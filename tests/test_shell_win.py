"""The native Windows shell backend: PowerShell + a per-run Job Object (src/smol_harness/shell_win.py).

These run real processes on Windows: no mocks for what the Job Object enforces (the whole process
tree dies at the timeout, the kernel applies the address-space limit) and no mocks for the honest
reporting of what it does not (writes and network are NOT confined; trusted repositories only).

The Linux-namespace pins live in tests/test_tools.py (`@linux_only`); these are their Windows
counterparts, plus the exit-code semantics of the PowerShell wrapper.
"""

from __future__ import annotations

import asyncio
import ctypes
import shutil
import platform
import sys
import tempfile
import time
from pathlib import Path

import pytest

import smol_harness.tools as tools_mod
from smol_harness.config import KNOWN_TOOLS
from smol_harness.shell_win import _KILLED_EXIT_CODE, WindowsShellTool, require_shell_host_windows
from smol_harness.tools import ToolBuildError, build_tools, resolve_workspace

windows_only = pytest.mark.skipif(platform.system() != "Windows", reason="the Windows shell backend runs on Windows")


def run(tool: WindowsShellTool, command: str, **kw):
    return asyncio.run(tool.execute(command=command, **kw))


def out(result) -> dict:
    assert result.output is not None, result.error
    return result.output


@pytest.fixture
def layout(tmp_path):
    """tmp/ws (the workspace) next to tmp/outside (must stay untouched)."""
    ws, outside = tmp_path / "ws", tmp_path / "outside"
    ws.mkdir()
    outside.mkdir()
    (ws / "a.txt").write_text("inside\n", encoding="utf-8")
    (outside / "secret.txt").write_text("outside secret\n", encoding="utf-8")
    return ws, outside


@pytest.fixture
def shell(layout):
    (tool,) = build_tools(["shell"], layout[0])
    assert isinstance(tool, WindowsShellTool)
    return tool


# ---- the wrapper's basic semantics ------------------------------------------------------------------------
@windows_only
def test_a_command_runs_streams_and_reports_honestly(shell):
    r = run(shell, "Write-Output hello; [Console]::Error.WriteLine('oops')")
    o = out(r)
    assert (o["exit_code"], o["stdout"], o["stderr"], o["timed_out"]) == (0, "hello\n", "oops\n", False)
    # what this backend does NOT enforce is stated in the result, never implied
    assert (o["filesystem_confined"], o["network_isolated"]) == (False, False)
    assert "hello" in o["result"]


@windows_only
def test_native_exit_codes_flow_through(shell):
    o = out(run(shell, "cmd /c exit 5"))
    assert o["exit_code"] == 5 and o["stdout"] == "" and not run(shell, "cmd /c exit 5").success
    assert "exit code: 5" in run(shell, "cmd /c exit 5").error


@windows_only
def test_an_explicit_exit_124_is_not_reported_as_a_timeout(shell):
    r = run(shell, "exit 124")
    assert not r.success and out(r)["exit_code"] == 124 and out(r)["timed_out"] is False


@windows_only
def test_a_failing_cmdlet_is_exit_code_1_and_reaches_the_model(shell):
    r = run(shell, "Get-Process id-nonexistent | Out-Null")
    assert not r.success and out(r)["exit_code"] == 1 and "id-nonexistent" in r.error


@windows_only
def test_the_command_starts_in_the_workspace_and_reads_its_files(shell, layout):
    ws = layout[0]
    assert out(run(shell, "(Get-Location).Path"))["stdout"].strip() == str(ws)
    assert out(run(shell, "Get-Content a.txt"))["stdout"] == "inside\n"
    run(shell, "New-Item -ItemType Directory -Force sub | Out-Null; Set-Content sub/made.txt built")
    assert (ws / "sub" / "made.txt").read_text(encoding="utf-8") == "built\n"  # Set-Content adds the engine newline


@windows_only
def test_shell_and_file_operations_share_the_workspace(layout):
    fo, sht = build_tools(["file_operations", "shell"], layout[0])
    assert asyncio.run(fo.execute(operation="write", path="t.txt", content="from file_operations")).success
    assert out(run(sht, "Get-Content t.txt"))["stdout"] == "from file_operations\n"  # the engine newline
    run(sht, "Set-Content s.txt -Value 'from shell'")
    assert asyncio.run(fo.execute(operation="read", path="s.txt")).output["data"] == "from shell\n"


@windows_only
def test_output_is_truncated_to_the_tail(shell):
    r = run(shell, f"& '{sys.executable}' -c \"print('x' * 30000); print('THE-END')\"")
    o = out(r)
    assert o["truncated"] and len(o["stdout"]) < WindowsShellTool.MAX_OUTPUT_CHARS + 100
    assert o["stdout"].rstrip().endswith("THE-END") and "earlier characters omitted" in o["stdout"]


@windows_only
def test_stdin_is_empty(shell):
    r = run(shell, "[Console]::In.ReadToEnd()")
    assert r.success and out(r)["stdout"].strip() == ""  # -ReadToEnd on an empty stdin leaves engine newlines


# ---- arguments and configuration ---------------------------------------------------------------------------
@windows_only
@pytest.mark.parametrize("args", [
    {}, {"command": ""}, {"command": "   \n"}, {"command": "echo a\x00b"}, {"command": 5},
    {"command": "x" * (WindowsShellTool.MAX_COMMAND_CHARS + 1)},
    {"command": "Write-Output hi", "timeout": 0}, {"command": "Write-Output hi", "timeout": -1},
    {"command": "Write-Output hi", "timeout": WindowsShellTool.MAX_TIMEOUT_S + 1},
    {"command": "Write-Output hi", "timeout": True}, {"command": "Write-Output hi", "timeout": 2.5},
    {"command": "Write-Output hi", "network_enabled": True}, {"command": "Write-Output hi", "memory_limit": "64g"},
])
def test_invalid_arguments_are_rejected_and_nothing_runs(shell, layout, args):
    if args.get("command") == "Write-Output hi":
        args = {**args, "command": "Set-Content ran.marker thanks"}
    r = run(shell, **args) if "command" in args else asyncio.run(shell.execute(**args))
    assert not r.success and r.error
    assert not (layout[0] / "ran.marker").exists()


@windows_only
def test_timeout_argument_is_capped_by_the_tool_configuration(layout):
    tool = WindowsShellTool(layout[0], max_timeout_s=3)
    assert not run(tool, "Write-Output hi", timeout=4).success
    assert run(tool, "Write-Output hi", timeout=3).success
    with pytest.raises(ToolBuildError, match="max_timeout_s"):
        WindowsShellTool(layout[0], max_timeout_s=WindowsShellTool.MAX_TIMEOUT_S + 1)
    with pytest.raises(ToolBuildError, match="memory_limit"):
        WindowsShellTool(layout[0], memory_limit="lots")


@windows_only
def test_the_parameter_schema_exposes_only_command_and_timeout(shell):
    assert [p.name for p in shell.metadata.model_facing_parameters] == ["command", "timeout"]


@windows_only
def test_the_built_tool_is_the_windows_backend_and_instances_are_fresh(layout):
    (a,), (b,) = build_tools(["shell"], layout[0]), build_tools(["shell"], layout[0])
    assert a is not b and isinstance(a, WindowsShellTool)


@windows_only
def test_the_shell_stays_in_the_registry_and_known_tools(layout):
    assert "shell" in tools_mod.TOOL_NAMES and "shell" in KNOWN_TOOLS
    assert resolve_workspace(str(layout[0])) == layout[0].resolve()


# ---- the Job Object: kernel-enforced tree kill and time limit ------------------------------------------------
def _process_alive(pid: int) -> bool:
    """True while the process exists (WAIT_TIMEOUT); False once it is gone."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    WAIT_TIMEOUT = 0x00000102
    handle = k32.OpenProcess(0x00100000 | 0x0001 | 0x0400, False, pid)  # SYNCHRONIZE | TERMINATE | QUERY
    if not handle:
        return False  # no handle: not ours to wait on, and OpenProcess fails for exited pids too
    try:
        return k32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
    finally:
        k32.CloseHandle(handle)


def _assert_dead(pid: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not _process_alive(pid):
            return
        time.sleep(0.05)
    pytest.fail(f"process {pid} survived the Job Object kill")


_DETACHED_CHILD = (
    "$p = Start-Process powershell -ArgumentList 'Start-Sleep','60' -PassThru -WindowStyle Hidden; "
    "Set-Content -Path {pid_file} -Value $p.Id; "
)


@windows_only
def test_a_command_exceeding_its_timeout_is_killed(shell):
    start = time.monotonic()
    r = run(shell, "Start-Sleep -Seconds 30", timeout=2)
    assert time.monotonic() - start < 8
    o = out(r)
    assert not r.success and o["timed_out"] and "timed out after 2s" in r.error
    assert o["exit_code"] == _KILLED_EXIT_CODE


@windows_only
def test_a_detached_child_attempts_no_escape_and_dies_with_the_killed_job(shell, tmp_path):
    """A Start-Process child (the Windows setsid/nohup) is inside the Job too: killed with it."""
    pid_file = tmp_path / "child.pid"
    r = run(shell, _DETACHED_CHILD.format(pid_file=pid_file) + "Start-Sleep -Seconds 30", timeout=2)
    assert out(r)["timed_out"]
    _assert_dead(int(pid_file.read_text(encoding="utf-8")))


@windows_only
def test_background_children_do_not_outlive_a_command_that_exits(shell, tmp_path):
    """Parity with the Linux PID namespace: when the command ends, nothing it started survives."""
    pid_file = tmp_path / "child.pid"
    r = run(shell, _DETACHED_CHILD.format(pid_file=pid_file) + "Write-Output done")
    assert r.success and out(r)["stdout"] == "done\n"
    _assert_dead(int(pid_file.read_text(encoding="utf-8")))


@windows_only
def test_the_memory_limit_is_integral_to_the_tool(layout):
    """The invariant: a memory limit either bites or the build refuses — never an advisory limit.

    On this Windows build the kernel records Job commit limits but does not apply them
    (512 MiB committed under 64m and 256m limits, all code paths, measured in batteries);
    elsewhere it does. Only `ToolBuildError` with the honest reason is acceptable there.
    """
    try:
        tool = WindowsShellTool(layout[0], memory_limit="256m")
    except ToolBuildError as error:
        assert "not enforceable" in str(error) and "probe" in str(error)
        return  # refused instead of claimed: the honest outcome on this build
    alloc = f"& '{sys.executable}' -c \"x = bytearray(512 << 20); print('allocated')\""
    r = run(tool, alloc)
    assert not r.success and out(r)["stdout"] == ""  # the allocation never completes: killed by the Job
    assert out(run(WindowsShellTool(layout[0], memory_limit="2g"), alloc))["stdout"] == "allocated\n"


@windows_only
def test_the_default_build_claims_no_memory_limit(shell):
    assert shell._memory_limit is None
    d = shell.metadata.description.lower()
    assert "memory is limited" not in d and "memory limit is enforced" not in d
    assert "no memory limit is configured" in d  # says so explicitly
    assert out(run(shell, "Write-Output hi"))["memory_limited"] is False


# ---- the honest statement of what is NOT enforced -----------------------------------------------------------
@windows_only
def test_reads_and_writes_outside_the_workspace_are_NOT_confined(shell, layout):
    # Documented (shell_win module docstring): the accepted scope is trusted repositories only.
    o = out(run(shell, f"Get-Content {layout[1] / 'secret.txt'}"))
    assert o["exit_code"] == 0 and o["stdout"] == "outside secret\n"
    run(shell, f"Set-Content {layout[1] / 'written.txt'} pwned")
    assert (layout[1] / "written.txt").read_text(encoding="utf-8") == "pwned\n"  # and it worked: NOT confined


# ---- fail closed --------------------------------------------------------------------------------------------
@windows_only
def test_without_powershell_the_tool_refuses_to_build(layout, monkeypatch):
    real = shutil.which
    monkeypatch.setattr(shutil, "which",
                        lambda name, *a, **k: None if name in ("powershell", "pwsh") else real(name, *a, **k))
    with pytest.raises(ToolBuildError, match="PowerShell"):
        build_tools(["shell"], layout[0])
    assert [t.name for t in build_tools(["file_operations"], layout[0])] == ["file_operations"]  # unaffected


@windows_only
def test_require_shell_host_windows_refuses_other_platforms(monkeypatch, tmp_path):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    with pytest.raises(ToolBuildError, match="needs Windows"):
        require_shell_host_windows()
