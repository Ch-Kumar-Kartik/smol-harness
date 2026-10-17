"""T4.1: tool factory and workspace-confined `file_operations`.

Everything goes through effGen's real `BaseTool.execute()` (and, at the end, its real
agent loop with scripted models) on temporary workspaces. CPU only, no network.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import sys
import tempfile
import time
from pathlib import Path, PureWindowsPath

import pytest
from conftest import ScriptedModel
from effgen import AgentConfig
from effgen.tools.base_tool import BaseTool
from effgen.tools.builtin.file_ops import FileOperations

import smol_harness.delegation as delegation_mod
import smol_harness.harness as harness_mod
import smol_harness.tools as tools_mod
from smol_harness.config import KNOWN_TOOLS, loads_config
from smol_harness.harness import build_harness
from effgen.security.sandbox import SandboxResult, SubprocessSandbox
from smol_harness.tools import (
    TOOL_NAMES, ShellSandboxError, ShellTool, ToolBuildError, _is_filesystem_root, build_tools, resolve_workspace,
)

WEB_LIKE = re.compile(r"web|search_engine|url_fetch|http", re.I)


def run(tool: BaseTool, **kw):
    return asyncio.run(tool.execute(**kw))


def data(result):
    assert result.success, result.error
    return result.output["data"]


def refused(result, reason: str = "outside allowed directories") -> str:
    assert not result.success
    assert reason in result.error, result.error
    return result.error


@pytest.fixture
def layout(tmp_path):
    """tmp/ws (the workspace) next to tmp/outside (must stay untouched)."""
    ws, outside = tmp_path / "ws", tmp_path / "outside"
    ws.mkdir()
    outside.mkdir()
    (ws / "a.txt").write_text("inside\n")
    (outside / "secret.txt").write_text("outside secret\n")
    return ws, outside


@pytest.fixture
def fo(layout):
    (tool,) = build_tools(["file_operations"], layout[0])
    return tool


def outside_unchanged(outside: Path) -> None:
    assert sorted(p.name for p in outside.iterdir()) == ["secret.txt"]
    assert (outside / "secret.txt").read_text() == "outside secret\n"


# ---- 1-3: valid reads, writes, nested paths ----------------------------------------------------------
def test_reads_a_workspace_file(fo, layout):
    assert data(run(fo, operation="read", path="a.txt")) == "inside\n"
    assert data(run(fo, operation="read", path=str(layout[0] / "a.txt"))) == "inside\n"  # absolute, inside


def test_writes_a_workspace_file(fo, layout):
    summary = data(run(fo, operation="write", path="b.py", content="x = 1\n"))
    assert summary == f"wrote b.py (6 bytes, 1 line, sha256 {hashlib.sha256(b'x = 1' + b'\n').hexdigest()[:8]})"
    assert (layout[0] / "b.py").read_text() == "x = 1\n"
    data(run(fo, operation="write", path="a.txt", content="replaced"))
    assert (layout[0] / "a.txt").read_text() == "replaced"


def test_nested_paths_create_parents_and_read_back(fo, layout):
    data(run(fo, operation="write", path="pkg/sub/mod.py", content="def f():\n    return 1\n"))
    assert (layout[0] / "pkg/sub/mod.py").is_file()
    assert data(run(fo, operation="read", path="pkg/sub/mod.py")).startswith("def f()")
    assert data(run(fo, operation="read", path="pkg/./sub/../sub/mod.py")).startswith("def f()")  # stays inside


def test_relative_paths_resolve_against_the_workspace_not_the_cwd(fo, layout, tmp_path, monkeypatch):
    monkeypatch.chdir(layout[1])
    data(run(fo, operation="write", path="c.txt", content="c"))
    assert (layout[0] / "c.txt").exists()
    outside_unchanged(layout[1])


def test_list_and_search_discover_repository_files(fo, layout):
    data(run(fo, operation="write", path="src/m.py", content="pass\n"))
    listed = {Path(e["path"]).relative_to(layout[0]).as_posix() for e in data(run(fo, operation="list", path=".", recursive=True))}
    assert {"a.txt", "src", "src/m.py"} <= listed
    found = data(run(fo, operation="search", path=".", pattern="*.py", recursive=True))
    assert [Path(p) for p in found] == [layout[0] / "src/m.py"]


# ---- 4-6: escapes are refused, for reads and writes ----------------------------------------------------
@pytest.mark.parametrize("path", ["../outside/secret.txt", "sub/../../outside/secret.txt", "..", "../../x.txt"])
def test_relative_traversal_is_refused(fo, layout, path):
    refused(run(fo, operation="read", path=path))
    refused(run(fo, operation="write", path=path.replace("secret", "new"), content="pwned"))
    outside_unchanged(layout[1])


def test_absolute_path_escapes_are_refused(fo, layout):
    refused(run(fo, operation="read", path=str(layout[1] / "secret.txt")))
    refused(run(fo, operation="write", path=str(layout[1] / "new.txt"), content="pwned"))
    refused(run(fo, operation="write", path=str(layout[1] / "secret.txt"), content="pwned"))
    refused(run(fo, operation="read", path="/etc/hostname"))
    refused(run(fo, operation="list", path="/"))
    refused(run(fo, operation="metadata", path=str(layout[1])))
    outside_unchanged(layout[1])


def test_symlink_escapes_are_refused_for_reads_and_writes(fo, layout):
    ws, outside = layout
    (ws / "file_link.txt").symlink_to(outside / "secret.txt")
    (ws / "dir_link").symlink_to(outside, target_is_directory=True)
    (ws / "dangling.txt").symlink_to(outside / "created.txt")  # target does not exist yet
    for path in ("file_link.txt", "dir_link/secret.txt"):
        refused(run(fo, operation="read", path=path))
        refused(run(fo, operation="metadata", path=path))
    refused(run(fo, operation="write", path="file_link.txt", content="pwned"))
    refused(run(fo, operation="write", path="dir_link/new.txt", content="pwned"))
    refused(run(fo, operation="write", path="dangling.txt", content="pwned"))
    refused(run(fo, operation="list", path="dir_link"))
    outside_unchanged(outside)


def test_listing_and_search_do_not_leak_symlinked_or_parent_entries(fo, layout):
    ws, outside = layout
    (ws / "dir_link").symlink_to(outside, target_is_directory=True)
    listed = [e["path"] for e in data(run(fo, operation="list", path=".", recursive=True))]
    assert all(Path(p).resolve().is_relative_to(ws) for p in listed) and not any("dir_link" in p for p in listed)
    assert data(run(fo, operation="search", path=".", pattern="../outside/*")) == []
    assert data(run(fo, operation="search", path=".", pattern="**/secret.txt", recursive=True)) == []


def test_symlink_inside_the_workspace_is_followed(fo, layout):
    (layout[0] / "alias.txt").symlink_to(layout[0] / "a.txt")
    assert data(run(fo, operation="read", path="alias.txt")) == "inside\n"


def test_a_symlinked_workspace_confines_to_its_target(layout, tmp_path):
    link = tmp_path / "ws_link"
    link.symlink_to(layout[0], target_is_directory=True)
    (tool,) = build_tools(["file_operations"], link)
    assert data(run(tool, operation="read", path="a.txt")) == "inside\n"
    refused(run(tool, operation="read", path="../outside/secret.txt"))


def test_malformed_paths_fail_cleanly(fo):
    assert not run(fo, operation="read", path="a\x00.txt").success
    assert not run(fo, operation="write", path="", content="x").success


# ---- 7-8: workspace validation --------------------------------------------------------------------------
def test_missing_workspace_is_rejected(tmp_path):
    with pytest.raises(ToolBuildError, match="does not exist"):
        build_tools(["file_operations"], tmp_path / "missing")


def test_workspace_that_is_a_regular_file_is_rejected(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(ToolBuildError, match="not a directory"):
        build_tools(["file_operations"], f)


def test_empty_and_broken_symlink_workspaces_are_rejected(tmp_path):
    with pytest.raises(ToolBuildError, match="non-empty"):
        resolve_workspace("")
    (tmp_path / "broken").symlink_to(tmp_path / "nowhere")
    with pytest.raises(ToolBuildError, match="does not exist"):
        resolve_workspace(tmp_path / "broken")


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses permission bits")
def test_inaccessible_workspace_is_rejected(tmp_path):
    ws = tmp_path / "locked"
    ws.mkdir()
    ws.chmod(0o500)  # no write
    try:
        with pytest.raises(ToolBuildError, match="not readable, writable"):
            build_tools(["file_operations"], ws)
    finally:
        ws.chmod(0o700)


def test_workspace_is_validated_even_for_an_empty_allowlist(tmp_path):
    with pytest.raises(ToolBuildError):
        build_tools([], tmp_path / "missing")


def test_resolve_workspace_returns_the_resolved_directory(layout):
    assert resolve_workspace(str(layout[0] / "." / ".." / "ws")) == layout[0].resolve()


# ---- 9-12: allowlists, unknown and forbidden tools --------------------------------------------------------
def test_allowlist_is_exact(layout):
    assert build_tools([], layout[0]) == []
    tools = build_tools(["file_operations"], layout[0])
    assert [t.name for t in tools] == ["file_operations"] and isinstance(tools[0], FileOperations)


def test_each_call_returns_fresh_instances(layout):
    (a,), (b,) = build_tools(["file_operations"], layout[0]), build_tools(["file_operations"], layout[0])
    assert a is not b


@pytest.mark.parametrize("name", ["bash", "code_executor", "edit_file", "FILE_OPERATIONS", "", "web_search",
                                  "url_fetch", "http_request", "browser", "agentic_search", "wikipedia"])
def test_unknown_and_web_tools_are_rejected(layout, name):
    with pytest.raises(ToolBuildError, match="unknown tool"):
        build_tools(["file_operations", name], layout[0])


def test_delegate_is_never_built(layout):
    with pytest.raises(ToolBuildError, match="orchestrator only"):
        build_tools(["delegate"], layout[0])


def test_duplicates_and_bare_strings_are_rejected(layout):
    with pytest.raises(ToolBuildError, match="duplicate"):
        build_tools(["file_operations", "file_operations"], layout[0])
    with pytest.raises(ToolBuildError, match="list of names"):
        build_tools("file_operations", layout[0])


def test_names_are_checked_before_the_workspace(tmp_path):
    with pytest.raises(ToolBuildError, match="unknown tool"):
        build_tools(["web_search"], tmp_path / "missing")


def test_registry_matches_config_and_holds_no_web_tool():
    assert TOOL_NAMES == KNOWN_TOOLS
    assert not any(WEB_LIKE.search(n) for n in TOOL_NAMES)


CONFIG = """
workspace: ws
orchestrator:
  model: {provider: ollama, model: "qwen2.5:7b", context_length: 8192}
  tools: [file_operations, delegate]
  max_iterations: 6
workers:
  - name: reader
    description: reads files
    model: {provider: ollama, model: "qwen2.5:1.5b", context_length: 4096}
    tools: [file_operations]
    max_iterations: 4
  - name: thinker
    description: no tools
    model: {provider: ollama, model: "qwen2.5:1.5b", context_length: 4096}
    tools: []
    max_iterations: 4
"""


def test_orchestrator_and_workers_get_their_own_allowlists(layout):
    config = loads_config(CONFIG, base_dir=layout[0].parent)
    assert config.workspace == layout[0]
    built = {spec.name: [t.name for t in build_tools(spec.tools, config.workspace)]
             for spec in (config.orchestrator, *config.workers)}
    assert built == {"orchestrator": ["file_operations"], "reader": ["file_operations"], "thinker": []}


# ---- the harness: real effGen loop, scripted models -------------------------------------------------------
def cfg(model, name, tools) -> AgentConfig:
    return AgentConfig(name=name, model=model, tools=list(tools), tool_calling_mode="react", capability_probe=False,
                       enable_memory=False, max_iterations=6, raise_on_error=False)


def test_harness_agents_hold_exactly_their_tools_and_no_web_tool(layout, monkeypatch):
    ws, outside = layout
    held: dict[str, list[str]] = {}

    class RecordingAgent(harness_mod.Agent):
        def __init__(self, config, *a, **kw):
            super().__init__(config, *a, **kw)
            held[config.name] = sorted(self.tools)

    monkeypatch.setattr(harness_mod, "Agent", RecordingAgent)
    monkeypatch.setattr(delegation_mod, "Agent", RecordingAgent)

    write = json.dumps({"operation": "write", "path": "notes/out.txt", "content": "from worker"})
    escape = json.dumps({"operation": "write", "path": "../outside/pwned.txt", "content": "x"})
    worker = ScriptedModel([
        f"Thought: write.\nAction: file_operations\nAction Input: {write}",
        f"Thought: try escaping.\nAction: file_operations\nAction Input: {escape}",
        "Thought: done.\nFinal Answer: wrote notes/out.txt",
    ], name="worker-model")
    orch = ScriptedModel([
        'Thought: delegate.\nAction: delegate\nAction Input: {"worker": "coder", "task": "write the note"}',
        "Thought: done.\nFinal Answer: ok",
    ], name="orch-model")
    harness = build_harness(
        cfg(orch, "orch", build_tools(["file_operations"], ws)),
        {"coder": cfg(worker, "coder", build_tools(["file_operations"], ws)),
         "idle": cfg(ScriptedModel(["Final Answer: -"]), "idle", build_tools([], ws))},
        {"coder": "writes files", "idle": "unused"},
    )
    result = harness.run("write a note")

    assert result.success, result.output
    assert held == {"orch": ["delegate", "file_operations"], "coder": ["file_operations"]}
    assert not any(WEB_LIKE.search(n) for names in held.values() for n in names)
    assert (ws / "notes/out.txt").read_text() == "from worker"
    assert [(r.worker, r.status) for r in result.workers] == [("coder", "completed")]
    # The escape attempt really ran and came back to the worker as a refusal observation.
    assert len(worker.calls) == 3 and "outside allowed directories" in str(worker.calls[2].prompt)
    outside_unchanged(outside)


# ---- regression pins for surprising effGen 1.3.0 behaviour (audit §1.9) -----------------------------------
def test_effgen_widens_a_missing_allowed_directory_to_the_cwd(tmp_path, monkeypatch):
    # Why build_tools validates the workspace and reads back the kept root.
    monkeypatch.delenv("EFFGEN_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    assert FileOperations(allowed_directories=[str(tmp_path / "missing")])._allowed_directories == [tmp_path.resolve()]


def test_effgen_falls_back_to_and_creates_effgen_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFGEN_WORKSPACE", str(tmp_path / "env_ws"))
    tool = FileOperations(allowed_directories=[str(tmp_path / "missing")])
    assert tool._allowed_directories == [(tmp_path / "env_ws").resolve()] and (tmp_path / "env_ws").is_dir()


@pytest.mark.parametrize("env", [None, "env_ws"])
def test_a_workspace_vanishing_after_validation_fails_closed(tmp_path, monkeypatch, env):
    if env:
        monkeypatch.setenv("EFFGEN_WORKSPACE", str(tmp_path / env))
    else:
        monkeypatch.delenv("EFFGEN_WORKSPACE", raising=False)
    monkeypatch.setattr(tools_mod, "resolve_workspace", lambda ws: tmp_path / "gone")  # simulates the race
    with pytest.raises(ToolBuildError, match="did not confine"):
        build_tools(["file_operations"], tmp_path / "gone")


def test_effgen_workspace_env_does_not_override_an_explicit_workspace(layout, tmp_path, monkeypatch):
    monkeypatch.setenv("EFFGEN_WORKSPACE", str(tmp_path / "env_ws"))
    (tool,) = build_tools(["file_operations"], layout[0])
    data(run(tool, operation="write", path="x.txt", content="x"))
    assert (layout[0] / "x.txt").exists() and not (tmp_path / "env_ws").exists()


@pytest.mark.parametrize("name", ["Makefile", "README", ".gitignore", "Dockerfile"])
def test_effgen_refuses_extensionless_files(fo, layout, name):
    (layout[0] / name).write_text("x\n")
    refused(run(fo, operation="read", path=name), "not allowed for read")
    refused(run(fo, operation="write", path=name, content="y"), "not allowed for write")


def test_effgen_refuses_credential_filenames_and_content(fo, layout):
    (layout[0] / ".env").write_text("A=1\n")
    refused(run(fo, operation="read", path=".env"), "credentials")
    (layout[0] / "settings.txt").write_text("API_KEY=abc\nDB_PASSWORD=xyz\n")
    refused(run(fo, operation="read", path="settings.txt"), "credentials")


# ======================================================================================================
# T4.2: sandboxed shell. These run effGen's real SubprocessSandbox (Linux user/mount/net/PID namespaces)
# on temporary workspaces: no mocks for the security properties. A host that cannot provide the namespaces
# makes them FAIL, not skip: the shell is unusable there by design (see test_degraded_sandbox_*).
# ======================================================================================================
linux_only = pytest.mark.skipif(platform.system() != "Linux", reason="the shell sandbox is Linux-only (namespaces)")


def sh(tool: ShellTool, command: str, **kw):
    return run(tool, command=command, **kw)


def out(result) -> dict:
    assert result.output is not None, result.error
    return result.output


@pytest.fixture
def shell(layout):
    (tool,) = build_tools(["shell"], layout[0])
    return tool


def proc_cmdlines() -> dict[int, str]:
    found = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                found[int(entry.name)] = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                pass
    return found


def survivors(marker: str) -> list[int]:
    return [pid for pid, cmd in proc_cmdlines().items() if marker in cmd and pid != os.getpid()]


@pytest.fixture
def marker():
    """A unique `sleep` duration that identifies this test's processes; any survivor is killed on exit."""
    m = f"{5000 + (os.getpid() % 1000)}.{time.monotonic_ns() % 1_000_000:06d}"
    yield m
    for pid in survivors(f"sleep {m}"):
        os.kill(pid, signal.SIGKILL)


# ---- valid execution, workspace placement ------------------------------------------------------------------
@linux_only
def test_harmless_command_runs(shell):
    r = sh(shell, "echo hello; echo oops >&2")
    assert r.success, r.error
    o = out(r)
    assert (o["exit_code"], o["stdout"], o["stderr"], o["timed_out"]) == (0, "hello\n", "oops\n", False)
    assert (o["filesystem_confined"], o["network_isolated"]) == (True, True)
    assert "hello" in o["result"]  # what effGen shows the model for a successful call


@linux_only
def test_command_starts_in_the_workspace_and_writes_land_there(shell, layout):
    ws = layout[0]
    assert out(sh(shell, "pwd"))["stdout"].strip() == str(ws)
    assert out(sh(shell, "cat a.txt"))["stdout"] == "inside\n"
    sh(shell, "mkdir -p sub && echo built > sub/made.txt && cd sub && pwd")
    assert (ws / "sub/made.txt").read_text() == "built\n"
    assert out(sh(shell, "cd sub && pwd"))["stdout"].strip() == str(ws / "sub")


@linux_only
def test_shell_and_file_operations_share_the_workspace(layout):
    fo, sht = build_tools(["file_operations", "shell"], layout[0])
    data(run(fo, operation="write", path="t.txt", content="from file_operations"))
    assert out(sh(sht, "cat t.txt"))["stdout"] == "from file_operations"
    sh(sht, "echo from shell > s.txt")
    assert data(run(fo, operation="read", path="s.txt")) == "from shell\n"


@linux_only
def test_non_zero_exit_is_a_failure_whose_output_reaches_the_model(shell):
    r = sh(shell, "echo partial; echo boom-marker >&2; exit 3")
    assert not r.success
    assert out(r)["exit_code"] == 3 and out(r)["stdout"] == "partial\n"
    # effGen shows a model only `error` for a failed result, so the output must be in it.
    assert "exit code: 3" in r.error and "boom-marker" in r.error and "partial" in r.error


@linux_only
def test_a_commands_own_exit_124_is_not_reported_as_a_timeout(shell):
    r = sh(shell, "exit 124")
    assert not r.success and out(r)["exit_code"] == 124 and out(r)["timed_out"] is False


@linux_only
def test_stdin_is_empty_and_cannot_consume_the_script(shell):
    r = sh(shell, "cat; echo after")
    assert r.success and out(r)["stdout"] == "after\n"


@linux_only
def test_output_is_truncated_to_the_tail(shell):
    r = sh(shell, "python3 -c \"print('x' * 30000); print('THE-END')\"")
    o = out(r)
    assert o["truncated"] and len(o["stdout"]) < ShellTool.MAX_OUTPUT_CHARS + 100
    assert o["stdout"].rstrip().endswith("THE-END") and "earlier characters omitted" in o["stdout"]


# ---- invalid workspace / arguments ---------------------------------------------------------------------------
def test_shell_rejects_invalid_workspaces(tmp_path):
    with pytest.raises(ToolBuildError, match="does not exist"):
        build_tools(["shell"], tmp_path / "missing")
    f = tmp_path / "f.txt"
    f.write_text("x")
    with pytest.raises(ToolBuildError, match="not a directory"):
        ShellTool(f)
    with pytest.raises(ToolBuildError, match="non-empty"):
        ShellTool("")


@linux_only
@pytest.mark.parametrize("args", [
    {}, {"command": ""}, {"command": "   \n"}, {"command": "echo a\x00b"}, {"command": 5}, {"command": None},
    {"command": "x" * (ShellTool.MAX_COMMAND_CHARS + 1)},
    {"command": "echo hi", "timeout": 0}, {"command": "echo hi", "timeout": -1},
    {"command": "echo hi", "timeout": ShellTool.MAX_TIMEOUT_S + 1}, {"command": "echo hi", "timeout": "soon"},
    {"command": "echo hi", "timeout": True}, {"command": "echo hi", "timeout": 2.5},
    {"command": "echo hi", "network_enabled": True}, {"command": "echo hi", "memory_limit": "64g"},
    {"command": "echo hi", "language": "python"}, {"command": "echo hi", "files": {"a": "b"}},
])
def test_invalid_arguments_are_rejected_and_nothing_runs(shell, layout, args):
    if args.get("command") == "echo hi":
        args = {**args, "command": "touch ran.marker"}
    r = run(shell, **args)
    assert not r.success and r.error
    assert not (layout[0] / "ran.marker").exists()


@linux_only
def test_timeout_argument_is_capped_by_the_tool_configuration(layout):
    tool = ShellTool(layout[0], max_timeout_s=3)
    assert not sh(tool, "echo hi", timeout=4).success
    assert sh(tool, "echo hi", timeout=3).success
    with pytest.raises(ToolBuildError, match="max_timeout_s"):
        ShellTool(layout[0], max_timeout_s=ShellTool.MAX_TIMEOUT_S + 1)
    with pytest.raises(ToolBuildError, match="memory_limit"):
        ShellTool(layout[0], memory_limit="lots")


# ---- write confinement, read characterisation -----------------------------------------------------------------
@linux_only
def test_writes_outside_the_workspace_are_refused(shell, layout):
    ws, outside = layout
    for cmd in (f"echo pwned > {outside}/new.txt", f"echo pwned > {outside}/secret.txt", "echo pwned > ../outside/new2.txt",
                f"echo pwned > {ws.parent}/sibling.txt", f"mkdir {outside}/d", f"rm {outside}/secret.txt",
                f"mv a.txt {outside}/moved.txt", f"cp a.txt {outside}/copied.txt"):
        r = sh(shell, cmd)
        assert not r.success and "Read-only file system" in out(r)["stderr"], (cmd, out(r))
    outside_unchanged(outside)
    assert not (ws.parent / "sibling.txt").exists() and (ws / "a.txt").exists()


@linux_only
def test_symlinks_cannot_be_used_to_write_outside(shell, layout):
    ws, outside = layout
    r = sh(shell, f"ln -s {outside} dir_link && ln -s {outside}/secret.txt file_link && echo pwned > dir_link/x.txt; "
                  "echo pwned > file_link; echo pwned > dir_link/secret.txt")
    assert "Read-only file system" in out(r)["stderr"]
    outside_unchanged(outside)  # the links exist inside the workspace, the writes through them were refused


@linux_only
def test_the_sandbox_cannot_remount_its_way_out(shell, layout):
    r = sh(shell, f"mount -o remount,rw,bind {layout[0].parent} 2>&1; mount -o remount,rw / 2>&1; "
                  f"unshare --user --map-root-user --mount sh -c 'mount -o remount,rw,bind {layout[0].parent} 2>&1; "
                  f"echo pwned > {layout[1]}/r.txt' 2>&1; true")
    outside_unchanged(layout[1])
    assert not (layout[0].parent / "r.txt").exists(), out(r)


@linux_only
def test_host_tmp_writes_do_not_reach_the_host(shell):
    name = f"smol-t42-{os.getpid()}-{time.monotonic_ns()}"
    sh(shell, f"echo x > /tmp/{name} 2>/dev/null; echo x > /var/tmp/{name} 2>/dev/null; true")
    assert not Path("/tmp", name).exists() and not Path("/var/tmp", name).exists()


@linux_only
def test_reads_outside_the_workspace_are_NOT_confined(shell, layout):
    # Documented limitation (audit Part 3, T4.2): reads are a deny-list, not confinement. If this ever
    # starts failing, the boundary got stronger: update the docs, do not weaken the test silently.
    assert out(sh(shell, f"cat {layout[1]}/secret.txt"))["stdout"] == "outside secret\n"
    assert out(sh(shell, "cat ../outside/secret.txt"))["stdout"] == "outside secret\n"
    (layout[0] / "dir_link").symlink_to(layout[1], target_is_directory=True)
    assert out(sh(shell, "cat dir_link/secret.txt"))["stdout"] == "outside secret\n"
    assert out(sh(shell, "cat /etc/hostname"))["exit_code"] == 0


@linux_only
def test_shell_reads_what_file_operations_refuses(shell, layout):
    # The shell is a wider tool than file_operations: no extension allowlist, no credential-name filter.
    (layout[0] / ".env").write_text("API_KEY=abc\nDB_PASSWORD=xyz\n")
    (layout[0] / "Makefile").write_text("all:\n")
    assert "API_KEY=abc" in out(sh(shell, "cat .env"))["stdout"]
    assert out(sh(shell, "cat Makefile"))["stdout"] == "all:\n"


@linux_only
def test_credential_stores_under_home_are_masked(shell, layout, monkeypatch):
    home = layout[1] / "fakehome"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_test").write_text("PRIVATE-KEY-MATERIAL\n")
    (home / ".netrc").write_text("machine x password PASSWORD-MATERIAL\n")
    monkeypatch.setenv("HOME", str(home))  # effGen's sandbox masks "$HOME/.ssh", "$HOME/.netrc", ...
    r = sh(shell, "cat $HOME/.ssh/id_test $HOME/.netrc 2>&1; ls $HOME/.ssh | wc -l")
    assert "PRIVATE-KEY-MATERIAL" not in out(r)["stdout"] and "PASSWORD-MATERIAL" not in out(r)["stdout"]
    assert (home / ".ssh" / "id_test").read_text() == "PRIVATE-KEY-MATERIAL\n"  # the host copy is untouched


# ---- network ----------------------------------------------------------------------------------------------------
@linux_only
def test_network_is_blocked(shell):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port = srv.getsockname()[1]
    try:
        socket.create_connection(("127.0.0.1", port), timeout=2).close()  # control: reachable from the host
        for cmd in (f"bash -c 'exec 3<>/dev/tcp/127.0.0.1/{port}'",
                    f"python3 -c \"import socket; socket.create_connection(('127.0.0.1', {port}), 2)\"",
                    "python3 -c \"import socket; socket.create_connection(('93.184.216.34', 80), 2)\"",
                    "python3 -c \"import socket; socket.getaddrinfo('example.com', 80)\""):
            r = sh(shell, cmd)
            assert not r.success, (cmd, out(r))
        assert out(sh(shell, "awk 'NR>2 {sub(/:.*/, \"\", $1); print $1}' /proc/net/dev"))["stdout"].split() == ["lo"]
    finally:
        srv.close()


# ---- host Unix sockets (not covered by effGen's network namespace) ----------------------------------------
HOST_SOCKETS = ["/run/systemd/resolve/io.systemd.Resolve", "/run/dbus/system_bus_socket", "/run/docker.sock",
                f"/run/user/{os.getuid()}/bus", f"/run/user/{os.getuid()}/systemd/private"]
CONNECT = "python3 -c \"import socket,sys; s=socket.socket(socket.AF_UNIX); s.settimeout(2); s.connect(sys.argv[1])\" "


def host_can_connect(path: str) -> bool:
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(2)
    try:
        s.connect(path)
        return True
    except OSError:
        return False
    finally:
        s.close()


@linux_only
@pytest.mark.parametrize("path", HOST_SOCKETS)
def test_host_sockets_under_run_are_unreachable(shell, path):
    if not host_can_connect(path):
        pytest.skip(f"{path} is absent or not connectable on this host")
    r = sh(shell, CONNECT + shlex.quote(path))
    assert not r.success and ("No such file" in out(r)["stderr"]), out(r)  # hidden, not merely refused


@linux_only
def test_effgen_sandbox_alone_reaches_host_sockets(layout):
    # Regression pin (audit Part 3, T4.2): effGen's net namespace leaves pathname Unix sockets reachable, which
    # is why ShellTool hides /run. If this fails, effGen closed the gap: review _HIDE_HOST_SOCKETS.
    from effgen.security.sandbox import SandboxConfig
    path = next((p for p in HOST_SOCKETS if host_can_connect(p)), None)
    if path is None:
        pytest.skip("no connectable host socket under /run on this host")
    cfg = SandboxConfig(backend="subprocess", timeout=10, memory_limit="1g", docker_image="unused", workdir=str(layout[0]))
    result = asyncio.run(SubprocessSandbox().run(CONNECT + shlex.quote(path), "bash", cfg))
    assert result.network_isolated and result.exit_code == 0, result.stderr


@linux_only
def test_dns_cannot_leak_through_the_host_resolver(shell):
    for cmd in ("getent hosts example.com", "python3 -c \"import socket; socket.getaddrinfo('example.com', 80)\""):
        assert not sh(shell, cmd).success, cmd


@linux_only
def test_run_is_an_empty_read_only_directory_inside_the_sandbox(shell):
    r = sh(shell, "ls -A /run | wc -l; touch /run/x 2>&1; true")
    assert out(r)["stdout"].splitlines()[0] == "0" and "Read-only" in out(r)["stdout"]


@linux_only
def test_host_tmp_is_hidden_only_when_the_workspace_is_outside_tmp(layout):
    # Documented limitation: with the workspace under /tmp (as in these tests) effGen cannot give a private /tmp,
    # so sockets in the host /tmp stay reachable. A workspace elsewhere gets a private, empty /tmp.
    probe_dir = Path(tempfile.mkdtemp(prefix="smol-t42-host-tmp-", dir="/tmp"))
    sock = socket.socket(socket.AF_UNIX)
    sock.bind(str(probe_dir / "s.sock"))
    sock.listen(1)
    try:
        under_tmp = ShellTool(layout[0])
        assert sh(under_tmp, f"test -S {probe_dir}/s.sock").success  # visible: the documented gap
        elsewhere = Path(tempfile.mkdtemp(prefix=".t42-ws-", dir=Path(__file__).resolve().parents[1]))
        try:
            assert not str(elsewhere).startswith("/tmp")
            assert not sh(ShellTool(elsewhere), f"test -e {probe_dir}").success  # hidden
        finally:
            shutil.rmtree(elsewhere, ignore_errors=True)
    finally:
        sock.close()
        shutil.rmtree(probe_dir, ignore_errors=True)


# ---- timeout and child processes ----------------------------------------------------------------------------------
@linux_only
def test_command_exceeding_its_timeout_is_killed(shell, marker):
    start = time.monotonic()
    r = sh(shell, f"echo started; sleep {marker}", timeout=2)
    assert time.monotonic() - start < 8
    o = out(r)
    assert not r.success and o["timed_out"] and "timed out after 2s" in r.error and o["stdout"] == "started\n"
    assert survivors(f"sleep {marker}") == []


@linux_only
def test_every_spawned_process_dies_with_the_timeout(shell, marker):
    cmd = (f"sleep {marker} & setsid sleep {marker} & nohup sleep {marker} >/dev/null 2>&1 & "
           f"(trap '' TERM; sleep {marker}) & sleep {marker}")
    start = time.monotonic()
    r = sh(shell, cmd, timeout=2)
    assert time.monotonic() - start < 10 and out(r)["timed_out"]
    assert survivors(f"sleep {marker}") == []


@linux_only
def test_a_term_ignoring_command_is_killed_after_the_grace_period(shell, marker):
    start = time.monotonic()
    r = sh(shell, f"trap '' TERM; while true; do sleep {marker}; done", timeout=2)
    assert out(r)["timed_out"] and time.monotonic() - start < 10
    assert survivors(f"sleep {marker}") == []


@linux_only
def test_background_processes_do_not_outlive_a_command_that_exits(shell, marker):
    start = time.monotonic()
    r = sh(shell, f"sleep {marker} & setsid sleep {marker} & echo done")
    assert r.success and out(r)["stdout"] == "done\n" and time.monotonic() - start < 8
    assert survivors(f"sleep {marker}") == []


@linux_only
def test_effgen_timeout_path_leaves_children_and_blocks(layout):
    # Regression pin (audit Part 3, T4.2): on its own deadline (timeout + 2 s) SubprocessSandbox.run kills only the
    # outer `unshare`, leaves the descendants running, and then waits for their pipes to close, so it returns only
    # when they end by themselves. ShellTool avoids this by timing out inside the PID namespace. If upstream fixes it,
    # this test fails: review ShellTool's workaround and relax the pin.
    from effgen.security.sandbox import SandboxConfig
    duration = f"4.{time.monotonic_ns() % 1_000_000:06d}"
    cfg = SandboxConfig(backend="subprocess", timeout=1, memory_limit="256m", docker_image="unused", workdir=str(layout[0]))

    async def scenario():
        task = asyncio.create_task(SubprocessSandbox().run(f"sleep {duration} & sleep {duration}", "bash", cfg))
        await asyncio.sleep(3.5)  # effGen's own deadline (1 + 2 s) has passed
        alive, done = survivors(f"sleep {duration}"), task.done()
        return alive, done, await task

    start = time.monotonic()
    alive, done_early, _ = asyncio.run(scenario())
    assert alive and not done_early  # children still running, and run() still has not returned
    assert time.monotonic() - start >= 3.9  # it returned only once they exited on their own


# ---- memory ---------------------------------------------------------------------------------------------------
ALLOC = "python3 -c \"x = bytearray(512 << 20); print('allocated')\""


@linux_only
def test_memory_limit_is_enforced(layout):
    small, large = ShellTool(layout[0], memory_limit="256m"), ShellTool(layout[0], memory_limit="2g")
    r = sh(small, ALLOC)
    assert not r.success and "MemoryError" in out(r)["stderr"] and out(r)["stdout"] == ""
    assert out(sh(large, ALLOC))["stdout"] == "allocated\n"
    assert out(sh(small, "ulimit -v"))["stdout"].strip() == str(256 * 1024)


@linux_only
def test_default_memory_limit_runs_a_real_pytest_run(shell, layout):
    (layout[0] / "test_sample.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n\ndef test_bad():\n    assert 1 + 1 == 3\n")
    assert out(sh(shell, "ulimit -v"))["stdout"].strip() == str(1024 * 1024)
    r = sh(shell, f"{shlex.quote(sys.executable)} -m pytest -q -p no:cacheprovider test_sample.py", timeout=60)
    assert not r.success and out(r)["exit_code"] == 1  # a failing test is a failed result...
    assert "1 failed, 1 passed" in r.error and "assert 1 + 1 == 3" in r.error  # ...that shows the model why


# ---- the sandbox is required: fail closed ---------------------------------------------------------------------------
def test_shell_is_refused_off_linux(layout, monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    with pytest.raises(ToolBuildError, match="needs Linux") as e:
        build_tools(["shell"], layout[0])
    assert "WSL2" in str(e.value)
    assert [t.name for t in build_tools(["file_operations"], layout[0])] == ["file_operations"]  # unaffected


@pytest.mark.parametrize("missing", ["unshare", "timeout", "bash"])
def test_shell_is_refused_without_its_host_binaries(layout, monkeypatch, missing):
    import shutil
    real = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None if name == missing else real(name, *a, **k))
    with pytest.raises(ToolBuildError, match=missing):
        build_tools(["shell"], layout[0])


@linux_only
def test_degraded_sandbox_runs_nothing_and_fails_closed(layout, monkeypatch):
    # Simulate a host without user namespaces: this is the state in which effGen quietly runs plain `bash`.
    monkeypatch.setattr(SubprocessSandbox, "_caps_probed", True)
    monkeypatch.setattr(SubprocessSandbox, "_userns_ok", False)
    monkeypatch.setattr(SubprocessSandbox, "_mountns_ok", False)
    monkeypatch.setattr(SubprocessSandbox, "_pidns_ok", False)
    tool = ShellTool(layout[0])
    r = sh(tool, "touch ran.marker; echo hi")
    assert not r.success and "did not enforce" in r.error and "refusing to use the shell" in r.error
    assert not (layout[0] / "ran.marker").exists()  # only the fixed no-op probe ran, never the command
    assert r.output is None


@linux_only
@pytest.mark.parametrize("flags", [dict(filesystem_confined=False), dict(network_isolated=False),
                                   dict(process_table_isolated=False), dict(writable_root="/elsewhere"),
                                   dict(error="spawn failed")])
def test_each_missing_guarantee_is_fatal(layout, monkeypatch, flags):
    good = dict(stdout="", stderr="", exit_code=0, filesystem_confined=True, network_isolated=True,
                process_table_isolated=True, writable_root=str(layout[0]))

    async def fake_run(self, code, language, config):
        return SandboxResult(**{**good, **flags})

    monkeypatch.setattr(SubprocessSandbox, "run", fake_run)
    tool = ShellTool(layout[0])
    r = sh(tool, "echo hi")
    assert not r.success and (r.error.startswith("Tool execution failed") or "sandbox" in r.error)
    assert tool._verified is False  # a failed proof is never cached


@linux_only
def test_a_sandbox_that_never_returns_is_cut_off(layout, monkeypatch):
    good = dict(stdout="", stderr="", exit_code=0, filesystem_confined=True, network_isolated=True,
                process_table_isolated=True, writable_root=str(layout[0]))

    async def fake_run(self, code, language, config):
        if "sleep" in code:
            await asyncio.sleep(60)
        return SandboxResult(**good)

    monkeypatch.setattr(SubprocessSandbox, "run", fake_run)
    monkeypatch.setattr(ShellTool, "SANDBOX_SLACK_S", 0)
    monkeypatch.setattr(ShellTool, "HARD_DEADLINE_SLACK_S", 1)
    start = time.monotonic()
    r = sh(ShellTool(layout[0]), "sleep 100", timeout=1)
    assert not r.success and "hard deadline" in r.error and time.monotonic() - start < 6


@linux_only
def test_the_sandbox_backend_is_pinned_to_subprocess(layout, monkeypatch):
    seen = []
    real_run = SubprocessSandbox.run

    async def spy(self, code, language, config):
        seen.append((config.backend, config.network_enabled, config.workdir))
        return await real_run(self, code, language, config)

    monkeypatch.setattr(SubprocessSandbox, "run", spy)
    monkeypatch.setenv("EFFGEN_SANDBOX_BACKEND", "off")  # must not be honoured
    monkeypatch.setenv("EFFGEN_SANDBOX_MEMORY", "4g")
    tool = ShellTool(layout[0])
    assert sh(tool, "echo hi").success
    assert out(sh(tool, "ulimit -v"))["stdout"].strip() == str(1024 * 1024)
    assert seen and all(s == ("subprocess", False, str(layout[0])) for s in seen)


# ---- allowlists ---------------------------------------------------------------------------------------------------
def test_shell_is_absent_unless_allowlisted(layout):
    assert [t.name for t in build_tools(["file_operations"], layout[0])] == ["file_operations"]
    assert build_tools([], layout[0]) == []
    assert "shell" in TOOL_NAMES and "shell" in KNOWN_TOOLS


@linux_only
def test_shell_instances_are_fresh_and_isolated_per_agent(layout):
    (a,), (b,) = build_tools(["shell"], layout[0]), build_tools(["shell"], layout[0])
    assert a is not b and isinstance(a, ShellTool)


@linux_only
def test_shell_parameter_schema_exposes_only_command_and_timeout(shell):
    assert [p.name for p in shell.metadata.model_facing_parameters] == ["command", "timeout"]


@linux_only
def test_an_agent_without_shell_cannot_reach_it_and_one_with_shell_can(layout):
    ws, outside = layout
    held: dict[str, list[str]] = {}

    class RecordingAgent(harness_mod.Agent):
        def __init__(self, config, *a, **kw):
            super().__init__(config, *a, **kw)
            held[config.name] = sorted(self.tools)

    call = lambda tool, **args: f"Thought: go.\nAction: {tool}\nAction Input: {json.dumps(args)}"  # noqa: E731
    final = "Thought: done.\nFinal Answer: ok"
    shell_worker = ScriptedModel([
        call("shell", command="pwd > where.txt; echo boom-marker >&2; exit 3"), call("shell", command="touch built.txt"), final,
    ], name="shell-worker")
    files_only = ScriptedModel([call("shell", command="touch smuggled.txt"), final], name="files-worker")
    orch = ScriptedModel([
        'Thought: d.\nAction: delegate\nAction Input: {"worker": "builder", "task": "build"}',
        'Thought: d.\nAction: delegate\nAction Input: {"worker": "reader", "task": "try to use the shell"}',
        'Thought: done.\nFinal Answer: finished',
    ], name="orch")
    harness = build_harness(
        cfg(orch, "orch", build_tools(["file_operations"], ws)),
        {"builder": cfg(shell_worker, "builder", build_tools(["file_operations", "shell"], ws)),
         "reader": cfg(files_only, "reader", build_tools(["file_operations"], ws))},
        {"builder": "runs commands", "reader": "reads files"},
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(harness_mod, "Agent", RecordingAgent)
        mp.setattr(delegation_mod, "Agent", RecordingAgent)
        result = harness.run("go")

    assert result.success, result.output
    assert held == {"orch": ["delegate", "file_operations"], "builder": ["file_operations", "shell"], "reader": ["file_operations"]}
    assert not any(WEB_LIKE.search(n) for names in held.values() for n in names)
    assert (ws / "where.txt").read_text().strip() == str(ws) and (ws / "built.txt").exists()
    assert not (ws / "smuggled.txt").exists()  # the worker without `shell` could not run it
    assert any("boom-marker" in str(c.prompt) for c in shell_worker.calls)  # a failing command's stderr reached the model
    assert any("not found" in str(c.prompt).lower() or "unknown tool" in str(c.prompt).lower()
               or "available" in str(c.prompt).lower() for c in files_only.calls[1:])
    outside_unchanged(outside)


# ---- regression pins for effGen behaviour the shell depends on (audit §1.8) ----------------------------------------
@linux_only
def test_effgen_strips_unknown_parameters_so_the_tool_must_reject_them(layout):
    from effgen.tools.builtin.file_ops import FileOperations as FO
    fo = FO(allowed_directories=[str(layout[0])])
    assert run(fo, operation="read", path="a.txt", network_enabled=True).success  # silently ignored by effGen itself
    assert not run(ShellTool(layout[0]), command="echo hi", network_enabled=True).success  # rejected by ShellTool


# ---- T4.1 correction: dangerous workspace roots ---------------------------------------------------------------------
def test_filesystem_roots_are_rejected_as_workspaces():
    with pytest.raises(ToolBuildError, match="filesystem root"):
        resolve_workspace(Path("/").anchor)
    with pytest.raises(ToolBuildError, match="filesystem root"):
        build_tools(["file_operations"], "/")


def test_home_directory_and_its_ancestors_are_rejected_but_subdirectories_are_not(tmp_path, monkeypatch):
    home = tmp_path / "users" / "me"
    project = home / "projects" / "repo"
    project.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    for bad in (home, home.parent, tmp_path):
        with pytest.raises(ToolBuildError, match="contains the home directory"):
            build_tools(["file_operations"], bad)
    assert resolve_workspace(project) == project.resolve()
    assert resolve_workspace(home / "projects") == (home / "projects").resolve()


def test_a_symlink_to_the_filesystem_root_is_rejected(tmp_path):
    (tmp_path / "rootlink").symlink_to("/", target_is_directory=True)
    with pytest.raises(ToolBuildError, match="filesystem root"):
        resolve_workspace(tmp_path / "rootlink")


def test_root_detection_follows_windows_drive_semantics():
    # Pure-path semantics only: this is NOT a test on Windows (no Windows host was available).
    assert _is_filesystem_root(PureWindowsPath("C:\\")) and _is_filesystem_root(PureWindowsPath("D:/"))
    assert _is_filesystem_root(PureWindowsPath("\\\\server\\share\\"))
    assert not _is_filesystem_root(PureWindowsPath("C:\\work\\repo")) and not _is_filesystem_root(PureWindowsPath("C:\\work"))


# ======================================================================================================
# Review F2 (phase A): results that say what was done. Each claim is checked against the real disk / process.
# ======================================================================================================
def sha8(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def test_write_summary_is_read_back_from_the_file_on_disk(fo, layout):
    ws = layout[0]
    for name, content in (("f.py", "x = 1\n"), ("sub/g.py", "a\nb\nc"), ("empty.txt", ""), ("é.md", "héllo\n")):
        summary = data(run(fo, operation="write", path=name, content=content))
        on_disk = ws / name
        body = on_disk.read_bytes()
        assert body == content.encode()  # the side effect really happened
        lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        assert summary == f"wrote {name} ({len(body)} bytes, {lines} line{'s' if lines != 1 else ''}, sha256 {sha8(on_disk)})"


def test_write_summary_changes_with_the_content_not_with_the_call(fo, layout):
    first = data(run(fo, operation="write", path="f.py", content="x = 1\n"))
    other = data(run(fo, operation="write", path="f.py", content="x = 2\n"))  # same path, same size
    again = data(run(fo, operation="write", path="f.py", content="x = 2\n"))
    assert first != other  # different content reads differently (what effGen's bare path could not do)
    assert other == again  # an identical rewrite IS the same result: no counter hides a real repeat
    assert "x = 2" not in other  # a summary, not the content


def test_write_summary_reflects_the_formatted_file_for_json(fo, layout):
    summary = data(run(fo, operation="write", path="d.json", content='{"a":1}', format="json"))
    assert (layout[0] / "d.json").read_text() == '{\n  "a": 1\n}'  # effGen re-indents; the summary describes this file
    size = len((layout[0] / "d.json").read_bytes())
    assert summary == f"wrote d.json ({size} bytes, 3 lines, sha256 {sha8(layout[0] / 'd.json')})"


def test_write_summary_does_not_touch_failures_or_other_operations(fo, layout):
    ws, outside = layout
    refused(run(fo, operation="write", path="../outside/x.txt", content="pwned"))  # same wording as before
    r = run(fo, operation="write", path="Makefile", content="all:")
    assert not r.success and "not allowed for write" in r.error
    outside_unchanged(outside)
    assert data(run(fo, operation="read", path="a.txt")) == "inside\n"  # reads are still the raw content
    assert any(e["name"] == "a.txt" for e in data(run(fo, operation="list", path=".")))
    assert data(run(fo, operation="write", path="a.json", content='{"k": 1}', format="json")).startswith("wrote a.json")
    converted = data(run(fo, operation="convert", path="a.json", format="json", target_format="xml"))
    assert converted == "a.xml" and (ws / "a.xml").exists()  # convert keeps effGen's own result (its path)


def test_the_built_tool_is_effgens_file_operations_with_the_same_identity(fo, layout):
    assert isinstance(fo, FileOperations) and fo.name == "file_operations"
    assert [p.name for p in fo.metadata.parameters] == [p.name for p in FileOperations(allowed_directories=[str(layout[0])]).metadata.parameters]


@linux_only
def test_shell_report_names_the_command_and_keeps_every_field(shell):
    r = sh(shell, "echo out; echo err >&2; exit 4")
    o = out(r)
    assert not r.success and (o["exit_code"], o["stdout"], o["stderr"], o["timed_out"]) == (4, "out\n", "err\n", False)  # raw, unprefixed
    assert r.error == "$ echo out; echo err >&2; exit 4\nexit code: 4\nstdout:\nout\nstderr:\nerr"  # failure semantics intact
    ok = sh(shell, "echo fine")
    assert ok.success and out(ok)["result"] == "$ echo fine\nexit code: 0\nstdout:\nfine" and out(ok)["success"] is True


@linux_only
def test_two_different_silent_commands_read_differently_and_both_really_ran(shell, layout):
    a, b = sh(shell, "mkdir -p one"), sh(shell, "touch two.txt")
    assert a.success and b.success and out(a)["result"] != out(b)["result"]
    assert (layout[0] / "one").is_dir() and (layout[0] / "two.txt").exists()  # the effects, not the text, are the proof


@linux_only
def test_an_identical_command_with_identical_output_reads_identically(shell):
    # No call counter on purpose: effGen's repeat guards must still be able to see a genuinely repeated command.
    assert out(sh(shell, "echo same"))["result"] == out(sh(shell, "echo same"))["result"]


@linux_only
def test_long_commands_keep_their_start_their_size_and_a_hash_of_the_whole(shell, layout):
    head = "cat > " + "x" * 10 + ".txt <<'EOF'\n" + "y" * 400
    one, two = head + "\none\nEOF", head + "\ntwo\nEOF"
    ra, rb = sh(shell, one), sh(shell, two)
    assert ra.success and rb.success and out(ra)["result"] != out(rb)["result"]  # identical first 200 chars, different commands
    first_line = out(ra)["result"].splitlines()[0]
    assert first_line.startswith("$ cat > xxxxxxxxxx.txt <<'EOF' yyyy") and "more chars, sha256" in first_line
    assert len(first_line) < 300
    assert (layout[0] / "xxxxxxxxxx.txt").read_text().endswith("two\n")  # both ran; the second replaced the first


@linux_only
def test_timeout_report_still_says_timed_out_with_the_command(shell):
    r = sh(shell, "sleep 30", timeout=1)
    assert not r.success and out(r)["timed_out"] and r.error.startswith("$ sleep 30\ntimed out after 1s")
    assert "exit code:" in r.error


@linux_only
def test_command_echo_cannot_pass_for_an_effgen_failure_or_hide_one(shell):
    from effgen.core.agent_tool_loop import is_error_result
    failing = sh(shell, "echo Error executing tool; exit 2")  # output that imitates effGen's own error prefix
    assert not failing.success and "exit code: 2" in failing.error
    assert not is_error_result(out(sh(shell, "echo hi"))["result"])  # a success never reads as an error to effGen
