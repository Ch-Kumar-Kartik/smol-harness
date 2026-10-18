"""Native Windows `shell` tool: PowerShell inside a Job Object.

Windows has no unprivileged user namespaces, so effGen's Linux sandbox (audit
T4.2: read-only mounts, a network namespace, a PID namespace) has no exact
equivalent. This backend keeps what a Job Object CAN enforce and is explicit,
in its tool description and in every result, about what it cannot:

Enforced (tests/test_shell_win.py, Windows hosts): the command runs in the
workspace; every process it starts is assigned to a Job Object at creation, so
the wall-clock timeout kills the whole tree (detached children included);
stdin is empty; output is tail-truncated; a Job Object that cannot be created
or assigned fails closed (`ShellSandboxError`, nothing runs). Exactly as on
Linux, every leftover process dies when the command ends (Job close mirrors
the PID namespace exit).

Memory: an *optional* `memory_limit` (a per-process Job commit limit) is verified
empirically before it is accepted, because enforcement is NOT a given: the
kernel stores the configured limit verbatim (it queries back) yet commits far
past it anyway on some Windows builds — measured here, 512 MiB allocated under
both 256m and 64m limits while `TerminateJobObject` on the same Job worked.
A limit that is configured but advisory is a lie, so a build-time probe runs a
child that commits above a small Job limit and refuses `memory_limit`
(`ToolBuildError`) unless the commit actually fails. The default build claims
no memory limit.

Not enforced (and never claimed): writes (any writable path is writable,
workspace siblings included), reads (same as Linux, audit Part 3 R1) and
network access. This is the accepted scope for the shell on every platform:
trusted local development and controlled test repositories only
(docs/new-session.md, 2026-10-17).

The command never appears on a command line: prelude + command + epilogue are
composed into one temp script run with `-File` (BOM-encoded, so Windows
PowerShell 5.1 reads it as UTF-8). A command whose first statement is a
`param(...)` block broke under the old `& { }` wrapper anyway; as a script
file it is simply a parse error like any other syntax error.
"""

from __future__ import annotations

import asyncio
import ctypes
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from effgen.tools.base_tool import BaseTool, ParameterSpec, ParameterType, ToolCategory, ToolMetadata

from smol_harness.delegation import remaining_worker_seconds
from smol_harness.tools import (  # noqa: F401  (ShellSandboxError/ToolBuildError re-exported)
    ShellSandboxError, ToolBuildError, _echo, _report, _tail, resolve_workspace,
)


# ---- kernel32 Job Object API (loaded once, lazily; pure stdlib) -----------------
_k32: ctypes.WinDLL | None = None

_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
_JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JobObjectExtendedLimitInformation = 9
_CREATE_SUSPENDED = 0x00000004
_CREATE_NO_WINDOW = 0x08000000
_PROCESS_SET_QUOTA = 0x00000100
_PROCESS_TERMINATE = 0x0001
_TH32CS_SNAPTHREAD = 0x00000004
_THREAD_SUSPEND_RESUME = 0x0002
_INVALID_HANDLE = ctypes.c_size_t(-1).value
# `timeout`'s SIGKILL exit code on Linux; reused so timeout reports read the same.
_KILLED_EXIT_CODE = 137


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                 "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _THREADENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint32),
        ("cntUsage", ctypes.c_uint32),
        ("th32ThreadID", ctypes.c_uint32),
        ("th32OwnerProcessID", ctypes.c_uint32),
        ("tpBasePri", ctypes.c_int32),
        ("tpDeltaPri", ctypes.c_int32),
        ("dwFlags", ctypes.c_uint32),
    ]


def _kernel32() -> ctypes.WinDLL:
    """kernel32 with the Job Object functions bound exactly once per process."""
    global _k32
    if _k32 is None:
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateJobObjectW.restype = ctypes.c_void_p
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        k.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
        k.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        k.Thread32First.restype = ctypes.c_int
        k.Thread32First.argtypes = [ctypes.c_void_p, ctypes.POINTER(_THREADENTRY32)]
        k.Thread32Next.restype = ctypes.c_int
        k.Thread32Next.argtypes = [ctypes.c_void_p, ctypes.POINTER(_THREADENTRY32)]
        k.OpenProcess.restype = ctypes.c_void_p
        k.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        k.OpenThread.restype = ctypes.c_void_p
        k.OpenThread.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        k.ResumeThread.restype = ctypes.c_uint32
        k.ResumeThread.argtypes = [ctypes.c_void_p]
        k.AssignProcessToJobObject.restype = ctypes.c_int
        k.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        k.SetInformationJobObject.restype = ctypes.c_int
        k.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        k.TerminateJobObject.restype = ctypes.c_int
        k.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k.TerminateProcess.restype = ctypes.c_int
        k.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k.CloseHandle.restype = ctypes.c_int
        k.CloseHandle.argtypes = [ctypes.c_void_p]
        _k32 = k
    return _k32


class _JobObject:
    """One run's Job Object. `kill()` ends every process in it; closing also does
    (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE), so stragglers cannot survive this object.
    `memory_limit_bytes=None` sets no memory limit at all (nothing is claimed)."""

    def __init__(self, memory_limit_bytes: int | None) -> None:
        k = _kernel32()
        self._k = k
        self._handle = k.CreateJobObjectW(None, None)
        if not self._handle or self._handle == _INVALID_HANDLE:
            raise OSError(f"CreateJobObjectW failed (error {ctypes.get_last_error()})")
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        flags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if memory_limit_bytes is not None:
            info.ProcessMemoryLimit = info.JobMemoryLimit = memory_limit_bytes
            flags |= _JOB_OBJECT_LIMIT_PROCESS_MEMORY | _JOB_OBJECT_LIMIT_JOB_MEMORY
        info.BasicLimitInformation.LimitFlags = flags
        if not k.SetInformationJobObject(self._handle, _JobObjectExtendedLimitInformation,
                                         ctypes.byref(info), ctypes.sizeof(info)):
            code = ctypes.get_last_error()
            self.close()
            raise OSError(f"SetInformationJobObject failed (error {code})")

    def assign(self, pid: int) -> None:
        """Put a suspended process into this Job before it runs a single instruction."""
        k = self._k
        handle = k.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
        if not handle or handle == _INVALID_HANDLE:
            raise OSError(f"OpenProcess({pid}) failed (error {ctypes.get_last_error()})")
        try:
            if not k.AssignProcessToJobObject(self._handle, handle):
                code = ctypes.get_last_error()
                raise OSError(f"AssignProcessToJobObject({pid}) failed (error {code})")
        finally:
            k.CloseHandle(handle)

    def kill(self, exit_code: int = _KILLED_EXIT_CODE) -> None:
        if self._handle:
            self._k.TerminateJobObject(self._handle, exit_code)

    def close(self) -> None:
        """KILL_ON_JOB_CLOSE: closing the last handle ends everything still inside."""
        if self._handle:
            self._k.CloseHandle(self._handle)
            self._handle = None


def _resume_primary_thread(pid: int) -> None:
    """Resume the suspended main thread (CREATE_SUSPENDED children race us no more)."""
    k = _kernel32()
    snapshot = k.CreateToolhelp32Snapshot(_TH32CS_SNAPTHREAD, 0)
    if not snapshot or snapshot == _INVALID_HANDLE:
        raise OSError(f"CreateToolhelp32Snapshot failed (error {ctypes.get_last_error()})")
    entry = _THREADENTRY32()
    entry.dwSize = ctypes.sizeof(entry)
    resumed = False
    try:
        ok = k.Thread32First(snapshot, ctypes.byref(entry))
        while ok:
            if entry.th32OwnerProcessID == pid:
                thread = k.OpenThread(_THREAD_SUSPEND_RESUME, False, entry.th32ThreadID)
                if thread:
                    try:
                        if k.ResumeThread(thread) == 0xFFFFFFFF:
                            raise OSError(f"ResumeThread({pid}) failed (error {ctypes.get_last_error()})")
                    finally:
                        k.CloseHandle(thread)
                    resumed = True
                break
            ok = k.Thread32Next(snapshot, ctypes.byref(entry))
    finally:
        k.CloseHandle(snapshot)
    if not resumed:
        raise OSError(f"could not resume the suspended command (pid {pid})")


def _terminate_pid(pid: int) -> None:
    """Best-effort kill for a process that never joined its Job (nothing left running)."""
    k = _kernel32()
    handle = k.OpenProcess(_PROCESS_TERMINATE, False, pid)
    if handle and handle != _INVALID_HANDLE:
        try:
            k.TerminateProcess(handle, _KILLED_EXIT_CODE)
        finally:
            k.CloseHandle(handle)


# ---- host checks ---------------------------------------------------------------------------
def require_shell_host_windows() -> None:
    """Fail closed unless this host can run the native Windows shell (checked statically)."""
    if platform.system() != "Windows":
        raise ToolBuildError(f"tool 'shell' needs Windows (found {platform.system()!r}): this is the native "
                             "Windows backend; the Linux-namespace backend builds only on Linux")
    if shutil.which("powershell") is None and shutil.which("pwsh") is None:
        raise ToolBuildError("tool 'shell' needs PowerShell on PATH (powershell.exe or pwsh)")


def _limit_bytes(memory_limit: str) -> int:
    m = re.fullmatch(r"([1-9]\d*)([mg])", memory_limit)
    if not m:
        raise ToolBuildError(f"memory_limit must look like '512m' or '2g', got {memory_limit!r}")
    return int(m.group(1)) * (1 << 30 if m.group(2) == "g" else 1 << 20)


# ---- the one-time empirical memory-limit probe ----------------------------------------------
# The kernel ACCEPTS Job commit limits (they query back verbatim) but on some Windows
# builds it never applies them: a process commits 512 MiB under a 64m limit, silently.
# Measured here (2026-10-18, Windows 11) on every code path tried — per-process flag
# only, per-job flag only, nested and single Job membership, CREATE_SUSPENDED and not.
# Kill limits and ActiveProcessLimit on the same Job DO enforce. Since a configured
# limit that is not enforced is a lie, a `memory_limit` builds only where this probe
# proves the commit actually fails. Cached for the process: positive results are cheap
# to reuse; refusals re-test on the next build (cheap anyway, ~2 s).

_memory_probe_limit = 16 << 20          # a Job limit small enough that ~4 commits must fail
_memory_probe_commits = 64 << 20        # 4x the limit: one `bytearray` must blow through it
_memory_limit_enforced: bool | None = None
_PROBE_SUCCESS = "ALLOC-SUCCEEDED"


def _memory_limits_are_enforced() -> bool:
    """One Python child vs a 16m Job commit limit: enforcement is proven, never assumed."""
    global _memory_limit_enforced
    if _memory_limit_enforced is None:
        job: _JobObject | None = None
        enforced = False
        try:
            job = _JobObject(_memory_probe_limit)
            child = subprocess.Popen(  # the probe never touches PowerShell or a workspace
                [sys.executable, "-c",
                 f"x = bytearray({_memory_probe_commits}); print({_PROBE_SUCCESS!r}, flush=True)"],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                creationflags=_CREATE_SUSPENDED | _CREATE_NO_WINDOW,
            )
            job.assign(child.pid)
            _resume_primary_thread(child.pid)
            try:
                out, _err = child.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                out = b""
            enforced = _PROBE_SUCCESS not in out.decode("utf-8", errors="replace")
        except OSError:
            enforced = False  # without a working probe there is no proof; fail closed
        finally:
            if job is not None:
                job.close()  # kills anything the probe child left behind
        _memory_limit_enforced = enforced
    return _memory_limit_enforced


# ---- the tool ------------------------------------------------------------------------------
_PROBE_COMMAND = "Write-Output 'smol-probe-ok'"

# The command text and this epilogue share one script file and one scope, so the exit
# code is the last statement's, like `bash -c`: a native command's exit code, a failing
# cmdlet -> 1, an explicit `exit N` anywhere -> N (it ends the engine immediately).
# Scope matters: after `& { block }`, `$?` does NOT say the block's last command failed
# (measured on Windows PowerShell 5.1), so the epilogue cannot wrap the command.
_PS_PRELUDE = """\
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = 'Continue'
$global:LASTEXITCODE = 0
"""

_PS_EPILOGUE = """\
if ($?) { exit 0 }
if ($LASTEXITCODE -is [int] -and $LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
exit 1
"""


def _script_text(command: str) -> str:
    return f"{_PS_PRELUDE}{command}\n{_PS_EPILOGUE}"


def _temp_file(prefix: str, suffix: str, content: str, *, encoding: str = "utf-8") -> Path:
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
    except BaseException:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise
    return Path(name)


class WindowsShellTool(BaseTool):
    """PowerShell in the workspace, inside a per-run Job Object (see the module docstring)."""

    executes_every_call = True  # a repeated command must run again; see STATEFUL_TOOL_MARKER

    DEFAULT_TIMEOUT_S = 30
    MAX_TIMEOUT_S = 300
    # No memory limit by default: Job commit limits are not trusted until
    # _memory_limits_are_enforced() proves them (see the module docstring). An
    # explicit memory_limit is probe-gated and refuses to build where unenforced.
    DEFAULT_MEMORY_LIMIT: str | None = None
    MAX_COMMAND_CHARS = 10_000
    MAX_OUTPUT_CHARS = 8_000  # per stream, tail kept; a small model's context is the scarce resource
    HARD_DEADLINE_SLACK_S = 15  # our own cancel, in case even the killed job never closes its pipes

    def __init__(self, workspace: Path | str, *, memory_limit: str | None = None,
                 max_timeout_s: int = MAX_TIMEOUT_S) -> None:
        require_shell_host_windows()
        if memory_limit is not None and not re.fullmatch(r"[1-9]\d*[mg]", memory_limit):
            raise ToolBuildError(f"memory_limit must look like '512m' or '2g', got {memory_limit!r}")
        if isinstance(max_timeout_s, bool) or not isinstance(max_timeout_s, int) or not 1 <= max_timeout_s <= self.MAX_TIMEOUT_S:
            raise ToolBuildError(f"max_timeout_s must be an integer in 1..{self.MAX_TIMEOUT_S}, got {max_timeout_s!r}")
        self._workspace = resolve_workspace(workspace)
        self._memory_limit = memory_limit
        self._max_timeout_s = max_timeout_s
        self._verified = False
        if memory_limit is not None and not _memory_limits_are_enforced():
            # The limit configures cleanly but does not bind on this system; claiming
            # it would be exactly the silent lie this backend exists to avoid.
            raise ToolBuildError(
                f"memory_limit {memory_limit!r} is not enforceable on this Windows system: the "
                "kernel records Job commit limits but does not apply them (a probe child "
                "committed past the limit, measured here and now). Build without "
                "memory_limit; the Job Object still bounds the process tree and timeout."
            )
        memory_note = (f"The {memory_limit} per-process memory limit is enforced by the kernel."
                       if memory_limit is not None else "No memory limit is configured.")
        super().__init__(metadata=ToolMetadata(
            name="shell",
            description=(
                "Run a PowerShell command in the project directory (it starts there). Use it to run tests and "
                "search code (Select-String -Recurse). Every process it started is killed when the timeout "
                f"ends; stdin is empty; output is truncated to the last characters. {memory_note} Windows "
                "cannot confine writes or network here: use only in trusted repositories, write paths inside "
                "the project directory only. The result gives the exit code, stdout and stderr."
            ),
            category=ToolCategory.CODE_EXECUTION,
            parameters=[
                ParameterSpec(name="command", type=ParameterType.STRING,
                              description="PowerShell command text, executed as a script",
                              required=True, min_length=1, max_length=self.MAX_COMMAND_CHARS),
                ParameterSpec(name="timeout", type=ParameterType.INTEGER, description="Seconds before the command is killed",
                              required=False, default=min(self.DEFAULT_TIMEOUT_S, max_timeout_s),
                              min_value=1, max_value=max_timeout_s),
            ],
            returns={"type": "object", "properties": {"success": {"type": "boolean"}, "exit_code": {"type": "integer"},
                                                      "stdout": {"type": "string"}, "stderr": {"type": "string"},
                                                      "timed_out": {"type": "boolean"}}},
            timeout_seconds=max_timeout_s + self.HARD_DEADLINE_SLACK_S,  # informational: effGen does not enforce it
            tags=["shell", "job-object", "windows"],
        ))

    def validate_parameters(self, **kwargs: Any) -> tuple[bool, str | None]:
        # effGen drops unknown parameters with a warning (audit §1.8); for a sandbox, a model
        # asking for `network_enabled` or `memory_limit` should be told no, not silently ignored.
        unknown = sorted(set(kwargs) - {"command", "timeout"})
        if unknown:
            return False, f"Unknown parameter(s) {unknown}: this tool accepts only 'command' and 'timeout'"
        command = kwargs.get("command")
        if isinstance(command, str) and (not command.strip() or "\x00" in command):
            return False, "Parameter 'command' must contain a command and no NUL characters"
        return super().validate_parameters(**kwargs)

    async def _execute(self, command: str, timeout: int | None = None) -> dict[str, Any]:
        timeout_s = self.DEFAULT_TIMEOUT_S if timeout is None else timeout
        timeout_s = min(timeout_s, self._max_timeout_s)
        # A worker's deadline is only checked between calls (DeadlineMiddleware), so a command
        # started with a longer timeout would outlive it by up to that timeout (review F8).
        # Bound the effective timeout by the time the worker has left; the model's own
        # argument is untouched, and the report says when this applied.
        left = remaining_worker_seconds()
        limited = left is not None and max(1, math.ceil(left)) < timeout_s
        if limited:
            timeout_s = max(1, math.ceil(left))
        if not self._verified:
            # Before the first real command, prove the wrapper works: a fixed no-op through
            # the exact path the command will take (a degraded host fails here, not later).
            probe = await self._run_wrapped(_PROBE_COMMAND, 5)
            if probe["exit_code"] != 0 or "smol-probe-ok" not in probe["stdout"]:
                raise ShellSandboxError(f"the shell prologue failed (exit {probe['exit_code']}): "
                                        f"{probe['stderr'].strip()}")
            self._verified = True
        start = time.monotonic()
        result = await self._run_wrapped(command, timeout_s)
        elapsed = time.monotonic() - start
        timed_out = result["timed_out"] or (result["exit_code"] == _KILLED_EXIT_CODE and elapsed >= timeout_s - 0.5)
        ok = result["exit_code"] == 0 and not timed_out
        stdout, out_cut = _tail(result["stdout"], self.MAX_OUTPUT_CHARS)
        stderr, err_cut = _tail(result["stderr"], self.MAX_OUTPUT_CHARS)
        report = _report(_echo(command), result["exit_code"], timed_out, timeout_s, stdout, stderr, limited)
        out: dict[str, Any] = {
            "success": ok, "exit_code": result["exit_code"], "timed_out": timed_out, "stdout": stdout, "stderr": stderr,
            "truncated": out_cut or err_cut,
            "memory_limited": self._memory_limit is not None,  # a documented result field: only True where enforced
            # The honest statement of what this backend does NOT enforce on Windows:
            "filesystem_confined": False, "network_isolated": False, "writable_root": None,
            "process_table_isolated": True,  # the Job object: every process dies with the command
        }
        # effGen shows a model only `error` for a failed result and only `result` for a successful
        # one, so the report goes in both: a failing test run must reach the model with its output.
        out["result" if ok else "error"] = report
        return out

    async def _run_wrapped(self, command: str, timeout_s: int) -> dict[str, Any]:
        """Run one command under the Job Object; returns raw stdout/stderr/exit_code/timed_out."""
        script_file = _temp_file("smol-win-run-", ".ps1", _script_text(command), encoding="utf-8-sig")
        job: _JobObject | None = None
        try:
            try:
                job = _JobObject(_limit_bytes(self._memory_limit) if self._memory_limit else None)
            except OSError as e:
                raise ShellSandboxError(f"the Job Object could not be created; refusing to run anything: {e}") from e
            try:
                proc = await asyncio.create_subprocess_exec(
                    *self._powershell_argv(str(script_file)),
                    cwd=str(self._workspace),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    creationflags=_CREATE_SUSPENDED | _CREATE_NO_WINDOW,
                )
            except OSError as e:
                raise ShellSandboxError(f"PowerShell could not be started: {e}") from e
            try:
                try:
                    job.assign(proc.pid)  # fails closed: an unassignable process never runs
                except OSError as e:
                    _terminate_pid(proc.pid)
                    raise ShellSandboxError(f"the command could not be put into the Job Object; "
                                            f"refusing to run it: {e}") from e
                try:
                    _resume_primary_thread(proc.pid)
                except OSError as e:
                    _terminate_pid(proc.pid)  # never leave a suspended process behind
                    raise ShellSandboxError(f"the command could not be started inside the Job Object: {e}") from e
                timed_out = False
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
                except (asyncio.TimeoutError, TimeoutError):
                    timed_out = True
                    job.kill()  # the wall clock ends the whole process tree, not just the shell
                    try:
                        stdout, stderr = await asyncio.wait_for(proc.communicate(),
                                                                timeout=self.HARD_DEADLINE_SLACK_S)
                    except (asyncio.TimeoutError, TimeoutError):
                        raise ShellSandboxError("the killed job did not close its pipes before the hard "
                                                "deadline; processes it started may still be running") from None
            finally:
                if job:  # even a completed command loses its background children (Job close)
                    job.kill(exit_code=0)
                    job.close()
        finally:
            _unlink(script_file)
        return {
            "exit_code": proc.returncode if proc.returncode is not None else _KILLED_EXIT_CODE,
            "stdout": _decode(stdout), "stderr": _decode(stderr), "timed_out": timed_out,
        }

    def _powershell_argv(self, wrapper_path: str) -> tuple[str, ...]:
        host = shutil.which("pwsh") or shutil.which("powershell")
        if host is None:  # checked in __init__; a PATH change in between fails closed
            raise ShellSandboxError("PowerShell disappeared from PATH")
        # `Start-Process`-free family: -File cannot be quoted wrong and -NonInteractive keeps
        # the host from breaking on a prompt; the command never appears on the command line.
        return host, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-OutputFormat", "Text", \
            "-File", wrapper_path


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _decode(raw: bytes | None) -> str:
    """UTF-8 output (the wrapper sets the console encoding), CRLF normalised like on Linux."""
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
