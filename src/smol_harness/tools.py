"""Per-agent tool factory: fresh effGen tool instances for exactly the allowlisted names.

The registry is fixed. Web or network tools, `delegate` (the harness adds it to the
orchestrator only) and unknown names are never built. Path confinement for
`file_operations` is effGen's own (`FileOperations`, audit §1.9). The `shell` backend is
per platform and never degrades silently: on Linux it is effGen's `SubprocessSandbox`
(Linux namespaces, fail closed without them, T4.2); on Windows it is the native Job-Object
backend (`smol_harness.shell_win`), which enforces fewer guarantees and says so. Neither
is a VM or container boundary.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path, PurePath
from typing import Any

from effgen.security.sandbox import SandboxConfig, SandboxResult, SubprocessSandbox
from effgen.tools.base_tool import BaseTool, ParameterSpec, ParameterType, ToolCategory, ToolMetadata
from effgen.tools.builtin.file_ops import FileOperations

from smol_harness.delegation import DELEGATE_TOOL_NAME, STATEFUL_TOOL_MARKER, remaining_worker_seconds


class ToolBuildError(ValueError):
    """A tool cannot be built: unknown/forbidden name, bad workspace, or unavailable tool."""


class ShellSandboxError(RuntimeError):
    """The shell sandbox is unavailable or did not enforce what is required; nothing was trusted."""


def _is_filesystem_root(path: PurePath) -> bool:
    return path.parent == path  # "/" on POSIX, "C:\\" or a UNC share root on Windows


def resolve_workspace(workspace: Path | str) -> Path:
    """Return the resolved workspace; it must be an existing, accessible, non-dangerous directory."""
    if not str(workspace).strip():  # Path("") would silently mean the current directory
        raise ToolBuildError("workspace must be a non-empty path")
    try:
        path = Path(workspace).resolve(strict=True)
    except (OSError, RuntimeError) as e:
        raise ToolBuildError(f"workspace {str(workspace)!r} does not exist or cannot be resolved: {e}") from e
    if not path.is_dir():
        raise ToolBuildError(f"workspace {str(path)!r} is not a directory")
    try:
        home = Path.home().resolve()
    except (OSError, RuntimeError):  # no resolvable home: only the root check can apply
        home = None
    # A filesystem root, or the home directory and everything above it, is never a project workspace.
    if _is_filesystem_root(path) or (home is not None and (path == home or path in home.parents)):
        raise ToolBuildError(f"workspace {str(path)!r} is a filesystem root or contains the home directory; "
                             "use a project directory")
    if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
        raise ToolBuildError(f"workspace {str(path)!r} is not readable, writable and searchable by this process")
    return path


class WorkspaceFileOperations(FileOperations):
    """effGen's `FileOperations`, except that a successful `write` says what it wrote, and on
    Windows what it wrote is byte-exact (effGen's text-mode write translates \\n to \\r\\n there;
    `_restore_translated_write` undoes exactly that).

    effGen answers a write with the bare path, so rewriting a file (or writing two files) yields the same text
    each time. effGen's loop guards read "the same result text again" as a repeat and end the run (review F2,
    audit Part 3). The summary is read back from the file that is now on disk (path, size, lines, short hash),
    so it changes exactly when the content does; a failed write, and every other operation, are untouched.
    """

    executes_every_call = True  # a repeated read/write must reach the disk; see STATEFUL_TOOL_MARKER

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        out = await super()._execute(**kwargs)
        if kwargs.get("operation") == "write" and out.get("success") and isinstance(out.get("data"), str):
            self._restore_translated_write(Path(out["data"]), kwargs)
            try:
                written = Path(out["data"])
                body = written.read_bytes()
            except OSError:
                return out  # the write succeeded; a summary that cannot be read back is not worth failing it
            shown = written.relative_to(self._allowed_directories[0]).as_posix() \
                if written.is_relative_to(self._allowed_directories[0]) else str(written)
            lines = body.count(b"\n") + (1 if body and not body.endswith(b"\n") else 0)
            out = {**out, "data": f"wrote {shown} ({len(body)} bytes, {lines} line{"" if lines == 1 else "s"}, sha256 {hashlib.sha256(body).hexdigest()[:8]})"}
        return out

    @staticmethod
    def _restore_translated_write(path: Path, kwargs: dict[str, Any]) -> None:
        """Rewrite a just-written file byte-exactly when Windows text-mode translation changed it.

        effGen writes with `open(path, "w")`, so on Windows the "\\n" the agent sent reaches the
        disk as "\\r\\n", and an actual "\\r\\n" as "\\r\\r\\n" — neither round-trips, and hashes,
        tests and git then see different bytes than the agent wrote. The rewrite is exact: the
        expected bytes are recomputed the way effGen formats the content (plain text, or JSON
        re-indented with the same `dumps` call). Anything else (csv's own \\r\\n terminators, a
        non-UTF-8 encoding, an unrecognised shape) is left untouched.
        """
        fmt = kwargs.get("format")
        if os.name != "nt" or fmt == "csv" or kwargs.get("encoding", "utf-8") != "utf-8":
            return
        try:
            if fmt == "json":
                expected = json.dumps(json.loads(kwargs["content"]), indent=2, ensure_ascii=False).encode("utf-8")
            else:
                expected = kwargs["content"].encode("utf-8")
            body = path.read_bytes()
        except (KeyError, TypeError, ValueError, OSError):
            return
        if body == expected:
            return
        if body.replace(b"\r\n", b"\n") == expected:  # only the translation differs
            try:
                path.write_bytes(expected)  # the agent's bytes, exactly, on every platform
            except OSError:
                pass


def _file_operations(workspace: Path) -> BaseTool:
    tool = WorkspaceFileOperations(allowed_directories=[str(workspace)])
    # effGen silently replaces a missing allowed directory with EFFGEN_WORKSPACE or the
    # current directory (audit §1.9, T4.1). Read back the root it kept so a workspace that
    # vanished after validation fails closed instead of widening confinement. The attribute
    # is private; there is no public accessor.
    if tool._allowed_directories != [workspace]:
        raise ToolBuildError(
            f"file_operations did not confine to {str(workspace)!r} (effGen kept "
            f"{[str(p) for p in tool._allowed_directories]}); refusing to build it"
        )
    return tool


def require_shell_host() -> None:
    """Fail closed unless this host can run effGen's Linux-namespace sandbox (checked statically)."""
    if platform.system() != "Linux":
        # effGen's SubprocessSandbox returns an *unsandboxed* `bash -c` plan off Linux, and
        # DockerSandbox does not mount the workspace, so neither is usable (audit §1.9, T4.2).
        # Native Windows does not need this check: there `shell` builds as the Job-Object
        # backend (smol_harness.shell_win) instead of a WSL2-only arrangement.
        raise ToolBuildError(f"tool 'shell' needs Linux (found {platform.system()!r}): on non-Linux POSIX and "
                             "macOS hosts effGen's sandbox provides no isolation (native Windows is served by the "
                             "Job-Object backend instead; this Linux backend would need WSL2).")
    missing = [b for b in ("bash", "unshare", "timeout") if shutil.which(b) is None]
    if missing:
        raise ToolBuildError(f"tool 'shell' needs {', '.join(missing)} on PATH (bash, util-linux unshare, GNU timeout)")


# effGen's network namespace does not stop *pathname* Unix sockets, which are reached through the filesystem
# (a read-only mount does not stop connect() either). Measured on a desktop host: the Docker daemon, the session
# and system D-Bus, the systemd user manager and gpg-agent were all connectable from inside the sandbox, and
# systemd-resolved answered DNS, which is an egress channel (audit Part 3, T4.2). They all live under /run, so
# an empty read-only tmpfs goes over it before any command. Sockets elsewhere (under $HOME, or in /tmp when the
# workspace is itself under /tmp) stay reachable: a path deny-list cannot cover them.
_HIDE_HOST_SOCKETS = (
    "if [ -d /run ]; then mount -t tmpfs -o ro,nosuid,nodev,noexec none /run 2>/dev/null || "
    "{ echo 'sandbox: could not hide /run, so host sockets would stay reachable' >&2; exit 125; }; fi\n"
)


class _LinuxShellTool(BaseTool):
    """`bash -c <command>` in the workspace, inside effGen's `SubprocessSandbox`.

    Enforced (verified by tests, Linux with unprivileged user namespaces): writes only inside the
    workspace, no network, a private PID namespace, wall-clock timeout that kills every process the
    command started, an address-space limit. Not enforced: **reads** (any file the user can read is
    readable, credential stores are masked by a deny-list), CPU/disk/output quotas beyond the
    address-space limit and a process-count `ulimit` that is per user, not per sandbox.
    On Windows, `_shell_tool_class()` swaps in the native Job-Object backend instead of this class.
    """

    executes_every_call = True  # a repeated command must run again; see STATEFUL_TOOL_MARKER

    DEFAULT_TIMEOUT_S = 30
    MAX_TIMEOUT_S = 300
    DEFAULT_MEMORY_LIMIT = "1g"  # `ulimit -v`; effGen's 256m default is too small for real toolchains (node needs more)
    MAX_COMMAND_CHARS = 10_000
    MAX_OUTPUT_CHARS = 8_000  # per stream, tail kept; a small model's context is the scarce resource
    KILL_AFTER_S = 2  # `timeout -k`: SIGKILL this long after SIGTERM
    # effGen's own timeout path kills only the outer `unshare`, leaves descendants running and then
    # waits for their pipes to close (audit Part 3, T4.2). So the real deadline is enforced *inside*
    # the PID namespace (`timeout`), and effGen's deadline is kept later than it as a backstop.
    SANDBOX_SLACK_S = 5
    HARD_DEADLINE_SLACK_S = 15  # our own cancel, in case even the backstop never returns

    def __init__(self, workspace: Path | str, *, memory_limit: str = DEFAULT_MEMORY_LIMIT,
                 max_timeout_s: int = MAX_TIMEOUT_S) -> None:
        require_shell_host()
        if not re.fullmatch(r"[1-9]\d*[mg]", memory_limit):
            raise ToolBuildError(f"memory_limit must look like '512m' or '2g', got {memory_limit!r}")
        if isinstance(max_timeout_s, bool) or not isinstance(max_timeout_s, int) or not 1 <= max_timeout_s <= self.MAX_TIMEOUT_S:
            raise ToolBuildError(f"max_timeout_s must be an integer in 1..{self.MAX_TIMEOUT_S}, got {max_timeout_s!r}")
        self._workspace = resolve_workspace(workspace)
        self._memory_limit = memory_limit
        self._max_timeout_s = max_timeout_s
        self._sandbox = SubprocessSandbox()  # the backend is pinned: never `auto`, never `off`
        self._verified = False
        temp = Path(tempfile.gettempdir()).resolve()  # the same test effGen applies when it points TMPDIR at the workspace
        self._clean_litter = self._workspace == temp or temp in self._workspace.parents
        super().__init__(metadata=ToolMetadata(
            name="shell",
            description=(
                "Run a bash command in the project directory (it starts there). Use it to run tests and "
                "search code (grep -rn). Writes are allowed only inside the project directory; there is no "
                "network; stdin is empty; every process is killed when the timeout ends; output is truncated "
                "to the last characters. The result gives the exit code, stdout and stderr."
            ),
            category=ToolCategory.CODE_EXECUTION,
            parameters=[
                ParameterSpec(name="command", type=ParameterType.STRING, description="Command line, run with bash -c",
                              required=True, min_length=1, max_length=self.MAX_COMMAND_CHARS),
                ParameterSpec(name="timeout", type=ParameterType.INTEGER, description="Seconds before the command is killed",
                              required=False, default=min(self.DEFAULT_TIMEOUT_S, max_timeout_s),
                              min_value=1, max_value=max_timeout_s),
            ],
            returns={"type": "object", "properties": {"success": {"type": "boolean"}, "exit_code": {"type": "integer"},
                                                       "stdout": {"type": "string"}, "stderr": {"type": "string"},
                                                       "timed_out": {"type": "boolean"}}},
            timeout_seconds=max_timeout_s + self.SANDBOX_SLACK_S,  # informational: effGen does not enforce it
            tags=["shell", "sandbox"],
        ))

    def validate_parameters(self, **kwargs: Any) -> tuple[bool, str | None]:
        # effGen drops unknown parameters with a warning (audit §1.8); for a sandbox, a model asking for
        # `network_enabled` or `memory_limit` should be told no rather than silently ignored.
        unknown = sorted(set(kwargs) - {"command", "timeout"})
        if unknown:
            return False, f"Unknown parameter(s) {unknown}: this tool accepts only 'command' and 'timeout'"
        command = kwargs.get("command")
        if isinstance(command, str) and (not command.strip() or "\x00" in command):
            return False, "Parameter 'command' must contain a command and no NUL characters"
        return super().validate_parameters(**kwargs)

    def _litter(self) -> set[str]:
        try:
            return {name for name in os.listdir(self._workspace) if SANDBOX_LITTER.fullmatch(name)}
        except OSError:
            return set()

    def _remove_new_litter(self, before: set[str]) -> None:
        for name in self._litter() - before:  # only what this very run created, only if empty (rmdir refuses anything else)
            try:
                os.rmdir(self._workspace / name)
            except OSError:
                pass

    async def _run_in_sandbox(self, script: str, timeout_s: int) -> SandboxResult:
        before = self._litter() if self._clean_litter else None
        try:
            return await self._run_in_sandbox_unchecked(script, timeout_s)
        finally:
            if before is not None:
                self._remove_new_litter(before)

    async def _run_in_sandbox_unchecked(self, script: str, timeout_s: int) -> SandboxResult:
        config = SandboxConfig(
            backend="subprocess", timeout=timeout_s + self.SANDBOX_SLACK_S, memory_limit=self._memory_limit,
            docker_image="unused", network_enabled=False, workdir=str(self._workspace),
        )
        try:
            result = await asyncio.wait_for(self._sandbox.run(script, "bash", config),
                                            timeout_s + self.SANDBOX_SLACK_S + self.HARD_DEADLINE_SLACK_S)
        except TimeoutError:
            raise ShellSandboxError("the sandbox did not return before its hard deadline; processes it "
                                    "started may still be running") from None
        if result.error:
            raise ShellSandboxError(f"sandbox error: {result.error}")
        # effGen degrades silently (to plain `bash`) when namespaces are unavailable and only says so in
        # these flags, so every run is checked against the full set we depend on.
        lacking = [name for name, ok in (
            ("write confinement", result.filesystem_confined and result.writable_root == str(self._workspace)),
            ("network isolation", result.network_isolated),
            ("a private process table", result.process_table_isolated),
        ) if not ok]
        if lacking:
            raise ShellSandboxError(f"the sandbox did not enforce {', '.join(lacking)} on this host "
                                    "(unprivileged user namespaces unavailable?); refusing to use the shell")
        return result

    async def _execute(self, command: str, timeout: int | None = None) -> dict[str, Any]:
        timeout_s = self.DEFAULT_TIMEOUT_S if timeout is None else timeout
        timeout_s = min(timeout_s, self._max_timeout_s)
        # A worker's deadline is only checked between calls (DeadlineMiddleware), so a command started with a longer
        # timeout would outlive it by up to that timeout (review F8). Bound the effective timeout by the time the
        # worker has left; the model's own argument is not touched, and the report says when this applied.
        left = remaining_worker_seconds()
        limited = left is not None and max(1, math.ceil(left)) < timeout_s
        if limited:
            timeout_s = max(1, math.ceil(left))
        if not self._verified:
            # Before the first real command, prove the sandbox is enforced by running a fixed no-op through
            # the same path. A degraded effGen sandbox would run *that* unconfined, never a model command.
            probe = await self._run_in_sandbox(_HIDE_HOST_SOCKETS + "true\n", 5)
            if probe.exit_code != 0:
                raise ShellSandboxError(f"the sandbox prologue failed (exit {probe.exit_code}): {probe.stderr.strip()}")
            self._verified = True
        # `timeout` runs inside the PID namespace: when bash -s (PID 1 there) exits, the kernel kills every
        # process left in it, including setsid/nohup daemons. </dev/null keeps the command from reading the script.
        script = (_HIDE_HOST_SOCKETS +
                  f"timeout -k {self.KILL_AFTER_S} {timeout_s} bash -c {shlex.quote(command)} </dev/null\nexit $?\n")
        start = time.monotonic()
        result = await self._run_in_sandbox(script, timeout_s)
        elapsed = time.monotonic() - start
        # `timeout` exits 124 (137 after SIGKILL); a command's own `exit 124` is told apart by elapsed time.
        timed_out = result.timed_out or (result.exit_code in (124, 137) and elapsed >= timeout_s - 0.5)
        ok = result.exit_code == 0 and not timed_out
        stdout, out_cut = _tail(result.stdout, self.MAX_OUTPUT_CHARS)
        stderr, err_cut = _tail(result.stderr, self.MAX_OUTPUT_CHARS)
        report = _report(_echo(command), result.exit_code, timed_out, timeout_s, stdout, stderr, limited)
        out: dict[str, Any] = {
            "success": ok, "exit_code": result.exit_code, "timed_out": timed_out, "stdout": stdout, "stderr": stderr,
            "truncated": out_cut or err_cut, "filesystem_confined": True, "network_isolated": True,
            "writable_root": str(self._workspace),
        }
        # effGen shows a model only `error` for a failed result and only `result` for a successful one, so the
        # report goes in both: a failing test run must reach the model with its output.
        out["result" if ok else "error"] = report
        return out


def _tail(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return f"[... {len(text) - limit} earlier characters omitted]\n{text[-limit:]}", True


ECHO_CHARS = 200

# effGen's sandbox prologue runs `mktemp -d` (to mask credential stores) and never removes the directory, so every command
# leaves one empty `tmp.XXXXXXXXXX` behind (audit Part 3, T5.2a). Without a workspace under the system temp directory it lands
# in the host /tmp, which is shared with other programs and therefore left alone; with one, TMPDIR is the workspace itself
# and the litter lands where the agent can list it.
SANDBOX_LITTER = re.compile(r"tmp\.[A-Za-z0-9]{10}")


def _echo(command: str) -> str:
    """The command as one line, for the report. A long one keeps its start, its size and a hash of the whole.

    Without the hash two long commands that share their first lines (for example two heredoc file writes) would
    read the same when they print nothing.
    """
    line = " ".join(command.split())
    if len(line) <= ECHO_CHARS:
        return line
    digest = hashlib.sha256(command.encode()).hexdigest()[:8]
    return f"{line[:ECHO_CHARS]}... [{len(line) - ECHO_CHARS} more chars, sha256 {digest}]"


def _report(echo: str, exit_code: int, timed_out: bool, timeout_s: int, stdout: str, stderr: str, limited: bool = False) -> str:
    # The echo makes two different silent commands (`exit code: 0` each) read differently: effGen ends a run in
    # which different calls return the same result text (review F2). The structured fields stay unprefixed.
    lines = [f"$ {echo}"]
    if timed_out:
        why = " (all the time the worker had left)" if limited else ""
        lines.append(f"timed out after {timeout_s}s{why}: the command and every process it started were killed")
    lines.append(f"exit code: {exit_code}")
    for label, text in (("stdout", stdout), ("stderr", stderr)):
        if text.strip():
            lines.append(f"{label}:\n{text.rstrip()}")
    return "\n".join(lines)


def _shell_tool_class() -> type:
    """The shell tool for this platform, chosen once per process, never guessed.

    `shell` on POSIX is `_LinuxShellTool`: effGen's `SubprocessSandbox` with Linux user
    namespaces. On Windows there are no unprivileged user namespaces, so the native
    Job-Object backend (`smol_harness.shell_win.WindowsShellTool`) takes over: same name,
    same parameters, same result shape, differently enforced guarantees (each reported
    honestly by the tool itself). Neither degrades silently: each refuses to build off
    its platform.
    """
    if platform.system() == "Windows":
        from smol_harness.shell_win import WindowsShellTool
        return WindowsShellTool
    return _LinuxShellTool


def _build_shell(workspace: Path) -> BaseTool:
    return _shell_tool_class()(workspace)


_REGISTRY: dict[str, Callable[[Path], BaseTool]] = {
    "file_operations": _file_operations,
    "shell": _build_shell,
}
TOOL_NAMES = tuple(_REGISTRY)


def __getattr__(name: str):  # PEP 562: `ShellTool` resolves to this platform's shell class.
    if name == "ShellTool":
        return _shell_tool_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_tools(names: Sequence[str], workspace: Path | str) -> list[BaseTool]:
    """Build one fresh tool instance per allowlisted name, confined to `workspace`.

    `[]` builds nothing. Names are checked before the workspace, and nothing is built
    unless every name is valid. `shell` raises `ToolBuildError` on hosts that cannot run
    its sandbox (static check here; the namespaces themselves are proven before its first
    command, which then fails with `ShellSandboxError` instead of running).
    """
    if isinstance(names, str):
        raise ToolBuildError(f"tool names must be a list of names, not the string {names!r}")
    for name in names:
        if name == DELEGATE_TOOL_NAME:
            raise ToolBuildError(f"{DELEGATE_TOOL_NAME!r} is not built here: the harness adds it to the orchestrator only")
        if name not in _REGISTRY:
            raise ToolBuildError(f"unknown tool {name!r}. Available: {', '.join(TOOL_NAMES)} "
                                 "(web and network tools are never available)")
    if len(set(names)) != len(names):
        raise ToolBuildError(f"duplicate tool names in {list(names)!r}")
    root = resolve_workspace(workspace)
    return [_REGISTRY[name](root) for name in names]
