"""Per-agent tool factory: fresh effGen tool instances for exactly the allowlisted names.

The registry is fixed. Web or network tools, `delegate` (the harness adds it to the
orchestrator only) and unknown names are never built. Path confinement for
`file_operations` is effGen's own (`FileOperations`, audit §1.9); `shell` runs commands
in effGen's `SubprocessSandbox` (Linux namespaces) and fails closed when that cannot
be enforced (audit Part 3, T4.2). Neither is a VM or container boundary. `edit_file` (T4.3) replaces one exact
occurrence of a text in an existing file without following symbolic links (`EditFileTool`); it is not a sandbox either.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import hashlib
import math
import os
import platform
import re
import secrets
import shlex
import shutil
import stat
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
    """effGen's `FileOperations`, except that a successful `write` says what it wrote.

    effGen answers a write with the bare path, so rewriting a file (or writing two files) yields the same text
    each time. effGen's loop guards read "the same result text again" as a repeat and end the run (review F2,
    audit Part 3). The summary is read back from the file that is now on disk (path, size, lines, short hash),
    so it changes exactly when the content does; a failed write, and every other operation, are untouched.
    """

    executes_every_call = True  # a repeated read/write must reach the disk; see STATEFUL_TOOL_MARKER

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        out = await super()._execute(**kwargs)
        if kwargs.get("operation") == "write" and out.get("success") and isinstance(out.get("data"), str):
            try:
                written = Path(out["data"])
                body = written.read_bytes()
            except OSError:
                return out  # the write succeeded; a summary that cannot be read back is not worth failing it
            # as_posix: the model sees the same "sub/file.txt" on every platform (Windows would print a backslash).
            shown = written.relative_to(self._allowed_directories[0]).as_posix() \
                if written.is_relative_to(self._allowed_directories[0]) else str(written)
            lines = body.count(b"\n") + (1 if body and not body.endswith(b"\n") else 0)
            out = {**out, "data": f"wrote {shown} ({len(body)} bytes, {lines} line{"" if lines == 1 else "s"}, sha256 {hashlib.sha256(body).hexdigest()[:8]})"}
        return out


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
        raise ToolBuildError(f"tool 'shell' needs Linux (found {platform.system()!r}): effGen's subprocess "
                             "sandbox provides no isolation elsewhere. On Windows, run the harness inside WSL2 "
                             "(not tested).")
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


class ShellTool(BaseTool):
    """`bash -c <command>` in the workspace, inside effGen's `SubprocessSandbox`.

    Enforced (verified by tests, Linux with unprivileged user namespaces): writes only inside the
    workspace, no network, a private PID namespace, wall-clock timeout that kills every process the
    command started, an address-space limit. Not enforced: **reads** (any file the user can read is
    readable, credential stores are masked by a deny-list), CPU/disk/output quotas beyond the
    address-space limit and a process-count `ulimit` that is per user, not per sandbox.
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


class _Refused(Exception):
    """`edit_file` refuses the call; the message goes to the model, nothing was changed."""


class EditFileTool(BaseTool):
    """Replace ONE exact piece of text in an existing file, without rewriting the file (plan T4.3).

    Why it exists: `file_operations` can only overwrite a whole file, which a small model cannot do for a file of a few
    thousand characters inside its output limit (docs/new-session.md, rich_3278 preflight).

    Safety, all enforced here and tested (`tests/test_edit_file.py`): the path is workspace-relative; every component is
    opened with `O_NOFOLLOW` through directory file descriptors starting at the resolved workspace, so no symbolic link
    in the path (parent directories included) is ever followed and a path swapped for a link after validation cannot
    redirect the edit; only a regular, single-link, UTF-8 text file within the size limit is edited; `old_text` must occur
    exactly once (overlapping occurrences count); nothing is normalised (line endings, whitespace, BOM stay as they are);
    the new content goes to a temporary file in the same directory that atomically replaces the target, after the
    target is re-read and found unchanged since it was first read. Not enforced: that re-check is detection, not a lock
    (a writer that changes the file between it and the rename still wins the race), ownership and extended attributes
    are not preserved by the replace, and `.git` components are refused but there is no other deny-list (unlike
    `file_operations`, credential-shaped file names are not special-cased).
    """

    executes_every_call = True  # an identical edit must run again; see STATEFUL_TOOL_MARKER

    MAX_FILE_BYTES = FileOperations.DEFAULT_MAX_FILE_SIZE  # the limit `file_operations` applies to reads
    MAX_TEXT_CHARS = 10_000  # effGen cuts any tool input at 10,000 characters anyway (audit Part 3, 2026-10-28)
    MAX_SHOWN_LINES = 5
    # effGen reads a tool call's JSON input leniently: text that is not valid JSON (a file's `\x1b` copied with ONE backslash) is
    # re-read with `ast.literal_eval` (core/agent_tool_execution.py), which turns it into a real control character. So a model that
    # copies backslashes from a file verbatim sends text that cannot match; the refusal explains the doubling (review P1-1).
    _CONTROL_NAMES = {0x07: "BEL", 0x08: "BS", 0x0B: "VT", 0x0C: "FF", 0x1B: "ESC"}
    # The same mistake in `new_text` is worse than in `old_text`: the call would MATCH and the control character would be written into the
    # file, so `new_text` may not contain C0 control characters other than TAB, LF and CR. Escapes that decode to ordinary characters
    # (`\x41` -> `A`, a naive `\t` -> TAB, `\n` -> LF) cannot be told apart from intended text after parsing and are NOT detected.
    _ESCAPE_RULE = ' In the JSON input write every backslash as two backslashes, in old_text and in new_text: to match or write the text \\x1b send "\\\\x1b"'

    def __init__(self, workspace: Path | str) -> None:
        if not (hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY") and os.open in os.supports_dir_fd and os.rename in os.supports_dir_fd and os.unlink in os.supports_dir_fd):
            raise ToolBuildError("edit_file needs POSIX file APIs (O_NOFOLLOW and directory file descriptors) and is unavailable on this platform")
        self._workspace = resolve_workspace(workspace)
        super().__init__(metadata=ToolMetadata(
            name="edit_file",
            description=(
                "Replace ONE exact piece of text in an existing file without rewriting the whole file. Give the file path "
                "(relative to the project directory), old_text (copied exactly from the file: indentation, quotes and line breaks "
                "included) and new_text (what replaces it; an empty new_text deletes old_text). old_text must occur exactly once in "
                "the file. If it is missing or occurs more than once, nothing is changed and you are told why: copy a few more "
                "neighbouring lines into old_text to make it unique. Read the file first. Prefer this over file_operations write "
                "for changing an existing file. Line endings and whitespace are never altered. In the JSON input write every backslash as two "
                "backslashes, in old_text and in new_text: to match or write the text \\x1b send \\\\x1b. new_text may not contain control "
                "characters other than tab, line feed and carriage return."),
            category=ToolCategory.FILE_OPERATIONS,
            parameters=[
                ParameterSpec(name="path", type=ParameterType.STRING, description="File to edit, relative to the project directory (for example rich/ansi.py)",
                              required=True, min_length=1, max_length=500),
                ParameterSpec(name="old_text", type=ParameterType.STRING, description="The exact text to replace; must occur exactly once in the file",
                              required=True, min_length=1, max_length=self.MAX_TEXT_CHARS),
                ParameterSpec(name="new_text", type=ParameterType.STRING, description="The replacement text (may be empty to delete old_text); control characters other than tab, line feed and carriage return are refused",
                              required=True, max_length=self.MAX_TEXT_CHARS),
            ],
            returns={"type": "object", "properties": {"success": {"type": "boolean"}, "result": {"type": "string"}, "error": {"type": "string"}}},
            tags=["files", "edit"],
            examples=[{"path": "pkg/mod.py", "old_text": "    return a - b", "new_text": "    return a + b"}],
        ))

    def validate_parameters(self, **kwargs: Any) -> tuple[bool, str | None]:
        unknown = sorted(set(kwargs) - {"path", "old_text", "new_text"})
        if unknown:  # effGen would drop them silently; a call with a misspelled parameter must not run
            return False, f"Unknown parameter(s) {unknown}: this tool accepts only 'path', 'old_text' and 'new_text'"
        return super().validate_parameters(**kwargs)

    async def _execute(self, path: str, old_text: str, new_text: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(self._edit, path, old_text, new_text)
        except _Refused as exc:
            return {"success": False, "error": f"edit_file refused: {exc}. Nothing was changed."}
        except OSError as exc:
            return {"success": False, "error": f"edit_file failed with {errno.errorcode.get(exc.errno or 0, 'an OS error')}. Nothing was changed."}

    # -- path handling -------------------------------------------------------------------------------------------------
    @classmethod
    def _control_names(cls, text: str) -> str:
        """Names of the C0 control characters in `text` other than TAB, LF and CR (at most 3, bounded); empty when there are none. Never quotes anything else."""
        found = sorted({ord(c) for c in text if ord(c) < 32 and c not in "\t\n\r"})
        return ", ".join(f"{cls._CONTROL_NAMES.get(o, 'control')} (U+{o:04X})" for o in found[:3])

    @staticmethod
    def _components(path: str) -> list[str]:
        if not isinstance(path, str) or not path.strip() or "\x00" in path:
            raise _Refused("the path must be a non-empty workspace-relative path")
        if path.startswith(("/", "\\")) or re.match(r"[A-Za-z]:", path) or "\\" in path:
            raise _Refused("absolute paths and backslashes are not accepted: use a path relative to the project directory")
        parts = [p for p in path.split("/") if p not in ("", ".")]
        if not parts:
            raise _Refused("the path names no file")
        if ".." in parts:
            raise _Refused("'..' is not allowed in a path")
        if ".git" in parts:
            raise _Refused("files inside .git are not editable")
        return parts

    @staticmethod
    def _open(name: str, flags: int, dir_fd: int, what: str) -> int:
        try:
            return os.open(name, flags | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NOCTTY, dir_fd=dir_fd)
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                raise _Refused(f"{what} does not exist") from None
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise _Refused(f"{what} is a symbolic link or not a directory (symbolic links are never followed)") from None
            raise

    @staticmethod
    def _read(fd: int, limit: int) -> bytes:
        chunks, total = [], 0
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                return b"".join(chunks)
            total += len(chunk)
            if total > limit:
                raise _Refused(f"the file is larger than the {limit} byte limit")
            chunks.append(chunk)

    @staticmethod
    def _occurrences(text: str, old: str) -> list[int]:
        found, i = [], text.find(old)
        while i != -1 and len(found) <= EditFileTool.MAX_SHOWN_LINES:
            found.append(i)
            i = text.find(old, i + 1)  # +1: overlapping occurrences are ambiguous too
        return found

    # -- the edit ------------------------------------------------------------------------------------------------------
    def _edit(self, path: str, old_text: str, new_text: str) -> dict[str, Any]:
        parts = self._components(path)
        shown = "/".join(parts)[:200]
        for label, value in (("old_text", old_text), ("new_text", new_text)):
            if not isinstance(value, str) or "\x00" in value:
                raise _Refused(f"{label} must be text without NUL characters." + (self._ESCAPE_RULE if isinstance(value, str) else ""))
            if len(value) > self.MAX_TEXT_CHARS:
                raise _Refused(f"{label} is longer than {self.MAX_TEXT_CHARS} characters")
            if label == "new_text" and (names := self._control_names(value)):  # checked before the filesystem is touched, like the other argument checks
                raise _Refused(f"new_text contains the control character(s) {names}, which edit_file never writes: a backslash sequence was probably read as an escape."
                               + self._ESCAPE_RULE)
        if not old_text:
            raise _Refused("old_text must not be empty")
        if old_text == new_text:
            raise _Refused("old_text and new_text are identical")
        try:
            new_text.encode("utf-8")
        except UnicodeEncodeError:
            raise _Refused("new_text is not valid text") from None
        fds: list[int] = []
        tmp_name = parent = None
        try:
            fd = os.open(self._workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            fds.append(fd)
            for part in parts[:-1]:
                fd = self._open(part, os.O_RDONLY | os.O_DIRECTORY, fd, f"directory '{part}'")
                fds.append(fd)
            parent, name = fd, parts[-1]
            ffd = self._open(name, os.O_RDONLY | os.O_NONBLOCK, parent, f"file '{shown}'")  # O_NONBLOCK: a FIFO must not hang the open
            fds.append(ffd)
            st = os.fstat(ffd)
            if not stat.S_ISREG(st.st_mode):
                raise _Refused(f"'{shown}' is not a regular file")
            if st.st_nlink != 1:
                raise _Refused(f"'{shown}' has several hard links")
            if not st.st_mode & stat.S_IWUSR:  # a rename would replace it regardless of its mode; an open-for-write would not
                raise _Refused(f"'{shown}' is read-only")
            if st.st_size > self.MAX_FILE_BYTES:
                raise _Refused(f"'{shown}' is larger than the {self.MAX_FILE_BYTES} byte limit")
            data = self._read(ffd, self.MAX_FILE_BYTES)
            if b"\x00" in data:
                raise _Refused(f"'{shown}' looks like a binary file")
            try:
                text = data.decode("utf-8")  # strict; no newline translation: line endings stay byte for byte
            except UnicodeDecodeError:
                raise _Refused(f"'{shown}' is not valid UTF-8 text") from None
            at = self._occurrences(text, old_text)
            if not at:
                hint = " The file uses CRLF line endings: write them (\\r\\n) in old_text." if "\r\n" in text and "\n" in old_text and "\r\n" not in old_text else ""
                raise _Refused(f"old_text was not found in '{shown}'. It must match the file exactly, indentation and line breaks included.{hint}"
                               + (f" old_text contains the control character(s) {names}: a backslash sequence was probably read as an escape.{self._ESCAPE_RULE}"
                                  if (names := self._control_names(old_text)) else ""))
            if len(at) > 1:
                lines = ", ".join(str(text.count("\n", 0, i) + 1) for i in at[:self.MAX_SHOWN_LINES])
                more = "more than " if len(at) > self.MAX_SHOWN_LINES else ""
                raise _Refused(f"old_text occurs {more}{min(len(at), self.MAX_SHOWN_LINES)} times in '{shown}' (starting at lines {lines}); "
                               "add neighbouring lines to old_text so that it matches exactly once")
            line = text.count("\n", 0, at[0]) + 1
            result = text[:at[0]] + new_text + text[at[0] + len(old_text):]
            out = result.encode("utf-8")
            if len(out) > self.MAX_FILE_BYTES:
                raise _Refused("the edited file would exceed the size limit")
            tmp_name = f".{name}.edit-{secrets.token_hex(6)}"
            tfd = os.open(tmp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent)
            try:
                os.fchmod(tfd, stat.S_IMODE(st.st_mode))
                view = memoryview(out)
                while view:
                    view = view[os.write(tfd, view):]
                os.fsync(tfd)
            finally:
                os.close(tfd)
            # Detection, not a lock: the target must still be the file (same inode, same bytes) that the edit was computed from.
            cfd = self._open(name, os.O_RDONLY | os.O_NONBLOCK, parent, f"file '{shown}'")
            try:
                now = os.fstat(cfd)
                unchanged = (now.st_ino, now.st_dev, now.st_size, now.st_mtime_ns) == (st.st_ino, st.st_dev, st.st_size, st.st_mtime_ns) and self._read(cfd, self.MAX_FILE_BYTES) == data
            finally:
                os.close(cfd)
            if not unchanged:
                raise _Refused(f"'{shown}' was changed by someone else while it was being edited; read it again and retry")
            os.rename(tmp_name, name, src_dir_fd=parent, dst_dir_fd=parent)  # POSIX rename(2) replaces the target atomically
            tmp_name = None
            with contextlib.suppress(OSError):
                os.fsync(parent)  # the replace has happened; a failed directory sync does not undo it
            digest = hashlib.sha256(out).hexdigest()[:8]
            return {"success": True, "sha256": digest,
                    "result": (f"edited {shown}: replaced 1 occurrence at line {line} ({old_text.count(chr(10)) + 1} -> {new_text.count(chr(10)) + 1} lines); "
                               f"the file is now {len(out)} bytes, sha256 {digest}")}
        finally:
            if tmp_name is not None and parent is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_name, dir_fd=parent)  # a refused or failed edit leaves no temporary file behind
            for f in fds:
                with contextlib.suppress(OSError):
                    os.close(f)


_REGISTRY: dict[str, Callable[[Path], BaseTool]] = {
    "file_operations": _file_operations,
    "shell": ShellTool,
    "edit_file": EditFileTool,
}
TOOL_NAMES = tuple(_REGISTRY)


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
