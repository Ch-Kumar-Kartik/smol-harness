"""One-task coding-evaluation pilot driver for `rich_3278` (an EXPERIMENT utility, not part of `src/`).

Pipeline, all host-orchestrated:
  1. extract the pinned snapshot twice: an AGENT workspace (no `.git`) and an EVALUATOR pristine copy;
  2. run the existing harness (public API) in a child process against the agent workspace only, with a wall-clock limit;
  3. collect the candidate change by comparing the two trees (never by running git in a directory the agent can write);
  4. evaluate in a disposable, network-less container: pristine + candidate patch + official test patch + pinned command.

The agent has `file_operations` only (confined to its workspace by path validation) and NO shell: repository code is executed
only by the evaluator, inside the container. The gold patch and the test patch are read by this process alone and never
written into anything the agent can reach. See README.md for the security limits. Python 3.12, stdlib + the project's own deps.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
TASK_ID = "rich_3278"
SNAPSHOT_SHA256 = {"rich_3278": "bda96e09e9e02fa372b2cf6c5f3f2760f97d292fb531cc631564632940d6a453"}
EVAL_PACKAGES = ("pytest==8.3.4", "pygments", "markdown-it-py")  # what rich_3278's tests need; installed offline from the dataset's wheels
MAX_CHANGED_FILES, MAX_PATCH_BYTES = 50, 2_000_000
WORKER = "coder"
# Added when edit_file was introduced (the file_operations write tool can only rewrite a whole file, which does not fit a small output limit).
WORKER_PROMPT = ("You change code in a repository. Read the relevant file first with file_operations, then change it with edit_file: copy a short piece of the file exactly "
                 "as old_text (it must occur once) and give its replacement as new_text. In the JSON input write every backslash as two backslashes, in old_text "
                 "and in new_text (to match or write the text \\x1b send \\\\x1b); new_text may not contain control characters. Do not rewrite whole files. "
                 "You cannot run tests. Report what you changed.")

# README section 8.1/8.2 of the competition's HARNESS_README (the official code is NOT in the dataset, so this is reproduced from its text only).
PROTECTED_NAMES = {"conftest.py", "pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg", ".pytest.ini", "sitecustomize.py", "usercustomize.py", "_swegemma_stubs.py"}
PROTECTED_DIRS = {"tests", "test", "testing"}
IGNORED_DIRS = {"__pycache__", ".pytest_cache", "build", "dist"}

# What is known about the snapshot's relation to the declared base commit (docs/new-session.md, T6.1 and the pilot validation of 2026-11-03). Recorded in every
# run so that a result is never read as a base-commit-exact reproduction.
BASIS = {"rich_3278": {"experiment_basis": "snapshot-based",
                       "declared_base_commit_in_snapshot": False,
                       "snapshot_head": "200f23b28e7e848d59063a9c2c96cd67208707ff",
                       "note": "the declared base commit is absent from the snapshot's object store; the snapshot is the repository at its own HEAD (a merge of PR #3232, 2024-07-01), "
                               "its working tree equals that HEAD's tree, and no commit is dated after the task's creation time"}}

DEVIATIONS = (
    "the agent has file_operations (read/list/search/write) and edit_file (exact single-occurrence replacement, this project's tool) and no shell or test runner; the competition agent has run_command, read_file/edit_file/write_file and the code-graph tools",
    "the agent child runs in a systemd-scope + bubblewrap sandbox (no network except a TCP relay to the one model host:port; read-only code; only <run>/agent is a writable bind, plus ephemeral memory-backed /tmp, /dev/shm and the bridge directory); tests and the evaluator are unchanged",
    "the agent workspace and the evaluator copy have no .git directory (not run by the agent; avoids executing a git config the agent can write)",
    "the prompt is a minimal reconstruction of the competition's task prompt, not the official one (no budget text, no graph tools, no workspace layout)",
    "protected-file handling is reproduced from the README text; the official verifier code was not available",
    "success is judged from pytest's exit code and -rA output, not the official JUnit XML check",
    "the declared base_commit is not in the snapshot; the snapshot is the repository at its own HEAD (see README)",
)


class PilotError(Exception):
    """A condition under which the driver refuses to continue (fail closed)."""


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


# ---- the task (evaluator-side data) --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    repo: str
    base_commit: str
    statement: str  # the ONLY task text the agent receives
    snapshot: Path
    snapshot_sha256: str
    test_patch: str = field(repr=False)  # evaluator-only
    test_files: tuple[str, ...] = ()
    required_tests: tuple[str, ...] = ()


def _task_record(tasks_path: Path, task_id: str) -> dict:
    found = [r for r in (json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()) if r.get("instance_id") == task_id]
    if len(found) != 1:
        raise PilotError(f"expected exactly one task {task_id!r} in {tasks_path}, found {len(found)}")
    return found[0]


def with_newline(text: str) -> str:
    """The dataset's patch strings have no final newline (GNU patch then reports 'ends in middle of line'); a patch is a text file."""
    return text if not text or text.endswith("\n") else text + "\n"


def patch_files(text: str) -> list[str]:
    return re.findall(r"^\+\+\+ b/(.+)$", text, re.M)


def load_task(tasks_path: Path, snapshot: Path, task_id: str, snapshot_sha256: str | None = None) -> TaskSpec:
    """Reads the statement and the test patch. The gold patch is deliberately not loaded here (see `load_gold`)."""
    r = _task_record(tasks_path, task_id)
    expected = snapshot_sha256 or SNAPSHOT_SHA256.get(task_id)
    if not expected:
        raise PilotError(f"no pinned snapshot SHA-256 for {task_id!r}: pass snapshot_sha256")
    return TaskSpec(task_id, r["repo"], r["base_commit"], r["problem_statement"], snapshot, expected, with_newline(r["test_patch"]),
                    tuple(patch_files(r["test_patch"])), tuple(dict.fromkeys(re.findall(r"^\+def (test_\w+)", r["test_patch"], re.M))))


def load_gold(tasks_path: Path, task_id: str) -> str:
    """Evaluator-only: used by `reference-check` and the leak scan, never by the agent phase."""
    return with_newline(_task_record(tasks_path, task_id)["patch"])


def build_prompt(spec: TaskSpec) -> str:
    return (f"You are evaluating a software engineering task for repository {spec.repo}.\n\nProblem Statement:\n{spec.statement}\n\n"
            f"## Environment\n- The repository is your workspace. Delegate the change to your worker `{WORKER}` with the delegate tool: it reads files and changes them with small exact text replacements. You may read files yourself with file_operations.\n"
            "- You cannot run commands or tests. Do not modify tests or configuration files: such changes are discarded.\n"
            "- When the change is complete, reply with a short summary.\n")


# ---- snapshots ------------------------------------------------------------------------------------------------------------------
def _member_name(m: tarfile.TarInfo) -> str:
    return m.name[2:] if m.name.startswith("./") else m.name


def extract_snapshot(archive: Path, dest: Path, expected_sha256: str, *, include_git: bool) -> None:
    if sha256_file(archive) != expected_sha256:
        raise PilotError(f"snapshot {archive} does not match the pinned SHA-256 {expected_sha256}")
    if dest.exists() and any(dest.iterdir()):
        raise PilotError(f"destination {dest} is not empty")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        for m in members:
            name = _member_name(m)
            if m.name.startswith("/") or ".." in name.split("/") or "\\" in m.name:
                raise PilotError(f"unsafe archive member {m.name!r}")
            if not (m.isreg() or m.isdir()):
                raise PilotError(f"archive member {m.name!r} is not a plain file or directory (links and special files are refused)")
        keep = [m for m in members if include_git or not (_member_name(m) == ".git" or _member_name(m).startswith(".git/"))]
        tf.extractall(dest, members=keep, filter="data")


def manifest(root: Path) -> dict[str, str]:
    """relative path -> sha256 of every file; refuses links and special files (an agent workspace must contain none)."""
    out = {}
    for d, dirs, files in os.walk(root):
        for name in [*dirs, *files]:
            p = Path(d, name)
            st = p.lstat()
            if p.is_symlink():
                raise PilotError(f"symbolic link in the tree: {p.relative_to(root)}")
            if not (p.is_dir() or p.is_file()):
                raise PilotError(f"special file in the tree: {p.relative_to(root)}")
            if p.is_file():
                if st.st_nlink > 1:
                    raise PilotError(f"hard-linked file in the tree: {p.relative_to(root)}")
                out[p.relative_to(root).as_posix()] = sha256_file(p)
    return out


def leak_scan(workspace: Path, secrets_: list[str], names: tuple[str, ...] = ("tasks.jsonl", "gold.diff", "test.diff")) -> None:
    """Content-based (never extension-based): no file in the agent workspace may contain an evaluator-only text or carry an evaluator file name."""
    manifest(workspace)  # also refuses links
    needles = [s.encode() for s in secrets_ if len(s) >= 40]
    for d, _dirs, files in os.walk(workspace):
        for name in files:
            p = Path(d, name)
            if name in names:
                raise PilotError(f"evaluator file name in the agent workspace: {p.relative_to(workspace)}")
            data = p.read_bytes()
            for n in needles:
                if n in data:
                    raise PilotError(f"evaluator-only text found in the agent workspace: {p.relative_to(workspace)}")


# ---- patches --------------------------------------------------------------------------------------------------------------------
def is_protected(path: str, extra: frozenset[str] | set[str] = frozenset()) -> bool:
    parts = path.split("/")
    base = parts[-1]
    return (path in extra or base in PROTECTED_NAMES or base.endswith(".pth")
            or (base.endswith(".py") and (base.startswith("test_") or base.endswith("_test.py") or any(p in PROTECTED_DIRS for p in parts[:-1]))))


# The exact name EditFileTool gives its temporary file (`.<name>.edit-<12 hex>`). A kill between its creation and the rename can leave one behind.
EDIT_TEMPORARY = re.compile(r"\..+\.edit-[0-9a-f]{12}")


def _ignored(path: str) -> bool:
    parts = path.split("/")
    return any(p in IGNORED_DIRS or p.endswith(".egg-info") for p in parts[:-1]) or parts[-1].endswith(".pyc") or parts[-1] == ".coverage"


def _diff(path: str, old: str | None, new: str | None) -> str:
    def lines(text):
        return [] if text is None else text.splitlines(keepends=True)

    out = []
    for line in difflib.unified_diff(lines(old), lines(new), "/dev/null" if old is None else f"a/{path}", "/dev/null" if new is None else f"b/{path}"):
        out.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    return "".join(out)


def collect_patch(pristine: Path, workspace: Path, extra_protected: frozenset[str] | set[str] = frozenset()) -> tuple[str, dict]:
    """Candidate change = workspace vs pristine, as a plain unified diff; protected files are listed and left out; anything doubtful raises."""
    before, after = manifest(pristine), manifest(workspace)
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p) and not _ignored(p))
    temporaries = [p for p in changed if EDIT_TEMPORARY.fullmatch(p.rsplit("/", 1)[-1])]  # exactly this pattern, nothing broader
    protected = [p for p in changed if is_protected(p, extra_protected) and p not in temporaries]
    included = [p for p in changed if p not in protected and p not in temporaries]
    if len(included) > MAX_CHANGED_FILES:
        raise PilotError(f"{len(included)} changed files exceed the limit of {MAX_CHANGED_FILES}")
    chunks = []
    for p in included:
        try:
            old = (pristine / p).read_bytes().decode() if p in before else None
            new = (workspace / p).read_bytes().decode() if p in after else None
        except UnicodeDecodeError:
            raise PilotError(f"{p} is not valid UTF-8 text: binary changes are not supported") from None
        chunks.append(_diff(p, old, new))
    text = "".join(chunks)
    if len(text.encode()) > MAX_PATCH_BYTES:
        raise PilotError("candidate patch is larger than the limit")
    return text, {"changed_files": changed, "included_files": included, "protected_changed": protected, "edit_temporaries_ignored": temporaries, "patch_sha256": sha256_bytes(text.encode()), "patch_bytes": len(text.encode())}


def split_patch(text: str) -> list[tuple[str, str]]:
    """[(path, block)] of a PLAIN unified diff. Git extended headers, renames, modes and binary patches are refused."""
    blocks, cur, path = [], [], None
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            if cur:
                blocks.append((path, "".join(cur)))
            new, old = lines[i + 1][4:].split("\t")[0].strip(), line[4:].split("\t")[0].strip()
            target = old if new == "/dev/null" else new
            m = re.fullmatch(r"[ab]/(.+)", target)
            if not m or m.group(1).startswith("/") or ".." in m.group(1).split("/"):
                raise PilotError(f"unsupported or unsafe patch path {target!r}")
            path, cur = m.group(1), [line]
        elif cur and (line[:1] in (" ", "+", "-", "@", "\\") or line in ("\n", "")):
            cur.append(line)
        elif line.strip():
            raise PilotError(f"unsupported patch syntax: {line.rstrip()[:60]!r}")
    if cur:
        blocks.append((path, "".join(cur)))
    return blocks


def strip_protected(text: str, extra: frozenset[str] | set[str] = frozenset()) -> tuple[str, list[str]]:
    kept, stripped = [], []
    for path, block in split_patch(text):
        if is_protected(path, extra):
            stripped.append(path)
        else:
            kept.append(block)
    return "".join(kept), stripped


def apply_patch(workspace: Path, text: str) -> dict:
    """GNU patch, no fuzz, dry run first. Paths were validated by split_patch; nothing here executes repository code."""
    base = ["patch", "-p1", "--forward", "--batch", "--fuzz=0", "--no-backup-if-mismatch"]
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"}
    for flags in (["--dry-run"], []):
        done = subprocess.run([*base, *flags], input=text, cwd=workspace, capture_output=True, text=True, env=env, timeout=60)
        if done.returncode != 0:
            return {"ok": False, "dry_run": bool(flags), "output": (done.stdout + done.stderr)[-2000:]}
    return {"ok": True}


# ---- evaluator container --------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ContainerLimits:
    memory: str = "1g"
    cpus: str = "2"
    pids: int = 256
    nofile: int = 1024
    fsize_bytes: int = 104_857_600
    tmp_size: str = "256m"
    test_timeout_s: int = 120
    outer_timeout_s: int = 300


def build_test_script(spec: TaskSpec, limits: ContainerLimits) -> str:
    """The pinned test command, run from /workspace by a non-root user; prints markers the parser relies on."""
    files = " ".join(spec.test_files)
    return ('cd /workspace || exit 2; echo "python: $(/opt/python/bin/python3 --version 2>&1)"; echo "pytest: $(/opt/python/bin/python3 -m pytest --version 2>&1)"; '
            'S=$(date +%s%N); '
            f'timeout -k 5 {limits.test_timeout_s} /opt/python/bin/python3 -m pytest {files} -q -rA -p no:cacheprovider; '
            'RC=$?; E=$(date +%s%N); echo "pytest_exit=$RC wall_ms=$(( (E-S)/1000000 ))"; exit $RC')


def create_argv(name: str, image: str, limits: ContainerLimits, script: str, run_id: str) -> list[str]:
    """Every isolation setting in one place. Deliberately absent: -v/--mount, --privileged, --network (other than none), any socket."""
    return ["docker", "create", "--pull", "never", "--name", name, "--label", "purpose=coding-eval-pilot", "--label", f"run={run_id}",
            "--network", "none", "--ipc", "none", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", f"{os.getuid()}:{os.getgid()}", "--memory", limits.memory, "--memory-swap", limits.memory, "--cpus", limits.cpus,
            "--pids-limit", str(limits.pids), "--ulimit", f"nofile={limits.nofile}:{limits.nofile}", "--ulimit", f"fsize={limits.fsize_bytes}", "--ulimit", "core=0",
            "--tmpfs", f"/tmp:rw,noexec,nosuid,nodev,size={limits.tmp_size}", "--tmpfs", "/data:rw,noexec,nosuid,nodev,size=16m",  # /data: the image's VOLUME, kept off disk
            "--hostname", "pilot", "--stop-timeout", "5",
            "-e", "HOME=/tmp", "-e", "PYTHONPATH=/opt/site", "-e", "PYTHONNOUSERSITE=1", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONHASHSEED=0", "-e", "LC_ALL=C.UTF-8",
            "--entrypoint", "bash", image, "-c", script]


class DockerRunner:
    """Runs one script against a COPY of a workspace. Python and site-packages are copied in too; nothing is bind-mounted."""

    def __init__(self, python_dir: Path, site_dir: Path, run_id: str, *, image: str = "redis:7", limits: ContainerLimits = ContainerLimits()) -> None:
        self.python_dir, self.site_dir, self.run_id, self.image, self.limits = Path(python_dir).resolve(), Path(site_dir), run_id, image, limits

    def _docker(self, *args: str, timeout: float = 60) -> subprocess.CompletedProcess:
        return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)

    def run(self, workspace: Path, script: str, label: str, log_dir: Path) -> dict:
        name = f"coding-eval-{self.run_id}-{label}"
        info: dict = {"image": self.image, "name": name, "limits": vars(self.limits), "timed_out": False}
        try:
            image_id = self._docker("image", "inspect", "--format", "{{.Id}}", self.image)
            if image_id.returncode != 0:
                raise PilotError(f"image {self.image!r} is not available locally (the driver never pulls): {image_id.stderr.strip()}")
            info["image_id"] = image_id.stdout.strip()
            created = subprocess.run(create_argv(name, self.image, self.limits, script, self.run_id), capture_output=True, text=True, timeout=60)
            if created.returncode != 0:
                raise PilotError(f"docker create failed: {created.stderr.strip()}")
            for src, dst in ((self.python_dir, "/opt/python"), (self.site_dir, "/opt/site"), (workspace, "/workspace")):
                cp = self._docker("cp", "-q", str(src), f"{name}:{dst}", timeout=300)
                if cp.returncode != 0:
                    raise PilotError(f"docker cp to {dst} failed: {cp.stderr.strip()}")
            host = json.loads(self._docker("inspect", name).stdout)[0]["HostConfig"]
            wrong = [k for k, ok in (("NetworkMode", host["NetworkMode"] == "none"), ("Privileged", host["Privileged"] is False), ("CapDrop", host["CapDrop"] == ["ALL"]),
                                     ("Binds", not host.get("Binds")), ("Mounts", not host.get("Mounts"))) if not ok]
            if wrong:
                raise PilotError(f"container isolation not as configured ({', '.join(wrong)}); refusing to run")
            info["isolation"] = {k: host[k] for k in ("NetworkMode", "Privileged", "CapDrop", "Memory", "PidsLimit", "NanoCpus")}
            start = time.monotonic()
            try:
                done = subprocess.run(["docker", "start", "-a", name], capture_output=True, text=True, timeout=self.limits.outer_timeout_s)
                out, err, code = done.stdout, done.stderr, done.returncode
            except subprocess.TimeoutExpired as exc:
                self._docker("kill", name)
                info["timed_out"] = True
                out, err, code = "", "", 124  # partial output of a timed-out run is not trusted
            info["wall_s"] = round(time.monotonic() - start, 2)
            state = json.loads(self._docker("inspect", name).stdout)[0]["State"]
            info["oom_killed"] = bool(state.get("OOMKilled"))
        finally:
            self._docker("rm", "-f", "-v", name)  # only the container this run created
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{label}.stdout.txt").write_text(out)
        (log_dir / f"{label}.stderr.txt").write_text(err)
        return {**info, "exit_code": code, "stdout": out, "stderr": err}


def parse_pytest(stdout: str) -> dict:
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    summary = next((ln for ln in reversed(stdout.splitlines()) if re.search(r"\b(passed|failed|error)", ln) and " in " in ln), "")
    for n, word in re.findall(r"(\d+) (passed|failed|errors?|skipped)", summary):
        counts["errors" if word.startswith("error") else word] = int(n)
    status: dict[str, str] = {}
    for m in re.finditer(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS) (\S+)", stdout, re.M):
        status[m.group(2).split(" - ")[0]] = m.group(1)
    exit_code = re.search(r"^pytest_exit=(\d+)", stdout, re.M)
    return {**counts, "node_status": status, "pytest_exit": int(exit_code.group(1)) if exit_code else None,
            "python": (re.search(r"^python: (.+)$", stdout, re.M) or [None, None])[1], "pytest": (re.search(r"^pytest: (.+)$", stdout, re.M) or [None, None])[1]}


def judge(parsed: dict, exit_code: int, required: tuple[str, ...]) -> tuple[bool, list[str]]:
    """Resolved only if pytest exited 0, nothing failed or errored, something passed, and every required test node passed (never skipped)."""
    problems = []
    if exit_code != 0:
        problems.append(f"pytest exit status {exit_code}")
    if parsed["failed"] or parsed["errors"]:
        problems.append("failures or errors reported")
    if not parsed["passed"]:
        problems.append("no test passed")
    for name in required:
        nodes = {k: v for k, v in parsed["node_status"].items() if k.split("::", 1)[-1].split("[")[0] == name}
        if not nodes or any(v != "PASSED" for v in nodes.values()):
            problems.append(f"required test {name} did not fully pass")
    return not problems, problems


def evaluate(spec: TaskSpec, candidate: str, work_dir: Path, runner, *, label: str = "candidate", run_even_if_empty: bool = False,
             limits: ContainerLimits = ContainerLimits()) -> dict:
    """Fresh pristine copy -> candidate patch (protected files stripped) -> official test patch -> pinned command in the container."""
    candidate = with_newline(candidate)
    rec: dict = {"schema_version": SCHEMA_VERSION, "task_id": spec.task_id, "label": label, "started": now(), "snapshot_sha256": spec.snapshot_sha256,
                 "test_command": build_test_script(spec, limits), "required_tests": list(spec.required_tests), "candidate_sha256": sha256_bytes(candidate.encode()),
                 "candidate_bytes": len(candidate.encode()), "steps": {}, "result": None, "resolved": False, "outcome": "error", "problems": []}
    workspace = work_dir / "workspace"
    try:
        extract_snapshot(spec.snapshot, workspace, spec.snapshot_sha256, include_git=False)
        rec["steps"]["pristine"] = "extracted"
        extra = frozenset(spec.test_files)
        kept, stripped = strip_protected(candidate, extra) if candidate.strip() else ("", [])
        rec["steps"]["protected_stripped"] = stripped
        if not kept.strip() and not run_even_if_empty:
            rec.update(outcome="unresolved_empty_patch", problems=["the candidate patch is empty after protected files were removed"])
            return rec
        if kept.strip():
            applied = apply_patch(workspace, kept)
            rec["steps"]["candidate_patch"] = applied
            if not applied["ok"]:
                rec.update(outcome="unresolved_patch_failed", problems=["the candidate patch does not apply to the pristine snapshot"])
                return rec
        official = apply_patch(workspace, spec.test_patch)
        rec["steps"]["test_patch"] = {"ok": official["ok"]}
        if not official["ok"]:
            rec.update(outcome="error", problems=["the official test patch does not apply after the candidate patch"])
            return rec
        ran = runner.run(workspace, build_test_script(spec, limits), label, work_dir / "logs")
        parsed = parse_pytest(ran["stdout"])
        resolved, problems = judge(parsed, ran["exit_code"], spec.required_tests)
        rec["result"] = {k: v for k, v in ran.items() if k not in ("stdout", "stderr")} | {k: v for k, v in parsed.items() if k != "node_status"}
        rec["result"]["stdout_sha256"] = sha256_bytes(ran["stdout"].encode())
        rec["result"]["output_tail"] = ran["stdout"][-1500:]
        if ran.get("timed_out"):  # the OUTER timeout (`docker start` was killed, its output discarded): no marker can exist. The INNER one (`timeout` inside the script) prints the marker and is judged below.
            rec.update(outcome="error", problems=[f"the evaluation container exceeded the outer timeout of {limits.outer_timeout_s}s (output discarded, so no pytest_exit marker); not a verdict on the candidate"])
        elif ran.get("oom_killed"):  # the engine killed the container for memory: the test result, if any, is not trustworthy (a very memory-hungry patch looks the same)
            rec.update(outcome="error", problems=["the container was OOM-killed; the result is not a verdict on the candidate"])
        elif parsed["pytest_exit"] is None:  # the script prints the marker after pytest returns, however pytest ended: no marker = the container or the shell died
            rec.update(outcome="error", problems=[f"the test run did not complete (no pytest_exit marker; container exit code {ran['exit_code']})"])
        else:
            rec.update(resolved=resolved, problems=problems, outcome="resolved" if resolved else "unresolved_tests_failed")
    except PilotError as exc:
        rec.update(outcome="error", problems=[str(exc)])
    except Exception as exc:  # fail closed: an evaluator fault is never a pass
        rec.update(outcome="error", problems=[f"{type(exc).__name__}: {exc}"])
    finally:
        rec["finished"] = now()
        shutil.rmtree(workspace, ignore_errors=True)  # the evaluator's own copy only
    return rec


# ---- the agent sandbox -----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SandboxLimits:
    memory_max: str = "2G"  # cgroup MemoryMax (no swap); the child needs about 100 MiB
    tasks_max: int = 256  # cgroup TasksMax: processes AND threads
    cpu_quota: str = "200%"
    fsize_bytes: int = 104_857_600  # RLIMIT_FSIZE: the largest single file the child can write
    nofile: int = 1024


def _bytes(size: str) -> int:
    return int(size[:-1]) * {"K": 1024, "M": 1024**2, "G": 1024**3}[size[-1]]


def model_endpoint(base_url: str) -> tuple[str, int]:
    parts = urllib.parse.urlsplit(base_url)
    if parts.scheme != "http" or not parts.hostname or not parts.port:
        raise PilotError(f"the model endpoint must be an http URL with an explicit port (got {base_url!r})")
    return parts.hostname, parts.port


CONNECT_TIMEOUT_S = 10  # only for ESTABLISHING the upstream connection; it is removed once connected (a non-streaming generation sends nothing for as long as it runs)
RELAY_IDLE_S = 600  # a connection that moves no bytes in either direction for this long is closed: the bound on a relay's life (effGen's own client gives up after 60 s)


def _pipe(src: socket.socket, dst: socket.socket) -> None:
    try:
        while data := src.recv(65536):
            dst.sendall(data)
    except OSError:  # includes the idle timeout, a reset and a close from the other thread
        pass
    finally:
        with contextlib.suppress(OSError):
            dst.shutdown(socket.SHUT_WR)


def _drop(sock: socket.socket) -> None:
    """shutdown first: it wakes a thread blocked in recv on this socket, close alone does not."""
    with contextlib.suppress(OSError):
        sock.shutdown(socket.SHUT_RDWR)
    with contextlib.suppress(OSError):
        sock.close()


def _relay(a: socket.socket, b: socket.socket) -> None:
    for s in (a, b):
        s.settimeout(RELAY_IDLE_S)
    other = threading.Thread(target=_pipe, args=(a, b), daemon=True)
    other.start()
    _pipe(b, a)
    other.join(timeout=5)
    _drop(a)
    _drop(b)
    other.join(timeout=5)


class ModelBridge:
    """Host side of the only network path out of the sandbox: a unix socket that forwards every connection to ONE host:port (the model endpoint)."""

    def __init__(self, directory: Path, host: str, port: int) -> None:
        self.path, self.target, self.server, self.closed = directory / "m.sock", (host, port), None, False
        self._lock, self._open = threading.Lock(), set()  # the sockets of the relays in flight, so that shutdown can end them

    def __enter__(self) -> ModelBridge:
        self.server = socket.socket(socket.AF_UNIX)
        self.server.bind(str(self.path))
        os.chmod(self.path, 0o600)
        self.server.listen(16)
        self.thread = threading.Thread(target=self._accept, daemon=True)
        self.thread.start()
        return self

    def _accept(self) -> None:
        while not self.closed:
            try:
                client, _ = self.server.accept()
            except OSError:
                return
            try:
                upstream = socket.create_connection(self.target, timeout=CONNECT_TIMEOUT_S)
                upstream.settimeout(None)  # the connect timeout must not stay on the connected socket (_relay sets the idle bound)
            except OSError:
                client.close()
                continue
            with self._lock:
                if self.closed:  # shut down between accept and here
                    _drop(client)
                    _drop(upstream)
                    return
                self._open.update((client, upstream))
            threading.Thread(target=self._serve, args=(client, upstream), daemon=True).start()

    def _serve(self, client: socket.socket, upstream: socket.socket) -> None:
        try:
            _relay(client, upstream)
        finally:
            with self._lock:
                self._open.difference_update((client, upstream))

    @property
    def active(self) -> int:
        with self._lock:
            return len(self._open) // 2

    def __exit__(self, *exc) -> None:
        with self._lock:
            self.closed = True
            live = list(self._open)
        _drop(self.server)  # shutdown wakes the accept thread, which close alone would leave blocked
        for s in live:  # ends every relay in flight; their threads see EOF/an error and return
            _drop(s)
        with contextlib.suppress(OSError):
            self.path.unlink()


def serve_loopback_bridge(port: int, socket_path: str) -> None:
    """Child side: 127.0.0.1:<port> inside the sandbox (its own loopback) is forwarded to the bridge socket. Runs as a daemon thread."""
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(16)

    def accept() -> None:
        while True:
            try:
                client, _ = server.accept()
            except OSError:
                return
            upstream = socket.socket(socket.AF_UNIX)
            try:
                upstream.connect(socket_path)
            except OSError:
                client.close()
                upstream.close()
                continue
            threading.Thread(target=_relay, args=(client, upstream), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()


def sandbox_command(agent_dir: Path, inner: list[str], limits: SandboxLimits = SandboxLimits(), *, extra_binds: tuple[tuple[str, str], ...] = (), unit: str | None = None) -> list[str]:
    """systemd scope (cgroup limits) > bubblewrap (namespaces, mounts) > prlimit > the command. Mounted: read-only system libraries, the project's venv and
    `src`, the interpreter and this script; read-write: the agent directory and the extra binds. NOT mounted: the home directory, the repository (docs, dataset,
    tests), the evaluator's directories, /etc (but the loader cache), /run, /var, the D-Bus and Docker sockets. A fresh network namespace has only its own loopback."""
    prefix = Path(sys.prefix)
    link = Path(sys.executable)
    base = None
    while link.is_symlink():  # follow the venv's interpreter link by hand: the first absolute target outside the venv is what must exist inside
        target = Path(os.readlink(link))
        target = target if target.is_absolute() else link.parent / target
        if prefix not in target.parents:
            base = target.parent.parent
            break
        link = target
    base = base or Path(sys.executable).resolve().parent.parent
    ro = [("/usr", "/usr"), (str(prefix), str(prefix)), (str(base), str(base)), (str(REPO / "src"), str(REPO / "src")), (str(Path(__file__).resolve()), str(Path(__file__).resolve()))]
    if Path("/etc/ld.so.cache").exists():
        ro.append(("/etc/ld.so.cache", "/etc/ld.so.cache"))
    argv = ["systemd-run", "--user", "--scope", "--quiet", "--collect", *(["--unit", unit] if unit else []),
            "-p", f"MemoryMax={limits.memory_max}", "-p", "MemorySwapMax=0", "-p", f"TasksMax={limits.tasks_max}", "-p", f"CPUQuota={limits.cpu_quota}", "--",
            "bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv"]
    for top in ("/lib", "/lib64", "/bin", "/sbin"):  # mirror the host's usr-merge links
        if os.path.islink(top):
            argv += ["--symlink", os.readlink(top), top]
    for src, dst in ro:
        argv += ["--ro-bind", src, dst]
    argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--bind", str(agent_dir), str(agent_dir)]
    for src, dst in extra_binds:
        argv += ["--bind", src, dst]
    argv += ["--chdir", str(agent_dir)]
    for key, value in (("PATH", "/usr/bin"), ("HOME", "/tmp"), ("LANG", "C.UTF-8"), ("VIRTUAL_ENV", str(prefix)), ("PYTHONDONTWRITEBYTECODE", "1"), ("PYTHONNOUSERSITE", "1")):
        argv += ["--setenv", key, value]
    return argv + ["prlimit", f"--fsize={limits.fsize_bytes}", f"--nofile={limits.nofile}", "--core=0", "--", *inner]


def sandbox_problem(limits: SandboxLimits = SandboxLimits()) -> str | None:
    """None when the sandbox can really be built here; otherwise why not. The driver never falls back to running the agent as the plain host user."""
    if platform.system() != "Linux":
        return "the agent sandbox needs Linux"
    missing = [tool for tool in ("bwrap", "systemd-run", "prlimit") if not shutil.which(tool)]
    if missing:
        return f"{', '.join(missing)} not found"
    probe = ("import json, os, socket\nnet = [l.split(':')[0].strip() for l in open('/proc/net/dev').read().splitlines()[2:]]\ns = socket.socket(); s.settimeout(2)\n"
             "try:\n    s.connect(('1.1.1.1', 53)); out = False\nexcept OSError:\n    out = True\nprint(json.dumps({'net': net, 'external_blocked': out, 'pid': os.getpid()}))\n")
    with tempfile.TemporaryDirectory(prefix="sbx-probe-") as tmp:
        done = subprocess.run(sandbox_command(Path(tmp), [sys.executable, "-c", probe], limits), capture_output=True, text=True, timeout=60, env=_launcher_env())
        if done.returncode != 0:
            return f"the sandbox could not be started: {done.stderr.strip()[-300:]}"
        seen = json.loads(done.stdout.strip().splitlines()[-1])
        if seen["net"] != ["lo"] or not seen["external_blocked"]:
            return f"the sandbox does not isolate the network: {seen}"
        scope = ("import re\npath = re.search(r'^0::(.*)$', open('/proc/self/cgroup').read(), re.M).group(1)\n"
                 "print(open('/sys/fs/cgroup' + path + '/memory.max').read().strip(), open('/sys/fs/cgroup' + path + '/pids.max').read().strip())\n")
        cg = subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={limits.memory_max}", "-p", f"TasksMax={limits.tasks_max}", "--", sys.executable, "-c", scope],
                            capture_output=True, text=True, timeout=60, env=_launcher_env())
        if cg.returncode != 0 or cg.stdout.split() != [str(_bytes(limits.memory_max)), str(limits.tasks_max)]:
            return f"the cgroup limits are not enforced as configured: {(cg.stdout + cg.stderr).strip()[-200:]}"
    return None


def _launcher_env() -> dict[str, str]:
    """The environment of the launcher (systemd-run needs the user session bus). The sandbox itself starts from an empty environment (--clearenv)."""
    keys = ("PATH", "HOME", "LANG", "LC_ALL", "VIRTUAL_ENV", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")
    return {k: os.environ[k] for k in keys if k in os.environ}


# ---- agent phase ----------------------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def generation_overrides(max_tokens: int, temperature: float, seed: int):
    """Same experiment-only override the RTX/DGX runners use (the YAML has no fields for these yet; review F7)."""
    import dataclasses

    import smol_harness.harness as harness_mod

    real = harness_mod._agent_config
    harness_mod._agent_config = lambda spec, model, tools, mode: dataclasses.replace(real(spec, model, tools, mode), max_tokens=max_tokens, temperature=temperature, seed=seed)
    try:
        yield
    finally:
        harness_mod._agent_config = real


def render_config(workspace: Path, a: dict) -> str:
    import yaml

    model = {"provider": a["provider"], "model": a["model"], "context_length": a["context_length"], "base_url": a["base_url"]}
    return yaml.safe_dump({
        "workspace": str(workspace),
        "limits": {"worker_timeout_s": a["worker_timeout_s"]},
        "orchestrator": {"model": model, "tools": ["file_operations"], "max_iterations": a["max_iterations"],
                         "system_prompt": f"You coordinate one worker, {WORKER}. Delegate the code change to it with the delegate tool, naming the file and the change, then answer from its report."},
        "workers": [{"name": WORKER, "description": "reads and edits files in the repository workspace", "model": model, "tools": ["file_operations", "edit_file"],
                     "max_iterations": a["max_iterations"], "system_prompt": WORKER_PROMPT}],
    }, sort_keys=False)


def model_failure(result) -> dict | None:
    """What effGen recorded about WHY the orchestrator's run failed (None for an answered run). Raw facts only; `classify_run` interprets them.

    `stop_reason` and `metadata["reason"]`/`["error"]` are effGen's own (AgentResponse); for a failed provider call `error` carries the provider error's
    `category` (unreachable, timeout, transient, auth, truncation, reasoning_only, ...). A run that failed for another reason has no `category`."""
    response = getattr(result, "response", None)
    if response is None or getattr(response, "success", True):
        return None
    meta = getattr(response, "metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    error = meta.get("error") if isinstance(meta.get("error"), dict) else {}
    return {"stop_reason": str(getattr(response, "stop_reason", None)), "reason": str(meta.get("reason")), "category": error.get("category"),
            "type": error.get("type"), "retryable": error.get("retryable"), "message": str(error.get("message") or "")[:300]}


def run_agent(params: dict) -> int:
    """Child-process entry: the existing harness through its public API, against the agent workspace only."""
    from smol_harness.config import load_config
    from smol_harness.harness import build_harness_from_config, initialize_runtime
    from smol_harness.recorder import ExecutionRecorder

    out = Path(params["out_dir"])
    started, result_doc = time.monotonic(), {"started": now(), "generation_overrides_requested": {k: params[k] for k in ("max_tokens", "temperature", "seed")},
                                             "note": "limits are what was REQUESTED; the server-enforced output limit was not verified (docs/new-session.md U17)"}
    code = 3
    try:
        if params.get("bridge"):  # inside the sandbox: the model endpoint is reached through the bridge, the only way out
            serve_loopback_bridge(params["bridge"]["port"], params["bridge"]["socket"])
        initialize_runtime()
        config = load_config(params["config_path"])
        recorder = ExecutionRecorder(out / "history")
        with generation_overrides(params["max_tokens"], params["temperature"], params["seed"]):
            with build_harness_from_config(config, recorder=recorder) as harness:
                result = harness.run(Path(params["prompt_path"]).read_text())
        result_doc.update(output=result.output, agent_reported_success=result.success, termination=result.termination, model_failure=model_failure(result),
                          workers=[{"worker": r.worker, "status": r.status, "delegation_id": r.result.delegation_id, "iterations": r.result.iterations,
                                    "tool_calls": r.result.tool_calls, "tokens_used": r.result.tokens_used, "error_type": r.result.error_type,
                                    "termination": r.result.termination} for r in result.workers],
                          history_dir=str(recorder.dir), error=None)
        code = 0 if result.success else 1
    except BaseException as exc:  # recorded, never raised past the driver
        result_doc["error"] = f"{type(exc).__name__}: {exc}"[:500]
    result_doc.update(seconds=round(time.monotonic() - started, 2), finished=now())
    write_json(out / "agent_result.json", result_doc)
    return code


def run_agent_phase(agent_dir: Path, params: dict, timeout_s: float, cmd: list[str] | None = None) -> dict:
    """The harness in its own process group so the wall-clock limit can really stop it. A minimal environment: no credentials are forwarded."""
    write_json(agent_dir / "params.json", params)
    cmd = cmd or [sys.executable, str(Path(__file__).resolve()), "agent-child", "--params", str(agent_dir / "params.json")]
    env = _launcher_env() if cmd[0] == "systemd-run" else {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "VIRTUAL_ENV") if k in os.environ}
    start, timed_out = time.monotonic(), False
    with open(agent_dir / "child.stdout.txt", "wb") as so, open(agent_dir / "child.stderr.txt", "wb") as se:
        proc = subprocess.Popen(cmd, cwd=agent_dir, env=env, stdout=so, stderr=se, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            for sig, grace in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, sig)
                try:
                    code = proc.wait(timeout=grace)
                    break
                except subprocess.TimeoutExpired:
                    code = None
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)  # nothing the agent process started survives the phase
    return {"command": cmd, "exit_code": code, "timed_out": timed_out, "seconds": round(time.monotonic() - start, 2)}


# ---- one run --------------------------------------------------------------------------------------------------------------------
def fetch_model_identity(base_url: str, model: str, timeout: float = 5.0, opener=None) -> dict:
    """Ask the Ollama server which model it holds under this tag (`GET /api/tags`: metadata only, no inference). Never raises. Environment proxies are ignored."""
    root = base_url.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({})).open
    try:
        with opener(root + "/api/tags", timeout=timeout) as response:
            data = json.loads(response.read(2_000_000))
    except Exception as exc:  # unreachable endpoint, refused connection, not JSON, ...
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}
    for entry in data.get("models", []) if isinstance(data, dict) else []:
        if model in (entry.get("name"), entry.get("model")):
            details = entry.get("details") or {}
            return {"available": True, "tag": model, "digest": entry.get("digest"), "size_bytes": entry.get("size"), "modified_at": entry.get("modified_at"),
                    "family": details.get("family"), "parameter_size": details.get("parameter_size"), "quantization_level": details.get("quantization_level"), "provider": "ollama"}
    return {"available": False, "reason": f"the server at {root} does not list the model {model!r}"}


def collect_provenance() -> dict:
    """What produced this run: code hashes, repository state, versions and host. Each field is best effort and None when it cannot be read."""
    def attempt(fn):
        try:
            return fn()
        except Exception:
            return None

    def git(*args):
        done = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=30, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"})
        return done.stdout.strip() if done.returncode == 0 else None

    def meminfo():
        return int(next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal")).split()[1])

    return {"driver_sha256": attempt(lambda: sha256_file(Path(__file__).resolve())), "tools_py_sha256": attempt(lambda: sha256_file(REPO / "src" / "smol_harness" / "tools.py")),
            "repo_head": attempt(lambda: git("rev-parse", "HEAD")), "repo_status": attempt(lambda: git("status", "--short").splitlines()),
            "python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(), "mem_total_kib": attempt(meminfo),
            "effgen": attempt(lambda: importlib.metadata.version("effgen")), "pyyaml": attempt(lambda: importlib.metadata.version("pyyaml"))}


# effGen's provider-error categories (models/errors.py) split by what they say about the CAUSE of a failed model call.
TRANSPORT_CATEGORIES = frozenset({"unreachable", "timeout", "transient", "rate_limited", "auth", "not_found", "not_loaded", "resource_exhausted", "fatal"})  # endpoint, tunnel, server, configuration
GENUINE_WORKER_FAILURES = frozenset({"stuck", "tool_failed"})  # a worker termination that states a genuine agent failure (the loop gave up, or the tools failed)
MODEL_LIMIT_CATEGORIES = frozenset({"truncation", "reasoning_only", "refusal"})  # the model answered, but not usably: a property of the model under test, not of the infrastructure


def _agent_state(agent: dict | None, record: dict, reasons: list[str], ambiguous: list[str]) -> str:
    """The agent phase, in precedence order: not run > timed out > no result > harness raised > how the run ended."""
    if not agent:
        reasons.append(f"the agent phase did not run: {record.get('error') or record.get('status')}")
        return "not_run"
    if agent.get("timed_out"):
        return "timed_out"
    result = agent.get("result")
    if result is None:
        reasons.append(f"the agent process left no result (exit code {agent.get('exit_code')}; a cgroup OOM kill or a crash looks like this)")
        return "infrastructure_failure"
    if result.get("error"):
        reasons.append(f"the harness raised: {result['error']}")
        return "infrastructure_failure"
    state = "completed" if result.get("agent_reported_success") else "agent_reported_failure"
    failure = result.get("model_failure") or {}
    # Workers FIRST, before any return below can bypass them. effGen's report to the parent carries no error category, and a worker that did not complete may have failed for a
    # reason the report does not state: `_from_response` sets termination (error, stuck, tool_failed), but `_timed_out` and `_failed` (a timeout, an exception) leave it None.
    # Only a termination that establishes a genuine agent failure (stuck, tool_failed) is taken as one; every other non-completed worker is unexplained.
    unexplained = [w.get("worker") for w in result.get("workers") or []
                   if (w.get("status") != "completed" and w.get("termination") not in GENUINE_WORKER_FAILURES) or w.get("termination") == "error"]
    if unexplained:
        ambiguous.append(f"worker run(s) {unexplained} did not complete and the cause is not established (status/termination in `workers`; possibly a model-call failure)")
    if state == "agent_reported_failure" and result.get("termination") == "error":
        # termination "error" = "the run could not be carried out" (provider failure, empty task, blocked run, written-out call, schema failure): several producers.
        category = failure.get("category")
        if category in TRANSPORT_CATEGORIES:
            reasons.append(f"the model call failed ({category}: {failure.get('type')}): transport, server or configuration, not a coding result")
            return "infrastructure_failure"
        if category in MODEL_LIMIT_CATEGORIES:
            # a genuine model-side failure (output budget or reasoning exhausted, refusal), unless a worker failed for an unstated reason: then the orchestrator may only be
            # reacting to that (a dead endpoint makes the next generation fail too). The category stays in `model_failure` either way.
            return "indeterminate" if unexplained else "agent_reported_failure"
        ambiguous.append(f"the run ended with termination=error and the cause is not established (category={category!r}, reason={failure.get('reason')!r}, stop_reason={failure.get('stop_reason')!r})")
        return "indeterminate"
    return state


def classify_run(record: dict) -> dict:
    """Keeps apart: what the agent did, what was collected, what the evaluator said, and whether the infrastructure failed.

    Precedence (the first that applies decides `attempt_outcome`):
      1. infrastructure failure: the setup was refused, the agent left no result, the harness raised, the model call failed in transport/server/configuration
         (effGen category unreachable, timeout, transient, rate_limited, auth, not_found, not_loaded, resource_exhausted, fatal), or the evaluator failed
         (error, missing pytest completion marker, container OOM kill, the OUTER container timeout). Never a coding failure.
      2. agent timeout: reported on its own (the partial work is still evaluated and recorded).
      3. indeterminate: the cause is not established and the change did not resolve the task: a `termination=error` without a transport or model-limit category, or a
         worker that did not complete for an unstated reason (status other than completed, termination None or error; stuck and tool_failed ARE stated). The worker
         check runs before the orchestrator's own category is read, so a worker failure followed by a truncated or reasoning-only orchestrator is indeterminate, not a coding
         failure. Not counted as a coding failure; the raw fields (`model_failure`, `workers`) stay in the record. A reasoning-only, truncated or refused generation with no
         unexplained worker is NOT indeterminate: it is a genuine model-side failure (`agent_reported_failure`) with its category recorded.
      4. resolved / unresolved by the evaluator. `unresolved` (the only coding failure) covers a genuine agent failure, an empty or unapplicable candidate and failing tests.
    The agent's own success flag is recorded and never decides `resolved`.
    """
    agent, evaluation = record.get("agent"), record.get("evaluation")
    reasons: list[str] = []
    ambiguous: list[str] = []
    agent_state = _agent_state(agent, record, reasons, ambiguous)
    outcome = (evaluation or {}).get("outcome")
    if record.get("candidate") is None:
        candidate = "not_collected"
    elif outcome == "unresolved_patch_failed":
        candidate = "does_not_apply"
    else:
        candidate = "collected" if record["candidate"].get("included_files") else "empty"
    if evaluation is None:
        evaluator = "not_run"
        if agent:
            reasons.append(f"no evaluation: {record.get('error') or record.get('status')}")
    elif outcome == "error":
        evaluator = "infrastructure_failure"
        reasons.append("the evaluator failed: " + "; ".join(evaluation.get("problems") or ["unknown"]))
    else:
        evaluator = outcome
    infrastructure = agent_state in ("not_run", "infrastructure_failure") or evaluator in ("not_run", "infrastructure_failure")
    if infrastructure:
        attempt = "infrastructure_failure"
    elif agent_state == "timed_out":
        attempt = "timed_out"
    elif ambiguous and not record.get("resolved"):
        attempt = "indeterminate"
    else:
        attempt = "resolved" if record.get("resolved") else "unresolved"
    return {"agent": agent_state, "candidate": candidate, "evaluator": evaluator, "infrastructure_failure": infrastructure, "infrastructure_reasons": reasons,
            "indeterminate_reasons": ambiguous, "attempt_outcome": attempt, "counts_as_coding_failure": attempt == "unresolved"}


@dataclass
class RunConfig:
    out_dir: Path
    dataset_dir: Path = REPO / "dataset"
    task_id: str = TASK_ID
    snapshot_sha256: str | None = None
    provider: str = "ollama"
    model: str = ""
    context_length: int = 8192
    base_url: str = "http://127.0.0.1:11434/v1"
    max_tokens: int = 768
    temperature: float = 0.1
    seed: int = 0
    max_iterations: int = 8
    worker_timeout_s: float = 300
    agent_timeout_s: float = 1800
    keep_workspaces: bool = False
    limits: ContainerLimits = ContainerLimits()
    sandbox_limits: SandboxLimits = SandboxLimits()


def check_out_dir(out: Path, dataset: Path) -> Path:
    out = out.resolve()
    for forbidden in (REPO.resolve(), dataset.resolve()):
        if out == forbidden or forbidden in out.parents or out in forbidden.parents:
            raise PilotError(f"artifact directory {out} must be outside {forbidden}")
    return out


def run_pilot(cfg: RunConfig, runner, *, agent_cmd: list[str] | None = None) -> dict:
    out = check_out_dir(cfg.out_dir, cfg.dataset_dir)
    spec = load_task(cfg.dataset_dir / "tasks.jsonl", cfg.dataset_dir / "snapshots" / f"{cfg.task_id}.tgz", cfg.task_id, cfg.snapshot_sha256)
    gold = load_gold(cfg.dataset_dir / "tasks.jsonl", cfg.task_id)  # evaluator-side only: used to scan the agent workspace, never passed on
    run_id = f"{cfg.task_id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(2)}"
    run_dir = out / run_id
    run_dir.mkdir(parents=True, mode=0o700)  # fresh: a reused directory is an error
    agent_dir, eval_dir = run_dir / "agent", run_dir / "evaluator"
    created = [agent_dir / "workspace", eval_dir / "pristine"]
    record: dict = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "task_id": spec.task_id, "started": now(), "snapshot_sha256": spec.snapshot_sha256,
                    "declared_base_commit": spec.base_commit, "deviations": list(DEVIATIONS), "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items() if k not in ("limits", "sandbox_limits")},
                    "container_limits": vars(cfg.limits), "basis": BASIS.get(cfg.task_id), "provenance": collect_provenance(), "model_identity": None, "sandbox": None,
                    "agent": None, "candidate": None, "evaluation": None, "resolved": False, "status": "started"}
    sandboxed = agent_cmd is None  # an injected command is for tests only; a real run is ALWAYS sandboxed, or refused
    try:
        if sandboxed:
            problem = sandbox_problem(cfg.sandbox_limits)
            if problem:
                raise PilotError(f"the agent sandbox is unavailable ({problem}); refusing to run the agent as the plain host user")
            endpoint = model_endpoint(cfg.base_url)
            record["sandbox"] = {"kind": "systemd scope + bubblewrap + prlimit", "network": f"none; a unix-socket bridge to {endpoint[0]}:{endpoint[1]} only", "limits": vars(cfg.sandbox_limits),
                                 "bwrap": subprocess.run(["bwrap", "--version"], capture_output=True, text=True).stdout.strip()}
        else:
            record["sandbox"] = {"kind": "none (an injected agent command: tests only)"}
        agent_dir.mkdir(mode=0o700)
        eval_dir.mkdir(mode=0o700)
        workspace, pristine = agent_dir / "workspace", eval_dir / "pristine"
        extract_snapshot(spec.snapshot, workspace, spec.snapshot_sha256, include_git=False)
        extract_snapshot(spec.snapshot, pristine, spec.snapshot_sha256, include_git=False)
        leak_scan(workspace, [spec.test_patch, gold])
        record["workspace_files"] = len(manifest(workspace))
        (agent_dir / "prompt.txt").write_text(build_prompt(spec))
        a = {"provider": cfg.provider, "model": cfg.model, "context_length": cfg.context_length, "base_url": f"http://127.0.0.1:{endpoint[1]}/v1" if sandboxed else cfg.base_url, "worker_timeout_s": cfg.worker_timeout_s, "max_iterations": cfg.max_iterations}
        (agent_dir / "harness.yaml").write_text(render_config(workspace, a))
        record["hashes"] = {"prompt_sha256": sha256_file(agent_dir / "prompt.txt"), "harness_yaml_sha256": sha256_file(agent_dir / "harness.yaml"),
                            "worker_prompt_sha256": sha256_bytes(WORKER_PROMPT.encode()), "task_statement_sha256": sha256_bytes(spec.statement.encode())}
        record["model_identity"] = fetch_model_identity(cfg.base_url, cfg.model)  # asked of the server (metadata only) BEFORE the agent starts
        if not record["model_identity"]["available"]:  # an unreachable endpoint or an unknown tag is an infrastructure failure, not a coding result
            raise PilotError(f"the model identity could not be established, so the agent was not started: {record['model_identity']['reason']}")
        params = {"config_path": str(agent_dir / "harness.yaml"), "prompt_path": str(agent_dir / "prompt.txt"), "out_dir": str(agent_dir),
                  "max_tokens": cfg.max_tokens, "temperature": cfg.temperature, "seed": cfg.seed, "workspace": str(workspace)}
        if sandboxed:
            bridge_dir = Path(tempfile.mkdtemp(prefix="cb-"))  # a short path: unix socket paths are limited to about 100 characters
            try:
                with ModelBridge(bridge_dir, *endpoint):
                    params["bridge"] = {"port": endpoint[1], "socket": "/bridge/m.sock"}
                    inner = [sys.executable, str(Path(__file__).resolve()), "agent-child", "--params", str(agent_dir / "params.json")]
                    record["agent"] = run_agent_phase(agent_dir, params, cfg.agent_timeout_s, sandbox_command(agent_dir, inner, cfg.sandbox_limits, extra_binds=((str(bridge_dir), "/bridge"),)))
            finally:
                shutil.rmtree(bridge_dir, ignore_errors=True)
        else:
            record["agent"] = run_agent_phase(agent_dir, params, cfg.agent_timeout_s, agent_cmd)
        result_path = agent_dir / "agent_result.json"
        record["agent"]["result"] = json.loads(result_path.read_text()) if result_path.exists() else None
        patch, meta = collect_patch(pristine, workspace, frozenset(spec.test_files))  # fails closed on links, binary files, too many files
        (run_dir / "candidate.patch").write_text(patch)
        record["candidate"] = meta
        record["evaluation"] = evaluate(spec, patch, eval_dir / "eval", runner, limits=cfg.limits)
        write_json(run_dir / "evaluation.json", record["evaluation"])
        record["resolved"] = bool(record["evaluation"]["resolved"])  # the evaluator's verdict; the agent's own success flag is recorded but never used
        record["status"] = "complete"
    except PilotError as exc:
        record.update(status="refused", error=str(exc))
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["finished"] = now()
        try:
            record["classification"] = classify_run(record)
        except Exception as exc:  # never hide the real outcome behind a classifier fault
            record["classification"] = {"error": f"{type(exc).__name__}: {exc}"}
        write_json(run_dir / "run.json", record)
        if not cfg.keep_workspaces:
            for p in created:  # only what this run created, only inside its own run directory
                if p.is_relative_to(run_dir):
                    shutil.rmtree(p, ignore_errors=True)
    return record


# ---- command line ---------------------------------------------------------------------------------------------------------------
def prepare_env(python_dir: Path, wheels_dir: Path, site_dir: Path) -> dict:
    """Offline install of the task's test dependencies from the dataset's wheels (wheels are unpacked, never built or executed)."""
    site_dir.parent.mkdir(parents=True, exist_ok=True)
    cache = site_dir.parent / "uvcache"
    cmd = ["uv", "--no-config", "pip", "install", "--offline", "--no-index", "--find-links", str(wheels_dir), "--only-binary", ":all:", "--no-compile",
           "--python", str(Path(python_dir).resolve() / "bin" / "python3"), "--target", str(site_dir), *EVAL_PACKAGES]
    done = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "UV_CACHE_DIR": str(cache)}, timeout=300)
    if done.returncode != 0:
        raise PilotError(f"offline dependency install failed: {done.stderr.strip()[-500:]}")
    return {"command": cmd, "packages": sorted(p.name for p in site_dir.glob("*.dist-info"))}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rich3278_pilot", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset-dir", type=Path, default=REPO / "dataset")
    common.add_argument("--task-id", default=TASK_ID)
    common.add_argument("--python-dir", type=Path, default=Path("~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu").expanduser(), help="a standalone Python 3.12 copied into the container")
    common.add_argument("--site-dir", type=Path, required=False, help="site-packages prepared by prepare-env (default: <out>/env/site)")
    common.add_argument("--image", default="redis:7", help="a LOCAL Debian-based image providing bash and timeout; never pulled")
    r = sub.add_parser("run", parents=[common], help="agent phase + evaluation (calls the configured model: needs --allow-model-call)")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--provider", default="ollama", choices=["ollama"])
    r.add_argument("--model", required=True)
    r.add_argument("--context-length", type=int, default=8192)
    r.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    r.add_argument("--max-tokens", type=int, default=768)
    r.add_argument("--temperature", type=float, default=0.1)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--max-iterations", type=int, default=8)
    r.add_argument("--agent-timeout-s", type=float, default=1800)
    r.add_argument("--keep-workspaces", action="store_true")
    r.add_argument("--allow-model-call", action="store_true", help="required: confirms a real model may be called")
    e = sub.add_parser("evaluate", parents=[common], help="evaluate an existing candidate patch (no agent, no model)")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--patch", type=Path, required=True)
    c = sub.add_parser("reference-check", parents=[common], help="baseline (test patch only) and gold patch through the evaluator (no agent, no model)")
    c.add_argument("--out", type=Path, required=True)
    pe = sub.add_parser("prepare-env", parents=[common], help="install the test dependencies offline from the dataset wheels")
    pe.add_argument("--out", type=Path, required=True)
    ch = sub.add_parser("agent-child", help=argparse.SUPPRESS)
    ch.add_argument("--params", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "agent-child":
        return run_agent(json.loads(args.params.read_text()))
    try:
        out = check_out_dir(args.out, args.dataset_dir)
        out.mkdir(parents=True, exist_ok=True)
        site = args.site_dir or out / "env" / "site"
        if args.command == "prepare-env":
            print(json.dumps(prepare_env(args.python_dir, args.dataset_dir / "wheels", site), indent=1))
            return 0
        runner = DockerRunner(args.python_dir, site, secrets.token_hex(3), image=args.image)
        tasks, snap = args.dataset_dir / "tasks.jsonl", args.dataset_dir / "snapshots" / f"{args.task_id}.tgz"
        if args.command == "run":
            if not args.allow_model_call:
                raise PilotError("refusing to call a model without --allow-model-call")
            rec = run_pilot(RunConfig(out, args.dataset_dir, args.task_id, None, args.provider, args.model, args.context_length, args.base_url, args.max_tokens, args.temperature,
                                      args.seed, args.max_iterations, 300, args.agent_timeout_s, args.keep_workspaces), runner)
            print(json.dumps({k: rec.get(k) for k in ("run_id", "status", "resolved")} | {"outcome": (rec.get("evaluation") or {}).get("outcome")}
                             | {k: (rec.get("classification") or {}).get(k) for k in ("attempt_outcome", "infrastructure_failure", "counts_as_coding_failure")}))
            return 0 if rec["status"] == "complete" else 1
        spec = load_task(tasks, snap, args.task_id)
        if args.command == "evaluate":
            records = [evaluate(spec, args.patch.read_text(), out / f"eval-{secrets.token_hex(2)}", runner)]
        else:
            records = [evaluate(spec, "", out / f"baseline-{secrets.token_hex(2)}", runner, label="baseline", run_even_if_empty=True),
                       evaluate(spec, load_gold(tasks, args.task_id), out / f"gold-{secrets.token_hex(2)}", runner, label="gold")]
        for rec in records:
            write_json(out / f"{rec['label']}-{secrets.token_hex(2)}.evaluation.json", rec)
            print(json.dumps({k: rec[k] for k in ("label", "outcome", "resolved")} | {"counts": {k: (rec["result"] or {}).get(k) for k in ("passed", "failed", "errors", "pytest_exit")}}))
        return 0
    except PilotError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
