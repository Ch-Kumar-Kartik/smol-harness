"""T5.2a: a scripted coding workflow through the whole application, on a disposable Git repository.

SCRIPTED end-to-end verification only: the models are scripted (only `build_model` is replaced), everything else is
real: the CLI, `initialize_runtime()`, `load_config()`, `build_harness_from_config()`, `Harness.run()/close()`, the
delegation layer, effGen's real agent loop, the real `file_operations` and the real sandboxed `shell` (git and pytest run
inside it), and the filesystem. This says nothing about what a real model can do (that is T5.2b).

The task: `calc.add` subtracts instead of adding; `test_calc.py` exposes it. The scripted worker reads the source, checks
the tree, runs the tests (fails), fixes the source, runs the SAME test command again (passes), and stages the fix.
Success is never judged from the models' words: the repository is inspected and its tests are run again by this file.

It exercises the earlier review fixes: F1 (the identical test command must run again after the edit), F2 (silent commands
and rewrites must not end the run) and F3 (a stopped orchestrator keeps its worker records). Trusted, disposable repos only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from effgen import AgentConfig
from effgen.models.errors import BackendUnreachableError
from test_cli import REPO, CliModels, run_driver
from test_harness_config import BIG, FINAL, SMALL, TrackedModel, call, delegate

import smol_harness.__main__ as cli
import smol_harness.models as models_mod
from smol_harness.harness import Harness, build_harness
from smol_harness.tools import ShellTool, build_tools

pytestmark = [
    pytest.mark.skipif(platform.system() != "Linux", reason="the shell sandbox (Linux namespaces) is required; no unsandboxed fallback exists"),
    pytest.mark.skipif(shutil.which("git") is None, reason="git is required to build the disposable repository"),
]

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
TEST_FILE = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
TEST_CMD = f"{shlex.quote(sys.executable)} -m pytest -q -p no:cacheprovider test_calc.py"  # the one command the worker repeats
WORKER_TASK = "Fix the failing test in calc.py: add() returns the wrong result. Run test_calc.py to confirm."
ORCH_ANSWER = "coder fixed calc.py and test_calc.py passes"


@pytest.fixture(scope="module", autouse=True)
def sandbox_or_skip(tmp_path_factory):
    """Skip, with the reason, when this host cannot run the sandbox; never fall back to an unrestricted shell."""
    probe = tmp_path_factory.mktemp("sandbox-probe")
    try:
        result = asyncio.run(ShellTool(probe).execute(command="true"))
    except Exception as exc:  # ToolBuildError / ShellSandboxError: the tool fails closed by itself
        pytest.skip(f"the shell sandbox is unavailable on this host ({type(exc).__name__}: {exc})")
    if not result.success:
        pytest.skip(f"the shell sandbox is unavailable on this host: {result.error}")


# ---- the disposable repository and its independent verification -----------------------------------------------------------------
def git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "HOME": str(repo.parent / "home")}
    done = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args], cwd=repo, env=env,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def project(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text(BUGGY)
    (repo / "test_calc.py").write_text(TEST_FILE)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "initial: add() is wrong")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "canary.txt").write_text("must never change\n")
    return SimpleNamespace(root=tmp_path, repo=repo, outside=outside, head=git(repo, "rev-parse", "HEAD").strip(),
                           test_sha=sha(repo / "test_calc.py"), canary=tree_digest(outside))


def host_pytest(repo: Path) -> subprocess.CompletedProcess:
    """Run the repository's tests ourselves, outside the agent and outside the sandbox."""
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_calc.py"], cwd=repo, capture_output=True,
                          text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def processes_left_in(path: Path) -> list[str]:
    """Any process whose working directory is inside `path` (a surviving shell or test run)."""
    root, found = str(path.resolve()), []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                cwd = os.readlink(entry / "cwd")
            except OSError:
                continue
            if cwd == root or cwd.startswith(root + os.sep):
                found.append(f"{entry.name}:{cwd}")
    return found


def assert_repo_fixed(project) -> None:
    repo = project.repo
    assert (repo / "calc.py").read_text() == FIXED  # the file really changed...
    assert sha(repo / "test_calc.py") == project.test_sha  # ...and only that file: the test was not touched
    assert git(repo, "rev-parse", "HEAD").strip() == project.head  # nothing was committed
    assert git(repo, "status", "--porcelain").splitlines() == ["M  calc.py"]  # exactly one staged change, nothing else
    assert sorted(p.name for p in repo.iterdir()) == [".git", "calc.py", "test_calc.py"]  # and no stray files or directories (git ignores empty ones)
    assert git(repo, "diff", "--cached", "--numstat").split() == ["1", "1", "calc.py"]  # one line replaced
    host = host_pytest(repo)
    assert host.returncode == 0 and "1 passed" in host.stdout, host.stdout  # and the tests genuinely pass, run by us


def assert_repo_untouched(project) -> None:
    assert (project.repo / "calc.py").read_text() == BUGGY and git(project.repo, "status", "--porcelain") == ""
    assert sorted(p.name for p in project.repo.iterdir()) == [".git", "calc.py", "test_calc.py"]
    assert host_pytest(project.repo).returncode != 0  # the bug is still there, as the failure report said


def assert_nothing_outside_changed(project) -> None:
    assert tree_digest(project.outside) == project.canary
    assert sorted(p.name for p in project.root.iterdir()) == sorted(["repo", "outside", "harness.yaml", "home"] if (project.root / "home").exists() else ["repo", "outside", "harness.yaml"])


# ---- configuration and the scripted models ---------------------------------------------------------------------------------------------
def write_config(project, *, orchestrator_tools=(), **limits) -> Path:
    data = {
        "workspace": str(project.repo),
        "orchestrator": {"model": {"provider": "ollama", "model": BIG, "context_length": 8192}, "tools": list(orchestrator_tools),
                         "max_iterations": 6},
        "workers": [{"name": "coder", "description": "reads, edits and tests code in the repository",
                     "model": {"provider": "ollama", "model": SMALL, "context_length": 4096},
                     "tools": ["file_operations", "shell"], "max_iterations": 10}],
    }
    if limits:
        data["limits"] = limits
    path = project.root / "harness.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def worker_script(*, fix: bool = True) -> list[str]:
    script = [
        call("file_operations", operation="read", path="calc.py"),  # 1 read the source
        call("shell", command="git status --short"),  # 2 silent: the tree is clean ("exit code: 0", no output)
        call("shell", command=TEST_CMD),  # 3 the tests fail
    ]
    if fix:
        script += [
            call("file_operations", operation="write", path="calc.py", content=FIXED),  # 4 the fix
            call("shell", command=TEST_CMD),  # 5 the IDENTICAL command: must run again and now pass (F1)
            call("shell", command="git add calc.py"),  # 6 silent again, like step 2 (F2)
        ]
    return script + ["Thought: done.\nFinal Answer: Fixed add() in calc.py (a - b became a + b); test_calc.py passes."]


OK_SCRIPTS = {BIG: [delegate("coder", WORKER_TASK), f"Thought: done.\nFinal Answer: {ORCH_ANSWER}"], SMALL: worker_script()}


@pytest.fixture
def models(monkeypatch):
    fake = CliModels()
    monkeypatch.setattr(models_mod, "build_model", fake.build)
    return fake


def run_cli(project, capsys, *flags, task="Fix the failing test in this repository", **config):
    argv = ["run", "--config", str(write_config(project, **config)), "--task", task, *flags]
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err, argv


# ===============================================================================================================================
# The workflow, through the CLI
# ===============================================================================================================================
def test_scripted_edit_test_fix_retest_through_the_cli(project, models, capsys):
    models.scripts.update(OK_SCRIPTS)
    code, out, err, _ = run_cli(project, capsys, "--json")

    doc = json.loads(out)
    assert code == 0 and doc["success"] is True and doc["output"] == ORCH_ANSWER and doc["termination"] == "done"
    (worker,) = doc["workers"]
    assert (worker["worker"], worker["delegation_id"], worker["status"]) == ("coder", "d1", "completed")
    assert worker["tool_calls"] == 6 and worker["error"] is None  # effGen counts executed dispatches: all six really ran
    assert "d1 coder: completed" in err

    # What the worker actually saw, in order: the failing run, then (after the edit) the same command passing.
    seen = str(models.instances[SMALL].calls[-1].prompt)
    assert seen.count(f"$ {TEST_CMD}") == 2  # the identical command, twice
    first_fail, last_pass = seen.find("1 failed"), seen.rfind("1 passed")
    assert 0 < first_fail < seen.find("wrote calc.py") < last_pass  # fail -> edit -> pass: the second run was fresh, not replayed
    assert seen.count("1 passed") == 1 and seen.count("1 failed") == 1  # (one of each: not a stale copy of the first result)

    assert_repo_fixed(project)  # independent of everything the models said
    assert_nothing_outside_changed(project)
    assert processes_left_in(project.repo) == []
    assert {name: m.unloads for name, m in models.instances.items()} == {BIG: 1, SMALL: 1}  # both models released, once


def test_the_human_output_is_the_orchestrators_answer_and_a_short_summary(project, models, capsys):
    models.scripts.update(OK_SCRIPTS)
    code, out, err, _ = run_cli(project, capsys)
    assert code == 0 and out == ORCH_ANSWER + "\n"
    assert "delegations: 1" in err and "d1 coder: completed" in err and "tool calls 6" in err
    assert "def add" not in out + err  # no file content on either stream
    assert_repo_fixed(project)


def test_the_workers_side_effects_match_its_claims_and_the_records_name_the_worker(project, models, capsys):
    models.scripts.update(OK_SCRIPTS)
    _, out, _, _ = run_cli(project, capsys, "--json")
    worker = json.loads(out)["workers"][0]
    assert worker["worker"] == "coder" and "calc.py" in worker["output"] and "test_calc.py passes" in worker["output"]
    assert (project.repo / "calc.py").read_text() == FIXED  # the claim "calc.py is fixed" is true on disk
    assert host_pytest(project.repo).returncode == 0  # and so is "test_calc.py passes"


# ---- F1 / F2 in the real workflow ------------------------------------------------------------------------------------------------------------
def test_without_the_loop_patch_the_cli_refuses_rather_than_replaying_a_stale_result(project, models, capsys, monkeypatch, compat_uninstalled):
    # The CLI's own initialisation is what makes the identical rerun real. If it were skipped, the run must not start.
    from smol_harness.harness import EffgenCompatError

    def refuse():
        raise EffgenCompatError("patch missing")

    monkeypatch.setattr(cli, "initialize_runtime", refuse)
    models.scripts.update(OK_SCRIPTS)
    code, out, err, _ = run_cli(project, capsys)
    assert code == 3 and out == "" and models.loads == []
    assert_repo_untouched(project)


def test_a_rewrite_with_identical_text_is_not_mistaken_for_a_loop(project, models, capsys):
    # F2: writing calc.py twice (same path) and several silent commands in a row all return near-identical effGen text.
    script = [call("shell", command="git status --short"), call("shell", command="git add -N calc.py"),
              call("file_operations", operation="write", path="calc.py", content=FIXED),
              call("file_operations", operation="write", path="calc.py", content=FIXED + "\n"),  # a second write to the same path
              call("shell", command=TEST_CMD), call("shell", command="git add calc.py"),
              "Thought: done.\nFinal Answer: fixed"]
    models.scripts.update({BIG: OK_SCRIPTS[BIG], SMALL: script})
    code, out, _, _ = run_cli(project, capsys, "--json")
    worker = json.loads(out)["workers"][0]
    assert code == 0 and worker["status"] == "completed" and worker["tool_calls"] == 6  # none of the six steps was cut off
    assert (project.repo / "calc.py").read_text() == FIXED + "\n" and host_pytest(project.repo).returncode == 0


# ---- failures --------------------------------------------------------------------------------------------------------------------------------
def test_a_stopped_orchestrator_exits_1_keeps_the_worker_record_and_the_fix(project, models, capsys):
    # The orchestrator delegates, then loops on identical reads until effGen stops it (F3: the record survives).
    models.scripts.update({BIG: [delegate("coder", WORKER_TASK)] + [call("file_operations", operation="read", path="test_calc.py")] * 8,
                           SMALL: worker_script()})
    code, out, err, _ = run_cli(project, capsys, "--json", orchestrator_tools=["file_operations"])
    doc = json.loads(out)
    assert code == 1 and doc["success"] is False and doc["termination"] == "stuck" and doc["stop_reason"]
    assert [(w["worker"], w["status"], w["tool_calls"]) for w in doc["workers"]] == [("coder", "completed", 6)]
    assert "the task failed" in err and "d1 coder: completed" in err
    assert_repo_fixed(project)  # the harness-level failure does not undo the work the worker really did
    assert processes_left_in(project.repo) == [] and {m.unloads for m in models.instances.values()} == {1}


def test_a_stopped_orchestrator_with_effgens_default_raise_on_error_still_preserves_the_records(project):
    # F3 proper. The CLI's builder sets raise_on_error=False, so exercise effGen's default (True) with the same real tools:
    # Agent.run raises RunStoppedError underneath, and Harness.run must still return the result with the worker's record.
    from test_cli import OK_FLOW  # noqa: F401  (documentation: same shape of scripted flow)
    from smol_harness.harness import initialize_runtime

    initialize_runtime()
    coder_model = TrackedModel(worker_script(), name=SMALL)
    orch_model = TrackedModel([delegate("coder", WORKER_TASK)] + [call("file_operations", operation="read", path="test_calc.py")] * 8, name=BIG)

    def agent(model, name, tool_names):
        return AgentConfig(name=name, model=model, tools=build_tools(tool_names, project.repo), tool_calling_mode="react",
                           capability_probe=False, enable_memory=False, max_iterations=10, raise_on_error=True)  # effGen's default

    harness = build_harness(agent(orch_model, "orchestrator", ["file_operations"]), {"coder": agent(coder_model, "coder", ["file_operations", "shell"])})
    assert isinstance(harness, Harness)
    result = harness.run("Fix the failing test in this repository")
    assert result.success is False and result.termination == "stuck"
    assert [(r.worker, r.status, r.result.tool_calls) for r in result.workers] == [("coder", "completed", 6)]
    assert_repo_fixed(project)
    assert processes_left_in(project.repo) == []


def test_a_worker_that_fails_mid_task_leaves_the_repo_untouched_and_the_cli_exits_non_zero(project, models, capsys):
    attempted: list[str] = []

    class DiesAfterTheFirstTestRun(TrackedModel):
        def generate(self, prompt, config=None, **kw):
            attempted.append(str(prompt))  # every request, including the one that fails
            if len(self.calls) >= 3:
                raise BackendUnreachableError("openai_compatible", SMALL, "connection refused", "http://127.0.0.1:11434/v1")
            return super().generate(prompt, config, **kw)

    models.custom[SMALL] = lambda: DiesAfterTheFirstTestRun(worker_script(), name=SMALL)
    models.scripts[BIG] = [delegate("coder", WORKER_TASK)] * 8  # the orchestrator keeps retrying and never answers
    code, out, err, _ = run_cli(project, capsys, "--json", max_delegations=3)
    doc = json.loads(out)
    assert code == 1 and doc["success"] is False
    statuses = [w["status"] for w in doc["workers"]]
    assert statuses[0] == "failed" and set(statuses) <= {"failed"} and len(statuses) >= 2
    assert doc["workers"][0]["error_type"] == "BackendUnreachableError"  # (an exception carries no counters: tool_calls is 0 by design)
    died_on = attempted[3]  # the 4th request: it carries the observation of the (real) failing test run
    assert f"$ {TEST_CMD}" in died_on and "1 failed" in died_on and "1 passed" not in died_on  # ran, saw it fail, never got to fix it
    assert_repo_untouched(project)  # the failure report is true: nothing was fixed
    assert processes_left_in(project.repo) == [] and {m.unloads for m in models.instances.values()} == {1}


def test_documented_limit_the_exit_status_follows_the_orchestrator_not_the_repository(project, models, capsys):
    # The harness does not run the repository's tests itself. If the worker fails and the orchestrator nevertheless
    # answers, the run "succeeds" (exit 0) while the bug remains. A caller who needs more must verify the repository.
    class WorkerDown(TrackedModel):
        def generate(self, prompt, config=None, **kw):
            raise BackendUnreachableError("openai_compatible", SMALL, "connection refused", "http://127.0.0.1:11434/v1")

    models.custom[SMALL] = lambda: WorkerDown(["unused"], name=SMALL)
    models.scripts[BIG] = [delegate("coder", WORKER_TASK), "Thought: done.\nFinal Answer: the worker failed; the bug is NOT fixed"]
    code, out, err, _ = run_cli(project, capsys)
    assert code == 0 and "NOT fixed" in out and "d1 coder: failed" in err
    assert_repo_untouched(project)  # which is why this test inspects the repo instead of trusting the exit code


# ---- a fresh process, nothing from pytest's session -----------------------------------------------------------------------------------------------
def test_a_fresh_process_cli_run_fixes_the_repository(project):
    scripts = {BIG: OK_SCRIPTS[BIG], SMALL: OK_SCRIPTS[SMALL]}
    config = write_config(project)
    argv = ["run", "--config", str(config), "--task", "Fix the failing test in this repository"]
    before = git(REPO, "status", "--porcelain")
    done, state = run_driver(project.root, scripts, argv)
    assert state["installed_before"] is False  # no pytest fixture installed the loop patch for this process
    assert done.returncode == 0 and done.stdout == ORCH_ANSWER + "\n", done.stderr[-2000:]
    assert state["installed_after"] is True and state["unloaded"] == [BIG, SMALL]
    assert_repo_fixed(project)
    assert_nothing_outside_changed(project)
    assert processes_left_in(project.repo) == []
    assert git(REPO, "status", "--porcelain") == before  # and the smol-harness project itself was not touched


def test_a_fresh_process_failed_run_exits_1_and_leaves_no_processes(project):
    scripts = {BIG: [delegate("coder", WORKER_TASK)] + [call("file_operations", operation="read", path="test_calc.py")] * 8, SMALL: worker_script()}
    config = write_config(project, orchestrator_tools=["file_operations"])
    done, state = run_driver(project.root, scripts, ["run", "--config", str(config), "--task", "Fix the failing test", "--json"])
    assert done.returncode == 1 and json.loads(done.stdout)["success"] is False and state["unloaded"] == [BIG, SMALL]
    assert [w["worker"] for w in json.loads(done.stdout)["workers"]] == ["coder"]
    assert_repo_fixed(project)
    assert processes_left_in(project.repo) == []
