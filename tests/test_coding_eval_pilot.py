"""The one-task coding-evaluation pilot driver (experiments/coding_eval/rich3278_pilot.py): driver MECHANICS only.

CPU only, no network, no model, no Docker. The evaluator container is replaced by `FakeRunner` (it records what it was given and
returns canned pytest output); the agent phase is either a fake child process that edits the workspace or the real harness with a
scripted model (`run_agent`). Nothing here says whether a real agent can solve the task, or that the real container works: the
opt-in test at the bottom (SMOL_RUN_DOCKER_TESTS=1, needs the dataset and a local Docker image) covers the reference check.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_harness_config import BIG, FINAL, FakeModels, call, delegate

import smol_harness.models as models_mod

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "coding_eval"))
import rich3278_pilot as pilot  # noqa: E402

REAL_FETCH_MODEL_IDENTITY = pilot.fetch_model_identity  # the autouse fixture below replaces pilot.fetch_model_identity

needs_patch = pytest.mark.skipif(shutil.which("patch") is None, reason="GNU patch is required")
linux_only = pytest.mark.skipif(platform.system() != "Linux", reason="process inspection uses /proc")

CORE_OLD = "def add(a, b):\n    return a - b\n"
CORE_NEW = "def add(a, b):\n    return a + b\n"
TEST_OLD = "from mini.core import add\n\n\ndef test_add():\n    assert add(1, 1) == 2\n"
TEST_NEW = TEST_OLD + "\n\ndef test_add_three():\n    assert add(2, 3) == 5\n"
STATEMENT = "MINI-STATEMENT: add() subtracts instead of adding. Make add(2, 3) return 5."
DECOY = "DECOY-STATEMENT: a different task that must never reach this run"
GOLD = pilot._diff("mini/core.py", CORE_OLD, CORE_NEW)
TEST_PATCH = pilot._diff("tests/test_core.py", TEST_OLD, TEST_NEW)


def pytest_output(passed: int, failed: int = 0, required: tuple[str, ...] = ("test_add_three",), exit_code: int | None = None) -> str:
    lines = ["python: Python 3.12.13", "pytest: pytest 8.3.4"]
    lines += [f"PASSED tests/test_core.py::{n}" for n in ("test_add", *required)][:passed]
    lines += [f"FAILED tests/test_core.py::{n} - assert 1 == 2" for n in required][:failed]
    lines.append(f"{failed} failed, {passed} passed in 0.01s" if failed else f"{passed} passed in 0.01s")
    code = exit_code if exit_code is not None else (1 if failed else 0)
    return "\n".join(lines) + f"\npytest_exit={code} wall_ms=12\n"


class FakeRunner:
    """Stands in for the container: remembers the workspace content it was handed and returns canned output."""

    def __init__(self, stdout: str | None = None, exit_code: int = 0, timed_out: bool = False, raises: Exception | None = None) -> None:
        self.stdout, self.exit_code, self.timed_out, self.raises, self.calls = stdout, exit_code, timed_out, raises, []

    def run(self, workspace: Path, script: str, label: str, log_dir: Path) -> dict:
        if self.raises:
            raise self.raises
        tree = {p.relative_to(workspace).as_posix(): p.read_text() for p in sorted(workspace.rglob("*")) if p.is_file()}
        self.calls.append({"label": label, "tree": tree, "script": script})
        return {"exit_code": self.exit_code, "stdout": self.stdout if self.stdout is not None else pytest_output(2), "stderr": "", "timed_out": self.timed_out,
                "wall_s": 0.01, "image": "fake", "isolation": {"NetworkMode": "none"}}


@pytest.fixture(autouse=True)
def model_identity(monkeypatch):
    """The driver asks the Ollama server for the model's identity before starting the agent; no server exists here."""
    identity = {"available": True, "provider": "ollama", "digest": "sha256:" + "ab" * 32, "family": "qwen3", "parameter_size": "14.8B", "quantization_level": "Q4_K_M"}
    monkeypatch.setattr(pilot, "fetch_model_identity", lambda base_url, model, **kw: {**identity, "tag": model})
    return identity


# ---- a miniature dataset: two tasks, a snapshot with a .git directory, evaluator-only patches --------------------------------------
@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "ds"
    (root / "snapshots").mkdir(parents=True)
    src = tmp_path / "src"
    for rel, text in {"mini/__init__.py": "", "mini/core.py": CORE_OLD, "tests/__init__.py": "", "tests/test_core.py": TEST_OLD, "README.md": "mini\n", ".git/config": "[core]\n"}.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(text)
    archive = root / "snapshots" / "mini_1.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(src, arcname=".")
    rows = [{"instance_id": "mini_1", "repo": "acme/mini", "base_commit": "0" * 40, "problem_statement": STATEMENT, "patch": GOLD.rstrip("\n"), "test_patch": TEST_PATCH.rstrip("\n"), "hints_text": "", "created_at": "2026-01-01T00:00:00Z"},  # like the real dataset: no final newline
            {"instance_id": "mini_2", "repo": "acme/other", "base_commit": "1" * 40, "problem_statement": DECOY, "patch": "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n" * 3, "test_patch": TEST_PATCH, "hints_text": "", "created_at": "2026-01-02T00:00:00Z"}]
    (root / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root, pilot.sha256_file(archive)


@pytest.fixture
def spec(dataset):
    root, sha = dataset
    return pilot.load_task(root / "tasks.jsonl", root / "snapshots" / "mini_1.tgz", "mini_1", sha)


def cfg_for(dataset, tmp_path, **over) -> pilot.RunConfig:
    root, sha = dataset
    base = dict(out_dir=tmp_path / "out", dataset_dir=root, task_id="mini_1", snapshot_sha256=sha, model="big:1", agent_timeout_s=60)
    return pilot.RunConfig(**{**base, **over})


FAKE_AGENT = ("import json, pathlib\np = json.load(open('params.json')); ws = pathlib.Path(p['workspace'])\n"
              "f = ws / 'mini/core.py'; f.write_text(f.read_text().replace('a - b', 'a + b'))\n"
              "(ws / 'tests/test_core.py').write_text('tampered\\n')\n(ws / 'notes.txt').write_text('scratch\\n')\n"
              "(ws / 'mini/__pycache__').mkdir(); (ws / 'mini/__pycache__/x.pyc').write_bytes(b'\\0')\n"
              "json.dump({'agent_reported_success': True, 'output': 'edited'}, open('agent_result.json', 'w'))\n")


def fake_agent(code: str = FAKE_AGENT) -> list[str]:
    return [sys.executable, "-c", code]


# ---- task selection and snapshots ------------------------------------------------------------------------------------------------
def test_the_task_is_selected_by_id_and_the_decoy_never_reaches_the_prompt(dataset, spec):
    assert (spec.task_id, spec.statement, spec.test_files, spec.required_tests) == ("mini_1", STATEMENT, ("tests/test_core.py",), ("test_add_three",))
    prompt = pilot.build_prompt(spec)
    assert STATEMENT in prompt and DECOY not in prompt and "acme/mini" in prompt
    root, sha = dataset
    with pytest.raises(pilot.PilotError, match="exactly one task"):
        pilot.load_task(root / "tasks.jsonl", root / "snapshots" / "mini_1.tgz", "nope_9", sha)


def test_dataset_patches_without_a_final_newline_are_normalised_for_the_evaluator(dataset, spec):
    root, _ = dataset
    assert pilot.load_gold(root / "tasks.jsonl", "mini_1") == GOLD and spec.test_patch == TEST_PATCH
    assert pilot.with_newline("") == "" and pilot.with_newline("x") == "x\n" and pilot.with_newline("x\n") == "x\n"


def test_only_the_worker_is_given_edit_file_and_the_prompts_describe_it(spec, tmp_path):
    import yaml

    cfg = yaml.safe_load(pilot.render_config(tmp_path / "ws", {"provider": "ollama", "model": "m:1", "context_length": 8192, "base_url": "http://127.0.0.1:11434/v1",
                                                               "worker_timeout_s": 300, "max_iterations": 8}))
    assert cfg["orchestrator"]["tools"] == ["file_operations"] and cfg["workers"][0]["tools"] == ["file_operations", "edit_file"]
    assert "edit_file" in cfg["workers"][0]["system_prompt"] and "\\u005c" in cfg["workers"][0]["system_prompt"]
    assert cfg["orchestrator"]["model"] == cfg["workers"][0]["model"]  # model, context length and the other settings are unchanged
    assert any("edit_file" in d for d in pilot.DEVIATIONS)


def test_the_orchestrator_prompt_delegates_the_edit_and_names_only_tools_the_orchestrator_holds(spec, tmp_path):
    import re

    import yaml

    from smol_harness.tools import TOOL_NAMES

    cfg = yaml.safe_load(pilot.render_config(tmp_path / "ws", {"provider": "ollama", "model": "m:1", "context_length": 8192, "base_url": "http://127.0.0.1:11434/v1",
                                                               "worker_timeout_s": 300, "max_iterations": 8}))
    held = {*cfg["orchestrator"]["tools"], "delegate"}  # the harness adds delegate to the orchestrator only
    prompt = pilot.build_prompt(spec)
    named = {n for n in (*TOOL_NAMES, "delegate") if re.search(rf"\b{n}\b", prompt)}
    assert named <= held, f"the prompt names tools the orchestrator does not hold: {named - held}"
    assert "edit_file" not in prompt and "delegate" in prompt and f"`{pilot.WORKER}`" in prompt and STATEMENT in prompt
    assert cfg["orchestrator"]["tools"] == ["file_operations"] and cfg["workers"][0]["tools"] == ["file_operations", "edit_file"]  # the boundary is unchanged


def test_the_default_task_is_pinned_to_the_validated_snapshot_hash():
    assert pilot.TASK_ID == "rich_3278"
    assert pilot.SNAPSHOT_SHA256["rich_3278"] == "bda96e09e9e02fa372b2cf6c5f3f2760f97d292fb531cc631564632940d6a453"


def test_a_snapshot_with_the_wrong_hash_is_refused_before_anything_is_extracted(dataset, tmp_path):
    root, _ = dataset
    dest = tmp_path / "ws"
    with pytest.raises(pilot.PilotError, match="pinned SHA-256"):
        pilot.extract_snapshot(root / "snapshots" / "mini_1.tgz", dest, "0" * 64, include_git=False)
    assert not dest.exists()


def test_extraction_leaves_out_git_and_refuses_a_non_empty_destination(dataset, tmp_path):
    root, sha = dataset
    dest = tmp_path / "ws"
    pilot.extract_snapshot(root / "snapshots" / "mini_1.tgz", dest, sha, include_git=False)
    assert not (dest / ".git").exists() and (dest / "mini/core.py").read_text() == CORE_OLD
    with pytest.raises(pilot.PilotError, match="not empty"):
        pilot.extract_snapshot(root / "snapshots" / "mini_1.tgz", dest, sha, include_git=False)


@pytest.mark.parametrize("member", ["abs", "traversal", "symlink", "fifo"])
def test_an_unsafe_archive_is_refused(tmp_path, member):
    archive = tmp_path / "bad.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo({"abs": "/etc/evil", "traversal": "../evil"}.get(member, "./x"))
        if member == "symlink":
            info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
        elif member == "fifo":
            info.type = tarfile.FIFOTYPE
        tf.addfile(info, io.BytesIO(b"") if member in ("abs", "traversal") else None)
    with pytest.raises(pilot.PilotError):
        pilot.extract_snapshot(archive, tmp_path / "out", pilot.sha256_file(archive), include_git=False)
    assert not (tmp_path / "evil").exists()


# ---- the agent workspace and the evaluator-only material -------------------------------------------------------------------------
def test_gold_and_test_patches_never_enter_the_agent_workspace_or_its_inputs(dataset, tmp_path):
    record = pilot.run_pilot(cfg_for(dataset, tmp_path), FakeRunner(), agent_cmd=fake_agent())
    run_dir = tmp_path / "out" / record["run_id"]
    agent_files = [run_dir / "agent" / n for n in ("harness.yaml", "prompt.txt", "params.json")]
    for p in agent_files:
        text = p.read_text()
        assert GOLD not in text and TEST_PATCH not in text and DECOY not in text, p.name
    assert STATEMENT in (run_dir / "agent" / "prompt.txt").read_text()
    argv = " ".join(record["agent"]["command"])
    assert GOLD not in argv and TEST_PATCH not in argv


def test_the_workspace_is_fresh_has_no_git_and_sits_beside_not_under_the_artifacts(dataset, tmp_path):
    cfg = cfg_for(dataset, tmp_path, keep_workspaces=True)
    record = pilot.run_pilot(cfg, FakeRunner(), agent_cmd=fake_agent("pass"))
    run_dir = tmp_path / "out" / record["run_id"]
    ws = run_dir / "agent" / "workspace"
    assert sorted(p.name for p in ws.iterdir()) == ["README.md", "mini", "tests"]  # no .git, no run artifacts
    artifacts = {"run.json", "candidate.patch", "evaluation.json", "params.json", "harness.yaml", "prompt.txt", "agent_result.json", "child.stdout.txt"}
    assert artifacts.isdisjoint(p.name for p in ws.rglob("*"))
    assert (run_dir / "evaluator" / "pristine").exists() and not (run_dir / "evaluator" / "pristine").samefile(ws)


def test_the_leak_scan_is_content_based_not_extension_based(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "ok.py").write_text("x = 1\n")
    pilot.leak_scan(ws, [GOLD, TEST_PATCH])
    (ws / "innocent.png").write_text("header\n" + TEST_PATCH + "\n")  # the extension says nothing
    with pytest.raises(pilot.PilotError, match="evaluator-only text"):
        pilot.leak_scan(ws, [GOLD, TEST_PATCH])
    (ws / "innocent.png").unlink()
    (ws / "tasks.jsonl").write_text("{}\n")
    with pytest.raises(pilot.PilotError, match="evaluator file name"):
        pilot.leak_scan(ws, [GOLD])
    (ws / "tasks.jsonl").unlink()
    (ws / "link").symlink_to("/etc/passwd")
    with pytest.raises(pilot.PilotError, match="symbolic link"):
        pilot.leak_scan(ws, [GOLD])


# ---- candidate patch collection --------------------------------------------------------------------------------------------------
def trees(tmp_path, files_before: dict, files_after: dict):
    a, b = tmp_path / "a", tmp_path / "b"
    for root, files in ((a, files_before), (b, files_after)):
        for rel, text in files.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text)
        root.mkdir(exist_ok=True)
    return a, b


@needs_patch
def test_collected_patch_covers_modify_add_delete_and_missing_final_newline_and_round_trips(tmp_path):
    before = {"m.py": "a\nb\nc\n", "gone.py": "x\n", "nonl.py": "one\ntwo"}
    after = {"m.py": "a\nB\nc\n", "new/n.py": "hello\n", "nonl.py": "one\ntwo\nthree"}
    a, b = trees(tmp_path, before, after)
    text, meta = pilot.collect_patch(a, b)
    assert meta["included_files"] == ["gone.py", "m.py", "new/n.py", "nonl.py"] and meta["protected_changed"] == []
    assert pilot.apply_patch(a, text) == {"ok": True}
    assert pilot.manifest(a) == pilot.manifest(b) or {k: v for k, v in pilot.manifest(a).items() if k != "gone.py"} == pilot.manifest(b)
    assert (a / "m.py").read_text() == "a\nB\nc\n" and (a / "nonl.py").read_text() == "one\ntwo\nthree" and (a / "new/n.py").read_text() == "hello\n"


def test_protected_and_ignored_files_are_left_out_of_the_candidate_and_listed(tmp_path):
    before = {"pkg/a.py": "1\n", "tests/test_a.py": "t\n", "pytest.ini": "[pytest]\n", "pkg/conftest.py": "c\n"}
    after = {"pkg/a.py": "2\n", "tests/test_a.py": "tampered\n", "pytest.ini": "x\n", "pkg/conftest.py": "d\n", "tests/test_new.py": "n\n", "pkg/__pycache__/a.pyc": "z", "x.egg-info/PKG": "z", "notes.pth": "p"}
    a, b = trees(tmp_path, before, after)
    text, meta = pilot.collect_patch(a, b)
    assert meta["included_files"] == ["pkg/a.py"]
    assert meta["protected_changed"] == ["notes.pth", "pkg/conftest.py", "pytest.ini", "tests/test_a.py", "tests/test_new.py"]
    assert "tampered" not in text and "__pycache__" not in text and "egg-info" not in text


def test_collection_fails_closed_on_binary_changes_links_and_too_many_files(tmp_path):
    a, b = trees(tmp_path, {"a.py": "1\n"}, {"a.py": "1\n"})
    (b / "blob.bin").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(pilot.PilotError, match="not valid UTF-8"):
        pilot.collect_patch(a, b)
    (b / "blob.bin").unlink()
    (b / "sneaky").symlink_to("/etc/passwd")
    with pytest.raises(pilot.PilotError, match="symbolic link"):
        pilot.collect_patch(a, b)
    (b / "sneaky").unlink()
    for i in range(pilot.MAX_CHANGED_FILES + 1):
        (b / f"f{i}.py").write_text("x\n")
    with pytest.raises(pilot.PilotError, match="exceed the limit"):
        pilot.collect_patch(a, b)


def test_a_leftover_edit_file_temporary_never_becomes_a_candidate_file_but_is_reported(tmp_path):
    a, b = trees(tmp_path, {"pkg/mod.py": "1\n"}, {"pkg/mod.py": "2\n", "pkg/.mod.py.edit-0123456789ab": "2\n", ".x.edit-abcdefabcdef": "x\n"})
    text, meta = pilot.collect_patch(a, b)
    assert meta["included_files"] == ["pkg/mod.py"] and meta["edit_temporaries_ignored"] == [".x.edit-abcdefabcdef", "pkg/.mod.py.edit-0123456789ab"]
    assert ".edit-" not in text and "pkg/mod.py" in text
    assert pilot.EDIT_TEMPORARY.fullmatch(".mod.py.edit-" + "0123456789ab")


@pytest.mark.parametrize("name", ["mod.py.edit-0123456789ab", ".mod.py.edit-0123456789", ".mod.py.edit-0123456789abc", ".mod.py.edit-0123456789AB", ".mod.py.edit-", "notes.txt", ".edit-0123456789ab"])
def test_only_the_exact_temporary_name_pattern_is_ignored(tmp_path, name):
    a, b = trees(tmp_path, {"keep.py": "1\n"}, {"keep.py": "1\n", name: "new\n"})
    text, meta = pilot.collect_patch(a, b)
    assert meta["edit_temporaries_ignored"] == [] and meta["included_files"] == [name] and name in text


@pytest.mark.parametrize("bad", ["diff --git a/x b/x\nindex 1..2\n--- a/x\n+++ b/x\n", "--- a/../x\n+++ b/../x\n@@ -1 +1 @@\n-a\n+b\n", "--- a/x\n+++ b//etc/x\n@@ -1 +1 @@\n-a\n+b\n",
                                 "--- x\n+++ y\n@@ -1 +1 @@\n-a\n+b\n", "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\nGIT binary patch\n"])
def test_unsupported_or_unsafe_patch_syntax_is_refused(bad):
    with pytest.raises(pilot.PilotError):
        pilot.split_patch(bad)


def test_protected_hunks_are_stripped_from_an_external_patch_including_the_official_test_files():
    patch = pilot._diff("pkg/a.py", "1\n", "2\n") + pilot._diff("tests/test_x.py", "1\n", "2\n") + pilot._diff("docs/guide.py", "1\n", "2\n") + pilot._diff("pyproject.toml", "1\n", "2\n")
    kept, stripped = pilot.strip_protected(patch, {"docs/guide.py"})
    assert stripped == ["tests/test_x.py", "docs/guide.py", "pyproject.toml"] and "pkg/a.py" in kept and "tests/test_x.py" not in kept


# ---- evaluation -------------------------------------------------------------------------------------------------------------------
@needs_patch
def test_evaluation_applies_candidate_then_official_test_patch_to_a_fresh_pristine_copy(spec, tmp_path):
    runner = FakeRunner(pytest_output(2))
    rec = pilot.evaluate(spec, GOLD, tmp_path / "ev", runner)
    tree = runner.calls[0]["tree"]
    assert tree["mini/core.py"] == CORE_NEW and tree["tests/test_core.py"] == TEST_NEW and ".git/config" not in tree
    assert (rec["outcome"], rec["resolved"], rec["problems"]) == ("resolved", True, [])
    assert not (tmp_path / "ev" / "workspace").exists()  # the evaluator removes its own copy


@needs_patch
@needs_patch
def test_a_candidate_patch_without_a_final_newline_still_applies(spec, tmp_path):
    rec = pilot.evaluate(spec, GOLD.rstrip("\n"), tmp_path / "ev", FakeRunner(pytest_output(2)))
    assert rec["outcome"] == "resolved" and rec["steps"]["candidate_patch"] == {"ok": True}


@needs_patch
def test_a_candidate_that_edits_the_official_test_file_cannot_change_the_outcome(spec, tmp_path):
    hostile = GOLD + pilot._diff("tests/test_core.py", TEST_OLD, "def test_add():\n    assert True\n")
    runner = FakeRunner(pytest_output(2))
    rec = pilot.evaluate(spec, hostile, tmp_path / "ev", runner)
    assert runner.calls[0]["tree"]["tests/test_core.py"] == TEST_NEW  # the official version, whatever the candidate said
    assert rec["steps"]["protected_stripped"] == ["tests/test_core.py"]


@needs_patch
def test_the_verdict_is_the_evaluators_not_the_agents_and_requires_the_new_tests_to_pass(spec, tmp_path):
    base = dict(spec=spec, candidate=GOLD)
    failing = pilot.evaluate(spec, GOLD, tmp_path / "e1", FakeRunner(pytest_output(1, 1), exit_code=1))
    assert (failing["outcome"], failing["resolved"]) == ("unresolved_tests_failed", False) and failing["result"]["failed"] == 1
    skipped_required = pilot.evaluate(spec, GOLD, tmp_path / "e2", FakeRunner(pytest_output(1, 0, required=()), exit_code=0))
    assert skipped_required["resolved"] is False and any("required test test_add_three" in p for p in skipped_required["problems"])
    lying = pilot.evaluate(spec, GOLD, tmp_path / "e3", FakeRunner(pytest_output(2), exit_code=1))  # PASSED lines but a failing exit status
    assert lying["resolved"] is False and any("exit status 1" in p for p in lying["problems"])
    assert base["candidate"] == GOLD


@needs_patch
def test_empty_non_applying_and_timed_out_candidates_are_unresolved_and_infrastructure_faults_fail_closed(spec, tmp_path):
    runner = FakeRunner()
    empty = pilot.evaluate(spec, "", tmp_path / "e1", runner)
    assert (empty["outcome"], empty["resolved"]) == ("unresolved_empty_patch", False) and runner.calls == []
    nonapplying = pilot.evaluate(spec, pilot._diff("mini/core.py", "something else\n", "x\n"), tmp_path / "e2", runner)
    assert (nonapplying["outcome"], nonapplying["resolved"]) == ("unresolved_patch_failed", False) and runner.calls == []
    timed = pilot.evaluate(spec, GOLD, tmp_path / "e3", FakeRunner(timed_out=True, exit_code=124, stdout=""))
    assert (timed["outcome"], timed["resolved"]) == ("error", False) and "outer timeout" in " ".join(timed["problems"])  # the OUTER container timeout is an evaluator failure
    boom = pilot.evaluate(spec, GOLD, tmp_path / "e4", FakeRunner(raises=pilot.PilotError("container isolation not as configured (Binds)")))
    assert (boom["outcome"], boom["resolved"]) == ("error", False) and "isolation" in boom["problems"][0]
    broken = dataclasses.replace(spec, test_patch=pilot._diff("tests/test_core.py", "not the file\n", "x\n"))
    assert pilot.evaluate(broken, GOLD, tmp_path / "e5", runner)["outcome"] == "error"
    assert not any((tmp_path / n / "workspace").exists() for n in ("e1", "e2", "e3", "e4", "e5"))


@needs_patch
def test_the_run_even_if_empty_switch_gives_the_baseline_condition(spec, tmp_path):
    runner = FakeRunner(pytest_output(1, 1), exit_code=1)
    rec = pilot.evaluate(spec, "", tmp_path / "ev", runner, label="baseline", run_even_if_empty=True)
    assert runner.calls[0]["tree"]["mini/core.py"] == CORE_OLD and runner.calls[0]["tree"]["tests/test_core.py"] == TEST_NEW
    assert rec["outcome"] == "unresolved_tests_failed" and rec["label"] == "baseline"


@needs_patch
def test_the_evaluation_record_has_a_stable_machine_readable_schema(spec, tmp_path):
    rec = pilot.evaluate(spec, GOLD, tmp_path / "ev", FakeRunner(pytest_output(2)))
    assert {"schema_version", "task_id", "label", "started", "finished", "snapshot_sha256", "test_command", "required_tests", "candidate_sha256", "candidate_bytes", "steps",
            "result", "resolved", "outcome", "problems"} <= set(rec)
    assert {"exit_code", "passed", "failed", "errors", "skipped", "pytest_exit", "python", "pytest", "stdout_sha256", "output_tail", "wall_s"} <= set(rec["result"])
    assert rec["schema_version"] == 1 and json.loads(json.dumps(rec)) == rec


def test_the_container_command_is_locked_down_and_limits_are_explicit():
    argv = pilot.create_argv("n", "img:1", pilot.ContainerLimits(), "true", "rid")
    joined = " ".join(argv)
    for needed in ("--pull never", "--network none", "--ipc none", "--cap-drop ALL", "no-new-privileges", "--memory 1g", "--memory-swap 1g", "--cpus 2", "--pids-limit 256",
                   "nofile=1024:1024", "fsize=104857600", "core=0", "--tmpfs /data:"):
        assert needed in joined, needed
    for forbidden in ("-v", "--volume", "--mount", "--privileged", "docker.sock", "--network host", "--device", "--pid host", "--add-host"):
        assert forbidden not in argv and forbidden not in joined, forbidden
    assert "--user" in argv and argv[argv.index("--user") + 1] != "0:0"


class FakeDocker:
    """A stand-in for the `docker` command (patched over subprocess.run): records every call, answers from a table."""

    def __init__(self, *, host_config=None, image_ok=True, start_timeout=False) -> None:
        self.calls, self.image_ok, self.start_timeout = [], image_ok, start_timeout
        self.host_config = host_config or {"NetworkMode": "none", "Privileged": False, "CapDrop": ["ALL"], "Binds": None, "Mounts": None,
                                           "Memory": 1073741824, "PidsLimit": 256, "NanoCpus": 2000000000}

    def __call__(self, args, **kw):
        import subprocess

        self.calls.append(list(args))
        verb = args[1]
        done = lambda out="", code=0, err="": subprocess.CompletedProcess(args, code, out, err)  # noqa: E731
        if verb == "image":
            return done("sha256:abc\n") if self.image_ok else done("", 1, "No such image")
        if verb == "inspect":
            return done(json.dumps([{"HostConfig": self.host_config, "State": {"OOMKilled": False}}]))
        if verb == "start":
            if self.start_timeout:
                raise subprocess.TimeoutExpired(args, 1)
            return done(pytest_output(2))
        return done()

    def verbs(self):
        return [c[1] for c in self.calls]


def docker_runner(tmp_path):
    return pilot.DockerRunner(tmp_path / "py", tmp_path / "site", "rid1", image="img:1")


def test_the_docker_runner_copies_in_only_python_site_and_the_workspace_and_always_removes_its_container(tmp_path, monkeypatch):
    fake = FakeDocker()
    monkeypatch.setattr(pilot.subprocess, "run", fake)
    out = docker_runner(tmp_path).run(tmp_path / "ws", "true", "cand", tmp_path / "logs")
    assert out["exit_code"] == 0 and out["isolation"]["NetworkMode"] == "none" and out["image_id"] == "sha256:abc" and out["timed_out"] is False
    copies = [c[4] for c in fake.calls if c[1] == "cp"]
    assert [d.split(":", 1)[1] for d in copies] == ["/opt/python", "/opt/site", "/workspace"] and all(d.startswith("coding-eval-rid1-cand:") for d in copies)
    assert fake.verbs()[-1] == "rm" and fake.calls[-1][-1] == "coding-eval-rid1-cand" and fake.calls[-1][2:4] == ["-f", "-v"]
    assert (tmp_path / "logs" / "cand.stdout.txt").exists()


def test_system_python_allows_only_readonly_runtime_binds(tmp_path, monkeypatch):
    fake = FakeDocker(host_config={
        "NetworkMode": "none", "Privileged": False, "CapDrop": ["ALL"],
        "Binds": None,
        "Mounts": [
            {"Type": "bind", "Target": "/usr", "ReadOnly": True},
            {"Type": "bind", "Target": "/lib", "ReadOnly": True},
        ],
        "Memory": 1073741824, "PidsLimit": 256, "NanoCpus": 2000000000,
    })
    monkeypatch.setattr(pilot.subprocess, "run", fake)
    runner = pilot.DockerRunner(Path("/usr"), tmp_path / "site", "rid1", image="img:1")
    out = runner.run(tmp_path / "ws", "true", "cand", tmp_path / "logs")
    assert out["exit_code"] == 0


@pytest.mark.parametrize("field,value", [("NetworkMode", "bridge"), ("Privileged", True), ("CapDrop", None), ("Binds", ["/home:/h"]), ("Mounts", [{"Source": "/"}])])
def test_a_container_whose_isolation_differs_from_the_configuration_is_never_started(tmp_path, monkeypatch, field, value):
    fake = FakeDocker()
    fake.host_config[field] = value
    monkeypatch.setattr(pilot.subprocess, "run", fake)
    with pytest.raises(pilot.PilotError, match="isolation not as configured"):
        docker_runner(tmp_path).run(tmp_path / "ws", "true", "cand", tmp_path / "logs")
    assert "start" not in fake.verbs() and fake.verbs()[-1] == "rm"


def test_a_missing_local_image_fails_closed_without_pulling_or_creating_anything(tmp_path, monkeypatch):
    fake = FakeDocker(image_ok=False)
    monkeypatch.setattr(pilot.subprocess, "run", fake)
    with pytest.raises(pilot.PilotError, match="never pulls"):
        docker_runner(tmp_path).run(tmp_path / "ws", "true", "cand", tmp_path / "logs")
    assert set(fake.verbs()) <= {"image", "rm"} and "pull" not in fake.verbs()


def test_the_outer_timeout_kills_the_container_and_reports_it(tmp_path, monkeypatch):
    fake = FakeDocker(start_timeout=True)
    monkeypatch.setattr(pilot.subprocess, "run", fake)
    out = docker_runner(tmp_path).run(tmp_path / "ws", "true", "cand", tmp_path / "logs")
    assert out["timed_out"] is True and out["exit_code"] == 124 and out["stdout"] == ""
    assert fake.verbs().index("kill") < len(fake.verbs()) - 1 and fake.verbs()[-1] == "rm"


def test_pytest_output_parsing_and_judging():
    parsed = pilot.parse_pytest(pytest_output(2))
    assert (parsed["passed"], parsed["failed"], parsed["pytest_exit"], parsed["pytest"]) == (2, 0, 0, "pytest 8.3.4")
    assert pilot.judge(parsed, 0, ("test_add_three",)) == (True, [])
    assert pilot.judge(parsed, 0, ("test_missing",))[0] is False
    assert pilot.parse_pytest("garbage")["pytest_exit"] is None


# ---- the whole run, with a fake agent process -----------------------------------------------------------------------------------
@needs_patch
def test_a_run_collects_the_candidate_and_evaluates_it_independently(dataset, tmp_path):
    runner = FakeRunner(pytest_output(2))
    record = pilot.run_pilot(cfg_for(dataset, tmp_path), runner, agent_cmd=fake_agent())
    run_dir = tmp_path / "out" / record["run_id"]
    assert (record["status"], record["task_id"], record["resolved"]) == ("complete", "mini_1", True)
    assert record["candidate"]["included_files"] == ["mini/core.py", "notes.txt"] and record["candidate"]["protected_changed"] == ["tests/test_core.py"]
    patch = (run_dir / "candidate.patch").read_text()
    assert "a + b" in patch and "tampered" not in patch
    assert runner.calls[0]["tree"]["tests/test_core.py"] == TEST_NEW  # the official test file, not the agent's
    saved = json.loads((run_dir / "evaluation.json").read_text())
    assert saved["resolved"] is True and saved["candidate_sha256"] == record["candidate"]["patch_sha256"]
    assert json.loads((run_dir / "run.json").read_text())["deviations"]
    assert record["agent"]["exit_code"] == 0 and record["agent"]["timed_out"] is False


@needs_patch
def test_an_agent_that_claims_success_does_not_make_the_run_resolved(dataset, tmp_path):
    claims = "import json, pathlib\np = json.load(open('params.json'))\njson.dump({'agent_reported_success': True, 'output': 'fixed it'}, open('agent_result.json', 'w'))\n"
    record = pilot.run_pilot(cfg_for(dataset, tmp_path), FakeRunner(), agent_cmd=fake_agent(claims))
    assert record["agent"]["result"]["agent_reported_success"] is True
    assert (record["resolved"], record["evaluation"]["outcome"]) == (False, "unresolved_empty_patch")


@needs_patch
def test_cleanup_removes_only_this_runs_workspaces_and_keeps_artifacts_and_neighbours(dataset, tmp_path):
    out = tmp_path / "out"
    (out / "other-run").mkdir(parents=True)
    (out / "other-run" / "keep.txt").write_text("not mine\n")
    record = pilot.run_pilot(cfg_for(dataset, tmp_path), FakeRunner(), agent_cmd=fake_agent())
    run_dir = out / record["run_id"]
    assert not (run_dir / "agent" / "workspace").exists() and not (run_dir / "evaluator" / "pristine").exists()
    for kept in ("run.json", "candidate.patch", "evaluation.json", "agent/params.json", "agent/harness.yaml", "agent/child.stderr.txt"):
        assert (run_dir / kept).exists(), kept
    assert (out / "other-run" / "keep.txt").read_text() == "not mine\n"


@needs_patch
def test_repeated_runs_of_the_same_agent_give_the_same_candidate_and_verdict(dataset, tmp_path):
    first = pilot.run_pilot(cfg_for(dataset, tmp_path), FakeRunner(pytest_output(2)), agent_cmd=fake_agent())
    second = pilot.run_pilot(cfg_for(dataset, tmp_path), FakeRunner(pytest_output(2)), agent_cmd=fake_agent())
    assert first["run_id"] != second["run_id"]
    assert first["candidate"] == second["candidate"]
    assert (first["resolved"], first["evaluation"]["outcome"]) == (second["resolved"], second["evaluation"]["outcome"])


@linux_only
@needs_patch
def test_the_wall_clock_limit_really_stops_the_agent_and_the_partial_work_is_still_evaluated(dataset, tmp_path):
    slow = FAKE_AGENT + "import time; time.sleep(120)\n"
    started = __import__("time").monotonic()
    record = pilot.run_pilot(cfg_for(dataset, tmp_path, agent_timeout_s=2), FakeRunner(pytest_output(2)), agent_cmd=fake_agent(slow))
    assert __import__("time").monotonic() - started < 30
    assert record["agent"]["timed_out"] is True and record["agent"]["exit_code"] != 0
    assert record["classification"]["agent"] == "timed_out" and record["classification"]["attempt_outcome"] == "timed_out" and record["classification"]["counts_as_coding_failure"] is False
    assert record["candidate"]["included_files"] == ["mini/core.py", "notes.txt"]  # the edits made before the limit are collected
    run_dir = str((tmp_path / "out" / record["run_id"]).resolve())
    left = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                cwd = os.readlink(entry / "cwd")
            except OSError:
                continue
            if cwd.startswith(run_dir):
                left.append((entry.name, cwd))
    assert left == []


def test_artifacts_must_live_outside_the_repository_and_the_dataset(dataset, tmp_path):
    root, _ = dataset
    for bad in (pilot.REPO / "experiments" / "x", pilot.REPO, root / "inside", root):
        with pytest.raises(pilot.PilotError, match="must be outside"):
            pilot.check_out_dir(bad, root)
    assert pilot.check_out_dir(tmp_path / "fine", root) == (tmp_path / "fine").resolve()


def test_the_cli_refuses_to_call_a_model_without_explicit_permission(tmp_path, capsys):
    code = pilot.main(["run", "--out", str(tmp_path / "o"), "--model", "x:1"])
    assert code == 2 and "--allow-model-call" in capsys.readouterr().err


# ---- failure classification, provenance and model identity (readiness for the first real attempt) ----------------------------------
def run_record(**over):
    base = {"status": "complete", "resolved": False, "agent": {"timed_out": False, "exit_code": 0, "result": {"agent_reported_success": True, "error": None}},
            "candidate": {"included_files": ["mini/core.py"]}, "evaluation": {"outcome": "unresolved_tests_failed", "problems": ["x"]}}
    return {**base, **over}


@pytest.mark.parametrize("label,record,expected", [
    ("resolved", run_record(resolved=True, evaluation={"outcome": "resolved"}), ("resolved", False, False)),
    ("coding failure: agent finished, tests fail", run_record(), ("unresolved", False, True)),
    ("coding failure: agent finished, nothing changed", run_record(candidate={"included_files": []}, evaluation={"outcome": "unresolved_empty_patch"}), ("unresolved", False, True)),
    ("coding failure: the agent reported failure", run_record(agent={"timed_out": False, "exit_code": 1, "result": {"agent_reported_success": False, "error": None}}), ("unresolved", False, True)),
    ("coding failure: the patch does not apply", run_record(evaluation={"outcome": "unresolved_patch_failed"}), ("unresolved", False, True)),
    ("INFRASTRUCTURE: the harness raised (endpoint unreachable)", run_record(agent={"timed_out": False, "exit_code": 3, "result": {"error": "BackendUnreachableError: connection refused"}},
                                                                             candidate={"included_files": []}, evaluation={"outcome": "unresolved_empty_patch"}), ("infrastructure_failure", True, False)),
    ("INFRASTRUCTURE: no result file", run_record(agent={"timed_out": False, "exit_code": -9, "result": None}), ("infrastructure_failure", True, False)),
    ("INFRASTRUCTURE: the evaluator errored", run_record(evaluation={"outcome": "error", "problems": ["container isolation not as configured"]}), ("infrastructure_failure", True, False)),
    ("INFRASTRUCTURE: evaluation never ran", run_record(evaluation=None, status="refused", error="candidate collection refused"), ("infrastructure_failure", True, False)),
    ("INFRASTRUCTURE: setup refused before the agent", {"status": "refused", "error": "the model identity could not be established", "agent": None, "candidate": None, "evaluation": None, "resolved": False},
     ("infrastructure_failure", True, False)),
    ("timeout is reported on its own", run_record(agent={"timed_out": True, "exit_code": -9, "result": None}), ("timed_out", False, False)),
])
def test_runs_are_classified_so_that_infrastructure_failures_never_count_as_coding_failures(label, record, expected):
    c = pilot.classify_run(record)
    assert (c["attempt_outcome"], c["infrastructure_failure"], c["counts_as_coding_failure"]) == expected, (label, c)
    assert (c["infrastructure_reasons"] != []) == c["infrastructure_failure"]


@needs_patch
def test_a_run_records_classification_provenance_hashes_identity_and_basis(dataset, tmp_path):
    cfg = cfg_for(dataset, tmp_path)
    record = pilot.run_pilot(cfg, FakeRunner(pytest_output(2)), agent_cmd=fake_agent())
    run_dir = tmp_path / "out" / record["run_id"]
    saved = json.loads((run_dir / "run.json").read_text())
    assert saved["classification"] == {"agent": "completed", "candidate": "collected", "evaluator": "resolved", "infrastructure_failure": False, "infrastructure_reasons": [], "indeterminate_reasons": [],
                                       "attempt_outcome": "resolved", "counts_as_coding_failure": False}
    assert saved["hashes"]["prompt_sha256"] == pilot.sha256_file(run_dir / "agent" / "prompt.txt") and saved["hashes"]["harness_yaml_sha256"] == pilot.sha256_file(run_dir / "agent" / "harness.yaml")
    assert saved["hashes"]["worker_prompt_sha256"] == pilot.sha256_bytes(pilot.WORKER_PROMPT.encode())
    assert saved["model_identity"]["digest"] == "sha256:" + "ab" * 32 and saved["model_identity"]["tag"] == "big:1" and saved["model_identity"]["quantization_level"] == "Q4_K_M"
    for key in ("driver_sha256", "tools_py_sha256", "repo_head", "repo_status", "python", "platform", "effgen", "cpu_count"):
        assert key in saved["provenance"]
    assert saved["provenance"]["driver_sha256"] == pilot.sha256_file(Path(pilot.__file__)) and len(saved["provenance"]["repo_head"]) == 40
    assert saved["config"]["max_tokens"] == 768 and saved["config"]["temperature"] == 0.1 and saved["config"]["seed"] == 0 and saved["container_limits"]["memory"] == "1g"
    assert saved["snapshot_sha256"] == cfg.snapshot_sha256 and saved["declared_base_commit"] == "0" * 40


def test_the_rich_3278_basis_is_always_labelled_snapshot_based():
    basis = pilot.BASIS["rich_3278"]
    assert basis["experiment_basis"] == "snapshot-based" and basis["declared_base_commit_in_snapshot"] is False and basis["snapshot_head"].startswith("200f23b2")


@needs_patch
def test_an_unreachable_model_endpoint_stops_the_run_before_the_agent_and_is_not_a_coding_failure(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "fetch_model_identity", lambda *a, **kw: {"available": False, "reason": "URLError: connection refused"})
    marker_cmd = fake_agent("open('agent_started.txt', 'w').write('started')\n")  # would leave a marker if the agent were started
    record = pilot.run_pilot(cfg_for(dataset, tmp_path), FakeRunner(), agent_cmd=marker_cmd)
    run_dir = tmp_path / "out" / record["run_id"]
    assert record["status"] == "refused" and "agent was not started" in record["error"] and record["agent"] is None
    assert not (run_dir / "agent" / "agent_started.txt").exists() and not (run_dir / "agent" / "params.json").exists()
    c = record["classification"]
    assert (c["agent"], c["attempt_outcome"], c["infrastructure_failure"], c["counts_as_coding_failure"]) == ("not_run", "infrastructure_failure", True, False)
    assert json.loads((run_dir / "run.json").read_text())["classification"] == c


@needs_patch
def test_an_agent_exception_is_an_infrastructure_failure_even_though_the_candidate_is_empty(dataset, tmp_path):
    raising = fake_agent("import json\njson.dump({'error': 'BackendUnreachableError: connection refused', 'agent_reported_success': None}, open('agent_result.json', 'w'))\nraise SystemExit(3)\n")
    record = pilot.run_pilot(cfg_for(dataset, tmp_path), FakeRunner(), agent_cmd=raising)
    assert record["evaluation"]["outcome"] == "unresolved_empty_patch" and record["resolved"] is False  # what the evaluator literally says...
    c = record["classification"]
    assert (c["agent"], c["attempt_outcome"], c["counts_as_coding_failure"]) == ("infrastructure_failure", "infrastructure_failure", False)  # ...is not a coding result
    assert "BackendUnreachableError" in c["infrastructure_reasons"][0]


@needs_patch
def test_an_evaluator_fault_after_a_normal_agent_run_is_an_infrastructure_failure(dataset, tmp_path):
    record = pilot.run_pilot(cfg_for(dataset, tmp_path), FakeRunner(raises=pilot.PilotError("docker create failed")), agent_cmd=fake_agent())
    c = record["classification"]
    assert (c["agent"], c["evaluator"], c["attempt_outcome"], c["counts_as_coding_failure"]) == ("completed", "infrastructure_failure", "infrastructure_failure", False)


@needs_patch
def test_an_agent_that_finished_but_failed_the_tests_counts_as_a_coding_failure(dataset, tmp_path):
    record = pilot.run_pilot(cfg_for(dataset, tmp_path), FakeRunner(pytest_output(1, 1), exit_code=1), agent_cmd=fake_agent())
    c = record["classification"]
    assert (c["agent"], c["evaluator"], c["attempt_outcome"], c["infrastructure_failure"], c["counts_as_coding_failure"]) == ("completed", "unresolved_tests_failed", "unresolved", False, True)


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=-1):
        return self.body[:n] if n >= 0 else self.body


def test_the_model_identity_is_read_from_the_ollama_tags_endpoint_without_inference():
    body = json.dumps({"models": [{"name": "other:1", "digest": "x"}, {"name": "smol-qwen3-14b:8k", "model": "smol-qwen3-14b:8k", "digest": "bb17a48da83e4e84", "size": 9, "modified_at": "2026-10-01",
                                   "details": {"family": "qwen3", "parameter_size": "14.8B", "quantization_level": "Q4_K_M"}}]}).encode()
    seen = []

    def opener(url, timeout):
        seen.append((url, timeout))
        return _Response(body)

    found = REAL_FETCH_MODEL_IDENTITY("http://127.0.0.1:11434/v1", "smol-qwen3-14b:8k", timeout=3, opener=opener)
    assert seen == [("http://127.0.0.1:11434/api/tags", 3)]  # the /v1 suffix is dropped; a GET of metadata only
    assert found == {"available": True, "tag": "smol-qwen3-14b:8k", "digest": "bb17a48da83e4e84", "size_bytes": 9, "modified_at": "2026-10-01", "family": "qwen3",
                     "parameter_size": "14.8B", "quantization_level": "Q4_K_M", "provider": "ollama"}
    missing = REAL_FETCH_MODEL_IDENTITY("http://h:1/v1", "absent:1", opener=lambda u, timeout: _Response(body))
    assert missing["available"] is False and "does not list" in missing["reason"]

    def boom(url, timeout):
        raise OSError("refused")

    assert REAL_FETCH_MODEL_IDENTITY("http://h:1/v1", "x:1", opener=boom) == {"available": False, "reason": "OSError: refused"}
    assert REAL_FETCH_MODEL_IDENTITY("http://h:1", "x:1", opener=lambda u, timeout: _Response(b"not json"))["available"] is False


# ---- the agent sandbox (bubblewrap + systemd scope + prlimit): policy, bridge, escape probes, limits, cleanup ---------------------------------
SANDBOX_PROBLEM = pilot.sandbox_problem()
needs_sandbox = pytest.mark.skipif(SANDBOX_PROBLEM is not None, reason=f"the agent sandbox cannot be built on this host: {SANDBOX_PROBLEM}")
LIMITS_SMALL = pilot.SandboxLimits(memory_max="256M", tasks_max=32, fsize_bytes=1_048_576, nofile=64)


def mounts(argv):
    ro = [(argv[i + 1], argv[i + 2]) for i, a in enumerate(argv) if a == "--ro-bind"]
    rw = [(argv[i + 1], argv[i + 2]) for i, a in enumerate(argv) if a == "--bind"]
    return ro, rw


def test_the_sandbox_command_mounts_only_what_the_agent_needs_and_states_every_limit(tmp_path):
    agent = tmp_path / "run" / "agent"
    argv = pilot.sandbox_command(agent, ["python", "child.py"], extra_binds=(("/tmp/cb-x", "/bridge"),))
    assert argv[:4] == ["systemd-run", "--user", "--scope", "--quiet"]
    for needed in ("MemoryMax=2G", "MemorySwapMax=0", "TasksMax=256", "CPUQuota=200%", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv"):
        assert needed in argv, needed
    assert "--share-net" not in argv  # --unshare-all gives the child its own network namespace and nothing re-shares it
    ro, rw = mounts(argv)
    interpreter_dirs = {src for src, _ in ro if "python" in src.lower()}  # the uv-managed interpreter and its alias directory
    assert {src for src, _ in ro} <= {"/usr", sys.prefix, str(pilot.REPO / "src"), str(Path(pilot.__file__).resolve()), "/etc/ld.so.cache"} | interpreter_dirs
    assert all(src == dst for src, dst in ro)
    assert [dst for _, dst in rw] == [str(agent), "/bridge"]  # the ONLY writable mounts: the agent directory and the bridge socket directory
    home, repo = os.path.expanduser("~"), str(pilot.REPO)
    for src, _ in (*ro, *rw):
        assert src not in (home, repo, "/", "/etc", "/run", "/var", "/home", str(pilot.REPO / "dataset"), str(pilot.REPO / "docs"), str(pilot.REPO / "tests"))
        assert "docker.sock" not in src and "dataset" not in src and "evaluator" not in src
    keys = [argv[i + 1] for i, a in enumerate(argv) if a == "--setenv"]
    assert sorted(keys) == sorted(["PATH", "HOME", "LANG", "VIRTUAL_ENV", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE"])  # the sandbox starts from an empty environment
    tail = argv[argv.index("prlimit"):]  # the innermost layer: resource limits, then the child
    assert tail == ["prlimit", "--fsize=104857600", "--nofile=1024", "--core=0", "--", "python", "child.py"]


def test_the_model_endpoint_must_be_an_http_url_with_a_port():
    assert pilot.model_endpoint("http://127.0.0.1:11434/v1") == ("127.0.0.1", 11434)
    assert pilot.model_endpoint("http://100.126.219.9:11434") == ("100.126.219.9", 11434)
    for bad in ("http://127.0.0.1/v1", "https://h:1/v1", "ftp://h:1", "not a url", ""):
        with pytest.raises(pilot.PilotError, match="explicit port"):
            pilot.model_endpoint(bad)


class _Stub(__import__("http.server", fromlist=["x"]).BaseHTTPRequestHandler):
    """A stand-in for the Ollama server (NOT a model): answers the metadata calls and returns a fixed completion. Logs the paths it is asked for."""

    def _send(self, body, code=200):
        data = json.dumps(body).encode() if not isinstance(body, bytes) else body
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.server.paths.append(("GET", self.path))
        if self.path.startswith("/api/tags"):
            self._send({"models": [{"name": "big:1", "model": "big:1", "digest": "sha256:" + "cd" * 32, "size": 1, "details": {"family": "stub", "parameter_size": "0B", "quantization_level": "none"}}]})
        elif self.path.rstrip("/").endswith("/models"):
            self._send({"object": "list", "data": [{"id": "big:1", "object": "model"}]})
        else:
            self._send(self.server.tag if isinstance(self.server.tag, bytes) else b"stub")

    def do_POST(self):
        self.server.paths.append(("POST", self.path))
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._send({"id": "x", "object": "chat.completion", "created": 0, "model": "big:1", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Thought: nothing to do.\nFinal Answer: stub"}}]})

    def log_message(self, *a):
        pass


@pytest.fixture
def stub_servers():
    from http.server import ThreadingHTTPServer
    import threading

    servers = []

    def make(tag=b"STUB"):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
        srv.paths, srv.tag = [], tag
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return srv

    yield make
    for srv in servers:
        srv.shutdown()


def test_the_bridge_forwards_to_the_one_configured_endpoint_and_nothing_else(stub_servers):
    import socket
    import tempfile
    import urllib.request

    model, other = stub_servers(b"MODEL"), stub_servers(b"OTHER")
    with tempfile.TemporaryDirectory(prefix="cb-") as d:
        with pilot.ModelBridge(Path(d), "127.0.0.1", model.server_address[1]) as bridge:
            assert oct(bridge.path.stat().st_mode & 0o777) == "0o600"
            with socket.socket() as probe:  # find a free port for the loopback half
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            pilot.serve_loopback_bridge(port, str(bridge.path))
            assert urllib.request.urlopen(f"http://127.0.0.1:{port}/x", timeout=5).read() == b"MODEL"  # TCP -> unix socket -> TCP -> the model endpoint
            assert urllib.request.urlopen(f"http://127.0.0.1:{port}/y", timeout=5).read() == b"MODEL"  # and again (new connection)
        assert not bridge.path.exists()  # removed on exit
    assert other.paths == []  # the other server was never contacted


def test_a_real_run_is_refused_when_the_sandbox_cannot_be_built_and_never_runs_the_agent_as_the_host_user(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "sandbox_problem", lambda limits=None: "bwrap not found")
    record = pilot.run_pilot(cfg_for(dataset, tmp_path), FakeRunner())  # no injected agent command: this is the real path
    assert record["status"] == "refused" and "refusing to run the agent as the plain host user" in record["error"] and record["agent"] is None
    assert not (tmp_path / "out" / record["run_id"] / "agent" / "params.json").exists()  # nothing was launched, not even prepared
    c = record["classification"]
    assert (c["agent"], c["attempt_outcome"], c["infrastructure_failure"], c["counts_as_coding_failure"]) == ("not_run", "infrastructure_failure", True, False)


@needs_sandbox
def test_escape_probes_inside_the_sandbox(tmp_path, stub_servers):
    import socket
    import subprocess

    run_dir = tmp_path / "run"
    agent, evaluator = run_dir / "agent", run_dir / "evaluator"
    (agent / "workspace").mkdir(parents=True)
    evaluator.mkdir()
    (evaluator / "secret_gold.txt").write_text("GOLD\n")
    (agent / "workspace" / "ok.txt").write_text("mine\n")
    service = stub_servers(b"HOST-SERVICE")
    abstract = socket.socket(socket.AF_UNIX)
    abstract.bind("\0probe-host-abstract-" + str(os.getpid()))
    abstract.listen(1)
    repo, home = str(pilot.REPO), os.path.expanduser("~")
    probe = f'''
import json, os, socket, ctypes
def read(path):
    try:
        (os.listdir if os.path.isdir(path) else (lambda p: open(p, "rb").read(8)))(path); return "READABLE"
    except OSError as e: return type(e).__name__
def write(path):
    try: open(path, "w").write("x"); return "WROTE"
    except OSError as e: return type(e).__name__
def connect(addr, fam=socket.AF_INET):
    s = socket.socket(fam); s.settimeout(2)
    try: s.connect(addr); return "CONNECTED"
    except OSError as e: return type(e).__name__
R = {{"read": {{n: read(p) for n, p in {[("dataset", repo + "/dataset/tasks.jsonl"), ("docs", repo + "/docs/new-session.md"), ("repo tests", repo + "/tests"), ("repo git", repo + "/.git/config"), ("ssh", home + "/.ssh"), ("bashrc", home + "/.bashrc"), ("evaluator secret", str(evaluator / "secret_gold.txt")), ("passwd", "/etc/passwd"), ("shadow", "/etc/shadow"), ("root", "/root"), ("var", "/var/log"), ("run", "/run/user")]!r}}},
      "own": read({str(agent / "workspace" / "ok.txt")!r}), "write_own": write({str(agent / "PROBE_OK")!r}),
      "write_readonly": {{n: write(p) for n, p in {[("usr", "/usr/PROBE"), ("venv", sys.prefix + "/PROBE"), ("src", repo + "/src/PROBE")]!r}}},
      "net": {{"external": connect(("1.1.1.1", 53)), "host_service": connect(("127.0.0.1", {service.server_address[1]})), "abstract_unix": connect("\\0probe-host-abstract-{os.getpid()}", socket.AF_UNIX), "docker_sock": connect("/var/run/docker.sock", socket.AF_UNIX)}},
      "devices": [l.split(":")[0].strip() for l in open("/proc/net/dev").read().splitlines()[2:]], "pids": len([e for e in os.listdir("/proc") if e.isdigit()]), "env": sorted(os.environ)}}
st = dict(l.split(":\\t") for l in open("/proc/self/status").read().splitlines() if ":\\t" in l)
R["caps"] = [st["CapEff"].strip(), st["CapBnd"].strip(), st["NoNewPrivs"].strip()]
R["mount"] = ctypes.CDLL(None, use_errno=True).mount(b"none", b"/tmp", b"tmpfs", 0, None)
print(json.dumps(R))
'''
    done = subprocess.run(pilot.sandbox_command(agent, [sys.executable, "-c", probe]), capture_output=True, text=True, timeout=60, env=pilot._launcher_env())
    assert done.returncode == 0, done.stderr
    R = json.loads(done.stdout.strip().splitlines()[-1])
    assert set(R["read"].values()) == {"FileNotFoundError"}, R["read"]  # none of the host files exists inside the sandbox
    assert R["own"] == "READABLE" and R["write_own"] == "WROTE" and (agent / "PROBE_OK").exists()  # the agent directory is the one place it can write, and the host sees it
    assert set(R["write_readonly"].values()) == {"OSError"}
    assert R["net"]["external"] == "OSError" and R["net"]["host_service"] == "ConnectionRefusedError" and R["net"]["abstract_unix"] == "ConnectionRefusedError" and R["net"]["docker_sock"] == "FileNotFoundError"
    assert R["devices"] == ["lo"] and R["pids"] <= 3 and R["caps"] == ["0000000000000000", "0000000000000000", "1"] and R["mount"] == -1
    assert R["env"] == sorted(["HOME", "LANG", "PATH", "PWD", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE", "VIRTUAL_ENV"])  # no credentials, no session variables
    for leaked in (Path(repo) / "PROBE", Path(home) / "PROBE", evaluator / "PROBE", Path(repo) / "src" / "PROBE", run_dir / "PROBE"):
        assert not leaked.exists()
    assert service.paths == []  # the host service was never contacted
    abstract.close()


@needs_sandbox
def test_the_model_endpoint_is_reachable_from_the_sandbox_only_through_the_bridge(tmp_path, stub_servers):
    import subprocess
    import tempfile

    agent = tmp_path / "agent"
    agent.mkdir()
    model, service = stub_servers(b"MODEL-STANDIN"), stub_servers(b"HOST-SERVICE")
    probe = f'''
import json, sys, urllib.request, socket
sys.path.insert(0, {str(pilot.REPO / "experiments" / "coding_eval")!r}); import rich3278_pilot as p
R = {{}}
try: socket.create_connection(("127.0.0.1", {model.server_address[1]}), timeout=2); R["before"] = "CONNECTED"
except OSError: R["before"] = "unreachable"
p.serve_loopback_bridge({model.server_address[1]}, "/bridge/m.sock")
R["after"] = urllib.request.urlopen("http://127.0.0.1:{model.server_address[1]}/", timeout=5).read().decode()
try: socket.create_connection(("127.0.0.1", {service.server_address[1]}), timeout=2); R["other"] = "CONNECTED"
except OSError: R["other"] = "unreachable"
print(json.dumps(R))
'''
    with tempfile.TemporaryDirectory(prefix="cb-") as bridge_dir, pilot.ModelBridge(Path(bridge_dir), "127.0.0.1", model.server_address[1]):
        done = subprocess.run(pilot.sandbox_command(agent, [sys.executable, "-c", probe], extra_binds=((bridge_dir, "/bridge"),)), capture_output=True, text=True, timeout=60, env=pilot._launcher_env())
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {"before": "unreachable", "after": "MODEL-STANDIN", "other": "unreachable"}
    assert service.paths == []


@needs_sandbox
@pytest.mark.parametrize("name,code,check", [
    ("memory", "b = bytearray(1 << 30)\nfor i in range(0, len(b), 4096): b[i] = 1\nprint('ALLOCATED')", lambda rc, out: rc != 0 and "ALLOCATED" not in out),
    ("tasks", "import threading, time\nok = 0\nfor i in range(100):\n    try:\n        threading.Thread(target=time.sleep, args=(3,)).start(); ok += 1\n    except RuntimeError: break\nprint(ok)", lambda rc, out: rc == 0 and 0 < int(out) < 40),
    ("file size", "try:\n    open('big.bin', 'wb').write(b'x' * (4 << 20)); print('WROTE')\nexcept OSError as e: print('blocked', e.errno)", lambda rc, out: out == "blocked 27"),
    ("open files", "fs = []\ntry:\n    for i in range(200): fs.append(open('/proc/self/status'))\n    print('opened 200')\nexcept OSError: print('blocked at', len(fs))", lambda rc, out: out.startswith("blocked at") and int(out.split()[-1]) <= 64),
    ("core dumps", "import resource; print(resource.getrlimit(resource.RLIMIT_CORE))", lambda rc, out: out == "(0, 0)"),
])
def test_resource_limits_are_enforced_inside_the_sandbox(tmp_path, name, code, check):
    import subprocess

    agent = tmp_path / "agent"
    agent.mkdir()
    done = subprocess.run(pilot.sandbox_command(agent, [sys.executable, "-c", code], LIMITS_SMALL), capture_output=True, text=True, timeout=60, env=pilot._launcher_env())
    assert check(done.returncode, done.stdout.strip()), (name, done.returncode, done.stdout, done.stderr)


@linux_only
@needs_sandbox
def test_the_wall_clock_kill_removes_every_process_the_sandboxed_agent_started(tmp_path):
    import subprocess
    import time

    agent = tmp_path / "agent"
    agent.mkdir()
    marker = "SBX-CLEANUP-" + os.urandom(4).hex()
    code = f"import subprocess, sys, time\n# {marker}\nsubprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)  # {marker}'])\ntime.sleep(300)"
    result = pilot.run_agent_phase(agent, {"x": 1}, 3, pilot.sandbox_command(agent, [sys.executable, "-c", code], LIMITS_SMALL))
    time.sleep(1)
    assert result["timed_out"] is True and result["seconds"] < 15
    assert subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout.split() == []  # the child, its grandchild and the sandbox init are all gone


@needs_sandbox
def test_a_real_run_executes_the_child_in_the_sandbox_edits_a_file_and_reaches_the_endpoint_only_through_the_bridge(dataset, tmp_path, monkeypatch):
    servers = []  # the stand-in answers /api/tags like Ollama and scripts the chat completions (see editing_script): the coder really edits mini/core.py
    record = run_scripted_endpoint(dataset, tmp_path, monkeypatch, editing_script, server_out=servers)
    stub = servers[0]
    assert record["status"] == "complete" and record["model_identity"]["available"] is True
    assert record["sandbox"]["kind"].startswith("systemd scope + bubblewrap") and f"127.0.0.1:{stub.server_address[1]}" in record["sandbox"]["network"]
    assert record["agent"]["command"][:4] == ["systemd-run", "--user", "--scope", "--quiet"] and "--unshare-all" in record["agent"]["command"]
    assert ("GET", "/api/tags") in stub.paths  # asked by the host before the agent started
    assert any(m == "POST" and p.startswith("/v1/") for m, p in stub.paths)  # and the CHILD, inside the sandbox, reached the endpoint through the bridge
    run_dir = tmp_path / "out" / record["run_id"]
    config = (run_dir / "agent" / "harness.yaml").read_text()
    assert f"http://127.0.0.1:{stub.server_address[1]}/v1" in config and json.loads((run_dir / "agent" / "params.json").read_text())["bridge"]["socket"] == "/bridge/m.sock"
    result = record["agent"]["result"]
    assert result is not None and result["error"] is None and record["agent"]["exit_code"] == 0
    # the success path: the edit really happened inside the sandbox, so the harness's own verification accepts the agent's answer and the candidate reaches the evaluator
    assert result["agent_reported_success"] is True and result["workspace_changed"] is True and result["workers"][0]["status"] == "completed"
    assert record["candidate"]["included_files"] == ["mini/core.py"] and "+    return a + b" in (run_dir / "candidate.patch").read_text()
    assert record["evaluation"]["outcome"] == "resolved" and record["resolved"] is True  # (the container is a fake here; this checks the hand-over, not the tests)
    assert not list((run_dir / "agent").glob("*.sock")) and not (run_dir / "agent" / "workspace").exists()  # no socket in the run directory; the workspace was cleaned up


# ---- B1: the bridge must not cut off a slow non-streaming generation ---------------------------------------------------------------------------
class _Slow(__import__("http.server", fromlist=["x"]).BaseHTTPRequestHandler):
    """Stand-in HTTP server (not a model). `delay`: seconds before the response starts; `gap`: idle seconds in the middle of the body; `release`: an Event the response waits for."""

    protocol_version = "HTTP/1.1"

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.server.started.append(time.monotonic())
        self.server.release.wait(self.server.hold) if self.server.hold else None
        time.sleep(self.server.delay)
        body = b"0123456789"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body[:5])
            self.wfile.flush()
            time.sleep(self.server.gap)
            self.wfile.write(body[5:])
        except OSError:
            pass  # the client went away

    def log_message(self, *a):
        pass


@pytest.fixture
def slow_server():
    from http.server import ThreadingHTTPServer
    import threading

    made = []

    def make(delay=0.0, gap=0.0, hold=None):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _Slow)
        srv.daemon_threads = True
        srv.delay, srv.gap, srv.hold, srv.release, srv.started = delay, gap, hold, threading.Event(), []
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        made.append(srv)
        return srv

    yield make
    for srv in made:
        srv.release.set()
        srv.shutdown()
        srv.server_close()


def ask_through_bridge(bridge, timeout=40):
    """One POST over the bridge's unix socket; returns (response bytes, seconds)."""
    import socket

    c = socket.socket(socket.AF_UNIX)
    c.settimeout(timeout)
    c.connect(str(bridge.path))
    c.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
    start, data = time.monotonic(), b""
    while chunk := c.recv(65536):
        data += chunk
    c.close()
    return data, time.monotonic() - start


def open_fds():
    return len(os.listdir("/proc/self/fd"))


@pytest.mark.parametrize("shape,delay,gap", [("a response that starts late", 2.5, 0.0), ("an idle gap inside the response", 0.0, 2.5), ("both", 1.5, 2.5)])
def test_the_connect_timeout_does_not_cut_a_slow_generation_off(tmp_path, slow_server, monkeypatch, shape, delay, gap):
    monkeypatch.setattr(pilot, "CONNECT_TIMEOUT_S", 1)  # the old bug: this timeout stayed on the connected socket; 2.5 s of silence now exceeds it
    server = slow_server(delay, gap)
    with pilot.ModelBridge(tmp_path, "127.0.0.1", server.server_address[1]) as bridge:
        data, took = ask_through_bridge(bridge)
    assert data.startswith(b"HTTP/1.1 200") and data.endswith(b"0123456789"), (shape, data)
    assert took >= delay + gap


def test_a_response_that_starts_after_more_than_ten_seconds_arrives_with_the_real_constants(tmp_path, slow_server):
    assert pilot.CONNECT_TIMEOUT_S == 10  # unpatched: the exact configuration that failed in the review (12 s -> empty reply at 10.0 s)
    server = slow_server(delay=11.5)
    with pilot.ModelBridge(tmp_path, "127.0.0.1", server.server_address[1]) as bridge:
        data, took = ask_through_bridge(bridge)
    assert data.endswith(b"0123456789") and took >= 11.5


def test_ordinary_fast_responses_and_concurrent_connections_still_work(tmp_path, slow_server):
    import concurrent.futures

    server = slow_server()
    with pilot.ModelBridge(tmp_path, "127.0.0.1", server.server_address[1]) as bridge, concurrent.futures.ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: ask_through_bridge(bridge, 10), range(8)))
    assert all(d.endswith(b"0123456789") and t < 3 for d, t in results)


def test_connect_timeout_still_bounds_connection_establishment_and_a_dead_target_closes_the_client(tmp_path, monkeypatch):
    import socket

    seen = {}
    real = socket.create_connection
    monkeypatch.setattr(socket, "create_connection", lambda addr, timeout=None, **kw: (seen.setdefault("timeout", timeout), real(addr, timeout=timeout, **kw))[1])
    with socket.socket() as s:  # a port nobody listens on
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    with pilot.ModelBridge(tmp_path, "127.0.0.1", dead) as bridge:
        c = socket.socket(socket.AF_UNIX)
        c.settimeout(10)
        c.connect(str(bridge.path))
        assert c.recv(10) == b""  # closed promptly, not hung
        c.close()
    assert seen["timeout"] == pilot.CONNECT_TIMEOUT_S


def test_an_idle_connection_is_closed_after_the_idle_bound(tmp_path, slow_server, monkeypatch):
    monkeypatch.setattr(pilot, "RELAY_IDLE_S", 1)  # the relay's life is bounded: nothing moving for this long ends it
    server = slow_server(delay=4)
    with pilot.ModelBridge(tmp_path, "127.0.0.1", server.server_address[1]) as bridge:
        data, took = ask_through_bridge(bridge)
    assert data == b"" and 0.9 < took < 3.5


@linux_only
def test_a_client_that_disconnects_leaves_no_relay_thread_or_socket_behind(tmp_path, slow_server):
    import socket

    server = slow_server(delay=1.5)
    base_fds, base_threads = open_fds(), threading.active_count()
    with pilot.ModelBridge(tmp_path, "127.0.0.1", server.server_address[1]) as bridge:
        for _ in range(5):
            c = socket.socket(socket.AF_UNIX)
            c.connect(str(bridge.path))
            c.sendall(b"POST /v1/x HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n")
            c.close()  # gone while the server is still thinking
        deadline = time.monotonic() + 3
        while bridge.active < 5 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert bridge.active > 0  # the relays existed...
        deadline = time.monotonic() + 15
        while (bridge.active or open_fds() > base_fds + 1 or threading.active_count() > base_threads + 1) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert bridge.active == 0  # ...and ended once the upstream answered into a dead client
    server.release.set()
    deadline = time.monotonic() + 5
    while (open_fds() > base_fds or threading.active_count() > base_threads) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert (open_fds(), threading.active_count()) <= (base_fds, base_threads), "sockets or threads leaked"


@linux_only
def test_shutting_the_bridge_down_ends_a_relay_in_flight_and_leaks_nothing(tmp_path, slow_server):
    import socket

    server = slow_server(hold=60)  # the server never answers until released
    base_fds, base_threads = open_fds(), threading.active_count()
    bridge = pilot.ModelBridge(tmp_path, "127.0.0.1", server.server_address[1])
    bridge.__enter__()
    c = socket.socket(socket.AF_UNIX)
    c.settimeout(10)
    c.connect(str(bridge.path))
    c.sendall(b"POST /v1/x HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n")
    deadline = time.monotonic() + 3
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started and bridge.active == 1  # a generation is in flight
    start = time.monotonic()
    bridge.__exit__(None, None, None)
    assert c.recv(10) == b"" and time.monotonic() - start < 3  # the client was cut off at shutdown, not after 60 s
    c.close()
    bridge.thread.join(3)
    deadline = time.monotonic() + 3
    while bridge.active and time.monotonic() < deadline:  # the relay thread removes itself a moment after its sockets are cut
        time.sleep(0.05)
    assert not bridge.thread.is_alive() and bridge.active == 0 and not bridge.path.exists()
    server.release.set()
    deadline = time.monotonic() + 5
    while (open_fds() > base_fds or threading.active_count() > base_threads) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert (open_fds(), threading.active_count()) <= (base_fds, base_threads), "sockets or threads leaked"


# ---- B2: outcome classification ------------------------------------------------------------------------------------------------------------------
def agent_with(success=False, termination="error", category=None, workers=(), error=None, **failure):
    """An agent block as the child writes it: `model_failure` is effGen's own account of a failed run (None for an answered one)."""
    mf = None if success else {"stop_reason": "run_failed", "reason": "generation_failed", "category": category, "type": "X", "retryable": None, "message": "m", **failure}
    return {"timed_out": False, "exit_code": 0 if success else 1, "result": {"agent_reported_success": success, "termination": termination, "model_failure": mf, "workers": list(workers), "error": error}}


WORKER_ERROR = {"worker": "coder", "status": "failed", "error_type": "worker_run_failed", "termination": "error"}
WORKER_TIMEOUT = {"worker": "coder", "status": "timeout", "error_type": "WorkerTimeout", "termination": None}  # effGen's _timed_out leaves termination unset
WORKER_EXCEPTION = {"worker": "coder", "status": "failed", "error_type": "BackendUnreachableError", "termination": None}  # effGen's _failed leaves it unset
EMPTY, CHANGED = {"included_files": []}, {"included_files": ["mini/core.py"]}
UNRES = {"outcome": "unresolved_tests_failed", "problems": ["x"]}
NOEDIT = {"outcome": "unresolved_empty_patch"}
OK = {"outcome": "resolved"}
EVAL_ERR = {"outcome": "error", "problems": ["the container was OOM-killed"]}


def rec(agent, candidate=CHANGED, evaluation=UNRES, resolved=False, **more):
    return run_record(agent=agent, candidate=candidate, evaluation=evaluation, resolved=resolved, **more)


# (label, record, (attempt_outcome, infrastructure_failure, counts_as_coding_failure), agent state)
CLASSIFICATION_CASES = [
    # model-call transport / server / configuration errors: infrastructure, whatever the evaluator then says
    *[(f"model call failed: {c}", rec(agent_with(category=c), EMPTY, NOEDIT), ("infrastructure_failure", True, False), "infrastructure_failure")
      for c in ("unreachable", "timeout", "transient", "rate_limited", "auth", "not_found", "not_loaded", "resource_exhausted", "fatal")],
    # genuine model-side failures keep their category and DO count (the model under test failed to produce a usable answer)
    *[(f"genuine model failure: {c}", rec(agent_with(category=c), EMPTY, NOEDIT), ("unresolved", False, True), "agent_reported_failure") for c in ("truncation", "reasoning_only", "refusal")],
    # unknown cause: preserved as such, neither infrastructure nor coding failure
    *[(f"cause not established: {c!r}", rec(agent_with(category=c), EMPTY, NOEDIT), ("indeterminate", False, False), "indeterminate") for c in (None, "unknown", "invalid_request", "empty_response")],
    ("stuck is a genuine agent failure", rec(agent_with(termination="stuck")), ("unresolved", False, True), "agent_reported_failure"),
    ("tool_failed is a genuine agent failure", rec(agent_with(termination="tool_failed")), ("unresolved", False, True), "agent_reported_failure"),
    ("no termination recorded (legacy result) is a genuine agent failure", rec(agent_with(termination=None)), ("unresolved", False, True), "agent_reported_failure"),
    ("a worker ended in error and nothing resolved: ambiguous", rec(agent_with(success=True, termination="done", workers=[{"worker": "coder", "status": "failed", "termination": "error"}]), EMPTY, NOEDIT), ("indeterminate", False, False), "completed"),
    ("a worker ended in error but the task resolved: resolved", rec(agent_with(success=True, termination="done", workers=[{"worker": "coder", "status": "failed", "termination": "error"}]), CHANGED, OK, True), ("resolved", False, False), "completed"),
    ("a worker ended normally", rec(agent_with(success=True, termination="done", workers=[{"worker": "coder", "status": "completed", "termination": "done"}])), ("unresolved", False, True), "completed"),
    # C1: a worker that did not complete for a cause effGen's report does not state (timeout and exception leave termination None)
    ("S2b: worker timeout (status=timeout, termination=None), nothing resolved", rec(agent_with(success=True, termination="done", workers=[WORKER_TIMEOUT]), EMPTY, NOEDIT), ("indeterminate", False, False), "completed"),
    ("worker exception (status=failed, termination=None), nothing resolved", rec(agent_with(success=True, termination="done", workers=[WORKER_EXCEPTION]), EMPTY, NOEDIT), ("indeterminate", False, False), "completed"),
    ("worker timeout but the task resolved: resolved", rec(agent_with(success=True, termination="done", workers=[WORKER_TIMEOUT]), CHANGED, OK, True), ("resolved", False, False), "completed"),
    ("genuine worker failure: stuck", rec(agent_with(success=True, termination="done", workers=[{"worker": "coder", "status": "failed", "termination": "stuck"}])), ("unresolved", False, True), "completed"),
    ("genuine worker failure: tool_failed", rec(agent_with(success=True, termination="done", workers=[{"worker": "coder", "status": "failed", "termination": "tool_failed"}])), ("unresolved", False, True), "completed"),
    ("one genuine and one unexplained worker failure: ambiguous", rec(agent_with(success=True, termination="done", workers=[{"worker": "a", "status": "failed", "termination": "stuck"}, WORKER_TIMEOUT]), EMPTY, NOEDIT), ("indeterminate", False, False), "completed"),
    # C2: the worker check comes before the model-failure early returns
    *[(f"S3: worker transport failure, then the orchestrator's {c}", rec(agent_with(category=c, workers=[WORKER_ERROR]), EMPTY, NOEDIT), ("indeterminate", False, False), "indeterminate") for c in ("truncation", "reasoning_only", "refusal")],
    ("S3 but the task resolved: resolved", rec(agent_with(category="truncation", workers=[WORKER_ERROR]), CHANGED, OK, True), ("resolved", False, False), "indeterminate"),
    ("orchestrator truncation with a worker that ended normally stays a genuine failure", rec(agent_with(category="truncation", workers=[{"worker": "coder", "status": "completed", "termination": "done"}]), EMPTY, NOEDIT), ("unresolved", False, True), "agent_reported_failure"),
    ("orchestrator transport error beats an unexplained worker failure", rec(agent_with(category="unreachable", workers=[WORKER_TIMEOUT]), EMPTY, NOEDIT), ("infrastructure_failure", True, False), "infrastructure_failure"),
    ("orchestrator unknown cause plus an unexplained worker failure: both reasons kept", rec(agent_with(category="unknown", workers=[WORKER_ERROR]), EMPTY, NOEDIT), ("indeterminate", False, False), "indeterminate"),
    # agent success and evaluator correctness are separate
    ("agent says success, tests fail", rec(agent_with(success=True, termination="done")), ("unresolved", False, True), "completed"),
    ("agent says failure, tests pass", rec(agent_with(category="reasoning_only"), CHANGED, OK, True), ("resolved", False, False), "agent_reported_failure"),
    ("agent says success, tests pass", rec(agent_with(success=True, termination="done"), CHANGED, OK, True), ("resolved", False, False), "completed"),
    # precedence of simultaneous signals
    ("timeout beats a transport error in the same run", rec({"timed_out": True, "exit_code": -9, "result": None}, EMPTY, NOEDIT), ("timed_out", False, False), "timed_out"),
    ("transport error beats an empty candidate and a failing evaluator verdict", rec(agent_with(category="unreachable"), EMPTY, UNRES), ("infrastructure_failure", True, False), "infrastructure_failure"),
    ("evaluator infrastructure failure beats a genuine agent failure", rec(agent_with(category="truncation"), CHANGED, EVAL_ERR), ("infrastructure_failure", True, False), "agent_reported_failure"),
    ("evaluator infrastructure failure beats an indeterminate agent", rec(agent_with(category="unknown"), CHANGED, EVAL_ERR), ("infrastructure_failure", True, False), "indeterminate"),
    ("evaluator infrastructure failure beats an agent timeout", rec({"timed_out": True, "exit_code": -9, "result": None}, CHANGED, EVAL_ERR), ("infrastructure_failure", True, False), "timed_out"),
    ("a harness exception beats everything after it", rec(agent_with(success=True, termination="done", error="BackendUnreachableError: x"), CHANGED, UNRES), ("infrastructure_failure", True, False), "infrastructure_failure"),
    ("the agent timeout beats an indeterminate error", rec({**agent_with(category="unknown"), "timed_out": True}), ("timed_out", False, False), "timed_out"),
    ("indeterminate beats a plain unresolved verdict", rec(agent_with(category=None), CHANGED, UNRES), ("indeterminate", False, False), "indeterminate"),
]


@pytest.mark.parametrize("label,record,expected,agent_state", CLASSIFICATION_CASES, ids=[c[0] for c in CLASSIFICATION_CASES])
def test_every_classification_branch_and_the_precedence_of_simultaneous_signals(label, record, expected, agent_state):
    c = pilot.classify_run(record)
    assert (c["attempt_outcome"], c["infrastructure_failure"], c["counts_as_coding_failure"]) == expected, c
    assert c["agent"] == agent_state, c
    assert (c["infrastructure_reasons"] != []) == c["infrastructure_failure"]
    assert (c["indeterminate_reasons"] != []) == (c["agent"] == "indeterminate" or "worker run(s)" in " ".join(c["indeterminate_reasons"]))


def test_an_indeterminate_outcome_keeps_the_raw_fields_and_says_why():
    record = rec(agent_with(category="unknown", reason="generation_failed"), EMPTY, NOEDIT)
    c = pilot.classify_run(record)
    assert c["attempt_outcome"] == "indeterminate" and "cause is not established" in c["indeterminate_reasons"][0] and "'unknown'" in c["indeterminate_reasons"][0]
    assert record["agent"]["result"]["model_failure"]["category"] == "unknown"  # the evidence stays in the record; the classification only reads it


class _Resp:
    def __init__(self, success, stop_reason="run_failed", metadata=None):
        self.success, self.stop_reason, self.metadata = success, stop_reason, metadata


@pytest.mark.parametrize("response,expected", [
    (None, None),
    (_Resp(True), None),
    (_Resp(False, metadata={"reason": "generation_failed", "error": {"category": "unreachable", "type": "BackendUnreachableError", "retryable": True, "message": "x" * 500}}),
     {"stop_reason": "run_failed", "reason": "generation_failed", "category": "unreachable", "type": "BackendUnreachableError", "retryable": True, "message": "x" * 300}),
    (_Resp(False, stop_reason="max_iterations_exhausted", metadata={}), {"stop_reason": "max_iterations_exhausted", "reason": "None", "category": None, "type": None, "retryable": None, "message": ""}),
    (_Resp(False, metadata=None), {"stop_reason": "run_failed", "reason": "None", "category": None, "type": None, "retryable": None, "message": ""}),
])
def test_the_child_records_effgens_own_account_of_a_failed_run(response, expected):
    assert pilot.model_failure(SimpleNamespace(response=response)) == expected


class FlaggedRunner(FakeRunner):
    def __init__(self, stdout, exit_code=0, **flags):
        super().__init__(stdout, exit_code)
        self.flags = flags

    def run(self, *a, **kw):
        return {**super().run(*a, **kw), **self.flags}


@needs_patch
@pytest.mark.parametrize("label,runner,outcome,resolved", [
    ("tests pass", FlaggedRunner(pytest_output(2)), "resolved", True),
    ("tests fail", FlaggedRunner(pytest_output(1, failed=1), 1), "unresolved_tests_failed", False),
    ("OUTER container timeout (docker start killed: output discarded, no marker)", FlaggedRunner("", 124, timed_out=True), "error", False),
    ("INNER pytest timeout (`timeout` fired inside the script: marker printed, exit 124)", FlaggedRunner("python: x\npytest: y\npytest_exit=124 wall_ms=120000\n", 124), "unresolved_tests_failed", False),
    ("confirmed OOM kill, partial output", FlaggedRunner("python: x\n", 137, oom_killed=True), "error", False),
    ("confirmed OOM kill although the output looks like a pass", FlaggedRunner(pytest_output(2), 0, oom_killed=True), "error", False),
    ("no completion marker (engine or shell died)", FlaggedRunner("", 125), "error", False),
    ("no completion marker, truncated output", FlaggedRunner("python: Python 3.12\npytest: pytest 8.3.4\n", 137), "error", False),
])
def test_an_evaluation_without_a_completion_marker_or_with_an_oom_kill_is_an_evaluator_error_never_a_test_verdict(spec, tmp_path, label, runner, outcome, resolved):
    record = pilot.evaluate(spec, GOLD, tmp_path / "e", runner)
    assert (record["outcome"], record["resolved"]) == (outcome, resolved), record["problems"]
    assert record["result"] is not None and record["result"]["exit_code"] == runner.exit_code  # the raw result stays
    c = pilot.classify_run(run_record(evaluation=record, resolved=record["resolved"]))
    assert c["infrastructure_failure"] is (outcome == "error") and c["counts_as_coding_failure"] is (outcome == "unresolved_tests_failed")  # an inner pytest timeout (a candidate whose tests run too long) is its own failure; the outer one is not


# ---- B1 end to end: the real driver, the real sandbox, a stand-in endpoint whose first answer takes 12 s ----------------------------------------------
@needs_sandbox
def test_a_sandboxed_run_whose_first_model_call_takes_twelve_seconds_still_succeeds_and_that_call_is_not_retried(dataset, tmp_path, monkeypatch):
    # The first completion (the orchestrator rewriting mini/core.py through file_operations) arrives after 12 s; the second answers at once. A real change keeps the success path:
    # `agent_reported_success` needs a workspace change. effGen retries a call that fails, so a cut-off first request would show up as extra POSTs.
    slow_first = lambda worker, n: ("sleep", 12, _completion(WRITE_CALL)) if n == 1 else _completion(EDITED_ANSWER)  # noqa: E731
    before_bridge_dirs = set(Path(tempfile.gettempdir()).glob("cb-*"))
    servers = []
    record = run_scripted_endpoint(dataset, tmp_path, monkeypatch, slow_first, server_out=servers)
    result = record["agent"]["result"]
    posts = [p for m, p in servers[0].paths if m == "POST" and p.startswith("/v1/")]
    assert len(posts) == 2, posts  # the 12 s call exactly once plus the follow-up that carries its tool result (before the bridge fix: a connection error at 10 s and a dozen retries)
    assert record["agent"]["seconds"] >= 12 and record["agent"]["exit_code"] == 0 and not record["agent"]["timed_out"]
    assert (result["agent_reported_success"], result["workspace_changed"], result["termination"], result["error"], result["model_failure"]) == (True, True, "done", None, None)
    assert result["output"] == "add now adds"
    assert record["candidate"]["included_files"] == ["mini/core.py"]  # the edit made during the slow call reached the candidate
    assert record["classification"]["agent"] == "completed" and record["classification"]["infrastructure_failure"] is False
    assert set(Path(tempfile.gettempdir()).glob("cb-*")) <= before_bridge_dirs  # the driver removed its bridge directory


@needs_sandbox
def test_a_sandboxed_run_whose_endpoint_drops_every_request_is_an_infrastructure_failure_with_the_evidence_kept(dataset, tmp_path, monkeypatch):
    from http.server import ThreadingHTTPServer

    class Drop(_Stub):
        def do_POST(self):
            self.server.paths.append(("POST", self.path))
            self.close_connection = True  # no response at all: effGen sees a connection error

    server = ThreadingHTTPServer(("127.0.0.1", 0), Drop)
    server.daemon_threads, server.paths, server.tag = True, [], b"x"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(pilot, "fetch_model_identity", REAL_FETCH_MODEL_IDENTITY)
    try:
        record = pilot.run_pilot(cfg_for(dataset, tmp_path, base_url=f"http://127.0.0.1:{server.server_address[1]}/v1", agent_timeout_s=240, max_iterations=2), FakeRunner(pytest_output(2)))
    finally:
        server.shutdown()
        server.server_close()
    result, c = record["agent"]["result"], record["classification"]
    assert result["agent_reported_success"] is False and result["termination"] == "error" and result["model_failure"]["category"] in pilot.TRANSPORT_CATEGORIES, result
    assert (c["agent"], c["attempt_outcome"], c["infrastructure_failure"], c["counts_as_coding_failure"]) == ("infrastructure_failure", "infrastructure_failure", True, False), c
    assert "model call failed" in c["infrastructure_reasons"][0]


# ---- C1/C2 end to end (real driver, real sandbox, real effGen): the worker's failure record, then the classification ---------------------------------
DELEGATE_CALL = 'Thought: next.\nAction: delegate\nAction Input: {"worker": "coder", "task": "fix mini/core.py"}'
FINAL_ANSWER = "Thought: done.\nFinal Answer: the worker could not finish"
READ_CALL = 'Thought: read it.\nAction: file_operations\nAction Input: {"operation": "read", "path": "mini/core.py"}'


# The success path needs a REAL change: the child's `agent_reported_success` is `result.success and workspace_changed`, and it exits 1 when nothing changed. These scripted stand-ins
# (not models) make the coder edit mini/core.py through the real edit_file tool, or the orchestrator rewrite it through file_operations, inside the real sandbox.
EDIT_CALL = "Thought: fix it.\nAction: edit_file\nAction Input: " + json.dumps({"path": "mini/core.py", "old_text": "a - b", "new_text": "a + b"})
WRITE_CALL = "Thought: fix it.\nAction: file_operations\nAction Input: " + json.dumps({"operation": "write", "path": "mini/core.py", "content": CORE_NEW})
EDITED_ANSWER = "Thought: done.\nFinal Answer: add now adds"


def editing_script(worker, n):
    """The orchestrator delegates once; the coder REALLY edits mini/core.py through edit_file; both then answer."""
    if worker:
        return _completion(EDIT_CALL if n == 1 else EDITED_ANSWER)
    return _completion(DELEGATE_CALL if n == 1 else EDITED_ANSWER)


def _completion(content, finish="stop", reasoning=None):
    message = {"role": "assistant", "content": content, **({"reasoning": reasoning, "reasoning_content": reasoning} if reasoning else {})}
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "big:1", "usage": {"prompt_tokens": 1, "completion_tokens": 5, "total_tokens": 6},
            "choices": [{"index": 0, "finish_reason": finish, "message": message}]}


def run_scripted_endpoint(dataset, tmp_path, monkeypatch, script, *, server_out=None, **cfg):
    """`script(is_worker, n)` -> a completion body, 'drop' (close without answering) or ('sleep', seconds, body). The worker is told apart by its system prompt.
    `server_out`, when a list, receives the stand-in server (its `.paths` and `.log` stay readable after it is shut down)."""
    from http.server import ThreadingHTTPServer

    class Scripted(_Stub):
        def do_POST(self):
            self.server.paths.append(("POST", self.path))
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
            worker = pilot.WORKER_PROMPT[:40] in raw
            self.server.log.append("W" if worker else "O")
            action = self.server.script(worker, self.server.log.count("W" if worker else "O"))
            if action == "drop":
                self.close_connection = True
                return
            if isinstance(action, tuple):
                time.sleep(action[1])
                action = action[2]
            self._send(action)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Scripted)
    server.daemon_threads, server.paths, server.tag, server.log, server.script = True, [], b"x", [], script
    if server_out is not None:
        server_out.append(server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(pilot, "fetch_model_identity", REAL_FETCH_MODEL_IDENTITY)
    try:
        return pilot.run_pilot(cfg_for(dataset, tmp_path, base_url=f"http://127.0.0.1:{server.server_address[1]}/v1", agent_timeout_s=240, max_iterations=3, **cfg), FakeRunner(pytest_output(2)))
    finally:
        server.shutdown()
        server.server_close()


@needs_sandbox
def test_S2b_a_worker_that_hits_its_deadline_leaves_termination_unset_and_the_run_is_indeterminate(dataset, tmp_path, monkeypatch):
    slow_worker = lambda worker, n: ("sleep", 5, _completion(READ_CALL)) if worker else _completion(DELEGATE_CALL if n == 1 else FINAL_ANSWER)  # noqa: E731
    record = run_scripted_endpoint(dataset, tmp_path, monkeypatch, slow_worker, worker_timeout_s=3)
    worker = record["agent"]["result"]["workers"][0]
    assert (worker["status"], worker["error_type"], worker["termination"]) == ("timeout", "WorkerTimeout", None)  # the real effGen/driver record: the shape the old check missed
    result = record["agent"]["result"]
    # the orchestrator went on and answered ("done"), but nothing in the workspace changed, so the harness does not accept the answer as a success (the new verification) and the child exits 1
    assert result["termination"] == "done" and result["workspace_changed"] is False and result["agent_reported_success"] is False
    assert "no workspace change was detected" in result["output"] and record["agent"]["exit_code"] == 1
    c = record["classification"]
    assert (c["attempt_outcome"], c["counts_as_coding_failure"], c["infrastructure_failure"]) == ("indeterminate", False, False), c
    assert "worker run(s) ['coder']" in c["indeterminate_reasons"][0]
    assert json.loads((tmp_path / "out" / record["run_id"] / "run.json").read_text())["agent"]["result"]["workers"][0]["status"] == "timeout"  # the evidence is in the saved artifact


@needs_sandbox
def test_S3_a_worker_transport_failure_followed_by_orchestrator_truncation_is_indeterminate_with_the_category_kept(dataset, tmp_path, monkeypatch):
    script = lambda worker, n: "drop" if worker else (_completion(DELEGATE_CALL) if n == 1 else _completion("", "length", "thinking " * 200))  # noqa: E731
    record = run_scripted_endpoint(dataset, tmp_path, monkeypatch, script)
    result, c = record["agent"]["result"], record["classification"]
    assert result["workers"][0]["termination"] == "error" and result["workers"][0]["status"] == "failed"
    assert result["agent_reported_success"] is False and result["model_failure"]["category"] == "truncation"  # the orchestrator's own category is preserved
    assert (c["agent"], c["attempt_outcome"], c["counts_as_coding_failure"], c["infrastructure_failure"]) == ("indeterminate", "indeterminate", False, False), c
    assert "worker run(s) ['coder']" in c["indeterminate_reasons"][0]


# ---- the Docker agent sandbox: a minimal mount set that never reaches the dataset or the evaluator (no Docker is started here) ------------------------
GOLD_LINE = r"(?:\x1b[0-?])|"  # the reference fix's added line; it must never be given to the agent


def create_kwargs(**over) -> dict:
    """The name, run id and limits `docker_agent_command` now needs (the limits are the dict that is both passed to docker and recorded)."""
    return {"name": "smol-agent-t", "run_id": "t", "limits": pilot.docker_limits(pilot.SandboxLimits()), **over}


def docker_params(agent_dir: Path) -> dict:
    (agent_dir / "workspace").mkdir(parents=True, exist_ok=True)
    return {"config_path": str(agent_dir / "harness.yaml"), "prompt_path": str(agent_dir / "prompt.txt"), "out_dir": str(agent_dir), "workspace": str(agent_dir / "workspace"),
            "max_tokens": 1, "temperature": 0.1, "seed": 0, "bridge": {"port": 11434, "socket": "unused"}}


def docker_mounts(command: list[str]) -> list[tuple[str, str, str]]:
    out, i = [], 0
    while i < len(command):
        if command[i] == "-v":
            out.append(tuple(command[i + 1].rsplit(":", 2)))
            i += 2
        else:
            i += 1
    return out


def test_the_docker_agent_gets_exactly_the_mounts_it_needs_and_never_the_repo_root_or_the_dataset(tmp_path):
    run = tmp_path / "run"
    agent_dir, bridge_dir = run / "agent", tmp_path / "cb-x"
    bridge_dir.mkdir()
    protected = (tmp_path / "ds", run / "evaluator", run / "evaluator" / "pristine")
    command = pilot.docker_agent_command(agent_dir, bridge_dir, docker_params(agent_dir), protected=protected, **create_kwargs())
    mounts = docker_mounts(command)
    by_target = {dst: (src, mode) for src, dst, mode in mounts}
    assert len(by_target) == len(mounts)  # no target mounted twice
    expected = {"/agent", "/bridge", "/repo/src", "/repo/experiments/coding_eval/rich3278_pilot.py", "/venv", "/usr", "/lib"} | ({"/lib64"} if Path("/lib64").exists() else set())
    assert set(by_target) == expected
    assert sorted(t for t in by_target if t == "/repo" or t.startswith("/repo/")) == ["/repo/experiments/coding_eval/rich3278_pilot.py", "/repo/src"]  # no /repo mount, only these two
    assert "/python" not in by_target and "/repo" not in by_target
    assert {t for t, (_, mode) in by_target.items() if mode == "rw"} == {"/agent", "/bridge"}  # everything else is read-only
    assert by_target["/agent"][0] == str(agent_dir) and by_target["/bridge"][0] == str(bridge_dir)
    assert by_target["/repo/src"][0] == str(pilot.REPO / "src") and by_target["/venv"][0] == str(Path(sys.prefix).resolve())
    assert by_target["/repo/experiments/coding_eval/rich3278_pilot.py"][0] == str(Path(pilot.__file__).resolve())
    # no source may contain, equal or lie inside the repository's real dataset directory (compared as pure paths: the real dataset is never touched)
    real_dataset = Path(os.path.abspath(pilot.REPO / "dataset"))
    for src, dst, _ in mounts:
        resolved = Path(src).resolve()
        assert resolved != real_dataset and resolved not in real_dataset.parents and real_dataset not in resolved.parents, (src, dst)
    assert "PYTHONPATH=/repo/src" in command and command[-1].startswith("exec /venv/bin/python /repo/experiments/coding_eval/rich3278_pilot.py agent-child")
    assert json.loads((agent_dir / "params.docker.json").read_text())["workspace"] == "/agent/workspace"


def test_a_mount_source_that_equals_contains_or_lies_inside_a_protected_directory_is_refused(tmp_path):
    repo = tmp_path / "repo"
    dataset, evaluator = repo / "dataset", tmp_path / "run" / "evaluator"
    pristine = evaluator / "pristine"
    for d in (dataset / "sub", pristine / "rich", tmp_path / "other"):
        d.mkdir(parents=True)
    (tmp_path / "link").symlink_to(dataset)
    protected = (dataset, evaluator, pristine)
    for label, source in (("a repository root that holds the dataset", repo), ("a higher ancestor of the dataset", tmp_path), ("the dataset itself", dataset), ("a directory inside the dataset", dataset / "sub"),
                          ("a symlink to the dataset", tmp_path / "link"), ("the evaluator directory", evaluator), ("the pristine copy", pristine), ("a directory inside the pristine copy", pristine / "rich")):
        try:
            pilot.check_mount_sources([(str(source), "/x")], protected)
        except pilot.PilotError as exc:
            assert "protected directory" in str(exc), label
        else:
            pytest.fail(f"not refused: {label}")
    pilot.check_mount_sources([(str(tmp_path / "other"), "/o"), ("/usr", "/usr")], protected)  # unrelated sources pass
    pilot.check_mount_sources([(str(repo), "/repo")], ())  # (the caller must pass the protected set; docker_agent_command requires it)


def test_docker_agent_command_refuses_before_writing_anything_when_a_mount_conflicts(tmp_path):
    agent_dir, bridge_dir = tmp_path / "run" / "agent", tmp_path / "cb-x"
    agent_dir.mkdir(parents=True)
    bridge_dir.mkdir()
    params = docker_params(agent_dir)
    with pytest.raises(pilot.PilotError, match="protected directory"):
        pilot.docker_agent_command(agent_dir, bridge_dir, params, protected=(pilot.REPO / "src",), **create_kwargs())  # the src mount equals a protected directory
    with pytest.raises(pilot.PilotError, match="protected directory"):
        pilot.docker_agent_command(agent_dir, bridge_dir, params, protected=(agent_dir.parent,), **create_kwargs())  # the agent directory lies inside one
    assert not (agent_dir / "params.docker.json").exists()  # refused before the parameters file was written
    with pytest.raises(TypeError):
        pilot.docker_agent_command(agent_dir, bridge_dir, params)  # `protected` is required: the check cannot be forgotten


@needs_patch
def test_a_docker_run_passes_the_dataset_the_evaluator_and_the_pristine_copy_as_protected_directories(dataset, tmp_path, monkeypatch):
    seen = {}

    def spy(agent_dir, bridge_dir, params, timeout_s, **kw):
        seen.update(kw)
        return {"command": ["docker", "create"], "exit_code": 0, "timed_out": False, "seconds": 0.0,
                "container": {"name": kw["name"], "inspect_file": "x.json", "isolation": {}, "cleanup": {}}}

    monkeypatch.setattr(pilot, "run_agent_container", spy)
    cfg = cfg_for(dataset, tmp_path, agent_sandbox="docker")
    record = pilot.run_pilot(cfg, FakeRunner())
    run_dir = tmp_path / "out" / record["run_id"]
    assert [Path(p).resolve() for p in seen["protected"]] == [Path(cfg.dataset_dir).resolve(), (run_dir / "evaluator").resolve(), (run_dir / "evaluator" / "pristine").resolve()]
    assert seen["image"] == cfg.agent_image and seen["name"] == f"smol-agent-{record['run_id']}" and seen["limits"] == pilot.docker_limits(cfg.sandbox_limits)


def test_the_rendered_worker_and_orchestrator_prompts_do_not_contain_the_reference_fix_line():
    import yaml

    cfg = yaml.safe_load(pilot.render_config(Path("/agent/workspace"), {"provider": "ollama", "model": "m:1", "context_length": 8192, "base_url": "http://127.0.0.1:11434/v1",
                                                                        "worker_timeout_s": 300, "max_iterations": 8}))
    worker, orchestrator = cfg["workers"][0]["system_prompt"], cfg["orchestrator"]["system_prompt"]
    for name, text in (("worker", worker), ("orchestrator", orchestrator), ("WORKER_PROMPT", pilot.WORKER_PROMPT)):
        assert GOLD_LINE not in text, name
        assert "For this task, insert" not in text and "one-line alternative" not in text, name
    assert "\\u005c" in worker and "After a successful edit_file call" in worker and "with no extra Action or Observation text. After a successful" in worker  # the rest is unchanged


# ---- the Docker agent container: create -> inspect -> verify -> start, and ALWAYS killed and removed (a stubbed docker CLI: no container exists) --------
REAL_RUN = pilot.subprocess.run  # captured before any test patches subprocess.run; the stub below hands every non-docker command to it


def inspect_from_create(argv: list[str]) -> dict:
    """What a real `docker inspect` would show of the container this `docker create` command describes (the stub's idea of Docker's honest answer)."""
    def values(flag):
        return [argv[i + 1] for i, a in enumerate(argv) if a == flag]

    def size(text):
        return int(text) if text.isdigit() else int(text[:-1]) * {"k": 1024, "m": 1024**2, "g": 1024**3}[text[-1].lower()]

    ulimits = []
    for item in values("--ulimit"):
        name, _, rest = item.partition("=")
        soft, _, hard = rest.partition(":")
        ulimits.append({"Name": name, "Soft": int(soft), "Hard": int(hard or soft)})
    binds = values("-v")
    return {"HostConfig": {"NetworkMode": values("--network")[0], "IpcMode": values("--ipc")[0], "Privileged": False, "ReadonlyRootfs": "--read-only" in argv,
                           "CapDrop": values("--cap-drop"), "SecurityOpt": values("--security-opt"), "Init": "--init" in argv, "Memory": size(values("--memory")[0]),
                           "MemorySwap": size(values("--memory-swap")[0]), "PidsLimit": int(values("--pids-limit")[0]), "NanoCpus": round(float(values("--cpus")[0]) * 1e9),
                           "Ulimits": ulimits, "Tmpfs": dict(t.split(":", 1) for t in values("--tmpfs")), "Binds": binds, "Devices": None, "PortBindings": {}},
            "Mounts": [{"Type": "bind", "Source": b.split(":")[0], "Destination": b.split(":")[1], "Mode": b.split(":")[2], "RW": b.split(":")[2] == "rw"} for b in binds],
            "Config": {"User": values("--user")[0], "WorkingDir": values("-w")[0], "Entrypoint": [values("--entrypoint")[0]], "Cmd": ["-c", argv[-1]]}}


class AgentDocker:
    """A stand-in for the docker CLI, patched over subprocess.run (every non-docker command runs for real). Appends each docker verb to `events`.

    `edit(info)` changes what `docker inspect` reports; `kill` / `rm` / `create` are (returncode, stderr); `survives=True` makes `docker rm` succeed while the container stays."""

    def __init__(self, events, *, edit=None, kill=(0, ""), rm=(0, ""), create=(0, ""), survives=False):
        self.events, self.edit, self.kill, self.rm, self.create, self.survives = events, edit, kill, rm, create, survives
        self.argv, self.removed = None, False

    def __call__(self, args, **kw):
        if args[0] != "docker":
            return REAL_RUN(args, **kw)
        verb = args[1]
        self.events.append(verb)
        done = lambda code=0, out="", err="": pilot.subprocess.CompletedProcess(args, code, out, err)  # noqa: E731
        if verb == "create":
            if self.create[0] == 0:
                self.argv = list(args)
            return done(self.create[0], "cid\n" if self.create[0] == 0 else "", self.create[1])
        if verb == "inspect":
            if self.argv is None or self.removed:
                return done(1, "", f"Error: No such object: {args[-1]}")
            info = inspect_from_create(self.argv)
            if self.edit:
                self.edit(info)
            return done(0, json.dumps([info]))
        if verb == "kill":
            return done(self.kill[0], "", self.kill[1])
        if verb == "rm":
            if self.rm[0] == 0 and not self.survives:
                self.removed = True
            return done(self.rm[0], "", self.rm[1])
        raise AssertionError(f"an unexpected docker call: {args}")

    def flag(self, name):
        return [self.argv[i + 1] for i, a in enumerate(self.argv) if a == name]


def docker_pilot(dataset, tmp_path, monkeypatch, *, phase=None, cfg_over=None, **stub):
    """run_pilot with the Docker agent sandbox, a stubbed docker CLI and a stubbed `docker start` (run_agent_phase). Returns (events, docker, cfg).

    `events` is the order of everything that matters: docker verbs, "start" (the agent phase) and "collect" (candidate collection)."""
    events = []
    docker = AgentDocker(events, **stub)
    monkeypatch.setattr(pilot.subprocess, "run", docker)
    default = lambda cmd: {"command": cmd, "exit_code": 0, "timed_out": False, "seconds": 0.1}  # noqa: E731
    monkeypatch.setattr(pilot, "run_agent_phase", lambda agent_dir, params, timeout_s, cmd=None: (events.append("start"), (phase or default)(cmd))[1])
    real_collect = pilot.collect_patch
    monkeypatch.setattr(pilot, "collect_patch", lambda *a, **k: (events.append("collect"), real_collect(*a, **k))[1])
    return events, docker, cfg_for(dataset, tmp_path, agent_sandbox="docker", **(cfg_over or {}))


def test_the_docker_create_command_has_the_name_init_matching_memory_and_swap_tmpfs_and_workdir(tmp_path):
    agent_dir, bridge_dir = tmp_path / "run" / "agent", tmp_path / "cb-x"
    bridge_dir.mkdir()
    argv = pilot.docker_agent_command(agent_dir, bridge_dir, docker_params(agent_dir), protected=(tmp_path / "ds",), **create_kwargs(name="smol-agent-abc", run_id="abc"))
    assert argv[:2] == ["docker", "create"] and "run" not in argv[:3] and "--rm" not in argv  # created, not run: the driver reads it back, starts it and removes it itself

    def values(flag):
        return [argv[i + 1] for i, a in enumerate(argv) if a == flag]

    assert values("--name") == ["smol-agent-abc"] and "--init" in argv and "run=abc" in values("--label")
    assert values("--memory") == ["2g"] and values("--memory-swap") == ["2g"] and values("--memory-swap") != ["0"]  # swap limited to the memory limit: "0" would mean "unset"
    assert sorted(t.split(":")[0] for t in values("--tmpfs")) == ["/data", "/tmp"] and values("-w") == ["/agent"]  # the image's VOLUME /data is masked; the cwd is the agent directory
    assert values("--cpus") == ["2"] and values("--pids-limit") == ["256"] and "nofile=1024:1024" in values("--ulimit") and "core=0" in values("--ulimit")
    custom = pilot.docker_limits(pilot.SandboxLimits(memory_max="3G", tasks_max=100, cpu_quota="150%", fsize_bytes=12345, nofile=77))
    argv = pilot.docker_agent_command(agent_dir, bridge_dir, docker_params(agent_dir), protected=(tmp_path / "ds",), **create_kwargs(limits=custom))
    assert [argv[i + 1] for i, a in enumerate(argv) if a in ("--memory", "--memory-swap", "--cpus", "--pids-limit")] == ["3g", "3g", "1.5", "100"]


@pytest.mark.parametrize("how", ["normal", "timeout"])
def test_the_container_is_killed_then_removed_and_confirmed_gone_before_the_candidate_is_collected(dataset, tmp_path, monkeypatch, how):
    phase = None if how == "normal" else (lambda cmd: {"command": cmd, "exit_code": None, "timed_out": True, "seconds": 5.0})
    events, docker, cfg = docker_pilot(dataset, tmp_path, monkeypatch, phase=phase)
    record = pilot.run_pilot(cfg, FakeRunner())
    assert events == ["create", "inspect", "start", "kill", "rm", "inspect", "collect"]  # kill, then rm, then the confirmation, then (only then) the candidate
    assert record["status"] == "complete" and record["agent"]["timed_out"] is (how == "timeout")
    assert record["agent"]["container"]["name"] == f"smol-agent-{record['run_id']}" and record["agent"]["container"]["cleanup"]["gone"] is True


def test_the_container_is_killed_and_removed_when_the_agent_phase_raises_and_no_candidate_is_collected(dataset, tmp_path, monkeypatch):
    def boom(cmd):
        raise RuntimeError("boom")

    events, docker, cfg = docker_pilot(dataset, tmp_path, monkeypatch, phase=boom)
    with pytest.raises(RuntimeError, match="boom"):
        pilot.run_pilot(cfg, FakeRunner())
    assert events == ["create", "inspect", "start", "kill", "rm", "inspect"] and docker.removed  # the finally path ran; collection never did


@pytest.mark.parametrize("stub,message", [
    ({"kill": (1, "Error response from daemon: permission denied")}, "docker kill failed"),
    ({"kill": (127, "OSError: docker vanished")}, "docker kill failed"),
    ({"survives": True}, "not confirmed gone"),
    ({"rm": (1, "Error: removal of container x is already in progress"), "survives": True}, "not confirmed gone"),
])
def test_a_container_that_cannot_be_stopped_and_removed_blocks_candidate_collection_with_an_infrastructure_error(dataset, tmp_path, monkeypatch, stub, message):
    events, docker, cfg = docker_pilot(dataset, tmp_path, monkeypatch, **stub)
    record = pilot.run_pilot(cfg, FakeRunner())
    assert record["status"] == "refused" and message in record["error"] and "the candidate is not collected" in record["error"]
    assert "collect" not in events and record["candidate"] is None and record["evaluation"] is None and "rm" in events  # removal was still attempted after a failed kill
    c = record["classification"]
    assert (c["infrastructure_failure"], c["counts_as_coding_failure"], c["attempt_outcome"]) == (True, False, "infrastructure_failure")
    assert record["agent"]["exit_code"] == 0  # what is known about the agent phase is kept


@pytest.mark.parametrize("kill_error", ["Error response from daemon: cannot kill container: abc: Container 123 is not running", "Error: No such container: abc"])
def test_a_container_that_already_stopped_or_never_existed_is_not_a_cleanup_failure(dataset, tmp_path, monkeypatch, kill_error):
    events, docker, cfg = docker_pilot(dataset, tmp_path, monkeypatch, kill=(1, kill_error))
    record = pilot.run_pilot(cfg, FakeRunner())
    assert record["status"] == "complete" and events[-1] == "collect"


def test_a_missing_docker_binary_during_cleanup_is_a_cleanup_failure_not_a_pass(monkeypatch):
    def no_docker(args, **kw):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(pilot.subprocess, "run", no_docker)
    with pytest.raises(pilot.ContainerCleanupError, match="docker kill failed"):
        pilot.remove_agent_container("smol-agent-x")


def direct_container_run(tmp_path, monkeypatch, **stub):
    """pilot.run_agent_container with the stubbed docker; the agent phase must only run when the read-back accepted the container."""
    events = []
    docker = AgentDocker(events, **stub)
    monkeypatch.setattr(pilot.subprocess, "run", docker)
    monkeypatch.setattr(pilot, "run_agent_phase", lambda agent_dir, params, timeout_s, cmd=None: (events.append("start"), {"command": cmd, "exit_code": 0, "timed_out": False, "seconds": 0.1})[1])
    agent_dir, bridge_dir = tmp_path / "run" / "agent", tmp_path / "cb-x"
    bridge_dir.mkdir(exist_ok=True)
    inspect_path = tmp_path / "run" / "agent_container_inspect.json"
    kwargs = dict(name="smol-agent-t", run_id="t", protected=(tmp_path / "ds",), image="img:1", limits=pilot.docker_limits(pilot.SandboxLimits()), inspect_path=inspect_path)
    return events, docker, inspect_path, lambda: pilot.run_agent_container(agent_dir, bridge_dir, docker_params(agent_dir), 5, **kwargs)


def test_an_untouched_inspect_result_passes_the_read_back_and_the_container_is_started_and_recorded(tmp_path, monkeypatch):
    events, docker, inspect_path, run = direct_container_run(tmp_path, monkeypatch)
    result = run()
    assert events == ["create", "inspect", "start", "kill", "rm", "inspect"]
    assert result["container"]["isolation"]["NetworkMode"] == "none" and result["container"]["isolation"]["MemorySwap"] == result["container"]["isolation"]["Memory"]
    assert result["container"]["inspect_file"] == inspect_path.name and json.loads(inspect_path.read_text())["HostConfig"]["ReadonlyRootfs"] is True
    assert result["command"][:2] == ["docker", "create"] and result["start_command"] == ["docker", "start", "-a", "smol-agent-t"]


def _extra_mount(info):
    info["HostConfig"]["Binds"].append("/home/x:/h:rw")
    info["Mounts"].append({"Type": "bind", "Source": "/home/x", "Destination": "/h", "Mode": "rw", "RW": True})


@pytest.mark.parametrize("label,edit,field", [
    ("an extra bind mount", _extra_mount, "Binds"),
    ("an extra (image) volume", lambda i: i["Mounts"].append({"Type": "volume", "Destination": "/data", "RW": True}), "Mounts"),
    ("a read-only mount made writable", lambda i: i["HostConfig"].update(Binds=[b[:-3] + ":rw" if b.endswith(":ro") else b for b in i["HostConfig"]["Binds"]]), "Binds"),
    ("network not none", lambda i: i["HostConfig"].update(NetworkMode="bridge"), "NetworkMode"),
    ("ipc not none", lambda i: i["HostConfig"].update(IpcMode="host"), "IpcMode"),
    ("swap unset", lambda i: i["HostConfig"].update(MemorySwap=0), "MemorySwap"),
    ("swap unlimited", lambda i: i["HostConfig"].update(MemorySwap=-1), "MemorySwap"),
    ("memory differs", lambda i: i["HostConfig"].update(Memory=1 << 40), "Memory"),
    ("pids limit differs", lambda i: i["HostConfig"].update(PidsLimit=-1), "PidsLimit"),
    ("cpus differ", lambda i: i["HostConfig"].update(NanoCpus=0), "NanoCpus"),
    ("privileged", lambda i: i["HostConfig"].update(Privileged=True), "Privileged"),
    ("writable root filesystem", lambda i: i["HostConfig"].update(ReadonlyRootfs=False), "ReadonlyRootfs"),
    ("capabilities not dropped", lambda i: i["HostConfig"].update(CapDrop=[]), "CapDrop"),
    ("no-new-privileges missing", lambda i: i["HostConfig"].update(SecurityOpt=[]), "SecurityOpt"),
    ("no init", lambda i: i["HostConfig"].update(Init=False), "Init"),
    ("a device", lambda i: i["HostConfig"].update(Devices=[{"PathOnHost": "/dev/kvm"}]), "Devices"),
    ("a published port", lambda i: i["HostConfig"].update(PortBindings={"80/tcp": [{"HostPort": "80"}]}), "PortBindings"),
    ("no /data tmpfs", lambda i: i["HostConfig"]["Tmpfs"].pop("/data"), "Tmpfs"),
    ("running as root", lambda i: i["Config"].update(User="0:0"), "User"),
    ("another working directory", lambda i: i["Config"].update(WorkingDir="/data"), "WorkingDir"),
    ("another entrypoint", lambda i: i["Config"].update(Entrypoint=["sh"]), "Entrypoint"),
])
def test_an_inspect_read_back_that_differs_from_the_configuration_refuses_to_start_and_still_cleans_up(tmp_path, monkeypatch, label, edit, field):
    events, docker, inspect_path, run = direct_container_run(tmp_path, monkeypatch, edit=edit)
    with pytest.raises(pilot.PilotError, match="refusing to start") as caught:
        run()
    assert field in str(caught.value), (label, str(caught.value))
    assert "start" not in events and events[:2] == ["create", "inspect"] and events[2:] == ["kill", "rm", "inspect"] and docker.removed  # never started; the created container was removed
    assert inspect_path.exists()  # the evidence is kept even for a refusal


def test_a_failed_docker_create_never_starts_anything(tmp_path, monkeypatch):
    events, docker, inspect_path, run = direct_container_run(tmp_path, monkeypatch, create=(125, "Error: image not found locally"), kill=(1, "Error: No such container: smol-agent-t"))
    with pytest.raises(pilot.PilotError, match="docker create failed"):
        run()
    assert events == ["create", "kill", "rm", "inspect"] and not inspect_path.exists()  # nothing to read back or start; the cleanup still runs and confirms there is nothing left


def test_run_json_limits_equal_the_values_passed_to_docker_and_the_inspect_evidence_is_recorded(dataset, tmp_path, monkeypatch):
    custom = pilot.SandboxLimits(memory_max="3G", tasks_max=100, cpu_quota="150%", fsize_bytes=12345, nofile=77)
    events, docker, cfg = docker_pilot(dataset, tmp_path, monkeypatch, cfg_over={"sandbox_limits": custom})
    record = pilot.run_pilot(cfg, FakeRunner())
    run_dir = tmp_path / "out" / record["run_id"]
    saved = json.loads((run_dir / "run.json").read_text())["sandbox"]["limits"]
    passed = {"memory": docker.flag("--memory")[0], "memory_swap": docker.flag("--memory-swap")[0], "cpus": docker.flag("--cpus")[0], "pids_limit": int(docker.flag("--pids-limit")[0]),
              "nofile": int(next(u for u in docker.flag("--ulimit") if u.startswith("nofile=")).split("=")[1].split(":")[0]),
              "fsize_bytes": int(next(u for u in docker.flag("--ulimit") if u.startswith("fsize=")).split("=")[1])}
    assert passed == {"memory": "3g", "memory_swap": "3g", "cpus": "1.5", "pids_limit": 100, "nofile": 77, "fsize_bytes": 12345}  # what the command carries
    assert {k: saved[k] for k in passed} == passed == {k: record["sandbox"]["limits"][k] for k in passed}  # and what run.json says: the same values, not a second copy of the config
    assert record["sandbox"]["isolation"]["Memory"] == 3 * 1024**3 == record["sandbox"]["isolation"]["MemorySwap"]
    evidence = json.loads((run_dir / "agent_container_inspect.json").read_text())
    assert evidence["HostConfig"]["NetworkMode"] == "none" and not (run_dir / "agent" / "agent_container_inspect.json").exists()  # kept beside run.json, not where the agent can read it


# ---- --tool-calling-mode: chosen on the command line, passed to the agent child, recorded in run.json (default react: unchanged) --------------------
def test_the_driver_accepts_exactly_the_effgen_tool_calling_modes_and_defaults_to_react():
    import smol_harness.harness as harness_mod

    assert pilot.TOOL_CALLING_MODES == harness_mod.TOOL_CALLING_MODES  # the driver's list cannot drift from the harness's
    base = ["run", "--out", "/tmp/x", "--model", "m:1"]
    assert pilot.build_parser().parse_args(base).tool_calling_mode == "react"
    for mode in pilot.TOOL_CALLING_MODES:
        assert pilot.build_parser().parse_args([*base, "--tool-calling-mode", mode]).tool_calling_mode == mode
    with pytest.raises(SystemExit):
        pilot.build_parser().parse_args([*base, "--tool-calling-mode", "telepathy"])
    assert pilot.RunConfig(out_dir=Path("/tmp/x")).tool_calling_mode == "react"


def test_main_passes_the_chosen_mode_into_the_run_config(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_run_pilot(cfg, runner, **kw):
        seen["cfg"] = cfg
        return {"run_id": "r", "status": "complete", "resolved": False}

    monkeypatch.setattr(pilot, "run_pilot", fake_run_pilot)
    base = ["run", "--out", str(tmp_path / "out"), "--dataset-dir", str(tmp_path / "ds"), "--model", "m:1", "--allow-model-call"]  # a temporary dataset path: the real one is never touched
    assert pilot.main(base) == 0
    assert seen["cfg"].tool_calling_mode == "react"
    assert pilot.main([*base, "--tool-calling-mode", "hybrid"]) == 0
    assert seen["cfg"].tool_calling_mode == "hybrid"
    capsys.readouterr()


@needs_patch
@pytest.mark.parametrize("mode", [None, "react", "hybrid"])
def test_run_json_params_and_the_child_result_record_the_tool_calling_mode(dataset, tmp_path, mode):
    over = {} if mode is None else {"tool_calling_mode": mode}
    record = pilot.run_pilot(cfg_for(dataset, tmp_path, **over), FakeRunner(), agent_cmd=fake_agent())
    expected = mode or "react"  # not given: the default, exactly what the harness used before the option existed
    run_dir = tmp_path / "out" / record["run_id"]
    saved = json.loads((run_dir / "run.json").read_text())
    assert record["config"]["tool_calling_mode"] == expected and saved["config"]["tool_calling_mode"] == expected  # next to the other run settings
    assert record["provenance"]["tool_calling_mode"] == expected and saved["provenance"]["tool_calling_mode"] == expected  # and in the provenance
    assert json.loads((run_dir / "agent" / "params.json").read_text())["tool_calling_mode"] == expected  # the child receives it through the same params file as every other setting


@pytest.mark.parametrize("params_mode,expected", [(None, "react"), ("hybrid", "hybrid"), ("auto", "auto")])
def test_the_agent_child_builds_the_harness_with_the_requested_mode_and_records_it(tmp_path, monkeypatch, params_mode, expected):
    import smol_harness.harness as harness_mod

    seen = {}

    def stop_here(config, **kw):  # records what the child asked the builder for, then ends the run deterministically
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(harness_mod, "build_harness_from_config", stop_here)
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "harness.yaml").write_text(pilot.render_config(agent_dir / "workspace", {"provider": "ollama", "model": BIG, "context_length": 8192, "base_url": "http://127.0.0.1:11434/v1",
                                                                                           "worker_timeout_s": 60, "max_iterations": 6}))
    (agent_dir / "prompt.txt").write_text("task")
    (agent_dir / "workspace").mkdir()
    params = {"config_path": str(agent_dir / "harness.yaml"), "prompt_path": str(agent_dir / "prompt.txt"), "out_dir": str(agent_dir), "max_tokens": 1, "temperature": 0.1, "seed": 0,
              "workspace": str(agent_dir / "workspace")}
    if params_mode is not None:
        params["tool_calling_mode"] = params_mode
    assert pilot.run_agent(params) == 3  # the stand-in builder raised: recorded, not raised past the driver
    assert seen["tool_calling_mode"] == expected  # reached the harness builder (the default is react: the value the builder itself defaults to)
    result = json.loads((agent_dir / "agent_result.json").read_text())
    assert result["tool_calling_mode"] == expected and result["error"] == "RuntimeError: stop"


def test_an_unknown_tool_calling_mode_is_refused_before_anything_starts(dataset, tmp_path):
    record = pilot.run_pilot(cfg_for(dataset, tmp_path, tool_calling_mode="telepathy"), FakeRunner(), agent_cmd=fake_agent())
    assert record["status"] == "refused" and "unknown tool calling mode" in record["error"] and record["agent"] is None
    assert record["classification"]["infrastructure_failure"] is True and record["classification"]["counts_as_coding_failure"] is False


# ---- the real harness through its public API, with a scripted model --------------------------------------------------------------
@needs_patch
def test_the_agent_phase_runs_the_real_harness_with_a_scripted_model_and_records_everything(dataset, spec, tmp_path, monkeypatch):
    fake = FakeModels()
    fake.scripts[BIG] = [delegate("coder", "fix add in mini/core.py"), call("edit_file", path="mini/core.py", old_text="a - b", new_text="a + b"),
                         "Thought: edited.\nFinal Answer: add now adds", FINAL]
    monkeypatch.setattr(models_mod, "build_model", fake.build)
    agent_dir, ws = tmp_path / "agent", tmp_path / "agent" / "workspace"
    root, sha = dataset
    pilot.extract_snapshot(root / "snapshots" / "mini_1.tgz", ws, sha, include_git=False)
    (agent_dir / "prompt.txt").write_text(pilot.build_prompt(spec))
    (agent_dir / "harness.yaml").write_text(pilot.render_config(ws, {"provider": "ollama", "model": BIG, "context_length": 8192, "base_url": "http://127.0.0.1:11434/v1",
                                                                      "worker_timeout_s": 60, "max_iterations": 6}))
    params = {"config_path": str(agent_dir / "harness.yaml"), "prompt_path": str(agent_dir / "prompt.txt"), "out_dir": str(agent_dir),
              "max_tokens": 321, "temperature": 0.2, "seed": 7, "workspace": str(ws)}
    assert pilot.run_agent(params) == 0
    assert (ws / "mini/core.py").read_text() == CORE_NEW  # the worker edited the agent workspace through edit_file
    assert any("edit_file" in str(c.prompt) and "exactly once" in str(c.prompt) and "\\u005c" in str(c.prompt)
               for c in fake.instances[BIG].calls)  # and was offered it, with its contract
    result = json.loads((agent_dir / "agent_result.json").read_text())
    assert result["agent_reported_success"] is True and result["workers"][0]["worker"] == "coder" and result["workers"][0]["status"] == "completed"
    assert result["workspace_changed"] is True  # the verification the success flag now rests on
    assert result["generation_overrides_requested"] == {"max_tokens": 321, "temperature": 0.2, "seed": 7} and "U17" in result["note"]
    assert (Path(result["history_dir"]) / "trace.jsonl").exists()
    text, meta = pilot.collect_patch(_pristine(dataset, tmp_path), ws)
    assert meta["included_files"] == ["mini/core.py"] and "a + b" in text
    calls = fake.instances[BIG].calls
    assert "edit_file" not in str(calls[0].prompt)  # calls[0] is the orchestrator's first turn: it is not told to use a tool it does not hold
    assert len(calls) == 4 and any(STATEMENT in str(c.prompt) for c in calls)  # the task statement reached the model...
    assert all(GOLD not in str(c.prompt) and TEST_PATCH not in str(c.prompt) and DECOY not in str(c.prompt) for c in calls)  # ...and nothing evaluator-only did
    assert {c.config.max_tokens for c in calls} == {321} and {c.config.seed for c in calls} == {7}  # the requested generation settings really reached every call


@needs_patch
def test_an_agent_that_answers_without_changing_the_workspace_is_not_a_success_and_the_child_exits_1(dataset, spec, tmp_path, monkeypatch):
    # The harness verifies the claim: `agent_reported_success = result.success and workspace_changed`. Here effGen itself sees a normal answer (termination "done"), but the
    # coder never calls edit_file and the workspace is byte-identical, so the child records a failure, says why in the output and exits 1.
    fake = FakeModels()
    fake.scripts[BIG] = [delegate("coder", "fix add in mini/core.py"), "Thought: looked.\nFinal Answer: I changed add to add", FINAL]
    monkeypatch.setattr(models_mod, "build_model", fake.build)
    agent_dir, ws = tmp_path / "agent", tmp_path / "agent" / "workspace"
    root, sha = dataset
    pilot.extract_snapshot(root / "snapshots" / "mini_1.tgz", ws, sha, include_git=False)
    (agent_dir / "prompt.txt").write_text(pilot.build_prompt(spec))
    (agent_dir / "harness.yaml").write_text(pilot.render_config(ws, {"provider": "ollama", "model": BIG, "context_length": 8192, "base_url": "http://127.0.0.1:11434/v1",
                                                                      "worker_timeout_s": 60, "max_iterations": 6}))
    params = {"config_path": str(agent_dir / "harness.yaml"), "prompt_path": str(agent_dir / "prompt.txt"), "out_dir": str(agent_dir),
              "max_tokens": 321, "temperature": 0.2, "seed": 7, "workspace": str(ws)}
    assert pilot.run_agent(params) == 1  # not 0: the run did not do what it claims
    result = json.loads((agent_dir / "agent_result.json").read_text())
    assert result["error"] is None and result["termination"] == "done"  # effGen saw an ordinary answer...
    assert result["workspace_changed"] is False and result["agent_reported_success"] is False  # ...the harness did not accept it
    assert result["output"].startswith("all done") and result["output"].endswith("Harness verification: no workspace change was detected; the edit is unresolved.")
    assert result["workers"][0]["status"] == "completed" and result["workers"][0]["tool_calls"] == 0  # the coder really made no tool call
    assert (ws / "mini/core.py").read_text() == CORE_OLD
    _, meta = pilot.collect_patch(_pristine(dataset, tmp_path), ws)
    assert meta["included_files"] == []


def _pristine(dataset, tmp_path) -> Path:
    root, sha = dataset
    dest = tmp_path / "pristine"
    pilot.extract_snapshot(root / "snapshots" / "mini_1.tgz", dest, sha, include_git=False)
    return dest


def test_a_harness_failure_in_the_child_is_recorded_not_raised(tmp_path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "prompt.txt").write_text("task")
    params = {"config_path": str(tmp_path / "missing.yaml"), "prompt_path": str(agent_dir / "prompt.txt"), "out_dir": str(agent_dir), "max_tokens": 1, "temperature": 0.1, "seed": 0}
    assert pilot.run_agent(params) == 3
    assert json.loads((agent_dir / "agent_result.json").read_text())["error"]


# ---- opt-in: the real container against the real dataset (no model) -------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("SMOL_RUN_DOCKER_TESTS") != "1" or not (ROOT / "dataset" / "tasks.jsonl").exists() or shutil.which("docker") is None,
                    reason="opt-in: SMOL_RUN_DOCKER_TESTS=1 with the dataset present and Docker available")
def test_reference_check_on_the_real_container(tmp_path):
    out = tmp_path / "ref"
    assert pilot.main(["prepare-env", "--out", str(out)]) == 0
    assert pilot.main(["reference-check", "--out", str(out)]) == 0
    records = {json.loads(p.read_text())["label"]: json.loads(p.read_text()) for p in out.glob("*.evaluation.json")}
    assert (records["baseline"]["result"]["failed"], records["baseline"]["result"]["passed"]) == (16, 7) and records["baseline"]["resolved"] is False
    assert records["gold"]["result"]["passed"] == 23 and records["gold"]["resolved"] is True
