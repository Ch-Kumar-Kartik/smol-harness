"""`dgx_diag.py e3-repeat`: repeated, independent E3 trials with a fresh random transfer value each. CPU only, no network, no real model, no live inference.

The scripted model below does not know any value in advance. It answers from what is in the prompt it receives (the reader reads `source.txt`, the orchestrator copies the value it finds
in the reader's report, the writer writes the value it was given), so the real harness, tools, tracing, CLI path and verification run on values the runner generated. Behaviours that
go wrong ("modes") are scripted per trial. `model_identity` (the only HTTP call that matters) is replaced; nothing here can reach Ollama.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import ModelCall, ScriptedModel
from test_dgx_diag import CTX, TAG, act, args, e3_script, final, host, scripted, write_e0, write_e2  # noqa: F401  (host is a fixture)

import smol_harness.models as models_mod

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "dgx"))
import dgx_diag as dgx  # noqa: E402

drv, sr = dgx.drv, dgx.sr
TOKEN = re.compile(r"TRANSFER-[A-HJKMNP-Z2-9]{8}")
DIGEST = "sha256:" + "ab" * 32
IDENTITY = {"tag": TAG, "digest": DIGEST, "modified_at": "2026-10-28T00:00:00Z", "size_bytes": 1, "details": {"family": "scripted", "parameter_size": "14B"},
            "capabilities": ["completion"], "parameters": f"num_ctx {CTX}", "modelfile_from": "scripted-base"}


class FollowsTheValue(ScriptedModel):
    """Answers from the prompt it is given. `mode` (fixed per trial, see Plan) says how this run misbehaves."""

    def __init__(self, mode: str):
        super().__init__(["unused"], name="scripted")
        self.mode = mode

    def _next(self, prompt, config, kwargs):
        text = self.respond(str(prompt))
        self.calls.append(ModelCall(prompt, config, dict(kwargs), text))
        return text

    def generate(self, prompt, config=None, **kw):
        if self.mode == "interrupt":
            raise KeyboardInterrupt("operator pressed Ctrl-C")
        return super().generate(prompt, config, **kw)

    def respond(self, prompt: str) -> str:
        mode, token = self.mode, TOKEN.search(prompt)
        if "You are the worker agent 'reader'" in prompt:
            if mode == "reader_skips_read":
                return final("The transfer value is TRANSFER-ZZZZZZZZ.")
            if token is None:
                return act("file_operations", {"operation": "read", "path": "source.txt"})
            return final(f"The transfer value is {token.group()}.")
        if "You are the worker agent 'writer'" in prompt:
            if "wrote result.txt" in prompt:
                return final("I created result.txt.")
            if mode in ("bypass", "bypass_with_value") and "The transfer value is" not in prompt:
                return act("file_operations", {"operation": "read", "path": "source.txt"})
            value = token.group() if token else "UNKNOWN"
            return act("file_operations", {"operation": "write", "path": "result.txt", "content": (value[::-1] if mode == "wrong_file" else value) + "\n"})
        delegated = len(re.findall(r'"delegation_id": "d\d+"', prompt))  # the orchestrator: how many reports has it received
        if mode == "no_delegation":
            return final("Done.")
        if delegated == 0:
            return act("delegate", {"worker": "reader", "task": "Read the file source.txt and report the transfer value it contains."})
        if delegated == 1:
            value = token.group() if token else "UNKNOWN"
            task = {"alter": f"Create the file result.txt containing exactly {value}9 on one line.",
                    "no_value": "Create result.txt containing the transfer value from source.txt.", "bypass": "Create result.txt containing the transfer value from source.txt."}.get(
                mode, f"Create the file result.txt containing exactly {value} on one line.")
            return act("delegate", {"worker": "writer", "task": task})
        return final("Both steps are done.")


class Plan:
    """One model per CLI run (each trial builds its own), whose behaviour is the trial's mode."""

    def __init__(self, modes):
        self.modes, self.models = list(modes), []

    def next_model(self):
        model = FollowsTheValue(self.modes[len(self.models)] if len(self.models) < len(self.modes) else "ok")
        model.unload = lambda: None
        self.models.append(model)
        return model


def repeat_args(out: Path, trials: int = 3, **over):
    return SimpleNamespace(**{"out": str(out), "tag": TAG, "num_ctx": CTX, "trials": trials, "seed_start": 0, "value": None, "capture_prompts": False, **over})


@pytest.fixture
def ready(host, monkeypatch):
    """E0 and E2 evidence for the tag, a known model identity, fresh trace state (from `host`)."""
    write_e0(host)
    write_e2(host)
    monkeypatch.setattr(dgx, "model_identity", lambda tag: dict(IDENTITY, tag=tag))
    return host


def run_suite(monkeypatch, out: Path, modes, **over):
    plan = Plan(modes)
    monkeypatch.setattr(models_mod, "build_model", lambda spec: plan.next_model())
    before = set((out / "diag_E3R").glob("e3r-*")) if (out / "diag_E3R").exists() else set()
    code = dgx.cmd_e3_repeat(repeat_args(out, over.pop("trials", len(modes)), **over))
    created = sorted(set((out / "diag_E3R").glob("e3r-*")) - before) if (out / "diag_E3R").exists() else []  # the suite THIS call created (names within one second sort by a random suffix)
    return code, (created[-1] if created else None), plan


def jload(path: Path):
    return json.loads(path.read_text())


def trials_of(suite: Path):
    return sorted(suite.glob("trial-*"))


def value_of(trial: Path) -> str:
    return jload(trial / "trial.json")["record"]["expected_value"]


# ---- unique ids, directories, values and isolation ---------------------------------------------------------------------------------------------------
def test_every_trial_has_its_own_id_directory_trace_workspace_and_fresh_value(ready, monkeypatch):
    code, suite, plan = run_suite(monkeypatch, ready, ["ok", "ok", "ok"])
    assert code == 0
    trials = trials_of(suite)
    assert [t.name for t in trials] == ["trial-01", "trial-02", "trial-03"]
    ids = [jload(t / "trial.json")["trial_id"] for t in trials]
    assert len(set(ids)) == 3 and all(i.startswith(suite.name + "-t") for i in ids)
    values = [value_of(t) for t in trials]
    assert len(set(values)) == 3 and all(dgx.VALUE_RE.match(v) for v in values)
    traces = [[json.loads(line) for line in (t / "trace.jsonl").read_text().splitlines()] for t in trials]
    assert len({lines[0]["invocation"] for lines in traces}) == 3 and [lines[0]["run_id"] for lines in traces] == ids  # one invocation and one run id per trial
    for i, trial in enumerate(trials):
        text = (trial / "trace.jsonl").read_text()
        assert values[i] in text and not [v for j, v in enumerate(values) if j != i and v in text]  # nothing of another trial reached this trace
        kept = [json.loads(line) for line in (trial / "attempts.jsonl").read_text().splitlines()]
        assert [k["attempt_id"] for k in kept] == [ids[i]]  # one attempt per trial file
        assert (trial / "cli.json").exists() and list((trial / "configs_rendered").glob("*.yaml")) and not (trial / "work").exists()  # the disposable workspace is gone
    hashes = [jload(t / "trial.json")["record"]["file_hashes"] for t in trials]
    assert len({h["result_after"] for h in hashes}) == 3 and len({h["source_before"] for h in hashes}) == 3 and all(h["source_before"] == h["source_after"] for h in hashes)
    for i, model in enumerate(plan.models):  # a model prompt never holds a value from a different trial
        seen = " ".join(str(c.prompt) for c in model.calls)
        assert values[i] in seen and not [v for j, v in enumerate(values) if j != i and v in seen]


def test_values_come_from_the_secrets_module_and_never_repeat(monkeypatch):
    drawn = []
    real_choice = dgx.secrets.choice
    monkeypatch.setattr(dgx.secrets, "choice", lambda alphabet: drawn.append(1) or real_choice(alphabet))
    values = [dgx.new_transfer_value() for _ in range(300)]
    assert len(set(values)) == 300 and all(dgx.VALUE_RE.match(v) for v in values) and len(drawn) == 300 * 8
    assert all(dgx.has_token(f"The transfer value is {v}.", v) and not dgx.has_token(f"{v}9", v) and not dgx.has_token(f"X{v}", v) for v in values[:20])  # compatible with the token-boundary matcher
    sequence = iter("AAAAAAAA" + "BBBBBBBB")
    monkeypatch.setattr(dgx.secrets, "choice", lambda alphabet: next(sequence))
    assert dgx.new_transfer_value(used={"TRANSFER-AAAAAAAA"}) == "TRANSFER-BBBBBBBB"  # a value already used in the suite is skipped


def test_the_value_is_not_given_to_anyone_before_the_reader_retrieves_it(ready, monkeypatch):
    code, suite, plan = run_suite(monkeypatch, ready, ["ok", "ok"])
    assert not TOKEN.search(dgx.E3_TASK) and "TRANSFER" not in dgx.E3_TASK.upper().replace("TRANSFER VALUE", "")
    for i, trial in enumerate(trials_of(suite)):
        value = value_of(trial)
        for name in ("suite.json", "summary.json", "report.md"):
            assert value not in (suite / name).read_text()  # aggregates and metadata never carry the clear value
        assert value not in "".join(p.read_text() for p in (trial / "configs_rendered").glob("*.yaml"))
        first_prompt = str(plan.models[i].calls[0].prompt)
        assert value not in first_prompt  # the orchestrator's first prompt
        lines = (trial / "trace.jsonl").read_text().splitlines()
        reader_read = next(n for n, line in enumerate(lines) if '"tool_call_start"' in line and '"source.txt"' in line)
        first_with_value = next(n for n, line in enumerate(lines) if value in line)
        assert first_with_value > reader_read  # nothing in the evidence holds the value before the reader's own read was issued
        summary_row = jload(suite / "summary.json")["per_trial"][i]
        assert summary_row["value_sha256"] == hashlib.sha256(value.encode()).hexdigest()[:12]


# ---- verification: the existing E3 semantics, applied to every trial --------------------------------------------------------------------------------
@pytest.mark.parametrize("mode, exact, task, failure, mechanics", [
    ("ok", True, True, "none", True),
    ("alter", False, True, "value_altered", True),
    ("no_value", False, False, "value_not_transferred", True),
    ("bypass_with_value", False, True, "writer_bypass", True),
    ("wrong_file", True, False, "wrong_result_file", True),
    ("reader_skips_read", False, False, "reader_failed", True),
    ("no_delegation", False, False, "no_delegation", False),
])
def test_each_trial_is_judged_by_the_unchanged_e3_checks_and_classified(ready, monkeypatch, mode, exact, task, failure, mechanics):
    code, suite, plan = run_suite(monkeypatch, ready, [mode])
    row = jload(suite / "summary.json")["per_trial"][0]
    assert (row["status"], row["context_transfer_exact"], row["task_verified"], row["failure_class"], row["mechanics_ok"]) == ("completed", exact, task, failure, mechanics)
    assert (code == 0) is mechanics
    record = jload(next(iter(trials_of(suite))) / "trial.json")["record"]
    assert record["success"] is (exact and task and mechanics)  # success needs all three, never one of them


def test_exact_transfer_and_task_success_stay_separate_in_the_aggregate(ready, monkeypatch):
    code, suite, _ = run_suite(monkeypatch, ready, ["ok", "ok", "alter", "wrong_file", "no_delegation"])
    s = jload(suite / "summary.json")
    assert code == 1 and s["complete"] is True and s["interrupted"] is False
    assert s["trials"] == {"requested": 5, "attempted": 5, "completed": 5, "incomplete": 0, "blocked": 0, "not_run": 0}
    assert (s["mechanics_demonstrated"]["count"], s["context_transfer_exact"]["count"], s["task_verified"]["count"], s["all_three"]["count"]) == (4, 3, 3, 2)
    assert s["context_transfer_exact"]["of_requested"] == "3/5" and s["task_verified"]["rate_of_requested"] == 0.6
    assert s["failure_classes"] == {"no_delegation": 1, "value_altered": 1, "wrong_result_file": 1}
    assert s["latency"]["trial_seconds"]["n"] == 5 and s["latency"]["generation_seconds"]["n"] > 5 and "median" in s["latency"]["trial_seconds"]
    report = (suite / "report.md").read_text()
    assert "not a measurement" in report and "Do not quote these rates" in report and "3/5 of requested" in report


# ---- failure, interruption and partial results ------------------------------------------------------------------------------------------------------------
def test_an_interrupted_suite_keeps_its_finished_trials_and_counts_nothing_unfinished_as_success(ready, monkeypatch):
    released = []
    monkeypatch.setattr(drv, "release_ollama", lambda profile, label, *a, **k: released.append(label) or {"verified": True})
    plan = Plan(["ok", "interrupt", "ok", "ok"])
    monkeypatch.setattr(models_mod, "build_model", lambda spec: plan.next_model())
    with pytest.raises(KeyboardInterrupt):
        dgx.cmd_e3_repeat(repeat_args(ready, 4))
    suite = next((ready / "diag_E3R").glob("e3r-*"))
    s = jload(suite / "summary.json")
    assert (s["complete"], s["interrupted"]) == (False, True)
    assert [r["status"] for r in s["per_trial"]] == ["completed", "interrupted", "not_run", "not_run"]
    assert s["trials"] == {"requested": 4, "attempted": 2, "completed": 1, "incomplete": 1, "blocked": 0, "not_run": 2}
    assert (s["task_verified"]["count"], s["context_transfer_exact"]["of_requested"], s["all_three"]["count"]) == (1, "1/4", 1)  # the denominator is what was requested
    assert s["failure_classes"] == {"interrupted": 1, "not_run": 2}
    assert [t.name for t in trials_of(suite)] == ["trial-01", "trial-02"]  # no later trial was started
    marker = [json.loads(line) for line in (suite / "trial-02" / "trace.jsonl").read_text().splitlines()][-1]
    assert marker["event"] == "trial_interrupted" and marker["exit_code"] == 130  # the CLI caught the interrupt and returned 130; the trace says the trial did not finish
    assert jload(suite / "trial-02" / "trial.json")["record"]["exit_code"] == 130
    assert not (suite / "trial-02" / "work").exists() and sr.SINK["path"] is None and sr.TRACE["path"] is None
    assert len(released) >= 3  # our model was released after each trial and at the end
    assert "INCOMPLETE" in (suite / "report.md").read_text() and jload(suite / "trial-02" / "trial.json")["status"] == "interrupted"


def test_a_keyboard_interrupt_raised_inside_a_trial_also_stops_the_suite_with_a_partial_summary(ready, monkeypatch):
    real, calls = dgx.run_e3_trial, []

    def interrupting(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*a, **k)

    monkeypatch.setattr(dgx, "run_e3_trial", interrupting)
    plan = Plan(["ok", "ok", "ok"])
    monkeypatch.setattr(models_mod, "build_model", lambda spec: plan.next_model())
    with pytest.raises(KeyboardInterrupt):
        dgx.cmd_e3_repeat(repeat_args(ready, 3))
    s = jload(next((ready / "diag_E3R").glob("e3r-*")) / "summary.json")
    assert [r["status"] for r in s["per_trial"]] == ["completed", "interrupted", "not_run"] and s["complete"] is False and s["all_three"]["count"] == 1


def test_a_trial_that_crashes_is_recorded_and_the_next_one_still_runs_independently(ready, monkeypatch):
    real = sr.run_cli
    calls = []

    def flaky(config, task):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("the runner broke in trial two")
        return real(config, task)

    monkeypatch.setattr(sr, "run_cli", flaky)
    code, suite, _ = run_suite(monkeypatch, ready, ["ok", "ok", "ok"])
    s = jload(suite / "summary.json")
    assert code == 1 and [r["status"] for r in s["per_trial"]] == ["completed", "runner_error", "completed"]
    assert s["per_trial"][1]["failure_class"] == "runner_error" and "broke in trial two" in s["per_trial"][1]["error"]
    assert (s["context_transfer_exact"]["count"], s["trials"]["incomplete"], s["complete"]) == (2, 1, True)
    assert not (suite / "trial-02" / "work").exists() and jload(suite / "trial-03" / "trial.json")["failure_class"] == "none"


def test_a_summary_with_incomplete_trials_never_counts_them_as_successes():
    meta = {"suite_id": "e3r-x", "tag": TAG, "num_ctx": CTX, "trials_requested": 6, "model_identity": IDENTITY,
            "configuration": {"seed_start": 0, "runner_overrides": {}, "value_source": "secrets", "capture_prompts": False}}
    good = dgx.trial_row(1, "t1", "completed", None, "none", seed=0, seconds=10.0)
    good.update(mechanics_ok=True, context_transfer_exact=True, task_verified=True, generation_seconds=[1.0, 3.0], generation_count=2)
    bad = dict(good, trial=2, context_transfer_exact=False, task_verified=False, failure_class="wrong_result_file", seconds=30.0)
    rows = [good, bad, dgx.trial_row(3, "t3", "interrupted", None, "interrupted", seconds=4.0), dgx.trial_row(4, "t4", "runner_error", None, "runner_error", seconds=1.0, error="boom"),
            dgx.trial_row(5, None, "not_run", None, "not_run"), dgx.trial_row(6, None, "not_run", None, "not_run")]
    s = dgx.summarize_suite(meta, rows, complete=False, interrupted=True)
    assert s["trials"] == {"requested": 6, "attempted": 4, "completed": 2, "incomplete": 2, "blocked": 0, "not_run": 2}
    assert (s["context_transfer_exact"]["of_requested"], s["context_transfer_exact"]["of_completed"], s["context_transfer_exact"]["rate_of_requested"]) == ("1/6", "1/2", 0.167)
    assert s["all_three"]["count"] == 1 and s["failure_classes"] == {"interrupted": 1, "not_run": 2, "runner_error": 1, "wrong_result_file": 1}
    assert s["latency"]["trial_seconds"] == {"n": 2, "min": 10.0, "median": 20.0, "mean": 20.0, "max": 30.0, "total": 40.0}  # only completed trials are timed; the interrupted one is not mixed in
    assert s["latency"]["generation_seconds"]["median"] == 2.0
    assert dgx.summarize_suite(meta, rows[2:], complete=False, interrupted=True)["all_three"]["count"] == 0


def test_the_failure_classes_follow_the_root_cause_order():
    def checks(*failed):
        names = ["reader_read_source", "reader_delegated_before_writer", "value_absent_before_the_reader", "reader_report_has_value", "value_in_writer_delegation",
                 "value_reached_writer_model", "writer_did_not_read_source", "value_first_seen_by_reader"]
        return [{"name": n, "passed": n not in failed} for n in names]

    def task(*failed):
        return [{"name": n, "passed": n not in failed} for n in ("result_file_exists", "result_file_exact", "source_unchanged", "no_other_changes")]

    d = lambda worker, **kw: {"worker": worker, "dispatched": True, "task": kw.get("task", ""), "context": kw.get("context")}  # noqa: E731
    base = {"exit_code": 1, "mechanics_ok": True, "expected_value": "TRANSFER-AAAAAAAA", "delegations": [d("reader"), d("writer", task="use TRANSFER-AAAAAAAA")]}

    def classify(transfer=(), failed_task=(), **over):
        return dgx.classify_trial({**base, **over, "context_transfer": {"checks": checks(*transfer)}, "verification": {"checks": task(*failed_task)}})

    assert classify() == "none"
    assert classify(transfer=("reader_read_source", "value_in_writer_delegation"), failed_task=("result_file_exact",)) == "reader_failed"
    assert classify(transfer=("value_in_writer_delegation", "writer_did_not_read_source")) == "value_not_transferred"
    assert classify(transfer=("value_in_writer_delegation",), delegations=[d("reader"), d("writer", task="use TRANSFER-AAAAAAAA9")]) == "value_altered"
    assert classify(transfer=("writer_did_not_read_source",)) == "writer_bypass"
    assert classify(transfer=("value_first_seen_by_reader",)) == "value_known_before_reader"
    assert classify(failed_task=("result_file_exact", "no_other_changes")) == "wrong_result_file"
    assert classify(delegations=[d("reader")]) == "writer_not_delegated" and classify(delegations=[]) == "no_delegation" and classify(exit_code=3) == "harness_error"
    assert classify(mechanics_ok=False) == "mechanics_failed"
    assert [dgx.classify_trial(None, s) for s in ("interrupted", "runner_error", "not_run")] == ["interrupted", "runner_error", "not_run"]


V = "TRANSFER-AAAAAAAA"
READ = {"tool": "file_operations", "delegation_id": "d1", "args": {"operation": "read", "path": "source.txt"}, "success": True, "visible": f"The transfer value is {V}.", "start_seq": 10}
LOG = [{"tool": "delegate", "delegation_id": "d1", "args": {"worker": "reader", "task": "t"}, "success": True, "dispatched": True,
        "visible": json.dumps({"status": "completed", "output": f"The transfer value is {V}."})}, READ,
       {"tool": "delegate", "delegation_id": "d2", "args": {"worker": "writer", "task": f"write {V}"}, "success": True, "dispatched": True, "visible": "{}"}]


def call(label, role="worker", did="d1", seq=12, obs=True, value=True):
    body = f"Observation: The transfer value is {V}." if obs else "Observation: (nothing yet)"
    return {"prompt": f"You are the worker agent '{label}'. {body if value else ''}", "role": role, "delegation_id": did, "trace_seq": seq}


def first_seen(calls, log=LOG):
    judged = dgx.e3_judgements(log, {"success": True, "workers": []}, calls, Path("/nonexistent"), {}, {}, V)
    return next(c for c in judged["context_transfer"]["checks"] if c["name"] == "value_first_seen_by_reader")["passed"]


WRITER = call("writer", did="d2", seq=20)


def test_the_value_must_first_appear_in_a_reader_prompt_issued_after_its_successful_read():
    assert first_seen([call("reader", seq=12), WRITER]) is True  # the reader's prompt after its read (seq 12 > 10), holding the observation
    assert first_seen([call("reader", seq=7, obs=False, value=False), call("reader", seq=12), WRITER]) is True  # its earlier prompt (before the read) has no value: fine


@pytest.mark.parametrize("why, calls", [
    ("the reader's INITIAL prompt already held the value (issued before its read)", [call("reader", seq=7), call("reader", seq=12), WRITER]),
    ("an orchestrator prompt held it first", [{"prompt": f"You are orchestrator. {V}", "role": "orchestrator", "delegation_id": None, "trace_seq": 5}, call("reader", seq=12), WRITER]),
    ("the writer's prompt held it before the reader's observation", [dict(WRITER, trace_seq=8), call("reader", seq=12)]),
    ("a reader prompt after the read that does not contain the read's observation", [{"prompt": f"You are the worker agent 'reader'. I recall {V}", "role": "worker", "delegation_id": "d1", "trace_seq": 12}, WRITER]),
    ("the prompt belongs to a different delegation than the read", [call("reader", did="d9", seq=12), WRITER]),
    ("the sequence numbers are missing (fails closed)", [{"prompt": call("reader")["prompt"], "role": "worker", "delegation_id": "d1"}, WRITER]),
    ("no prompt ever held it", [call("reader", obs=False, value=False)]),
])
def test_the_first_appearance_check_fails_when_the_ordering_is_not_established(why, calls):
    assert first_seen(calls) is False, why


def test_the_first_appearance_check_needs_a_successful_reader_read():
    failed_read = dict(READ, success=False)
    assert first_seen([call("reader", seq=12), WRITER], [LOG[0], failed_read, LOG[2]]) is False
    assert first_seen([call("reader", seq=12), WRITER], [LOG[0], dict(READ, visible="empty file"), LOG[2]]) is False


# ---- duplicate-output protection, prerequisites and options ----------------------------------------------------------------------------------------
def test_two_suites_never_share_a_folder_and_an_existing_one_is_never_written_into(ready, monkeypatch):
    run_suite(monkeypatch, ready, ["ok"])
    run_suite(monkeypatch, ready, ["ok"])
    suites = sorted((ready / "diag_E3R").glob("e3r-*"))
    assert len(suites) == 2 and suites[0].name != suites[1].name and all(len(trials_of(s)) == 1 for s in suites)
    frozen = SimpleNamespace(datetime=SimpleNamespace(now=lambda tz=None: dt.datetime(2026, 10, 29, 12, 0, 0, tzinfo=dt.timezone.utc)), timezone=dt.timezone)
    monkeypatch.setattr(dgx, "dt", frozen)
    monkeypatch.setattr(dgx.uuid, "uuid4", lambda: SimpleNamespace(hex="0123456789abcdef" * 2))
    assert run_suite(monkeypatch, ready, ["ok"])[0] == 0  # the first run with these frozen ids creates its folder
    before = {p: p.read_bytes() for p in (ready / "diag_E3R" / "e3r-20261029T120000Z-012345").rglob("*") if p.is_file()}
    code, _, _ = run_suite(monkeypatch, ready, ["ok"])
    assert code == 2  # refused: the same folder again
    assert {p: p.read_bytes() for p in before} == before and len(list((ready / "diag_E3R").glob("e3r-*"))) == 3


def test_a_trial_workspace_that_already_exists_is_an_error_not_reused(ready, monkeypatch):
    monkeypatch.setattr(models_mod, "build_model", lambda spec: pytest.fail("a model was loaded"))
    ctx = dgx.make_ctx(repeat_args(ready), ready)
    (ready / "work" / "t1").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        dgx.run_e3_trial(ctx, repeat_args(ready), ready / "work", attempt_id="t1", number=1, seed=0, value="TRANSFER-AAAAAAAA", config_name="t1")


@pytest.mark.parametrize("setup, fragment", [
    (lambda out, mp: [(out / "diag_E0" / "e0.json").unlink()], "no E0 report"),
    (lambda out, mp: [(out / "diag_E2" / "result.json").unlink()], "no E2 result"),
    (lambda out, mp: write_e2(out, tag="other:tag"), "E2 result was made for tag 'other:tag'"),
    (lambda out, mp: write_e2(out, summary=[{"attempt_id": "x", "mechanics_demonstrated": False}]), "did not demonstrate the delegation mechanics"),
    (lambda out, mp: mp.setattr(dgx, "model_identity", lambda tag: None), "does not list the tag"),
    (lambda out, mp: write_e0(out, results=[{"id": "ollama", "status": "pass", "evidence": {"digest": "sha256:" + "cd" * 32}}]), "changed since E0"),
])
def test_repeat_refuses_without_its_prerequisites_and_loads_no_model(ready, monkeypatch, capsys, setup, fragment):
    setup(ready, monkeypatch)
    monkeypatch.setattr(models_mod, "build_model", lambda spec: pytest.fail("a model was loaded although the suite was refused"))
    assert dgx.cmd_e3_repeat(repeat_args(ready)) == 2
    assert fragment in capsys.readouterr().out and not (ready / "diag_E3R").exists()


@pytest.mark.parametrize("over, fragment", [({"trials": 0}, "--trials must be between"), ({"trials": 11}, "--trials must be between"), ({"seed_start": -1}, "--seed-start"),
                                             ({"value": "TRANSFER-AAAAAAAA", "trials": 3}, "--trials 1"), ({"value": "TRANSFER-4172", "trials": 1}, "must look like")])
def test_repeat_refuses_unsound_options(ready, monkeypatch, capsys, over, fragment):
    monkeypatch.setattr(models_mod, "build_model", lambda spec: pytest.fail("a model was loaded"))
    a = repeat_args(ready)
    for key, value in over.items():
        setattr(a, key, value)
    assert dgx.cmd_e3_repeat(a) == 2 and fragment in capsys.readouterr().out and not (ready / "diag_E3R").exists()


def test_a_replayed_value_is_used_only_when_asked_and_is_labelled(ready, monkeypatch):
    plan = Plan(["ok"])
    monkeypatch.setattr(models_mod, "build_model", lambda spec: plan.next_model())
    a = repeat_args(ready, 1)
    a.value, a.seed_start = "TRANSFER-ABCDEFGH", 7
    assert dgx.cmd_e3_repeat(a) == 0
    suite = next((ready / "diag_E3R").glob("e3r-*"))
    trial = jload(suite / "trial-01" / "trial.json")
    assert trial["record"]["expected_value"] == "TRANSFER-ABCDEFGH" and trial["seed"] == 7 and trial["value_source"].startswith("operator")
    assert jload(suite / "summary.json")["configuration"]["value_source"].startswith("operator") and "TRANSFER-ABCDEFGH" not in (suite / "summary.json").read_text()


# ---- evidence: identity, configuration, prompts, packaging ---------------------------------------------------------------------------------------------
def test_the_model_identity_prerequisites_and_configuration_are_recorded(ready, monkeypatch):
    code, suite, _ = run_suite(monkeypatch, ready, ["ok"])
    meta, trial, summary = jload(suite / "suite.json"), jload(suite / "trial-01" / "trial.json"), jload(suite / "summary.json")
    assert meta["model_identity"]["digest"] == trial["model_identity_checked"]["digest"] == summary["model_identity"]["digest"] == DIGEST  # the trial records what was checked for IT
    assert meta["prerequisites"]["e0_report_sha256"] == hashlib.sha256((ready / "diag_E0" / "e0.json").read_bytes()).hexdigest()
    assert meta["prerequisites"]["e2_result_sha256"] == hashlib.sha256((ready / "diag_E2" / "result.json").read_bytes()).hexdigest()
    cfg = meta["configuration"]
    assert cfg["runner_overrides"] == sr.SAMPLING and cfg["max_delegations"] == 3 and cfg["capture_prompts"] is False and cfg["driver_sha256"] and cfg["task"] == dgx.E3_TASK
    assert set(meta["git"]) >= {"commit", "branch", "dirty"} and meta["host"]["machine"]
    record = trial["record"]
    assert record["mechanics_ok"] and record["context_transfer"]["exact"] and record["verification"]["task_verified"]  # the three verdicts, separately
    assert record["delegations"] and record["worker_tool_calls"] and record["generations"] and record["file_hashes"]["expected_value_sha256"] == dgx.value_fingerprint(record["expected_value"])
    assert trial["wall_clock_seconds"] >= record["seconds"] >= 0 and trial["started_utc"] <= trial["ended_utc"]


PROMPT_MARKER = "Delegate a self-contained task to a worker agent"  # part of the delegate tool's description, i.e. of EVERY orchestrator prompt and of nothing else we write


def test_prompts_are_not_captured_by_default_and_never_reach_a_shared_file_when_they_are(ready, monkeypatch):
    code, suite, plan = run_suite(monkeypatch, ready, ["ok"])
    assert any(PROMPT_MARKER in str(c.prompt) for c in plan.models[0].calls)  # the marker really is in the prompts the model received...
    assert not list(suite.rglob("prompts.local.jsonl")) and not [p for p in suite.rglob("*") if p.is_file() and PROMPT_MARKER in p.read_text(errors="replace")]  # ...and in no file we wrote
    code, suite, plan = run_suite(monkeypatch, ready, ["ok"], capture_prompts=True)
    captured = suite / "trial-01" / "prompts.local.jsonl"
    lines = [json.loads(line) for line in captured.read_text().splitlines()]
    assert stat.S_IMODE(captured.stat().st_mode) == 0o600 and drv.exclusion_reason(captured) is not None  # private, and the packager leaves it out
    assert len(lines) == len(plan.models[0].calls) and sorted(l["text"] for l in lines) == sorted(str(c.prompt) for c in plan.models[0].calls)  # exactly what the model received
    assert all(l["sha256"] == hashlib.sha256(l["text"].encode()).hexdigest() for l in lines) and any(PROMPT_MARKER in l["text"] for l in lines)
    shared = [p for p in suite.rglob("*") if p.is_file() and p != captured]
    assert shared and not [p for p in shared if PROMPT_MARKER in p.read_text(errors="replace")]  # every OTHER file in the suite is free of prompt text
    assert jload(suite / "suite.json")["configuration"]["capture_prompts"] is True


def test_the_archive_keeps_trial_evidence_but_no_private_files_and_no_clear_values_in_the_aggregates(ready, monkeypatch):
    import tarfile

    code, suite, _ = run_suite(monkeypatch, ready, ["ok", "ok"], capture_prompts=True)
    values = [value_of(t) for t in trials_of(suite)]
    results = ready.parent / "archive-results"
    results.mkdir()
    monkeypatch.setattr(drv, "RESULTS", results)
    assert drv.cmd_package(SimpleNamespace(run_dir=str(ready))) == 0
    with tarfile.open(results / f"{ready.name}.tar.gz") as tar:
        members = {m.name: tar.extractfile(m).read().decode("utf-8", "replace") for m in tar.getmembers() if m.isfile()}
    inside = {n.split(suite.name + "/", 1)[1]: text for n, text in members.items() if suite.name + "/" in n}
    assert not [n for n in members if "local.jsonl" in n or n.endswith(".db") or n.endswith(".sqlite")]  # no private prompt file, no database
    assert not [n for n, text in members.items() if PROMPT_MARKER in text]  # and no prompt text anywhere in what was archived
    for name in ("summary.json", "report.md", "suite.json"):
        assert name in inside and not [v for v in values if v in inside[name]], name  # aggregates never carry a clear value
    for i, v in enumerate(values, 1):
        assert v in inside[f"trial-0{i}/trial.json"] and v in inside[f"trial-0{i}/trace.jsonl"] and not [o for o in values if o != v and o in inside[f"trial-0{i}/trace.jsonl"]]  # the audit evidence keeps its own value only


# ---- the original single-run path is unchanged -------------------------------------------------------------------------------------------------------------
def test_single_run_e3_still_uses_its_fixed_value_folders_and_adds_no_suite(host, monkeypatch):
    write_e0(host)
    write_e2(host)
    scripted(monkeypatch, e3_script())
    assert dgx.cmd_e3(args(host)) == 0
    assert not (host / "diag_E3R").exists() and (host / "diag_E3" / "result.json").exists() and (host / "stage_X" / "cli_ctx-transfer-a1.json").exists()
    record = json.loads((host / "diag_E3" / "attempts.jsonl").read_text().splitlines()[0])
    assert record["attempt_id"] == "ctx-transfer-a1" and record["expected_value"] == dgx.E3_VALUE == "TRANSFER-4172" and record["seed"] == 0
    row = json.loads((host / "diag_E3" / "result.json").read_text())["summary"][0]
    assert (row["mechanics_demonstrated"], row["task_verified"], row["context_transfer_exact"]) == (True, True, True)
    assert "value_first_seen_by_reader" in [c["name"] for c in record["context_transfer"]["checks"]] and record["context_transfer"]["exact"] is True


# ---- review fixes: the per-trial lifecycle is interruption-safe ---------------------------------------------------------------------------------------
def once(fn, when):
    """Wrap `fn` so that the first call for which `when(*args, **kwargs)` is true raises KeyboardInterrupt (once)."""
    state = {"fired": False}

    def wrapper(*args, **kwargs):
        if not state["fired"] and when(*args, **kwargs):
            state["fired"] = True
            raise KeyboardInterrupt
        return fn(*args, **kwargs)

    return wrapper


def interrupt_at(point, monkeypatch):
    if point == "release":
        monkeypatch.setattr(drv, "release_ollama", once(lambda *a, **k: {"verified": True}, lambda profile, label, *a, **k: "-t01" in label))
    elif point == "summary_write":
        monkeypatch.setattr(dgx, "write_suite_files", once(dgx.write_suite_files, lambda d, m, rows, **k: rows[0]["status"] == "completed"))
    elif point == "trial_json_write":
        monkeypatch.setattr(dgx, "write_json_atomic", once(dgx.write_json_atomic, lambda path, data: path.name == "trial.json" and data.get("record") is not None))
    elif point == "classification":
        monkeypatch.setattr(dgx, "classify_trial", once(dgx.classify_trial, lambda record, status="completed": record is not None))
    else:
        raise AssertionError(point)


@pytest.mark.parametrize("point", ["release", "summary_write", "trial_json_write", "classification"])
def test_an_interrupt_after_a_successful_trial_never_loses_or_downgrades_it(ready, monkeypatch, point):
    interrupt_at(point, monkeypatch)
    plan = Plan(["ok", "ok", "ok"])
    monkeypatch.setattr(models_mod, "build_model", lambda spec: plan.next_model())
    with pytest.raises(KeyboardInterrupt):
        dgx.cmd_e3_repeat(repeat_args(ready, 3))
    suite = next((ready / "diag_E3R").glob("e3r-*"))
    s = jload(suite / "summary.json")  # a valid summary, whatever was interrupted
    assert [r["trial"] for r in s["per_trial"]] == [1, 2, 3]  # one row per trial: no duplicate, no missing row
    assert [r["status"] for r in s["per_trial"]] == ["completed", "not_run", "not_run"]  # the finished trial stays completed
    assert (s["interrupted"], s["complete"]) == (True, False) and s["trials"]["completed"] == 1 and s["trials"]["incomplete"] == 0 and s["trials"]["not_run"] == 2
    assert (s["all_three"]["count"], s["task_verified"]["count"], s["context_transfer_exact"]["of_requested"]) == (1, 1, "1/3")
    trial = jload(suite / "trial-01" / "trial.json")  # its evidence record survived
    assert trial["status"] == "completed" and trial["record"]["context_transfer"]["exact"] is True and trial["record"]["verification"]["task_verified"] is True
    assert len(plan.models) == 1 and [t.name for t in trials_of(suite)] == ["trial-01"]  # no later trial was started
    assert sr.SINK["path"] is None and sr.TRACE["path"] is None
    assert not list(suite.glob("*.tmp")) and not list(suite.rglob("*.tmp"))  # atomic writes leave no half-written file behind


def test_an_interrupt_during_the_last_trials_release_leaves_a_complete_but_interrupted_suite(ready, monkeypatch):
    monkeypatch.setattr(drv, "release_ollama", once(lambda *a, **k: {"verified": True}, lambda profile, label, *a, **k: "-t02" in label))
    plan = Plan(["ok", "ok"])
    monkeypatch.setattr(models_mod, "build_model", lambda spec: plan.next_model())
    with pytest.raises(KeyboardInterrupt):
        dgx.cmd_e3_repeat(repeat_args(ready, 2))
    s = jload(next((ready / "diag_E3R").glob("e3r-*")) / "summary.json")
    assert [r["status"] for r in s["per_trial"]] == ["completed", "completed"] and (s["complete"], s["interrupted"]) == (True, True)  # every trial finished; the command was interrupted
    assert s["all_three"]["count"] == 2 and "(the command was interrupted)" in (next((ready / "diag_E3R").glob("e3r-*")) / "report.md").read_text()


def test_an_interrupt_before_the_model_is_even_checked_for_a_later_trial_is_recorded_for_that_trial(ready, monkeypatch):
    calls = []
    monkeypatch.setattr(dgx, "model_identity", lambda tag: (calls.append(1), (_ for _ in ()).throw(KeyboardInterrupt()) if len(calls) == 3 else dict(IDENTITY, tag=tag))[1])  # gate, trial 1, then trial 2
    plan = Plan(["ok", "ok", "ok"])
    monkeypatch.setattr(models_mod, "build_model", lambda spec: plan.next_model())
    with pytest.raises(KeyboardInterrupt):
        dgx.cmd_e3_repeat(repeat_args(ready, 3))
    suite = next((ready / "diag_E3R").glob("e3r-*"))
    s = jload(suite / "summary.json")
    assert [r["status"] for r in s["per_trial"]] == ["completed", "interrupted", "not_run"] and s["interrupted"] is True and len(plan.models) == 1
    assert jload(suite / "trial-02" / "trial.json")["record"] is None and not (suite / "trial-02" / "trace.jsonl").exists()  # nothing was executed for it


def test_a_summary_exists_from_the_start_and_shows_the_trial_in_flight(ready, monkeypatch):
    seen = {}

    class Peeking(FollowsTheValue):
        def generate(self, prompt, config=None, **kw):
            if "snapshot" not in seen:
                seen["snapshot"] = jload(next((ready / "diag_E3R").glob("e3r-*")) / "summary.json")
                seen["trial_json"] = (next((ready / "diag_E3R").glob("e3r-*")) / "trial-01" / "trial.json").exists()
            return super().generate(prompt, config, **kw)

    monkeypatch.setattr(models_mod, "build_model", lambda spec: Peeking("ok"))
    assert dgx.cmd_e3_repeat(repeat_args(ready, 2)) == 0
    snap = seen["snapshot"]  # what a kill -9 at this very moment would have left on disk
    assert [r["status"] for r in snap["per_trial"]] == ["running", "not_run"] and snap["complete"] is False and snap["interrupted"] is False
    assert snap["trials"] == {"requested": 2, "attempted": 1, "completed": 0, "incomplete": 1, "blocked": 0, "not_run": 1} and snap["all_three"]["count"] == 0
    assert seen["trial_json"] is False  # the trial is not claimed as finished
    assert jload(next((ready / "diag_E3R").glob("e3r-*")) / "summary.json")["complete"] is True


# ---- review fixes: the model identity is verified before EVERY trial -------------------------------------------------------------------------------
def test_with_a_matching_e0_digest_every_trial_runs_and_records_the_identity_it_was_checked_against(ready, monkeypatch):
    write_e0(ready, results=[{"id": "ollama", "status": "pass", "evidence": {"digest": DIGEST}}])
    code, suite, plan = run_suite(monkeypatch, ready, ["ok", "ok"])
    assert code == 0 and jload(suite / "suite.json")["prerequisites"]["e0_recorded_digest"] == DIGEST
    for trial in trials_of(suite):
        assert jload(trial / "trial.json")["model_identity_checked"]["digest"] == DIGEST
    assert [r["model_digest"] for r in jload(suite / "summary.json")["per_trial"]] == [DIGEST, DIGEST]


def drifting(monkeypatch, later):
    """Ollama reports the validated digest for the gate and the first trial, then `later` (a dict, None, or an exception to raise)."""
    calls = []

    def identity(tag):
        calls.append(1)
        if len(calls) <= 2:
            return dict(IDENTITY, tag=tag)
        if isinstance(later, Exception):
            raise later
        return later(tag) if callable(later) else later

    monkeypatch.setattr(dgx, "model_identity", identity)


@pytest.mark.parametrize("later, failure", [
    (lambda tag: dict(IDENTITY, tag=tag, digest="sha256:" + "ee" * 32), "model_identity_changed"),
    (None, "model_identity_unverifiable"),
    ({"tag": TAG}, "model_identity_unverifiable"),                       # malformed: no digest
    ({"tag": TAG, "digest": 12345}, "model_identity_unverifiable"),      # malformed: not a string
    ({"tag": TAG, "digest": ""}, "model_identity_unverifiable"),
    ("not a dict", "model_identity_unverifiable"),
    (OSError("connection refused"), "model_identity_unverifiable"),      # the lookup itself crashed
])
def test_a_model_that_cannot_be_verified_before_a_trial_stops_the_suite_without_running_it(ready, monkeypatch, later, failure):
    drifting(monkeypatch, later)
    code, suite, plan = run_suite(monkeypatch, ready, ["ok", "ok", "ok"])
    s = jload(suite / "summary.json")
    assert code == 1 and [r["status"] for r in s["per_trial"]] == ["completed", "blocked", "not_run"]
    assert s["per_trial"][1]["failure_class"] == failure and s["per_trial"][1]["task_verified"] is None and s["per_trial"][1]["mechanics_ok"] is None  # never an ordinary task result
    assert (s["complete"], s["interrupted"]) == (False, False) and s["trials"]["blocked"] == 1 and s["trials"]["not_run"] == 1 and s["failure_classes"] == {failure: 1, "not_run": 1}
    assert (s["all_three"]["count"], s["task_verified"]["count"]) == (1, 1)  # the finished trial is intact and the blocked one is no success
    assert len(plan.models) == 1 and not (suite / "trial-02" / "trace.jsonl").exists() and not (suite / "trial-02" / "work").exists()  # the blocked trial executed nothing
    blocked = jload(suite / "trial-02" / "trial.json")
    assert blocked["status"] == "blocked" and blocked["record"] is None and failure in blocked["failure_class"] and blocked["error"]


def test_a_digest_that_changes_right_after_the_gate_blocks_the_first_trial(ready, monkeypatch):
    calls = []
    monkeypatch.setattr(dgx, "model_identity", lambda tag: dict(IDENTITY, tag=tag, digest=DIGEST if not calls.append(1) and len(calls) == 1 else "sha256:" + "ff" * 32))
    code, suite, plan = run_suite(monkeypatch, ready, ["ok", "ok"])
    s = jload(suite / "summary.json")
    assert code == 1 and [r["status"] for r in s["per_trial"]] == ["blocked", "not_run"] and plan.models == [] and s["trials"]["completed"] == 0


def test_verify_identity_checks_the_suite_start_and_the_e0_digest():
    good = {"digest": "sha256:aa"}
    assert dgx.verify_identity(good, TAG, "sha256:aa", "sha256:aa") is None and dgx.verify_identity(good, TAG, "sha256:aa", None) is None
    assert dgx.verify_identity(good, TAG, "sha256:bb", None)[0] == "model_identity_changed"
    assert dgx.verify_identity(good, TAG, "sha256:aa", "sha256:bb")[0] == "model_identity_changed"  # equal to the suite start but not to what E0 validated
    assert [dgx.verify_identity(x, TAG, "sha256:aa", None)[0] for x in (None, {}, {"digest": None}, [], "x")] == ["model_identity_unverifiable"] * 5


# ---- review fixes: a REAL Ctrl-C while a worker is running (controlled subprocess; scripted, blocking model; no inference) ------------------------------------
SIGINT_DRIVER = r'''
import json, os, signal, sys, time
from pathlib import Path
ROOT = {root!r}
sys.path.insert(0, ROOT + "/tests"); sys.path.insert(0, ROOT + "/experiments/dgx")
signal.signal(signal.SIGINT, signal.default_int_handler)  # a test runner may have left SIGINT ignored
out, marker, stop = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
import dgx_diag as dgx
import smol_harness.models as models_mod
from test_dgx_e3_repeat import FollowsTheValue, IDENTITY
drv = dgx.drv
(out / "diag_E0").mkdir(parents=True, exist_ok=True); (out / "diag_E2").mkdir(parents=True, exist_ok=True)
(out / "diag_E0" / "e0.json").write_text(json.dumps({{"tag": "t:8k", "num_ctx": 8192, "gates_e1_e2": True, "gating": {{}}}}))
(out / "diag_E2" / "result.json").write_text(json.dumps({{"tag": "t:8k", "num_ctx": 8192, "summary": [{{"mechanics_demonstrated": True}}]}}))
dgx.model_identity = lambda tag: dict(IDENTITY, tag=tag)
drv.release_ollama = lambda *a, **k: {{"verified": True}}
drv.snapshot_ollama = lambda *a, **k: None

class Blocking(FollowsTheValue):
    """The reader's FIRST model call blocks in the worker thread until the stop file exists: a stand-in for one long local generation, which nothing can interrupt."""
    def respond(self, prompt):
        if "You are the worker agent 'reader'" in prompt and not os.path.exists(marker):
            open(marker, "w").write("running")
            while not os.path.exists(stop):
                time.sleep(0.05)
        return super().respond(prompt)

models_mod.build_model = lambda spec: (lambda m: (setattr(m, "unload", lambda: None), m)[1])(Blocking("ok"))
sys.exit(dgx.main(["e3-repeat", "--out", str(out), "--tag", "t:8k", "--num-ctx", "8192", "--trials", "3"]))
'''


@pytest.mark.parametrize("interrupts", [1, 2])
def test_a_real_sigint_during_a_worker_call_waits_for_the_call_then_leaves_an_accurate_partial_summary(tmp_path, interrupts):
    """What really happens (measured, 3.12): SIGINT reaches the main thread, but asyncio's executor shutdown joins the worker thread, so after ONE Ctrl-C the process keeps running until
    the in-flight model call returns and only then writes the interrupted state. A SECOND Ctrl-C breaks that wait: the suite writes the final interrupted state at once (the process still
    lingers until the call returns, because Python cannot stop a running thread). A summary exists the whole time and is never half-written."""
    import signal
    import subprocess
    import time

    driver, out, marker, stop = tmp_path / "driver.py", tmp_path / "out", tmp_path / "marker", tmp_path / "stop"
    driver.write_text(SIGINT_DRIVER.format(root=str(ROOT)))
    process = subprocess.Popen([sys.executable, "-I", str(driver), str(out), str(marker), str(stop)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT)
    try:
        deadline = time.time() + 90
        while not marker.exists():
            assert process.poll() is None and time.time() < deadline, process.communicate()
            time.sleep(0.05)
        suite = next((out / "diag_E3R").glob("e3r-*"))
        before = jload(suite / "summary.json")  # a kill -9 right now would leave exactly this
        assert [r["status"] for r in before["per_trial"]] == ["running", "not_run", "not_run"] and before["complete"] is False and not (suite / "trial-01" / "trial.json").exists()
        for n in range(1, interrupts + 1):
            process.send_signal(signal.SIGINT)
            time.sleep(1.5)
            assert process.poll() is None  # still alive: the worker thread is still inside its model call
            now = jload(suite / "summary.json")  # always valid JSON, whatever the timing of the signal
            assert now["per_trial"][0]["status"] == ("running" if n == 1 else "interrupted")  # one Ctrl-C: still waiting; the second: the interrupted state is written NOW
            if n == 2:
                assert (now["interrupted"], now["complete"]) == (True, False) and jload(suite / "trial-01" / "trial.json")["status"] == "interrupted"
        stop.write_text("go")  # the long call returns
        stdout, stderr = process.communicate(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode in (130, -signal.SIGINT), (process.returncode, stderr[-500:])
    s = jload(suite / "summary.json")
    assert [r["status"] for r in s["per_trial"]] == ["interrupted", "not_run", "not_run"] and (s["interrupted"], s["complete"]) == (True, False)
    assert (s["trials"]["completed"], s["all_three"]["count"], s["task_verified"]["count"]) == (0, 0, 0)  # an interrupted trial is no success
    trial = jload(suite / "trial-01" / "trial.json")
    assert trial["status"] == "interrupted" and trial["record"]["exit_code"] == 130 and [t.name for t in trials_of(suite)] == ["trial-01"]
    assert json.loads((suite / "trial-01" / "trace.jsonl").read_text().splitlines()[-1])["event"] == "trial_interrupted"
    assert "INTERRUPTED" in stdout and not list(suite.rglob("*.tmp"))


def test_the_finished_trial_is_on_disk_before_the_model_release_even_starts(ready, monkeypatch):
    """The end state alone cannot show the ORDER (the interrupt handler persists again), so look at the moment the release runs: a kill -9 or a second Ctrl-C there must lose nothing."""
    at_release = []

    def release(profile, label, *a, **k):
        if "-t01" in label:
            suite = next((ready / "diag_E3R").glob("e3r-*"))
            at_release.append((jload(suite / "trial-01" / "trial.json"), jload(suite / "summary.json")))
        return {"verified": True}

    monkeypatch.setattr(drv, "release_ollama", release)
    code, suite, _ = run_suite(monkeypatch, ready, ["ok", "ok"])
    trial, summary = at_release[0]
    assert trial["status"] == "completed" and trial["record"]["context_transfer"]["exact"] is True  # the full evidence record is already written
    assert [r["status"] for r in summary["per_trial"]] == ["completed", "not_run"] and summary["trials"]["completed"] == 1 and summary["all_three"]["count"] == 1
    assert jload(suite / "trial-01" / "release.json") == {"verified": True} and jload(suite / "summary.json")["per_trial"][0]["release_verified"] is True  # the release outcome is added afterwards


def test_an_accurate_incomplete_summary_exists_before_the_first_trial_starts(ready, monkeypatch):
    snapshots = []
    calls = []

    def identity(tag):
        calls.append(1)
        if len(calls) == 2:  # the per-trial lookup of trial 1: nothing of the trial exists yet
            suite = next((ready / "diag_E3R").glob("e3r-*"))
            snapshots.append((jload(suite / "summary.json"), (suite / "suite.json").exists(), (suite / "trial-01" / "trial.json").exists()))
        return dict(IDENTITY, tag=tag)

    monkeypatch.setattr(dgx, "model_identity", identity)
    run_suite(monkeypatch, ready, ["ok", "ok"])
    summary, has_meta, has_trial = snapshots[0]
    assert [r["status"] for r in summary["per_trial"]] == ["not_run", "not_run"] and summary["complete"] is False and summary["trials"]["attempted"] == 0 and has_meta and not has_trial
