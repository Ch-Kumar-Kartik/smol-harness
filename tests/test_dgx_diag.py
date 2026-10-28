"""The DGX Spark diagnostic driver (experiments/dgx/dgx_diag.py): CPU only, no network, no real model, no DGX.

What is pinned: the unified-memory GPU reader (and that the unmodified RTX reader raises on `[N/A]`), the E0 gate that E1 and E2 refuse to run without, and, with scripted
models and the real file tool, that E1 and E2 keep "the mechanics ran" apart from "the task was solved" and write the evidence the runbook promises. Nothing here says the
driver works on a DGX Spark: that needs the hardware.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import ScriptedModel

import smol_harness.models as models_mod

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "dgx"))
import dgx_diag as dgx  # noqa: E402

drv, sr = dgx.drv, dgx.sr
CODE = "7391-QX"
TAG, CTX = "smol-diag:8k", 8192


def act(tool: str, args: dict) -> str:
    return f"Thought: next.\nAction: {tool}\nAction Input: {json.dumps(args)}"


def final(text: str) -> str:
    return f"Thought: done.\nFinal Answer: {text}"


def args(out: Path, attempts: int = 1):
    return SimpleNamespace(out=str(out), tag=TAG, num_ctx=CTX, attempts=attempts)


@pytest.fixture
def host(tmp_path, monkeypatch):
    """No Ollama, no GPU reads, fresh trace state; whatever the commands change in drv/sr is restored afterwards."""
    monkeypatch.setattr(drv, "nvidia_smi_gpus", drv.nvidia_smi_gpus)
    monkeypatch.setattr(sr, "TRACE", sr.fresh_trace())
    monkeypatch.setitem(sr.SINK, "path", None)
    monkeypatch.setattr(drv, "release_ollama", lambda *a, **k: {"verified": True, "reason": "test double"})
    monkeypatch.setattr(drv, "snapshot_ollama", lambda *a, **k: None)
    return tmp_path


def write_e0(out: Path, **over):
    report = {"experiment": "E0", "tag": TAG, "num_ctx": CTX, "gates_e1_e2": True, "gating": {i: "pass" for i in dgx.GATING}, **over}
    (out / "diag_E0").mkdir(parents=True, exist_ok=True)
    (out / "diag_E0" / "e0.json").write_text(json.dumps(report))


# ---- unified memory ---------------------------------------------------------------------------------------------------------------------------
def fake_smi(monkeypatch, line: str):
    real = drv.run
    monkeypatch.setattr(drv, "run", lambda cmd, *a, **k: (0, line + "\n", "") if cmd and cmd[0] == "nvidia-smi" else real(cmd, *a, **k))


def test_the_unmodified_rtx_reader_raises_when_nvidia_smi_reports_no_numbers(monkeypatch):
    """The reason the adapter exists: on a unified-memory GPU nvidia-smi may print [N/A] for memory. (If the RTX package ever tolerates this, update this pin and the runbook.)"""
    fake_smi(monkeypatch, "NVIDIA GB10, [N/A], [N/A], [N/A], 580.95.05")
    with pytest.raises(ValueError):
        drv.nvidia_smi_gpus()


def test_the_dgx_reader_falls_back_to_system_ram_and_says_so(monkeypatch):
    fake_smi(monkeypatch, "NVIDIA GB10, [N/A], [N/A], [N/A], 580.95.05")
    gpus, err = dgx.gpus_tolerant()
    mem = dgx.meminfo_mib()
    assert err == "" and gpus[0]["name"] == "NVIDIA GB10" and gpus[0]["driver"] == "580.95.05"
    assert gpus[0]["total_mib"] == mem["MemTotal"] and gpus[0]["free_mib"] == mem["MemAvailable"] and gpus[0]["used_mib"] == mem["MemTotal"] - mem["MemAvailable"]
    assert "unified memory" in gpus[0]["memory_source"] and "[N/A]" in gpus[0]["memory_source"]


def test_the_dgx_reader_keeps_real_numbers_when_nvidia_smi_gives_them(monkeypatch):
    fake_smi(monkeypatch, "NVIDIA GeForce RTX 4070, 12282, 1000, 11282, 576.02")
    gpus, _ = dgx.gpus_tolerant()
    assert (gpus[0]["total_mib"], gpus[0]["used_mib"], gpus[0]["free_mib"], gpus[0]["memory_source"]) == (12282, 1000, 11282, "nvidia-smi")


# ---- E0 and the gate -------------------------------------------------------------------------------------------------------------------------
def check(status, detail="x"):
    return lambda a: (status, detail, {"seen": True})


def test_e0_gates_on_the_core_checks_and_not_on_the_sandbox():
    checks = [(i, check("pass")) for i in dgx.GATING] + [("sandbox", check("fail", "no user namespaces"))]
    report = dgx.run_e0(args(Path("/unused")), checks)
    assert report["gates_e1_e2"] is True  # E1/E2 use file_operations only; the sandbox result is recorded, not gating
    assert {r["id"]: r["status"] for r in report["results"]}["sandbox"] == "fail"


@pytest.mark.parametrize("bad", ["fail", "unverified"])
def test_e0_does_not_open_the_gate_when_a_gating_check_is_not_ok(bad):
    checks = [(i, check(bad if i == "project_env" else "pass")) for i in dgx.GATING]
    assert dgx.run_e0(args(Path("/unused")), checks)["gates_e1_e2"] is False


def test_e0_accepts_warnings_on_gating_checks_and_reports_a_crashing_check_as_unverified():
    checks = [(i, check("warn")) for i in dgx.GATING]
    assert dgx.run_e0(args(Path("/unused")), checks)["gates_e1_e2"] is True
    crashing = [("gpu", lambda a: 1 / 0)] + [(i, check("pass")) for i in dgx.GATING if i != "gpu"]
    report = dgx.run_e0(args(Path("/unused")), crashing)
    assert report["results"][0]["status"] == "unverified" and "ZeroDivisionError" in report["results"][0]["detail"] and report["gates_e1_e2"] is False


def test_the_inference_check_is_skipped_when_ollama_itself_failed():
    checks = [("ollama", check("fail")), ("ollama_inference", lambda a: pytest.fail("must not run"))]
    assert [r["status"] for r in dgx.run_e0(args(Path("/unused")), checks)["results"]] == ["fail", "unverified"]


def test_e1_and_e2_refuse_without_a_passing_e0_for_the_same_model(host, capsys):
    assert dgx.cmd_e1(args(host)) == 2 and "no E0 report" in capsys.readouterr().out
    write_e0(host, tag="other:tag")
    assert dgx.cmd_e2(args(host)) == 2 and "must use the same" in capsys.readouterr().out
    write_e0(host, gates_e1_e2=False, gating={"ollama": "fail"})
    assert dgx.cmd_e1(args(host)) == 2 and "ollama=fail" in capsys.readouterr().out
    assert not (host / "diag_E1").exists() and not (host / "diag_E2").exists()  # a refused run leaves no trace of having run


# ---- E1: a real-model file read, judged by the runner, not by the model -----------------------------------------------------------------------
def scripted(monkeypatch, turns):
    model = ScriptedModel(turns, name="scripted")
    model.unload = lambda: None
    monkeypatch.setattr(models_mod, "build_model", lambda spec: model)
    return model


def read_secret() -> str:
    return act("file_operations", {"operation": "read", "path": "secret.txt"})


def events(path: Path):
    return [json.loads(line) for line in (path / "trace.jsonl").read_text().splitlines()]


def test_e1_records_the_real_read_and_verifies_the_task_independently(host, monkeypatch, capsys):
    write_e0(host)
    scripted(monkeypatch, [read_secret(), final(f"The access code is {CODE}.")])
    assert dgx.cmd_e1(args(host)) == 0
    e1 = host / "diag_E1"
    result = json.loads((e1 / "result.json").read_text())
    row = result["summary"][0]
    assert row["mechanics_demonstrated"] is True and row["agent_completed"] is True and row["task_verified"] is True and row["failed_checks"] == []
    trace = events(e1)
    reads = [e for e in trace if e["event"] == "tool_call"]
    assert [(e["tool"], e["args"]["path"], e["success"]) for e in reads] == [("file_operations", "secret.txt", True)] and CODE in reads[0]["visible"]
    assert trace[0]["event"] == "stage_start" and trace[0]["models"]["hf"] is None and trace[0]["models"]["ollama"]["tag"] == TAG
    assert trace[-1]["event"] == "attempt_end" and trace[-1]["task_verified"] is True
    assert (e1 / "attempts.jsonl").exists() and not (e1 / "work").exists()  # evidence kept, disposable workspace removed
    assert "TASK VERIFIED=True" in capsys.readouterr().out


def test_e1_a_completed_run_with_a_wrong_answer_is_a_task_failure_but_the_mechanics_still_count(host, monkeypatch):
    write_e0(host)
    scripted(monkeypatch, [read_secret(), final("The access code is 0000-ZZ.")])
    assert dgx.cmd_e1(args(host)) == 0  # exit status is about the harness; the task result is a separate field
    row = json.loads((host / "diag_E1" / "result.json").read_text())["summary"][0]
    assert (row["mechanics_demonstrated"], row["agent_completed"], row["task_verified"]) == (True, True, False) and row["failed_checks"] == ["answer_has_code"]


def test_e1_a_model_that_never_calls_the_tool_demonstrates_no_mechanics(host, monkeypatch):
    write_e0(host)
    scripted(monkeypatch, [final(f"The access code is {CODE}.")])  # claims the right code without reading anything
    assert dgx.cmd_e1(args(host)) == 1
    row = json.loads((host / "diag_E1" / "result.json").read_text())["summary"][0]
    assert row["mechanics_demonstrated"] is False and row["task_verified"] is False and "secret_read_executed" in row["failed_checks"]
    assert not [e for e in events(host / "diag_E1") if e["event"] == "tool_call"]


# ---- E2: delegation, with the delegated task text and the worker's real tool call -------------------------------------------------------------
def test_e2_records_the_delegation_the_worker_attribution_and_the_independent_verdict(host, monkeypatch):
    write_e0(host)
    delegated = "Read secret.txt and report the access code."
    scripted(monkeypatch, [act("delegate", {"worker": "reader", "task": delegated}), read_secret(), final("The file holds an access code."), final(f"The access code is {CODE}.")])
    assert dgx.cmd_e2(args(host)) == 0
    e2 = host / "diag_E2"
    row = json.loads((e2 / "result.json").read_text())["summary"][0]
    assert (row["mechanics_demonstrated"], row["agent_completed"], row["task_verified"]) == (True, True, True)
    trace = events(e2)
    tools = [(e["tool"], e["role"], e["delegation_id"]) for e in trace if e["event"] == "tool_call"]
    assert tools == [("file_operations", "worker", "d1"), ("delegate", "orchestrator", "d1")]
    delegate_call = next(e for e in trace if e["event"] == "tool_call" and e["tool"] == "delegate")
    assert delegate_call["args"]["task"] == delegated and delegate_call["dispatched"] is True
    record = json.loads((e2 / "attempts.jsonl").read_text().splitlines()[0])
    assert [d["task"] for d in record["delegations"]] == [delegated] and record["delegations"][0]["report_status"] == "completed" and record["role_split_agrees"] is True
    assert list(host.glob("stage_X/cli_*.json"))  # the CLI output of the attempt


def test_e2_a_completed_delegation_with_the_wrong_answer_is_not_a_task_success(host, monkeypatch):
    write_e0(host)
    scripted(monkeypatch, [act("delegate", {"worker": "reader", "task": "Read secret.txt and report the code."}), read_secret(), final("It holds a code."), final("The access code is 0000-ZZ.")])
    assert dgx.cmd_e2(args(host)) == 0
    row = json.loads((host / "diag_E2" / "result.json").read_text())["summary"][0]
    assert (row["mechanics_demonstrated"], row["agent_completed"], row["task_verified"]) == (True, True, False) and row["failed_checks"] == ["answer_has_code"]


# ---- an interrupted run leaves nothing of ours loaded and keeps what it finished ------------------------------------------------------------------
class InterruptsOnTheSecondAttempt(ScriptedModel):
    def __init__(self, marker: Path):
        super().__init__([final("I do not know the code.")], name="scripted")
        self.marker = marker

    def generate(self, prompt, config=None, **kw):
        if self.marker.exists():  # attempts.jsonl exists once the first attempt was kept
            raise KeyboardInterrupt("operator pressed Ctrl-C")
        return super().generate(prompt, config, **kw)


def test_ctrl_c_during_e1_releases_our_ollama_model_closes_the_sinks_and_keeps_the_finished_attempt(host, monkeypatch):
    write_e0(host)
    released = []
    monkeypatch.setattr(drv, "release_ollama", lambda profile, label, *a, **k: released.append(label) or {"verified": True})
    model = InterruptsOnTheSecondAttempt(host / "diag_E1" / "attempts.jsonl")
    model.unload = lambda: released.append("model.unload")
    monkeypatch.setattr(models_mod, "build_model", lambda spec: model)
    with pytest.raises(KeyboardInterrupt):
        dgx.cmd_e1(args(host, attempts=3))
    assert released == ["after E1 (interrupted)", "model.unload"]  # the server-side model is released; the in-process one is unloaded
    assert sr.SINK["path"] is None and sr.TRACE["path"] is None
    kept = [json.loads(line) for line in (host / "diag_E1" / "attempts.jsonl").read_text().splitlines()]
    assert [r["attempt_id"] for r in kept] == ["tooluse-a1"] and kept[0]["success"] is False
    last = events(host / "diag_E1")[-1]
    assert last["event"] == "attempt_aborted" and last["attempt"] == "tooluse-a2"  # the interrupted attempt is marked, never silently dropped
    assert not (host / "diag_E1" / "result.json").exists()  # no summary is invented for a run that did not finish


# ---- a results folder under $HOME (the default for a checkout in a home directory) ---------------------------------------------------------------
def test_e2_runs_when_the_results_folder_is_under_the_home_directory(host, monkeypatch):
    """The rendered config is read back by the harness, so its workspace path must not be rewritten to `~/...` by the redactor (it was, and the run could not start)."""
    home = host / "home"
    out = home / "checkout" / "experiments" / "dgx" / "results" / "run"
    out.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    write_e0(out)
    scripted(monkeypatch, [act("delegate", {"worker": "reader", "task": "Read secret.txt and report the access code."}), read_secret(), final("It holds an access code."), final(f"The access code is {CODE}.")])
    assert dgx.cmd_e2(args(out)) == 0
    row = json.loads((out / "diag_E2" / "result.json").read_text())["summary"][0]
    assert (row["mechanics_demonstrated"], row["task_verified"]) == (True, True)
    rendered = next((out / "configs_rendered").glob("*.yaml")).read_text()
    assert str(home) in rendered and "workspace: \"~" not in rendered  # verbatim in the run folder...
    assert drv.redact(rendered).count(str(home)) == 0  # ...and the packager's pass still removes it from anything that is shared


# ---- E3: context transfer between two workers through the orchestrator ------------------------------------------------------------------------------
VALUE = dgx.E3_VALUE


def write_e2(out: Path, **over):
    report = {"experiment": "E2", "tag": TAG, "num_ctx": CTX, "summary": [{"attempt_id": "X2-a1", "mechanics_demonstrated": True}], **over}
    (out / "diag_E2").mkdir(parents=True, exist_ok=True)
    (out / "diag_E2" / "result.json").write_text(json.dumps(report))


def e3_script(*, writer_delegation=None, writer_calls=None, reader_task="Read source.txt and report the transfer value.", writer_content=f"{VALUE}\n"):
    """The turns of ONE shared scripted model, in the order the agents call it: orchestrator, reader (x2), orchestrator, writer (x2), orchestrator."""
    writer_delegation = writer_delegation or {"worker": "writer", "task": f"Create the file result.txt containing exactly {VALUE} on one line."}
    writer_calls = writer_calls or [act("file_operations", {"operation": "write", "path": "result.txt", "content": writer_content})]
    return [act("delegate", {"worker": "reader", "task": reader_task}),
            act("file_operations", {"operation": "read", "path": "source.txt"}), final(f"The transfer value is {VALUE}."),
            act("delegate", writer_delegation), *writer_calls, final("I created result.txt."),
            final(f"I wrote {VALUE} to result.txt.")]


def e3_row(host):
    return json.loads((host / "diag_E3" / "result.json").read_text())["summary"][0]


def e3_record(host):
    return json.loads((host / "diag_E3" / "attempts.jsonl").read_text().splitlines()[0])


def test_e3_exact_transfer_and_a_verified_result_are_both_recorded_and_the_arguments_are_in_the_trace(host, monkeypatch):
    write_e0(host)
    write_e2(host)
    model = scripted(monkeypatch, e3_script())
    assert dgx.cmd_e3(args(host)) == 0
    row = e3_row(host)
    assert (row["mechanics_demonstrated"], row["agent_completed"], row["task_verified"], row["context_transfer_exact"]) == (True, True, True, True)
    assert row["failed_checks"] == [] and row["transfer_failed_checks"] == []
    record = e3_record(host)
    first, second = record["delegations"]
    assert (first["worker"], first["delegation_id"], VALUE in first["report_output"]) == ("reader", "d1", True)
    assert (second["worker"], second["delegation_id"], VALUE in second["task"], second["context"]) == ("writer", "d2", True, None) and record["value_carried_in"] == ["task"]
    assert [(t["worker"], t["tool"], t["args"].get("operation")) for t in record["worker_tool_calls"]] == [("reader", "file_operations", "read"), ("writer", "file_operations", "write")]
    assert record["result_file"] == {"exists": True, "content": f"{VALUE}\n"} and record["writer_read_source"] is False
    # what the SQLite index shows about the transfer: the second delegate call's arguments, and the writer's own calls attributed to it
    trace = events(host / "diag_E3")
    starts = [e for e in trace if e["event"] == "tool_call_start" and e["tool"] == "delegate"]
    assert [s["args"]["worker"] for s in starts] == ["reader", "writer"] and VALUE in starts[1]["args"]["task"] and VALUE not in starts[0]["args"]["task"]
    sys.path.insert(0, str(ROOT / "experiments" / "history"))
    import history

    conn = history.connect(host / "h.db")
    history.import_source(conn, host / "diag_E3" / "trace.jsonl", "2026-10-28T00:00:00Z")
    sent = [dict(r) for r in conn.execute("SELECT json_extract(args, '$.worker') w, json_extract(args, '$.task') t, delegation_id d FROM tool_calls WHERE tool = 'delegate' ORDER BY started_seq")]
    assert [(r["w"], r["d"]) for r in sent] == [("reader", "d1"), ("writer", "d2")] and VALUE in sent[1]["t"]
    assert [dict(r) for r in conn.execute("SELECT agent_name a, delegation_id d FROM tool_calls WHERE role = 'worker' ORDER BY started_seq")] == [{"a": "reader", "d": "d1"}, {"a": "writer", "d": "d2"}]
    # independent contexts: the writer's model never saw the reader's task or tool call, the reader's never saw the writer's
    prompts = [str(c.prompt) for c in model.calls]
    writer_prompts = [p for p in prompts if "You are the worker agent 'writer'" in p]
    reader_prompts = [p for p in prompts if "You are the worker agent 'reader'" in p]
    assert writer_prompts and reader_prompts and record["worker_context_isolated"] is True
    assert not any("Read source.txt and report" in p or '"path": "source.txt"' in p for p in writer_prompts) and not any("result.txt" in p for p in reader_prompts)


def test_e3_exact_transfer_does_not_mean_the_task_succeeded(host, monkeypatch):
    write_e0(host)
    write_e2(host)
    scripted(monkeypatch, e3_script(writer_content="WRONG-0000\n"))
    assert dgx.cmd_e3(args(host)) == 0
    row = e3_row(host)
    assert row["context_transfer_exact"] is True and row["task_verified"] is False and row["failed_checks"] == ["result_file_exact"]


def test_e3_a_correct_file_that_bypassed_the_orchestrator_is_not_an_exact_transfer(host, monkeypatch):
    """The writer reads source.txt itself: the file is right, but nothing was carried through the delegation."""
    write_e0(host)
    write_e2(host)
    scripted(monkeypatch, e3_script(writer_delegation={"worker": "writer", "task": "Create result.txt containing the transfer value from source.txt."},
                                    writer_calls=[act("file_operations", {"operation": "read", "path": "source.txt"}),
                                                  act("file_operations", {"operation": "write", "path": "result.txt", "content": f"{VALUE}\n"})]))
    assert dgx.cmd_e3(args(host)) == 0
    row = e3_row(host)
    assert row["task_verified"] is True and row["context_transfer_exact"] is False
    # the value also never reached the writer's FIRST prompt (it only appeared later, as the result of its own read), which is why that check fails as well
    assert row["transfer_failed_checks"] == ["value_in_writer_delegation", "value_reached_writer_model", "writer_did_not_read_source"] and e3_record(host)["writer_read_source"] is True


def test_e3_the_value_may_travel_in_the_context_argument(host, monkeypatch):
    write_e0(host)
    write_e2(host)
    scripted(monkeypatch, e3_script(writer_delegation={"worker": "writer", "task": "Create result.txt with the value below on one line.", "context": f"value: {VALUE}"}))
    assert dgx.cmd_e3(args(host)) == 0
    assert (e3_row(host)["context_transfer_exact"], e3_row(host)["task_verified"]) == (True, True) and e3_record(host)["value_carried_in"] == ["context"]


def test_e3_a_value_known_before_the_reader_ran_is_flagged(host, monkeypatch):
    write_e0(host)
    write_e2(host)
    scripted(monkeypatch, e3_script(reader_task=f"Read source.txt; it should say {VALUE}."))
    assert dgx.cmd_e3(args(host)) == 0
    assert e3_row(host)["transfer_failed_checks"] == ["value_absent_before_the_reader"]


def test_e3_a_model_that_never_delegates_demonstrates_nothing(host, monkeypatch):
    write_e0(host)
    write_e2(host)
    scripted(monkeypatch, [final(f"I wrote {VALUE} to result.txt.")])
    assert dgx.cmd_e3(args(host)) == 1
    row = e3_row(host)
    assert (row["mechanics_demonstrated"], row["task_verified"], row["context_transfer_exact"]) == (False, False, False)
    assert "result_file_exists" in row["failed_checks"] and "reader_read_source" in row["transfer_failed_checks"]


@pytest.mark.parametrize("prepare, fragment", [
    (lambda out: None, "no E0 report"),
    (lambda out: write_e0(out), "no E2 result"),
    (lambda out: (write_e0(out), write_e2(out, tag="other:tag")), "E2 result was made for tag 'other:tag'"),
    (lambda out: (write_e0(out), write_e2(out, summary=[{"attempt_id": "X2-a1", "mechanics_demonstrated": False}])), "did not demonstrate the delegation mechanics"),
    (lambda out: (write_e0(out, gates_e1_e2=False), write_e2(out)), "E0 did not pass"),
])
def test_e3_refuses_to_run_until_its_prerequisites_are_met(host, monkeypatch, capsys, prepare, fragment):
    prepare(host)
    monkeypatch.setattr(models_mod, "build_model", lambda spec: pytest.fail("a model was loaded although E3 was refused"))
    assert dgx.cmd_e3(args(host)) == 2
    assert fragment in capsys.readouterr().out and not (host / "diag_E3").exists()


# ---- E3 review remediation (Opus review of the E3 verifier): B1 the report the orchestrator really received, B2 whole-token matching, the bypass check ----------------------------
def reader_report_turns(report: str):
    turns = e3_script()
    turns[2] = final(report)  # the reader worker's final answer, i.e. its report
    return turns


def test_e3_b1_the_report_is_read_from_the_delegate_result_not_the_500_character_cli_preview(host, monkeypatch):
    write_e0(host)
    write_e2(host)
    scripted(monkeypatch, reader_report_turns("Report: " + "padding " * 90 + f"The transfer value is {VALUE}."))  # the value sits after ~720 characters
    assert dgx.cmd_e3(args(host)) == 0
    first = e3_record(host)["delegations"][0]
    assert VALUE in first["report_output"] and VALUE not in first["report_output_cli_preview"] and len(first["report_output_cli_preview"]) < 600  # the preview WAS cut
    row = e3_row(host)
    assert row["context_transfer_exact"] is True and row["transfer_failed_checks"] == [] and row["task_verified"] is True


def test_e3_b1_a_value_beyond_what_the_orchestrator_received_is_not_credited_even_if_the_orchestrator_names_it(host, monkeypatch):
    """max_report_chars (3000 in the E3 config) cuts the report the orchestrator gets. If the value is past the cut, a later correct delegation did not come from the report."""
    write_e0(host)
    write_e2(host)
    scripted(monkeypatch, reader_report_turns("Report: " + "padding " * 450 + f"The transfer value is {VALUE}."))  # ~3,600 characters
    assert dgx.cmd_e3(args(host)) == 0
    first = e3_record(host)["delegations"][0]
    assert first["report_truncated"] is True and VALUE not in first["report_output"]
    assert e3_row(host)["transfer_failed_checks"] == ["reader_report_has_value"] and e3_row(host)["context_transfer_exact"] is False


def test_e3_b1_an_altered_value_in_the_reader_report_fails_that_check(host, monkeypatch):
    write_e0(host)
    write_e2(host)
    scripted(monkeypatch, reader_report_turns(f"The transfer value is {VALUE}9."))
    assert dgx.cmd_e3(args(host)) == 0
    assert e3_row(host)["transfer_failed_checks"] == ["reader_report_has_value"]


@pytest.mark.parametrize("text, expected", [
    (VALUE, True), (f"{VALUE}.", True), (f'"{VALUE}"', True), (f"value: {VALUE}\n", True), (f"({VALUE})", True), (f"a {VALUE}, b", True),
    (f"{VALUE}9", False), (f"X{VALUE}", False), (f"{VALUE}-b", False), (f"{VALUE}_x", False), (f"a-{VALUE}", False), (f"{VALUE.lower()}", False), ("TRANSFER-417", False), ("", False), (None, False),
])
def test_e3_b2_the_value_is_matched_as_a_whole_token(text, expected):
    assert dgx.has_token(text, VALUE) is expected


@pytest.mark.parametrize("altered", [f"{VALUE}9", f"X{VALUE}", f"{VALUE}-b", f"{VALUE}_x"])
def test_e3_b2_an_altered_value_is_not_an_exact_transfer_even_when_the_file_is_right(host, monkeypatch, altered):
    write_e0(host)
    write_e2(host)
    scripted(monkeypatch, e3_script(writer_delegation={"worker": "writer", "task": f"Create the file result.txt containing exactly {altered} on one line."}))  # the writer still writes the right value
    assert dgx.cmd_e3(args(host)) == 0
    row = e3_row(host)
    assert row["task_verified"] is True  # result.txt is exactly right
    assert row["context_transfer_exact"] is False and row["transfer_failed_checks"] == ["value_in_writer_delegation", "value_reached_writer_model"]


def test_e3_b2_the_writers_own_later_output_does_not_count_as_the_value_reaching_it(host, monkeypatch):
    """The writer's second prompt contains its own earlier `Action Input` with the value; only its FIRST prompt shows what the delegation gave it."""
    write_e0(host)
    write_e2(host)
    model = scripted(monkeypatch, e3_script(writer_delegation={"worker": "writer", "task": "Create result.txt containing the transfer value."}))
    assert dgx.cmd_e3(args(host)) == 0
    writer_prompts = [str(c.prompt) for c in model.calls if "You are the worker agent 'writer'" in str(c.prompt)]
    assert dgx.has_token(writer_prompts[-1]) and not dgx.has_token(writer_prompts[0])  # present later (its own text), absent at the start
    assert "value_reached_writer_model" in e3_row(host)["transfer_failed_checks"]


def test_e3_b2_the_value_next_to_punctuation_still_counts(host, monkeypatch):
    write_e0(host)
    write_e2(host)
    scripted(monkeypatch, e3_script(writer_delegation={"worker": "writer", "task": f'Create result.txt containing "{VALUE}".'}))
    assert dgx.cmd_e3(args(host)) == 0
    assert e3_row(host)["context_transfer_exact"] is True


# ---- the writer's access to the source ------------------------------------------------------------------------------------------------------------------------
def test_e3_names_only_operations_on_the_source_are_recorded_but_are_not_a_bypass(host, monkeypatch):
    """effGen's search is a filename glob and list/metadata return names and sizes: none of them can show the writer the value."""
    write_e0(host)
    write_e2(host)
    scripted(monkeypatch, e3_script(writer_calls=[act("file_operations", {"operation": "list", "path": "."}), act("file_operations", {"operation": "search", "path": ".", "pattern": "*.txt"}),
                                                  act("file_operations", {"operation": "metadata", "path": "source.txt"}),
                                                  act("file_operations", {"operation": "write", "path": "result.txt", "content": f"{VALUE}\n"})]))
    assert dgx.cmd_e3(args(host)) == 0
    row, record = e3_row(host), e3_record(host)
    assert row["context_transfer_exact"] is True and row["task_verified"] is True and record["writer_read_source"] is False
    writer = [t for t in record["worker_tool_calls"] if t["worker"] == "writer"]
    assert [(t["args"]["operation"], t["names_source"], t["content_access"], t["observed_value"]) for t in writer] == [
        ("list", False, False, False), ("search", False, False, False), ("metadata", True, False, False), ("write", False, False, False)]


def test_e3_convert_and_read_of_the_source_are_a_bypass(host, monkeypatch):
    """`convert` reads the source and writes a copy; reading the copy (source.json) names no `source.txt`, so a path check on source.txt alone would miss it."""
    write_e0(host)
    write_e2(host)
    scripted(monkeypatch, e3_script(writer_calls=[act("file_operations", {"operation": "convert", "path": "source.txt", "target_format": "json"}),
                                                  act("file_operations", {"operation": "read", "path": "source.json"}),
                                                  act("file_operations", {"operation": "write", "path": "result.txt", "content": f"{VALUE}\n"})]))
    assert dgx.cmd_e3(args(host)) == 0
    row, record = e3_row(host), e3_record(host)
    assert "writer_did_not_read_source" in row["transfer_failed_checks"] and row["context_transfer_exact"] is False and record["writer_read_source"] is True
    writer = [t for t in record["worker_tool_calls"] if t["worker"] == "writer"]
    assert [(t["args"]["operation"], t["content_access"]) for t in writer[:2]] == [("convert", True), ("read", True)]  # source.json also names `source.`


def log_entry(delegation_id, tool, args, visible="", **extra):
    return {"tool": tool, "delegation_id": delegation_id, "args": args, "success": True, "visible": visible, **extra}


def judge(extra_writer_calls):
    """e3_judgements on a hand-made log in which everything passes except what `extra_writer_calls` adds: for routes the scripted tools cannot produce."""
    delegate = lambda did, worker, **kw: log_entry(did, "delegate", {"worker": worker, "task": kw.get("task", "t")}, kw.get("visible", ""), dispatched=True)  # noqa: E731
    report = json.dumps({"worker": "reader", "status": "completed", "output": f"The transfer value is {VALUE}.", "truncated": False})
    log = [log_entry("d1", "file_operations", {"operation": "read", "path": "source.txt"}, f"The transfer value is {VALUE}."), delegate("d1", "reader", visible=report),
           delegate("d2", "writer", task=f"Create result.txt with {VALUE}"), *extra_writer_calls]
    calls = [{"prompt": f"You are the worker agent 'writer'. Task: create result.txt with {VALUE}"}]
    return dgx.e3_judgements(log, {"success": True, "workers": [{"status": "completed"}, {"status": "completed"}]}, calls, Path("/nonexistent"), {}, {})


def failed_transfer(judged):
    return [c["name"] for c in judged["context_transfer"]["checks"] if not c["passed"]]


def test_e3_a_value_returned_to_the_writer_by_any_other_file_is_a_bypass_but_its_own_result_file_is_not():
    other = log_entry("d2", "file_operations", {"operation": "read", "path": "notes.dat"}, f"copied: {VALUE}")
    own = log_entry("d2", "file_operations", {"operation": "read", "path": "result.txt"}, f"{VALUE}\n")
    wrote = log_entry("d2", "file_operations", {"operation": "write", "path": "result.txt", "content": VALUE}, f"wrote result.txt ({VALUE})")
    assert failed_transfer(judge([other])) == ["writer_did_not_read_source"]
    assert failed_transfer(judge([own, wrote])) == []
    assert failed_transfer(judge([])) == []


def test_e3_the_reader_must_have_been_shown_the_value_by_its_read():
    judged = dgx.e3_judgements([log_entry("d1", "file_operations", {"operation": "read", "path": "source.txt"}, "empty file"),
                                log_entry("d1", "delegate", {"worker": "reader", "task": "t"}, json.dumps({"status": "completed", "output": VALUE}), dispatched=True),
                                log_entry("d2", "delegate", {"worker": "writer", "task": VALUE}, json.dumps({"status": "completed", "output": "ok"}), dispatched=True)],
                               {"success": True, "workers": []}, [{"prompt": f"You are the worker agent 'writer'. {VALUE}"}], Path("/nonexistent"), {}, {})
    assert "reader_read_source" in failed_transfer(judged)
