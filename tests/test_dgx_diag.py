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
