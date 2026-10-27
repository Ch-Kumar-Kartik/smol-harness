"""SQLite execution history (experiments/history/history.py) and the instrumentation changes behind it: tool_call_start, the recorded generation-to-tool link, private prompt capture.

CPU only, no network, no real model. Real evidence is produced by scripted models through the real harness, the real file tool and the DGX E2 path; hand-written JSONL is used
only where a real run cannot produce the situation (a half-written line, an older trace format, odd usage metadata). One test really kills a process in the middle of a tool call.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import ScriptedModel
from effgen import AgentConfig

import smol_harness.models as models_mod
from smol_harness.harness import build_harness, initialize_runtime
from smol_harness.tools import build_tools

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "history"))
sys.path.insert(0, str(ROOT / "experiments" / "dgx"))
import dgx_diag as dgx  # noqa: E402
import history  # noqa: E402

drv, sr = dgx.drv, dgx.sr
CODE = "7391-QX"
TAG, CTX = "smol-diag:8k", 8192


def act(tool: str, args: dict) -> str:
    return f"Thought: next.\nAction: {tool}\nAction Input: {json.dumps(args)}"


def final(text: str) -> str:
    return f"Thought: done.\nFinal Answer: {text}"


def delegate(task: str, worker: str = "reader") -> str:
    return act("delegate", {"worker": worker, "task": task})


def read_secret() -> str:
    return act("file_operations", {"operation": "read", "path": "secret.txt"})


# ---- fixtures and helpers -----------------------------------------------------------------------------------------------------------------
@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "nvidia_smi_gpus", drv.nvidia_smi_gpus)
    monkeypatch.setattr(sr, "TRACE", sr.fresh_trace())
    monkeypatch.setitem(sr.SINK, "path", None)
    monkeypatch.setattr(drv, "release_ollama", lambda *a, **k: {"verified": True, "reason": "test double"})
    monkeypatch.setattr(drv, "snapshot_ollama", lambda *a, **k: None)
    return tmp_path


@pytest.fixture
def db(tmp_path):
    conn = history.connect(tmp_path / "history" / "h.db")
    yield conn
    conn.close()


def run_args(out: Path, capture: bool = False, attempts: int = 1):
    return SimpleNamespace(out=str(out), tag=TAG, num_ctx=CTX, attempts=attempts, capture_prompts=capture)


def e2(host, monkeypatch, turns, capture=True):
    """The DGX E2 path with a scripted model: the real CLI, harness, delegation and file tool. Returns the evidence folder (diag_E2)."""
    (host / "diag_E0").mkdir(parents=True, exist_ok=True)
    (host / "diag_E0" / "e0.json").write_text(json.dumps({"experiment": "E0", "tag": TAG, "num_ctx": CTX, "gates_e1_e2": True, "gating": {}}))
    model = ScriptedModel(turns, name="scripted")
    model.unload = lambda: None
    monkeypatch.setattr(models_mod, "build_model", lambda spec: model)
    dgx.cmd_e2(run_args(host, capture))
    return host / "diag_E2"


GOOD = [delegate("Read secret.txt and report the access code."), read_secret(), final("The file holds an access code."), final(f"The access code is {CODE}.")]


def load(conn, folder: Path):
    return history.import_source(conn, Path(folder) / "trace.jsonl", "2026-10-27T00:00:00Z")


def one(conn, sql, *params):
    return conn.execute(sql, params).fetchone()


def table_dump(conn) -> dict:
    """Every table except the import timestamp, to compare two databases built differently from the same evidence."""
    dump = {}
    for name in ("sessions", "messages", "tool_calls", "events", "issues"):
        order = "ROWID" if name == "issues" else "1"
        dump[name] = [tuple(r) for r in conn.execute(f"SELECT * FROM {name} ORDER BY {order}")] if name != "issues" else [tuple(r)[1:] for r in conn.execute("SELECT * FROM issues ORDER BY kind, source, line_no, detail")]
    dump["imports"] = [tuple(r)[:4] for r in conn.execute("SELECT * FROM imports ORDER BY invocation")]
    return dump


def lines_of(path: Path) -> list:
    return path.read_text().splitlines(keepends=True)


# ---- hand-written evidence (only where a real run cannot produce it) -----------------------------------------------------------------------
INV = "inv000000001"


def ev(seq, event, attempt="a1", **fields):
    return {"seq": seq, "t_utc": f"2026-10-27T10:00:{seq:02d}Z", "run_id": "run-1", "stage": "S", "invocation": INV, "attempt": attempt, "event": event, **fields}


def gen(seq, role="single", ordinal=1, delegation=None, meta=None, **fields):
    base = dict(provider="hf", model="m", role=role, delegation_id=delegation, generation_in_agent=ordinal, raw_output="Thought: x", consumed_output="Thought: x", consumed_status="established",
                stop_handling="provider", stop_sequences=["\nObservation:"], stop_sequences_local=None, discarded_chars=0, finish="stop", seconds=0.1, tokens=3, prompt_chars=10,
                prompt_sha256="0" * 64, limits={"max_tokens": 768}, reasoning={"reported": []}, provider_metadata=meta)
    return ev(seq, "generation", **{**base, **fields})


def write_lines(path: Path, events: list, tail: str = "", raw: list | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for e in events:
            handle.write(json.dumps(e) + "\n")
        for line in raw or []:
            handle.write(line)
        handle.write(tail)
    return path


# ---- 1. a complete orchestrator-worker experiment, reconstructed -------------------------------------------------------------------------------
def test_the_complete_timeline_of_a_scripted_orchestrator_worker_experiment(host, monkeypatch, db):
    folder = e2(host, monkeypatch, GOOD)
    summary = load(db, folder)
    assert summary["issues"] == 0 and len(summary["imported"]) == 1
    session = one(db, "SELECT * FROM sessions")
    assert session["status"] == "finished" and session["agent_completed"] == 1 and session["task_verified"] == 1 and session["session_id"].endswith(":X2-plain-a1")
    messages = db.execute("SELECT * FROM messages ORDER BY seq").fetchall()
    assert [(m["role"], m["delegation_id"], m["agent_name"], m["generation_in_agent"]) for m in messages] == [("orchestrator", None, None, 1), ("worker", "d1", "reader", 1), ("worker", "d1", "reader", 2), ("orchestrator", None, None, 2)]
    tools = {t["tool"]: t for t in db.execute("SELECT * FROM tool_calls")}
    assert tools["delegate"]["status"] == "completed" and tools["delegate"]["delegation_id"] == tools["delegate"]["delegation_id_expected"] == "d1" and tools["delegate"]["dispatched"] == 1
    assert json.loads(tools["delegate"]["args"])["task"] == "Read secret.txt and report the access code."
    assert tools["delegate"]["requested_by_message_id"] == messages[0]["message_id"] and tools["delegate"]["link_basis"] == "recorded"
    assert (tools["file_operations"]["role"], tools["file_operations"]["delegation_id"], tools["file_operations"]["agent_name"]) == ("worker", "d1", "reader")
    assert tools["file_operations"]["requested_by_message_id"] == messages[1]["message_id"] and CODE in tools["file_operations"]["result"]
    text = history.timeline_lines(db, session["session_id"])
    joined = "\n".join(text)
    order = [joined.index(marker) for marker in ("MESSAGE " + messages[0]["message_id"], "TOOL START delegate", "MESSAGE " + messages[1]["message_id"], "TOOL START file_operations",
                                                 "TOOL END   file_operations", "MESSAGE " + messages[2]["message_id"], "TOOL END   delegate", "MESSAGE " + messages[3]["message_id"], "=== verdict")]
    assert order == sorted(order) and len(set(order)) == len(order)
    assert any(line.startswith("    #") and "[worker d1 #1] MESSAGE" in line for line in text)  # the worker's turns are nested under the delegation
    assert "task_verified=yes" in joined and f"The access code is {CODE}." in joined and "UNFINISHED" not in joined


def test_tool_call_start_comes_first_and_shares_the_id_of_its_completion(host, monkeypatch):
    folder = e2(host, monkeypatch, GOOD)
    trace = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
    starts = {e["seq"]: e for e in trace if e["event"] == "tool_call_start"}
    ends = [e for e in trace if e["event"] == "tool_call"]
    assert len(starts) == len(ends) == 2
    for end in ends:
        assert end["start_seq"] in starts and end["seq"] > end["start_seq"] and end["tool_call_id"] == f"{end['invocation']}:{end['start_seq']}"
        assert starts[end["start_seq"]]["args"] == end["args"] and starts[end["start_seq"]]["requested_by_generation_seq"] == end["requested_by_generation_seq"]
    delegate_start = next(e for e in starts.values() if e["tool"] == "delegate")
    assert delegate_start["delegation_id_expected"] == "d1" and "delegation_id" not in delegate_start and delegate_start["args"]["worker"] == "reader"


# ---- 2. a tool that never completes (the process dies) --------------------------------------------------------------------------------------
KILLED_CHILD = r'''
import asyncio, json, sys
from pathlib import Path
root, out = sys.argv[1], Path(sys.argv[2])
sys.path[:0] = [root + "/tests", root + "/experiments/rtx4070"]
from conftest import ScriptedModel
from effgen import AgentConfig
import stage_runner as sr
from smol_harness.harness import build_harness, initialize_runtime
from smol_harness.tools import WorkspaceFileOperations, build_tools

async def hang(self, **kwargs):
    await asyncio.sleep(120)

WorkspaceFileOperations.execute = hang
initialize_runtime()
out.mkdir(parents=True, exist_ok=True)
sr.TRACE.clear()
sr.TRACE.update(sr.fresh_trace(path=out / "trace.jsonl", stage="K", run_id="killed-run", invocation="kill00000001"))
workspace = out / "ws"
workspace.mkdir()
(workspace / "secret.txt").write_text("code\n")
def act(tool, args): return "Thought: n.\nAction: %s\nAction Input: %s" % (tool, json.dumps(args))
model = ScriptedModel([act("delegate", {"worker": "reader", "task": "Read secret.txt and report the access code."}), act("file_operations", {"operation": "read", "path": "secret.txt"}), "Thought: d\nFinal Answer: x"], name="scripted")
model.generate = sr.traced_generate(model.generate, "scripted", [])
cfg = lambda n, tools: AgentConfig(name=n, model=model, tools=tools, max_iterations=6, tool_calling_mode="react", raise_on_error=False, enable_memory=False, capability_probe=False, enable_sub_agents=False)
harness = build_harness(cfg("orchestrator", []), {"reader": cfg("reader", build_tools(["file_operations"], workspace))}, {"reader": "reads"})
with sr.attempt_scope("a1", "get the code", "orchestrator"), sr.tool_log(), harness:
    harness.run("get the code")
'''


def test_a_process_killed_inside_a_tool_call_leaves_an_unfinished_call_with_its_arguments(tmp_path, db):
    script = tmp_path / "child.py"
    script.write_text(KILLED_CHILD)
    out = tmp_path / "killed"
    child = subprocess.Popen([sys.executable, str(script), str(ROOT), str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            text = (out / "trace.jsonl").read_text() if (out / "trace.jsonl").exists() else ""
            if '"tool": "file_operations"' in text and '"event": "tool_call_start"' in text:
                break
            time.sleep(0.1)
        else:
            pytest.fail("the child never reached the file tool")
        child.kill()  # SIGKILL: nothing can be written after this
    finally:
        child.kill()
        child.wait()
    summary = load(db, out)
    session = one(db, "SELECT * FROM sessions")
    assert session["status"] == "unfinished" and session["end_seq"] is None
    tools = {t["tool"]: t for t in db.execute("SELECT * FROM tool_calls")}
    assert {t["status"] for t in tools.values()} == {"unfinished"} and all(t["completed_seq"] is None and t["success"] is None for t in tools.values())
    assert json.loads(tools["file_operations"]["args"]) == {"operation": "read", "path": "secret.txt"}  # the arguments survived
    d = tools["delegate"]
    assert json.loads(d["args"]) == {"worker": "reader", "task": "Read secret.txt and report the access code."}  # so did the delegated task text
    assert d["delegation_id"] is None and d["delegation_id_expected"] == "d1" and d["dispatched"] is None  # not confirmed: only expected
    assert (tools["file_operations"]["role"], tools["file_operations"]["delegation_id"], tools["file_operations"]["agent_name"]) == ("worker", "d1", "reader")  # worker still resolved
    assert tools["file_operations"]["requested_by_message_id"] and d["requested_by_message_id"]
    counts = {(r["tool"], r["role"]): (r["completed"], r["failed"], r["unfinished"]) for r in db.execute("SELECT * FROM tool_counts")}
    assert counts == {("delegate", "orchestrator"): (0, 0, 1), ("file_operations", "worker"): (0, 0, 1)}  # unfinished calls are not executions
    text = "\n".join(history.timeline_lines(db, session["session_id"]))
    assert text.count("UNFINISHED") == 2 and "status=unfinished" in text and summary["issues"] == 0


# ---- 3. attribution of generations to tool calls ---------------------------------------------------------------------------------------------
def harness_run(directory: Path, model, workers=True, attempt="a1"):
    """Scripted agents through the real harness and file tool, writing the real trace; returns the folder."""
    initialize_runtime()
    folder = directory / "stage"
    folder.mkdir(exist_ok=True)
    workspace = directory / "ws"
    workspace.mkdir(exist_ok=True)
    (workspace / "secret.txt").write_text(f"The access code is {CODE}.\n")
    sr.TRACE.clear()
    sr.TRACE.update(sr.fresh_trace(path=folder / "trace.jsonl", stage="T", run_id="run-1", invocation=INV))
    sr.SINK["path"] = folder / "attempts.jsonl"
    cfg = lambda n, tools: AgentConfig(name=n, model=model, tools=tools, max_iterations=8, tool_calling_mode="react", raise_on_error=False, enable_memory=False, capability_probe=False, enable_sub_agents=False)
    model.generate = sr.traced_generate(model.generate, "scripted", [])
    harness = build_harness(cfg("orchestrator", build_tools(["file_operations"], workspace) if not workers else []), {"reader": cfg("reader", build_tools(["file_operations"], workspace))} if workers else {}, {"reader": "reads"})
    with sr.attempt_scope(attempt, "task", "orchestrator" if workers else "single"), sr.tool_log(), harness:
        harness.run("task")
    sr.keep({"attempt_id": attempt, "kind": "t", "success": True, "final_answer": "done", "verification": sr.verdict(True, [("c", True, "x")])})
    return folder


class FailsOnce(ScriptedModel):
    """Its first call raises (effGen retries after a pause); the retry and everything after it succeed."""

    def __init__(self, turns):
        super().__init__(turns, name="scripted")
        self.raised = False

    def generate(self, prompt, config=None, **kw):
        if not self.raised:
            self.raised = True
            raise RuntimeError("backend hiccup")
        return super().generate(prompt, config, **kw)


def test_a_tool_call_is_attributed_to_the_successful_retry_not_to_the_failed_generation(host, db):
    folder = harness_run(host, FailsOnce([read_secret(), final(f"The code is {CODE}.")]), workers=False)
    load(db, folder)
    messages = db.execute("SELECT * FROM messages ORDER BY seq").fetchall()
    assert [(m["status"], m["generation_in_agent"]) for m in messages] == [("error", 1), ("ok", 2), ("ok", 3)]
    assert "backend hiccup" in messages[0]["error"] and messages[0]["prompt_sha256"] and messages[0]["raw_output"] is None
    tool = one(db, "SELECT * FROM tool_calls")
    assert tool["requested_by_message_id"] == messages[1]["message_id"] and tool["link_basis"] == "recorded" and tool["generation_in_agent"] == 2


def test_two_sequential_delegations_each_link_to_their_own_generations(host, db):
    turns = [delegate("first job"), read_secret(), final("a"), delegate("second job"), read_secret(), final("b"), final("done")]
    load(db, harness_run(host, ScriptedModel(turns, name="scripted")))
    delegates = db.execute("SELECT * FROM delegations ORDER BY started_seq").fetchall()
    assert [(d["delegation_id"], d["worker"], d["task"], d["confirmed"]) for d in delegates] == [("d1", "reader", "first job", 1), ("d2", "reader", "second job", 1)]
    by_id = {m["message_id"]: m for m in db.execute("SELECT * FROM messages")}
    assert [by_id[d["requested_by_message_id"]]["generation_in_agent"] for d in delegates] == [1, 2]  # the 1st and 2nd ORCHESTRATOR generation
    for t in db.execute("SELECT * FROM tool_calls WHERE tool = 'file_operations'"):
        message = by_id[t["requested_by_message_id"]]
        assert (message["role"], message["delegation_id"]) == ("worker", t["delegation_id"]) and t["agent_name"] == "reader"


def test_a_refused_delegation_owns_no_id_and_does_not_steal_the_next_ones(host, db):
    turns = [act("delegate", {"worker": "reader"}), delegate("the real job"), read_secret(), final("a"), final("done")]  # the first call lacks `task`: effGen refuses it
    load(db, harness_run(host, ScriptedModel(turns, name="scripted")))
    refused, real = db.execute("SELECT * FROM tool_calls WHERE tool = 'delegate' ORDER BY started_seq").fetchall()
    assert (refused["status"], refused["dispatched"], refused["delegation_id"]) == ("failed", 0, None)
    assert (real["status"], real["dispatched"], real["delegation_id"]) == ("completed", 1, "d1")
    worker = one(db, "SELECT * FROM messages WHERE role = 'worker'")
    assert worker["agent_name"] == "reader" and one(db, "SELECT COUNT(*) FROM issues")[0] == 0


def test_several_tool_calls_from_one_generation_share_it_and_a_missing_generation_is_not_invented(tmp_path, db):
    events = [ev(1, "attempt_start", task="t"), gen(2),
              ev(3, "tool_call_start", tool="file_operations", role="single", generation_in_agent=1, delegation_id=None, args={"path": "a"}, requested_by_generation_seq=2),
              ev(4, "tool_call", tool_call_id=f"{INV}:3", start_seq=3, tool="file_operations", role="single", generation_in_agent=1, args={"path": "a"}, success=True, exception=False, result="A", visible="A",
                 seconds=0.1, requested_by_generation_seq=2),
              ev(5, "tool_call_start", tool="file_operations", role="single", generation_in_agent=1, delegation_id=None, args={"path": "b"}, requested_by_generation_seq=2),
              ev(6, "tool_call", tool_call_id=f"{INV}:5", start_seq=5, tool="file_operations", role="single", generation_in_agent=1, args={"path": "b"}, success=True, exception=False, result="B", visible="B",
                 seconds=0.1, requested_by_generation_seq=2),
              ev(7, "tool_call_start", tool="shell", role="single", generation_in_agent=1, delegation_id=None, args={"command": "ls"}, requested_by_generation_seq=99),  # names a generation that is not there
              ev(8, "attempt_end", kind="t", success=True, agent_completed=True, task_verified=True)]
    load(db, write_lines(tmp_path / "t" / "trace.jsonl", events).parent)
    links = {t["tool_call_id"]: (t["requested_by_message_id"], t["link_basis"]) for t in db.execute("SELECT * FROM tool_calls")}
    assert links == {f"{INV}:3": (f"{INV}:2", "recorded"), f"{INV}:5": (f"{INV}:2", "recorded"), f"{INV}:7": (None, None)}
    assert one(db, "SELECT kind, detail FROM issues WHERE kind = 'requested_by_unresolved'") is not None


def test_a_legacy_trace_is_imported_without_inventing_what_it_never_recorded(tmp_path, db):
    legacy = [ev(1, "attempt_start", task="t"), gen(2, ordinal=1), gen(3, ordinal=2),
              ev(4, "tool_call", tool="file_operations", role="single", generation_in_agent=2, delegation_id=None, args={"path": "a"}, success=True, exception=False, result="A", visible="A", seconds=0.1),
              ev(5, "tool_call", tool="file_operations", role="single", generation_in_agent=7, delegation_id=None, args={"path": "b"}, success=False, exception=False, error="no", result=None, visible="no", seconds=0.1),
              gen(6, role="orchestrator", ordinal=1, delegation=None),  # a delegate call has the orchestrator's role but the NEW delegation id; its generation has none
              ev(7, "tool_call", tool="delegate", role="orchestrator", generation_in_agent=1, delegation_id="d1", args={"worker": "reader", "task": "t"}, success=True, exception=False, result="{}", visible="{}",
                 seconds=0.1, dispatched=True, delegation_id_reported="d1"),
              ev(8, "attempt_end", kind="t", success=False, agent_completed=True, task_verified=False)]
    folder = tmp_path / "legacy"
    without_identity = json.dumps({"event": "tool_call", "tool": "x", "t_utc": "2026-01-01T00:00:00Z"}) + "\n"
    write_lines(folder / "trace.jsonl", legacy, raw=[without_identity])
    summary = load(db, folder)
    tools = {t["tool_call_id"]: t for t in db.execute("SELECT * FROM tool_calls")}
    assert set(tools) == {f"{INV}:4", f"{INV}:5", f"{INV}:7"} and all(t["started_seq"] is None and t["started_utc"] is None for t in tools.values())  # no start was ever recorded: none is made up
    assert (tools[f"{INV}:4"]["status"], tools[f"{INV}:4"]["requested_by_message_id"], tools[f"{INV}:4"]["link_basis"]) == ("completed", f"{INV}:3", "derived-ordinal")
    assert (tools[f"{INV}:5"]["status"], tools[f"{INV}:5"]["requested_by_message_id"], tools[f"{INV}:5"]["link_basis"]) == ("failed", None, None)  # no generation 7: not linked
    assert (tools[f"{INV}:7"]["requested_by_message_id"], tools[f"{INV}:7"]["link_basis"], tools[f"{INV}:7"]["delegation_id"]) == (f"{INV}:6", "derived-ordinal", "d1")  # matched on role and ordinal, not on the id
    assert {r["kind"] for r in db.execute("SELECT kind FROM issues")} == {"missing_identity", "attempt_record_missing"} and summary["issues"] == 2  # both stated, neither papered over
    assert one(db, "SELECT final_answer, verification FROM sessions")[:] == (None, None)  # no attempts.jsonl: no verification details are made up
    assert one(db, "SELECT COUNT(*) FROM messages WHERE prompt_text IS NOT NULL")[0] == 0


# ---- 4. aborted and failed sessions -------------------------------------------------------------------------------------------------------------
class InterruptsOnTheSecondAttempt(ScriptedModel):
    def __init__(self, marker: Path):
        super().__init__([final("I do not know the code.")], name="scripted")
        self.marker = marker

    def generate(self, prompt, config=None, **kw):
        if self.marker.exists():
            raise KeyboardInterrupt("operator pressed Ctrl-C")
        return super().generate(prompt, config, **kw)


def test_an_aborted_attempt_and_a_finished_but_failed_one_are_both_accurate(host, monkeypatch, db):
    (host / "diag_E0").mkdir()
    (host / "diag_E0" / "e0.json").write_text(json.dumps({"tag": TAG, "num_ctx": CTX, "gates_e1_e2": True, "gating": {}}))
    model = InterruptsOnTheSecondAttempt(host / "diag_E1" / "attempts.jsonl")
    model.unload = lambda: None
    monkeypatch.setattr(models_mod, "build_model", lambda spec: model)
    with pytest.raises(KeyboardInterrupt):
        dgx.cmd_e1(run_args(host, attempts=3))
    load(db, host / "diag_E1")
    first, second = db.execute("SELECT * FROM sessions ORDER BY start_seq").fetchall()
    assert (first["status"], first["agent_completed"], first["task_verified"]) == ("finished", 1, 0)  # finished, and the task was NOT achieved
    assert json.loads(first["verification"])["task_verified"] is False and first["final_answer"] == "I do not know the code."
    assert (second["status"], second["task_verified"], second["end_seq"] is not None) == ("aborted", None, True) and "KeyboardInterrupt" in second["error"]
    failed = one(db, "SELECT * FROM messages WHERE session_id = ?", second["session_id"])
    assert failed["status"] == "error" and failed["raw_output"] is None
    listing = history.cmd_sessions(SimpleNamespace(db=db_path(db)))
    assert listing == 0


def db_path(conn) -> Path:
    return Path(conn.execute("PRAGMA database_list").fetchone()["file"])


# ---- 5. growing files, repeated imports, interrupted imports ----------------------------------------------------------------------------------
def test_importing_a_growing_trace_and_a_partial_last_line_keeps_statuses_accurate(host, monkeypatch, db, tmp_path):
    folder = e2(host, monkeypatch, GOOD, capture=False)
    trace_lines, attempt_lines = lines_of(folder / "trace.jsonl"), lines_of(folder / "attempts.jsonl")
    live = tmp_path / "live"
    live.mkdir()
    end_index = next(i for i, line in enumerate(trace_lines) if '"event": "attempt_end"' in line)
    # 1. the run is half done: no end line yet
    (live / "trace.jsonl").write_text("".join(trace_lines[:end_index - 3]))
    load(db, live)
    assert one(db, "SELECT status FROM sessions")["status"] == "unfinished"
    half = one(db, "SELECT COUNT(*) FROM messages")[0] + one(db, "SELECT COUNT(*) FROM tool_calls")[0]
    # 2. everything but the end line, which is half written; the attempt record is already there (keep() writes it first)
    (live / "trace.jsonl").write_text("".join(trace_lines[:end_index]) + trace_lines[end_index][:40])
    (live / "attempts.jsonl").write_text("".join(attempt_lines))
    summary = load(db, live)
    assert one(db, "SELECT status FROM sessions")["status"] == "unfinished" and one(db, "SELECT COUNT(*) FROM issues WHERE kind = 'partial_tail'")[0] == 1
    assert summary["partial_tail_bytes"] == 40 and one(db, "SELECT COUNT(*) FROM issues WHERE kind = 'malformed'")[0] == 0  # not mistaken for a corrupt line
    # 3. complete
    (live / "trace.jsonl").write_text("".join(trace_lines))
    load(db, live)
    session = one(db, "SELECT * FROM sessions")
    assert session["status"] == "finished" and session["task_verified"] == 1 and one(db, "SELECT COUNT(*) FROM issues")[0] == 0
    assert one(db, "SELECT COUNT(*) FROM sessions")[0] == 1 and one(db, "SELECT COUNT(*) FROM messages")[0] + one(db, "SELECT COUNT(*) FROM tool_calls")[0] > half


def test_repeating_an_import_changes_nothing_and_a_rebuild_gives_the_same_database(host, monkeypatch, db, tmp_path):
    folder = e2(host, monkeypatch, GOOD)
    load(db, folder)
    before = table_dump(db)
    again = load(db, folder)
    assert again["imported"] == [] and len(again["unchanged"]) == 1 and table_dump(db) == before
    copy = tmp_path / "moved"
    shutil.copytree(folder, copy)  # the same evidence from another place is the same session
    assert load(db, copy)["imported"] == [] and table_dump(db) == before
    other = history.connect(tmp_path / "rebuilt.db")
    load(other, folder)
    assert table_dump(other) == before
    other.close()


def test_an_older_copy_of_a_trace_never_replaces_a_newer_import(host, monkeypatch, db, tmp_path):
    folder = e2(host, monkeypatch, GOOD)
    lines = lines_of(folder / "trace.jsonl")
    old = tmp_path / "old"
    old.mkdir()
    (old / "trace.jsonl").write_text("".join(lines[:5]))
    load(db, folder)
    before = table_dump(db)
    summary = load(db, old)
    assert summary["skipped_older"] and summary["imported"] == []
    after = table_dump(db)
    assert {k: v for k, v in after.items() if k != "issues"} == {k: v for k, v in before.items() if k != "issues"}
    assert one(db, "SELECT COUNT(*) FROM issues WHERE kind = 'older_copy_skipped'")[0] == 1


def test_a_malformed_complete_line_is_recorded_and_the_rest_is_still_imported(host, monkeypatch, db, tmp_path):
    folder = e2(host, monkeypatch, GOOD)
    lines = lines_of(folder / "trace.jsonl")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "trace.jsonl").write_text("".join(lines[:3]) + '{"seq": 4, "event": "gener\n' + "[1, 2]\n" + "".join(lines[3:]))
    (broken / "attempts.jsonl").write_text((folder / "attempts.jsonl").read_text())
    load(db, broken)
    kinds = [(r["kind"], r["line_no"]) for r in db.execute("SELECT * FROM issues ORDER BY line_no")]
    assert kinds == [("malformed", 4), ("malformed", 5)]
    assert one(db, "SELECT status FROM sessions")["status"] == "finished"
    load(db, broken)
    assert one(db, "SELECT COUNT(*) FROM issues")[0] == 2  # the same lines again are not new problems


def test_an_import_that_fails_halfway_leaves_the_database_as_it_was(host, monkeypatch, db):
    folder = e2(host, monkeypatch, GOOD)
    real = history.insert

    def failing(conn, table, rows):
        if table == "tool_calls":
            raise sqlite3.OperationalError("disk I/O error")
        real(conn, table, rows)

    monkeypatch.setattr(history, "insert", failing)
    with pytest.raises(sqlite3.OperationalError):
        load(db, folder)
    assert [one(db, f"SELECT COUNT(*) FROM {t}")[0] for t in ("sessions", "messages", "tool_calls", "events", "imports", "issues")] == [0] * 6
    monkeypatch.setattr(history, "insert", real)
    load(db, folder)
    assert one(db, "SELECT COUNT(*) FROM sessions")[0] == 1


# ---- 6. generations, consumed text, usage and cost -----------------------------------------------------------------------------------------------
def test_a_pending_generation_is_settled_by_its_consumed_line_and_an_orphan_is_reported(tmp_path, db):
    events = [ev(1, "attempt_start", task="t"), gen(2, stop_sequences=[], stop_handling="undetermined", consumed_output=None, consumed_status="pending", raw_output="raw text\nObservation: x"),
              gen(3, ordinal=2, stop_sequences=[], stop_handling="undetermined", consumed_output=None, consumed_status="pending"),
              ev(4, "generation_consumed", generation_seq=2, stop_handling="local", stop_sequences_local=["\nObservation:"], consumed_output="raw text", discarded_chars=14, consumed_status="established"),
              ev(5, "generation_consumed", generation_seq=77, stop_handling="none", stop_sequences_local=[], consumed_output="?", discarded_chars=0, consumed_status="established"),
              ev(6, "attempt_end", kind="t", success=False, agent_completed=True, task_verified=False)]
    load(db, write_lines(tmp_path / "t" / "trace.jsonl", events).parent)
    settled, pending = db.execute("SELECT * FROM messages ORDER BY seq").fetchall()
    assert (settled["stop_handling"], settled["consumed_output"], settled["consumed_seq"], settled["discarded_chars"], settled["raw_output"]) == ("local", "raw text", 4, 14, "raw text\nObservation: x")
    assert (pending["consumed_output"], pending["consumed_status"], pending["consumed_seq"]) == (None, "pending", None)  # never settled: stays unknown
    assert one(db, "SELECT kind FROM issues WHERE kind = 'generation_consumed_without_generation'") is not None
    assert one(db, "SELECT payload FROM events WHERE type = 'generation_consumed'") is not None  # the orphan is kept, not dropped


def test_token_usage_and_cost_are_only_what_the_provider_reported(tmp_path, db):
    metas = [None, {}, {"prompt_tokens": 100, "completion_tokens": 20}, {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10, "cost_usd": None}, {"prompt_tokens": 5, "completion_tokens": 5, "cost_usd": 0.0123}]
    events = [ev(1, "attempt_start", task="t")] + [gen(2 + i, ordinal=i + 1, meta=m, tokens=99) for i, m in enumerate(metas)] + [ev(7, "attempt_end", kind="t", success=True, agent_completed=True, task_verified=True)]
    load(db, write_lines(tmp_path / "t" / "trace.jsonl", events).parent)
    rows = [(r["usage_status"], r["prompt_tokens"], r["completion_tokens"], r["total_tokens"], r["cost_status"], r["cost_usd"], r["tokens_used_reported"]) for r in db.execute("SELECT * FROM messages ORDER BY seq")]
    assert rows == [("not_reported", None, None, None, "not_reported", None, 99), ("not_reported", None, None, None, "not_reported", None, 99), ("reported", 100, 20, None, "not_reported", None, 99),
                    ("reported", 7, 3, 10, "unpriced", None, 99), ("reported", 5, 5, None, "reported", 0.0123, 99)]  # total_tokens is never computed
    usage = one(db, "SELECT * FROM session_usage")
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["cost_usd"], usage["messages_without_usage"], usage["messages_without_cost"]) == (112, 28, 0.0123, 2, 4)  # the gaps are stated beside the sums
    empty = [ev(1, "attempt_start", task="t", attempt="b"), gen(2, attempt="b", meta=None), ev(3, "attempt_end", attempt="b", kind="t", success=False, agent_completed=True, task_verified=False)]
    load(db, write_lines(tmp_path / "u" / "trace.jsonl", [{**e, "invocation": "inv000000002"} for e in empty]).parent)
    assert one(db, "SELECT prompt_tokens, completion_tokens, cost_usd FROM session_usage WHERE session_id = 'inv000000002:b'")[:] == (None, None, None)  # nothing reported: NULL, not 0


# ---- 7. prompt capture ------------------------------------------------------------------------------------------------------------------------
def test_prompt_capture_writes_a_private_verified_artifact_and_nothing_into_the_trace(host, monkeypatch, db):
    folder = e2(host, monkeypatch, GOOD, capture=True)
    prompts = folder / "prompts.local.jsonl"
    assert prompts.exists() and (prompts.stat().st_mode & 0o777) == 0o600
    records = [json.loads(line) for line in prompts.read_text().splitlines()]
    assert len(records) == 4 and all(set(r) == {"invocation", "run_id", "attempt", "seq", "sha256", "chars", "text"} for r in records)
    assert '"text"' not in (folder / "trace.jsonl").read_text() and all(r["text"] not in (folder / "trace.jsonl").read_text() for r in records)  # the trace never holds a prompt
    note = next(json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines() if '"prompt_capture_enabled"' in line)
    assert "adapter INPUT" in note["what"] and "not the provider's final payload" in note["what"]
    load(db, folder)
    messages = db.execute("SELECT * FROM messages ORDER BY seq").fetchall()
    assert [m["prompt_integrity"] for m in messages] == ["verified"] * 4
    assert all(m["prompt_text"] and len(m["prompt_text"]) == m["prompt_chars"] for m in messages)
    assert "You are the worker agent" in messages[1]["prompt_text"] and "You are the worker agent" not in messages[0]["prompt_text"]
    shown = "\n".join(history.timeline_lines(db, one(db, "SELECT session_id FROM sessions")["session_id"], prompts=True))
    assert "prompt text:" in shown and "(verified)" in shown


def test_without_capture_no_prompt_is_stored_anywhere(host, monkeypatch, db):
    folder = e2(host, monkeypatch, GOOD, capture=False)
    assert not (folder / "prompts.local.jsonl").exists()
    load(db, folder)
    assert [(m["prompt_text"], m["prompt_integrity"]) for m in db.execute("SELECT * FROM messages")] == [(None, "not_captured")] * 4
    assert all(m["prompt_sha256"] and m["prompt_chars"] for m in db.execute("SELECT * FROM messages"))  # the hash and length are always there


def test_a_tampered_prompt_is_flagged_and_a_missing_one_is_distinguished_from_one_never_captured(host, monkeypatch, db, tmp_path):
    folder = e2(host, monkeypatch, GOOD, capture=True)
    records = [json.loads(line) for line in (folder / "prompts.local.jsonl").read_text().splitlines()]
    records[0]["text"] += " (edited afterwards)"
    del records[2]
    records.append({**records[0], "seq": 999})
    copy = tmp_path / "tampered"
    shutil.copytree(folder, copy)
    (copy / "prompts.local.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    load(db, copy)
    assert [m["prompt_integrity"] for m in db.execute("SELECT * FROM messages ORDER BY seq")] == ["mismatch", "verified", "missing", "verified"]
    assert {r["kind"] for r in db.execute("SELECT * FROM issues")} == {"prompt_hash_mismatch", "prompt_without_message"}


def test_a_failed_generation_keeps_its_prompt_too(host, db):
    host_folder = harness_run_capture(host, FailsOnce([read_secret(), final(f"The code is {CODE}.")]))
    load(db, host_folder)
    first = one(db, "SELECT * FROM messages ORDER BY seq")
    assert first["status"] == "error" and first["prompt_integrity"] == "verified" and first["prompt_text"]


def harness_run_capture(directory: Path, model):
    initialize_runtime()
    folder = directory / "stage"
    folder.mkdir(exist_ok=True)
    workspace = directory / "ws"
    workspace.mkdir(exist_ok=True)
    (workspace / "secret.txt").write_text(f"The access code is {CODE}.\n")
    sr.TRACE.clear()
    sr.TRACE.update(sr.fresh_trace(path=folder / "trace.jsonl", stage="T", run_id="run-1", invocation=INV))
    sr.enable_prompt_capture(folder / "prompts.local.jsonl")
    model.generate = sr.traced_generate(model.generate, "scripted", [])
    agent = AgentConfig(name="a", model=model, tools=build_tools(["file_operations"], workspace), max_iterations=6, tool_calling_mode="react", raise_on_error=False, enable_memory=False, capability_probe=False, enable_sub_agents=False)
    harness = build_harness(agent, {}, {})
    with sr.attempt_scope("a1", "task", "single"), sr.tool_log(), harness:
        harness.run("task")
    return folder


# ---- 8. private artifacts stay out of Git and out of the packager's hands -----------------------------------------------------------------------
def test_private_files_are_not_archived_not_rewritten_and_not_tracked(host, monkeypatch, tmp_path):
    run = tmp_path / "run"
    (run / "stage_X").mkdir(parents=True)
    secret = 'a prompt that quotes a token hf_abcdefghijklmnopqrstuvwxyz0123 read by a tool\n'
    private = {"stage_X/prompts.local.jsonl": json.dumps({"text": secret}) + "\n", "history.db": "SQLite format 3\x00 " + secret, "history.db-wal": secret}
    for name, content in private.items():
        (run / name).write_text(content)
    (run / "stage_X" / "trace.jsonl").write_text(json.dumps({"event": "x", "text": "ok"}) + "\n")
    changed = drv.re_redact_text_files(run)
    assert changed == [] and all((run / n).read_text() == c for n, c in private.items())  # never rewritten: the token is still there, so the SHA-256 in the trace still verifies
    manifest = drv.write_manifest(run)
    excluded = {item["path"] for item in manifest["excluded_from_archive"]}
    assert excluded == set(private) and "stage_X/trace.jsonl" in {f["path"] for f in manifest["files"]}
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(drv, "RESULTS", tmp_path / "results")
    assert drv.cmd_package(SimpleNamespace(run_dir=str(run))) == 0
    import tarfile
    with tarfile.open(next((tmp_path / "results").glob("*.tar.gz"))) as tar:
        names = tar.getnames()
    assert not any(n.endswith(("prompts.local.jsonl", ".db", ".db-wal")) for n in names) and any(n.endswith("trace.jsonl") for n in names)
    ignored = subprocess.run(["git", "check-ignore", "experiments/history/history.db", "experiments/dgx/results/x/diag_E1/prompts.local.jsonl", "tmp/h.db-wal", "x.sqlite"], cwd=ROOT, capture_output=True, text=True)
    assert ignored.stdout.split() == ["experiments/history/history.db", "experiments/dgx/results/x/diag_E1/prompts.local.jsonl", "tmp/h.db-wal", "x.sqlite"]


def test_the_database_is_created_private_and_a_foreign_schema_is_refused(tmp_path):
    path = tmp_path / "fresh" / "h.db"
    history.connect(path).close()
    assert (path.stat().st_mode & 0o777) == 0o600
    other = sqlite3.connect(tmp_path / "other.db")
    other.execute("PRAGMA user_version = 7")
    other.commit()
    other.close()
    with pytest.raises(SystemExit, match="schema version 7"):
        history.connect(tmp_path / "other.db")


# ---- 9. commands and retention -----------------------------------------------------------------------------------------------------------------------
def test_the_commands_work_end_to_end_and_purge_removes_only_the_database_copy(host, monkeypatch, tmp_path, capsys):
    folder = e2(host, monkeypatch, GOOD, capture=True)
    path = tmp_path / "cli.db"
    assert history.main(["import", str(host), "--db", str(path)]) == 0
    capsys.readouterr()
    conn = history.connect(path)
    sid = one(conn, "SELECT session_id FROM sessions")["session_id"]
    assert history.main(["sessions", "--db", str(path)]) == 0 and sid in capsys.readouterr().out
    assert history.main(["timeline", sid, "--db", str(path)]) == 0
    out = capsys.readouterr().out
    assert "TOOL START delegate" in out and "task_verified=yes" in out
    assert history.main(["usage", sid, "--db", str(path)]) == 0
    usage = capsys.readouterr().out
    assert "reported no token usage" in usage and "delegate" in usage
    assert history.main(["purge", "--prompts", "--db", str(path)]) == 0
    assert [(m["prompt_text"], m["prompt_integrity"], bool(m["prompt_sha256"])) for m in conn.execute("SELECT * FROM messages")] == [(None, "purged", True)] * 4
    assert history.main(["purge", "--session", sid, "--db", str(path)]) == 0
    assert one(conn, "SELECT COUNT(*) FROM sessions")[0] == 0 and (folder / "trace.jsonl").exists()  # the evidence is untouched
    assert history.main(["import", str(host), "--db", str(path)]) == 0
    assert one(conn, "SELECT COUNT(*) FROM sessions")[0] == 1  # and restores the session
    assert history.main(["rebuild", str(host), "--db", str(path)]) == 0
    capsys.readouterr()
    assert one(history.connect(path), "SELECT COUNT(*) FROM messages")[0] == 4
    conn.close()


# ======================================================================================================================================================
# Final remediation (review of Phase 1): purge that really removes text, imports that never lose evidence, and four safeguards
# ======================================================================================================================================================
def file_bytes(path: Path) -> bytes:
    """The database file plus anything SQLite keeps next to it: what is physically there, not what a query says."""
    return b"".join(p.read_bytes() for p in (path, Path(str(path) + "-wal"), Path(str(path) + "-journal")) if p.exists())


def prompt_records(folder: Path) -> list:
    return [json.loads(line) for line in (folder / "prompts.local.jsonl").read_text().splitlines()]


# ---- B1: purge ------------------------------------------------------------------------------------------------------------------------------------
def test_purge_prompts_removes_the_text_from_the_database_file_itself(host, monkeypatch, tmp_path):
    folder = e2(host, monkeypatch, GOOD, capture=True)
    path = tmp_path / "p.db"
    history.main(["import", str(folder), "--db", str(path)])
    samples = [r["text"][:100].encode() for r in prompt_records(folder)]
    assert all(s in file_bytes(path) for s in samples)  # sanity: the text really is in the file before the purge, so the check below means something
    assert history.main(["purge", "--prompts", "--db", str(path)]) == 0
    assert not any(s in file_bytes(path) for s in samples)  # gone from the bytes, not only from the query results
    conn = history.connect(path)
    assert [(m["prompt_text"], m["prompt_integrity"], bool(m["prompt_sha256"])) for m in conn.execute("SELECT * FROM messages")] == [(None, "purged", True)] * 4
    assert not Path(str(path) + "-wal").exists() and not Path(str(path) + "-journal").exists()
    conn.close()


def test_purge_prompts_also_clears_text_that_spilled_into_overflow_pages(tmp_path):
    big = "x" * 9000 + "CANARY-OVERFLOW-9f3a" + "y" * 9000
    events = [ev(1, "attempt_start", task="t"), gen(2, prompt_sha256=history.prompt_sha256(big), prompt_chars=len(big)), ev(3, "attempt_end", kind="t", success=True, agent_completed=True, task_verified=True)]
    folder = tmp_path / "big"
    write_lines(folder / "trace.jsonl", events, raw=[json.dumps(ev(0, "prompt_capture_enabled", attempt=None, file="prompts.local.jsonl", what="x")) + "\n"][:0])
    (folder / "prompts.local.jsonl").write_text(json.dumps({"invocation": INV, "run_id": "run-1", "attempt": "a1", "seq": 2, "sha256": history.prompt_sha256(big), "chars": len(big), "text": big}) + "\n")
    path = tmp_path / "big.db"
    history.main(["import", str(folder), "--db", str(path)])
    assert b"CANARY-OVERFLOW-9f3a" in file_bytes(path)
    history.main(["purge", "--prompts", "--db", str(path)])
    assert b"CANARY-OVERFLOW-9f3a" not in file_bytes(path)


def test_purge_session_removes_the_session_text_from_the_database_file(host, monkeypatch, tmp_path):
    folder = e2(host, monkeypatch, GOOD, capture=True)
    path = tmp_path / "s.db"
    history.main(["import", str(folder), "--db", str(path)])
    canaries = [b"Read secret.txt and report the access code", CODE.encode()]
    assert all(c in file_bytes(path) for c in canaries)
    sid = one(history.connect(path), "SELECT session_id FROM sessions")["session_id"]
    history.main(["purge", "--session", sid, "--db", str(path)])
    assert not any(c in file_bytes(path) for c in canaries)


def test_secure_delete_is_on_for_every_connection_and_wal_mode_is_never_kept(tmp_path):
    path = tmp_path / "w.db"
    conn = history.connect(path)
    assert conn.execute("PRAGMA secure_delete").fetchone()[0] == 1 and conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    conn.close()
    other = sqlite3.connect(path)  # someone switches the file to WAL (the mode is persistent)
    other.execute("PRAGMA journal_mode = WAL")
    other.close()
    assert history.connect(path).execute("PRAGMA journal_mode").fetchone()[0] == "delete"  # converted back: no -wal file can hold a copy
    assert not Path(str(path) + "-wal").exists() and not Path(str(path) + "-shm").exists()
    holder = sqlite3.connect(path)  # a second connection that keeps a WAL database open cannot be converted: refuse, do not run in WAL
    holder.execute("PRAGMA journal_mode = WAL")
    holder.execute("CREATE TABLE IF NOT EXISTS probe (x)")
    holder.execute("INSERT INTO probe VALUES (1)")
    holder.commit()
    with pytest.raises(SystemExit, match="journal mode"):
        history.connect(path)
    holder.close()


# ---- B2: an incomplete copy must not erase what an earlier import held --------------------------------------------------------------------
def provenance_free(conn) -> dict:
    """The rows a re-import must not change, without `source` (where the latest import read a row from), the import log and the issues, which are provenance."""
    out = {}
    for name in ("sessions", "messages", "tool_calls", "events"):
        columns = [c["name"] for c in conn.execute(f"PRAGMA table_info({name})") if c["name"] != "source"]
        out[name] = [tuple(r) for r in conn.execute(f"SELECT {', '.join(columns)} FROM {name} ORDER BY 1")]
    return out


def archive_copy(folder: Path, destination: Path) -> Path:
    """What the packaged archive holds of a stage folder: everything except the private prompts."""
    shutil.copytree(folder, destination)
    (destination / "prompts.local.jsonl").unlink()
    return destination


def test_importing_the_archive_copy_after_the_local_evidence_keeps_the_verified_prompts(host, monkeypatch, db, tmp_path):
    folder = e2(host, monkeypatch, GOOD, capture=True)
    load(db, folder)
    before = provenance_free(db)
    texts = [m["prompt_text"] for m in db.execute("SELECT * FROM messages ORDER BY seq")]
    assert all(texts)
    summary = load(db, archive_copy(folder, tmp_path / "archive" / "diag_E2"))
    assert summary["imported"] and not summary["skipped_older"]  # same line count, different auxiliary files: it is processed, not ignored
    assert [m["prompt_text"] for m in db.execute("SELECT * FROM messages ORDER BY seq")] == texts
    assert [m["prompt_integrity"] for m in db.execute("SELECT * FROM messages")] == ["verified"] * 4
    assert provenance_free(db) == before  # nothing else moved
    kept = one(db, "SELECT detail FROM issues WHERE kind = 'kept_from_earlier_import'")
    assert kept is not None and "4 prompt" in kept["detail"]  # and it says so: not silent


def test_a_prompt_that_was_purged_stays_purged_when_an_archive_copy_is_imported(host, monkeypatch, tmp_path):
    folder = e2(host, monkeypatch, GOOD, capture=True)
    path = tmp_path / "pp.db"
    history.main(["import", str(folder), "--db", str(path)])
    history.main(["purge", "--prompts", "--db", str(path)])
    history.main(["import", str(archive_copy(folder, tmp_path / "ac" / "diag_E2")), "--db", str(path)])
    conn = history.connect(path)
    assert [(m["prompt_integrity"], m["prompt_text"]) for m in conn.execute("SELECT * FROM messages")] == [("purged", None)] * 4  # not "missing", not restored
    history.main(["import", str(folder), "--db", str(path)])  # the local evidence is unchanged since the first import: nothing is re-read
    assert one(conn, "SELECT COUNT(*) FROM messages WHERE prompt_text IS NOT NULL")[0] == 0


def test_a_genuinely_newer_complete_trace_updates_the_session_and_keeps_the_prompts_it_had(host, monkeypatch, db, tmp_path):
    full = e2(host, monkeypatch, GOOD, capture=True)
    trace = lines_of(full / "trace.jsonl")
    cut = [i for i, line in enumerate(trace) if '"event": "generation"' in line][1] + 1  # up to and including the worker's first generation
    last_seq = json.loads(trace[cut - 1])["seq"]
    live = tmp_path / "live"
    live.mkdir()
    (live / "trace.jsonl").write_text("".join(trace[:cut]))
    (live / "prompts.local.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prompt_records(full) if r["seq"] <= last_seq))
    load(db, live)
    assert one(db, "SELECT status FROM sessions")["status"] == "unfinished" and one(db, "SELECT COUNT(*) FROM messages")[0] == 2
    summary = load(db, archive_copy(full, tmp_path / "newer" / "diag_E2"))  # complete and newer, but without the private prompts
    assert summary["imported"]
    session = one(db, "SELECT * FROM sessions")
    assert (session["status"], session["task_verified"]) == ("finished", 1) and one(db, "SELECT COUNT(*) FROM tool_calls")[0] == 2
    assert [m["prompt_integrity"] for m in db.execute("SELECT * FROM messages ORDER BY seq")] == ["verified", "verified", "missing", "missing"]  # earlier ones kept, later ones honestly missing
    assert one(db, "SELECT COUNT(*) FROM messages WHERE prompt_text IS NOT NULL")[0] == 2


def test_prompts_that_arrive_later_are_attached_when_the_same_trace_is_imported_again(host, monkeypatch, db, tmp_path):
    folder = e2(host, monkeypatch, GOOD, capture=True)
    load(db, archive_copy(folder, tmp_path / "first" / "diag_E2"))
    assert [m["prompt_integrity"] for m in db.execute("SELECT * FROM messages")] == ["missing"] * 4
    load(db, folder)  # the same trace, now with its private prompts
    assert [m["prompt_integrity"] for m in db.execute("SELECT * FROM messages")] == ["verified"] * 4
    assert one(db, "SELECT COUNT(*) FROM issues WHERE kind = 'trace_content_changed'")[0] == 0  # only auxiliary files grew: not a surprise


def test_a_failed_replacement_keeps_the_earlier_import_and_its_prompts(host, monkeypatch, db, tmp_path):
    full = e2(host, monkeypatch, GOOD, capture=True)
    trace = lines_of(full / "trace.jsonl")
    cut = [i for i, line in enumerate(trace) if '"event": "generation"' in line][1] + 1
    live = tmp_path / "live"
    live.mkdir()
    (live / "trace.jsonl").write_text("".join(trace[:cut]))
    (live / "prompts.local.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prompt_records(full) if r["seq"] <= json.loads(trace[cut - 1])["seq"]))
    load(db, live)
    before = table_dump(db)
    real = history.insert
    monkeypatch.setattr(history, "insert", lambda conn, table, rows: (_ for _ in ()).throw(sqlite3.OperationalError("disk full")) if table == "tool_calls" else real(conn, table, rows))
    with pytest.raises(sqlite3.OperationalError):
        load(db, archive_copy(full, tmp_path / "newer" / "diag_E2"))
    assert table_dump(db) == before and one(db, "SELECT COUNT(*) FROM messages WHERE prompt_text IS NOT NULL")[0] == 2


# ---- safeguards ----------------------------------------------------------------------------------------------------------------------------------------
def test_a_new_attempt_never_inherits_the_previous_attempts_generation(host):
    import asyncio

    from smol_harness.tools import build_tools
    initialize_runtime()
    folder = host / "stage"
    folder.mkdir()
    workspace = host / "ws"
    workspace.mkdir()
    (workspace / "a.txt").write_text("A\n")
    sr.TRACE.clear()
    sr.TRACE.update(sr.fresh_trace(path=folder / "trace.jsonl", stage="T", run_id="run-1", invocation=INV))
    model = SimpleNamespace(generate=lambda prompt, config=None, **kw: SimpleNamespace(text="Final Answer: x", finish_reason="stop", tokens_used=1, model_name="m", metadata={}))
    tool = build_tools(["file_operations"], workspace)[0]
    with sr.attempt_scope("a1", "task", "single"):
        sr.traced_generate(model.generate, "scripted", [])("p")
    assert sr.TRACE["last_gen"]  # attempt a1 produced a generation
    with sr.attempt_scope("a2", "task", "single"), sr.tool_log():
        assert sr.TRACE["last_gen"] == {}  # reset at the boundary
        asyncio.run(tool.execute(operation="read", path="a.txt"))  # a tool call before a2 has generated anything
    start = next(json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines() if '"tool_call_start"' in line)
    assert start["attempt"] == "a2" and start["requested_by_generation_seq"] is None  # not attributed to a1's generation


def test_a_recorded_link_into_another_session_is_rejected(tmp_path, db):
    events = [ev(1, "attempt_start", attempt="a", task="t"), gen(2, attempt="a"), ev(3, "attempt_end", attempt="a", kind="t", success=True, agent_completed=True, task_verified=True),
              ev(4, "attempt_start", attempt="b", task="t"),
              ev(5, "tool_call_start", attempt="b", tool="file_operations", role="single", generation_in_agent=1, delegation_id=None, args={}, requested_by_generation_seq=2),
              ev(6, "tool_call", attempt="b", tool_call_id=f"{INV}:5", start_seq=5, tool="file_operations", role="single", generation_in_agent=1, args={}, success=True, exception=False, result="r", visible="r",
                 seconds=0.1, requested_by_generation_seq=2),
              ev(7, "attempt_end", attempt="b", kind="t", success=True, agent_completed=True, task_verified=True)]
    load(db, write_lines(tmp_path / "x" / "trace.jsonl", events).parent)
    row = one(db, "SELECT * FROM tool_calls")
    assert (row["session_id"], row["requested_by_message_id"], row["link_basis"]) == (f"{INV}:b", None, None)  # generation 2 belongs to session a
    assert one(db, "SELECT detail FROM issues WHERE kind = 'requested_by_other_session'") is not None


def test_same_line_count_with_different_trace_content_is_recorded_as_an_import_issue(host, monkeypatch, db, tmp_path):
    folder = e2(host, monkeypatch, GOOD, capture=False)
    load(db, folder)
    changed = tmp_path / "changed" / "diag_E2"
    shutil.copytree(folder, changed)
    lines = lines_of(changed / "trace.jsonl")
    i = next(i for i, line in enumerate(lines) if '"event": "generation"' in line)
    lines[i] = lines[i].replace("Thought: next", "Thought: EDITED", 1) if "Thought: next" in lines[i] else lines[i].replace("Thought", "Thought EDITED", 1)
    (changed / "trace.jsonl").write_text("".join(lines))
    summary = load(db, changed)
    assert summary["imported"]  # the evidence on disk is the evidence: it is imported...
    assert "EDITED" in one(db, "SELECT raw_output FROM messages WHERE message_id = ?", f"{json.loads(lines[i])['invocation']}:{json.loads(lines[i])['seq']}")["raw_output"]
    issue = one(db, "SELECT * FROM issues WHERE kind = 'trace_content_changed'")
    assert issue is not None and "same number of lines" in issue["detail"]  # ...and the surprise is on record
    load(db, changed)
    assert one(db, "SELECT COUNT(*) FROM issues WHERE kind = 'trace_content_changed'")[0] == 1  # not duplicated by repeating the import


def test_packaging_redacts_the_rendered_yaml_and_leaves_private_files_out_and_untouched(monkeypatch, tmp_path):
    import tarfile
    home = tmp_path / "home"
    run = home / "checkout" / "results" / "run"
    (run / "configs_rendered").mkdir(parents=True)
    (run / "diag_E2").mkdir()
    monkeypatch.setenv("HOME", str(home))
    yaml = f'workspace: "{run}/diag_E2/work/X2-plain-0"\nbase_url: "http://operator:s3cretpw@127.0.0.1:11434/v1"\n'
    prompts = json.dumps({"text": f"quotes {home}/notes and hf_abcdefghijklmnopqrstuvwxyz0123"}) + "\n"
    (run / "configs_rendered" / "ollama_only-X2-plain-a1.yaml").write_text(yaml)
    (run / "diag_E2" / "prompts.local.jsonl").write_text(prompts)
    (run / "diag_E2" / "trace.jsonl").write_text(json.dumps({"event": "x"}) + "\n")
    (tmp_path / "out").mkdir()
    monkeypatch.setattr(drv, "RESULTS", tmp_path / "out")
    assert drv.cmd_package(SimpleNamespace(run_dir=str(run))) == 0
    with tarfile.open(next((tmp_path / "out").glob("*.tar.gz"))) as tar:
        names = tar.getnames()
        archived = tar.extractfile(next(n for n in names if n.endswith(".yaml"))).read().decode()
    assert str(home) not in archived and "s3cretpw" not in archived and 'workspace: "~/checkout/results/run/' in archived  # what is shared is redacted
    assert not any(n.endswith("prompts.local.jsonl") for n in names) and any(n.endswith("trace.jsonl") for n in names)  # the private file is not shared
    assert (run / "diag_E2" / "prompts.local.jsonl").read_text() == prompts  # and it was not rewritten: the token and the home path are still there


def test_a_copy_without_the_attempt_record_does_not_erase_the_verification_an_earlier_import_held(host, monkeypatch, db, tmp_path):
    folder = e2(host, monkeypatch, GOOD, capture=False)
    load(db, folder)
    before = one(db, "SELECT final_answer, verification, task_verified FROM sessions")
    assert before["verification"] and before["final_answer"]
    copy = tmp_path / "noattempts" / "diag_E2"
    shutil.copytree(folder, copy)
    (copy / "attempts.jsonl").unlink()
    load(db, copy)
    after = one(db, "SELECT final_answer, verification, task_verified, status FROM sessions")
    assert tuple(after)[:3] == tuple(before) and after["status"] == "finished"
    assert one(db, "SELECT COUNT(*) FROM issues WHERE kind = 'attempt_record_missing'")[0] == 0  # it was not missing: the database had it
    assert "1 attempt record(s)" in one(db, "SELECT detail FROM issues WHERE kind = 'kept_from_earlier_import'")["detail"]
    fresh = history.connect(tmp_path / "fresh.db")  # a database that never saw the record cannot invent it
    load(fresh, copy)
    assert one(fresh, "SELECT verification, final_answer FROM sessions")[:] == (None, None) and one(fresh, "SELECT COUNT(*) FROM issues WHERE kind = 'attempt_record_missing'")[0] == 1
    fresh.close()


def test_a_sqlite_build_that_ignores_secure_delete_is_refused_rather_than_trusted(tmp_path, monkeypatch):
    real = sqlite3.connect

    class IgnoresSecureDelete(sqlite3.Connection):
        def execute(self, sql, *args):
            return super().execute("PRAGMA secure_delete = OFF" if sql == "PRAGMA secure_delete = ON" else sql, *args)

    monkeypatch.setattr(history.sqlite3, "connect", lambda path: real(path, factory=IgnoresSecureDelete))
    with pytest.raises(SystemExit, match="does not honour PRAGMA secure_delete"):
        history.connect(tmp_path / "x.db")


def test_a_stored_prompt_that_no_longer_matches_its_hash_is_not_carried_into_a_new_import(host, monkeypatch, db, tmp_path):
    folder = e2(host, monkeypatch, GOOD, capture=True)
    load(db, folder)
    db.execute("UPDATE messages SET prompt_text = 'edited inside the database' WHERE message_id = (SELECT message_id FROM messages ORDER BY seq LIMIT 1)")
    load(db, archive_copy(folder, tmp_path / "ac" / "diag_E2"))
    states = [(m["prompt_integrity"], m["prompt_text"] is not None) for m in db.execute("SELECT * FROM messages ORDER BY seq")]
    assert states == [("missing", False), ("verified", True), ("verified", True), ("verified", True)]  # three still hash to the trace; the edited one is not trusted


def test_a_content_change_note_survives_the_next_time_the_invocation_is_analysed(host, monkeypatch, db, tmp_path):
    folder = e2(host, monkeypatch, GOOD, capture=False)
    load(db, folder)
    changed = tmp_path / "changed" / "diag_E2"
    shutil.copytree(folder, changed)
    lines = lines_of(changed / "trace.jsonl")
    i = next(i for i, line in enumerate(lines) if '"event": "generation"' in line)
    lines[i] = lines[i].replace("Thought", "Thought EDITED", 1)
    (changed / "trace.jsonl").write_text("".join(lines))
    load(db, changed)
    assert one(db, "SELECT COUNT(*) FROM issues WHERE kind = 'trace_content_changed'")[0] == 1
    later = tmp_path / "later" / "diag_E2"
    shutil.copytree(changed, later)
    (later / "attempts.jsonl").unlink()  # another change to the auxiliary files only: the invocation is analysed again
    assert load(db, later)["imported"]
    assert one(db, "SELECT COUNT(*) FROM issues WHERE kind = 'trace_content_changed'")[0] == 1  # the earlier surprise is still on record
