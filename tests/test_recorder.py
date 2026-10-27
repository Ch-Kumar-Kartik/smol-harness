"""SQLite execution history, Phase 2: `smol-harness run --history-dir` (src/smol_harness/recorder.py) and the shared redaction module.

CPU only, no network, no real model. Real effGen agents run the real loop with the real tools; only `build_model` is replaced by scripted models. The hooks are also driven
directly (real effGen context objects) where a real run cannot produce the situation: overlapping delegations, a failing disk, the exact return-value contract. Two tests
start a fresh interpreter and really send SIGINT / SIGKILL in the middle of a delegation.
"""

from __future__ import annotations

import contextvars
import copy
import dataclasses
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import ScriptedModel
from effgen import AgentConfig
from effgen.core.middleware import ModelCallContext, RunContext, ToolCallContext
from test_cli import CliModels, write_config
from test_harness_config import BIG, FINAL, SMALL, call, delegate

import smol_harness.__main__ as cli
import smol_harness.models as models_mod
import smol_harness.recorder as recorder_mod
from smol_harness import redaction
from smol_harness.harness import build_harness
from smol_harness.recorder import ExecutionRecorder, prompt_sha256

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "history"))
sys.path.insert(0, str(ROOT / "experiments" / "rtx4070"))
import history  # noqa: E402
import rtx4070 as drv  # noqa: E402

WORKER_FLOW = [call("file_operations", operation="read", path="missing.txt"),
               call("file_operations", operation="read", path="../outside.txt"),
               call("file_operations", operation="write", path="hello.txt", content="hi\n"),
               "Thought: ok.\nFinal Answer: wrote hello.txt"]
FLOW = {BIG: [delegate("writer", "create hello.txt"), FINAL], SMALL: WORKER_FLOW}


# ---- helpers ------------------------------------------------------------------------------------------------------------------
def run_cli(monkeypatch, capsys, tmp_path, scripts, *extra, config=None, task="make hello.txt"):
    """One `cli.main` run with scripted models. Returns the exit code, stdout, stderr and the models (to see the prompts they were given)."""
    fake = CliModels()
    fake.scripts.update(scripts)
    monkeypatch.setattr(models_mod, "build_model", fake.build)
    code = cli.main(["run", "--config", str(config or write_config(tmp_path)), "--task", task, *extra])
    captured = capsys.readouterr()
    return code, captured.out, captured.err, fake


def history_dir_of(err: str) -> Path:
    return Path(next(line for line in err.splitlines() if line.startswith("history: ")).removeprefix("history: ").split(" (")[0])


def events(directory: Path) -> list[dict]:
    return [json.loads(line) for line in jsonl_lines(directory / "trace.jsonl")]


def jsonl_lines(path: Path) -> list[str]:
    """Records split on "\n" only, as the importer does: str.splitlines() would also split inside a value holding U+2028 or U+0085."""
    return [line for line in path.read_text(encoding="utf-8").split("\n") if line]


def import_into(directory: Path, db: Path):
    conn = history.connect(db)
    history.import_source(conn, directory / "trace.jsonl", "2026-10-27T00:00:00Z")
    return conn


def rows(conn, sql, *args):
    return [dict(r) for r in conn.execute(sql, args)]


def prompts_seen(fake: CliModels) -> list[str]:
    return [str(c.prompt) for model in fake.instances.values() for c in model.calls]


@pytest.fixture
def recorded(monkeypatch, capsys, tmp_path):
    """The scripted orchestrator -> writer flow through the CLI, with history and prompt capture on, and its SQLite import."""
    code, out, err, fake = run_cli(monkeypatch, capsys, tmp_path, FLOW, "--history-dir", str(tmp_path / "hist"), "--capture-prompts")
    assert code == 0, err
    directory = history_dir_of(err)
    conn = import_into(directory, tmp_path / "h.db")
    session = rows(conn, "SELECT session_id FROM sessions")[0]["session_id"]
    return SimpleNamespace(dir=directory, conn=conn, session=session, out=out, err=err, fake=fake, events=events(directory))


# ---- 1. default behaviour is unchanged ----------------------------------------------------------------------------------------
def test_history_off_changes_nothing_and_writes_nothing(monkeypatch, capsys, tmp_path):
    seen = []
    real = cli.build_harness_from_config
    monkeypatch.setattr(cli, "build_harness_from_config", lambda config, **kw: (seen.append(kw), real(config, **kw))[1])
    code, out, err, fake = run_cli(monkeypatch, capsys, tmp_path, FLOW)
    assert (code, out) == (0, "all done\n") and "history" not in err
    assert seen == [{}]  # no recorder argument at all: the builder is called exactly as before this feature existed
    assert sorted(p.name for p in tmp_path.iterdir()) == ["harness.yaml", "ws"]  # nothing created beside the config and the workspace


def test_history_on_does_not_change_what_the_agents_see_or_the_cli_output(monkeypatch, capsys, tmp_path):
    config = write_config(tmp_path)
    code, out, err, plain = run_cli(monkeypatch, capsys, tmp_path, FLOW, config=config)
    (tmp_path / "ws" / "hello.txt").unlink()
    code2, out2, err2, with_history = run_cli(monkeypatch, capsys, tmp_path, FLOW, "--history-dir", str(tmp_path / "h"), "--capture-prompts", config=config)
    assert (code2, out2) == (code, out)
    assert [line for line in err2.splitlines() if not line.startswith("history:")] == err.splitlines()
    assert prompts_seen(with_history) == prompts_seen(plain)  # byte-identical prompts: the recorder added nothing to what any model was asked


@pytest.mark.parametrize("argv, message", [(["--capture-prompts"], "--capture-prompts needs --history-dir")])
def test_capture_prompts_without_a_history_dir_is_a_usage_error(monkeypatch, capsys, tmp_path, argv, message):
    code, out, err, fake = run_cli(monkeypatch, capsys, tmp_path, FLOW, *argv)
    assert code == 2 and out == "" and message in err and fake.loads == []


# ---- 2. a reconstructable timeline (end to end, CLI -> JSONL -> SQLite) -------------------------------------------------------------
def test_the_cli_run_is_reconstructable_from_jsonl_and_sqlite(recorded):
    conn, session = recorded.conn, recorded.session
    s = rows(conn, "SELECT * FROM sessions")[0]
    assert (s["status"], s["stage"], s["task"], s["final_answer"], s["agent_completed"], s["task_verified"]) == ("finished", "cli", "make hello.txt", "all done", 1, None)
    assert json.loads(s["model"]) == {"orchestrator": BIG, "workers": {"writer": SMALL}}
    msgs = rows(conn, "SELECT role, agent_name, generation_in_agent, status, prompt_integrity FROM messages ORDER BY seq")
    assert [(m["role"], m["generation_in_agent"]) for m in msgs] == [("orchestrator", 1), ("worker", 1), ("worker", 2), ("worker", 3), ("worker", 4), ("orchestrator", 2)]
    assert {m["prompt_integrity"] for m in msgs} == {"verified"} and {m["status"] for m in msgs} == {"ok"}
    assert [(t["role"], t["tool"]) for t in rows(conn, "SELECT role, tool FROM tool_calls ORDER BY started_seq")] == [
        ("orchestrator", "delegate"), ("worker", "file_operations"), ("worker", "file_operations"), ("worker", "file_operations")]
    assert rows(conn, "SELECT COUNT(*) n FROM issues")[0]["n"] == 0  # a clean import: nothing was lost or guessed
    text = "\n".join(history.timeline_lines(conn, session, full=True))
    for expected in ("status=finished", "task: make hello.txt", "TOOL START delegate (orchestrator)", "-> worker 'writer'", "[worker d1] TOOL START file_operations (writer)",
                     "[worker d1 #4] MESSAGE", "TOOL END   delegate RETURNED (outcome not confirmed)", " as d1", "final answer: all done", "task_verified=unknown"):
        assert expected in text, expected
    # the worker's lines sit between the delegate call's start and its end, indented under it
    assert text.index("TOOL START delegate") < text.index("[worker d1 #1]") < text.index("TOOL END   delegate")
    assert os.path.getsize(recorded.dir / "trace.jsonl") > 0 and (recorded.dir / "attempts.jsonl").read_text().count("\n") == 1


# ---- 3. parent-child delegation attribution -----------------------------------------------------------------------------------------
def test_every_worker_event_names_the_delegate_call_that_started_it(recorded):
    start = next(e for e in recorded.events if e["event"] == "tool_call_start" and e["tool"] == "delegate")
    parent = f"{start['invocation']}:{start['seq']}"
    worker_events = [e for e in recorded.events if e.get("role") == "worker"]
    assert worker_events and all(e["parent_tool_call_id"] == parent and e["agent_name"] == "writer" for e in worker_events)
    assert not [e for e in recorded.events if e.get("role") == "orchestrator" and "parent_tool_call_id" in e]
    assert start["delegation_id"] is None and start["delegation_id_expected"] is None  # an id is never predicted
    end = next(e for e in recorded.events if e["event"] == "tool_call" and e["tool"] == "delegate")
    assert end["delegation_id"] == "d1" == end["delegation_id_reported"] and end["tool_call_id"] == parent and end["worker_status"] == "completed"
    # in the database the worker rows carry the worker's name and the confirmed id, resolved through the parent call
    assert {(r["agent_name"], r["delegation_id"]) for r in rows(recorded.conn, "SELECT agent_name, delegation_id FROM messages WHERE role = 'worker'")} == {("writer", "d1")}
    assert {(r["agent_name"], r["delegation_id"]) for r in rows(recorded.conn, "SELECT agent_name, delegation_id FROM tool_calls WHERE role = 'worker'")} == {("writer", "d1")}
    d = rows(recorded.conn, "SELECT worker, task, confirmed, delegation_id, status FROM delegations")
    assert d == [{"worker": "writer", "task": "create hello.txt", "confirmed": 1, "delegation_id": "d1", "status": "returned"}]


def test_each_tool_call_is_linked_to_the_generation_that_requested_it_at_the_source(recorded):
    links = rows(recorded.conn, "SELECT t.tool_call_id, t.link_basis, m.role, m.generation_in_agent g, t.generation_in_agent tg "
                                "FROM tool_calls t JOIN messages m ON m.message_id = t.requested_by_message_id ORDER BY t.started_seq")
    assert len(links) == 4 and {r["link_basis"] for r in links} == {"recorded"}
    assert [r["g"] for r in links] == [r["tg"] for r in links] == [1, 1, 2, 3]


def test_sequential_delegations_to_two_workers_are_attributed_separately(monkeypatch, capsys, tmp_path):
    reader = {"name": "reader", "description": "reads", "tools": ["file_operations"], "max_iterations": 4, "model": {"provider": "ollama", "model": "reader:1", "context_length": 4096}}
    scripts = {BIG: [delegate("writer", "create hello.txt"), delegate("reader", "read hello.txt"), FINAL], SMALL: WORKER_FLOW[2:],
               "reader:1": [call("file_operations", operation="read", path="hello.txt"), "Thought: ok.\nFinal Answer: hi"]}
    code, out, err, fake = run_cli(monkeypatch, capsys, tmp_path, scripts, "--history-dir", str(tmp_path / "hist"), config=write_config(tmp_path, extra_workers=[reader]))
    assert code == 0, err
    conn = import_into(history_dir_of(err), tmp_path / "h.db")
    assert rows(conn, "SELECT worker, task, delegation_id, confirmed FROM delegations ORDER BY started_seq") == [
        {"worker": "writer", "task": "create hello.txt", "delegation_id": "d1", "confirmed": 1},
        {"worker": "reader", "task": "read hello.txt", "delegation_id": "d2", "confirmed": 1}]
    assert rows(conn, "SELECT DISTINCT agent_name, delegation_id FROM messages WHERE role = 'worker' ORDER BY delegation_id") == [
        {"agent_name": "writer", "delegation_id": "d1"}, {"agent_name": "reader", "delegation_id": "d2"}]
    assert rows(conn, "SELECT COUNT(*) n FROM tool_calls WHERE role = 'worker' AND agent_name = 'reader' AND delegation_id = 'd2' AND tool = 'file_operations'")[0]["n"] == 1


def simulate_overlapping_delegations(recorder: ExecutionRecorder, count: int = 3) -> None:
    """`count` delegate calls of ONE orchestrator run in flight at once, each with its worker run on its own thread (as DelegateTool does), finishing
    in the opposite order of starting. Real effGen context objects; the hooks are called the way effGen calls them."""
    recorder.begin("overlap", {})
    orchestrator = RunContext(task="overlap", agent_name="orchestrator")
    recorder.before_run(orchestrator)
    started, release = threading.Barrier(count), [threading.Event() for _ in range(count)]
    errors: list[BaseException] = []

    def one_delegation(i: int) -> None:
        try:
            tool = ToolCallContext(tool_name="delegate", tool_input=json.dumps({"worker": f"w{i}", "task": f"task {i}"}), run=orchestrator)
            assert recorder.before_tool_call(tool) is None
            started.wait(timeout=10)  # every delegate call has started before any worker runs

            def worker_run() -> None:
                run = RunContext(task=f"prompt {i}", agent_name=f"effgen-agent-{i}")
                recorder.before_run(run)
                model = ModelCallContext(prompt=f"worker {i} prompt", model_name=f"m{i}", run=run)
                recorder.before_model_call(model)
                assert release[i].wait(timeout=10)  # the worker that started last finishes first, then its predecessor
                recorder.after_model_call(model, {"text": f"worker {i} says", "tokens_used": 1, "finish_reason": "stop"})
                inner = ToolCallContext(tool_name="file_operations", tool_input=json.dumps({"path": f"f{i}"}), run=run)
                recorder.before_tool_call(inner)
                recorder.after_tool_call(inner, f"contents {i}")
                recorder.after_run(run, SimpleNamespace(success=True, termination="done", iterations=1, tool_call_count=1, tokens_used=1))

            thread = threading.Thread(target=contextvars.copy_context().run, args=(worker_run,))  # what DelegateTool does before its executor thread runs the worker
            thread.start()
            thread.join(timeout=10)
            if i > 0:
                release[i - 1].set()
            recorder.after_tool_call(tool, json.dumps({"worker": f"w{i}", "status": "completed", "delegation_id": f"d{i + 1}"}))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=one_delegation, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    release[count - 1].set()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors, errors
    recorder.after_run(orchestrator, SimpleNamespace(success=True, termination="done", iterations=1, tool_call_count=count, tokens_used=1))
    recorder.finish(SimpleNamespace(success=True, output="done", termination="done"))


def test_overlapping_delegations_keep_their_own_parent_worker_and_confirmed_id(tmp_path):
    recorder = ExecutionRecorder(tmp_path / "h")
    simulate_overlapping_delegations(recorder)
    assert recorder.error is None
    lines = events(recorder.dir)
    assert [e["seq"] for e in lines] == list(range(1, len(lines) + 1))  # every line complete, file order == seq order, no gaps, no interleaved bytes
    starts = {e["args"]["worker"]: f"{e['invocation']}:{e['seq']}" for e in lines if e["event"] == "tool_call_start" and e["tool"] == "delegate"}
    assert set(starts) == {"w0", "w1", "w2"}
    for e in lines:
        if e.get("role") == "worker":
            assert e["parent_tool_call_id"] == starts[e["agent_name"]]  # never another overlapping delegation's id
    conn = import_into(recorder.dir, tmp_path / "h.db")
    assert rows(conn, "SELECT worker, delegation_id, confirmed FROM delegations ORDER BY worker") == [
        {"worker": f"w{i}", "delegation_id": f"d{i + 1}", "confirmed": 1} for i in range(3)]
    for i in range(3):
        assert rows(conn, "SELECT DISTINCT delegation_id d FROM messages WHERE agent_name = ?", f"w{i}") == [{"d": f"d{i + 1}"}]
        assert rows(conn, "SELECT consumed_output c FROM messages WHERE agent_name = ?", f"w{i}") == [{"c": f"worker {i} says"}]
        assert [r["result"] for r in rows(conn, "SELECT result FROM tool_calls WHERE agent_name = ? AND tool = 'file_operations'", f"w{i}")] == [f"contents {i}"]
    assert rows(conn, "SELECT COUNT(*) n FROM issues")[0]["n"] == 0


def test_the_parent_is_released_after_the_delegate_call_and_after_a_new_session_begins(tmp_path):
    recorder = ExecutionRecorder(tmp_path / "h")
    recorder.begin("one", {})
    run = RunContext(task="one", agent_name="orchestrator")
    recorder.before_run(run)
    tool = ToolCallContext(tool_name="delegate", tool_input='{"worker": "w"}', run=run)
    recorder.before_tool_call(tool)
    assert recorder_mod._PARENT.get() is not None
    recorder.after_tool_call(tool, "{}")
    assert recorder_mod._PARENT.get() is None
    recorder.before_tool_call(tool)  # a delegate call that never reports (a hook raised): the next session must not start as a "worker"
    recorder.begin("two", {})
    second = RunContext(task="two", agent_name="orchestrator")
    recorder.before_run(second)
    assert [e["role"] for e in events(recorder.dir) if e["event"] == "agent_run_start"] == ["orchestrator", "orchestrator"]


# ---- 5/6. several tools, failed and refused actions; unknown versus confirmed outcomes ----------------------------------------------------------
def test_failed_and_refused_tool_calls_are_recorded_as_returned_with_unknown_outcome(recorded):
    tools = rows(recorded.conn, "SELECT tool, status, success, exception, error, result, visible, visible_unavailable FROM tool_calls WHERE role = 'worker' ORDER BY started_seq")
    assert [t["status"] for t in tools] == ["returned"] * 3  # the missing file and the path escape are NOT "failed": nothing confirms either outcome
    assert all(t["success"] is None and t["exception"] is None and t["error"] is None for t in tools)  # unknown stays NULL, never guessed
    assert "File not found" in tools[0]["result"] and "outside allowed directories" in tools[1]["result"] and tools[2]["result"].startswith("wrote hello.txt")
    assert all(t["visible"] is None and t["visible_unavailable"] for t in tools)
    # the guess exists only at display time, apart from the stored outcome
    text = "\n".join(history.timeline_lines(recorded.conn, recorded.session))
    assert text.count("heuristic (NOT a confirmed outcome): result starts like an error message") == 2
    assert not any(history.looks_like_error(t["result"]) for t in tools[2:])
    counts = rows(recorded.conn, "SELECT tool, role, completed, returned, failed, unfinished FROM tool_counts ORDER BY role, tool")
    assert counts == [{"tool": "delegate", "role": "orchestrator", "completed": 0, "returned": 1, "failed": 0, "unfinished": 0},
                      {"tool": "file_operations", "role": "worker", "completed": 0, "returned": 3, "failed": 0, "unfinished": 0}]
    assert (recorded.dir.parent.parent / "ws" / "hello.txt").read_text() == "hi\n"  # and the one real write did happen


def test_a_refused_delegation_has_a_confirmed_id_and_no_worker_events(monkeypatch, capsys, tmp_path):
    scripts = {BIG: [delegate("writer", "first"), delegate("writer", "second"), delegate("ghost", "third"), FINAL], SMALL: [WORKER_FLOW[2], WORKER_FLOW[3]]}
    code, out, err, fake = run_cli(monkeypatch, capsys, tmp_path, scripts, "--history-dir", str(tmp_path / "hist"), config=write_config(tmp_path, max_delegations=2))
    assert code == 0, err
    directory = history_dir_of(err)
    conn = import_into(directory, tmp_path / "h.db")
    assert rows(conn, "SELECT worker w, delegation_id d, confirmed, status FROM delegations ORDER BY started_seq") == [
        {"w": "writer", "d": "d1", "confirmed": 1, "status": "returned"}, {"w": "writer", "d": "d2", "confirmed": 1, "status": "returned"},
        {"w": "ghost", "d": "d3", "confirmed": 1, "status": "returned"}]
    reports = [json.loads(r["result"]) for r in rows(conn, "SELECT result FROM tool_calls WHERE tool = 'delegate' ORDER BY started_seq")]
    assert [r["status"] for r in reports] == ["completed", "completed", "failed"] and reports[2]["error_type"] == "delegation_limit_reached"
    ghost = next(e for e in events(directory) if e["event"] == "tool_call_start" and e["args"].get("worker") == "ghost")
    assert not [e for e in events(directory) if e.get("parent_tool_call_id") == f"{ghost['invocation']}:{ghost['seq']}"]  # nothing ran for the refused call
    assert "the delegate call itself returned" in "\n".join(history.timeline_lines(conn, rows(conn, "SELECT session_id FROM sessions")[0]["session_id"]))


# ---- 7. recorder failures and the middleware contracts ------------------------------------------------------------------------------------
def make_contexts():
    run = RunContext(task="t", agent_name="orchestrator", metadata={"keep": 1})
    return run, ModelCallContext(prompt="p", model_name="m", kwargs={"temperature": 0.1}, run=run), ToolCallContext(tool_name="file_operations", tool_input='{"path": "a"}', run=run)


def drive_all_hooks(recorder):
    """Every hook once, in effGen's order. Returns what each returned, and the sentinel objects it was given."""
    run, model, tool = make_contexts()
    result, tool_result, response = {"text": "x", "tokens_used": 1}, "tool output", SimpleNamespace(success=True)
    before = [copy.deepcopy(c) for c in (run, model, tool)]
    returned = {"before_run": recorder.before_run(run), "before_model_call": recorder.before_model_call(model), "after_model_call": recorder.after_model_call(model, result),
                "before_tool_call": recorder.before_tool_call(tool), "after_tool_call": recorder.after_tool_call(tool, tool_result), "after_run": recorder.after_run(run, response)}
    return returned, (result, tool_result, response), (run, model, tool), before


def assert_contract(returned, given, contexts, before):
    assert returned["before_run"] is None and returned["before_model_call"] is None and returned["before_tool_call"] is None  # None = "continue", never a short-circuit
    assert returned["after_model_call"] is given[0] and returned["after_tool_call"] is given[1] and returned["after_run"] is given[2]  # the very objects effGen passed in
    run, model, tool = contexts
    assert (model.prompt, model.kwargs, model.attempt) == (before[1].prompt, before[1].kwargs, before[1].attempt)  # nothing in a context was rewritten
    assert (tool.tool_name, tool.tool_input) == (before[2].tool_name, before[2].tool_input) and run.task == before[0].task
    assert {k: v for k, v in run.metadata.items() if k != recorder_mod._STATE_KEY} == before[0].metadata


def test_hooks_honour_effgens_return_contract_when_recording(tmp_path):
    recorder = ExecutionRecorder(tmp_path / "h", capture_prompts=True)
    recorder.begin("t", {})
    assert_contract(*drive_all_hooks(recorder))
    assert recorder.error is None and len(events(recorder.dir)) > 5


def test_hooks_honour_the_contract_after_recording_has_stopped(tmp_path):
    recorder = ExecutionRecorder(tmp_path / "h")
    recorder.error = "RuntimeError: stopped earlier"
    size = (recorder.dir / "trace.jsonl").stat().st_size
    assert_contract(*drive_all_hooks(recorder))
    assert (recorder.dir / "trace.jsonl").stat().st_size == size  # a stopped recorder writes nothing more


@pytest.mark.parametrize("fail_on", [1, 2, 4, 7])
def test_a_failing_write_never_breaks_a_hook_and_stops_recording_for_good(tmp_path, monkeypatch, fail_on):
    recorder = ExecutionRecorder(tmp_path / "h", capture_prompts=True)
    real, writes = recorder_mod._append, []

    def flaky(path, text):
        writes.append(path.name)
        if len(writes) == fail_on:
            raise OSError(28, "No space left on device")
        return real(path, text)

    monkeypatch.setattr(recorder_mod, "_append", flaky)
    recorder.begin("t", {})
    assert_contract(*drive_all_hooks(recorder))
    assert recorder.error is not None and "No space left" in recorder.error and len(writes) == fail_on  # the failing write was the last attempt: nothing after it
    recorder.finish(SimpleNamespace(success=True, output="o", termination="done"))
    recorder.abort(RuntimeError("late"))
    assert len(writes) == fail_on


def test_an_unserialisable_value_is_a_recorder_error_not_a_run_error(tmp_path, monkeypatch):
    recorder = ExecutionRecorder(tmp_path / "h")
    monkeypatch.setattr(recorder_mod, "redact_record", lambda record, **kw: (_ for _ in ()).throw(ValueError("redaction blew up")))
    recorder.begin("t", {})
    assert_contract(*drive_all_hooks(recorder))
    assert "redaction blew up" in recorder.error


def test_a_keyboard_interrupt_inside_a_recorder_hook_is_the_users_and_propagates(tmp_path, monkeypatch):
    recorder = ExecutionRecorder(tmp_path / "h")
    monkeypatch.setattr(recorder_mod, "_append", lambda path, text: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        recorder.before_run(RunContext(task="t", agent_name="a"))
    assert recorder.error is None


def test_the_cli_run_is_identical_when_the_disk_fails_midway(monkeypatch, capsys, tmp_path):
    config = write_config(tmp_path)
    code, out, err, plain = run_cli(monkeypatch, capsys, tmp_path, FLOW, config=config)
    (tmp_path / "ws" / "hello.txt").unlink()
    real, writes = recorder_mod._append, []

    def disk_full(path, text):
        writes.append(text)
        if len(writes) > 6:
            raise OSError(28, "No space left on device")
        return real(path, text)

    monkeypatch.setattr(recorder_mod, "_append", disk_full)
    code2, out2, err2, broken = run_cli(monkeypatch, capsys, tmp_path, FLOW, "--history-dir", str(tmp_path / "h"), "--capture-prompts", config=config)
    assert (code2, out2) == (code, out) == (0, "all done\n")
    assert prompts_seen(broken) == prompts_seen(plain)
    assert (tmp_path / "ws" / "hello.txt").read_text() == "hi\n"  # the task itself still did its work
    assert "history recording stopped early (the run itself was not affected): OSError: [Errno 28] No space left on device" in err2
    assert len(writes) == 7  # the first failure ended all recording


def test_a_history_dir_that_cannot_be_created_warns_and_the_run_proceeds(monkeypatch, capsys, tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("a file where the history folder should go")
    code, out, err, fake = run_cli(monkeypatch, capsys, tmp_path, FLOW, "--history-dir", str(blocker))
    assert (code, out) == (0, "all done\n")
    assert "history recording is OFF, nothing will be recorded" in err and blocker.read_text().startswith("a file")


def test_a_failed_task_and_a_run_that_raises_keep_their_exit_codes_and_are_recorded(monkeypatch, capsys, tmp_path):
    stuck = {BIG: [delegate("writer", "x")] + [call("file_operations", operation="read", path="a.txt")] * 8, SMALL: WORKER_FLOW[2:]}
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "a.txt").write_text("x\n")
    code, out, err, fake = run_cli(monkeypatch, capsys, tmp_path, stuck, "--history-dir", str(tmp_path / "hist"), "--json")
    assert code == 1 and json.loads(out)["success"] is False
    conn = import_into(history_dir_of(err), tmp_path / "h.db")
    assert rows(conn, "SELECT status, agent_completed, task_verified FROM sessions") == [{"status": "finished", "agent_completed": 0, "task_verified": None}]
    # an exception out of Harness.run: exit 4, the session is recorded as aborted with the exception type
    monkeypatch.setattr(models_mod, "build_model", CliModels().build)
    monkeypatch.setattr(cli.Harness, "run", lambda self, task: (self.recorder.begin(task, {}), self.recorder.abort(RuntimeError("boom")), (_ for _ in ()).throw(RuntimeError("boom")))[2])
    code2 = cli.main(["run", "--config", str(write_config(tmp_path)), "--task", "t", "--history-dir", str(tmp_path / "hist2")])
    err2 = capsys.readouterr().err
    assert code2 == 4
    assert [e["error"] for e in events(history_dir_of(err2)) if e["event"] == "attempt_aborted"] == ["RuntimeError: boom"]


# ---- 8. Ctrl-C and a killed process -----------------------------------------------------------------------------------------------------------
def test_ctrl_c_in_the_orchestrator_is_an_aborted_session_and_exit_130(monkeypatch, capsys, tmp_path):
    class Interrupting(ScriptedModel):
        def generate(self, prompt, config=None, **kw):
            if len(self.calls) == 1:
                raise KeyboardInterrupt
            return super().generate(prompt, config, **kw)

    fake = CliModels()
    fake.scripts.update(FLOW)
    fake.custom[BIG] = lambda: Interrupting(FLOW[BIG], name=BIG)  # first turn delegates; the second generation is interrupted
    monkeypatch.setattr(models_mod, "build_model", fake.build)
    code = cli.main(["run", "--config", str(write_config(tmp_path)), "--task", "t", "--history-dir", str(tmp_path / "hist")])
    err = capsys.readouterr().err
    assert code == 130 and "interrupted" in err
    directory = history_dir_of(err)
    conn = import_into(directory, tmp_path / "h.db")
    assert rows(conn, "SELECT status, error FROM sessions") == [{"status": "aborted", "error": "KeyboardInterrupt"}]
    assert rows(conn, "SELECT status FROM tool_calls WHERE tool = 'delegate'") == [{"status": "returned"}]  # the delegation had finished before the interrupt
    assert [m["status"] for m in rows(conn, "SELECT status FROM messages")][-1] == "ok" and rows(conn, "SELECT type FROM events WHERE type = 'generation_start'")  # the interrupted call: started, never completed


DRIVER = r'''
import json, os, signal, sys, time
sys.path.insert(0, {tests!r})
signal.signal(signal.SIGINT, signal.default_int_handler)  # a test runner may have left SIGINT ignored
from conftest import ScriptedModel
from effgen import AgentConfig, tool
mode = {mode!r}
hist, marker, stop = {hist!r}, {marker!r}, {stop!r}

class Blocking(ScriptedModel):
    """Stands in for a slow worker model: announces that it is running, then waits until told to stop (then raises, so the worker ends)."""
    def generate(self, prompt, config=None, **kw):
        open(marker, "w").write("running")
        deadline = time.time() + 60
        while not os.path.exists(stop) and time.time() < deadline:
            time.sleep(0.05)
        raise RuntimeError("stopped by the test")

if mode == "cli":
    import smol_harness.models as models_mod
    scripts = {scripts!r}
    models_mod.build_model = lambda spec: (Blocking if spec.model == {block!r} else ScriptedModel)(scripts[spec.model], name=spec.model)
    from smol_harness.__main__ import main
    sys.exit(main(["run", "--config", {config!r}, "--task", "interrupt me", "--history-dir", hist]))
else:
    from smol_harness.harness import build_harness, initialize_runtime
    from smol_harness.recorder import ExecutionRecorder
    initialize_runtime()

    @tool
    def hang(path: str) -> str:
        """Never returns."""
        open(marker, "w").write("running")
        time.sleep(120)
        return "never"

    def cfg(model, name, tools=()):
        return AgentConfig(name=name, model=model, tools=list(tools), tool_calling_mode="react", capability_probe=False, enable_memory=False, max_iterations=6, raise_on_error=False)
    orch = ScriptedModel(['Thought: go.\nAction: delegate\nAction Input: {{"worker": "coder", "task": "hang forever", "context": "ctx"}}'], name="orch")
    work = ScriptedModel(['Thought: go.\nAction: hang\nAction Input: {{"path": "a"}}'], name="work")
    harness = build_harness(cfg(orch, "orchestrator"), {{"coder": cfg(work, "coder", [hang])}}, recorder=ExecutionRecorder(hist))
    harness.run("kill me")
'''


def start_driver(tmp_path, mode, **fields):
    marker, stop = tmp_path / "running", tmp_path / "stop"
    code = DRIVER.format(tests=str(ROOT / "tests"), mode=mode, hist=str(tmp_path / "hist"), marker=str(marker), stop=str(stop), scripts=fields.get("scripts"),
                         block=fields.get("block"), config=fields.get("config"))
    process = subprocess.Popen([sys.executable, "-I", "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmp_path)
    deadline = time.time() + 60
    while not marker.exists():
        assert process.poll() is None and time.time() < deadline, process.communicate()
        time.sleep(0.05)
    return process, stop


def test_a_real_sigint_during_a_delegation_exits_130_with_an_aborted_session_and_the_worker_still_named(tmp_path):
    config = write_config(tmp_path)
    process, stop = start_driver(tmp_path, "cli", scripts={BIG: [delegate("writer", "do the thing"), FINAL], SMALL: [FINAL]}, block=SMALL, config=str(config))
    process.send_signal(signal.SIGINT)
    stop.write_text("go")  # let the blocked worker thread end so the interpreter can exit
    out, err = process.communicate(timeout=60)
    assert process.returncode == 130 and "interrupted" in err, err
    directory = history_dir_of(err)
    conn = import_into(directory, tmp_path / "h.db")
    assert rows(conn, "SELECT status, error FROM sessions") == [{"status": "aborted", "error": "KeyboardInterrupt"}]
    d = rows(conn, "SELECT worker, task, delegation_id, confirmed, status FROM delegations")
    assert d == [{"worker": "writer", "task": "do the thing", "delegation_id": None, "confirmed": 0, "status": "unfinished"}]  # identity and task kept, id NOT invented
    assert any(e["event"] == "generation_start" and e["role"] == "worker" and e["agent_name"] == "writer" for e in events(directory))
    assert "UNFINISHED" in "\n".join(history.timeline_lines(conn, rows(conn, "SELECT session_id FROM sessions")[0]["session_id"]))


def test_a_killed_process_leaves_an_unfinished_delegation_whose_worker_and_task_are_known(tmp_path):
    process, _ = start_driver(tmp_path, "tool")
    process.kill()  # SIGKILL: nothing more can be written
    process.communicate(timeout=30)
    assert process.returncode == -signal.SIGKILL
    directory = next((tmp_path / "hist").iterdir())
    conn = import_into(directory, tmp_path / "h.db")
    assert rows(conn, "SELECT status, end_seq FROM sessions") == [{"status": "unfinished", "end_seq": None}]
    assert rows(conn, "SELECT worker, task, delegation_id, confirmed, status FROM delegations") == [
        {"worker": "coder", "task": "hang forever", "delegation_id": None, "confirmed": 0, "status": "unfinished"}]
    hang = rows(conn, "SELECT role, agent_name, delegation_id, status, requested_by_message_id IS NOT NULL linked, success FROM tool_calls WHERE tool = 'hang'")
    assert hang == [{"role": "worker", "agent_name": "coder", "delegation_id": None, "status": "unfinished", "linked": 1, "success": None}]
    assert rows(conn, "SELECT COUNT(*) n FROM tool_calls WHERE status = 'unfinished'")[0]["n"] == 2  # the delegate call and the hung tool: neither counts as an execution
    text = "\n".join(history.timeline_lines(conn, rows(conn, "SELECT session_id FROM sessions")[0]["session_id"]))
    assert text.count("UNFINISHED") == 2 and "TOOL END   hang" not in text


# ---- 9. opt-in prompt capture, hashing, permissions -----------------------------------------------------------------------------------------------
def test_prompts_are_not_captured_unless_asked(monkeypatch, capsys, tmp_path):
    code, out, err, fake = run_cli(monkeypatch, capsys, tmp_path, FLOW, "--history-dir", str(tmp_path / "hist"))
    directory = history_dir_of(err)
    assert code == 0 and sorted(p.name for p in directory.iterdir()) == ["attempts.jsonl", "trace.jsonl"] and "prompts captured" not in err
    delegate_description = "Delegate a self-contained task to a worker agent"  # part of every orchestrator prompt, nowhere else
    assert any(delegate_description in p for p in prompts_seen(fake)) and delegate_description not in (directory / "trace.jsonl").read_text()
    conn = import_into(directory, tmp_path / "h.db")
    assert {r["prompt_integrity"] for r in rows(conn, "SELECT prompt_integrity FROM messages")} == {"not_captured"}
    assert all(r["prompt_text"] is None for r in rows(conn, "SELECT prompt_text FROM messages"))
    assert all(r["prompt_sha256"] and r["prompt_chars"] for r in rows(conn, "SELECT prompt_sha256, prompt_chars FROM messages"))  # the hash and size are still evidence


def test_captured_prompts_are_exactly_what_the_models_received_and_hash_to_the_trace(recorded):
    captured = [json.loads(line) for line in (recorded.dir / "prompts.local.jsonl").read_text().splitlines()]
    assert sorted(c["text"] for c in captured) == sorted(prompts_seen(recorded.fake))  # byte for byte the adapter input, every call
    by_seq = {e["seq"]: e for e in recorded.events if e["event"] == "generation"}
    assert len(captured) == len(by_seq) == 6
    for c in captured:
        assert c["sha256"] == prompt_sha256(c["text"]) == by_seq[c["seq"]]["prompt_sha256"] and c["chars"] == len(c["text"]) == by_seq[c["seq"]]["prompt_chars"]
    assert {r["prompt_integrity"] for r in rows(recorded.conn, "SELECT prompt_integrity FROM messages")} == {"verified"}
    assert "prompts captured in prompts.local.jsonl: private" in recorded.err


def test_a_tampered_captured_prompt_is_reported_as_a_mismatch(recorded, tmp_path):
    path = recorded.dir / "prompts.local.jsonl"
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["text"] += " tampered"
    path.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")
    conn = import_into(recorded.dir, tmp_path / "again.db")
    assert [r["prompt_integrity"] for r in rows(conn, "SELECT prompt_integrity FROM messages ORDER BY seq")].count("mismatch") == 1
    assert "prompt_hash_mismatch" in {r["kind"] for r in rows(conn, "SELECT kind FROM issues")}


@pytest.mark.parametrize("umask", [0o022, 0o000])
def test_history_files_are_private_whatever_the_umask(tmp_path, umask):
    old = os.umask(umask)
    try:
        recorder = ExecutionRecorder(tmp_path / "h", capture_prompts=True)
        recorder.begin("t", {})
        recorder.finish(SimpleNamespace(success=True, output="o", termination="done"))
    finally:
        os.umask(old)
    mode = lambda p: stat.S_IMODE(p.stat().st_mode)  # noqa: E731
    assert mode(recorder.dir) == 0o700
    assert [mode(recorder.dir / n) for n in ("trace.jsonl", "attempts.jsonl", "prompts.local.jsonl")] == [0o600] * 3
    assert recorder.prompts_path.name.endswith(".local.jsonl")  # the suffix the Git ignore list, the archiver and the packager key on


def test_captured_prompts_stay_out_of_git_and_out_of_the_experiment_archive():
    assert drv.exclusion_reason(Path("x/prompts.local.jsonl")) is not None
    ignored = subprocess.run(["git", "check-ignore", "-q", "hist/run/prompts.local.jsonl"], cwd=ROOT).returncode
    assert ignored == 0


# ---- 10. the shared redaction policy ------------------------------------------------------------------------------------------------------------------
SAMPLES = ["token hf_" + "a1" * 15, "Authorization: Bearer abcdef0123456789xyz", "https://user:hunter2pw@host.example/x", "API_KEY=sk-" + "b2" * 12, "-----BEGIN RSA PRIVATE KEY-----\nMIIEv\n-----END RSA PRIVATE KEY-----",
           "password: swordfish123", f"path {Path.home()}/project/file.py", "ghp_" + "c3" * 15, "nothing to hide here"]


def test_the_experiment_driver_and_the_recorder_share_one_redaction_implementation():
    assert drv.redact_obj.__code__ is redaction.redact_obj.__code__ or drv.redact_obj.__module__ == "smol_harness_redaction"
    assert drv._REDACTIONS is not None and [p.pattern for p, _ in drv._REDACTIONS] == [p.pattern for p, _ in redaction._REDACTIONS]
    for sample in SAMPLES:
        assert drv.redact(sample) == redaction.redact(sample)
        assert drv.redact_obj({"k": sample, "n": [sample]}) == redaction.redact_obj({"k": sample, "n": [sample]})
    assert "hunter2pw" not in redaction.redact(SAMPLES[2]) and "~/project" in redaction.redact(SAMPLES[6]) and redaction.redact(SAMPLES[-1]) == SAMPLES[-1]


def test_the_experiment_driver_still_starts_without_the_project_environment(tmp_path):
    done = subprocess.run([sys.executable, "-I", str(ROOT / "experiments" / "rtx4070" / "rtx4070.py"), "--help"], capture_output=True, text=True, cwd=tmp_path)
    assert done.returncode == 0 and "preflight" in done.stdout, done.stderr[-500:]  # the module is loaded by path: no installed smol_harness needed


def test_secrets_are_redacted_in_the_trace_and_kept_only_in_the_private_prompts(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("SMOL_TEST_SECRET", "env-secret-value-98765")
    token = "hf_" + "Z9" * 15
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "notes.txt").write_text(f"deploy with {token} and env-secret-value-98765\nhttps://bob:pa55word@host/x\n")
    scripts = {BIG: [delegate("writer", f"read notes.txt, my token is {token}"), FINAL], SMALL: [call("file_operations", operation="read", path="notes.txt"), "Thought: ok.\nFinal Answer: " + token]}
    code, out, err, fake = run_cli(monkeypatch, capsys, tmp_path, scripts, "--history-dir", str(tmp_path / "hist"), "--capture-prompts")
    directory = history_dir_of(err)
    shared = (directory / "trace.jsonl").read_text() + (directory / "attempts.jsonl").read_text()
    for secret in (token, "env-secret-value-98765", "pa55word"):
        assert secret not in shared, secret
    assert "hf_***" in shared and "***" in shared
    for line in shared.splitlines():
        json.loads(line)  # redaction left every line valid JSON
    assert token in (directory / "prompts.local.jsonl").read_text()  # the private file is the exact adapter input, by design
    db = import_into(directory, tmp_path / "h.db")
    assert token not in json.dumps([dict(r) for r in db.execute("SELECT task, final_answer, consumed_output FROM messages JOIN sessions USING (session_id)")])


# ---- 11. backward compatibility with Phase 1 -------------------------------------------------------------------------------------------------------------
def test_phase1_tool_call_lines_keep_their_statuses_and_only_an_explicit_null_means_returned(tmp_path):
    base = {"run_id": "r", "stage": "B", "invocation": "inv0", "attempt": "a1", "t_utc": "2026-10-27T00:00:00Z", "role": "orchestrator"}
    lines = [dict(base, seq=1, event="attempt_start", task="t"),
             dict(base, seq=2, event="tool_call", tool="a", success=True), dict(base, seq=3, event="tool_call", tool="b", success=False),
             dict(base, seq=4, event="tool_call", tool="c"), dict(base, seq=5, event="tool_call", tool="d", success=True, exception=True),
             dict(base, seq=6, event="tool_call", tool="e", success=None), dict(base, seq=7, event="attempt_end", agent_completed=True, task_verified=True)]
    trace = tmp_path / "trace.jsonl"
    trace.write_text("".join(json.dumps(line) + "\n" for line in lines))
    conn = history.connect(tmp_path / "p1.db")
    history.import_source(conn, trace, "2026-10-27T00:00:00Z")
    assert {r["tool"]: r["status"] for r in rows(conn, "SELECT tool, status FROM tool_calls")} == {"a": "completed", "b": "failed", "c": "failed", "d": "failed", "e": "returned"}


def test_views_are_refreshed_on_open_so_an_older_database_gains_the_returned_column(tmp_path):
    db = tmp_path / "old.db"
    conn = history.connect(db)
    conn.execute("DROP VIEW tool_counts")
    conn.execute("CREATE VIEW tool_counts AS SELECT session_id, tool, role, SUM(status = 'completed') AS completed, SUM(status = 'failed') AS failed, SUM(status = 'unfinished') AS unfinished FROM tool_calls GROUP BY session_id, tool, role")
    conn.close()
    again = history.connect(db)
    assert "returned" in [c[1] for c in again.execute("PRAGMA table_info(tool_counts)")]
    assert again.execute("PRAGMA user_version").fetchone()[0] == history.SCHEMA_VERSION == 1  # derived views changed, the schema version did not
    history.connect(db).close()  # opening again leaves the views alone


def test_a_trace_with_a_dangling_parent_is_imported_with_an_issue_not_dropped(tmp_path):
    recorder = ExecutionRecorder(tmp_path / "h")
    recorder.begin("t", {})
    run = RunContext(task="x", agent_name="w")
    recorder_mod._PARENT.set({"tool_call_id": "someone-else:99", "worker": "w"})  # a worker whose delegate call is not in this trace
    try:
        recorder.before_run(run)
    finally:
        recorder_mod._PARENT.set(None)
    recorder.before_tool_call(ToolCallContext(tool_name="file_operations", tool_input="{}", run=run))
    conn = import_into(recorder.dir, tmp_path / "d.db")
    assert "parent_tool_call_unresolved" in {r["kind"] for r in rows(conn, "SELECT kind FROM issues")}
    assert rows(conn, "SELECT COUNT(*) n FROM tool_calls")[0]["n"] == 1


# ---- review remediation (Opus review of Phase 2): B1 attribution, B2 escaped secrets, lifecycle safeguards ---------------------------------------
def cfg_agent(model, name, tools=()):
    return AgentConfig(name=name, model=model, tools=list(tools), tool_calling_mode="react", capability_probe=False, enable_memory=False, max_iterations=6, raise_on_error=False)


def test_b1_a_worker_whose_delegation_context_is_lost_is_unknown_never_orchestrator(tmp_path):
    """Harness.run called from async code: effGen bridges the tool coroutine through a bare thread (effgen/utils/async_bridge.py), so the
    delegation ContextVar never reaches the worker. The worker must not be recorded as a second orchestrator, nor be given an invented parent."""
    import asyncio

    orchestrator = ScriptedModel(['Thought: x\nAction: delegate\nAction Input: {"worker": "coder", "task": "t"}', FINAL], name="orch-model")
    worker = ScriptedModel(["Thought: d\nFinal Answer: w"], name="worker-model")
    recorder = ExecutionRecorder(tmp_path / "h")
    harness = build_harness(cfg_agent(orchestrator, "orchestrator"), {"coder": cfg_agent(worker, "coder")}, recorder=recorder)

    async def from_async_code():
        assert asyncio.get_running_loop() is not None  # a real running loop: the path the review reproduced
        return harness.run("task")

    result = asyncio.run(from_async_code())
    assert result.success and recorder.error is None
    lines = events(recorder.dir)
    runs = [(e["role"], e["agent_name"], e.get("parent_tool_call_id")) for e in lines if e["event"] == "agent_run_start"]
    assert runs == [("orchestrator", "orchestrator", None), ("unknown", "coder", None)]
    assert [e["role"] for e in lines if e["event"] == "generation" and e["agent_name"] == "coder"] == ["unknown"]
    flagged = [e for e in lines if e["event"] == "attribution_unknown"]
    assert len(flagged) == 1 and flagged[0]["agent_name"] == "coder" and flagged[0]["role"] == "unknown"
    conn = import_into(recorder.dir, tmp_path / "h.db")
    assert rows(conn, "SELECT role, agent_name, delegation_id FROM messages ORDER BY seq") == [
        {"role": "orchestrator", "agent_name": "orchestrator", "delegation_id": None}, {"role": "unknown", "agent_name": "coder", "delegation_id": None},
        {"role": "orchestrator", "agent_name": "orchestrator", "delegation_id": None}]
    assert rows(conn, "SELECT worker, delegation_id, confirmed FROM delegations") == [{"worker": "coder", "delegation_id": "d1", "confirmed": 1}]  # the report still confirms d1
    assert "attribution_unknown" in {r["kind"] for r in rows(conn, "SELECT kind FROM issues")}


def test_only_the_first_run_after_begin_is_the_orchestrator(tmp_path):
    recorder = ExecutionRecorder(tmp_path / "h")
    recorder.before_run(RunContext(task="t", agent_name="before-any-session"))  # no session yet: not an orchestrator either
    recorder.begin("s", {})
    for name in ("orchestrator", "stray-1", "stray-2"):
        recorder.before_run(RunContext(task="t", agent_name=name))
    recorder.begin("s2", {})
    recorder.before_run(RunContext(task="t", agent_name="orchestrator"))
    assert [(e["role"], e["agent_name"]) for e in events(recorder.dir) if e["event"] == "agent_run_start"] == [
        ("unknown", "before-any-session"), ("orchestrator", "orchestrator"), ("unknown", "stray-1"), ("unknown", "stray-2"), ("orchestrator", "orchestrator")]


def test_late_worker_events_keep_the_session_they_started_in(tmp_path):
    """A worker that outlives its session (Ctrl-C cannot stop its thread) must not write into the next session."""
    recorder = ExecutionRecorder(tmp_path / "h")
    recorder.begin("one", {})
    orchestrator = RunContext(task="one", agent_name="orchestrator")
    recorder.before_run(orchestrator)
    tool = ToolCallContext(tool_name="delegate", tool_input='{"worker": "w", "task": "x"}', run=orchestrator)
    recorder.before_tool_call(tool)
    started, go_on = threading.Event(), threading.Event()

    def late_worker():
        run = RunContext(task="x", agent_name="w")
        recorder.before_run(run)
        started.set()
        go_on.wait(timeout=10)
        model = ModelCallContext(prompt="p", model_name="m", run=run)
        recorder.before_model_call(model)
        recorder.after_model_call(model, {"text": "late"})

    thread = threading.Thread(target=contextvars.copy_context().run, args=(late_worker,))
    thread.start()
    started.wait(timeout=10)
    recorder.abort(KeyboardInterrupt())
    recorder.begin("two", {})
    recorder.before_run(RunContext(task="two", agent_name="orchestrator"))
    go_on.set()
    thread.join(timeout=10)
    late = [e for e in events(recorder.dir) if e.get("agent_name") == "w"]
    assert late and {e["attempt"] for e in late} == {"run1"}
    conn = import_into(recorder.dir, tmp_path / "h.db")
    assert rows(conn, "SELECT session_id, agent_name FROM messages WHERE agent_name = 'w'")[0]["session_id"].endswith(":run1")
    assert rows(conn, "SELECT COUNT(*) n FROM messages WHERE session_id LIKE '%:run2'")[0]["n"] == 0


def test_the_importer_refuses_a_parent_link_into_another_session(tmp_path):
    base = {"run_id": "r", "stage": "cli", "invocation": "inv1", "t_utc": "2026-10-27T00:00:00Z"}
    lines = [dict(base, seq=1, attempt="run1", event="attempt_start", task="a"),
             dict(base, seq=2, attempt="run1", event="tool_call_start", role="orchestrator", agent_name="orchestrator", tool="delegate", args={"worker": "w", "task": "x"}),
             dict(base, seq=3, attempt="run1", event="tool_call", role="orchestrator", tool="delegate", tool_call_id="inv1:2", success=None, delegation_id="d1", result="{}"),
             dict(base, seq=4, attempt="run2", event="attempt_start", task="b"),
             dict(base, seq=5, attempt="run2", event="generation", role="worker", agent_name="w", parent_tool_call_id="inv1:2", consumed_output="x")]
    (tmp_path / "trace.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    conn = history.connect(tmp_path / "x.db")
    history.import_source(conn, tmp_path / "trace.jsonl", "2026-10-27T00:00:00Z")
    assert rows(conn, "SELECT session_id, agent_name, delegation_id FROM messages") == [{"session_id": "inv1:run2", "agent_name": "w", "delegation_id": None}]
    assert "parent_in_other_session" in {r["kind"] for r in rows(conn, "SELECT kind FROM issues")}


def test_ctrl_c_while_the_history_folder_is_being_created_still_exits_130(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "ExecutionRecorder", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    code, out, err, fake = run_cli(monkeypatch, capsys, tmp_path, FLOW, "--history-dir", str(tmp_path / "h"))
    assert code == 130 and "interrupted" in err and fake.loads == []


def test_recording_only_metadata_that_fails_does_not_touch_the_run(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli.Harness, "_model_names", lambda self: 1 / 0)
    code, out, err, fake = run_cli(monkeypatch, capsys, tmp_path, FLOW, "--history-dir", str(tmp_path / "h"))
    assert (code, out) == (0, "all done\n") and "recording stopped" not in err
    assert [e["model"] for e in events(history_dir_of(err)) if e["event"] == "attempt_start"] == [None]


ESCAPED_SECRETS = {"SMOL_QUOTE_TOKEN": 'pa"ss\\word-123456', "SMOL_CTRL_SECRET": "tab\there\nnew\x01line-99", "SMOL_UNI_PASSWORD": "uniécode sep-7777"}


def assert_no_secret(path: Path, secrets) -> None:
    """Neither the raw bytes nor the JSON-escaped form nor any decoded value contains a secret."""
    text = path.read_text()
    for secret in secrets:
        assert secret not in text and json.dumps(secret)[1:-1] not in text and json.dumps(secret, ensure_ascii=False)[1:-1] not in text, (path.name, secret)

    def walk(value):
        if isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        elif isinstance(value, str):
            assert not any(s in value for s in secrets), (path.name, value[:80])

    for line in jsonl_lines(path):
        walk(json.loads(line))  # every line still valid JSON


def test_b2_escaped_secrets_are_redacted_in_the_recorder_trace_and_attempts_but_kept_exactly_in_private_prompts(tmp_path, monkeypatch):
    for name, value in ESCAPED_SECRETS.items():
        monkeypatch.setenv(name, value)
    secrets = list(ESCAPED_SECRETS.values())
    recorder = ExecutionRecorder(tmp_path / "h", capture_prompts=True)
    recorder.begin("task with " + secrets[0], {"orchestrator": secrets[1]})
    run = RunContext(task="t", agent_name="orchestrator")
    recorder.before_run(run)
    prompt = "prompt holding " + " | ".join(secrets)
    model = ModelCallContext(prompt=prompt, model_name="m", run=run)
    recorder.before_model_call(model)
    recorder.after_model_call(model, {"text": "answer " + secrets[2]})
    tool = ToolCallContext(tool_name="file_operations", tool_input=json.dumps({"path": secrets[0], "content": secrets[1]}), run=run)
    recorder.before_tool_call(tool)
    recorder.after_tool_call(tool, "read: " + " ".join(secrets))
    recorder.after_run(run, SimpleNamespace(success=True))
    recorder.finish(SimpleNamespace(success=True, output="final " + secrets[0], termination="done"))
    assert recorder.error is None
    assert_no_secret(recorder.dir / "trace.jsonl", secrets)
    assert_no_secret(recorder.dir / "attempts.jsonl", secrets)
    captured = [json.loads(line) for line in jsonl_lines(recorder.prompts_path)]
    assert [c["text"] for c in captured] == [prompt] and captured[0]["sha256"] == prompt_sha256(prompt)  # exact, unredacted, hash intact
    gen = next(e for e in events(recorder.dir) if e["event"] == "generation")
    assert gen["prompt_sha256"] == prompt_sha256(prompt) and gen["event"] == "generation" and "consumed_output" in gen  # same schema, the hash of the exact text
    assert drv.exclusion_reason(recorder.prompts_path) is not None  # never packaged


def test_b2_escaped_secrets_are_redacted_in_the_experiment_runner_trace_and_attempts(tmp_path, monkeypatch):
    import stage_runner as sr

    for name, value in ESCAPED_SECRETS.items():
        monkeypatch.setenv(name, value)
    secrets = list(ESCAPED_SECRETS.values())
    monkeypatch.setattr(sr, "TRACE", sr.fresh_trace(path=tmp_path / "trace.jsonl", stage="K", run_id="r", invocation="inv-b2", attempt="a1"))
    monkeypatch.setitem(sr.SINK, "path", tmp_path / "attempts.jsonl")
    sr.write_event("tool_call", tool="file_operations", args={"path": secrets[0]}, result="read: " + " ".join(secrets), success=True)
    sr.keep({"attempt_id": "a1", "kind": "t", "success": True, "final_answer": secrets[1], "verification": sr.verdict(True, [("c", True, secrets[2])])})
    assert_no_secret(tmp_path / "trace.jsonl", secrets)
    assert_no_secret(tmp_path / "attempts.jsonl", secrets)
    assert [json.loads(line)["event"] for line in jsonl_lines(tmp_path / "trace.jsonl")] == ["tool_call", "attempt_end"]


def test_redact_record_is_the_shared_structured_then_textual_policy():
    secret = 'abc"def\\ghi-123456'
    os.environ["SMOL_SHARED_TOKEN"] = secret
    try:
        record = {"result": "x " + secret, "nested": [{"v": secret}], "api_key": "plain-value-123456", "n": 3, "obj": Path("/p")}
        text = redaction.redact_record(record)
        assert secret not in json.loads(text)["result"] and json.loads(text)["nested"] == [{"v": "***"}] and json.loads(text)["api_key"] == "***"
        assert json.loads(text)["n"] == 3 and json.loads(text)["obj"] == "/p"
        assert drv.redact_record is not None and drv.redact_record(record) == text
    finally:
        del os.environ["SMOL_SHARED_TOKEN"]
