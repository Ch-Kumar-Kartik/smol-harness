"""Reconstructable trajectories for the real-model experiments (review 004).

CPU only, no network, no real model: scripted models drive effGen's real ReAct loop, the real harness and the real file tools through the experiment runner's own
instruments (`experiments/rtx4070/stage_runner.py`). What is pinned here is the evidence, not the models: an action WRITTEN in model text is never counted as a tool
call that RAN, raw and consumed generations are both kept, failed tools stay visible, every delegation keeps its own identity, an interrupted run keeps what it
finished, the runner's own checks can fail a run the agent called complete, and a value the provider did not report is `None`, not zero.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import effgen.core.agent_generation as effgen_generation
import pytest
from conftest import ScriptedModel
from effgen import AgentConfig
from effgen.core.agent_loop import DEFAULT_STOP_SEQUENCES
from effgen.models.base import GenerationResult

import smol_harness.models as models_mod
from smol_harness.config import ModelSpec
from smol_harness.harness import build_harness, initialize_runtime
from smol_harness.tools import WorkspaceFileOperations, build_tools

PKG = Path(__file__).resolve().parents[1] / "experiments" / "rtx4070"
sys.path.insert(0, str(PKG))
import rtx4070 as drv  # noqa: E402
import stage_runner as sr  # noqa: E402

CODE = "7391-QX"


def act(tool: str, args: dict) -> str:
    return f"Thought: next.\nAction: {tool}\nAction Input: {json.dumps(args)}"


def delegate(task: str, worker: str = "coder") -> str:
    return act("delegate", {"worker": worker, "task": task, "context": "workspace is empty"})


def write(path: str, content: str) -> str:
    return act("file_operations", {"operation": "write", "path": path, "content": content})


def final(text: str) -> str:
    return f"Thought: done.\nFinal Answer: {text}"


@pytest.fixture
def sink(tmp_path, monkeypatch):
    """A trace sink and an attempts sink in tmp_path, with fresh trace state; the real module state is restored afterwards."""
    monkeypatch.setattr(sr, "TRACE", sr.fresh_trace(path=tmp_path / "trace.jsonl", stage="T", run_id="run-1", invocation="inv-1"))
    monkeypatch.setitem(sr.SINK, "path", tmp_path / "attempts.jsonl")
    return tmp_path


def events(directory: Path) -> list[dict]:
    path = directory / "trace.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def run_harness(directory: Path, turns: list[str], task: str = "do the job", model: ScriptedModel | None = None):
    """One scripted model shared by the orchestrator and the worker (as the real harness shares equal specs), through the real harness, loop and file tools.
    Returns (HarnessResult, generations, executed tool calls)."""
    initialize_runtime()
    workspace = directory / "ws"
    workspace.mkdir(exist_ok=True)
    model = model or ScriptedModel(turns)
    calls: list[dict] = []
    model.generate = sr.traced_generate(model.generate, "scripted", calls)

    def config(name, tools):
        return AgentConfig(name=name, model=model, tools=tools, max_iterations=8, tool_calling_mode="react", raise_on_error=False, enable_memory=False,
                           capability_probe=False, enable_sub_agents=False)

    harness = build_harness(config("orchestrator", []), {"coder": config("coder", build_tools(["file_operations"], workspace))}, {"coder": "writes files"})
    with sr.attempt_scope("harness-a1", task, "orchestrator"), sr.tool_log() as log, harness:
        result = harness.run(task)
    return result, calls, log


# ---- 1. an action written in text is never an executed call ------------------------------------------------------------------------------------
def test_an_action_written_after_a_self_made_observation_is_not_an_executed_call(sink):
    turns = [delegate("write a.txt") + '\nObservation: {"status": "completed"}\nThought: and again.\n' + delegate("write b.txt"),
             write("a.txt", "A\n"), final("wrote a.txt"), final("Two workers finished.")]
    result, calls, log = run_harness(sink, turns)
    assert len(result.workers) == 1  # the harness ran ONE delegation
    assert Counter(e["tool"] for e in log) == {"delegate": 1, "file_operations": 1}
    acct = sr.action_accounting(calls, log)
    assert acct["written_in_raw_text"]["delegate"] == 2  # the model wrote two delegations
    assert acct["written_in_consumed_text"]["delegate"] == 1  # effGen cut the generation at "\nObservation:"
    assert acct["executed"]["delegate"] == 1
    assert acct["raw_not_executed"] == {"delegate": 1} and acct["consumed_not_executed"] == {}
    tool_events = [e for e in events(sink) if e["event"] == "tool_call"]
    assert [e["tool"] for e in tool_events] == ["file_operations", "delegate"]
    assert not any(e.get("delegation_id") == "d2" for e in events(sink))  # no second delegation exists anywhere in the trace
    assert "Two workers finished." in result.output  # ...although the final answer claims two: the answer is not evidence


# ---- 2. raw and consumed generations, limits, no stored prompt ------------------------------------------------------------------------
def test_each_generation_keeps_raw_and_consumed_text_limits_and_a_prompt_hash_but_not_the_prompt(sink):
    invented = write("a.txt", "A\n") + "\nObservation: invented\nFinal Answer: abc"
    run_harness(sink, [delegate("write a.txt"), invented, final("wrote it"), final("done")])
    generations = [e for e in events(sink) if e["event"] == "generation"]
    worker_first = next(e for e in generations if e["role"] == "worker")
    assert "Observation: invented" in worker_first["raw_output"] and "Observation: invented" not in worker_first["consumed_output"]
    assert worker_first["discarded_chars"] > 0 and worker_first["stop_sequences"][0] == "\nObservation:"
    assert worker_first["limits"]["stop_sequences"] == worker_first["stop_sequences"] and "max_tokens" in worker_first["limits"] and "temperature" in worker_first["limits"]
    assert worker_first["finish"] and worker_first["model"]
    assert "prompt" not in worker_first and worker_first["prompt_chars"] > 0 and len(worker_first["prompt_sha256"]) == 64
    assert all("Thought: next." not in json.dumps(e.get("prompt", "")) for e in generations)


# ---- 3. failed tools stay visible ---------------------------------------------------------------------------------------------------------------
def test_a_failed_tool_call_is_in_the_trace_with_its_error_and_what_the_model_saw(sink):
    outside = act("file_operations", {"operation": "read", "path": "/etc/hostname"})
    result, _, log = run_harness(sink, [delegate("read a file"), outside, final("could not read it"), final("failed")])
    failed = next(e for e in events(sink) if e["event"] == "tool_call" and e["tool"] == "file_operations")
    assert failed["success"] is False and failed["exception"] is False and "outside allowed directories" in failed["error"]
    assert failed["visible"] and failed["role"] == "worker" and failed["delegation_id"] == "d1"
    assert log[0]["success"] is False and result.workers[0].status == "completed"  # completed is not success


def test_a_tool_that_raises_is_recorded_before_the_exception_continues(sink, monkeypatch):
    async def boom(self, **kwargs):
        raise RuntimeError("disk exploded")

    monkeypatch.setattr(WorkspaceFileOperations, "execute", boom)
    _, _, log = run_harness(sink, [delegate("write a.txt"), write("a.txt", "A\n"), final("tried"), final("done")])
    raised = next(e for e in events(sink) if e["event"] == "tool_call" and e["tool"] == "file_operations")
    assert raised["exception"] is True and raised["success"] is False and raised["error"] == "RuntimeError: disk exploded"
    assert raised["visible"] is None and "does not see it" in raised["visible_unavailable"] and raised["result"] is None  # not an empty string: unknown, with the reason
    assert any(e["exception"] for e in log) and not sr.result_reached_model([e for e in log if e["tool"] == "file_operations"], ["anything"])


# ---- 4. several delegations keep their own identities, and nothing essential is cut ---------------------------------------------------------
def test_two_delegations_keep_distinct_identities_and_the_full_delegated_task(sink):
    long_task = "Create a.txt containing exactly one line. " + "Be precise. " * 60
    other_task = "Create b.txt containing exactly one line."
    turns = [delegate(long_task), write("a.txt", "A\n"), final("a done"), delegate(other_task), write("b.txt", "B\n"), final("b done"), final("both done")]
    result, _, log = run_harness(sink, turns)
    assert [r.result.delegation_id for r in result.workers] == ["d1", "d2"]
    delegates = [e for e in log if e["tool"] == "delegate"]
    assert [e["delegation_id"] for e in delegates] == [e["delegation_id_reported"] for e in delegates] == ["d1", "d2"]  # our numbering equals the tool's own
    assert delegates[0]["args"]["task"] == long_task and len(long_task) > 600 and delegates[1]["args"]["task"] == other_task
    generations = [(e["role"], e["delegation_id"], e["generation_in_agent"]) for e in events(sink) if e["event"] == "generation"]
    assert generations == [("orchestrator", None, 1), ("worker", "d1", 1), ("worker", "d1", 2), ("orchestrator", None, 2), ("worker", "d2", 1), ("worker", "d2", 2),
                           ("orchestrator", None, 3)]
    writes = [e for e in events(sink) if e["event"] == "tool_call" and e["tool"] == "file_operations"]
    assert [(e["delegation_id"], e["args"]["path"]) for e in writes] == [("d1", "a.txt"), ("d2", "b.txt")]
    assert all(e["generation_in_agent"] == 1 for e in writes)  # each call is tied to the worker generation that requested it


# ---- 5. an interrupted run keeps what it finished -------------------------------------------------------------------------------------------
class InterruptsAfterAnAttemptWasKept(ScriptedModel):
    def __init__(self, marker: Path):
        super().__init__([final("I do not know the code")], name="scripted")
        self.marker = marker

    def generate(self, prompt, config=None, **kw):
        if self.marker.exists():  # attempts.jsonl exists once the first attempt was kept: interrupt the second
            raise KeyboardInterrupt("operator pressed Ctrl-C")
        return super().generate(prompt, config, **kw)


def test_ctrl_c_in_the_second_attempt_keeps_the_first_and_marks_the_second_aborted(sink):
    model = InterruptsAfterAnAttemptWasKept(sink / "attempts.jsonl")
    spec = ModelSpec("ollama", "scripted", "server", base_url="http://127.0.0.1:11434/v1", context_length=4096)
    with pytest.raises(KeyboardInterrupt):
        sr.tool_use_attempts(model, spec, sink / "work", 3)
    kept = [json.loads(line) for line in (sink / "attempts.jsonl").read_text().splitlines()]
    assert [r["attempt_id"] for r in kept] == ["tooluse-a1"] and kept[0]["success"] is False
    trace = events(sink)
    by_attempt = {a: [e["event"] for e in trace if e["attempt"] == a] for a in ("tooluse-a1", "tooluse-a2")}
    assert by_attempt["tooluse-a1"][0] == "attempt_start" and by_attempt["tooluse-a1"][-1] == "attempt_end"
    assert by_attempt["tooluse-a2"] == ["attempt_start", "generation_error", "attempt_aborted"]  # started, interrupted, never ended
    assert next(e for e in trace if e["event"] == "attempt_start")["task"] == sr.READ_TASK  # the exact task, not a summary


# ---- 6. the runner's checks can fail a run that the agent called complete ----------------------------------------------------------------------
def delegation_run(tmp_path, monkeypatch, orchestrator_answer: str):
    """`delegation_attempts` end to end (rendered config, the CLI, the real harness and file tools) with one scripted model standing in for both Ollama specs."""
    model = ScriptedModel([delegate("Read secret.txt and report the access code.", worker="reader"), act("file_operations", {"operation": "read", "path": "secret.txt"}),
                           final("The file holds an access code."), final(orchestrator_answer)], name="scripted")
    model.unload = lambda: None
    monkeypatch.setattr(models_mod, "build_model", lambda spec: model)
    ctx = drv.Ctx("primary", tmp_path / "run")
    attempts = sr.delegation_attempts(ctx, tmp_path / "work", "ollama_only", {"__WORKER_NAME__": "reader", "__WORKER_DESCRIPTION__": "reads files", "__WORKER_TOOLS__": "[file_operations]"},
                                      sr.DELEGATION_TASKS[:1], 1, "B2")
    return attempts[0]


def test_a_completed_delegation_with_the_wrong_answer_is_a_task_failure(sink, monkeypatch):
    record = delegation_run(sink, monkeypatch, "The access code is 0000-ZZ.")
    assert record["exit_code"] == 0 and record["cli_success_flag"] is True and record["worker_records"][0]["status"] == "completed"
    verification = record["verification"]
    assert verification["agent_completed"] is True and verification["task_verified"] is False and record["success"] is False
    assert [c["name"] for c in verification["checks"] if not c["passed"]] == ["answer_has_code"]
    assert record["final_answer"] == "The access code is 0000-ZZ."  # in full
    assert [d["delegation_id"] for d in record["delegations"]] == ["d1"] and record["delegations"][0]["task"] == "Read secret.txt and report the access code."
    assert record["delegations"][0]["report_status"] == "completed" and record["workspace_changes"] == [] and record["role_split_agrees"] is True
    end = next(e for e in events(sink) if e["event"] == "attempt_end")
    assert (end["success"], end["agent_completed"], end["task_verified"]) == (False, True, False)
    assert {e["event"] for e in events(sink)} >= {"attempt_start", "model_loaded", "generation", "tool_call", "attempt_end"}


def test_the_same_run_with_the_right_answer_is_verified(sink, monkeypatch):
    record = delegation_run(sink, monkeypatch, f"The access code is {CODE}.")
    assert record["verification"]["task_verified"] is True and record["success"] is True


@pytest.mark.parametrize("worker_turns,fixed", [
    ([final("Fixed add(); all tests pass.")], False),  # claims a fix, changes nothing
    ([act("file_operations", {"operation": "write", "path": "calc.py", "content": "def add(a, b):\n    return a + b\n"}), final("Fixed add().")], True),
])
def test_the_coding_task_is_judged_by_the_repository_and_our_own_pytest_run(sink, monkeypatch, worker_turns, fixed):
    import asyncio
    import shutil

    from smol_harness.tools import ShellTool
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    probe = sink / "probe"
    probe.mkdir()
    if not asyncio.run(ShellTool(probe).execute(command="true")).success:
        pytest.skip("the shell sandbox is unavailable here")  # the coding config gives the worker the shell, which fails closed without it
    model = ScriptedModel([delegate("Fix add() in calc.py and run the tests.", worker="coder"), *worker_turns, final("The coder fixed add() and the tests pass.")], name="scripted")
    model.unload = lambda: None
    monkeypatch.setattr(models_mod, "build_model", lambda spec: model)
    ctx = drv.Ctx("primary", sink / "run")
    ctx.snapshot = "/unused/snapshot"
    record = sr.coding_attempts(ctx, sink / "work", 1)[0]
    verification = record["verification"]
    assert record["exit_code"] == 0 and record["cli_success_flag"] is True and verification["agent_completed"] is True  # the agent ALWAYS called itself done
    assert verification["task_verified"] is fixed and record["success"] is fixed and record["independent_pytest_returncode"] == (0 if fixed else 1)
    assert record["answer_claims_pass_but_repo_fails"] is (not fixed)
    assert [c["name"] for c in verification["checks"] if not c["passed"]] == ([] if fixed else ["source_changed", "independent_pytest_passes"])
    assert [(c["path"], c["change"]) for c in record["workspace_changes"]] == ([("calc.py", "modified")] if fixed else [])
    assert record["task"] == sr.CODING_TASK
    delegated = [e for e in record["tool_executions"] if e["tool"] == "delegate"]
    assert [e["args"]["task"] for e in delegated] == ["Fix add() in calc.py and run the tests."] and delegated[0]["delegation_id"] == "d1"


def test_no_checks_verify_nothing():
    assert sr.verdict(True, [])["task_verified"] is False
    assert sr.verdict(False, [("x", True, "")])["task_verified"] is True  # the agent's own flag is a separate field


def test_filesystem_changes_are_taken_from_disk(tmp_path):
    (tmp_path / "keep.txt").write_text("same")
    (tmp_path / "edit.txt").write_text("old")
    (tmp_path / "gone.txt").write_text("bye")
    before = sr.tree_manifest(tmp_path)
    (tmp_path / "edit.txt").write_text("new")
    (tmp_path / "gone.txt").unlink()
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "new.py").write_text("x = 1\n")
    changes = {c["path"]: c for c in sr.tree_changes(tmp_path, before, sr.tree_manifest(tmp_path))}
    assert {p: c["change"] for p, c in changes.items()} == {"edit.txt": "modified", "gone.txt": "removed", "pkg/new.py": "added"}
    assert set(changes["pkg/new.py"]) == {"path", "change", "sha256", "bytes"} and changes["pkg/new.py"]["bytes"] == 6  # no contents: path, size, hash
    assert changes["gone.txt"]["sha256"] is None and "bytes" not in changes["gone.txt"] and not any("text" in c for c in changes.values())


# ---- 7. missing provider metadata is not zero -------------------------------------------------------------------------------------------------
def generation(**fields):
    base = dict(text="Final Answer: x", finish_reason="stop", model_name="m", tokens_used=7, metadata=None)
    return SimpleNamespace(**{**base, **fields})


def test_missing_metadata_is_unavailable_and_not_zero():
    record = sr.generation_record("ollama", "p", None, generation(), 0.1)
    assert record["reasoning"] == {"metadata_available": False, "reported": [], "reasoning_tokens": None, "reasoning_chars": None, "reasoning_only": None, "empty_response_reason": None}
    assert record["provider_metadata"] is None and record["limits"] is None  # no config was passed: unknown, not an empty config
    empty = sr.generation_record("ollama", "p", None, generation(metadata={}), 0.1)["reasoning"]
    assert empty["metadata_available"] is True and empty["reported"] == [] and empty["reasoning_tokens"] is None  # absent: not reported (effGen also omits zero)
    assert sr.generation_record("ollama", "p", None, generation(tokens_used=0), 0.1)["tokens"] == 0  # a real zero stays zero
    assert sr.generation_record("ollama", "p", None, SimpleNamespace(text="t", finish_reason="stop"), 0.1)["tokens"] is None  # no such attribute: None


def test_reported_reasoning_metadata_is_kept_verbatim():
    meta = {"reasoning_tokens": 3598, "reasoning_chars": 15858, "reasoning_only": True, "empty_response_reason": "no visible text", "reasoning": "step by step..."}
    record = sr.generation_record("ollama", "p", None, generation(text="", finish_reason="stop", metadata=meta), 0.1)
    assert record["reasoning"]["reasoning_tokens"] == 3598 and record["reasoning"]["reasoning_only"] is True and set(record["reasoning"]["reported"]) == set(sr.REASONING_KEYS)
    assert record["provider_metadata"] == meta and record["raw_output"] == "" and record["finish"] == "stop"


def test_the_short_digest_says_when_it_cut_text():
    record = sr.generation_record("ollama", "p", None, generation(text="x" * 700), 0.1)
    short = sr.digest([record], limit=500)[0]
    assert len(short["raw_output"]) == 500 and short["raw_output_truncated"] is True and short["raw_output_chars"] == 700
    assert sr.digest([record], limit=1000)[0]["raw_output_truncated"] is False


# ---- P1-A: what effGen consumed when it cut the text itself (review 005) ----------------------------------------------------------------------------
class ReasoningOnlyFirst(ScriptedModel):
    """Its first generation is a reasoning-only turn (empty text, `reasoning_only` metadata, finish "stop"), as Ollama's Qwen3 produced in Run 21: effGen's answer is to
    retry once with its stop sequences applied locally instead of sent."""

    def generate(self, prompt, config=None, **kw):
        result = super().generate(prompt, config, **kw)
        if len(self.calls) == 1:
            return GenerationResult(text="", tokens_used=50, finish_reason="stop", model_name=self.model_name,
                                    metadata={"reasoning_only": True, "reasoning_tokens": 50, "empty_response_reason": "reasoning only"})
        return result


@pytest.fixture
def fresh_reasoning_registry(monkeypatch):
    """effGen remembers 'this model interleaves reasoning' for the whole process; a test must not leak that into the others."""
    monkeypatch.setattr(effgen_generation, "_reasoning_stream_models", set())


def test_the_reasoning_only_retry_is_recorded_with_the_text_effgen_really_consumed(sink, fresh_reasoning_registry):
    invented = delegate("write a.txt") + "\nObservation: INVENTED-RESULT\nFinal Answer: INVENTED"
    model = ReasoningOnlyFirst(["(replaced by the reasoning-only turn)", invented, write("a.txt", "A\n"), final("wrote it"), final("all done")], name="reasoner-retry")
    result, calls, log = run_harness(sink, [], model=model)
    trace = events(sink)
    generations = [e for e in trace if e["event"] == "generation"]
    consumed = {e["generation_seq"]: e for e in trace if e["event"] == "generation_consumed"}
    reasoning_only, retry = generations[0], generations[1]
    # the first turn: stops were SENT, the provider is in charge, nothing was cut by effGen; the provider's reasoning evidence is kept as reported
    assert reasoning_only["stop_handling"] == "provider" and reasoning_only["stop_sequences"] == list(DEFAULT_STOP_SEQUENCES) and reasoning_only["raw_output"] == ""
    assert reasoning_only["reasoning"]["reasoning_only"] is True and reasoning_only["reasoning"]["reasoning_tokens"] == 50 and reasoning_only["consumed_status"] == "established"
    # the retry: NO stops were sent (so `stop_sequences` is empty), the raw text is preserved, and the consumed text is not claimed until effGen's own cut was observed
    assert retry["stop_sequences"] == [] and retry["limits"]["stop_sequences"] is None and "INVENTED-RESULT" in retry["raw_output"]
    assert retry["stop_handling"] == "undetermined" and retry["consumed_output"] is None and retry["consumed_status"] == "pending"
    settled = consumed[retry["seq"]]
    assert settled["stop_handling"] == "local" and settled["stop_sequences_local"] == list(DEFAULT_STOP_SEQUENCES) and settled["consumed_status"] == "established"
    assert "INVENTED-RESULT" not in settled["consumed_output"] and settled["consumed_output"].endswith("}") and settled["discarded_chars"] == len(retry["raw_output"]) - len(settled["consumed_output"]) > 0
    assert retry["seq"] < settled["seq"] < generations[2]["seq"]  # settled before the next generation
    # the in-memory record that the ledgers use agrees, and the invented delegation is still not an executed call
    assert calls[1]["stop_handling"] == "local" and calls[1]["consumed_output"] == settled["consumed_output"] and "INVENTED-RESULT" in calls[1]["raw_output"]
    assert Counter(e["tool"] for e in log) == {"delegate": 1, "file_operations": 1} and result.output == "all done"
    assert sr.action_accounting(calls, log)["generations_with_consumed_text_unavailable"] == 0
    # effGen now cuts locally for this model for the rest of the process: every later generation is accounted for the same way
    assert [consumed[g["seq"]]["stop_handling"] for g in generations[1:]] == ["local"] * (len(generations) - 1)


def test_the_old_assumption_would_have_recorded_the_invented_text_as_consumed(sink, fresh_reasoning_registry):
    """The defect, stated as data: with no stops sent, `call_record` alone says consumed == raw. That is what the first version of the trace wrote."""
    invented = delegate("write a.txt") + "\nObservation: INVENTED-RESULT"
    _, calls, _ = run_harness(sink, [], model=ReasoningOnlyFirst(["x", invented, write("a.txt", "A\n"), final("ok"), final("done")], name="reasoner-old"))
    assert sr.call_record("scripted", "p", SimpleNamespace(stop_sequences=None), SimpleNamespace(text=calls[1]["raw_output"], finish_reason="stop", tokens_used=1), 0)["consumed_output"] == calls[1]["raw_output"]
    assert "INVENTED-RESULT" not in calls[1]["consumed_output"]  # the new record does not


def spy_with(sink_dir, raw, stop_sequences):
    model = SimpleNamespace(generate=lambda prompt, config=None, **kw: GenerationResult(text=raw, tokens_used=3, finish_reason="stop", model_name="m", metadata={}))
    calls: list[dict] = []
    return sr.traced_generate(model.generate, "scripted", calls), calls, SimpleNamespace(stop_sequences=stop_sequences)


def test_with_no_stops_sent_and_no_local_cut_the_text_was_consumed_as_returned(sink):
    spy, calls, config = spy_with(sink, "kept whole\nObservation: this stays because nothing cut it", None)
    with sr.attempt_scope("u-a1", "task", "single"):
        spy("p", config)
    settled = [e for e in events(sink) if e["event"] == "generation_consumed"][0]
    # not the default ReAct stops: they were not in force, so the text after "\nObservation:" was consumed
    assert settled["stop_handling"] == "none" and settled["stop_sequences_local"] == [] and settled["consumed_output"] == calls[0]["raw_output"] and settled["discarded_chars"] == 0


def test_consumed_text_stays_unavailable_when_effgens_handling_cannot_be_observed(sink):
    spy, calls, config = spy_with(sink, "x\nObservation: y", None)
    spy("p", config)  # outside an attempt scope nothing observes effGen's local cut
    assert calls[0]["consumed_output"] is None and calls[0]["consumed_status"].startswith("unavailable") and calls[0]["discarded_chars"] is None
    assert sr.action_accounting(calls, [])["generations_with_consumed_text_unavailable"] == 1
    assert sr.digest(calls)[0]["consumed_output"] is None and sr.digest(calls)[0]["consumed_status"].startswith("unavailable")


def test_an_attempt_interrupted_before_effgens_cut_was_seen_does_not_claim_the_text_was_kept(sink):
    spy, calls, config = spy_with(sink, "x\nObservation: y", None)
    with pytest.raises(KeyboardInterrupt):
        with sr.attempt_scope("u-a1", "task", "single"):
            spy("p", config)
            raise KeyboardInterrupt
    settled = next(e for e in events(sink) if e["event"] == "generation_consumed")
    assert settled["stop_handling"] == "undetermined" and settled["consumed_output"] is None and "could be observed" in settled["consumed_status"]
    assert [e["event"] for e in events(sink)][-2:] == ["generation_consumed", "attempt_aborted"]


def test_the_observer_is_restored_and_a_cut_it_cannot_attribute_is_reported(sink):
    real = effgen_generation.apply_stop_sequences
    with sr.attempt_scope("u-a1", "task", "single"):
        assert effgen_generation.apply_stop_sequences is not real
        assert effgen_generation.apply_stop_sequences("a\nObservation: b", ["\nObservation:"]) == "a"  # no generation of ours produced this text
    assert effgen_generation.apply_stop_sequences is real and sr.TRACE["observing"] is False
    unmatched = next(e for e in events(sink) if e["event"] == "local_stop_cut_unmatched")
    assert unmatched["stop_sequences_local"] == ["\nObservation:"] and (unmatched["chars_in"], unmatched["chars_out"]) == (16, 1)


# ---- P1-B: a refused delegation owns no id ----------------------------------------------------------------------------------------------------------
def test_a_delegation_refused_by_parameter_validation_does_not_share_the_id_of_the_next_one(sink):
    refused = act("delegate", {"worker": "coder"})  # the required `task` is missing: effGen refuses the call before DelegateTool runs
    turns = [refused, delegate("write a.txt"), write("a.txt", "A\n"), final("wrote it"), final("done")]
    result, calls, log = run_harness(sink, turns)
    assert [r.result.delegation_id for r in result.workers] == ["d1"]  # the harness ran one delegation
    delegates = [e for e in events(sink) if e["event"] == "tool_call" and e["tool"] == "delegate"]
    first, second = delegates
    assert first["success"] is False and first["dispatched"] is False and first["delegation_id"] is None and first["delegation_id_reported"] is None
    assert "Parameter validation failed" in first["error"] and first["role"] == "orchestrator"
    assert second["success"] is True and second["dispatched"] is True and second["delegation_id"] == second["delegation_id_reported"] == "d1"
    worker_events = [e for e in events(sink) if e.get("role") == "worker"]
    assert worker_events and {e["delegation_id"] for e in worker_events} == {"d1"}  # worker turns stay attributed to the delegation that ran
    orchestrator_generations = [e for e in events(sink) if e["event"] == "generation" and e["role"] == "orchestrator"]
    assert [e["delegation_id"] for e in orchestrator_generations] == [None] * len(orchestrator_generations)


def test_a_delegation_that_is_dispatched_but_fails_inside_still_owns_its_id(sink):
    turns = [delegate("write a.txt", worker="nobody"), final("could not delegate")]  # unknown worker: DelegateTool answers with a structured failure, after numbering the attempt
    result, _, _ = run_harness(sink, turns)
    only = next(e for e in events(sink) if e["event"] == "tool_call" and e["tool"] == "delegate")
    assert only["dispatched"] is True and only["delegation_id"] == "d1" == only["delegation_id_reported"]


# ---- P2-A: every run of a stage is its own invocation -------------------------------------------------------------------------------------------------
def test_running_a_stage_again_into_the_same_folder_cannot_be_confused_with_the_first_run(tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "hf_snapshot_path", lambda profile: ("/cache/snap", ""))
    monkeypatch.setattr(sr, "TRACE", sr.fresh_trace())
    state = {"runs": 0}

    def stage(ctx, args):
        state["runs"] += 1
        with sr.attempt_scope("tooluse-a1", "the task", "single"):
            if state["runs"] == 1:
                raise KeyboardInterrupt  # the first invocation is interrupted in attempt a1
            sr.trace("generation", raw_output="x")
        sr.keep({"attempt_id": "tooluse-a1", "kind": "tool_use", "success": False})
        return {"stage": "C", "status": drv.PASS}

    monkeypatch.setitem(sr.STAGE_FUNCS, "C", stage)
    with pytest.raises(KeyboardInterrupt):
        sr.main(["C", "--run-dir", str(tmp_path)])
    sr.main(["C", "--run-dir", str(tmp_path)])
    trace = events(tmp_path / "stage_C")
    assert all(e["run_id"] == tmp_path.name and e["stage"] == "C" and e["invocation"] for e in trace)  # existing identity kept, new one on EVERY line
    first, second = dict.fromkeys(e["invocation"] for e in trace)  # exactly two invocations, in the order they ran
    ended = lambda inv: [e["event"] for e in trace if e["invocation"] == inv and e["attempt"] == "tooluse-a1"]
    assert ended(first) == ["attempt_start", "attempt_aborted"] and ended(second) == ["attempt_start", "generation", "attempt_end"]  # same attempt id, told apart
    kept = [json.loads(line) for line in (tmp_path / "stage_C" / "attempts.jsonl").read_text().splitlines()]
    assert [(r["run_id"], r["stage"], r["invocation_id"], r["attempt_id"]) for r in kept] == [(tmp_path.name, "C", second, "tooluse-a1")]  # only the finished attempt is a record
    assert len({(e["invocation"], e["seq"]) for e in trace}) == len(trace)  # (invocation, seq) is unique, so a loader can key on it


# ---- P2-B: what the results say about workspace contents --------------------------------------------------------------------------------------------------
def test_file_contents_reach_the_results_only_through_tool_calls_and_the_docs_say_so(sink):
    run_harness(sink, [delegate("write notes.txt"), write("notes.txt", "customer: alice@example.com\n"), final("wrote it"), final("done")])
    trace = events(sink)
    written = next(e for e in trace if e["event"] == "tool_call" and e["tool"] == "file_operations")
    assert written["args"]["content"] == "customer: alice@example.com\n"  # the tool call is where contents are recorded (redaction applies there)
    manifest = drv.write_manifest(sink)
    claim = " ".join(manifest["never_collected"])
    assert "work/" in claim and "ARE in trace.jsonl" in claim  # no longer 'workspaces are never collected'
    text = (PKG / "RESULTS.md").read_text()
    assert "What does reach the results from a workspace" in text and "never its contents" in text
    assert "the disposable workspaces (`work/`)" not in text


# ---- shared artifacts ---------------------------------------------------------------------------------------------------------------------------------
def test_trace_lines_are_redacted_when_written(sink):
    sr.trace("note", text="token hf_abcdefghijklmnopqrstuvwxyz0123 leaked")
    assert "hf_abcdefghijklmnop" not in (sink / "trace.jsonl").read_text()


def test_packaging_scans_jsonl_files_a_second_time(tmp_path):
    (tmp_path / "stage_B").mkdir()
    leaky = tmp_path / "stage_B" / "trace.jsonl"
    leaky.write_text(json.dumps({"event": "x", "text": "key hf_abcdefghijklmnopqrstuvwxyz0123"}) + "\n")
    assert drv.re_redact_text_files(tmp_path) == ["stage_B/trace.jsonl"] and "hf_abcdefghijklmnop" not in leaky.read_text()


def test_a_stage_writes_a_header_first_and_clears_its_sink(tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "hf_snapshot_path", lambda profile: ("/cache/snap", ""))
    monkeypatch.setitem(sr.STAGE_FUNCS, "C", lambda ctx, args: {"stage": "C", "status": drv.PASS})
    monkeypatch.setattr(sr, "TRACE", dict(sr.TRACE))
    sr.main(["C", "--run-dir", str(tmp_path)])
    first = events(tmp_path / "stage_C")[0]
    assert first["event"] == "stage_start" and first["run_id"] == tmp_path.name and first["stage"] == "C" and first["versions"]["effgen"] == "1.3.0"
    assert "commit" in first["git"] and first["runner_overrides"] == sr.SAMPLING and "gpus" in first and "ollama_version" in first
    assert sr.TRACE["path"] is None
    assert dataclasses.is_dataclass(ModelSpec)  # the model_loaded event relies on it
