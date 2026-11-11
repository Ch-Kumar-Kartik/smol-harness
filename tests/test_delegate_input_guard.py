"""effGen cuts every tool input to 10,000 characters before it parses it, which breaks a long `delegate` call's JSON ("Parameter 'task' is required", no worker runs).
`DelegateInputGuard` refuses such a call first, with an accurate message. Pinned here: effGen's limit and how it counts, the guard's exact boundary (against the REAL
effGen, end to end), escaping overhead, that a refusal charges nothing, and that a refused call is still recorded. Scripted models only."""

from __future__ import annotations

import inspect
import json

import pytest
from conftest import ScriptedModel
from effgen import Agent, AgentConfig
from effgen.core.middleware import ToolCallContext

from smol_harness.delegation import MAX_TOOL_INPUT_CHARS, DelegateInputGuard, StructuredToolInputGuard, tool_input_size
from smol_harness.harness import build_harness
from smol_harness.recorder import ExecutionRecorder

FINAL = "Thought: done.\nFinal Answer: ok"
TAIL = "TAIL-MARKER-9"


def cfg(model, name):
    return AgentConfig(name=name, model=model, tools=[], tool_calling_mode="react", capability_probe=False, enable_memory=False, max_iterations=6, raise_on_error=False)


def call(**args) -> str:
    return f"Thought: n.\nAction: delegate\nAction Input: {json.dumps(args)}"


def make(orch_turns, recorder=None, **limits):
    orchestrator = ScriptedModel(orch_turns, name="orch", on_exhausted="raise")
    worker = ScriptedModel(["Thought: d.\nFinal Answer: worker done"], name="worker")
    return build_harness(cfg(orchestrator, "orchestrator"), {"A": cfg(worker, "A")}, {"A": "does things"}, recorder=recorder, **limits), orchestrator, worker


def guard(tool="delegate", text="") -> str | None:
    return DelegateInputGuard().before_tool_call(ToolCallContext(tool_name=tool, tool_input=text))


# ---- the effGen facts the guard relies on (a different effGen must fail here, loudly) ---------------------------------------------------------------
def test_effgen_truncates_tool_input_at_the_limit_the_guard_uses():
    assert inspect.signature(Agent._sanitize_tool_input).parameters["max_length"].default == MAX_TOOL_INPUT_CHARS == 10_000
    assert len(Agent._sanitize_tool_input("x" * 10_000)) == 10_000 and len(Agent._sanitize_tool_input("x" * 10_001)) == 10_000
    assert Agent._sanitize_tool_input("a\x01b\x7fc\nd\te") == "abc\nd\te"  # control characters other than newline and tab are removed first
    assert tool_input_size("a\x01b\x7fc\nd\te") == len(Agent._sanitize_tool_input("a\x01b\x7fc\nd\te"))


def test_the_sanitizer_runs_after_the_middleware_hooks_so_the_guard_sees_the_raw_input():
    source = inspect.getsource(Agent._execute_tool)
    assert source.index("before_tool_call") < source.index("_execute_tool_guarded(ctx.tool_name, ctx.tool_input)")


# ---- the guard itself ----------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("size, refused", [(10_000, False), (10_001, True), (1, False)])
def test_the_boundary_is_exact(size, refused):
    assert (guard(text="x" * size) is not None) is refused


def test_escaping_overhead_counts_because_the_input_is_the_serialised_json():
    context = '"' * 6_000  # 6,000 characters of context, 12,000 once the model writes them as JSON
    payload = json.dumps({"worker": "A", "task": "t", "context": context})
    assert len(context) < MAX_TOOL_INPUT_CHARS < len(payload)
    message = guard(text=payload)
    assert message is not None and f"is {len(payload)} characters" in message and message.startswith("Error: the delegate call was NOT run")


def test_control_characters_are_not_counted_because_effgen_removes_them_first():
    assert guard(text="x" * 9_999 + "\x01\x02") is None  # 10,001 raw, 9,999 as effGen counts
    assert guard(text="x" * 9_999 + "\n\t") is not None  # newline and tab stay


def test_malformed_delegate_json_is_refused_before_parameter_validation():
    message = guard(text='{"worker":"A","task":"t"} <channel|>Observation: {"status":"success"}')
    assert message is not None
    assert message.startswith("Error: the delegate call was NOT run")
    assert "malformed JSON" in message


def test_malformed_structured_worker_tool_json_is_refused_before_parameter_validation():
    message = StructuredToolInputGuard().before_tool_call(
        ToolCallContext(tool_name="edit_file", tool_input='{"path":"rich/ansi.py"} <channel|>Action: edit_file')
    )
    assert message is not None
    assert "edit_file call was NOT run" in message
    assert "exactly one JSON object" in message


def test_other_tools_and_non_text_input_are_left_alone():
    assert guard(tool="file_operations", text="x" * 50_000) is None
    assert DelegateInputGuard().before_tool_call(ToolCallContext(tool_name="delegate", tool_input={"a": 1})) is None  # type: ignore[arg-type]


# ---- end to end against the real effGen ---------------------------------------------------------------------------------------------------------------
def test_an_oversized_delegate_is_refused_before_anything_runs_and_charges_nothing():
    harness, orchestrator, worker = make([call(worker="A", task="t", context="x" * 11_000), call(worker="A", task="small task", context="short"), FINAL], max_delegations=1)
    result = harness.run("go")
    assert result.success and len(worker.calls) == 1  # only the second, valid call ran a worker
    assert [(w.worker, w.status, w.result.delegation_id) for w in result.workers] == [("A", "completed", "d1")]  # the refusal took no id and no max_delegations slot
    observation = str(orchestrator.calls[1].prompt)
    assert "Error: the delegate call was NOT run" in observation and "limit is 10000" in observation
    assert "Parameter 'task' is required" not in observation  # effGen's misleading message for the truncated JSON


def test_a_call_of_exactly_the_limit_reaches_the_worker_intact_and_one_more_character_does_not():
    base = len(json.dumps({"worker": "A", "task": "t", "context": ""}))
    for extra, runs in ((0, True), (1, False)):
        filler = "y" * (MAX_TOOL_INPUT_CHARS - base - len(TAIL) + extra) + TAIL
        text = f"Thought: n.\nAction: delegate\nAction Input: {json.dumps({'worker': 'A', 'task': 't', 'context': filler})}"
        assert len(json.dumps({"worker": "A", "task": "t", "context": filler})) == MAX_TOOL_INPUT_CHARS + extra
        harness, orchestrator, worker = make([text, FINAL])
        result = harness.run("go")
        assert bool(worker.calls) is runs
        if runs:
            assert TAIL in str(worker.calls[0].prompt)  # the end of the context arrived: nothing was cut at the real boundary
        else:
            assert not result.workers and "limit is 10000" in str(orchestrator.calls[1].prompt)


def test_a_refused_call_is_still_recorded_with_its_full_arguments_and_the_refusal(tmp_path):
    recorder = ExecutionRecorder(tmp_path / "h")
    harness, orchestrator, worker = make([call(worker="A", task="t", context="x" * 11_000), FINAL], recorder=recorder)
    harness.run("go")
    lines = [json.loads(line) for line in (recorder.dir / "trace.jsonl").read_text().splitlines()]
    start = next(e for e in lines if e["event"] == "tool_call_start" and e["tool"] == "delegate")
    end = next(e for e in lines if e["event"] == "tool_call" and e["tool"] == "delegate")
    assert len(start["args"]["context"]) == 11_000  # the recorder runs first and sees the whole call
    assert end["result"].startswith("Error: the delegate call was NOT run") and end.get("delegation_id") is None and end["tool_call_id"].endswith(f":{start['seq']}")
    assert not [e for e in lines if e.get("role") == "worker"] and recorder.error is None
