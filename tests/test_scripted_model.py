"""Tests for the ScriptedModel fixture, driven through effGen's real Agent loop."""

from __future__ import annotations

import pytest
from conftest import ModelCall, ScriptedModel, ScriptExhaustedError
from effgen import Agent, AgentConfig, tool

CALL_ADD = 'Thought: use the tool.\nAction: add\nAction Input: {"a": 2, "b": 3}'
ANSWER = "Thought: done.\nFinal Answer: 5"


def make_add():
    calls: list[tuple[int, int]] = []

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        calls.append((a, b))
        return a + b

    return add, calls


def make_agent(model, tools=(), **cfg) -> Agent:
    # Settings from plan T1.1 acceptance; enable_memory=False keeps runs independent.
    return Agent(AgentConfig(
        name="t", model=model, tools=list(tools), tool_calling_mode="react",
        capability_probe=False, enable_sub_agents=False, enable_memory=False,
        max_iterations=4, raise_on_error=False, **cfg,
    ))


# -- acceptance: tool called once, scripted final answer returned -------------
def test_agent_calls_tool_once_and_returns_scripted_answer():
    add, calls = make_add()
    model = ScriptedModel([CALL_ADD, ANSWER], on_exhausted="raise")
    with make_agent(model, [add]) as agent:
        resp = agent.run("What is 2+3?")
    assert resp.success is True
    assert resp.output == "5"
    assert calls == [(2, 3)]
    assert len(model.calls) == 2  # exact: no nudges consumed extra turns
    assert model.exhausted


def test_observation_reaches_the_next_prompt():
    add, _ = make_add()
    model = ScriptedModel([CALL_ADD, ANSWER], on_exhausted="raise")
    with make_agent(model, [add]) as agent:
        agent.run("What is 2+3?")
    first, second = model.calls
    assert "Question: What is 2+3?" in first.prompt
    assert "Observation: 5" not in first.prompt
    assert "Observation: 5" in second.prompt
    assert isinstance(first, ModelCall) and first.response == CALL_ADD


def test_no_tools_agent_answers_directly():
    model = ScriptedModel(["Final Answer: hello"], on_exhausted="raise")
    with make_agent(model) as agent:
        resp = agent.run("Say hello")
    assert resp.success and "hello" in resp.output
    assert len(model.calls) == 1


# -- recording ------------------------------------------------------------------
def test_records_each_call_with_prompt_config_and_response():
    model = ScriptedModel(["a", "b"])
    r1 = model.generate("p1")
    r2 = model.generate("p2", None, tools=["x"])
    assert (r1.text, r2.text) == ("a", "b")
    assert [c.prompt for c in model.calls] == ["p1", "p2"]
    assert [c.response for c in model.calls] == ["a", "b"]
    assert model.calls[1].tools == ["x"]
    assert model.calls[0].tools is None


def test_stream_yields_full_text_and_is_recorded():
    model = ScriptedModel(["hello streaming world"])
    assert "".join(model.generate_stream("p")) == "hello streaming world"
    assert len(model.calls) == 1


# -- exhaustion --------------------------------------------------------------------
def test_repeat_last_keeps_returning_final_turn():
    model = ScriptedModel(["one", "two"])
    assert [model.generate("p").text for _ in range(4)] == ["one", "two", "two", "two"]
    assert model.exhausted


def test_raise_mode_raises_when_script_runs_out():
    model = ScriptedModel(["only"], on_exhausted="raise")
    model.generate("p")
    with pytest.raises(ScriptExhaustedError, match="only 1 turn"):
        model.generate("p")
    assert len(model.calls) == 1  # the failed request is not recorded


def test_agent_surfaces_exhaustion_as_failed_run():
    # effGen 1.3.0 (raise_on_error=False) turns a model exception into a failed
    # response rather than propagating it, so a too-short script is visible
    # as success=False / termination="error", never a silent pass.
    add, _ = make_add()
    model = ScriptedModel([CALL_ADD], on_exhausted="raise")
    with make_agent(model, [add]) as agent:
        resp = agent.run("What is 2+3?")
    assert resp.success is False
    assert resp.termination == "error"
    assert "only 1 turn(s) scripted" in resp.output
    assert len(model.calls) == 1


def test_repeat_last_with_looping_script_is_stopped_by_effgen():
    add, calls = make_add()
    model = ScriptedModel([CALL_ADD])  # never answers
    with make_agent(model, [add]) as agent:
        resp = agent.run("What is 2+3?")
    assert resp.success is False
    assert resp.termination == "stuck"  # effGen's repeat guard (loop_detected)
    assert len(model.calls) <= 4  # max_iterations
    assert calls == [(2, 3)]  # the repeated call was not re-executed


# -- agent-level failure cases ---------------------------------------------------------
def test_unknown_tool_is_reported_to_the_model_as_an_observation():
    add, calls = make_add()
    bad = 'Thought: x.\nAction: nope\nAction Input: {"a": 1}'
    model = ScriptedModel([bad, "Final Answer: ok"])
    with make_agent(model, [add]) as agent:
        resp = agent.run("What is 2+3?")
    assert calls == []
    assert "No tool named 'nope' is available. The tools you can use are: add." in model.calls[1].prompt
    assert resp.success and resp.output == "ok"
    # effGen's nudge consumed an extra turn: this is why repeat_last is the default.
    assert len(model.calls) > 2


def test_tool_exception_becomes_observation_not_crash():
    @tool
    def boom(x: int) -> int:
        """Always fails."""
        raise ValueError("kaboom")

    model = ScriptedModel(['Thought: t.\nAction: boom\nAction Input: {"x": 1}', "Final Answer: gave up"])
    with make_agent(model, [boom]) as agent:
        resp = agent.run("go")
    assert "Observation: Error executing tool 'boom': Tool execution failed: kaboom" in model.calls[1].prompt
    assert resp.success and resp.output == "gave up"


# -- constructor validation -------------------------------------------------------------
def test_invalid_on_exhausted_rejected():
    with pytest.raises(ValueError, match="on_exhausted"):
        ScriptedModel(["x"], on_exhausted="sometimes")


def test_empty_script_with_repeat_last_rejected():
    with pytest.raises(ValueError, match="at least one"):
        ScriptedModel([])


def test_empty_script_with_raise_mode_raises_on_first_call():
    model = ScriptedModel([], on_exhausted="raise")
    with pytest.raises(ScriptExhaustedError):
        model.generate("p")


def test_scripted_model_factory_fixture(scripted_model):
    assert scripted_model is ScriptedModel
    assert scripted_model(["x"], name="orch").model_name == "orch"
