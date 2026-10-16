"""T1.2: WorkerResult / WorkerRegistry / DelegateTool, via effGen's real tool interface."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, fields

import pytest
from conftest import ScriptedModel
from effgen import Agent, AgentConfig, tool

from smol_harness.delegation import (
    DelegateTool,
    UnknownWorkerError,
    WorkerRegistry,
    WorkerResult,
)

ANSWER = "Thought: done.\nFinal Answer: {text}"
RESULT_KEYS = {f.name for f in fields(WorkerResult)}


def cfg(model, tools=(), name="w", **over) -> AgentConfig:
    base = dict(
        name=name, model=model, tools=list(tools), tool_calling_mode="react",
        capability_probe=False, enable_memory=False, max_iterations=4, raise_on_error=False,
    )
    base.update(over)
    return AgentConfig(**base)


def delegate(tool_: DelegateTool, **args):
    """Call the tool the way effGen does: BaseTool.execute() -> ToolResult."""
    return asyncio.run(tool_.execute(**args))


def payload(result) -> dict:
    assert result.success is True, result.error
    assert isinstance(result.output, str)  # JSON string, not a dict (audit §1.2)
    return json.loads(result.output)


@pytest.fixture
def setup():
    """registry with one worker 'coder' on a scripted model, plus its DelegateTool."""
    model = ScriptedModel([ANSWER.format(text="patched foo.py")], name="coder-model")
    registry = WorkerRegistry()
    registry.register("coder", cfg(model, name="coder"), description="edits code")
    return model, registry, DelegateTool(registry)


# 1. successful registration and delegation ------------------------------------
def test_successful_delegation(setup):
    model, registry, dt = setup
    assert registry.names() == ["coder"]
    data = payload(delegate(dt, worker="coder", task="fix the bug", context="see foo.py"))
    assert data["worker"] == "coder"
    assert data["status"] == "completed"
    assert data["output"] == "patched foo.py"
    assert data["error"] is None and data["error_type"] is None
    assert data["termination"] == "done"
    assert data["iterations"] >= 1 and data["tokens_used"] > 0
    assert len(model.calls) == 1
    prompt = model.calls[0].prompt
    assert "worker agent 'coder'" in prompt and "fix the bug" in prompt and "see foo.py" in prompt


def test_context_is_optional(setup):
    model, _, dt = setup
    assert payload(delegate(dt, worker="coder", task="t"))["status"] == "completed"
    assert "Context:" not in model.calls[0].prompt


def test_worker_tool_use_is_counted_but_not_exposed():
    @tool
    def lookup(q: str) -> str:
        """Look something up."""
        return "SECRET-OBSERVATION-TEXT"

    model = ScriptedModel([
        'Thought: private reasoning.\nAction: lookup\nAction Input: {"q": "x"}',
        ANSWER.format(text="all good"),
    ], on_exhausted="raise")
    registry = WorkerRegistry()
    registry.register("coder", cfg(model, [lookup]))
    result = delegate(DelegateTool(registry), worker="coder", task="t")
    data = payload(result)
    assert data["tool_calls"] == 1 and data["output"] == "all good"
    # The parent sees only the report: no worker thoughts, observations, or history field.
    assert set(data) == RESULT_KEYS
    assert "SECRET-OBSERVATION-TEXT" not in result.output
    assert "private reasoning" not in result.output


def test_tool_is_usable_by_an_effgen_agent_and_has_expected_schema(setup):
    _, _, dt = setup
    agent = Agent(cfg(ScriptedModel(["Final Answer: x"]), [dt], name="orch"))
    assert agent.tools["delegate"] is dt
    params = {p.name: p.required for p in dt.metadata.parameters}
    assert params == {"worker": True, "task": True, "context": False}
    assert "coder: edits code" in dt.metadata.description


# 2. unknown worker -------------------------------------------------------------
def test_unknown_worker_is_a_structured_failure(setup):
    model, registry, dt = setup
    data = payload(delegate(dt, worker="nobody", task="t"))
    assert data["status"] == "failed" and data["error_type"] == "unknown_worker"
    assert "'nobody'" in data["error"] and "coder" in data["error"]
    assert data["output"] == ""
    assert model.calls == []  # no worker was run
    with pytest.raises(UnknownWorkerError, match="Available workers: coder"):
        registry.create("nobody")


# 3. worker execution failure ----------------------------------------------------
@pytest.mark.parametrize("raise_on_error", [False, True])
def test_worker_model_failure_is_a_structured_failure(raise_on_error):
    model = ScriptedModel([], on_exhausted="raise")
    registry = WorkerRegistry()
    registry.register("w", cfg(model, raise_on_error=raise_on_error))
    data = payload(delegate(DelegateTool(registry), worker="w", task="t"))
    assert data["status"] == "failed"
    assert data["output"] == ""
    assert "scripted" in data["error"]  # effGen's wording differs per raise_on_error
    assert data["error_type"] in ("worker_run_failed", "RuntimeError")


def test_worker_that_never_finishes_is_a_structured_failure():
    @tool
    def noop(x: int) -> int:
        """No-op."""
        return x

    model = ScriptedModel(['Thought: t.\nAction: noop\nAction Input: {"x": 1}'])  # loops forever
    registry = WorkerRegistry()
    registry.register("w", cfg(model, [noop]))
    data = payload(delegate(DelegateTool(registry), worker="w", task="t"))
    assert data["status"] == "failed" and data["error_type"] == "worker_run_failed"
    assert data["termination"] == "stuck"
    assert data["error"]


def test_worker_construction_failure_is_a_structured_failure(monkeypatch, setup):
    _, registry, dt = setup

    def boom(name):
        raise RuntimeError("cannot build")

    monkeypatch.setattr(registry, "create", boom)
    data = payload(delegate(dt, worker="coder", task="t"))
    assert data["status"] == "failed" and data["error"] == "cannot build"


# 4. JSON serialization -------------------------------------------------------------
def test_success_and_failure_share_one_json_schema(setup):
    _, _, dt = setup
    ok = json.loads(delegate(dt, worker="coder", task="t").output)
    bad = json.loads(delegate(dt, worker="nobody", task="t").output)
    assert set(ok) == set(bad) == RESULT_KEYS
    for d in (ok, bad):
        assert json.loads(json.dumps(d)) == d


def test_worker_result_roundtrip():
    r = WorkerResult(worker="w", status="failed", error="x", error_type="E")
    assert json.loads(r.to_json()) == asdict(r)


# 5. fresh worker state ---------------------------------------------------------------
def test_each_delegation_gets_a_fresh_agent_with_no_history():
    model = ScriptedModel([ANSWER.format(text="one"), ANSWER.format(text="two")])
    registry = WorkerRegistry()
    registry.register("w", cfg(model))
    dt = DelegateTool(registry)
    payload(delegate(dt, worker="w", task="alpha-secret"))
    payload(delegate(dt, worker="w", task="beta"))
    assert "alpha-secret" in model.calls[0].prompt
    assert "beta" in model.calls[1].prompt
    assert "alpha-secret" not in model.calls[1].prompt


def test_control_reusing_one_agent_does_leak_history():
    # Why the registry builds a fresh Agent per delegation (audit §1.4): effGen
    # agents keep short-term memory across run() calls.
    model = ScriptedModel([ANSWER.format(text="one"), ANSWER.format(text="two")])
    with Agent(cfg(model, enable_sub_agents=False)) as agent:
        agent.run("alpha-secret")
        agent.run("beta")
    assert "alpha-secret" in model.calls[1].prompt


def test_registry_creates_distinct_agents_each_time(setup):
    _, registry, _ = setup
    a1, a2 = registry.create("coder"), registry.create("coder")
    assert a1 is not a2
    assert a1.short_term_memory is not a2.short_term_memory
    assert a1.state is not a2.state


# 6. shared model instance, separate conversations --------------------------------------
def test_model_instance_is_shared_but_histories_are_not():
    model = ScriptedModel([ANSWER.format(text="a")], name="shared")
    registry = WorkerRegistry()
    registry.register("w1", cfg(model, name="w1"))
    registry.register("w2", cfg(model, name="w2"))
    a1, a2, a3 = registry.create("w1"), registry.create("w2"), registry.create("w1")
    assert a1.model is a2.model is a3.model is model  # no reload, same weights object
    a1.run("only-for-w1")
    a2.run("only-for-w2")
    assert [("only-for-w1" in c.prompt, "only-for-w2" in c.prompt) for c in model.calls] == [
        (True, False), (False, True)
    ]
    assert a1.config is not a3.config  # per-agent config copy
    assert registry.get_config("w1").system_prompt == "You are a helpful AI assistant."  # registry copy untouched


def test_workers_cannot_spawn_and_registry_rejects_bad_definitions(setup):
    _, registry, dt = setup
    assert registry.create("coder").config.enable_sub_agents is False
    m = ScriptedModel(["x"])
    with pytest.raises(ValueError, match="already registered"):
        registry.register("coder", cfg(m))
    with pytest.raises(ValueError, match="non-empty"):
        registry.register("  ", cfg(m))
    with pytest.raises(TypeError, match="BaseModel instance"):
        registry.register("s", AgentConfig(name="s", model="some-model-id", require_model=False))
    with pytest.raises(ValueError, match="depth 1"):
        registry.register("nest", cfg(m, [dt]))


# 7. invalid delegation arguments ------------------------------------------------------------
@pytest.mark.parametrize("args, fragment", [
    ({"task": "t"}, "'worker' is required"),
    ({"worker": "coder"}, "'task' is required"),
    ({"worker": "coder", "task": ""}, "'task' must have length >= 1"),
    ({"worker": "coder", "task": 5}, "'task' must be of type string"),
    ({"worker": 3, "task": "t"}, "'worker' must be of type string"),
])
def test_invalid_arguments_rejected_by_effgen_validation(setup, args, fragment):
    model, _, dt = setup
    result = delegate(dt, **args)
    assert result.success is False
    assert "Parameter validation failed" in result.error and fragment in result.error
    assert model.calls == []


def test_blank_task_is_a_structured_failure(setup):
    model, _, dt = setup
    data = payload(delegate(dt, worker="coder", task="   "))
    assert data["status"] == "failed" and data["error_type"] == "invalid_arguments"
    assert model.calls == []
