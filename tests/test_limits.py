"""T2.1: cooperative per-worker deadline (DeadlineMiddleware) and the `timeout` status.

Most tests use a fake monotonic clock that tools/models advance, so they are
deterministic and fast; the real-time tests sleep ~0.3 s.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import fields

import pytest
from conftest import ScriptedModel
from effgen import AgentConfig, tool

from smol_harness.delegation import (
    DeadlineMiddleware,
    DelegateTool,
    WorkerRegistry,
    WorkerResult,
    WorkerTimeout,
)
from smol_harness.harness import build_harness

RESULT_KEYS = {f.name for f in fields(WorkerResult)}
CALL = 'Thought: t.\nAction: slow_tool\nAction Input: {"x": 1}'
ANSWER = "Thought: done.\nFinal Answer: all done"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, secs: float) -> None:
        self.now += secs


def cfg(model, name="w", tools=(), **over) -> AgentConfig:
    base = dict(
        name=name, model=model, tools=list(tools), tool_calling_mode="react",
        capability_probe=False, enable_memory=False, max_iterations=6, raise_on_error=False,
    )
    base.update(over)
    return AgentConfig(**base)


def make_slow_tool(clock: FakeClock, advance: float):
    ran = []

    @tool
    def slow_tool(x: int) -> int:
        """Pretend to take a long time (advances the fake clock)."""
        ran.append(x)
        clock.advance(advance)
        return x

    return slow_tool, ran


def delegate(dt: DelegateTool, **args):
    result = asyncio.run(dt.execute(**args))
    assert result.success is True, result.error  # a worker timeout is a successful tool call
    return json.loads(result.output)


def make_delegate(model, tools=(), timeout=5.0, clock=None, **over):
    reg = WorkerRegistry()
    reg.register("w", cfg(model, tools=tools, **over))
    records: list = []
    return DelegateTool(reg, records, worker_timeout_s=timeout, clock=clock or FakeClock()), records


# ---- DeadlineMiddleware unit behaviour ------------------------------------------
def test_deadline_passes_before_and_raises_at_deadline_in_both_hooks():
    clock = FakeClock()
    dl = DeadlineMiddleware(10, clock)
    dl.before_model_call(object())
    clock.advance(9.9)
    dl.before_model_call(object())
    assert dl.expired is False

    class Ctx:
        tool_name = "t"

    clock.advance(0.1)
    with pytest.raises(WorkerTimeout, match="10s deadline.*tool 't'"):
        dl.before_tool_call(Ctx())
    assert dl.expired is True
    with pytest.raises(WorkerTimeout, match="model call"):
        dl.before_model_call(object())


def test_deadline_state_is_per_instance():
    clock = FakeClock()
    a, b = DeadlineMiddleware(1, clock), DeadlineMiddleware(1, clock)
    clock.advance(2)
    with pytest.raises(WorkerTimeout):
        a.before_model_call(None)
    assert a.expired and not b.expired
    c = DeadlineMiddleware(1, clock)  # a new instance starts a new window
    c.before_model_call(None)
    assert not c.expired


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_non_positive_timeouts_rejected(bad):
    with pytest.raises(ValueError):
        DeadlineMiddleware(bad)
    with pytest.raises(ValueError):
        DelegateTool(WorkerRegistry(), worker_timeout_s=bad)


# ---- through effGen: DelegateTool + real Agent loop --------------------------------
def test_worker_completes_before_deadline():
    clock = FakeClock()
    slow_tool, ran = make_slow_tool(clock, advance=1.0)  # 1 s of a 5 s budget
    model = ScriptedModel([CALL, ANSWER])
    dt, records = make_delegate(model, [slow_tool], timeout=5.0, clock=clock)
    data = delegate(dt, worker="w", task="t")
    assert data["status"] == "completed" and data["output"] == "all done"
    assert ran == [1] and len(model.calls) == 2
    assert [r.status for r in records] == ["completed"]


@pytest.mark.parametrize("raise_on_error", [False, True])
def test_deadline_exceeded_during_a_tool_stops_the_worker(raise_on_error):
    clock = FakeClock()
    slow_tool, ran = make_slow_tool(clock, advance=10.0)  # tool "takes" 10 s of a 5 s budget
    model = ScriptedModel([CALL, ANSWER])
    dt, records = make_delegate(model, [slow_tool], timeout=5.0, clock=clock, raise_on_error=raise_on_error)
    data = delegate(dt, worker="w", task="t")
    assert data["status"] == "timeout" and data["error_type"] == "WorkerTimeout"
    assert "5s deadline" in data["error"] and data["output"] == ""
    assert ran == [1]  # cooperative: the tool in flight was NOT interrupted ...
    assert len(model.calls) == 1  # ... but the next model call never happened
    assert [r.status for r in records] == ["timeout"]


@pytest.mark.parametrize("raise_on_error", [False, True])
def test_deadline_exceeded_during_model_call_prevents_the_next_tool(raise_on_error):
    clock = FakeClock()
    slow_tool, ran = make_slow_tool(clock, advance=0)

    class SlowModel(ScriptedModel):  # first generation "takes" 10 s
        def generate(self, prompt, config=None, **kw):
            out = super().generate(prompt, config, **kw)
            if len(self.calls) == 1:
                clock.advance(10)
            return out

    model = SlowModel([CALL, ANSWER])
    dt, _ = make_delegate(model, [slow_tool], timeout=5.0, clock=clock, raise_on_error=raise_on_error)
    data = delegate(dt, worker="w", task="t")
    assert data["status"] == "timeout"
    assert len(model.calls) == 1  # the generation in flight finished; nothing after it
    assert ran == []  # before_tool_call stopped the tool from running


def test_timeout_result_is_valid_json_with_the_common_schema():
    clock = FakeClock()
    slow_tool, _ = make_slow_tool(clock, advance=10.0)
    dt, _ = make_delegate(ScriptedModel([CALL, ANSWER]), [slow_tool], timeout=5.0, clock=clock)
    raw = asyncio.run(dt.execute(worker="w", task="t")).output
    assert isinstance(raw, str)
    data = json.loads(raw)
    assert set(data) == RESULT_KEYS
    assert json.loads(json.dumps(data)) == data
    assert data["worker"] == "w" and data["termination"] is None and data["tokens_used"] == 0


def test_no_timeout_configured_means_no_deadline():
    clock = FakeClock()
    slow_tool, _ = make_slow_tool(clock, advance=10_000.0)
    reg = WorkerRegistry()
    reg.register("w", cfg(ScriptedModel([CALL, ANSWER]), tools=[slow_tool]))
    data = delegate(DelegateTool(reg), worker="w", task="t")
    assert data["status"] == "completed"


# ---- state isolation -------------------------------------------------------------------
def test_timeout_does_not_leak_into_the_next_delegation_or_agent():
    clock = FakeClock()
    ran = []

    @tool
    def slow_tool(x: int) -> int:
        """Slow only the first time."""
        ran.append(x)
        if len(ran) == 1:
            clock.advance(10)
        return x

    model = ScriptedModel([CALL, ANSWER, CALL, ANSWER])
    dt, records = make_delegate(model, [slow_tool], timeout=5.0, clock=clock)
    first = delegate(dt, worker="w", task="first")
    second = delegate(dt, worker="w", task="second")  # new deadline window, new Agent
    assert first["status"] == "timeout"
    assert second["status"] == "completed" and second["output"] == "all done"
    assert [r.status for r in records] == ["timeout", "completed"]


# ---- real clock, real orchestrator ------------------------------------------------------
@tool
def sleepy_tool(x: int) -> int:
    """Really sleeps 0.3 s."""
    time.sleep(0.3)
    return x


def _slept_harness(orch_turns, worker_turns, timeout):
    om = ScriptedModel(orch_turns, name="orch", on_exhausted="raise")
    wm = ScriptedModel(worker_turns, name="worker")
    h = build_harness(
        cfg(om, "orch"), {"w": cfg(wm, "w", [sleepy_tool])}, worker_timeout_s=timeout,
    )
    return h, om, wm


SLEEP_CALL = CALL.replace("slow_tool", "sleepy_tool")
DELEGATE = 'Thought: o.\nAction: delegate\nAction Input: {"worker": "w", "task": "do it"}'


def test_orchestrator_continues_after_worker_timeout_real_clock():
    h, om, wm = _slept_harness(
        [DELEGATE, "Thought: d.\nFinal Answer: worker timed out, reporting that"],
        [SLEEP_CALL, ANSWER], timeout=0.1,
    )
    t0 = time.monotonic()
    result = h.run("USER TASK")
    elapsed = time.monotonic() - t0
    assert result.success is True and result.termination == "done"
    assert [w.status for w in result.workers] == ["timeout"]
    report = json.loads(om.calls[1].prompt.split("Observation:")[-1])
    assert report["status"] == "timeout" and report["error_type"] == "WorkerTimeout"
    assert len(wm.calls) == 1
    # Cooperative limit, shown plainly: the 0.1 s deadline did not cut the 0.3 s tool short.
    assert elapsed >= 0.3


def test_timeout_state_does_not_leak_between_harness_runs():
    h, om, wm = _slept_harness(
        [DELEGATE, "Final Answer: first", DELEGATE, "Final Answer: second"],
        [SLEEP_CALL, ANSWER, ANSWER],  # run 2's worker answers without a tool call
        timeout=0.1,
    )
    r1 = h.run("one")
    r2 = h.run("two")
    assert [w.status for w in r1.workers] == ["timeout"]
    assert [w.status for w in r2.workers] == ["completed"]
    assert r1.workers is not r2.workers
