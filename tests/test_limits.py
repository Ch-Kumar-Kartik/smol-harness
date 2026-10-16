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
from effgen import Agent, AgentConfig, RunStoppedError, tool
from effgen.models.errors import BackendUnreachableError

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


# =====================================================================================
# T2.2: worker failure propagation
# =====================================================================================
@pytest.fixture(autouse=True)
def backoff_sleeps(monkeypatch):
    """effGen retries a failing generation 3 times with 0.5 s / 1.0 s sleeps and no config
    knob (agent_generation.py:_generate_instrumented, audit T2.2). Make the sleeps instant
    for tests, and record them so the retry behaviour itself can be asserted."""
    import effgen.core.ledger as ledger

    sleeps: list[float] = []
    monkeypatch.setattr(ledger, "backoff_sleep", lambda sleep_fn, seconds: sleeps.append(seconds))
    return sleeps


class RaisingModel(ScriptedModel):
    """A model whose every generation raises (what a dead backend or a bug looks like)."""

    def __init__(self, factory, **kw) -> None:
        super().__init__(["unused"], **kw)
        self.factory = factory
        self.attempts = 0

    def generate(self, prompt, config=None, **kw):
        self.attempts += 1
        raise self.factory()


def unreachable() -> Exception:
    return BackendUnreachableError("openai_compatible", "m", "connection refused", "http://127.0.0.1:11434/v1")


@tool
def step(n: int) -> int:
    """A harmless step."""
    return n


@tool
def flaky(q: str) -> str:
    """Always fails on its own side."""
    raise ConnectionError("upstream down")


STEP_TURNS = [f'Thought: t.\nAction: step\nAction Input: {{"n": {i}}}' for i in range(1, 10)]
FLAKY_TURNS = [f'Thought: t.\nAction: flaky\nAction Input: {{"q": "{c}"}}' for c in "abc"] + ["Final Answer: gave up"]

# kind -> (model factory, tools, worker max_iterations)
KINDS = {
    "iteration_cap": (lambda: ScriptedModel(STEP_TURNS), [step], 2),
    "backend_unreachable": (lambda: RaisingModel(unreachable), [], 4),
    "unexpected_exception": (lambda: RaisingModel(lambda: ValueError("boom")), [], 4),
    "worker_tool_failure": (lambda: ScriptedModel(FLAKY_TURNS), [flaky], 6),
}
# (kind, raise_on_error) -> (error_type, termination, text that must be in `error`)
EXPECTED = {
    ("iteration_cap", False): ("worker_run_failed", "stuck", "Stopped after 2 iterations"),
    ("iteration_cap", True): ("worker_run_failed", "stuck", "Stopped after 2 iterations"),  # same report as above
    ("backend_unreachable", False): ("BackendUnreachableError", None, "did not answer"),
    ("backend_unreachable", True): ("BackendUnreachableError", None, "did not answer"),
    ("unexpected_exception", False): ("worker_run_failed", "error", "boom"),
    ("unexpected_exception", True): ("RuntimeError", None, "boom"),  # effGen wraps it and drops the response
    ("worker_tool_failure", False): ("worker_run_failed", "tool_failed", "failed on its own side"),
    ("worker_tool_failure", True): ("worker_run_failed", "tool_failed", "failed on its own side"),
}
CASES = [pytest.param(k, roe, id=f"{k}-raise_on_error={roe}") for k in KINDS for roe in (False, True)]


def failing_worker_cfg(kind, roe, name="w") -> AgentConfig:
    make_model, tools, max_iter = KINDS[kind]
    return cfg(make_model(), name=name, tools=tools, max_iterations=max_iter, raise_on_error=roe)


# ---- how effGen 1.3.0 itself surfaces each failure (pins the assumptions) ------------
@pytest.mark.parametrize("kind, roe", CASES)
def test_effgen_surfaces_worker_failures_as_documented(kind, roe):
    agent = Agent(failing_worker_cfg(kind, roe, name="raw"))
    try:
        outcome = agent.run("q")
        raised = None
    except Exception as exc:  # noqa: BLE001 - recording which exception effGen chose
        outcome, raised = None, exc

    if kind == "backend_unreachable":  # raised whatever raise_on_error says, and not a RuntimeError
        assert type(raised) is BackendUnreachableError and not isinstance(raised, RuntimeError)
    elif roe is False:  # failures come back as a response
        assert raised is None and outcome.success is False
        assert outcome.termination == EXPECTED[(kind, roe)][1]
    elif kind == "unexpected_exception":  # plain RuntimeError, original type only in the text
        assert type(raised) is RuntimeError and str(raised) == "ValueError: boom"
    else:  # iteration cap / worker tool failure: RunStoppedError carrying the response
        assert isinstance(raised, RunStoppedError) and isinstance(raised, RuntimeError)
        assert raised.response.success is False
        assert raised.response.termination == EXPECTED[(kind, roe)][1]
        assert raised.stop_reason in ("max_iterations_partial", "tool_failed")


# ---- DelegateTool turns every one of them into the same structured failure ------------
@pytest.mark.parametrize("kind, roe", CASES)
def test_delegate_tool_reports_each_failure_as_structured_json(kind, roe):
    reg = WorkerRegistry()
    reg.register("w", failing_worker_cfg(kind, roe))
    raw = asyncio.run(DelegateTool(reg).execute(worker="w", task="t"))
    assert raw.success is True  # a worker failure is NOT a tool failure (audit §1.2)
    data = json.loads(raw.output)
    error_type, termination, text = EXPECTED[(kind, roe)]
    assert set(data) == RESULT_KEYS
    assert data["status"] == "failed" and data["worker"] == "w" and data["output"] == ""
    assert data["error_type"] == error_type and data["termination"] == termination
    assert text in data["error"]


def test_effgen_retries_a_failing_backend_before_the_worker_is_reported_failed(backoff_sleeps):
    model = RaisingModel(unreachable)
    reg = WorkerRegistry()
    reg.register("w", cfg(model, raise_on_error=False))
    data = json.loads(asyncio.run(DelegateTool(reg).execute(worker="w", task="t")).output)
    assert data["status"] == "failed" and data["error_type"] == "BackendUnreachableError"
    assert model.attempts == 3 and backoff_sleeps == [0.5, 1.0]  # real runs would wait ~1.5 s here


def test_both_raise_modes_give_the_same_report_for_run_stopped_failures():
    def report(kind, roe):
        reg = WorkerRegistry()
        reg.register("w", failing_worker_cfg(kind, roe))
        out = json.loads(asyncio.run(DelegateTool(reg).execute(worker="w", task="t")).output)
        return {k: out[k] for k in ("status", "error_type", "termination", "iterations", "tool_calls")}

    for kind in ("iteration_cap", "worker_tool_failure"):
        assert report(kind, True) == report(kind, False)
    assert report("iteration_cap", True)["iterations"] == 2  # progress survives the exception


def test_failure_is_not_mistaken_for_a_timeout_when_a_deadline_is_set():
    clock = FakeClock()  # never advances: the deadline cannot expire
    reg = WorkerRegistry()
    reg.register("w", failing_worker_cfg("backend_unreachable", True))
    data = json.loads(asyncio.run(DelegateTool(reg, worker_timeout_s=5.0, clock=clock).execute(worker="w", task="t")).output)
    assert data["status"] == "failed" and data["error_type"] == "BackendUnreachableError"


def test_timeout_and_failure_stay_distinct_statuses_in_one_run():
    clock = FakeClock()
    slow_tool, _ = make_slow_tool(clock, advance=10.0)
    reg = WorkerRegistry()
    reg.register("slow", cfg(ScriptedModel([CALL, ANSWER]), name="slow", tools=[slow_tool]))
    reg.register("dead", failing_worker_cfg("backend_unreachable", False, name="dead"))
    records: list = []
    dt = DelegateTool(reg, records, worker_timeout_s=5.0, clock=clock)
    assert delegate(dt, worker="slow", task="t")["status"] == "timeout"
    assert delegate(dt, worker="dead", task="t")["status"] == "failed"
    assert [r.status for r in records] == ["timeout", "failed"]


# ---- the parent keeps going ---------------------------------------------------------------------
def delegate_turn(task="do it") -> str:
    return f'Thought: o.\nAction: delegate\nAction Input: {json.dumps({"worker": "w", "task": task})}'


@pytest.mark.parametrize("kind, roe", CASES)
def test_orchestrator_continues_after_each_failure_kind(kind, roe):
    om = ScriptedModel([delegate_turn(), "Thought: d.\nFinal Answer: worker failed, reporting that"],
                       name="orch", on_exhausted="raise")
    h = build_harness(cfg(om, "orch"), {"w": failing_worker_cfg(kind, roe)})
    result = h.run("USER TASK")
    assert result.success is True and result.output == "worker failed, reporting that"
    assert result.termination == "done" and result.termination != "tool_failed"
    assert [w.status for w in result.workers] == ["failed"]
    report = json.loads(om.calls[1].prompt.split("Observation:")[-1])  # the parent saw the failure
    assert report["status"] == "failed" and report["error_type"] == EXPECTED[(kind, roe)][0]
    assert len(om.calls) == 2


def test_several_different_failing_delegations_do_not_end_the_parent_as_tool_failed():
    om = ScriptedModel([delegate_turn("a"), delegate_turn("b"), delegate_turn("c"), "Final Answer: all three failed"],
                       name="orch", on_exhausted="raise")
    h = build_harness(cfg(om, "orch"), {"w": failing_worker_cfg("backend_unreachable", True)})
    result = h.run("USER TASK")
    assert [w.status for w in result.workers] == ["failed"] * 3
    assert [w.result.delegation_id for w in result.workers] == ["d1", "d2", "d3"]
    assert result.success is True and result.termination == "done"


def test_identical_failure_reports_need_a_delegation_id_or_effgen_stops_the_parent():
    # Regression for the T2.2 finding: without unique reports, effGen's `repeated_tool_result`
    # guard ends a run after two identical 'delegate' results (success=False, stuck), even
    # though the parent made different calls. delegation_id keeps every report distinct.
    reg = WorkerRegistry()
    reg.register("w", failing_worker_cfg("backend_unreachable", True))
    dt = DelegateTool(reg)
    a, b = delegate(dt, worker="w", task="a"), delegate(dt, worker="w", task="b")
    assert {k: v for k, v in a.items() if k != "delegation_id"} == {k: v for k, v in b.items() if k != "delegation_id"}
    assert (a["delegation_id"], b["delegation_id"]) == ("d1", "d2")


def test_delegation_ids_restart_per_run_and_cover_successes():
    h = build_harness(
        cfg(ScriptedModel([delegate_turn("a"), "Final Answer: x", delegate_turn("b"), "Final Answer: y"], name="o"), "orch"),
        {"w": cfg(ScriptedModel([ANSWER]), "w")},
    )
    assert [w.result.delegation_id for w in h.run("1").workers] == ["d1"]
    assert [w.result.delegation_id for w in h.run("2").workers] == ["d1"]  # per run, not global


def test_retrying_the_identical_failing_delegation_forever_ends_stuck_not_tool_failed():
    # effGen's repeat guard, same as U11: the parent's own repetition ends the run.
    om = ScriptedModel([delegate_turn()], name="orch", on_exhausted="repeat_last")
    h = build_harness(cfg(om, "orch"), {"w": failing_worker_cfg("backend_unreachable", True)})
    result = h.run("USER TASK")
    assert result.success is False
    assert result.termination == "stuck" and result.termination != "tool_failed"
    assert result.response.stop_reason == "loop_detected"
    assert len(result.workers) == 1  # effGen does not re-dispatch the repeated identical call
    assert len(om.calls) == 4
