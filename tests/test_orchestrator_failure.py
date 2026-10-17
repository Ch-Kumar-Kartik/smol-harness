"""Review F3: an orchestrator that effGen stops must come back as a `HarnessResult`, with the run's worker records.

Under effGen's default `raise_on_error=True`, `Agent.run` raises `RunStoppedError` when the loop stops a run before
an answer (outcome "stopped": iteration cap, repeat guards, ...). The error carries the very `AgentResponse` that
`raise_on_error=False` would have returned (effgen/errors.py, core/agent_generation.py `_reconstruct_error`). Every
other failure under `True` is a different exception with NO response (provider errors, guardrail blocks, a model
that raises), which `Harness.run` must not turn into a result: it propagates, carrying the run's worker records.

Real effGen execution paths and scripted models throughout; `Agent.run` is replaced only where the test needs an
exception effGen itself would never construct.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from conftest import ScriptedModel
from effgen import Agent, AgentConfig, RunStoppedError, tool

from smol_harness import harness as harness_mod
from smol_harness.harness import HarnessResult, build_harness


def call(tool_name: str, /, **args) -> str:
    return f"Thought: next.\nAction: {tool_name}\nAction Input: {json.dumps(args)}"


DELEGATE = call("delegate", worker="w", task="do the job")
WORKER_OK = "Thought: ok.\nFinal Answer: worker done"


def cfg(model, name, tools=(), *, roe: bool, max_iterations: int = 5) -> AgentConfig:
    return AgentConfig(name=name, model=model, tools=list(tools), tool_calling_mode="react", capability_probe=False,
                       enable_memory=False, max_iterations=max_iterations, raise_on_error=roe)


@tool
def lookup(key: str) -> str:
    """Pure lookup."""
    return f"value of {key}"


class RaisingModel(ScriptedModel):
    """A model whose every generation fails (an unreachable backend, a crashed process)."""

    def generate(self, prompt, config=None, **kw):
        raise RuntimeError("worker model is down")


class DiesOnSecondCall(ScriptedModel):
    """Answers the first call (so a delegation happens), then fails."""

    def generate(self, prompt, config=None, **kw):
        if self.calls:
            raise RuntimeError("orchestrator model exploded")
        return super().generate(prompt, config, **kw)


def run_scenario(turns, *, roe: bool, worker_model=None, tools=(lookup,), max_iterations: int = 5):
    worker_model = worker_model or ScriptedModel([WORKER_OK], name="worker")
    orch = ScriptedModel(turns, name="orch")
    harness = build_harness(cfg(orch, "orch", tools, roe=roe, max_iterations=max_iterations),
                            {"w": cfg(worker_model, "w", roe=roe)}, {"w": "does the job"})
    return harness, harness.run("task"), orch


def view(result: HarnessResult) -> tuple:
    """Everything about a result that must not depend on raise_on_error."""
    return (result.success, result.termination, result.response.stop_reason, result.output,
            [(r.worker, r.status, r.result.delegation_id) for r in result.workers])


# Two stop reasons effGen produces for a stopped orchestrator (measured, deterministic with scripted models).
SCENARIOS = {
    "iteration cap after a good delegation": ([DELEGATE] + [call("lookup", key=k) for k in "abcdefg"], "max_iterations_partial"),
    "repeated identical call after a good delegation": ([DELEGATE] + [call("lookup", key="a")] * 7, "loop_detected"),
}


# ---- 1, 2, 4, 6: stopped orchestrators come back as results, in both modes, with the same content ---------------------------
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_a_stopped_orchestrator_returns_the_same_result_whatever_raise_on_error_says(name):
    turns, stop_reason = SCENARIOS[name]
    _, strict, _ = run_scenario(turns, roe=True)  # default effGen setting: Agent.run raises RunStoppedError underneath
    _, lenient, _ = run_scenario(turns, roe=False)
    for result in (strict, lenient):
        assert isinstance(result, HarnessResult)
        assert result.success is False
        assert result.termination == "stuck" and result.response.stop_reason == stop_reason
        assert result.response.success is False and result.output == str(result.response.output)
    assert view(strict) == view(lenient)


def test_the_run_really_is_stopped_by_raising_under_the_default_setting():
    # Guards the premise of the tests above: without Harness's handling, effGen raises for this orchestrator.
    turns, _ = SCENARIOS["iteration cap after a good delegation"]
    orch = ScriptedModel(turns, name="orch")
    config = cfg(orch, "orch", [lookup], roe=True)
    with Agent(config) as agent, pytest.raises(RunStoppedError) as stopped:
        agent.run("task")
    assert stopped.value.response.success is False and stopped.value.stop_reason == "max_iterations_partial"


# ---- 3, 4, 5: the worker records survive, after a good and after a bad delegation ------------------------------------------
@pytest.mark.parametrize("roe", [True, False])
def test_failure_after_a_successful_delegation_keeps_the_worker_record(roe):
    turns, _ = SCENARIOS["iteration cap after a good delegation"]
    _, result, _ = run_scenario(turns, roe=roe)
    assert result.success is False
    (record,) = result.workers
    assert (record.worker, record.status, record.task) == ("w", "completed", "do the job")
    assert record.result.output == "worker done" and record.result.delegation_id == "d1"


@pytest.mark.parametrize("roe", [True, False])
def test_failure_after_an_unsuccessful_delegation_keeps_the_failed_record(roe):
    turns, stop_reason = SCENARIOS["repeated identical call after a good delegation"]
    _, result, _ = run_scenario(turns, roe=roe, worker_model=RaisingModel(["unused"], name="worker"))
    assert result.success is False and result.response.stop_reason == stop_reason
    (record,) = result.workers
    assert record.status == "failed" and "worker model is down" in (record.result.error or "")
    # DelegateTool's existing, documented difference between the modes (test_delegation.py): the worker's own raise_on_error
    assert record.result.error_type == {True: "RuntimeError", False: "worker_run_failed"}[roe]


@pytest.mark.parametrize("roe", [True, False])
def test_failure_with_no_delegation_gives_an_empty_record_list(roe):
    _, result, _ = run_scenario([call("lookup", key=k) for k in "abcdefg"], roe=roe)
    assert result.success is False and result.workers == [] and result.termination == "stuck"


# ---- 7: no leaks between runs ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("roe", [True, False])
def test_records_do_not_leak_between_separate_runs(roe):
    turns, _ = SCENARIOS["iteration cap after a good delegation"]
    orch = ScriptedModel(turns + turns, name="orch", on_exhausted="repeat_last")
    harness = build_harness(cfg(orch, "orch", [lookup], roe=roe), {"w": cfg(ScriptedModel([WORKER_OK], name="worker"), "w", roe=roe)})
    first, second = harness.run("one"), harness.run("two")
    assert first.success is False and second.success is False
    assert first.workers is not second.workers
    assert [r.result.delegation_id for r in first.workers] == ["d1"] and [r.result.delegation_id for r in second.workers] == ["d1"]
    assert len(first.workers) == 1 and len(second.workers) == 1  # the second run neither inherited nor extended the first's


# ---- 8: unrelated exceptions are not swallowed (and keep the records) -------------------------------------------------------------
def test_a_model_exception_under_the_default_setting_propagates_with_the_worker_records():
    orch = DiesOnSecondCall([DELEGATE], name="orch")
    harness = build_harness(cfg(orch, "orch", [], roe=True), {"w": cfg(ScriptedModel([WORKER_OK], name="worker"), "w", roe=True)})
    with pytest.raises(RuntimeError, match="orchestrator model exploded") as err:
        harness.run("task")
    assert type(err.value) is RuntimeError and not isinstance(err.value, RunStoppedError)  # effGen's own exception, no result invented
    assert not hasattr(err.value, "response")
    (record,) = err.value.worker_records  # the delegation that did happen is not lost
    assert (record.worker, record.status, record.result.delegation_id) == ("w", "completed", "d1")
    assert any("1 worker delegation(s) ran before this error" in note for note in err.value.__notes__)


def test_the_same_model_exception_with_raise_on_error_false_is_a_failed_result():
    # Documented asymmetry in effGen, not in the harness: with False the failure is folded into a response.
    orch = DiesOnSecondCall([DELEGATE], name="orch")
    harness = build_harness(cfg(orch, "orch", [], roe=False), {"w": cfg(ScriptedModel([WORKER_OK], name="worker"), "w", roe=False)})
    result = harness.run("task")
    assert result.success is False and result.termination == "error" and result.response.stop_reason == "generation_failed"
    assert [r.status for r in result.workers] == ["completed"]


@pytest.mark.parametrize("roe", [True, False])
@pytest.mark.parametrize("exc", [ZeroDivisionError("bug"), KeyError("missing"), ValueError("bad")])
def test_programming_errors_are_never_swallowed(monkeypatch, roe, exc):
    def broken_run(self, *args, **kwargs):
        raise exc

    monkeypatch.setattr(Agent, "run", broken_run)
    harness = build_harness(cfg(ScriptedModel(["x"], name="orch"), "orch", [], roe=roe),
                            {"w": cfg(ScriptedModel([WORKER_OK], name="worker"), "w", roe=roe)})
    with pytest.raises(type(exc)) as err:
        harness.run("task")
    assert err.value is exc and err.value.worker_records == []  # no delegation happened, and the list says so


def test_base_exceptions_pass_through_untouched(monkeypatch):
    def interrupted(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(Agent, "run", interrupted)
    harness = build_harness(cfg(ScriptedModel(["x"], name="orch"), "orch", [], roe=True),
                            {"w": cfg(ScriptedModel([WORKER_OK], name="worker"), "w", roe=True)})
    with pytest.raises(KeyboardInterrupt) as err:
        harness.run("task")
    assert not hasattr(err.value, "worker_records")  # only Exception subclasses are annotated


def test_the_compatibility_patch_requirement_is_checked_before_and_independently_of_the_new_handling(compat_uninstalled):
    orch = ScriptedModel(["Thought: x\nFinal Answer: hi"], name="orch")
    harness = build_harness(cfg(orch, "orch", [], roe=True), {"w": cfg(ScriptedModel([WORKER_OK], name="worker"), "w", roe=True)})
    with pytest.raises(harness_mod.EffgenCompatError, match=r"install_effgen_loop_compat\(\)"):
        harness.run("task")
    assert orch.calls == []


# ---- 9: missing or malformed exception-response information is handled explicitly -----------------------------------------------
def fake_agent_run(monkeypatch, error):
    def run(self, *args, **kwargs):
        raise error

    monkeypatch.setattr(Agent, "run", run)
    return build_harness(cfg(ScriptedModel(["x"], name="orch"), "orch", [], roe=True),
                         {"w": cfg(ScriptedModel([WORKER_OK], name="worker"), "w", roe=True)})


def stopped_with(**fields) -> RunStoppedError:
    base = dict(output="stopped", stop_reason="loop_detected", partial=None, success=False)
    return RunStoppedError(SimpleNamespace(**{**base, **fields}))


def test_a_well_formed_error_from_a_stub_still_yields_a_result(monkeypatch):
    stopped = stopped_with(termination="stuck")
    result = fake_agent_run(monkeypatch, stopped).run("task")
    assert (result.success, result.termination, result.output) == (False, "stuck", "stopped") and result.response is stopped.response


def test_a_stopped_error_whose_response_was_removed_propagates_instead_of_inventing_a_result(monkeypatch):
    stopped = stopped_with()
    stopped.response = None
    with pytest.raises(RunStoppedError) as err:
        fake_agent_run(monkeypatch, stopped).run("task")
    assert err.value is stopped and err.value.worker_records == []


@pytest.mark.parametrize("response", [
    SimpleNamespace(stop_reason="x", partial=None, success=False),  # no output
    SimpleNamespace(output="x", stop_reason="x", partial=None),  # no success flag
    SimpleNamespace(output="x", stop_reason="x", partial=None, success=True),  # a "stopped" run that claims to have succeeded
    SimpleNamespace(output="x", stop_reason="x", partial=None, success=None),  # success is not a boolean False
])
def test_a_stopped_error_with_a_malformed_response_propagates(monkeypatch, response):
    stopped = RunStoppedError(SimpleNamespace(output="x", stop_reason="x", partial=None, success=False))
    stopped.response = response  # RunStoppedError.__init__ needs .output; a later change (or subclass) need not keep it valid
    with pytest.raises(RunStoppedError) as err:
        fake_agent_run(monkeypatch, stopped).run("task")
    assert err.value is stopped and err.value.worker_records == []


def test_records_reach_a_propagating_stopped_error_too(monkeypatch):
    orch = ScriptedModel([DELEGATE, "Thought: x\nFinal Answer: unreachable"], name="orch")
    harness = build_harness(cfg(orch, "orch", [], roe=True), {"w": cfg(ScriptedModel([WORKER_OK], name="worker"), "w", roe=True)})
    real_run = Agent.run

    def delegate_then_malfunction(self, *args, **kwargs):
        if self.name != "orch":  # only the orchestrator malfunctions; the worker's agent runs for real
            return real_run(self, *args, **kwargs)
        real_run(self, *args, **kwargs)  # the delegation really happens
        bad = stopped_with()
        bad.response = None
        raise bad

    monkeypatch.setattr(Agent, "run", delegate_then_malfunction)
    with pytest.raises(RunStoppedError) as err:
        harness.run("task")
    assert [r.result.delegation_id for r in err.value.worker_records] == ["d1"]
