"""T1.3: user task -> orchestrator -> delegate -> fresh worker -> JSON report -> orchestrator answer.

All models are scripted; every agent runs through effGen's real loop.
"""

from __future__ import annotations

import json
import re

import pytest
from conftest import ScriptedModel
from effgen import AgentConfig, tool

from smol_harness.harness import Harness, HarnessResult, build_harness

ORCH_SYSTEM = "ORCH-SYSTEM-MARKER: you coordinate workers."
USER_TASK = "USER-TASK-MARKER: please get foo.py fixed."
ORCH_THOUGHT = "ORCH-THOUGHT-MARKER"
WORKER_THOUGHT = "WORKER-PRIVATE-THOUGHT"
WORKER_OBSERVATION = "WORKER-INTERNAL-OBSERVATION"
WORKER_ANSWER = "patched foo.py (1 line)"


def D(**args) -> str:
    """An orchestrator turn that calls `delegate`."""
    return f"Thought: {ORCH_THOUGHT}.\nAction: delegate\nAction Input: {json.dumps(args)}"


def final(text: str) -> str:
    return f"Thought: done.\nFinal Answer: {text}"


def cfg(model, name, tools=(), **over) -> AgentConfig:
    base = dict(
        name=name, model=model, tools=list(tools), tool_calling_mode="react",
        capability_probe=False, enable_memory=False, max_iterations=6, raise_on_error=False,
    )
    base.update(over)
    return AgentConfig(**base)


@tool
def inspect_file(path: str) -> str:
    """Return the contents of a file."""
    return WORKER_OBSERVATION


def worker_turns(answer: str = WORKER_ANSWER) -> list[str]:
    return [
        f'Thought: {WORKER_THOUGHT}.\nAction: inspect_file\nAction Input: {{"path": "foo.py"}}',
        final(answer),
    ]


def make(orch_turns, worker_script=None, *, orch_exhaust="raise", worker_tools=(inspect_file,), worker_exhaust="repeat_last"):
    # NB: a worker that holds a tool but answers without calling it is nudged by
    # effGen ("You have not run a tool yet...") and consumes an extra scripted
    # turn, so tool-less scripts must use worker_tools=().
    orch_model = ScriptedModel(orch_turns, name="orch-model", on_exhausted=orch_exhaust)
    worker_model = ScriptedModel(worker_turns() if worker_script is None else worker_script,
                                 name="worker-model", on_exhausted=worker_exhaust)
    harness = build_harness(
        cfg(orch_model, "orch", system_prompt=ORCH_SYSTEM),
        {"coder": cfg(worker_model, "coder", worker_tools)},
        {"coder": "edits code"},
    )
    return harness, orch_model, worker_model


@pytest.fixture
def e2e():
    harness, om, wm = make([
        D(worker="coder", task="Fix the off-by-one in foo.py", context="file: foo.py"),
        final(f"The coder reported: {WORKER_ANSWER}"),
    ])
    return harness, om, wm, harness.run(USER_TASK)


# ---- acceptance 1-3, 7: the delegation path --------------------------------------
def test_flow_succeeds_and_final_answer_uses_worker_output(e2e):
    _, om, _, result = e2e
    assert isinstance(result, HarnessResult)
    assert result.success is True and result.termination == "done"
    assert WORKER_ANSWER in result.output  # (3)
    assert len(om.calls) == 2  # delegate turn + final turn, nothing extra


def test_worker_ran_exactly_once_with_templated_task(e2e):
    _, _, wm, _ = e2e
    assert len(wm.calls) == 2  # its own loop: one tool turn + one answer turn
    first = wm.calls[0].prompt
    assert "worker agent 'coder'" in first  # (1) template
    assert "Fix the off-by-one in foo.py" in first and "file: foo.py" in first


def test_orchestrator_receives_the_json_report_as_observation(e2e):
    _, om, _, _ = e2e
    after = om.calls[1].prompt  # (2)
    obs = after.split("Observation:")[-1].strip()
    report = json.loads(obs)
    assert report["worker"] == "coder" and report["status"] == "completed"
    assert report["output"] == WORKER_ANSWER and report["tool_calls"] == 1


def test_harness_result_has_worker_record(e2e):
    _, _, _, result = e2e
    assert [w.status for w in result.workers] == ["completed"]  # (7)
    rec = result.workers[0]
    assert rec.worker == "coder" and rec.task == "Fix the off-by-one in foo.py"
    assert rec.context == "file: foo.py" and rec.result.output == WORKER_ANSWER


# ---- acceptance 4-6: isolation ------------------------------------------------------
def test_worker_prompts_contain_no_orchestrator_context(e2e):
    _, om, wm, _ = e2e
    assert ORCH_SYSTEM in om.calls[0].prompt  # the marker is really in the parent's prompt
    for call in wm.calls:  # (4)
        for marker in (ORCH_SYSTEM, USER_TASK, ORCH_THOUGHT):
            assert marker not in call.prompt


def test_orchestrator_prompts_contain_only_the_report(e2e):
    _, om, _, _ = e2e
    for call in om.calls:  # (5)
        assert WORKER_THOUGHT not in call.prompt
        assert WORKER_OBSERVATION not in call.prompt
        assert "Fix the off-by-one" not in call.prompt.split("Question:")[0]
    assert WORKER_ANSWER in om.calls[1].prompt  # but the report's answer is there


def test_worker_does_not_hold_or_see_the_delegate_tool(e2e):
    harness, _, wm, _ = e2e
    assert "delegate" not in harness.registry.create("coder").tools  # (6)
    assert "inspect_file" in wm.calls[0].prompt
    assert not any(re.search(r"\bdelegate\b", c.prompt) for c in wm.calls)


def test_each_delegation_gets_a_fresh_worker_agent(monkeypatch):
    harness, om, wm = make([
        D(worker="coder", task="first-job"),
        D(worker="coder", task="second-job"),
        final("both done"),
    ], worker_script=[final("one"), final("two")], worker_tools=())
    created = []
    real_create = harness.registry.create
    monkeypatch.setattr(harness.registry, "create", lambda n: created.append(real_create(n)) or created[-1])
    result = harness.run(USER_TASK)
    assert result.success and [w.status for w in result.workers] == ["completed", "completed"]
    assert len(created) == 2 and created[0] is not created[1]
    assert created[0].model is created[1].model is wm  # shared model instance
    assert "first-job" in wm.calls[0].prompt and "first-job" not in wm.calls[1].prompt


# ---- lifecycle records ----------------------------------------------------------------
def test_failed_worker_is_recorded_and_parent_continues():
    harness, om, wm = make(
        [D(worker="coder", task="t"), final("worker failed; reporting that")],
        worker_script=[], worker_tools=(), worker_exhaust="raise",  # worker's model raises on first call
    )
    result = harness.run(USER_TASK)
    assert [w.status for w in result.workers] == ["failed"]
    assert result.workers[0].result.error
    assert result.success is True and result.termination == "done"  # U3: not ended as tool_failed
    assert json.loads(om.calls[1].prompt.split("Observation:")[-1])["status"] == "failed"


def test_worker_holding_a_tool_is_nudged_if_it_answers_without_using_it():
    # effGen 1.3.0 behaviour that shapes worker scripts (audit Part 3, T1.3).
    harness, _, wm = make([D(worker="coder", task="t"), final("ok")], worker_script=[final("a"), final("b")])
    result = harness.run(USER_TASK)
    assert len(wm.calls) == 2 and "You have not run a tool yet" in wm.calls[1].prompt
    assert result.workers[0].result.output == "b"


def test_unknown_worker_is_recorded_as_failed():
    harness, _, wm = make([D(worker="nobody", task="t"), final("no such worker")])
    result = harness.run(USER_TASK)
    assert [(w.worker, w.status) for w in result.workers] == [("nobody", "failed")]
    assert result.success and wm.calls == []


def test_no_delegation_means_no_records():
    harness, _, _ = make([final("answered directly")])
    result = harness.run(USER_TASK)
    assert result.success and result.workers == []


def test_runs_are_independent_and_records_do_not_leak():
    harness, om, wm = make(
        [D(worker="coder", task="job-A"), final("A done"), D(worker="coder", task="job-B"), final("B done")],
        worker_script=[final("a"), final("b")], worker_tools=(),
    )
    r1 = harness.run("USER-RUN-ONE")
    r2 = harness.run("USER-RUN-TWO")
    assert [w.task for w in r1.workers] == ["job-A"]
    assert [w.task for w in r2.workers] == ["job-B"]
    assert r1.workers is not r2.workers
    assert "USER-RUN-ONE" not in om.calls[2].prompt  # fresh orchestrator, no carried memory
    assert "job-A" not in wm.calls[1].prompt


def test_orchestrator_memory_does_not_carry_across_runs():
    """Focused regression: no delegation involved, so only orchestrator memory can leak.

    effGen agents keep short-term memory across run() calls (audit §1.4); Harness.run
    must therefore build a fresh orchestrator Agent each time.
    """
    harness, om, _ = make([final("first answer"), final("second answer")])
    harness.run("USER-RUN-ONE: remember the word PINEAPPLE")
    r2 = harness.run("USER-RUN-TWO")
    second_prompt = om.calls[1].prompt
    assert "USER-RUN-TWO" in second_prompt
    assert "USER-RUN-ONE" not in second_prompt
    assert "PINEAPPLE" not in second_prompt
    assert "first answer" not in second_prompt
    assert r2.output == "second answer"


def test_orchestrator_config_must_not_hold_delegate():
    harness, _, _ = make([final("x")])
    from smol_harness.delegation import DelegateTool

    bad = cfg(ScriptedModel(["x"]), "o", [DelegateTool(harness.registry)])
    with pytest.raises(ValueError, match="must not already hold"):
        Harness(bad, harness.registry)


# ---- U11: invalid delegate arguments from the orchestrator ----------------------------------
BAD_ARGS = [
    {"worker": "coder"},                  # missing task: pre-dispatch check
    {"task": "x"},                        # missing worker: pre-dispatch check
    {"worker": "coder", "task": ""},      # effGen parameter validation
    {"worker": "coder", "task": 5},
    {"worker": 3, "task": "x"},
]


@pytest.mark.parametrize("bad", BAD_ARGS)
def test_u11_single_invalid_call_gives_error_observation_and_parent_continues(bad):
    harness, om, wm = make([D(**bad), final("recovered")])
    result = harness.run(USER_TASK)
    obs = om.calls[1].prompt.split("Observation:")[-1]
    assert ("could not be run" in obs) or ("Parameter validation failed" in obs)
    assert result.success is True and result.output == "recovered"
    assert result.termination in ("done", "not_possible")
    assert result.workers == [] and wm.calls == []  # worker never ran; no record is created


def test_u11_several_different_invalid_calls_do_not_end_the_parent():
    harness, om, wm = make([*(D(**b) for b in BAD_ARGS), final("recovered")])
    result = harness.run(USER_TASK)
    assert result.success is True and result.output == "recovered"
    assert result.termination != "tool_failed"
    assert wm.calls == []


def test_u11_endless_identical_invalid_calls_end_as_stuck_not_tool_failed():
    harness, om, _ = make([D(worker="coder")], orch_exhaust="repeat_last")
    result = harness.run(USER_TASK)
    assert result.success is False
    assert result.termination == "stuck" and result.termination != "tool_failed"
    assert result.response.stop_reason == "loop_detected"
    assert len(om.calls) < 6  # ended by effGen's repeat guard before max_iterations


def test_u11_endless_varied_invalid_calls_end_at_max_iterations():
    harness, om, _ = make([*(D(**b) for b in BAD_ARGS), D(worker="coder")], orch_exhaust="repeat_last")
    result = harness.run(USER_TASK)
    assert result.success is False and result.termination == "stuck"
    assert result.response.stop_reason == "max_iterations_exhausted"
    assert len(om.calls) == 6
