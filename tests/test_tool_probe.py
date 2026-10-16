"""The tool-calling probe must not mistake a lucky answer for tool use (validated with scripted models)."""

from __future__ import annotations

from conftest import ScriptedModel
from tool_probe import probe_case

A, B, TOTAL = 4817, 9235, 14052
REACT_CALL = f'Thought: use it.\nAction: add\nAction Input: {{"a": {A}, "b": {B}}}'


def test_react_call_that_really_runs_is_a_success():
    model = ScriptedModel([REACT_CALL, f"Thought: done.\nFinal Answer: {TOTAL}"], on_exhausted="raise")
    r = probe_case(model, "react", a=A, b=B)
    assert r["selected_tool"] and r["parsed_by_effgen"] and r["tool_executed"] and r["tool_result_returned"]
    assert r["final_answer_success"] and r["tool_success"] and r["tool_arguments"] == [(A, B)]
    assert r["model_calls"] == 2 and r["tool_calls"] == 1 and r["termination"] == "done"


def test_a_correct_answer_without_a_tool_call_is_not_tool_success():
    model = ScriptedModel([f"Final Answer: {TOTAL}"])  # right number, tool never used
    r = probe_case(model, "react", a=A, b=B)
    assert r["answer_contains_expected"] is True
    assert not r["selected_tool"] and not r["tool_executed"] and r["tool_calls"] == 0
    assert r["tool_success"] is False


def test_prose_that_mentions_the_tool_is_not_a_selection():
    model = ScriptedModel([f"The tool can be used as follows: add({A},{B}) returns {TOTAL}. Final answer: {TOTAL}"])
    r = probe_case(model, "react", a=A, b=B)
    assert r["selected_tool"] is False and r["tool_executed"] is False and r["tool_success"] is False


def test_an_unparsed_native_style_call_is_selected_but_not_parsed_or_executed():
    text = f"<|tool_call_start|>[add(a={A}, b={B})]<|tool_call_end|>"
    model = ScriptedModel([text])  # repeats forever, as the real model did
    r = probe_case(model, "native", a=A, b=B)
    assert r["selected_tool"] is True  # the model chose the tool ...
    assert r["parsed_by_effgen"] is False and r["tool_executed"] is False  # ... but effGen never ran it
    assert r["tool_success"] is False and r["final_answer_success"] is False
    assert r["raw_outputs"][0] == text  # raw output is kept for diagnosis


def test_probe_restores_the_models_generate_and_records_every_call():
    model = ScriptedModel([REACT_CALL, f"Final Answer: {TOTAL}"])
    probe_case(model, "react", a=A, b=B)
    assert "generate" not in vars(model)  # the spy was removed
    assert len(model.calls) == 2


def test_probe_reports_a_failed_run_without_raising():
    model = ScriptedModel([], on_exhausted="raise")
    r = probe_case(model, "react", a=A, b=B)
    assert r["final_answer_success"] is False and r["tool_success"] is False and r["model_calls"] == 0


def test_result_returned_requires_a_prompt_sent_after_the_tool_ran():
    # The sum appears in the model's own first output (a hallucination), but the tool never runs.
    model = ScriptedModel([f"Final Answer: {TOTAL}"])
    r = probe_case(model, "react", a=A, b=B)
    assert r["tool_result_returned"] is False
    # When the tool does run, the observation reaches the next prompt.
    model = ScriptedModel([REACT_CALL, f"Final Answer: {TOTAL}"])
    assert probe_case(model, "react", a=A, b=B)["tool_result_returned"] is True
