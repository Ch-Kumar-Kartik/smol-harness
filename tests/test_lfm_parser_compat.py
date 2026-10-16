"""Record of how effGen 1.3.0 reads LFM2.5's tool-call formats (no model needed; pinned to this effGen).

Findings these tests pin (see docs/effgen-api-audit.md, T3.2c). If an effGen upgrade changes any of them,
the tests fail on purpose: revisit the audit and the recommended tool-calling mode for LFM models.
"""

from __future__ import annotations

import pytest
from effgen import tool
from effgen.core.tool_calling import get_strategy
from effgen.models.transformers_engine import TransformersEngine


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


TOOLS = {"add": add}
A, B = 4817, 9235
PYTHONIC = f"<|tool_call_start|>[add(a={A}, b={B})]<|tool_call_end|>"  # LFM2.5's default call format (model card)
PYTHONIC_POSITIONAL = f"<|tool_call_start|>[add({A}, {B})]<|tool_call_end|>"
JSON_LIST = f'<|tool_call_start|>[{{"name": "add", "arguments": {{"a": {A}, "b": {B}}}}}]<|tool_call_end|>'  # card's JSON option
BARE_JSON = f'{{"name": "add", "arguments": {{"a": {A}, "b": {B}}}}}'
REACT = f'Action: add  \nAction Input: {{"a": {A}, "b": {B}}}  \nObservation: the result is 51002\n\nThought: I now know the final answer.'


def parse(mode: str, text: str):
    return get_strategy(mode=mode, model=None).parse_response(text, TOOLS)


@pytest.mark.parametrize("mode, text, is_call", [
    ("native", PYTHONIC, False),  # the model's default format is NOT understood by effGen's native reader
    ("native", PYTHONIC_POSITIONAL, False),
    ("native", JSON_LIST, True),  # ... but the card's JSON variant is (via effGen's generic JSON reader)
    ("native", BARE_JSON, True),
    ("native", REACT, False),
    ("react", PYTHONIC, False),
    ("react", JSON_LIST, False),
    ("react", REACT, True),
    ("hybrid", PYTHONIC, False),  # hybrid adds nothing for the default format
    ("hybrid", JSON_LIST, True),
    ("hybrid", REACT, True),
])
def test_effgen_reads_which_lfm_call_formats(mode, text, is_call):
    result = parse(mode, text)
    assert result.is_tool_call is is_call
    if is_call:
        assert result.tool_name == "add" and result.arguments == {"a": A, "b": B}


def test_the_default_lfm_call_is_not_even_an_error_to_the_native_reader_it_looks_like_an_answer():
    # What the real agent run then did: it noticed a written tool call and stopped with `written_tool_call`.
    result = parse("native", PYTHONIC)
    assert result.is_tool_call is False and result.final_answer is not None


def test_transformers_engine_wraps_the_whole_prompt_as_one_user_message():
    """effGen cannot place AgentConfig.system_prompt in a system turn: this is why the model card's
    "request JSON calls in the system prompt" option had no effect through effGen."""
    seen: dict = {}

    class FakeTokenizer:
        chat_template = "x"

        def apply_chat_template(self, messages, **kwargs):
            seen["messages"], seen["kwargs"] = messages, kwargs
            return "rendered"

    fake_engine = type("E", (), {"tokenizer": FakeTokenizer(), "chat_template_kwargs": {}})()
    prompt = "Output function calls as JSON.\n\nUse the add tool.\n\nUse the tools to do this task."
    out = TransformersEngine._apply_chat_template(fake_engine, prompt, [{"type": "function"}])
    assert out == "rendered"
    assert seen["messages"] == [{"role": "user", "content": prompt}]  # one user turn, no system turn
    assert seen["kwargs"]["tools"] == [{"type": "function"}] and seen["kwargs"]["add_generation_prompt"] is True
