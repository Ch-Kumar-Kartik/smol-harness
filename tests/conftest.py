"""Shared test fixtures.

`ScriptedModel` is a deterministic effGen `BaseModel` that replays a fixed list
of response texts, so effGen agents can be driven end-to-end on CPU with no
model and no network. Modelled on effGen's own `_Scripted` fake
(tests/unit/test_written_action_runs.py upstream, commit 37de05a).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pytest
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount


class ScriptExhaustedError(RuntimeError):
    """The model was asked for more turns than were scripted."""


@dataclass
class ModelCall:
    """One generation request, as the model received it."""

    prompt: Any
    config: Any
    kwargs: dict[str, Any]
    response: str

    @property
    def tools(self) -> Any:
        return self.kwargs.get("tools")


class ScriptedModel(BaseModel):
    """Replays `turns` in order and records every call.

    Args:
        turns: Response texts, one per model call. For a ReAct agent a tool
            call turn is e.g. ``"Thought: ...\\nAction: tool\\nAction Input: {...}"``
            and an answer turn is ``"Thought: ...\\nFinal Answer: ..."``.
        on_exhausted: ``"repeat_last"`` (default) keeps returning the final turn,
            because effGen 1.3.0's nudges and closing request can consume more
            turns than a script anticipates (plan T1.1 risk). ``"raise"`` raises
            `ScriptExhaustedError`, for tests that assert exact call counts.
        name: Reported as the model name.
    """

    def __init__(
        self,
        turns: list[str],
        *,
        on_exhausted: str = "repeat_last",
        name: str = "scripted",
    ) -> None:
        if on_exhausted not in ("repeat_last", "raise"):
            raise ValueError(f"on_exhausted must be 'repeat_last' or 'raise', got {on_exhausted!r}")
        if not turns and on_exhausted == "repeat_last":
            raise ValueError("repeat_last needs at least one scripted turn")
        super().__init__(model_name=name, model_type=ModelType.OPENAI)
        self.turns = list(turns)
        self.on_exhausted = on_exhausted
        self.calls: list[ModelCall] = []

    # -- script ---------------------------------------------------------
    @property
    def exhausted(self) -> bool:
        """True once every scripted turn has been consumed."""
        return len(self.calls) >= len(self.turns)

    def _next(self, prompt: Any, config: Any, kwargs: dict[str, Any]) -> str:
        i = len(self.calls)
        if i < len(self.turns):
            text = self.turns[i]
        elif self.on_exhausted == "repeat_last":
            text = self.turns[-1]
        else:
            raise ScriptExhaustedError(
                f"{self.model_name!r}: call {i + 1} requested but only {len(self.turns)} turn(s) scripted"
            )
        self.calls.append(ModelCall(prompt, config, dict(kwargs), text))
        return text

    # -- BaseModel interface ---------------------------------------------
    def load(self) -> None: ...

    def unload(self) -> None: ...

    def generate(self, prompt, config=None, **kw) -> GenerationResult:
        text = self._next(prompt, config, kw)
        return GenerationResult(
            text=text, tokens_used=len(text.split()), finish_reason="stop",
            model_name=self.model_name, metadata={},
        )

    def generate_stream(self, prompt, config=None, **kw):
        text = self._next(prompt, config, kw)
        for i in range(0, len(text), 8):
            yield text[i:i + 8]

    def count_tokens(self, text) -> TokenCount:
        return TokenCount(count=len(str(text).split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 8192


@pytest.fixture
def scripted_model():
    """Factory: ``scripted_model(["turn 1", "turn 2"], name="orch")``."""
    return ScriptedModel


# ---- real-model opt-in -------------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-real-models", action="store_true", default=False,
        help="run tests marked `real_model`: they load real checkpoints and may download weights",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Tests marked `real_model` never run by accident: they need --run-real-models (or SMOL_RUN_REAL_MODELS=1)."""
    if config.getoption("--run-real-models") or os.environ.get("SMOL_RUN_REAL_MODELS") == "1":
        return
    skip = pytest.mark.skip(reason="real-model test (may download weights): opt in with --run-real-models")
    for item in items:
        if "real_model" in item.keywords:
            item.add_marker(skip)
