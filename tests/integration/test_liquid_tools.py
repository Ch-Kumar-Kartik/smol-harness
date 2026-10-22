"""T3.2c: can LiquidAI/LFM2.5-350M act as a tool-using worker under effGen 1.3.0? Native vs ReAct.

OPT-IN (real model, CPU only): skipped unless `--run-real-models` (or SMOL_RUN_REAL_MODELS=1).

    uv run pytest tests/integration/test_liquid_tools.py --run-real-models -s

Needs the snapshot cached by tests/integration/test_liquid_cpu.py (it downloads it if missing). Five short
runs only (<= 64 new tokens each, <= 3 agent iterations), CPU, network blocked, fixed seeds. Results go to
/tmp/smol-harness-lfm-tools.json (override with SMOL_TOOLS_REPORT) and are printed.

The assertions are about the experiment being sound (CPU only, the requested mode really engaged, accounting
consistent) and about the recommended mode working; what native mode does is RECORDED, not pinned, so a future
effGen that learns LFM's call format shows up in the report rather than as a failing test. A correct answer
without an executed tool is never counted as tool-calling success (see tests/tool_probe.py).
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest
import tempfile
import torch
from tool_probe import probe_case

pytestmark = pytest.mark.real_model

MODEL_ID = "LiquidAI/LFM2.5-350M"
REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"
FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
         "tokenizer_config.json", "chat_template.jinja", "LICENSE"]
CARD_WINDOW = 32768  # context length stated on the model card
REPORT_PATH = Path(os.environ.get("SMOL_TOOLS_REPORT", str(Path(tempfile.gettempdir()) / "smol-harness-lfm-tools.json")))
JSON_CALL_PROMPT = "Output function calls as JSON."  # the model card's documented way to ask for JSON calls

CASES = [
    {"mode": "native", "seed": 0},
    {"mode": "native", "seed": 1},
    {"mode": "react", "seed": 0},
    {"mode": "react", "seed": 1},
    {"mode": "native", "seed": 0, "system_prompt": JSON_CALL_PROMPT},  # supplementary, card-documented option
]


@pytest.fixture(scope="module")
def report():
    data: dict = {}
    yield data
    REPORT_PATH.write_text(json.dumps(data, indent=2, default=str))
    print("\n[real_model] LFM2.5-350M tool-calling compatibility\n" + json.dumps(data, indent=2, default=str))


@pytest.fixture(scope="module")
def model():
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    from smol_harness.config import ModelSpec
    from smol_harness.models import build_model

    try:
        path = snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=FILES, local_files_only=True)
    except LocalEntryNotFoundError:
        print(f"\n[real_model] {MODEL_ID}@{REVISION[:8]} is not cached: downloading ~714 MB (explicit opt-in)")
        path = snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=FILES)

    def refuse(*a, **k):
        raise AssertionError("network access attempted during real-model load/inference")

    mp = pytest.MonkeyPatch()
    for name in ("connect", "connect_ex"):
        mp.setattr(socket.socket, name, refuse)
    mp.setattr(socket, "getaddrinfo", refuse)
    mp.setattr(socket, "create_connection", refuse)
    loaded = build_model(ModelSpec("hf", path, "none", dtype="fp32", device="cpu"))
    try:
        yield loaded
    finally:
        loaded.unload()
        mp.undo()


def test_effective_context_window(model, report):
    """effGen sizes its budget from config.max_position_embeddings (128000); the card says 32,768."""
    from effgen.core.thread_budget import resolve_context_budget

    config = model.model.config
    tokenizer = model.tokenizer
    default = resolve_context_budget("auto", model=model)
    capped = resolve_context_budget("auto", model=model, window_override=CARD_WINDOW)

    assert config.max_position_embeddings == 128000 and model.get_context_length() == 128000
    assert default.window_tokens == 128000 and default.budget_tokens > CARD_WINDOW  # would allow ~100K tokens
    assert capped.window_tokens == CARD_WINDOW and capped.budget_tokens < CARD_WINDOW  # the override wins
    report["context_window"] = {
        "config_max_position_embeddings": config.max_position_embeddings,
        "config_rope_parameters": getattr(config, "rope_parameters", None),
        "tokenizer_model_max_length": tokenizer.model_max_length,
        "model_card_context_length": CARD_WINDOW,
        "effgen_get_context_length": model.get_context_length(),
        "effgen_auto_budget_default": {"window": default.window_tokens, "prompt_budget_tokens": default.budget_tokens,
                                      "reserve": default.reserve_tokens},
        "effgen_auto_budget_with_max_context_length_32768": {"window": capped.window_tokens,
                                                              "prompt_budget_tokens": capped.budget_tokens,
                                                              "reserve": capped.reserve_tokens},
        "note": "No input beyond a few hundred tokens was run; 32,768 is the card's figure, not measured here.",
    }


def test_tool_calling_modes(model, report):
    params = list(model.model.parameters())
    cuda_before = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved(), torch.cuda.is_initialized())

    runs = [probe_case(model, **case) for case in CASES]

    # --- the experiment itself is sound -------------------------------------------------------------------
    assert model.device == "cpu" and {p.device.type for p in params} == {"cpu"}
    assert (torch.cuda.memory_allocated(), torch.cuda.memory_reserved(), torch.cuda.is_initialized()) == cuda_before
    for run in runs:
        assert run["strategy_used"] == run["mode"], run  # the requested mode really engaged
        assert run["tool_calls"] == len(run["tool_arguments"])  # effGen's record matches the real executions
        assert run["model_calls"] >= 1 and all(isinstance(t, str) for t in run["raw_outputs"])
        if run["tool_success"]:  # success is only ever claimed with an executed tool and a returned result
            assert run["tool_executed"] and run["tool_result_returned"] and run["final_answer_success"]
        assert max(run["prompt_chars_per_call"]) < 20_000  # tiny prompts, far below any context limit

    by_mode: dict[str, list[dict]] = {}
    for run in runs:
        key = run["mode"] if not run["system_prompt"] else "native+json-system-prompt"
        by_mode.setdefault(key, []).append(run)
    report["tool_calling"] = {
        "summary": {key: {"runs": len(rs), "selected_tool": sum(r["selected_tool"] for r in rs),
                          "parsed_by_effgen": sum(r["parsed_by_effgen"] for r in rs),
                          "tool_executed": sum(r["tool_executed"] for r in rs),
                          "result_returned": sum(r["tool_result_returned"] for r in rs),
                          "final_answer_success": sum(r["final_answer_success"] for r in rs),
                          "tool_success": sum(r["tool_success"] for r in rs)} for key, rs in by_mode.items()},
        "runs": runs,
        "settings": {"model": f"{MODEL_ID}@{REVISION}", "device": "cpu", "dtype": "fp32", "max_tokens": 64,
                     "max_iterations": 3, "max_context_length": CARD_WINDOW, "temperature": 0.1, "top_k": 50,
                     "repetition_penalty": 1.05},
    }

    # --- the recommended mode must work (ReAct); native is only recorded --------------------------------------
    react = by_mode["react"]
    assert any(r["tool_success"] for r in react), [(r["raw_outputs"], r["termination"], r["stop_reason"]) for r in react]
