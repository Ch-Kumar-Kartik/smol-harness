"""Evidence accounting for the real-model experiments (review 002, R2-03/R2-04/R2-06/R2-08).

Three different things were being called "the output":

* the tool's RAW result (a dict such as `{'success': True, 'data': ..., 'message': ...}`), which a model never sees;
* the text effGen puts in front of the model for that call (`model_visible`);
* the RAW text an engine generated, versus the text effGen KEPT after cutting at its stop sequences (`consumed_text`). Whatever follows the earliest stop
  sequence (typically a model-written `\\nObservation:` after a real `Action:`) never reaches the next prompt or the answer.

These helpers state each one explicitly. `experiments/rtx4070/stage_runner.py` carries stand-alone copies (that package must run without the tests
directory); `tests/test_evidence_accounting.py` pins that both agree.
"""

from __future__ import annotations

import hashlib
import time


def model_visible(result) -> str:
    """What effGen shows the model for a tool call (effgen/core/agent_tool_execution.py, 1.3.0).

    A failed call: its error text (effGen prefixes it, so the error is a substring of what is shown). A successful call: the dict's `result` field (shell,
    delegate) or, for a file operation, its `data` field. NOT the tool's raw output dict.
    """
    if not getattr(result, "success", False):
        return str(getattr(result, "error", "") or "")
    out = getattr(result, "output", None)
    if isinstance(out, dict):
        if out.get("result") is not None:
            return str(out["result"])
        if "data" in out and "success" in out:
            return str(out["data"]) if out["data"] is not None else str(out.get("message", ""))
    return str(out if out is not None else "")


def result_reached_model(entries: list[dict], prompts: list[str]) -> bool:
    """True only if EVERY executed call's model-visible text (first 40 characters) appears in some prompt. Nothing visible proves nothing."""
    if not entries:
        return False
    for entry in entries:
        key = entry.get("visible", "").strip()[:40]
        if not key or not any(key in prompt for prompt in prompts):
            return False
    return True


def logged(original, log: list[dict]):
    """A replacement for a tool class's `execute` that records every call that REALLY ran: its arguments, the raw result and the visible text."""

    async def execute(self, **kwargs):
        started = time.perf_counter()
        result = await original(self, **kwargs)
        raw = result.output.get("result") if isinstance(result.output, dict) and "result" in result.output else result.output
        log.append({"tool": self.name, "args": {k: (v if len(str(v)) < 300 else str(v)[:300] + "...") for k, v in kwargs.items()},
                    "success": result.success, "seconds": round(time.perf_counter() - started, 2), "error": (result.error or "")[:300] or None,
                    "raw_result": str(raw)[:400], "visible": model_visible(result)[:400]})
        return result

    return execute


def consumed_text(raw: str, stops) -> str:
    """The text effGen keeps of a generation: everything before the EARLIEST requested stop sequence."""
    cut = len(raw)
    for sequence in stops or ():
        index = raw.find(sequence) if sequence else -1
        if index != -1:
            cut = min(cut, index)
    return raw[:cut]


def call_record(provider: str, prompt, config, result, seconds: float) -> dict:
    """One generation: RAW engine text, the stop sequences effGen requested, the text it kept. `tokens` are what the engine generated, discarded text included."""
    stops = list(getattr(config, "stop_sequences", None) or [])
    raw = result.text
    kept = consumed_text(raw, stops)
    return {"provider": provider, "prompt": str(prompt), "raw_output": raw, "stop_sequences": stops, "consumed_output": kept, "discarded_chars": len(raw) - len(kept),
            "finish": result.finish_reason, "seconds": round(seconds, 2), "tokens": getattr(result, "tokens_used", None)}


def observation_accounting(calls: list[dict]) -> dict:
    """Self-written `Observation:` per generation, raw versus after effGen's cut. Only the second can reach an answer."""
    return {"generations": len(calls), "raw_with_observation": sum("Observation:" in c["raw_output"] for c in calls),
            "consumed_with_observation": sum("Observation:" in c["consumed_output"] for c in calls), "generated_tokens": sum(c["tokens"] or 0 for c in calls)}


def trajectory_id(calls: list[dict]) -> str:
    return hashlib.sha256("\x00".join(c["raw_output"] for c in calls).encode()).hexdigest()[:12]


def trajectories(attempts: list[dict]) -> dict:
    """Attempts at a low temperature can repeat each other exactly; k distinct of n are k samples, not n."""
    ids = [a.get("trajectory") for a in attempts]
    return {"attempts": len(ids), "distinct": len(set(ids))}


def digest(calls: list[dict], limit: int = 600) -> list[dict]:
    return [{"seconds": c["seconds"], "tokens": c["tokens"], "finish": c["finish"], "raw_output": c["raw_output"][:limit], "consumed_output": c["consumed_output"][:limit],
             "discarded_chars": c["discarded_chars"]} for c in calls]
