"""Measures whether a model really *calls a tool* under effGen's native or ReAct path.

Used by the opt-in real-model test (tests/integration/test_liquid_tools.py) and validated with
scripted models in the default suite (tests/test_tool_probe.py), so the measurement itself is tested.

A run only counts as tool-calling success (`tool_success`) when the tool really executed, its result
came back in a later prompt, and the agent finished successfully with that result in the answer. A
correct answer without an executed tool is never a success.
"""

from __future__ import annotations

import re
import time
from typing import Any

from effgen import Agent, AgentConfig, tool

# Syntaxes that mean "the model chose to call `add`" (prose such as "add(1,2) returns 3" does not).
_CALL_SELECTED = re.compile(
    r"<\|tool_call_start\|>\s*\[?\s*add\s*\(|Action:\s*add\b|\"name\"\s*:\s*\"add\"|<tool_call>|<function=add",
)


def probe_case(
    model: Any, mode: str, *, a: int = 4817, b: int = 9235, seed: int = 0, system_prompt: str | None = None,
    max_tokens: int = 64, max_iterations: int = 3, window: int = 32768, raise_on_error: bool = False,
    **sampling: Any,
) -> dict[str, Any]:
    """Run one short tool task and report what happened at each step. Never raises for model failures."""
    total = a + b
    executed: list[tuple[int, int]] = []

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers and return the sum."""
        executed.append((a, b))
        return a + b

    seen: list[dict[str, Any]] = []
    original = model.generate

    def spy(prompt, config=None, **kw):
        ran_before = len(executed)  # how many tool executions preceded this model call
        result = original(prompt, config, **kw)
        seen.append({"prompt": prompt if isinstance(prompt, str) else str(prompt), "tools_passed": bool(kw.get("tools")),
                     "raw_output": result.text, "finish_reason": result.finish_reason, "tool_ran_before": ran_before > 0})
        return result

    sampling = {"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.05, **sampling}
    config = AgentConfig(
        name=f"probe-{mode}", model=model, tools=[add], tool_calling_mode=mode, capability_probe=False,
        enable_sub_agents=False, enable_memory=False, max_iterations=max_iterations, raise_on_error=raise_on_error,
        max_tokens=max_tokens, max_context_length=window, seed=seed,
        **({"system_prompt": system_prompt} if system_prompt else {}), **sampling,
    )
    task = f"Use the add tool to calculate {a} + {b}. Then state the result."
    model.generate = spy
    started = time.perf_counter()
    try:
        with Agent(config) as agent:
            strategy = agent._tool_calling_strategy.name
            response = agent.run(task)
    finally:
        del model.generate  # restore the class method
    seconds = time.perf_counter() - started

    raw = [s["raw_output"] for s in seen]
    dispatched = len(response.tool_calls)  # effGen's own record of calls it parsed and ran
    # The result "came back" only if a prompt sent AFTER the tool ran contains the sum.
    result_returned = bool(executed) and any(s["tool_ran_before"] and str(total) in s["prompt"] for s in seen)
    finished = bool(response.success) and response.termination == "done"
    return {
        "mode": mode, "strategy_used": strategy, "seed": seed, "system_prompt": system_prompt,
        "task": task, "expected": total,
        "selected_tool": any(_CALL_SELECTED.search(t) for t in raw),   # the model chose to call it
        "parsed_by_effgen": dispatched > 0,                            # effGen turned the text into a dispatch
        "tool_executed": bool(executed),                               # our counting tool really ran
        "tool_arguments": executed,
        "tool_result_returned": result_returned,                       # the result was in a later prompt
        "final_answer_success": finished,
        "answer_contains_expected": str(total) in str(response.output),
        "tool_success": bool(executed) and dispatched == len(executed) and result_returned and finished
                        and str(total) in str(response.output),
        "model_calls": len(seen), "tool_calls": dispatched, "iterations": response.iterations,
        "termination": response.termination, "stop_reason": response.stop_reason,
        "final_output": str(response.output), "raw_outputs": raw,
        "seconds": round(seconds, 2),
        "prompt_chars_per_call": [len(s["prompt"]) for s in seen],
        "tools_passed_to_model": [s["tools_passed"] for s in seen],
    }
