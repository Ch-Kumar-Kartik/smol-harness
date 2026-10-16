"""Wires one orchestrator `Agent` to a `WorkerRegistry` through `DelegateTool`.

Inputs are `AgentConfig`s that already hold loaded model *instances*; YAML and
provider/quantization mapping come later (T3.x) and just produce these configs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from effgen import Agent, AgentConfig

from smol_harness.delegation import (
    DELEGATE_TOOL_NAME,
    DelegateTool,
    WorkerRecord,
    WorkerRegistry,
)


@dataclass
class HarnessResult:
    """Outcome of one `Harness.run`: the parent's answer plus this run's delegations."""

    output: str
    success: bool
    termination: str | None
    response: Any  # effgen AgentResponse of the orchestrator
    workers: list[WorkerRecord]


class Harness:
    def __init__(
        self,
        orchestrator: AgentConfig,
        registry: WorkerRegistry,
        worker_timeout_s: float | None = None,
    ) -> None:
        if any(t.name == DELEGATE_TOOL_NAME for t in orchestrator.tools):
            raise ValueError(f"orchestrator config must not already hold {DELEGATE_TOOL_NAME!r}")
        self._orchestrator = orchestrator
        self.registry = registry
        self.worker_timeout_s = worker_timeout_s  # cooperative; see DeadlineMiddleware

    def run(self, task: str) -> HarnessResult:
        """Run one task. Builds a fresh orchestrator `Agent` and record list per call.

        A reused effGen `Agent` keeps conversation memory between `run()` calls
        (audit §1.4), so independent runs must not share one.
        """
        records: list[WorkerRecord] = []
        delegate = DelegateTool(self.registry, records, self.worker_timeout_s)
        config = replace(self._orchestrator, tools=[*self._orchestrator.tools, delegate], enable_sub_agents=False)
        with Agent(config) as agent:
            response = agent.run(task)
        return HarnessResult(
            output=str(response.output),
            success=bool(response.success),
            termination=getattr(response, "termination", None),
            response=response,
            workers=records,
        )


def build_harness(
    orchestrator: AgentConfig,
    workers: Mapping[str, AgentConfig],
    descriptions: Mapping[str, str] | None = None,
    worker_timeout_s: float | None = None,
) -> Harness:
    """Register `workers` (name -> config) and return a `Harness` for `orchestrator`.

    `worker_timeout_s` is a cooperative per-delegation deadline (None = none).
    """
    registry = WorkerRegistry()
    for name, config in workers.items():
        registry.register(name, config, (descriptions or {}).get(name, ""))
    return Harness(orchestrator, registry, worker_timeout_s)
