"""Orchestrator -> worker delegation as an effGen tool (docs/architecture.md §5-7).

The orchestrator is a normal `effgen.Agent`. Delegation is a tool it calls: the
tool builds a *fresh* worker `Agent`, runs it once, and hands back a small JSON
report. No reasoning loop lives here; effGen runs both agents.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from functools import partial
from typing import Any

from effgen import Agent, AgentConfig, AgentMiddleware
from effgen.models.base import BaseModel
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

DELEGATE_TOOL_NAME = "delegate"

# smolagents-style envelope (architecture §1, §5): tells the worker who asked and
# what shape of report is useful. Kept short for small models.
TASK_TEMPLATE = """You are the worker agent '{worker}'. Your manager delegated this task to you.
Work only from the task and context below; you cannot see the manager's conversation.

Task:
{task}
{context_block}
Finish with a short report: the outcome, the key details, any files you changed, and anything unresolved."""


class UnknownWorkerError(KeyError):
    """No worker is registered under this name."""

    def __str__(self) -> str:  # KeyError would repr() the message
        return str(self.args[0])


class WorkerTimeout(Exception):
    """A worker's deadline passed. Raised from effGen middleware hooks."""


class DeadlineMiddleware(AgentMiddleware):
    """Cooperative per-worker deadline (architecture §10, audit §1.10).

    COOPERATIVE, NOT A HARD WALL-CLOCK LIMIT. The deadline is only checked when
    effGen is about to start a model call or a tool call. A call already in
    flight (a long local generation, a slow tool) is never interrupted, so a
    worker can run past its deadline by up to one call before the next check
    stops it. Process/thread cancellation is deliberately not attempted.

    One instance per worker execution; `expired` becomes True when it raises,
    which is how callers recognise a timeout regardless of how effGen reports the
    exception (raised as-is when `raise_on_error=True`, folded into a failed
    response otherwise; audit §1.8, U4).
    """

    def __init__(self, timeout_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s!r}")
        self.timeout_s = timeout_s
        self._clock = clock
        self._deadline = clock() + timeout_s  # monotonic: immune to wall-clock changes
        self.expired = False

    def _check(self, before: str) -> None:
        if self._clock() >= self._deadline:
            self.expired = True
            raise WorkerTimeout(f"worker exceeded its {self.timeout_s:g}s deadline (checked before {before})")

    def before_model_call(self, ctx: Any) -> None:
        self._check("a model call")

    def before_tool_call(self, ctx: Any) -> None:
        self._check(f"tool {ctx.tool_name!r}")


@dataclass
class WorkerResult:
    """What the parent sees about one delegation. JSON-serializable by construction.

    `status` is "completed", "failed" or "timeout". Failure never raises into the
    parent loop: it is a normal result with `error` set (architecture §5; U3).
    """

    worker: str
    status: str
    output: str = ""
    error: str | None = None
    error_type: str | None = None
    termination: str | None = None
    iterations: int = 0
    tool_calls: int = 0
    tokens_used: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def _failed(worker: str, error_type: str, error: str) -> WorkerResult:
    return WorkerResult(worker=worker, status="failed", error=error, error_type=error_type)


def _timed_out(worker: str, deadline: DeadlineMiddleware) -> WorkerResult:
    return WorkerResult(
        worker=worker, status="timeout", error_type="WorkerTimeout",
        error=f"Worker exceeded its {deadline.timeout_s:g}s deadline and was stopped. "
              "No output was produced; retry with a smaller task or report the timeout.",
    )


@dataclass
class WorkerRecord:
    """One delegation as the harness saw it: what was asked and what came back.

    A plain per-run record (no state machine): created when the delegation ends,
    in `completed` or `failed` state. Holds the report only, never the worker's
    conversation.
    """

    worker: str
    task: str
    context: str | None
    result: WorkerResult

    @property
    def status(self) -> str:
        return self.result.status


class WorkerRegistry:
    """Named worker definitions; builds a fresh `Agent` on every `create`.

    A definition is an `AgentConfig` whose `model` is an already-loaded
    `BaseModel` instance: every worker built from it shares that instance (no
    weights reloaded per delegation) but gets its own `Agent`, so its
    short-term memory and run state are its own. The registry holds only
    definitions, never agents or conversation state.
    """

    def __init__(self) -> None:
        self._configs: dict[str, AgentConfig] = {}
        self._descriptions: dict[str, str] = {}

    def register(self, name: str, config: AgentConfig, description: str = "") -> None:
        if not name or not name.strip():
            raise ValueError("worker name must be a non-empty string")
        if name in self._configs:
            raise ValueError(f"worker {name!r} is already registered")
        if not isinstance(config.model, BaseModel):
            raise TypeError(
                f"worker {name!r}: config.model must be a loaded BaseModel instance so "
                "delegations share it; a model-id string would reload weights each time"
            )
        if any(t.name == DELEGATE_TOOL_NAME for t in config.tools):
            raise ValueError(f"worker {name!r} must not hold the {DELEGATE_TOOL_NAME!r} tool (depth 1)")
        self._configs[name] = config
        self._descriptions[name] = description

    def names(self) -> list[str]:
        return list(self._configs)

    def description(self, name: str) -> str:
        self.get_config(name)
        return self._descriptions[name]

    def get_config(self, name: str) -> AgentConfig:
        try:
            return self._configs[name]
        except KeyError:
            raise UnknownWorkerError(
                f"Unknown worker {name!r}. Available workers: {', '.join(self._configs) or 'none'}."
            ) from None

    def create(self, name: str) -> Agent:
        """A new `Agent` for worker `name`. Raises `UnknownWorkerError`."""
        config = self.get_config(name)
        # Shallow copy per agent: Agent.__init__ may rewrite config.system_prompt,
        # and the tools list must not be shared mutable state. The model instance
        # (and tool instances) are intentionally shared.
        # enable_sub_agents=False: workers never spawn (depth 1, architecture §6).
        return Agent(replace(config, tools=list(config.tools), enable_sub_agents=False))


class DelegateTool(BaseTool):
    """effGen tool: run a task on a named worker, return a `WorkerResult` as JSON.

    Returns a JSON *string*, never a dict: effGen reduces dict outputs to a few
    keys and treats `success: False` as a tool failure (audit §1.2). Worker
    failures are therefore successful tool calls whose payload says
    `status: "failed"`.
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        records: list[WorkerRecord] | None = None,
        worker_timeout_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if worker_timeout_s is not None and worker_timeout_s <= 0:
            raise ValueError(f"worker_timeout_s must be > 0 or None, got {worker_timeout_s!r}")
        self._registry = registry
        # None = no deadline. Otherwise a fresh DeadlineMiddleware per delegation.
        self._timeout_s = worker_timeout_s
        self._clock = clock
        # Per-run sink owned by the caller (Harness.run passes a fresh list), so
        # records never outlive or cross runs. Not global state.
        self.records: list[WorkerRecord] = records if records is not None else []
        workers = "; ".join(
            f"{n}: {registry.description(n)}" if registry.description(n) else n
            for n in registry.names()
        )
        super().__init__(metadata=ToolMetadata(
            name=DELEGATE_TOOL_NAME,
            description=(
                "Delegate a self-contained task to a worker agent and get back its JSON report "
                f"(status, output, error). Available workers - {workers or 'none'}. "
                "The worker cannot see your conversation, so put everything it needs in task/context."
            ),
            category=ToolCategory.COMMUNICATION,
            parameters=[
                ParameterSpec(name="worker", type=ParameterType.STRING, required=True, min_length=1,
                              description="Name of the worker to delegate to."),
                ParameterSpec(name="task", type=ParameterType.STRING, required=True, min_length=1,
                              description="Self-contained description of the task."),
                ParameterSpec(name="context", type=ParameterType.STRING, required=False,
                              description="Optional extra context for the worker, e.g. file paths or findings."),
            ],
        ))

    async def _execute(self, worker: str, task: str, context: str | None = None, **kwargs: Any) -> str:
        result = await self._delegate(worker, task, context)
        self.records.append(WorkerRecord(worker=worker, task=task, context=context, result=result))
        return result.to_json()

    async def _delegate(self, worker: str, task: str, context: str | None) -> WorkerResult:
        # Unknown worker / blank task are recoverable model mistakes: structured
        # failure for the parent to act on, not a tool failure. Missing or
        # non-string arguments never get here (effGen validates them first).
        if not task.strip():
            return _failed(worker, "invalid_arguments", "task must not be blank.")
        try:
            agent = self._registry.create(worker)
        except UnknownWorkerError as exc:
            return _failed(worker, "unknown_worker", str(exc))
        except Exception as exc:  # e.g. the worker's Agent failed to construct
            return _failed(worker, type(exc).__name__, str(exc))

        prompt = TASK_TEMPLATE.format(
            worker=worker, task=task.strip(),
            context_block=f"\nContext:\n{context.strip()}\n" if context and context.strip() else "",
        )
        # Fresh deadline per delegation, passed as per-call middleware so it never
        # outlives or leaks into another execution.
        deadline = DeadlineMiddleware(self._timeout_s, self._clock) if self._timeout_s is not None else None
        # Agent.run is synchronous: run it off the event loop in a copy of the
        # caller's context, as effGen's own FunctionTool does for nested agents.
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        try:
            response = await loop.run_in_executor(None, partial(ctx.run, self._run_worker, agent, prompt, deadline))
        except Exception as exc:  # RunStoppedError, backend errors, ... all become a report
            if deadline is not None and deadline.expired:
                return _timed_out(worker, deadline)
            return _failed(worker, type(exc).__name__, str(exc))
        # With raise_on_error=False effGen folds the hook's exception into a failed
        # response instead of raising it, so the flag is the reliable signal.
        if deadline is not None and deadline.expired:
            return _timed_out(worker, deadline)

        ok = bool(response.success)
        output = str(response.output)
        return WorkerResult(
            worker=worker,
            status="completed" if ok else "failed",
            output=output if ok else "",
            error=None if ok else output,
            error_type=None if ok else "worker_run_failed",
            termination=getattr(response, "termination", None),
            iterations=int(response.iterations or 0),
            tool_calls=int(response.tool_call_count),
            tokens_used=int(response.tokens_used or 0),
        )

    @staticmethod
    def _run_worker(agent: Agent, prompt: str, deadline: DeadlineMiddleware | None):
        with agent:  # close() after the run; the shared model is not unloaded (audit §1.4)
            return agent.run(prompt, middleware=[deadline]) if deadline else agent.run(prompt)
