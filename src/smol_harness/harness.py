"""Wires one orchestrator `Agent` to a `WorkerRegistry` through `DelegateTool`.

Inputs are `AgentConfig`s that already hold loaded model *instances*; YAML and
provider/quantization mapping come later (T3.x) and just produce these configs.

Also holds `install_effgen_loop_compat`, the one deliberate, process-wide patch of effGen 1.3.0 (review F1).
"""

from __future__ import annotations

import dataclasses
import inspect
import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from effgen import Agent, AgentConfig

from smol_harness.delegation import (
    DELEGATE_TOOL_NAME,
    STATEFUL_TOOL_MARKER,
    DelegateTool,
    WorkerRecord,
    WorkerRegistry,
    validate_limits,
)

# ---- effGen loop compatibility (docs/reviews/opus-review-001.md F1; audit Part 3, "F1 compatibility patch") ----------
#
# effGen's `NativeToolLoop` assumes tools are pure: an exact repeat of an earlier successful call `(tool, input)` is
# answered from a record WITHOUT running the tool, and an exact repeat of a failed call is declined. For `shell`,
# `file_operations` and `delegate` that returns stale results (the code changed between the two calls) or never
# re-runs a retry. effGen offers no switch, and the guard object is built at ONE site, `agent_loop.drive`, from the
# module-level name `NativeToolLoop`, so the only lever is rebinding that name.
#
# THIS IS A PROCESS-WIDE MUTATION. It is explicit (nothing calls it implicitly), idempotent, and never undone by the
# harness. It is inert for every tool that does not carry STATEFUL_TOOL_MARKER, whoever runs it. Installing or
# restoring it while any agent is running, or from several threads at once, is unsupported.
SUPPORTED_EFFGEN_VERSION = "1.3.0"
_CHECK_ACTION_PARAMETERS = ["self", "action", "action_input"]

_compat_lock = threading.Lock()  # guards the two names below, and nothing else
_compat_class: type | None = None  # our NativeToolLoop subclass while installed
_compat_original: type | None = None  # the effGen class it replaced


class EffgenCompatError(RuntimeError):
    """The effGen loop patch cannot be installed, or is required and missing. Nothing was patched."""


def _incompatible(problems: list[str]) -> str:
    return ("smol-harness cannot patch effGen's tool-loop guards: " + "; ".join(problems) + ". Nothing was patched, and "
            "running without the patch would replay stale results for shell/file_operations/delegate. Review "
            "docs/effgen-api-audit.md (F1 compatibility patch) and tests/test_loop_guards.py before changing the pin.")


def _verified_effgen_internals() -> tuple[Any, Any]:
    """`(agent_loop, agent_tool_loop)` after checking every private name the patch relies on, or raise."""
    import effgen

    problems: list[str] = []
    if getattr(effgen, "__version__", None) != SUPPORTED_EFFGEN_VERSION:
        problems.append(f"effgen is {getattr(effgen, '__version__', 'unknown')!r}, the patch was written for "
                        f"{SUPPORTED_EFFGEN_VERSION!r}")
    try:
        from effgen.core import agent_loop, agent_tool_loop
    except ImportError as exc:
        raise EffgenCompatError(_incompatible(problems + [f"cannot import effgen.core.agent_loop / agent_tool_loop: {exc}"])) from exc
    upstream = getattr(agent_tool_loop, "NativeToolLoop", None)
    if not isinstance(upstream, type):
        problems.append("effgen.core.agent_tool_loop.NativeToolLoop is missing")
    elif not callable(getattr(upstream, "check_action", None)):
        problems.append("NativeToolLoop.check_action is missing")
    elif list(inspect.signature(upstream.check_action).parameters) != _CHECK_ACTION_PARAMETERS:
        problems.append(f"NativeToolLoop.check_action parameters are {list(inspect.signature(upstream.check_action).parameters)}, "
                        f"expected {_CHECK_ACTION_PARAMETERS}")
    loop_check = getattr(agent_tool_loop, "LoopCheck", None)
    if not (loop_check is not None and dataclasses.is_dataclass(loop_check)
            and "is_exact_loop" in {f.name for f in dataclasses.fields(loop_check)}):
        problems.append("effgen.core.agent_tool_loop.LoopCheck is not a dataclass with an `is_exact_loop` field")
    if not hasattr(agent_loop, "NativeToolLoop"):
        problems.append("effgen.core.agent_loop has no module-level NativeToolLoop name to replace")
    try:
        constructs_here = "NativeToolLoop(" in inspect.getsource(agent_loop.drive)
    except (AttributeError, OSError, TypeError):
        constructs_here = False
    if not constructs_here:
        problems.append("effgen.core.agent_loop.drive no longer builds the guards with NativeToolLoop(...)")
    if problems:
        raise EffgenCompatError(_incompatible(problems))
    return agent_loop, agent_tool_loop


def install_effgen_loop_compat() -> None:
    """Make identical repeat calls of marked tools (`shell`, `file_operations`, `delegate`) actually run.

    Call it once, explicitly, before any agent runs (a CLI or application start-up, or a test fixture). Idempotent.
    Rebinds `effgen.core.agent_loop.NativeToolLoop` to a subclass whose `check_action` clears `is_exact_loop` only for
    tools carrying STATEFUL_TOOL_MARKER. Every other guard (repeated result, stall, drift, iteration cap, delegation
    limit, worker deadline) is inherited unchanged, and so is all behaviour for unmarked tools.

    PROCESS-WIDE: it changes effGen for every agent in this process, it is never undone by the harness (not by
    `Harness.run`, nor by closing anything), and installing or restoring it while an agent runs, or from several
    threads at once, is unsupported. Raises `EffgenCompatError`, patching nothing, if this effGen is not the one it was
    verified against, if any private name it needs changed, or if something else already replaced the class.
    """
    global _compat_class, _compat_original
    with _compat_lock:
        agent_loop, agent_tool_loop = _verified_effgen_internals()
        current = agent_loop.NativeToolLoop
        if _compat_class is not None and current is _compat_class:
            return
        if current is not agent_tool_loop.NativeToolLoop:
            raise EffgenCompatError(f"effgen.core.agent_loop.NativeToolLoop is already {current!r}, not effGen's own class; "
                                    "refusing to wrap an unknown replacement. Nothing was patched.")
        upstream = current

        class MarkerKeyedToolLoop(upstream):  # type: ignore[valid-type, misc]
            def check_action(self, action, action_input):
                check = super().check_action(action, action_input)
                if check.is_exact_loop and getattr(self.tools.get(action), STATEFUL_TOOL_MARKER, False):
                    return dataclasses.replace(check, is_exact_loop=False)
                return check

        agent_loop.NativeToolLoop = MarkerKeyedToolLoop
        _compat_class, _compat_original = MarkerKeyedToolLoop, upstream


def effgen_loop_compat_installed() -> bool:
    """Whether effGen's loop currently uses the patched guards."""
    try:
        from effgen.core import agent_loop
    except ImportError:
        return False
    return _compat_class is not None and getattr(agent_loop, "NativeToolLoop", None) is _compat_class


def uninstall_effgen_loop_compat_for_tests() -> None:
    """Restore effGen's own `NativeToolLoop`. Tests only, and only while no agent is running. The harness never calls it."""
    global _compat_class, _compat_original
    with _compat_lock:
        if _compat_class is None:
            return
        from effgen.core import agent_loop

        if agent_loop.NativeToolLoop is not _compat_class:
            raise EffgenCompatError(f"effgen.core.agent_loop.NativeToolLoop is {agent_loop.NativeToolLoop!r}, not the patched "
                                    "class; refusing to overwrite it.")
        agent_loop.NativeToolLoop = _compat_original
        _compat_class = _compat_original = None


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
        max_delegations: int | None = None,
        max_report_chars: int | None = None,
    ) -> None:
        validate_limits(max_delegations, max_report_chars)  # fail at build time, not mid-run
        self.max_delegations = max_delegations
        self.max_report_chars = max_report_chars
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
        delegate = DelegateTool(
            self.registry, records, self.worker_timeout_s,
            max_delegations=self.max_delegations, max_report_chars=self.max_report_chars,
        )
        config = replace(self._orchestrator, tools=[*self._orchestrator.tools, delegate], enable_sub_agents=False)
        if any(getattr(tool, STATEFUL_TOOL_MARKER, False) for tool in config.tools) and not effgen_loop_compat_installed():
            # `delegate` is always marked, so this always applies: without the patch effGen would silently replay a
            # recorded result instead of re-running an identical delegate/shell/file call (review F1).
            raise EffgenCompatError(
                "effGen's loop guards have not been patched for stateful tools: call "
                "smol_harness.harness.install_effgen_loop_compat() once, before any agent runs (it is process-wide and "
                "explicit by design). Refusing to run, because identical repeat calls would return stale recorded results.")
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
    max_delegations: int | None = None,
    max_report_chars: int | None = None,
) -> Harness:
    """Register `workers` (name -> config) and return a `Harness` for `orchestrator`.

    `worker_timeout_s` is a cooperative per-delegation deadline; `max_delegations` caps
    delegation attempts per `run`; `max_report_chars` caps each report's serialized size.
    None = no limit for all three.
    """
    registry = WorkerRegistry()
    for name, config in workers.items():
        registry.register(name, config, (descriptions or {}).get(name, ""))
    return Harness(orchestrator, registry, worker_timeout_s, max_delegations, max_report_chars)
