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

from effgen import Agent, AgentConfig, RunStoppedError
from effgen.models.base import BaseModel
from effgen.tools.base_tool import BaseTool

from smol_harness.delegation import (
    DELEGATE_TOOL_NAME,
    STATEFUL_TOOL_MARKER,
    DelegateTool,
    WorkerRecord,
    WorkerRegistry,
    validate_limits,
)
from smol_harness.config import AgentSpec, HarnessConfig
from smol_harness.models import ModelCache
from smol_harness.recorder import ExecutionRecorder
from smol_harness.tools import build_tools

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


def initialize_runtime() -> None:
    """The one call an application makes, once, before it builds or runs any harness.

    Today that is installing the effGen loop compatibility patch (`install_effgen_loop_compat`, see its docstring: it is
    explicit, idempotent and PROCESS-WIDE). The CLI calls this function first; so must any other entry point:

        initialize_runtime()
        config = load_config("harness.yaml")
        with build_harness_from_config(config) as harness:
            result = harness.run(task)

    Nothing else installs the patch: not `Harness.__init__`, `Harness.run`, `build_harness_from_config` or `Harness.close`,
    which also never uninstalls it. Raises `EffgenCompatError` (patching nothing) if this effGen is not the verified one.
    """
    install_effgen_loop_compat()


def _require_loop_compat() -> None:
    if not effgen_loop_compat_installed():
        # Without the patch effGen would silently replay a recorded result instead of re-running an identical
        # delegate/shell/file call (review F1); `delegate` is always marked, so every harness needs it.
        raise EffgenCompatError(
            "effGen's loop guards have not been patched for stateful tools: call "
            "smol_harness.harness.initialize_runtime() (or install_effgen_loop_compat()) once, before any agent runs "
            "(it is process-wide and explicit by design). Refusing to run, because identical repeat calls would return "
            "stale recorded results.")


@dataclass
class HarnessResult:
    """Outcome of one `Harness.run`: the parent's answer plus this run's delegations."""

    output: str
    success: bool
    termination: str | None
    response: Any  # effgen AgentResponse of the orchestrator
    workers: list[WorkerRecord]


class HarnessClosedError(RuntimeError):
    """`Harness.run` was called after `Harness.close`."""


class Harness:
    """Runs tasks on one orchestrator and its registered workers.

    `owned_cache` is a `ModelCache` this harness created and therefore closes (unloading its models, each once);
    `build_harness_from_config` sets it when it made the cache itself. A harness never closes a cache it was merely
    given, and `close()` never touches the effGen loop patch.
    """

    def __init__(
        self,
        orchestrator: AgentConfig,
        registry: WorkerRegistry,
        worker_timeout_s: float | None = None,
        max_delegations: int | None = None,
        max_report_chars: int | None = None,
        owned_cache: ModelCache | None = None,
        recorder: ExecutionRecorder | None = None,
    ) -> None:
        validate_limits(max_delegations, max_report_chars)  # fail at build time, not mid-run
        self._owned_cache = owned_cache
        self.recorder = recorder  # optional execution history; `build_harness` also adds it as middleware to every agent config
        self._closed = False
        self.max_delegations = max_delegations
        self.max_report_chars = max_report_chars
        if any(t.name == DELEGATE_TOOL_NAME for t in orchestrator.tools):
            raise ValueError(f"orchestrator config must not already hold {DELEGATE_TOOL_NAME!r}")
        self._orchestrator = orchestrator
        self.registry = registry
        self.worker_timeout_s = worker_timeout_s  # cooperative; see DeadlineMiddleware

    def close(self) -> None:
        """Release what this harness owns: the models of a cache it created. Idempotent; safe to call twice.

        An injected cache is the caller's and is left open. The harness is marked closed first, so a failing unload
        (`ExceptionGroup`, raised here once) cannot leave it half-open or make a second `close()` retry.
        """
        if self._closed:
            return
        self._closed = True
        if self._owned_cache is not None:
            self._owned_cache.close()

    def __enter__(self) -> Harness:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> None:
        try:
            self.close()
        except Exception as cleanup:
            if exc is None:
                raise
            exc.add_note(f"smol-harness: closing the harness also failed: {cleanup!r}")  # the original stays the visible error

    def run(self, task: str) -> HarnessResult:
        """Run one task. Builds a fresh orchestrator `Agent` and record list per call.

        A reused effGen `Agent` keeps conversation memory between `run()` calls
        (audit §1.4), so independent runs must not share one.

        An orchestrator that effGen stops without an answer (iteration cap, repeat guards, a tool failing on its own
        side) comes back as a normal failed `HarnessResult` whatever `raise_on_error` says: under the default `True`
        effGen raises `RunStoppedError` carrying the very response `False` would have returned, and that response is
        used (review F3, audit §1.2). Any other exception has no response to build a result from (a provider error, a
        guardrail block, a bug), so it propagates unchanged, with this run's worker records on its `worker_records`
        attribute so that the work already done is not lost.
        """
        if self._closed:
            raise HarnessClosedError("this Harness is closed: its models may be unloaded. Build a new one (or keep this "
                                     "one open for as long as it is needed, e.g. inside its `with` block).")
        records: list[WorkerRecord] = []
        delegate = DelegateTool(
            self.registry, records, self.worker_timeout_s,
            max_delegations=self.max_delegations, max_report_chars=self.max_report_chars,
        )
        config = replace(self._orchestrator, tools=[*self._orchestrator.tools, delegate], enable_sub_agents=False)
        if any(getattr(tool, STATEFUL_TOOL_MARKER, False) for tool in config.tools):
            _require_loop_compat()  # `delegate` is always marked, so this always applies
        if self.recorder is not None:
            try:
                models = self._model_names()
            except Exception:  # recording-only metadata: the session is recorded without it, the run is not touched
                models = None
            self.recorder.begin(task, models)  # never raises: a recorder error only stops recording
        try:
            with Agent(config) as agent:
                try:
                    response = agent.run(task)
                except RunStoppedError as exc:
                    response = _stopped_response(exc)
                    if response is None:  # a RunStoppedError without a usable response: say so, do not invent a result
                        _keep_records(exc, records)
                        raise
                except Exception as exc:
                    _keep_records(exc, records)
                    raise
        except BaseException as exc:  # includes Ctrl-C: the history says the session was aborted, then the exception goes on unchanged
            if self.recorder is not None:
                self.recorder.abort(exc)
            raise
        result = HarnessResult(
            output=str(response.output),
            success=bool(response.success),
            termination=getattr(response, "termination", None),
            response=response,
            workers=records,
        )
        if self.recorder is not None:
            self.recorder.finish(result)
        return result

    def _model_names(self) -> dict[str, Any]:
        """Model names by agent, for the history's session record. Names only; provider and quantization are not known here."""
        name = lambda config: getattr(config.model, "model_name", None) or type(config.model).__name__  # noqa: E731
        return {"orchestrator": name(self._orchestrator),
                "workers": {worker: name(self.registry.get_config(worker)) for worker in self.registry.names()}}


def _stopped_response(exc: RunStoppedError) -> Any | None:
    """The failed `AgentResponse` a `RunStoppedError` carries, or None when it is missing or not that.

    effGen builds the error from the response (`errors.py:RunStoppedError.__init__`), so a normal one always has it;
    the checks guard a hand-made or subclassed error. A stopped run's response has `success=False` and an `output`.
    """
    response = getattr(exc, "response", None)
    if response is None or not hasattr(response, "output") or getattr(response, "success", True) is not False:
        return None
    return response


def _keep_records(exc: BaseException, records: list[WorkerRecord]) -> None:
    """Hang this run's worker records on an exception that is about to propagate (it carries no result to put them in)."""
    try:
        exc.worker_records = list(records)  # type: ignore[attr-defined]
    except (AttributeError, TypeError):  # an exception type that refuses attributes: the note below still says it
        pass
    if records:
        exc.add_note(f"smol-harness: {len(records)} worker delegation(s) ran before this error (exception.worker_records)")


def build_harness(
    orchestrator: AgentConfig,
    workers: Mapping[str, AgentConfig],
    descriptions: Mapping[str, str] | None = None,
    worker_timeout_s: float | None = None,
    max_delegations: int | None = None,
    max_report_chars: int | None = None,
    owned_cache: ModelCache | None = None,
    recorder: ExecutionRecorder | None = None,
) -> Harness:
    """Register `workers` (name -> config) and return a `Harness` for `orchestrator`.

    `worker_timeout_s` is a cooperative per-delegation deadline; `max_delegations` caps
    delegation attempts per `run`; `max_report_chars` caps each report's serialized size.
    None = no limit for all three. `owned_cache`: a cache the returned harness will close.
    `recorder`: optional execution history; it is appended to every agent's middleware (after any the config
    already has, so its `after_*` hooks see the final values) and nothing else about the agents changes.
    """
    if recorder is not None:
        orchestrator = replace(orchestrator, middleware=[*orchestrator.middleware, recorder])
        workers = {name: replace(config, middleware=[*config.middleware, recorder]) for name, config in workers.items()}
    registry = WorkerRegistry()
    for name, config in workers.items():
        registry.register(name, config, (descriptions or {}).get(name, ""))
    return Harness(orchestrator, registry, worker_timeout_s, max_delegations, max_report_chars, owned_cache, recorder)


TOOL_CALLING_MODES = ("auto", "native", "react", "hybrid")  # AgentConfig.tool_calling_mode (effGen core/agent_config.py)


def _agent_config(spec: AgentSpec, model: BaseModel, tools: list[BaseTool], tool_calling_mode: str) -> AgentConfig:
    """One `AgentConfig` with every setting that matters stated, so no effGen default decides it silently (review F5).

    Set from our config: name, model (the shared instance), tools (this agent's own instances), `max_iterations`, the
    system prompt (only when configured; otherwise effGen's own "You are a helpful AI assistant."), and for Ollama the
    agent-side `max_context_length`. Fixed by the harness: `tool_calling_mode`; `raise_on_error=False` (an effGen run
    failure becomes a failed response, which `Harness.run` and `DelegateTool` turn into structured results; effGen's
    default `True` raises instead); `enable_memory=False` (workers must not share state; no long-term store);
    `capability_probe=False` (no extra generation to probe the model); `enable_sub_agents=False` (delegation is only the
    orchestrator's `delegate` tool, depth 1). Not set, because our config has no field for them (review F7): `temperature`
    (effGen 0.7), `seed`, `max_tokens`, sampling parameters.

    `max_context_length` limits the *agent's* prompt budget. It cannot change the Ollama server's own context window
    (`num_ctx`), which the server or its Modelfile fixes (review F4); Hugging Face specs have no such field yet.
    """
    extra: dict[str, Any] = {}
    if spec.system_prompt is not None:
        extra["system_prompt"] = spec.system_prompt
    if spec.model.provider == "ollama":
        extra["max_context_length"] = spec.model.context_length
    return AgentConfig(
        name=spec.name, model=model, tools=tools, max_iterations=spec.max_iterations, tool_calling_mode=tool_calling_mode,
        raise_on_error=False, enable_memory=False, capability_probe=False, enable_sub_agents=False, **extra,
    )


def build_harness_from_config(
    config: HarnessConfig, *, cache: ModelCache | None = None, tool_calling_mode: str = "react",
    recorder: ExecutionRecorder | None = None,
) -> Harness:
    """A runnable `Harness` for a validated `HarnessConfig`.

    Call `initialize_runtime()` first. Order matters: (1) the effGen patch must be installed; (2) every agent's tools are
    built, orchestrator first, so a bad workspace, a dangerous root or an unavailable shell (`ToolBuildError`) fails
    before any model is loaded; (3) models come from one `ModelCache`, so agents with equal `ModelSpec`s share one
    instance (their contexts stay independent: each delegation is a fresh `Agent`); (4) the agents are configured
    (`_agent_config`) and the workers registered with the configured limits.

    `tool_calling_mode` applies to every agent assembled here (it is not a YAML field); ReAct is the only mode verified
    with the small local model (audit §1.12).

    Ownership: with no `cache`, the harness creates one and owns it (`Harness.close()` unloads its models, each once).
    A `cache` you pass stays yours; the harness never closes it, and models already loaded into it stay loaded if
    assembly fails. If assembly fails, a cache this call created is closed before the original exception propagates.

    `recorder` (optional, see `recorder.py`) records the runs of the returned harness; it never changes them.
    """
    if tool_calling_mode not in TOOL_CALLING_MODES:
        raise ValueError(f"tool_calling_mode must be one of {TOOL_CALLING_MODES}, got {tool_calling_mode!r}")
    _require_loop_compat()
    specs = [config.orchestrator, *config.workers]
    tools = [build_tools(spec.tools, config.workspace) for spec in specs]  # before any model is loaded
    owned = ModelCache() if cache is None else None
    models = owned if owned is not None else cache
    try:
        configs = [_agent_config(spec, models.get(spec.model), agent_tools, tool_calling_mode)
                   for spec, agent_tools in zip(specs, tools, strict=True)]
        return build_harness(
            configs[0], {spec.name: agent for spec, agent in zip(config.workers, configs[1:], strict=True)},
            {spec.name: spec.description for spec in config.workers},
            worker_timeout_s=config.limits.worker_timeout_s, max_delegations=config.limits.max_delegations,
            max_report_chars=config.limits.max_report_chars, owned_cache=owned, recorder=recorder,
        )
    except BaseException as exc:
        if owned is not None:
            try:
                owned.close()
            except Exception as cleanup:  # the assembly error stays the one that is raised
                exc.add_note(f"smol-harness: releasing the models loaded so far also failed: {cleanup!r}")
        raise
