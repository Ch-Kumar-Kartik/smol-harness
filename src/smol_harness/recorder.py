"""Optional execution recorder: what a normal `smol-harness run` did, as JSONL the existing history importer reads.

An effGen `AgentMiddleware` (hooks around each run, model call and tool call) plus three lifecycle calls the `Harness` makes
(`begin`, `finish`, `abort`). It writes the same files the experiment runner does, `trace.jsonl`, `attempts.jsonl` and, only on
opt-in, `prompts.local.jsonl`, so `experiments/history/history.py import <dir>` indexes them into SQLite unchanged in kind.

RECORDING NEVER CHANGES A RUN. Every hook returns exactly what effGen's middleware contract expects (None from the `before_*`
hooks, i.e. "continue"; the received response/result unchanged from the `after_*` hooks) and never edits a context. The first
recorder error (disk full, a permission, an unserialisable value) disables recording for the rest of the process and is kept
in `error`; the CLI reports it on stderr. Nothing the recorder does can fail a task or alter an exit code.

What middleware can NOT see (these fields are recorded as absent, never guessed): the text a model produced before effGen cut it
at a stop sequence (only the consumed text arrives here), retries inside one generation, the provider's final request payload,
whether a tool succeeded (only its returned string), what effGen finally showed the model, and calls a later middleware blocked
(the `DeadlineMiddleware` raises after this recorder's `before_*` hook ran: such a call has a start line and no completion).

Delegation attribution: a worker is a separate effGen agent run on another thread. The orchestrator's `before_tool_call` for
`delegate` publishes the call's identity in a `ContextVar`, which `DelegateTool` copies into the worker thread (verified; audit
Part 3), so every worker event carries `parent_tool_call_id`. The delegation id (`d1`...) is NOT predicted: it is recorded once
the report that holds it comes back. Per-run state lives on `ctx.run.metadata` (effGen never reads it), not in shared stacks.
The ContextVar does NOT reach the worker when `Harness.run` is called under a running event loop: effGen then bridges the tool
coroutine through a bare thread (`effgen/utils/async_bridge.py`). So only the FIRST run after `begin()` is the orchestrator; any
later run without a parent is `role: "unknown"` with no parent and no delegation id, and an `attribution_unknown` line says so.
Each run keeps the session (`attempt`) it started in, so a worker that outlives its session never writes into the next one.
"""

from __future__ import annotations

import contextvars
import datetime as dt
import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from effgen import AgentMiddleware
from effgen.core.middleware import RunContext

from smol_harness.delegation import DELEGATE_TOOL_NAME
from smol_harness.redaction import redact_record

RECORDER_VERSION = 1
STAGE = "cli"  # the `stage` of every session this recorder writes; the experiment runner uses A..E

PROMPT_CAPTURE_NOTE = ("the prompt object effGen passed to the model call (the adapter INPUT), as text. It is not the provider's final payload: "
                       "the adapter wraps it in chat messages, and a local engine may apply a chat template or add tool definitions.")
VISIBLE_UNAVAILABLE = "middleware sees the string the tool returned, not the text effGen finally shows the model"
RAW_UNAVAILABLE = "middleware sees the text effGen consumed, not the generation before a stop sequence cut it"

# Identity of the delegate call in progress, for the worker run it starts. Set in the orchestrator's before_tool_call, copied by
# DelegateTool into the worker thread's context. {"tool_call_id": "<invocation>:<seq>", "worker": <name or None>, "attempt": <session>}.
_PARENT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("smol_recorder_parent_delegate", default=None)
_STATE_KEY = "smol_recorder"  # key in RunContext.metadata, which "effGen never reads"


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def prompt_text(prompt: Any) -> str:
    return prompt if isinstance(prompt, str) else str(prompt)


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()  # the same rule as the experiment runner and history.py


def _create_private(path: Path) -> None:
    os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600))
    os.chmod(path, 0o600)  # the umask can only narrow the mode above; this makes it exact


def _append(path: Path, text: str) -> None:
    """One `write` on an O_APPEND descriptor: a line is never interleaved with another writer's, and survives a kill after this returns."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        data = text.encode("utf-8") + b"\n"
        while data:
            data = data[os.write(fd, data):]
    finally:
        os.close(fd)


def _json_args(tool_input: Any) -> Any:
    """The tool input as the model supplied it: parsed when it is JSON, otherwise the raw text."""
    if isinstance(tool_input, str):
        try:
            return json.loads(tool_input)
        except ValueError:
            return tool_input
    return tool_input


class ExecutionRecorder(AgentMiddleware):
    """Records one process's `Harness.run` calls under `<history_dir>/<run id>/` (directory 0700, files 0600)."""

    name = "execution-recorder"

    def __init__(self, history_dir: str | os.PathLike[str], *, capture_prompts: bool = False) -> None:
        self.capture_prompts = capture_prompts
        self.invocation = uuid.uuid4().hex
        self.run_id = f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{self.invocation[:8]}"
        root = Path(history_dir)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.dir = root / self.run_id
        self.dir.mkdir(mode=0o700)
        os.chmod(self.dir, 0o700)
        self.trace_path = self.dir / "trace.jsonl"
        self.attempts_path = self.dir / "attempts.jsonl"
        self.prompts_path = self.dir / "prompts.local.jsonl"  # *.local.jsonl: Git-ignored and never archived
        for path in (self.trace_path, self.attempts_path, *((self.prompts_path,) if capture_prompts else ())):
            _create_private(path)
        self.error: str | None = None  # the first recorder error; once set, nothing more is recorded
        self._lock = threading.RLock()  # orders seq assignment and file writes across threads, and guards per-run state
        self._seq = 0
        self._attempt_count = 0
        self._attempt: str | None = None
        self._orchestrator_due = False  # True from begin() until the session's first agent run, which is the orchestrator
        self._anonymous: dict[str, Any] = {}  # metadata stand-in for a hook that arrives without a RunContext
        self._emit("history_start", recorder_version=RECORDER_VERSION, capture_prompts=capture_prompts, pid=os.getpid(),
                   unavailable={"raw_output": RAW_UNAVAILABLE, "visible": VISIBLE_UNAVAILABLE,
                                "tool_success": "middleware has no structured success flag: recorded as null"})
        if capture_prompts:
            self._emit("prompt_capture_enabled", file=self.prompts_path.name, what=PROMPT_CAPTURE_NOTE)

    # ------------------------------------------------------------------------------------------------------ writing
    def _emit(self, event: str, *, prompt: str | None = None, attempt: str | None = None, **fields: Any) -> int:
        """Append one redacted trace line and return its `seq`. With `prompt`, the exact text goes to the private file FIRST, so a
        generation line in the trace implies that its prompt was written. Raises on any failure: callers go through `_safely`."""
        with self._lock:
            self._seq += 1
            seq = self._seq
            line = {"seq": seq, "t_utc": _utc(), "run_id": self.run_id, "stage": STAGE, "invocation": self.invocation,
                    "attempt": attempt if attempt is not None else self._attempt, "event": event, **fields}
            if prompt is not None:
                # Unredacted on purpose, so that the SHA-256 in the trace still verifies; private by mode, location and name.
                _append(self.prompts_path, json.dumps({"invocation": self.invocation, "run_id": self.run_id, "attempt": line["attempt"], "seq": seq,
                                                       "sha256": prompt_sha256(prompt), "chars": len(prompt), "text": prompt}, ensure_ascii=False))
            _append(self.trace_path, self._redacted_line(line))
            return seq

    @staticmethod
    def _redacted_line(record: dict[str, Any]) -> str:
        """The shared policy, as the experiment runner applies it: values first, then the serialised text (`redaction.redact_record`)."""
        return redact_record(record, ensure_ascii=False)

    def _safely(self, fn: Any, *args: Any) -> None:
        """Run a recording step; on the first error stop recording for good. KeyboardInterrupt and SystemExit are not recorder errors."""
        if self.error is not None:
            return
        try:
            with self._lock:
                fn(*args)
        except Exception as exc:  # noqa: BLE001 - recording must never fail a run
            self.error = f"{type(exc).__name__}: {exc}"

    # ----------------------------------------------------------------------------------------- lifecycle (called by Harness)
    def begin(self, task: str, models: dict[str, Any] | None = None) -> None:
        """A session (`Harness.run`) starts."""
        _PARENT.set(None)  # a delegate call that never reported must not make the next run's orchestrator look like a worker
        self._safely(self._begin, task, models)

    def _begin(self, task: str, models: dict[str, Any] | None) -> None:
        self._attempt_count += 1
        self._attempt = f"run{self._attempt_count}"
        self._orchestrator_due = True
        self._emit("attempt_start", task=task, seed=None, model=models)

    def finish(self, result: Any) -> None:
        """The session ended with a result (which may be a failed one)."""
        self._safely(self._finish, result)

    def _finish(self, result: Any) -> None:
        ok = bool(getattr(result, "success", False))
        termination = getattr(result, "termination", None)
        _append(self.attempts_path, self._redacted_line({
            "attempt_id": self._attempt, "invocation_id": self.invocation, "run_id": self.run_id, "stage": STAGE,
            "final_answer": str(getattr(result, "output", "")), "termination": termination,
            "verification": {"agent_completed": ok, "task_verified": None, "checks": [],
                             "note": "nothing verified the task independently: a successful run is the agent's own claim"},
        }))  # before the end line, so a reader that sees the end finds the record
        self._emit("attempt_end", agent_completed=ok, task_verified=None, termination=termination)

    def abort(self, exc: BaseException) -> None:
        """The session ended by an exception (including Ctrl-C)."""
        self._safely(self._abort, exc)

    def _abort(self, exc: BaseException) -> None:
        message = str(exc)[:500]
        self._emit("attempt_aborted", error=f"{type(exc).__name__}: {message}" if message else type(exc).__name__)

    # ------------------------------------------------------------------------------------------------------ run state
    def _state(self, ctx: Any) -> dict[str, Any]:
        """The per-run state, kept on the `RunContext.metadata` of the run the call belongs to (a model/tool context reaches it as `ctx.run`;
        the run hooks receive the RunContext itself)."""
        run = ctx if isinstance(ctx, RunContext) else getattr(ctx, "run", None)
        meta = run.metadata if run is not None else self._anonymous
        state = meta.get(_STATE_KEY)
        if state is None:
            state = meta[_STATE_KEY] = self._new_state("unknown", getattr(run, "agent_name", None), None, self._attempt)
        return state

    @staticmethod
    def _new_state(role: str, agent_name: str | None, parent: str | None, attempt: str | None) -> dict[str, Any]:
        return {"role": role, "agent_name": agent_name, "parent": parent, "attempt": attempt, "generations": 0, "last_gen_seq": None, "models": {}, "tools": {}}

    @staticmethod
    def _who(state: dict[str, Any]) -> dict[str, Any]:
        """Who wrote a line, and the session it belongs to (the run's own, snapshot when it started; `_emit` takes `attempt` from here)."""
        who = {"attempt": state["attempt"], "role": state["role"], "agent_name": state["agent_name"]}
        if state["parent"] is not None:
            who["parent_tool_call_id"] = state["parent"]
        return who

    # --------------------------------------------------------------------------------------------------- the run hooks
    def before_run(self, ctx: Any) -> None:
        self._safely(self._run_started, ctx)
        return None

    def _run_started(self, ctx: Any) -> None:
        parent = _PARENT.get()
        if parent:
            worker = parent.get("worker")
            state = self._new_state("worker", worker if isinstance(worker, str) else ctx.agent_name, parent["tool_call_id"], parent.get("attempt") or self._attempt)
        elif self._orchestrator_due:
            self._orchestrator_due = False
            state = self._new_state("orchestrator", ctx.agent_name, None, self._attempt)
        else:  # not the session's first run and no parent reached it: never guess orchestrator, never invent a parent
            state = self._new_state("unknown", ctx.agent_name, None, self._attempt)
        ctx.metadata[_STATE_KEY] = state
        task = ctx.task if isinstance(ctx.task, str) else str(ctx.task)
        # The task text itself is in attempt_start (orchestrator) or in the parent's delegate arguments (worker): here only its size and hash.
        start = self._emit("agent_run_start", **self._who(state), effgen_agent=ctx.agent_name, task_chars=len(task), task_sha256=prompt_sha256(task))
        if state["role"] == "unknown":
            self._emit("attribution_unknown", **self._who(state), agent_run_start_seq=start,
                       reason="an agent run that is not the session's first and that no delegate call's context reached (e.g. Harness.run under a running "
                              "event loop, where effGen runs the tool on a thread without the caller's context): role, parent and delegation id are unknown")

    def after_run(self, ctx: Any, response: Any) -> Any:
        self._safely(self._run_ended, ctx, response)
        return response

    def _run_ended(self, ctx: Any, response: Any) -> None:
        state = self._state(ctx)
        self._emit("agent_run_end", **self._who(state), success=getattr(response, "success", None), termination=getattr(response, "termination", None),
                   iterations=getattr(response, "iterations", None), tool_calls=getattr(response, "tool_call_count", None),
                   tokens_used=getattr(response, "tokens_used", None))

    # --------------------------------------------------------------------------------------------------- model calls
    def before_model_call(self, ctx: Any) -> None:
        self._safely(self._model_started, ctx)
        return None

    def _model_started(self, ctx: Any) -> None:
        state = self._state(ctx)
        state["generations"] += 1
        text = prompt_text(ctx.prompt)
        start = self._emit("generation_start", **self._who(state), generation_in_agent=state["generations"], model=ctx.model_name,
                           prompt_sha256=prompt_sha256(text), prompt_chars=len(text))
        state["models"][id(ctx)] = {"start_seq": start, "t0": time.monotonic(), "n": state["generations"]}

    def after_model_call(self, ctx: Any, result: Any) -> Any:
        self._safely(self._model_ended, ctx, result)
        return result

    def _model_ended(self, ctx: Any, result: Any) -> None:
        state = self._state(ctx)
        call = state["models"].pop(id(ctx), None)
        text = prompt_text(ctx.prompt)  # read now, not at the start: what the model was finally asked, after any other hook's rewrite
        get = result.get if isinstance(result, dict) else (lambda key, default=None: getattr(result, key, default))
        output = get("text")
        metadata = get("metadata")
        usage = {k: metadata[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd", "estimated_usage")
                 if isinstance(metadata, dict) and k in metadata} or None
        seq = self._emit(
            "generation", prompt=text if self.capture_prompts else None, **self._who(state),
            generation_in_agent=call["n"] if call else None, started_seq=call["start_seq"] if call else None, provider=None, model=ctx.model_name,
            prompt_sha256=prompt_sha256(text), prompt_chars=len(text), prompt_type=type(ctx.prompt).__name__,
            raw_output=None, raw_unavailable=RAW_UNAVAILABLE,
            consumed_output=output if isinstance(output, str) else None, consumed_status="as_seen_by_middleware",
            finish=get("finish_reason"), tokens=get("tokens_used"), provider_metadata=usage,
            seconds=round(time.monotonic() - call["t0"], 4) if call else None, attempt_in_call=ctx.attempt)
        state["last_gen_seq"] = seq

    # ---------------------------------------------------------------------------------------------------- tool calls
    def before_tool_call(self, ctx: Any) -> None:
        self._safely(self._tool_started, ctx)
        return None

    def _tool_started(self, ctx: Any) -> None:
        state = self._state(ctx)
        args = _json_args(ctx.tool_input)
        fields: dict[str, Any] = {}
        if state["last_gen_seq"] is not None:  # the generation that asked for this call (absent: the importer derives the link, or leaves it unset)
            fields["requested_by_generation_seq"] = state["last_gen_seq"]
        seq = self._emit("tool_call_start", **self._who(state), tool=ctx.tool_name, generation_in_agent=state["generations"] or None, args=args,
                         delegation_id=None, delegation_id_expected=None, **fields)  # never predicted: the id comes back in the report
        call: dict[str, Any] = {"seq": seq, "t0": time.monotonic(), "token": None}
        state["tools"][id(ctx)] = call
        if ctx.tool_name == DELEGATE_TOOL_NAME:
            worker = args.get("worker") if isinstance(args, dict) else None
            call["token"] = _PARENT.set({"tool_call_id": f"{self.invocation}:{seq}", "worker": worker if isinstance(worker, str) else None, "attempt": state["attempt"]})

    def after_tool_call(self, ctx: Any, result: Any) -> Any:
        self._release_parent(ctx)  # even when recording has stopped: a stale parent must not outlive its delegate call
        self._safely(self._tool_ended, ctx, result)
        return result

    def _release_parent(self, ctx: Any) -> None:
        try:
            run = getattr(ctx, "run", None)
            state = (run.metadata if run is not None else self._anonymous).get(_STATE_KEY)
            call = state["tools"].get(id(ctx)) if state else None
            token = call.get("token") if call else None
            if token is not None:
                call["token"] = None
                try:
                    _PARENT.reset(token)
                except ValueError:  # after_tool_call ran in another context than before_tool_call: clear this one
                    _PARENT.set(None)
        except Exception:  # noqa: BLE001 - never let bookkeeping reach effGen
            pass

    def _tool_ended(self, ctx: Any, result: Any) -> None:
        state = self._state(ctx)
        call = state["tools"].pop(id(ctx), None)
        fields: dict[str, Any] = {}
        if ctx.tool_name == DELEGATE_TOOL_NAME and isinstance(result, str):
            report = _json_args(result)
            confirmed = report.get("delegation_id") if isinstance(report, dict) else None
            if isinstance(confirmed, str) and confirmed:
                fields.update(delegation_id=confirmed, delegation_id_reported=confirmed)
            if isinstance(report, dict):
                fields["worker_status"] = report.get("status")  # the worker's reported status; says nothing about the tool call's own success
        self._emit("tool_call", **self._who(state), tool=ctx.tool_name, tool_call_id=f"{self.invocation}:{call['seq']}" if call else None,
                   generation_in_agent=state["generations"] or None, success=None, exception=None,  # unknown, not "true": see the module docstring
                   result=result if isinstance(result, str) else None if result is None else str(result), visible=None, visible_unavailable=VISIBLE_UNAVAILABLE,
                   seconds=round(time.monotonic() - call["t0"], 4) if call else None, **fields)
