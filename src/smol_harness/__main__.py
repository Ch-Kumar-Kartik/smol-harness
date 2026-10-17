"""Command line: `smol-harness run --config FILE --task TEXT [--json]` (also `python -m smol_harness`).

A thin layer over the existing pieces: `initialize_runtime()`, `load_config()`, `build_harness_from_config()` and
`Harness.run()`, with the harness closed by its context manager. Nothing is re-implemented here.

Streams: stdout carries the task's answer (or, with `--json`, one JSON document); stderr carries everything else
(worker summary, failures, warnings). Exit codes: 0 success; 1 the task failed (a failed `HarnessResult`); 2 usage error;
3 start-up failure (effGen patch, configuration, tools, model loading); 4 the run raised; 130 interrupted.

Diagnostics are short on purpose: exception type and a scrubbed, truncated message, a one-line summary per delegation.
No tracebacks, environment, configuration contents or full tool output are printed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from typing import Any

from smol_harness.config import load_config
from smol_harness.delegation import WorkerRecord
from smol_harness.harness import Harness, HarnessResult, build_harness_from_config, initialize_runtime

EXIT_OK, EXIT_TASK_FAILED, EXIT_USAGE, EXIT_STARTUP, EXIT_RUN_ERROR, EXIT_INTERRUPTED = 0, 1, 2, 3, 4, 130

MESSAGE_CHARS = 500  # per exception message / failed-task detail
WORKER_ERROR_CHARS = 160  # per delegation line on stderr
PREVIEW_CHARS = 500  # a worker's report in --json output

# Credentials that can sit in a URL (`http://user:pw@host`, e.g. a base_url), a bearer header, or a key=value pair.
_SECRETS = (
    (re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)[^/\s:@]+(?::[^/\s@]*)?@"), r"\g<scheme>***@"),
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{8,}"), r"\1 ***"),
    # NAME=value where NAME contains key/token/password/secret (OPENAI_API_KEY, HF_TOKEN, db_password, SMOL_TEST_SECRET, ...)
    (re.compile(r"(?i)([\w.-]*(?:api[_-]?key|token|password|passwd|secret|credential)[\w.-]*)(\s*[=:]\s*)\S+"), r"\1\2***"),
)


def _scrub(text: str) -> str:
    for pattern, replacement in _SECRETS:
        text = pattern.sub(replacement, text)
    return text


def _short(text: object, limit: int) -> str:
    """One line, scrubbed of credentials, cut to `limit` characters."""
    line = " ".join(_scrub(str(text)).split())
    return line if len(line) <= limit else f"{line[:limit]}... [{len(line) - limit} more chars]"


def _err(message: str = "") -> None:
    print(message, file=sys.stderr)


def _worker_dict(record: WorkerRecord) -> dict[str, Any]:
    result = record.result
    output = _scrub(result.output)
    return {
        "worker": record.worker, "delegation_id": result.delegation_id, "status": record.status,
        "iterations": result.iterations, "tool_calls": result.tool_calls, "tokens_used": result.tokens_used,
        "termination": result.termination, "error_type": result.error_type,
        "error": _short(result.error, MESSAGE_CHARS) if result.error else None,
        "output": output if len(output) <= PREVIEW_CHARS else f"{output[:PREVIEW_CHARS]}... [{len(output) - PREVIEW_CHARS} more chars]",
    }


def _print_workers(records: Sequence[WorkerRecord]) -> None:
    _err(f"delegations: {len(records)}" if records else "delegations: none")
    for record in records:
        result = record.result
        line = f"  {result.delegation_id} {record.worker}: {record.status} (iterations {result.iterations}, tool calls {result.tool_calls})"
        if record.status != "completed":
            line += f" - {result.error_type}: {_short(result.error or '', WORKER_ERROR_CHARS)}"
        _err(line)


def _error_document(stage: str, exc: BaseException, records: Sequence[WorkerRecord]) -> dict[str, Any]:
    return {"success": False, "output": "", "termination": None, "stop_reason": None,
            "error": {"stage": stage, "type": type(exc).__name__, "message": _short(exc, MESSAGE_CHARS)},
            "workers": [_worker_dict(r) for r in records]}


def _report_error(stage: str, exc: BaseException, records: Sequence[WorkerRecord], as_json: bool, code: int) -> int:
    what = {"startup": "start-up failed", "run": "the run raised"}[stage]
    _err(f"smol-harness: {what}: {type(exc).__name__}: {_short(exc, MESSAGE_CHARS)}")
    for note in getattr(exc, "__notes__", ()):
        _err(f"  note: {_short(note, MESSAGE_CHARS)}")
    if records:
        _print_workers(records)
    if as_json:
        print(json.dumps(_error_document(stage, exc, records), ensure_ascii=False))
    return code


def _report_result(result: HarnessResult, as_json: bool) -> int:
    stop_reason = getattr(result.response, "stop_reason", None)
    code = EXIT_OK if result.success else EXIT_TASK_FAILED
    if as_json:
        print(json.dumps({"success": result.success, "output": _scrub(result.output), "termination": result.termination,
                          "stop_reason": stop_reason, "error": None, "workers": [_worker_dict(r) for r in result.workers]},
                         ensure_ascii=False))
    elif result.success:
        print(result.output)  # the answer, and only the answer, on stdout
    _print_workers(result.workers)
    if not result.success:
        _err(f"smol-harness: the task failed (termination={result.termination}, stop_reason={stop_reason}): "
             f"{_short(result.output, MESSAGE_CHARS)}")
    return code


def _run_task(harness: Harness, task: str, as_json: bool) -> int:
    try:
        result = harness.run(task)
    except Exception as exc:  # a failure with no result to show (provider error, guardrail block, a bug); never reported as success
        return _report_error("run", exc, getattr(exc, "worker_records", ()), as_json, EXIT_RUN_ERROR)
    return _report_result(result, as_json)


def run_command(config_path: str, task: str, as_json: bool) -> int:
    try:
        try:
            initialize_runtime()  # the process-wide effGen patch must be in place before any agent exists
            harness = build_harness_from_config(load_config(config_path))
        except Exception as exc:
            return _report_error("startup", exc, (), as_json, EXIT_STARTUP)
        code = EXIT_OK
        try:
            with harness:  # closes the models this harness loaded, on success, failure or interrupt
                code = _run_task(harness, task, as_json)
        except Exception as exc:  # only releasing resources can raise here: the task's outcome stands, the problem is shown
            _err(f"smol-harness: warning: releasing resources failed: {type(exc).__name__}: {_short(exc, MESSAGE_CHARS)}")
        return code
    except KeyboardInterrupt:
        _err("smol-harness: interrupted")
        return EXIT_INTERRUPTED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smol-harness", description="Run a task on a configured orchestrator and its workers.",
        epilog="Exit codes: 0 ok, 1 task failed, 2 usage, 3 start-up failure, 4 the run raised, 130 interrupted. "
               "The shell tool is for trusted repositories only.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    run = commands.add_parser("run", help="run one task", description="Run one task and print the answer on stdout.")
    run.add_argument("--config", required=True, metavar="FILE", help="YAML configuration (workspace, models, tools, limits)")
    run.add_argument("--task", required=True, metavar="TEXT", help="what the orchestrator should get done")
    run.add_argument("--json", action="store_true", help="print one JSON document on stdout instead of the bare answer")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse: 2 after printing usage to stderr, 0 after --help
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    task = args.task.strip()
    if not task:
        parser.print_usage(sys.stderr)
        _err("smol-harness run: error: --task must not be empty")
        return EXIT_USAGE
    return run_command(args.config, task, args.json)


if __name__ == "__main__":
    sys.exit(main())
