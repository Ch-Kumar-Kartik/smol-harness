#!/usr/bin/env python
"""SQLite execution history: an INDEX over the experiment evidence, never its source. The JSONL files stay the evidence; this database is rebuilt from them at will.

    python experiments/history/history.py import <run folder or trace.jsonl>... [--db FILE]
    python experiments/history/history.py rebuild <run folder>... [--db FILE]      (deletes the database first)
    python experiments/history/history.py sessions [--db FILE]
    python experiments/history/history.py timeline <session id> [--full] [--prompts] [--db FILE]
    python experiments/history/history.py usage <session id> [--db FILE]
    python experiments/history/history.py issues [--db FILE]
    python experiments/history/history.py purge (--prompts | --session ID) [--session ID] [--db FILE]

Reads, per stage folder, `trace.jsonl` (the events), `attempts.jsonl` (the finished attempts, with the independent verification) and, if the run was made with prompt
capture, `prompts.local.jsonl` (private). Standard library only. Nothing here runs inside a model run: it is a separate offline process, so a database error cannot touch a task.

Identity: a session is one attempt, `session_id = <invocation>:<attempt>`; a message is one model call, `<invocation>:<seq>`; a tool call is `<invocation>:<seq of its
tool_call_start line>`. `invocation` is a uuid created per runner process, `seq` counts the lines of one invocation, so these ids are stable across re-imports and moves.

Honesty rules: a value the evidence does not hold is NULL, never a guess (token usage, cost, links); a line that cannot be imported is recorded in `issues`, never dropped silently.
The database and the prompts file can hold anything a tool read: they are Git-ignored, created with mode 0600 and never packaged. See experiments/history/README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DB = Path(os.environ.get("SMOL_HISTORY_DB") or HERE / "history.db")
SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE imports (          -- what was imported, per invocation: decides whether a re-import changes anything
  invocation TEXT PRIMARY KEY, source TEXT NOT NULL, trace_lines INTEGER NOT NULL, trace_sha256 TEXT NOT NULL,   -- hash of the trace lines alone
  bundle_sha256 TEXT NOT NULL,                                                                                  -- hash of trace + attempts + prompts
  imported_utc TEXT NOT NULL);

CREATE TABLE sessions (         -- one task execution = one attempt
  session_id TEXT PRIMARY KEY,  -- "<invocation>:<attempt>"
  run_id TEXT, stage TEXT, invocation TEXT NOT NULL, attempt TEXT NOT NULL, source TEXT,
  task TEXT, final_answer TEXT, seed INTEGER, model JSON,
  status TEXT NOT NULL,         -- finished | aborted | unfinished (no end line: killed, or still running) | inconsistent
  start_seq INTEGER, end_seq INTEGER, started_utc TEXT, ended_utc TEXT,
  agent_completed INTEGER, task_verified INTEGER, verification JSON, error TEXT);

CREATE TABLE messages (         -- one model call
  message_id TEXT PRIMARY KEY,  -- "<invocation>:<seq of the generation line>"
  session_id TEXT NOT NULL REFERENCES sessions, seq INTEGER NOT NULL, t_utc TEXT,
  role TEXT, agent_name TEXT, delegation_id TEXT, generation_in_agent INTEGER,
  provider TEXT, model TEXT, status TEXT NOT NULL,                -- ok | error
  prompt_sha256 TEXT, prompt_chars INTEGER,
  prompt_text TEXT, prompt_integrity TEXT NOT NULL,               -- verified | mismatch | missing (capture was on) | not_captured | purged
  raw_output TEXT, consumed_output TEXT, consumed_status TEXT, consumed_seq INTEGER, stop_handling TEXT,
  stop_sequences_sent JSON, stop_sequences_local JSON, discarded_chars INTEGER,
  finish_reason TEXT, seconds REAL, limits JSON, reasoning JSON, provider_metadata JSON,
  prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER, usage_status TEXT NOT NULL,   -- reported | not_reported
  cost_usd REAL, cost_status TEXT NOT NULL,                                                              -- reported | unpriced | not_reported
  usage_estimated INTEGER, tokens_used_reported INTEGER, error TEXT,
  UNIQUE (session_id, seq));

CREATE TABLE tool_calls (
  tool_call_id TEXT PRIMARY KEY,  -- "<invocation>:<seq of tool_call_start>" ("<invocation>:<seq of tool_call>" for a legacy trace without starts)
  session_id TEXT NOT NULL REFERENCES sessions,
  started_seq INTEGER, completed_seq INTEGER, started_utc TEXT, completed_utc TEXT,
  status TEXT NOT NULL,           -- completed | failed (it ran and reported failure, or raised) | unfinished (started, no completion line)
  tool TEXT, role TEXT, agent_name TEXT, generation_in_agent INTEGER,
  delegation_id TEXT,             -- agent context for a worker's call; for a `delegate` call the id it was CONFIRMED to get (NULL if refused or unfinished)
  delegation_id_expected TEXT,    -- `delegate` only: the id the call expected when it started
  requested_by_message_id TEXT REFERENCES messages, link_basis TEXT,   -- recorded | derived-ordinal | NULL (not established)
  args JSON, success INTEGER, exception INTEGER, error TEXT, result TEXT, visible TEXT, visible_unavailable TEXT,
  seconds REAL, dispatched INTEGER, delegation_id_reported TEXT);

CREATE TABLE events (           -- every line that is not a message, a tool call or an attempt boundary, kept whole
  event_id TEXT PRIMARY KEY, invocation TEXT NOT NULL, session_id TEXT, seq INTEGER, t_utc TEXT, type TEXT, payload JSON, source TEXT);

CREATE TABLE issues (           -- everything that could not be imported cleanly, so nothing is lost silently
  issue_id INTEGER PRIMARY KEY, source TEXT NOT NULL, invocation TEXT, line_no INTEGER, kind TEXT NOT NULL, detail TEXT, excerpt TEXT);

CREATE VIEW timeline AS
  SELECT session_id, start_seq AS seq, 'session_start' AS kind, session_id AS ref, NULL AS role, NULL AS delegation_id FROM sessions WHERE start_seq IS NOT NULL
  UNION ALL SELECT session_id, seq, 'message', message_id, role, delegation_id FROM messages
  UNION ALL SELECT session_id, started_seq, 'tool_start', tool_call_id, role, COALESCE(delegation_id, delegation_id_expected) FROM tool_calls WHERE started_seq IS NOT NULL
  UNION ALL SELECT session_id, completed_seq, 'tool_end', tool_call_id, role, delegation_id FROM tool_calls WHERE completed_seq IS NOT NULL
  UNION ALL SELECT session_id, end_seq, 'session_end', session_id, NULL, NULL FROM sessions WHERE end_seq IS NOT NULL;

CREATE VIEW delegations AS      -- a delegate call: what was asked, of whom, and how it ended
  SELECT session_id, tool_call_id, started_seq, COALESCE(delegation_id, delegation_id_expected) AS delegation_id, delegation_id IS NOT NULL AS confirmed,
         json_extract(args, '$.worker') AS worker, json_extract(args, '$.task') AS task, status, dispatched, requested_by_message_id
  FROM tool_calls WHERE tool = 'delegate';

CREATE VIEW tool_counts AS      -- unfinished calls are NOT executions that completed
  SELECT session_id, tool, role, SUM(status = 'completed') AS completed, SUM(status = 'failed') AS failed, SUM(status = 'unfinished') AS unfinished
  FROM tool_calls GROUP BY session_id, tool, role;

CREATE VIEW session_usage AS    -- sums of what providers REPORTED; the *_without_usage columns say how incomplete the sums are
  SELECT session_id, COUNT(*) AS messages, SUM(status = 'ok') AS ok_messages,
         SUM(prompt_tokens) AS prompt_tokens, SUM(completion_tokens) AS completion_tokens, SUM(cost_usd) AS cost_usd,
         SUM(status = 'ok' AND usage_status = 'not_reported') AS messages_without_usage, SUM(status = 'ok' AND cost_status != 'reported') AS messages_without_cost
  FROM messages GROUP BY session_id;
"""


# ------------------------------------------------------------------------------------------------------------------ database
def connect(path: Path) -> sqlite3.Connection:
    """Open (creating, mode 0600, if needed) the database. Transactions are explicit; a database of another schema version is refused, not migrated."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT, 0o600))
    conn = sqlite3.connect(path)
    conn.isolation_level = None
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # Deleted or updated content is overwritten with zeros in the database file instead of being left in free pages (purge relies on it), and verified: a build that ignored the
    # pragma would make `purge` a promise it cannot keep. Rollback-journal mode only: a WAL database keeps old page copies in a separate -wal file, so it is converted back (and
    # refused if another connection prevents that). This is not secure erasure: see "Retention" in the README for what it does not touch.
    conn.execute("PRAGMA secure_delete = ON")
    if conn.execute("PRAGMA secure_delete").fetchone()[0] != 1:
        conn.close()
        raise SystemExit(f"{path}: this SQLite build does not honour PRAGMA secure_delete, so purge cannot overwrite deleted text; refusing to open")
    try:
        mode = conn.execute("PRAGMA journal_mode = DELETE").fetchone()[0]
    except sqlite3.OperationalError as exc:  # raised (rather than returning 'wal') when another connection holds the file
        mode = f"wal ({exc})"
    if str(mode).lower() != "delete":
        conn.close()
        raise SystemExit(f"{path}: journal mode is {mode!r} and could not be switched (another connection holds the database open). WAL files (-wal/-shm) can keep copies of deleted text; close the other connection and retry")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == 0:
        conn.executescript(SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    elif version != SCHEMA_VERSION:
        conn.close()
        raise SystemExit(f"{path}: schema version {version}, this tool writes {SCHEMA_VERSION}. Rebuild: history.py rebuild <run folder> --db {path}")
    return conn


def dumps(value) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False)


def as_int(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def flag(value) -> int | None:
    return None if value is None else int(bool(value))


# ------------------------------------------------------------------------------------------------------------------ reading evidence
def read_jsonl(path: Path, issues: list) -> tuple[list, int]:
    """(complete records as (line_no, raw line bytes, dict), size of an unterminated last line). A final line without its newline is a write in progress or a write that was cut
    off: it is reported as `partial_tail`, not as malformed, and read again once it is complete. A complete line that is not a JSON object is an issue, never skipped silently."""
    if not path.exists():
        return [], 0
    data = path.read_bytes()
    pieces = data.split(b"\n")
    tail = pieces.pop()  # b"" when the file ends with a newline
    records = []
    for number, raw in enumerate(pieces, start=1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw.decode("utf-8"))
            if not isinstance(record, dict):
                raise ValueError("not a JSON object")
        except (UnicodeDecodeError, ValueError) as exc:
            issues.append({"source": str(path), "invocation": None, "line_no": number, "kind": "malformed", "detail": f"{type(exc).__name__}: {exc}", "excerpt": raw[:200].decode("utf-8", "replace")})
            continue
        records.append((number, raw, record))
    if tail:
        issues.append({"source": str(path), "invocation": None, "line_no": len(pieces) + 1, "kind": "partial_tail", "detail": f"{len(tail)} bytes without a newline: a write in progress or cut off; imported once complete",
                       "excerpt": tail[:200].decode("utf-8", "replace")})
    return records, len(tail)


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()  # the same rule as stage_runner.prompt_digest


def usage_of(metadata) -> dict:
    """Token usage and cost exactly as the provider reported them; anything else is NULL. Nothing is computed, estimated or defaulted to zero."""
    if not isinstance(metadata, dict):
        return dict(prompt_tokens=None, completion_tokens=None, total_tokens=None, usage_status="not_reported", cost_usd=None, cost_status="not_reported", usage_estimated=None)
    prompt, completion, total = as_int(metadata.get("prompt_tokens")), as_int(metadata.get("completion_tokens")), as_int(metadata.get("total_tokens"))
    reported = prompt is not None or completion is not None
    cost, status = None, "not_reported"
    if "cost_usd" in metadata:
        value = metadata["cost_usd"]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            cost, status = float(value), "reported"
        else:
            status = "unpriced"  # effGen reports None for a model with no published price instead of inventing $0
    return dict(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total, usage_status="reported" if reported else "not_reported", cost_usd=cost, cost_status=status,
                usage_estimated=flag(bool(metadata.get("estimated_usage") or metadata.get("estimated"))) if reported else None)


# ------------------------------------------------------------------------------------------------------------------ one invocation -> rows
def previous_evidence(conn: sqlite3.Connection, inv: str) -> dict:
    """What the database already holds for this invocation and a later copy might lack: hash-verified prompt text (or the fact that it was purged) and the attempt record's contents.
    A missing file in a copy is missing evidence, not an instruction to forget."""
    prompts = {r["message_id"]: (r["prompt_text"], r["prompt_integrity"], r["prompt_sha256"]) for r in conn.execute(
        "SELECT m.message_id, m.prompt_text, m.prompt_integrity, m.prompt_sha256 FROM messages m JOIN sessions s ON s.session_id = m.session_id "
        "WHERE s.invocation = ? AND m.prompt_integrity IN ('verified', 'purged')", (inv,))}
    attempts = {r["session_id"]: (r["final_answer"], r["verification"]) for r in conn.execute("SELECT session_id, final_answer, verification FROM sessions WHERE invocation = ? AND verification IS NOT NULL", (inv,))}
    return {"prompts": prompts, "attempts": attempts}


def build_invocation(inv: str, events: list, attempts: dict, prompts: dict, source: str, issues: list, previous: dict | None = None) -> dict:
    kept = {"prompts": 0, "purged": 0, "attempts": 0}
    sessions: dict = {}
    messages: dict = {}
    tools: dict = {}
    extra: list = []
    capture_on = any(e.get("event") == "prompt_capture_enabled" for _, e in events)

    def issue(kind, line_no, detail, excerpt=None):
        issues.append({"source": source, "invocation": inv, "line_no": line_no, "kind": kind, "detail": detail, "excerpt": excerpt})

    def session(attempt, line_no, e):
        sid = f"{inv}:{attempt}"
        if sid not in sessions:
            sessions[sid] = dict(session_id=sid, run_id=e.get("run_id"), stage=e.get("stage"), invocation=inv, attempt=attempt, source=source, task=None, final_answer=None, seed=None, model=None,
                                 status="unfinished", start_seq=None, end_seq=None, started_utc=None, ended_utc=None, agent_completed=None, task_verified=None, verification=None, error=None)
            if e.get("event") != "attempt_start":
                issue("event_before_attempt_start", line_no, f"a {e.get('event')} line for attempt {attempt!r} came before (or without) its attempt_start")
        return sessions[sid]

    for line_no, e in events:
        kind, seq, attempt = e.get("event"), e.get("seq"), e.get("attempt")
        if attempt is None and kind in ("generation", "generation_error", "tool_call_start", "tool_call", "generation_consumed"):
            issue("event_without_attempt", line_no, f"a {kind} line (seq {seq}) names no attempt; kept as an event")
            extra.append((e, None))
            continue
        if kind == "attempt_start":
            s = session(attempt, line_no, e)
            s.update(task=e.get("task"), seed=as_int(e.get("seed")), model=dumps(e.get("model")), start_seq=seq, started_utc=e.get("t_utc"))
        elif kind in ("generation", "generation_error"):
            s = session(attempt, line_no, e)
            ok = kind == "generation"
            meta = e.get("provider_metadata") if ok else None
            row = dict(message_id=f"{inv}:{seq}", session_id=s["session_id"], seq=seq, t_utc=e.get("t_utc"), role=e.get("role"), agent_name=None, delegation_id=e.get("delegation_id"),
                       generation_in_agent=as_int(e.get("generation_in_agent")), provider=e.get("provider"), model=e.get("model"), status="ok" if ok else "error",
                       prompt_sha256=e.get("prompt_sha256"), prompt_chars=as_int(e.get("prompt_chars")), prompt_text=None, prompt_integrity="missing" if capture_on else "not_captured",
                       raw_output=e.get("raw_output"), consumed_output=e.get("consumed_output"), consumed_status=e.get("consumed_status"), consumed_seq=None, stop_handling=e.get("stop_handling"),
                       stop_sequences_sent=dumps(e.get("stop_sequences")), stop_sequences_local=dumps(e.get("stop_sequences_local")), discarded_chars=as_int(e.get("discarded_chars")),
                       finish_reason=e.get("finish"), seconds=e.get("seconds"), limits=dumps(e.get("limits")), reasoning=dumps(e.get("reasoning")), provider_metadata=dumps(meta),
                       tokens_used_reported=as_int(e.get("tokens")), error=e.get("error"), **usage_of(meta))
            messages[row["message_id"]] = row
        elif kind == "generation_consumed":
            target = messages.get(f"{inv}:{e.get('generation_seq')}")
            if target is None:
                issue("generation_consumed_without_generation", line_no, f"settles generation seq {e.get('generation_seq')}, which is not in the trace; kept as an event")
                extra.append((e, sessions.get(f"{inv}:{attempt}", {}).get("session_id")))
            else:
                target.update(stop_handling=e.get("stop_handling"), stop_sequences_local=dumps(e.get("stop_sequences_local")), consumed_output=e.get("consumed_output"),
                              discarded_chars=as_int(e.get("discarded_chars")), consumed_status=e.get("consumed_status"), consumed_seq=seq)
        elif kind == "tool_call_start":
            s = session(attempt, line_no, e)
            tid = f"{inv}:{seq}"
            tools[tid] = dict(tool_call_id=tid, session_id=s["session_id"], started_seq=seq, completed_seq=None, started_utc=e.get("t_utc"), completed_utc=None, status="unfinished", tool=e.get("tool"),
                              role=e.get("role"), agent_name=None, generation_in_agent=as_int(e.get("generation_in_agent")), delegation_id=e.get("delegation_id"),
                              delegation_id_expected=e.get("delegation_id_expected"), requested_by_message_id=None, link_basis=None, args=dumps(e.get("args")), success=None, exception=None,
                              error=None, result=None, visible=None, visible_unavailable=None, seconds=None, dispatched=None, delegation_id_reported=None,
                              _recorded_link=("requested_by_generation_seq" in e), _requested_seq=as_int(e.get("requested_by_generation_seq")))
        elif kind == "tool_call":
            s = session(attempt, line_no, e)
            tid = e.get("tool_call_id") or f"{inv}:{seq}"  # a legacy line has no id: it is its own, started_seq stays NULL
            row = tools.get(tid)
            if row is None:
                row = tools[tid] = dict(tool_call_id=tid, session_id=s["session_id"], started_seq=None, completed_seq=None, started_utc=None, completed_utc=None, status="unfinished", tool=e.get("tool"),
                                        role=e.get("role"), agent_name=None, generation_in_agent=as_int(e.get("generation_in_agent")), delegation_id=None, delegation_id_expected=None,
                                        requested_by_message_id=None, link_basis=None, args=dumps(e.get("args")), success=None, exception=None, error=None, result=None, visible=None,
                                        visible_unavailable=None, seconds=None, dispatched=None, delegation_id_reported=None,
                                        _recorded_link=("requested_by_generation_seq" in e), _requested_seq=as_int(e.get("requested_by_generation_seq")))
                if e.get("tool_call_id"):
                    issue("tool_call_without_start", line_no, f"completion of {tid} has no tool_call_start line")
            failed = bool(e.get("exception")) or not e.get("success")
            row.update(completed_seq=seq, completed_utc=e.get("t_utc"), status="failed" if failed else "completed", delegation_id=e.get("delegation_id"), success=flag(e.get("success")),
                       exception=flag(e.get("exception")), error=e.get("error"), result=e.get("result"), visible=e.get("visible"), visible_unavailable=e.get("visible_unavailable"),
                       seconds=e.get("seconds"), dispatched=flag(e.get("dispatched")), delegation_id_reported=e.get("delegation_id_reported"))
        elif kind in ("attempt_end", "attempt_aborted"):
            s = session(attempt, line_no, e)
            if s["end_seq"] is not None:
                issue("conflicting_terminal_events", line_no, f"attempt {attempt!r} ends twice (first seq {s['end_seq']}, now {kind} seq {seq})")
                s["status"] = "inconsistent"
                continue
            s.update(end_seq=seq, ended_utc=e.get("t_utc"))
            if kind == "attempt_aborted":
                s.update(status="aborted", error=e.get("error"))
            else:
                s.update(status="finished", agent_completed=flag(e.get("agent_completed")), task_verified=flag(e.get("task_verified")))
                record = attempts.get(attempt)
                if record is None and previous and s["session_id"] in previous["attempts"]:
                    s.update(final_answer=previous["attempts"][s["session_id"]][0], verification=previous["attempts"][s["session_id"]][1])
                    kept["attempts"] += 1
                elif record is None:
                    issue("attempt_record_missing", line_no, f"attempt {attempt!r} ended but attempts.jsonl has no record for it; verification details and the final answer are unavailable")
                else:
                    s.update(final_answer=record.get("final_answer"), verification=dumps(record.get("verification")))
        else:
            extra.append((e, sessions.get(f"{inv}:{attempt}", {}).get("session_id") if attempt else None))

    # an attempt record without any trace line is evidence too: say so
    for attempt in attempts:
        if f"{inv}:{attempt}" not in sessions:
            issue("attempt_record_without_trace", None, f"attempts.jsonl holds attempt {attempt!r} but the trace has no line for it")

    # ---- worker identity: the delegate call that claims the delegation id names the worker
    claims: dict = {}
    for row in tools.values():
        if row["tool"] != "delegate":
            continue
        claimed = row["delegation_id"] if row["delegation_id"] is not None else (row["delegation_id_expected"] if row["status"] == "unfinished" else None)
        if claimed is not None:
            claims.setdefault((row["session_id"], claimed), []).append(row)
    unresolved = set()
    for table in (messages, tools):
        for row in table.values():
            if row["role"] != "worker" or row["delegation_id"] is None:
                continue
            rows = claims.get((row["session_id"], row["delegation_id"]), [])
            worker = json.loads(rows[0]["args"]).get("worker") if len(rows) == 1 and rows[0]["args"] else None
            if len(rows) == 1 and isinstance(worker, str):
                row["agent_name"] = worker
            elif (row["session_id"], row["delegation_id"]) not in unresolved:
                unresolved.add((row["session_id"], row["delegation_id"]))
                issue("worker_unresolved", None, f"{len(rows)} delegate calls claim {row['delegation_id']} in {row['session_id']}: the worker is not named")

    # ---- which generation requested which tool call: recorded at the source when possible; otherwise only an unambiguous (role, ordinal) match; otherwise nothing
    for row in tools.values():
        if row["_recorded_link"]:
            target = f"{inv}:{row['_requested_seq']}" if row["_requested_seq"] is not None else None
            if target in messages and messages[target]["session_id"] != row["session_id"]:
                issue("requested_by_other_session", None, f"{row['tool_call_id']} names generation seq {row['_requested_seq']}, which belongs to {messages[target]['session_id']}, not to its own session; not linked")
            elif target in messages and messages[target]["status"] == "ok":
                row.update(requested_by_message_id=target, link_basis="recorded")
            elif row["_requested_seq"] is not None:
                issue("requested_by_unresolved", None, f"{row['tool_call_id']} names generation seq {row['_requested_seq']}, which is not a successful message in the trace")
            continue
        candidates = [m for m in messages.values() if m["session_id"] == row["session_id"] and m["status"] == "ok" and m["role"] == row["role"]
                      and m["generation_in_agent"] == row["generation_in_agent"] and (row["role"] != "worker" or m["delegation_id"] == row["delegation_id"])]
        if len(candidates) == 1 and row["generation_in_agent"] is not None:
            row.update(requested_by_message_id=candidates[0]["message_id"], link_basis="derived-ordinal")

    # ---- prompts: attach by seq, verify the hash against the text and against the trace
    seen = set()
    for seq, record in prompts.items():
        message = messages.get(f"{inv}:{seq}")
        text = record.get("text")
        if message is None or not isinstance(text, str):
            issue("prompt_without_message", None, f"a captured prompt for seq {seq} matches no message in the trace" if message is None else f"the captured prompt for seq {seq} has no text")
            continue
        digest = prompt_sha256(text)
        good = digest == record.get("sha256") and message["prompt_sha256"] in (None, digest)
        message.update(prompt_text=text, prompt_integrity="verified" if good else "mismatch")
        seen.add(message["message_id"])
        if not good:
            issue("prompt_hash_mismatch", None, f"the captured prompt for seq {seq} hashes to {digest[:12]}, its own record says {str(record.get('sha256'))[:12]}, the trace says {str(message['prompt_sha256'])[:12]}")
    # ---- what an earlier import held and this copy does not have (an archive never contains the private prompts file). Only text that still hashes to the trace's own SHA-256 is kept.
    # A purge is sticky: a prompt that was purged stays purged however it is imported again (the prompts file may well still be on disk, and an ordinary re-import must not undo a retention
    # decision). `rebuild` starts from an empty database and is the explicit way to bring the text back.
    for message in messages.values():
        old = (previous or {}).get("prompts", {}).get(message["message_id"])
        if old is None or message["prompt_sha256"] is None or old[2] != message["prompt_sha256"]:
            continue
        if old[1] == "purged":
            message.update(prompt_text=None, prompt_integrity="purged")
            kept["purged"] += 1
        elif message["prompt_integrity"] in ("missing", "not_captured") and old[0] is not None and prompt_sha256(old[0]) == old[2]:
            message.update(prompt_text=old[0], prompt_integrity="verified")
            kept["prompts"] += 1
    if any(kept.values()):
        issue("kept_from_earlier_import", None, f"kept from the earlier import of this invocation: {kept['prompts']} prompt text(s) this copy lacks (hash-verified), {kept['purged']} prompt(s) that were purged and stay purged, "
                                                f"{kept['attempts']} attempt record(s) this copy lacks")
    return dict(sessions=list(sessions.values()), messages=list(messages.values()), tool_calls=list(tools.values()), events=extra)


def insert(conn: sqlite3.Connection, table: str, rows: list) -> None:
    for row in rows:
        row = {k: v for k, v in row.items() if not k.startswith("_")}
        conn.execute(f"INSERT INTO {table} ({', '.join(row)}) VALUES ({', '.join(':' + k for k in row)})", row)


def import_source(conn: sqlite3.Connection, trace: Path, now: str) -> dict:
    """One stage folder in ONE transaction: either everything of it is imported, or (on any error) the database is exactly as it was."""
    issues: list = []
    lines, tail = read_jsonl(trace, issues)
    attempts_all, _ = read_jsonl(trace.parent / "attempts.jsonl", issues)
    prompts_all, _ = read_jsonl(trace.parent / "prompts.local.jsonl", issues)
    by_inv: dict = {}
    for number, raw, record in lines:
        inv, seq = record.get("invocation"), record.get("seq")
        if not isinstance(inv, str) or not isinstance(seq, int) or not isinstance(record.get("event"), str):
            issues.append({"source": str(trace), "invocation": None, "line_no": number, "kind": "missing_identity", "detail": "no invocation/seq/event: a trace from before stable ids; not imported (an id would have to be invented)",
                           "excerpt": raw[:200].decode("utf-8", "replace")})
            continue
        by_inv.setdefault(inv, []).append((number, raw, record))
    summary = {"source": str(trace), "imported": [], "unchanged": [], "skipped_older": [], "sessions": 0, "messages": 0, "tool_calls": 0}
    conn.execute("BEGIN IMMEDIATE")
    try:
        for inv, group in by_inv.items():
            seen_seq, events = set(), []
            for number, raw, record in group:
                if record["seq"] in seen_seq:
                    issues.append({"source": str(trace), "invocation": inv, "line_no": number, "kind": "duplicate_seq", "detail": f"seq {record['seq']} appears twice; the first line is used", "excerpt": raw[:200].decode("utf-8", "replace")})
                    continue
                seen_seq.add(record["seq"])
                events.append((number, record))
            attempts = {r.get("attempt_id"): r for _, _, r in attempts_all if r.get("invocation_id") == inv}
            prompts = {}
            for _, _, r in prompts_all:
                if r.get("invocation") == inv and isinstance(r.get("seq"), int):
                    prompts.setdefault(r["seq"], r)
            digest = hashlib.sha256()
            for part in ([raw for _, raw, _ in group], [json.dumps(a, sort_keys=True).encode() for a in attempts.values()], [json.dumps(p, sort_keys=True).encode() for p in prompts.values()]):
                for raw in part:
                    digest.update(raw + b"\n")
                digest.update(b"\x00")
            have = conn.execute("SELECT trace_lines, trace_sha256, bundle_sha256 FROM imports WHERE invocation = ?", (inv,)).fetchone()
            raws = [raw for _, raw, _ in group]

            def trace_hash(lines):
                h = hashlib.sha256()
                for raw in lines:
                    h.update(raw + b"\n")
                return h.hexdigest()
            if have and have["bundle_sha256"] == digest.hexdigest():
                summary["unchanged"].append(inv)
                continue
            if have and len(group) < have["trace_lines"]:
                summary["skipped_older"].append(inv)
                issues.append({"source": str(trace), "invocation": inv, "line_no": None, "kind": "older_copy_skipped", "detail": f"this copy has {len(group)} lines, the database already holds {have['trace_lines']}", "excerpt": None})
                continue
            if have and trace_hash(raws[:have["trace_lines"]]) != have["trace_sha256"]:
                issues.append({"source": str(trace), "invocation": inv, "line_no": None, "kind": "trace_content_changed", "excerpt": None,
                               "detail": f"the first {have['trace_lines']} lines differ from the import already in the database (this copy has {len(group)} lines: "
                                         f"{'the same number of lines' if len(group) == have['trace_lines'] else 'it is longer'}); for example `package` re-redacting a trace in place. Replaced by this copy"})
            rows = build_invocation(inv, events, attempts, prompts, str(trace), issues, previous_evidence(conn, inv) if have else None)
            conn.execute("DELETE FROM tool_calls WHERE session_id IN (SELECT session_id FROM sessions WHERE invocation = ?)", (inv,))
            conn.execute("DELETE FROM messages WHERE session_id IN (SELECT session_id FROM sessions WHERE invocation = ?)", (inv,))
            conn.execute("DELETE FROM sessions WHERE invocation = ?", (inv,))
            conn.execute("DELETE FROM events WHERE invocation = ?", (inv,))
            insert(conn, "sessions", rows["sessions"])
            insert(conn, "messages", rows["messages"])
            insert(conn, "tool_calls", rows["tool_calls"])
            insert(conn, "events", [dict(event_id=f"{inv}:{e['seq']}", invocation=inv, session_id=sid, seq=e["seq"], t_utc=e.get("t_utc"), type=e["event"], payload=dumps(e), source=str(trace)) for e, sid in rows["events"]])
            conn.execute("INSERT OR REPLACE INTO imports VALUES (?, ?, ?, ?, ?, ?)", (inv, str(trace), len(group), trace_hash(raws), digest.hexdigest(), now))
            summary["imported"].append(inv)
            for key in ("sessions", "messages", "tool_calls"):
                summary[key] += len(rows[key])
        # Issues follow their rows. File-level ones (malformed lines, a partial tail, lines without identity) are regenerated by every parse. An invocation's own issues are replaced
        # when that invocation is re-imported and left alone when it is unchanged (this run did not analyse it again, so anything it produced here would only duplicate them).
        imported, skipped = set(summary["imported"]), set(summary["skipped_older"])
        keep = [i for i in issues if i["invocation"] is None or i["invocation"] in imported or (i["invocation"] in skipped and i["kind"] == "older_copy_skipped")]
        conn.execute("DELETE FROM issues WHERE invocation IS NULL AND source IN (?, ?, ?)", (str(trace), str(trace.parent / "attempts.jsonl"), str(trace.parent / "prompts.local.jsonl")))
        for inv in imported:  # the two "this import changed something" notes are a log, not an analysis: they stay when the invocation is analysed again
            conn.execute("DELETE FROM issues WHERE invocation = ? AND kind NOT IN ('trace_content_changed', 'kept_from_earlier_import')", (inv,))
        for inv in skipped:
            conn.execute("DELETE FROM issues WHERE invocation = ? AND kind = 'older_copy_skipped'", (inv,))
        for item in keep:
            conn.execute("INSERT INTO issues (source, invocation, line_no, kind, detail, excerpt) VALUES (:source, :invocation, :line_no, :kind, :detail, :excerpt)", item)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    summary["issues"] = len(keep)
    summary["partial_tail_bytes"] = tail
    return summary


def find_traces(paths) -> list:
    found = []
    for path in map(Path, paths):
        found += sorted(path.rglob("trace.jsonl")) if path.is_dir() else [path]
    return found


def cmd_import(args) -> int:
    import datetime as dt
    conn = connect(args.db)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    traces = find_traces(args.paths)
    if not traces:
        print("no trace.jsonl found")
        return 1
    for trace in traces:
        s = import_source(conn, trace, now)
        print(f"{trace}: imported {len(s['imported'])} invocation(s) ({s['sessions']} sessions, {s['messages']} messages, {s['tool_calls']} tool calls), unchanged {len(s['unchanged'])}, "
              f"older copy skipped {len(s['skipped_older'])}, issues {s['issues']}")
    return 0


def cmd_rebuild(args) -> int:
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(str(args.db) + suffix).unlink(missing_ok=True)
    return cmd_import(args)


# ------------------------------------------------------------------------------------------------------------------ queries
def clip(value, width: int) -> str:
    text = "" if value is None else str(value).replace("\n", "\\n")
    return text if width <= 0 or len(text) <= width else text[:width] + f"... [{len(text) - width} more chars]"


def timeline_lines(conn: sqlite3.Connection, session_id: str, full: bool = False, prompts: bool = False) -> list:
    width = 0 if full else 160
    s = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if s is None:
        raise SystemExit(f"no such session: {session_id}")
    out = [f"=== session {s['session_id']}  run={s['run_id']} stage={s['stage']} status={s['status']}", f"task: {clip(s['task'], width)}"]
    for item in conn.execute("SELECT * FROM timeline WHERE session_id = ? ORDER BY seq, CASE kind WHEN 'tool_end' THEN 1 ELSE 0 END", (session_id,)):
        kind, ref = item["kind"], item["ref"]
        if kind in ("session_start", "session_end"):
            continue
        indent = "    " if item["role"] == "worker" else ""
        who = f"{item['role']}" + (f" {item['delegation_id']}" if item["delegation_id"] else "")
        if kind == "message":
            m = conn.execute("SELECT * FROM messages WHERE message_id = ?", (ref,)).fetchone()
            usage = f"tokens p={m['prompt_tokens']} c={m['completion_tokens']}" if m["usage_status"] == "reported" else "usage not reported"
            cost = f", cost ${m['cost_usd']}" if m["cost_status"] == "reported" else f", cost {m['cost_status'].replace('_', ' ')}"
            head = f"{indent}#{item['seq']} [{who} #{m['generation_in_agent']}] " + (f"MESSAGE {m['message_id']} finish={m['finish_reason']} {usage}{cost}" if m["status"] == "ok" else f"MESSAGE FAILED {m['message_id']}: {clip(m['error'], width)}")
            out += [head, f"{indent}    prompt: {m['prompt_chars']} chars sha256={str(m['prompt_sha256'])[:12]} ({m['prompt_integrity']})"]
            if prompts and m["prompt_text"] is not None:
                out.append(f"{indent}    prompt text: {clip(m['prompt_text'], width)}")
            if m["status"] == "ok":
                out.append(f"{indent}    raw: {clip(m['raw_output'], width)}")
                if m["consumed_output"] is None:
                    out.append(f"{indent}    consumed: UNAVAILABLE ({m['consumed_status']})")
                elif m["consumed_output"] != m["raw_output"]:
                    out.append(f"{indent}    consumed ({m['stop_handling']}, {m['discarded_chars']} chars cut): {clip(m['consumed_output'], width)}")
                else:
                    out.append(f"{indent}    consumed: same as raw ({m['stop_handling']})")
        else:
            t = conn.execute("SELECT * FROM tool_calls WHERE tool_call_id = ?", (ref,)).fetchone()
            name = t["agent_name"] and f" ({t['agent_name']})" or ""
            if kind == "tool_start":
                link = f" requested by {t['requested_by_message_id']} [{t['link_basis']}]" if t["requested_by_message_id"] else " requested by: not established"
                line = f"{indent}#{item['seq']} [{who}] TOOL START {t['tool']}{name}{link}"
                if t["tool"] == "delegate":
                    line += f" -> worker {json.loads(t['args'] or '{}').get('worker')!r} (expects {t['delegation_id_expected']})"
                out += [line, f"{indent}    args: {clip(t['args'], width)}"]
            else:
                out.append(f"{indent}#{item['seq']} [{who}] TOOL END   {t['tool']} {t['status'].upper()} in {t['seconds']}s" + (f" as {t['delegation_id']}" if t["tool"] == "delegate" and t["delegation_id"] else ""))
                out.append(f"{indent}    result: {clip(t['result'], width)}" if t["result"] is not None else f"{indent}    result: none ({t['error']})")
                if t["visible"] is None:
                    out.append(f"{indent}    the model saw: UNAVAILABLE ({t['visible_unavailable']})")
                elif t["visible"] != t["result"]:
                    out.append(f"{indent}    the model saw: {clip(t['visible'], width)}")
    for t in conn.execute("SELECT * FROM tool_calls WHERE session_id = ? AND status = 'unfinished' ORDER BY started_seq", (session_id,)):
        out.append(f"!!  UNFINISHED: {t['tool']} {t['tool_call_id']} started and never completed; NOT counted as an execution")
    verdict = f"=== verdict: status={s['status']}  agent_completed={fmt(s['agent_completed'])}  task_verified={fmt(s['task_verified'])}"
    failed = [c["name"] for c in json.loads(s["verification"] or "{}").get("checks", []) if not c.get("passed")]
    out += [verdict + (f"  failed checks: {failed}" if failed else ""), f"final answer: {clip(s['final_answer'], width)}" if s["final_answer"] is not None else "final answer: none recorded"]
    if s["error"]:
        out.append(f"error: {clip(s['error'], width)}")
    return out


def fmt(value) -> str:
    return "unknown" if value is None else ("yes" if value else "NO")


def cmd_timeline(args) -> int:
    print("\n".join(timeline_lines(connect(args.db), args.session, args.full, args.prompts)))
    return 0


def cmd_sessions(args) -> int:
    conn = connect(args.db)
    print(f"{'session_id':32} {'stage':14} {'status':11} {'agent_done':10} {'verified':9} {'msgs':>4} {'tools ok/fail/unfinished':>24}  task")
    for s in conn.execute("SELECT * FROM sessions ORDER BY started_utc, session_id"):
        n = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (s["session_id"],)).fetchone()[0]
        t = conn.execute("SELECT COALESCE(SUM(status='completed'),0), COALESCE(SUM(status='failed'),0), COALESCE(SUM(status='unfinished'),0) FROM tool_calls WHERE session_id = ?", (s["session_id"],)).fetchone()
        print(f"{s['session_id']:32} {str(s['stage']):14} {s['status']:11} {fmt(s['agent_completed']):10} {fmt(s['task_verified']):9} {n:>4} {f'{t[0]}/{t[1]}/{t[2]}':>24}  {clip(s['task'], 50)}")
    return 0


def cmd_usage(args) -> int:
    conn = connect(args.db)
    u = conn.execute("SELECT * FROM session_usage WHERE session_id = ?", (args.session,)).fetchone()
    if u is None:
        raise SystemExit(f"no messages for session {args.session}")
    print(f"{args.session}: {u['ok_messages']} successful model calls; prompt tokens {u['prompt_tokens']}, completion tokens {u['completion_tokens']}, cost {u['cost_usd']}")
    print(f"  incomplete: {u['messages_without_usage']} of {u['ok_messages']} calls reported no token usage; {u['messages_without_cost']} reported no cost (NULL sums mean nothing was reported)")
    for r in conn.execute("SELECT role, delegation_id, COUNT(*) n, SUM(prompt_tokens) p, SUM(completion_tokens) c FROM messages WHERE session_id = ? AND status = 'ok' GROUP BY role, delegation_id ORDER BY MIN(seq)", (args.session,)):
        print(f"  {r['role']:12} {str(r['delegation_id'] or '-'):4} calls={r['n']} prompt={r['p']} completion={r['c']}")
    for r in conn.execute("SELECT * FROM tool_counts WHERE session_id = ?", (args.session,)):
        print(f"  tool {r['tool']:16} role={r['role']:12} completed={r['completed']} failed={r['failed']} unfinished={r['unfinished']}")
    return 0


def cmd_issues(args) -> int:
    rows = connect(args.db).execute("SELECT * FROM issues ORDER BY source, line_no").fetchall()
    for r in rows:
        print(f"{r['kind']:38} {r['source']}:{r['line_no']}  {r['detail']}")
    print(f"{len(rows)} issue(s)")
    return 0


def cmd_purge(args) -> int:
    conn = connect(args.db)
    if not (args.prompts or args.session):
        raise SystemExit("purge needs --prompts and/or --session")
    where, params = ("AND session_id = ?", (args.session,)) if args.session else ("", ())
    conn.execute("BEGIN IMMEDIATE")
    try:
        if args.prompts:
            n = conn.execute(f"UPDATE messages SET prompt_text = NULL, prompt_integrity = 'purged' WHERE prompt_text IS NOT NULL {where}", params).rowcount
            print(f"prompt text removed from {n} message(s) in this database file (overwritten in place; hashes kept). Not touched: prompts.local.jsonl, backups, snapshots and any other copy")
        elif args.session:
            inv = conn.execute("SELECT invocation FROM sessions WHERE session_id = ?", (args.session,)).fetchone()
            conn.execute("DELETE FROM tool_calls WHERE session_id = ?", params)
            conn.execute("DELETE FROM messages WHERE session_id = ?", params)
            conn.execute("DELETE FROM sessions WHERE session_id = ?", params)
            if inv:
                conn.execute("DELETE FROM imports WHERE invocation = ?", (inv[0],))  # so that importing the evidence again restores it
            print(f"session {args.session} removed from this database file (overwritten in place). Not touched: the JSONL evidence, prompts.local.jsonl, backups, snapshots and any other copy")
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="history.py", description="SQLite index over the experiment JSONL evidence.")
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, fn, text, paths=False, session=False):
        p = sub.add_parser(name, help=text)
        p.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"database file (default {DEFAULT_DB}; SMOL_HISTORY_DB overrides)")
        if paths:
            p.add_argument("paths", nargs="+", help="run folders or trace.jsonl files")
        if session:
            p.add_argument("session", help="session id (see `sessions`)")
        p.set_defaults(func=fn)
        return p

    command("import", cmd_import, "import (or update from) run folders; safe to repeat", paths=True)
    command("rebuild", cmd_rebuild, "delete the database and import again from the evidence", paths=True)
    command("sessions", cmd_sessions, "list sessions")
    t = command("timeline", cmd_timeline, "one session in order: messages, tool calls, results, verdict", session=True)
    t.add_argument("--full", action="store_true", help="do not shorten texts")
    t.add_argument("--prompts", action="store_true", help="also print captured prompt text")
    command("usage", cmd_usage, "token usage, cost availability and tool counts of a session", session=True)
    command("issues", cmd_issues, "what could not be imported cleanly")
    p = command("purge", cmd_purge, "remove captured prompt text and/or a session from the database")
    p.add_argument("--prompts", action="store_true")
    p.add_argument("--session")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
