# Execution history (SQLite), Phase 1

A SQLite **index** over the experiment evidence, so one orchestrator-worker session can be read in order: every model turn, every tool call, the verdict. The JSONL files stay the
evidence; the database is derived from them and can be deleted and rebuilt at any time. Standard library only (`sqlite3`). It runs as a separate offline process, never inside a
model run, so a database problem cannot touch a task.

Scope of Phase 1: runs made through the experiment runner (`experiments/rtx4070/stage_runner.py`, so the RTX stages and `experiments/dgx/dgx_diag.py`). A plain `smol-harness run` writes
no trace and therefore has no history yet; that is a separate later milestone.

## Use

```bash
python experiments/history/history.py import  <run folder> [...]          # safe to repeat; also works on a run that is still going
python experiments/history/history.py rebuild <run folder> [...]          # deletes the database first, then imports
python experiments/history/history.py sessions
python experiments/history/history.py timeline <session id> [--full] [--prompts]
python experiments/history/history.py usage    <session id>
python experiments/history/history.py issues                             # everything that could not be imported cleanly
python experiments/history/history.py purge --prompts [--session ID]     # drop captured prompt text from the database (hashes stay); sticky, see Retention
python experiments/history/history.py purge --session ID                 # drop a session from the database (the JSONL is untouched; importing it again restores it)
```
`--db FILE` (or `SMOL_HISTORY_DB`) chooses the database; the default is `experiments/history/history.db`, outside any run folder.

To record exact prompts, ask for it explicitly: `dgx_diag.py e1|e2 ... --capture-prompts`, or `SMOL_CAPTURE_PROMPTS=1` for the RTX stage runner. Without it no prompt text is stored anywhere.

## Ids and what they mean

| Thing | Id | Notes |
|---|---|---|
| session | `<invocation>:<attempt>` | one task execution (one attempt). `invocation` is a uuid made per runner process, so rerunning a stage into the same folder is a different session |
| message | `<invocation>:<seq>` | one model call. **Not** a chat message: the ReAct prompt is one string with the history inside it |
| tool call | `<invocation>:<seq of its tool_call_start line>` | the start line and the completion line carry the same id |

## Tables and views

`sessions` (status `finished` / `aborted` / `unfinished` / `inconsistent`, the task, the final answer, the agent's own outcome and the runner's verification), `messages`, `tool_calls`,
`events` (every line that is not one of those, kept whole), `issues`, `imports`. Views: `timeline`, `delegations`, `tool_counts`, `session_usage`.

- **Tool call status:** `completed`, `failed` (it ran and reported failure, or raised), `unfinished` (the start line exists, the completion does not: the process died or the run is still going).
  An unfinished call is **not** counted as an execution (`tool_counts` keeps it in its own column) and keeps its arguments, so a killed delegation still shows the task text it was given.
- **Which generation asked for a tool call:** recorded at the source (`requested_by_generation_seq`, `link_basis = recorded`). A trace from before that existed is linked only when exactly one
  generation of the same role and ordinal matches (`derived-ordinal`; a worker also has to match the delegation id); otherwise the link is NULL. It is never guessed.
- **Delegations:** a `delegate` call has the orchestrator's role but the new delegation id. `delegation_id` is the id it was *confirmed* to get (NULL if effGen refused the call, or if it never finished);
  `delegation_id_expected` is what it expected when it started. A worker's turns carry the id; the worker's name comes from the delegate call that claims that id.
- **Consumed text:** `generation_consumed` lines are merged into their message by `generation_seq`. A generation that was never settled stays `consumed_status = pending` with a NULL consumed text.
- **Usage and cost:** exactly what the provider reported, otherwise NULL (`usage_status`, `cost_status`: `reported` / `unpriced` / `not_reported`). `total_tokens` is never computed.
  `tokens_used_reported` is `GenerationResult.tokens_used` as returned, whose meaning depends on the provider. `session_usage` states how many calls had no usage, so a sum is never mistaken for a total.

## Prompts and privacy

- Captured prompts live in a separate `prompts.local.jsonl` (mode 0600, **not redacted**, so the SHA-256 in the trace still verifies), keyed by invocation and the `seq` of the model call.
  The text is **the prompt string effGen passed to the model's `generate()`**: the adapter input. It is not the provider's final payload (the OpenAI-compatible adapter wraps it in chat messages;
  a Hugging Face engine may apply a chat template or add tool definitions).
- `prompt_integrity`: `verified` (the text hashes to the trace's SHA-256), `mismatch`, `missing` (capture was on but this call has no prompt), `not_captured`, `purged`.
- The database and `prompts.local.jsonl` can hold anything a tool read. They are created 0600, are Git-ignored (`*.db*`, `*.sqlite*`, `*.local.jsonl`), are excluded from the packaged archive and are
  never rewritten by the packager. **Do not share them.** The shareable evidence is `trace.jsonl` and `attempts.jsonl` (redacted when written).

## Retention (manual) and what `purge` does not do

- **What it does:** every connection turns on `PRAGMA secure_delete` (and refuses to open the database if the SQLite build ignores it), so text removed by `purge` is overwritten with zeros in
  the database file instead of staying in free pages. The database is kept in rollback-journal mode: a database found in WAL mode is converted back, or refused if another connection prevents
  that, because a `-wal` file would hold old copies.
- **What it does not do:** this is not secure erasure, cryptographic or filesystem-level. `purge` changes **this database file only**. It does not touch `prompts.local.jsonl`, `trace.jsonl`,
  backups, snapshots, copies of the database, or blocks the filesystem may still hold (the rollback journal used during a write is deleted, not overwritten; SSDs and copy-on-write filesystems keep
  old blocks). To remove prompts for good, delete `prompts.local.jsonl` as well, and every copy of the database.
- **Purge is sticky:** a prompt that was purged stays purged when the same run is imported again, from the local folder or from an archive copy. `rebuild` starts from an empty database and is the
  explicit way to bring prompt text back from `prompts.local.jsonl`. `purge --session` removes the whole session; importing the evidence again restores it, prompts included.

## Import guarantees

One transaction per stage folder: an import that fails leaves the database as it was. Each invocation is replaced as a whole when its evidence changed and skipped when it did not, so repeating
an import cannot duplicate anything, and a shorter older copy never replaces a longer import.

A copy that lacks a file is missing evidence, not an instruction to forget. A packaged archive never contains `prompts.local.jsonl`, so importing one after the local folder would otherwise erase every
captured prompt. Instead, when an invocation is replaced, prompt text the database already holds is kept for every message whose text still hashes to the trace's SHA-256, and so is an attempt record
(final answer, verification) the new copy lacks. Each such case is recorded in `issues` as `kept_from_earlier_import`. If the first lines of a copy differ from what was imported (for example `package`
re-redacted a trace in place), the copy is imported and `trace_content_changed` is recorded; both notes stay on record when the invocation is analysed again. `sessions.source`, `messages` and the other
rows say where the latest import read them from, which can differ from where a kept prompt originally came from. A final line without its newline is a write in progress: reported as `partial_tail`, imported once complete.
A complete line that is not a JSON object, a line without `invocation`/`seq` (an older trace format: an id would have to be invented), duplicate sequence numbers, a settlement with no generation,
a prompt that matches no message and every similar inconsistency become rows in `issues`; nothing is dropped silently.

## What Phase 1 cannot give you

- History for plain `smol-harness run` (no trace exists there).
- The provider's final payload or the tokenised input; effGen's loop iteration and ledger totals; a worker's own effGen trace.
- A model call that never returned (no trace line, hence no prompt); the start time of a tool call other than from its start line; sub-second timestamps (order by `seq`).
- Cost for local models: the provider reports none, and the database stores NULL.
