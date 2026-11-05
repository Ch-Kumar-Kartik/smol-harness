# DGX Spark validation: E0, E1, E2

**Status (corrected 2026-11-05): E0, E2 and a five-trial E3 repeat were run on a DGX Spark on 2026-10-30 with `smol-qwen3-14b:8k` (results: `docs/new-session.md` section 4, 2026-10-31 entry). No Gemma model has been run here.** The text below was written before that run and is otherwise unchanged; where it says "not run", read it as historical. This tool imports code from `experiments/rtx4070/` (a historical package) and is **not** used by the Gemma pilot runbook, `../../instructions.md`.

Purpose: Linux (ARM64) correctness of the harness with a real model, and a reproducible trajectory for every attempt. Not a benchmark; not an RTX 4070 stage.

| | What it shows | Real model | Uses `shell` |
|---|---|---|---|
| **E0** | the host, GPU, Python stack, Linux sandbox and Ollama are fit to run E1/E2 | one 16-token generation | sandbox probes only |
| **E1** | one agent reads `secret.txt` with `file_operations`; the runner checks the outcome itself | yes | no |
| **E2** | an orchestrator delegates to a worker that reads `secret.txt`; the runner checks the outcome itself | yes | no |

`dgx_diag.py` is a thin adapter over the instruments of `experiments/rtx4070/stage_runner.py` (trace, verification, action ledger). It does not use the RTX preflight, which
is wrong for this host (below), does not touch the RTX package, and changes no production code, model, prompt or sandbox policy.

## 1. Prerequisites on the DGX Spark

| Need | How E0 checks it |
|---|---|
| Linux, aarch64 (DGX OS), kernel recorded | `host` (WARN, not FAIL, on another architecture so the driver can be dry-run elsewhere) |
| NVIDIA driver and GPU visible; GB10 expected | `gpu`: the raw `nvidia-smi` memory columns are recorded. If they are not numbers (unified memory) the figures come from system RAM and say so |
| CUDA 13 capable driver | `cuda_driver` (reported; not gating) |
| At least 12 GiB of memory available (the GPU shares it; 24 GiB recommended) | `memory` |
| 20 GiB free disk | `disk` |
| `uv sync --locked` done; effGen 1.3.0; **torch can run real kernels on the GPU** (fp32 and bf16 matmul checked against CPU); no CPU fallback | `project_env` |
| Linux sandbox: runs commands, writes only inside the workspace, no network. Fails closed if user namespaces are blocked | `sandbox` (recorded; **not gating**: E1/E2 use only `file_operations`) |
| Ollama (ARM64 build) reachable at `127.0.0.1:11434` (or `SMOL_OLLAMA_URL`), a model tag **already installed** with `num_ctx` fixed in its Modelfile | `ollama` |
| A tiny real generation, and where Ollama placed the model (`/api/ps`) | `ollama_inference` (placement below 95 % on the GPU is a WARN with the numbers) |

Not needed: `bitsandbytes`, Hugging Face weights, Windows, WSL2. **No download happens anywhere in this sequence.** Which model to use is yours to decide: pick one tag that is
already installed and keep it for E1 and E2 and every rerun. For comparability with the RTX package the default would be `qwen2.5:7b-instruct-q4_K_M` (4.68 GB; manifest digest in
`experiments/rtx4070/models.json`); pulling it needs your approval. Prefer a non-reasoning model first (see the Qwen3 note in section 6).

## 2. One-time preparation (no download)

```bash
git rev-parse HEAD && git status --short          # record the commit; the tree should be clean
uv sync --locked                                  # Linux aarch64 wheels are in uv.lock for torch 2.14.1 and all 16 nvidia-* packages
ollama list                                       # choose a tag that is already here: BASE=<that tag>

cat > /tmp/Modelfile.dgx <<EOF
FROM $BASE
PARAMETER num_ctx 8192
PARAMETER num_predict 1024
EOF
ollama create smol-diag:8k -f /tmp/Modelfile.dgx  # local only: this is what rtx4070.py prepare does after its pull
```

## 3. Commands (run from the repository root; one results folder for all three)

```bash
export OUT=$PWD/experiments/dgx/results/dgx-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)
export TAG=smol-diag:8k
uv run python experiments/dgx/dgx_diag.py e0 --out "$OUT" --tag "$TAG" --num-ctx 8192     # read-only except one 16-token generation
uv run python experiments/dgx/dgx_diag.py e1 --out "$OUT" --tag "$TAG" --num-ctx 8192     # refuses unless E0's gate is open for this exact tag
uv run python experiments/dgx/dgx_diag.py e2 --out "$OUT" --tag "$TAG" --num-ctx 8192
python3 experiments/rtx4070/rtx4070.py package --run-dir "$OUT"                            # one redacted archive; refuses if a credential survives
```

Optional, explicit: add `--capture-prompts` to `e1`/`e2` to also record every prompt (the string effGen hands to the model; not the provider's final payload) in a **local-only** `prompts.local.jsonl`
(mode 0600; never archived or rewritten by `package`). `experiments/history/history.py` can then index the run into SQLite for reading (`import`, `timeline`); see `experiments/history/README.md`.

Run **one attempt first** (the default). Look at `trace.jsonl` before adding `--attempts 2` or `3` (seeds 0..N-1; E1 stops at the first verified attempt, E2 at the first
verified attempt with the mechanics shown). Exit status: E0 `1` if any check FAILED; E1/E2 `0` if the mechanics were demonstrated, `1` if not, `2` if refused.
**Exit `0` does not mean the task was solved.** (E3 is run after E2, see below.)

Fixed across E1, E2 and reruns: the model tag and its digest, `num_ctx 8192`, ReAct mode, the runner's sampling overrides (temperature 0.1, `max_tokens` 768, recorded in every
result), seeds 0..N-1, the task strings in `stage_runner.py`, tool allowlists (`file_operations` only), the limits in `experiments/rtx4070/configs/ollama_only.yaml`.
These are the runner's labelled overrides, not the production defaults.

### E3: context transfer between two workers (prepared, NOT RUN; needs a passed E0 and a demonstrated E2)

```bash
uv run python experiments/dgx/dgx_diag.py e3 --out "$OUT" --tag "$TAG" --num-ctx 8192 [--capture-prompts]   # one attempt, one short value; same $OUT and $TAG as e0 and e2
```
What it asks: one Ollama model plays the orchestrator and two workers (`reader`, `writer`, both `file_operations` only). Worker `reader` reads the value from `source.txt`
(`TRANSFER-4172`, written by the runner and never mentioned in the task text), reports it to the orchestrator, and the orchestrator must put the value into its delegation to `writer`
(the task text says so; the writer cannot see the conversation), which creates `result.txt`. Limits: `max_delegations 3`, report cap 3000 characters. E3 refuses (exit 2, nothing
loaded) unless E0 passed for this tag and context AND `diag_E2/result.json` exists for the same tag with the mechanics demonstrated.

Two verdicts, kept apart, both decided by the runner from executions, prompts and the disk, never from the model's answer:

| Verdict (field) | Checks |
|---|---|
| `context_transfer.exact` | the reader executed a successful read of `source.txt` whose result showed it the value; the reader was dispatched before the writer; the value is NOT in the first delegation; the report the orchestrator received from the reader contains the value; the value is in the writer delegation's `task` or `context`; the value is in the writer model's FIRST prompt (what the delegation handed it, not its own later output); no writer file call bypassed the orchestrator (below); **the value first appeared in the reader's own prompt**: the first model prompt, in trace order, that contains the value belongs to the reader's delegation, was issued AFTER its successful read of `source.txt` (its trace sequence number is greater than the read's) and contains that read's observation, so nobody (orchestrator, writer, configuration, the reader's initial prompt) held the value earlier; missing sequence numbers fail closed |
| `verification` (task) | `result.txt` exists; its stripped content is exactly the value; `source.txt` is byte-identical; `result.txt` is the only file added or changed |

**Matching.** The value is matched as a whole token (`has_token`): `TRANSFER-4172.`, `"TRANSFER-4172"` and `value: TRANSFER-4172` match; `TRANSFER-41729`, `XTRANSFER-4172`, `TRANSFER-4172-b` and `TRANSFER-4172_x` do not, so an altered value is never an exact transfer. `result.txt` is compared by exact equality of its stripped content.
**Which report.** The reader's report is the executed `delegate` call's own result (`diag_E3/trace.jsonl`, `tool_call.visible`), bounded only by `max_report_chars` (3000), i.e. what the orchestrator received. The CLI's `--json` worker output is a display preview cut to 500 characters; it is stored beside it as `report_output_cli_preview` and no check reads it. If the value lies beyond the 3000-character cut, the transfer fails even if a later delegation names the correct value. Nothing is inferred from the orchestrator's final answer.
**Writer bypass.** effGen's `file_operations` has six operations: `read` and `convert` read file content (`convert` also writes a copy); `search` is a filename glob, `list` and `metadata` return names, sizes and times, none of which can show the value. The writer fails the transfer if it calls `read` or `convert` with any argument naming `source.` (so `source.txt` and a converted `source.json`), or if any other call's observation returned the value to it (a read of a copy, say), except its own `write` and its own `result.txt`. `list`, `search` and `metadata` on the source are recorded (`names_source`) and are not a bypass.
Exact transfer does not imply a correct file, and a correct file does not imply an exact transfer (a writer that reads `source.txt` itself passes the task and fails the transfer). The
result row prints both (`TASK VERIFIED=..  CONTEXT TRANSFER EXACT=..`); exit `0` still means only that the mechanics (two dispatched delegations, a worker file call, reports back at the
orchestrator) ran. The attempt record holds the full `delegations` (task AND context as sent, worker, id, the report output), every worker tool call attributed to a worker and
delegation id, `value_carried_in`, `writer_read_source`, `result_file`, and `worker_context_isolated`. In SQLite (`history.py import`), the transferred text is
`SELECT json_extract(args,'$.task'), json_extract(args,'$.context') FROM tool_calls WHERE tool = 'delegate'`, and with `--capture-prompts` the writer's exact prompt is in `timeline --prompts`.
Delegate input is limited to 10,000 characters (see `docs/effgen-api-audit.md`, Part 3, "Subagent context transfer"): a longer call is refused with an explicit error.
What the evidence cannot show: the check proves no recorded writer tool call read or returned the source; it cannot prove how a model produced a value that is not in its prompt (that is the point of an unguessable value), and it knows only the writer's `file_operations` calls (the E3 config gives workers no other tool). Only the first dispatched writer delegation is judged: a retry that adds the value later scores as not exact.
Not part of E3 yet: several values, long values, other tags, parallel or direct worker-to-worker passing.

### E3 repeat suite: independent trials with a fresh random value (prepared, NOT RUN)

`e3-repeat` runs the E3 trial several times, one after another, and summarises them. Each trial is exactly the single-run E3 (same task, same config, same checks, nothing relaxed) in its own
folder with its own run id, trace, workspace and **fresh random transfer value** (`TRANSFER-` plus 8 random characters from the OS CSPRNG, generated by the runner, written only into that
trial's `source.txt`, never put in the task, the configuration or any prompt before the reader reads it). Our Ollama model is unloaded after every trial. Single-run `e3` is unchanged.

**Prerequisites, per model.** Gates read one results folder, so use ONE `--out` folder PER MODEL TAG that holds `diag_E0/e0.json` and `diag_E2/result.json` for that tag and the same `--num-ctx`
(run `e0` and `e2` into it first, or reuse the folder of an earlier session where both passed for that tag). `e3-repeat` also refuses unless Ollama lists the tag now and the model digest equals
the one E0 recorded (a re-created or replaced tag must go through E0 and E2 again). The model identity is resolved again BEFORE EVERY TRIAL and must still equal both the suite-start digest and E0's: if it changed, or Ollama gives no usable digest, that trial is `blocked` (not run, no model loaded for it), the remaining trials are `not_run`, the suite exits 1 and the finished trials stay intact. A blocked trial is never counted as a task failure or a success. Nothing is downloaded. Runs on the two target tags:

```bash
cd /path/to/smol-harness && git status --porcelain          # must print nothing; otherwise STOP and report
NUM_CTX=8192
logged() {  # logged NAME command...   -> $OUT/logs/NAME-{command.txt,stdout.log,stderr.log,exit-status.txt}; prints and returns the REAL exit status
  local n="$1"; shift; mkdir -p "$OUT/logs"
  if [ -e "$OUT/logs/$n-exit-status.txt" ]; then echo "STOP: logs named $n already exist; pick another name, for example $n-retry1" >&2; return 97; fi
  printf '%q ' "$@" > "$OUT/logs/$n-command.txt"; echo >> "$OUT/logs/$n-command.txt"
  trap ':' INT   # Ctrl-C reaches the whole foreground group; without a trap an interactive shell abandons this function when the command dies of SIGINT, and the status below is never written
  "$@" > >(tee -i "$OUT/logs/$n-stdout.log") 2> >(tee -i "$OUT/logs/$n-stderr.log" >&2)   # tee -i: the loggers survive Ctrl-C and keep the log tail
  local s=$?; sleep 1; echo "$s" > "$OUT/logs/$n-exit-status.txt"; echo "[$n] exit status $s"; trap - INT; return $s
}

# ---- model 1: smol-qwen3-14b:8k ----
TAG=smol-qwen3-14b:8k
OUT=$PWD/experiments/dgx/results/dgx-e3r-14b-$(date -u +%Y%m%d-%H%M%S)      # or the existing folder that already holds this tag's E0 and E2
mkdir -p "$OUT"
logged e0 uv run python experiments/dgx/dgx_diag.py e0 --out "$OUT" --tag "$TAG" --num-ctx "$NUM_CTX"          # then read the last line: it must say "E1/E2 may run: True"
logged e2 uv run python experiments/dgx/dgx_diag.py e2 --out "$OUT" --tag "$TAG" --num-ctx "$NUM_CTX"          # exit status 0 AND task verified are what you want to see
logged e3r uv run python experiments/dgx/dgx_diag.py e3-repeat --out "$OUT" --tag "$TAG" --num-ctx "$NUM_CTX"  # 5 trials (the default)
SUITE=$(ls -d "$OUT"/diag_E3R/e3r-* | tail -1); cat "$SUITE/report.md"

# ---- model 2: smol-qwen3.8-27b:8k  (a different folder; do not reuse model 1's) ----
TAG=smol-qwen3.8-27b:8k
OUT=$PWD/experiments/dgx/results/dgx-e3r-27b-$(date -u +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
logged e0 uv run python experiments/dgx/dgx_diag.py e0 --out "$OUT" --tag "$TAG" --num-ctx "$NUM_CTX"
logged e2 uv run python experiments/dgx/dgx_diag.py e2 --out "$OUT" --tag "$TAG" --num-ctx "$NUM_CTX"
logged e3r uv run python experiments/dgx/dgx_diag.py e3-repeat --out "$OUT" --tag "$TAG" --num-ctx "$NUM_CTX"
SUITE=$(ls -d "$OUT"/diag_E3R/e3r-* | tail -1); cat "$SUITE/report.md"
```
Run the lines one at a time and stop at the first one that is refused (exit 2) or fails its gate. Options: `--trials N` (1-10, default 5), `--seed-start S` (trial k uses seed S+k, default 0),
`--capture-prompts` (off by default; private per-trial `prompts.local.jsonl`, never archived). Exit status: `0` all requested trials completed and each demonstrated the mechanics; `1` anything
else (a failed trial, an interrupted run); `2` refused. Exit 0 says nothing about whether the value was transferred or the file is right: read the summary.

**Layout** (additive; nothing existing is touched):
```text
$OUT/diag_E3R/e3r-<UTC time>-<id>/        one folder per suite run, created once and never written into by another run
  suite.json       model identity (digest, details), E0/E2 hashes, git commit, host and versions, sampling overrides, task text, config and driver hashes, options
  summary.json     machine-readable aggregate, written when the suite starts and rewritten whenever a trial starts or ends (so even a run killed mid-trial leaves a valid, INCOMPLETE
                   summary in which that trial shows as `running`)
  report.md        the same in words; no transfer values, only short hashes
  trial-01/ .. trial-05/
    trial.json     status, failure class, seed, value hash, timings, the model identity CHECKED FOR THIS TRIAL, and the full record (delegations with task/context as sent, worker
                   tool calls, generations, verdicts, failed checks, file hashes). Written (atomically) as soon as the result exists, before the model is released
    release.json   the outcome of unloading our model after the trial (written afterwards; its absence means the run ended before the release)
    trace.jsonl attempts.jsonl cli.json configs_rendered/   the trial's own trace, attempt record, harness CLI output and exact config
    prompts.local.jsonl                                      only with --capture-prompts; private
```
**Reading the summary.** `report.md` first. Three separate results per trial and in total, each as `count/requested` and `count/completed`: **mechanics demonstrated** (two real delegations, real
worker file calls, reports back), **exact context transfer** (the random value travelled from the reader's report through the orchestrator's delegation to the writer's first prompt, no bypass) and
**task verified** (`result.txt` holds exactly the value, nothing else changed; checked on disk by the runner). A trial that was interrupted, crashed or never ran is in no numerator. `failure_classes`
gives each failed trial one primary label, by root cause: `reader_failed` (the reader never read the value), `reader_report_missing_value`, `value_not_transferred` (the orchestrator did not put the
value in the writer's task or context), `value_altered` (it put a different value), `writer_bypass` (the writer read the source itself), `wrong_result_file`, `no_delegation`,
`writer_not_delegated`, `wrong_delegation_order`, `value_known_before_reader`, `harness_error`, `mechanics_failed`, and the non-results `interrupted`, `runner_error`, `model_identity_changed` and `model_identity_unverifiable` (both: trial `blocked`), `running` (left by a killed run) and `not_run`. A trial's status is one of `completed`, `interrupted`, `runner_error`, `blocked`, `running`, `not_run`; only `completed` trials have verdicts. `complete` in the summary means every requested trial ran to its end; `interrupted` means the command was interrupted (both can be true if Ctrl-C came during cleanup after the last trial). The full failed-check lists stay in
`trial.json`. `latency` has the trial duration (min, median, mean, max, total) and the duration of individual model calls. **Five trials are a smoke test, not a measurement**: one failure moves a rate by 20
points. Do not quote the rates as the model's reliability.

**Pressing Ctrl-C (measured with a real signal against a blocking worker).** A model call that is already running cannot be stopped. After ONE Ctrl-C the command keeps running until that call returns (for a large model this can take minutes), then writes the interrupted state and stops; no further trial starts. A SECOND Ctrl-C makes the interrupted state final on disk immediately (`summary.json`, `trial.json`), after which the process only lingers until the call returns. Neither Ctrl-C damages evidence (every summary and trial file is written atomically, so it is always either the old or the new complete file). Use `kill -9` only as a last resort: the summary left behind still parses and shows the unfinished trial as `running`, but that trial has no `trial.json`. Our Ollama model may then still be loaded: `ollama ps`, `ollama stop <tag>`.

**Seeds.** Trial k uses sampling seed `--seed-start` + k (default 0, so trials use seeds 0 to 4). The transfer value is different in every trial and every suite, but the SEEDS are the same in every suite started with the default. Whether Ollama honours the seed through the harness is not verified, so do not assume that two trials with the same seed and value give the same text. To collect further independent samples for the same model, give a different `--seed-start` (for example 100) instead of repeating the default.

**One suite at a time per Ollama server.** Running two `e3-repeat` suites (or an `e0`/`e1`/`e2`/`e3` run) at once against the same Ollama instance is unsupported: every trial unloads our model afterwards, which would unload the other run's model in the middle of its generation, and two large models compete for the same GPU memory, which also distorts the durations. The folders themselves cannot collide (each suite creates its own), so nothing is overwritten, but the timings and possibly the outcomes of both runs would be wrong. Run the 14B suite to its end before starting the 27B one.

**Package without private prompts.** Prompt capture is off unless you asked for it, and even then `package` leaves every `prompts.local.jsonl` and `*.db` out (it lists them as `NOTE: left out`):
```bash
mkdir -p experiments/rtx4070/results
uv run python experiments/rtx4070/rtx4070.py package --run-dir "$OUT"          # writes experiments/rtx4070/results/<folder name>.tar.gz and .sha256
ARCHIVE=experiments/rtx4070/results/$(basename "$OUT").tar.gz
if tar -tzf "$ARCHIVE" | grep -E 'local\.jsonl|\.db|\.sqlite'; then echo "STOP: private files in the archive"; else echo "OK: none"; fi
```
`package` rewrites the text files of `$OUT` in place with the redaction applied, so run it last. The per-trial traces and `trial.json` contain that trial's random value (it is the audit evidence and
is a test fixture, not a secret); the summary and report do not.

**Reproduce a failed trial.** The trial's seed and value are in its `trial.json`:
```bash
SUITE=$(ls -d "$OUT"/diag_E3R/e3r-* | tail -1)
python3 -c "import json,sys; t=json.load(open(sys.argv[1])); print(t['trial_id'], t['failure_class'], 'seed', t['seed'], 'value', t['record']['expected_value'])" "$SUITE/trial-03/trial.json"
logged replay uv run python experiments/dgx/dgx_diag.py e3-repeat --out "$OUT" --tag "$TAG" --num-ctx "$NUM_CTX" --trials 1 --seed-start SEED --value VALUE [--capture-prompts]
```
This replays the same inputs (value, seed, task, configuration, model) into a NEW suite folder; the original is untouched and the replay is labelled `operator (reproduction)`. Local sampling is not
guaranteed to be bit-identical, so compare the two `trace.jsonl` files instead of expecting the same text. For a closer look, import the trial traces with `experiments/history/history.py` (each
`trial-NN/trace.jsonl` is a normal trace) and read `timeline`.

## 4. Evidence files (all under `$OUT`; every one is redacted when written)

| File | Must contain |
|---|---|
| `run.json` | run id, commit, dirty flag, note that this is a DGX diagnostic |
| `diag_E0/e0.json` | one entry per check with status, detail and raw evidence; `gates_e1_e2` |
| `diag_E1/trace.jsonl`, `diag_E2/trace.jsonl` | `stage_start` (versions, GPUs, Ollama version, model, overrides), then per attempt `attempt_start` (exact task), `generation` (raw text, sent stops, settings, finish reason, reasoning metadata, prompt length and hash, role, delegation id), `generation_consumed` (what effGen kept: `local`, `none`, or `null` with a reason), `tool_call` (only calls that really ran: arguments, result, the text the model saw, role, delegation id, `dispatched`), `attempt_end` or `attempt_aborted` |
| `diag_E1/attempts.jsonl`, `diag_E2/attempts.jsonl` | one record per finished attempt: task, delegations with the full delegated task text, final answer, `actions` (written in text / kept / executed), `verification`, `workspace_changes` (path, size, hash), `invocation_id` |
| `diag_E1/result.json`, `diag_E2/result.json` | per attempt: `mechanics_demonstrated`, `agent_completed`, `task_verified`, failed checks; the Ollama release event |
| `stage_X/cli_X2-*.json` | the harness CLI output of each E2 attempt |
| `configs_rendered/*.yaml` | the exact E2 configuration |
| `environment/ollama_after_E1.json`, `ollama_after_E2.json` | Ollama version, tags, `ps` (placement) after each |
| `diag_E3R/e3r-*/` (`suite.json`, `summary.json`, `report.md`, `trial-NN/`) | written by `e3-repeat` (see its section) |
| `manifest.json`, `summary.md`, `../*.tar.gz` (+ `.sha256`) | written by `package` |

## 5. Independent success criteria

Three different things, always reported separately: **mechanics** (the harness did what it is designed to do), **agent completed** (the agent produced a final answer), and
**task verified** (the runner's own checks passed; the model's words are never evidence). A completed agent with an unverified task is a task failure.

- **E0 passes** when every `gating` entry in `e0.json` is `pass` or `warn`. Read the WARNs: they are the unknowns (GPU placement, host architecture).
- **E1 mechanics**: a `tool_call` for `file_operations` with `path: secret.txt` exists in the trace (written by the tool wrapper, not parsed from text) and its result reached the model.
  **E1 task verified**: all of `secret_read_executed`, `secret_reached_model`, `answer_has_code` pass. The model's answer is checked against a code that exists only in the file.
- **E2 mechanics** (`mechanics_problems` empty): the orchestrator really executed `delegate` (`dispatched: true`, id `d1`, full task text recorded); a worker generated (role `worker`,
  `delegation_id d1`); the worker's `file_operations` read really ran; its result reached the worker model; the delegation report reached the orchestrator;
  `role_split_agrees` is true. **E2 task verified**: `delegate_executed`, `worker_read_secret`, `secret_reached_worker_model`, `answer_has_code` all pass.
- **Evidence integrity** (both): one `run_id`/`stage`/`invocation` on every line and `seq` contiguous; `actions.raw_not_executed` explained (an action written in text is not a call);
  `generation_consumed` present for every generation; the Ollama release `verified: true`; `package` exits 0 and the archive holds no `prompt` key.

## 6. Blockers and unknowns (nothing here has run on the DGX)

1. **The RTX preflight cannot be used on this host** (confirmed in the code): its `arch` check fails on anything but x86_64 (`rtx4070.py:436`), and `nvidia_smi_gpus` raises `ValueError`
   on `[N/A]` memory (`rtx4070.py:400`; reproduced with a simulated nvidia-smi answer), which would also abort the delegation attempt (`stage_runner.py:696`, unguarded). The adapter
   avoids both without editing the RTX package. Whether the GB10 really prints `[N/A]` is **unconfirmed**; E0 records the raw output.
2. **No model is known to be on the DGX.** E1/E2 cannot run until you name an installed tag or approve a pull.
3. **torch on aarch64 + CUDA 13 + GB10 (sm_121)** is a wheel that exists in `uv.lock`; whether its kernels run on this GPU is decided only by `project_env`.
4. **Ollama on unified memory**: `size_vram` accounting is unknown, hence WARN-with-evidence rather than FAIL.
5. **Sandbox**: an Ubuntu 24.04 based OS commonly sets `kernel.apparmor_restrict_unprivileged_userns=1`, which makes the shell tool refuse to build (fail closed). E1/E2 do not need it,
   but record it. Do not relax it for this validation; turning it off affects the whole host and needs a separate decision.
6. **Reasoning models** (the Qwen3-style turn that ends after its thinking) behave differently under effGen (review 004 / audit Part 3). Start with a non-reasoning tag.
7. **Not covered**: bitsandbytes on aarch64, Hugging Face workers, mixed-provider delegation, the shell tool inside a real model run, and any of the earlier DGX failures (no artifacts exist).

## 7. What was verified where

Verified on an x86_64 laptop (RTX 4060, no Ollama): `tests/test_dgx_diag.py` (15 tests; scripted models, the real file tool, the real CLI path for E2); E0 dry run (host/gpu
flagged as not the target, the sandbox and CUDA checks ran for real, Ollama FAIL, E1 refused behind the gate); `uv.lock` contents. **Not verified anywhere**: any command on a
DGX Spark, any real model through `dgx_diag.py`, any Ollama behaviour, the sandbox on the DGX kernel, CUDA kernels on the GB10.
