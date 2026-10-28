# DGX Spark validation: E0, E1, E2

**Status: PREPARED, NOT RUN ON A DGX SPARK.** The driver and its tests ran only on an x86_64 laptop with scripted models. Nothing below is hardware-validated until the commands
have run on the DGX and their evidence has been reviewed. What was and was not checked is listed at the end.

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
| `context_transfer.exact` | the reader executed a successful read of `source.txt` whose result showed it the value; the reader was dispatched before the writer; the value is NOT in the first delegation; the report the orchestrator received from the reader contains the value; the value is in the writer delegation's `task` or `context`; the value is in the writer model's FIRST prompt (what the delegation handed it, not its own later output); no writer file call bypassed the orchestrator (below) |
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
