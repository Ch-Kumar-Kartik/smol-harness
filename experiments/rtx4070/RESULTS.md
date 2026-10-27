# What the results contain

Every run creates one folder `experiments/rtx4070/results/<time>-<commit>-<profile>/` and `package` turns it into one archive next to it, `<same name>.tar.gz`, plus a `.sha256` file holding its checksum. **Send back only the `.tar.gz`.**

## Folder layout

| Path | Content |
|---|---|
| `manifest.json` | Run id, times, profile, git commit and whether the tree was dirty, pinned models, per-stage status, every file with size and SHA-256, files left out of the archive, files that needed extra redaction |
| `summary.md` | A table of stages A to E with status and reason, and a list of what stays unverified |
| `run.json` | Run metadata (profile, git state) |
| `environment/` | `nvidia_smi.txt`, `gpu_query.csv` (driver, memory, power), `os.txt`, `python_packages.txt`, `versions.json` (tool versions, locked stack, bitsandbytes pin, selected environment variables such as `OLLAMA_*` and `CUDA_*`; nothing else), `ollama.json` (version, models, effective settings where Ollama reports them) |
| `stage_A/` ... `stage_E/` | `result.json` (status, reason, measurements, and for stages that use Ollama a `residency` list: every unload of our model with free VRAM before and after, and whether Ollama confirmed it), `preflight.json` (checks that gated the stage), `attempts.jsonl` (every finished attempt, written as it finished, so it survives a timeout), `trace.jsonl` (one line per event, written as it happens: the stage header, each attempt's exact task, every model generation with its raw and consumed text, limits, finish reason and reported reasoning metadata, every tool call as a `tool_call_start` line written BEFORE the tool is dispatched (arguments, delegated task text, which agent and which generation asked) followed by a `tool_call` completion line with the same `tool_call_id` (only calls that really ran have one), and `attempt_end` or `attempt_aborted`; prompts are recorded only as length and SHA-256 unless the opt-in `prompts.local.jsonl` below is used), `stdout.log` / `stderr.log` (redacted), `cli_*.json` (harness output of each task attempt), GPU and `ollama ps` readings taken during and after the stage |
| `stage_E/` | additionally `repo_diff_attempt*.diff` (the final repository diff) and, in `result.json`, the **independent** pytest result (run by the stage runner itself, not by the model) |

Not collected, ever: model weights (any `.safetensors`, `.gguf`, `.bin` and similar), files over 25 MiB, the disposable workspace folders themselves (`work/`), tokens, credentials, your environment variables other than the short whitelist above.

**What does reach the results from a workspace:** the workspace folders are never archived, but anything a model read, wrote or ran through a tool is recorded: file contents and shell output appear in the `args`, `result` and `visible` fields of `tool_call` lines in `trace.jsonl`, in the raw and consumed text of `generation` lines (a model quotes what it read), in `cli_*.json`, and (stage E) in `repo_diff_attempt*.diff`. `workspace_changes` in the attempt records lists only the path, kind of change, size and SHA-256 of each file that really changed on disk, never its contents. Redaction is pattern matching on known credential formats, so use only disposable repositories and files without secrets.

## What the fields mean

- `status` is whether the stage's own criterion was met. `task_success` (stages B, C, D, E) is whether the models **solved the task**; it is recorded separately and never changes the status of stage D, which judges the mechanism.
- Generations are recorded twice: `raw_output` is what the model engine produced, `consumed_output` is that text cut at the stop sequences effGen asked for (what effGen actually kept). Text after, for example, a model-written `Observation:` on a new line is discarded and never reaches the next prompt. `generated_tokens` counts what the engine produced, discarded text included.
- In `trace.jsonl` a tool call exists only if the tool's `execute` ran; an `Action:` written in model text is counted separately (`actions`: written in the raw text, in the text effGen kept, and executed). `null` means not reported or not available, never zero. `verification` keeps what the agent reported (`agent_completed`) apart from the runner's own checks (`task_verified`); `workspace_changes` lists path, size and SHA-256 of the files that really changed on disk.
- **Private prompt capture (opt-in).** With `SMOL_CAPTURE_PROMPTS=1` the stage runner (or `--capture-prompts` for `experiments/dgx/dgx_diag.py`) also writes every prompt to `prompts.local.jsonl` in the stage folder: the string effGen passed to the
  model's `generate()` (the adapter input, not the provider's final payload), with its SHA-256, mode 0600, never redacted. It is **not archived and not rewritten by `package`** (it can hold anything a tool read); it is listed under
  `excluded_from_archive`. `experiments/history/` can index it into a local SQLite database, which is equally private.
- Every line of `trace.jsonl` carries `run_id`, `stage`, `invocation` (a fresh id for each run of the stage runner, so running a stage again into the same results folder cannot be mixed up with the first run) and `attempt`; attempt records carry `run_id`, `stage` and `invocation_id`. An attempt is one `(invocation, attempt)` pair: it ended (`attempt_end`), was interrupted (`attempt_aborted`) or has neither (the process was killed).
- A generation says how its stop sequences were handled. `stop_sequences` are the ones SENT to the provider (`stop_handling: provider`: the provider cuts, effGen does not). When none were sent the generation line has `consumed_output: null` and `consumed_status: pending`; a following `generation_consumed` line (same `generation_seq`) settles it as `local` (effGen cut the returned text itself, with `stop_sequences_local`) or `none` (no stops were in force; consumed equals raw). If effGen's handling could not be observed (an interrupted attempt, or no attempt scope) the value stays `null` with the reason in `consumed_status`.
- A `delegate` tool call has `dispatched: false` and `delegation_id: null` when effGen refused it (for example a missing argument) before the tool ran; only dispatched calls own an id (`d1`, `d2`, ...). A tool that raised has `visible: null` and `visible_unavailable` giving the reason: effGen builds that observation itself and the wrapper does not see it.
- `observations` counts generations with a self-written `Observation:` in the raw text and in the consumed text. Only the second can reach an answer.
- `trajectories` says how many of the attempts produced different text (`distinct` of `attempts`); at a low temperature repeated attempts can be identical, so 3 attempts may be fewer than 3 samples.
- Stage D's `coresidency` says whether both models were **measured** on the GPU together (Ollama still listed and almost fully on the GPU while the Hugging Face model was loaded, the worker's parameters all on `cuda`). If it says not established, nothing is claimed.

## Statuses

`pass` the stage's criterion was met. `fail` it was not, or a required check failed. `skip` a stage it depends on did not pass. `unreadable` / `unknown` the stage died before writing a usable record: its logs are still included. A stage stopped with Ctrl-C is recorded as `fail` with `interrupted: true` and the attempts it had finished. None of these ever counts as a pass.

## Redaction

Everything is redacted when it is written (Hugging Face and GitHub tokens, `sk-` keys, bearer tokens, `key=value` secrets, passwords inside URLs, private-key blocks, your home path and username). These are known credential formats; redaction is pattern matching and cannot recognise a secret in a form it does not know (a model's shell can read any file your Linux user can, and a result can be printed into the logs). `package` then scans every text file again, fixes any that still contain something credential-shaped (listed in `re_redacted_files`), and finally scans the finished archive. If a credential still survives, the archive is deleted and nothing is produced. If you ever see `REFUSING to keep the archive`, tell us.

## Before you send it

Please check what is inside: `tar -tzf <archive>.tar.gz`, and open anything you did not expect (extract it and search for passwords or keys if you used this machine for anything private). To verify it arrived intact, compare `sha256sum <archive>.tar.gz` with the checksum printed by `package`.
