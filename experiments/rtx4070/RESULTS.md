# What the results contain

Every run creates one folder `experiments/rtx4070/results/<time>-<commit>-<profile>/` and `package` turns it into one archive next to it, `<same name>.tar.gz`, plus a `.sha256` file holding its checksum. **Send back only the `.tar.gz`.**

## Folder layout

| Path | Content |
|---|---|
| `manifest.json` | Run id, times, profile, git commit and whether the tree was dirty, pinned models, per-stage status, every file with size and SHA-256, files left out of the archive, files that needed extra redaction |
| `summary.md` | A table of stages A to E with status and reason, and a list of what stays unverified |
| `run.json` | Run metadata (profile, git state) |
| `environment/` | `nvidia_smi.txt`, `gpu_query.csv` (driver, memory, power), `os.txt`, `python_packages.txt`, `versions.json` (tool versions, locked stack, bitsandbytes pin, selected environment variables such as `OLLAMA_*` and `CUDA_*`; nothing else), `ollama.json` (version, models, effective settings where Ollama reports them) |
| `stage_A/` ... `stage_E/` | `result.json` (status, reason, measurements, and for stages that use Ollama a `residency` list: every unload of our model with free VRAM before and after, and whether Ollama confirmed it), `preflight.json` (checks that gated the stage), `attempts.jsonl` (every finished attempt, written as it finished, so it survives a timeout), `stdout.log` / `stderr.log` (redacted), `cli_*.json` (harness output of each task attempt), GPU and `ollama ps` readings taken during and after the stage |
| `stage_E/` | additionally `repo_diff_attempt*.diff` (the final repository diff) and, in `result.json`, the **independent** pytest result (run by the stage runner itself, not by the model) |

Not collected, ever: model weights (any `.safetensors`, `.gguf`, `.bin` and similar), files over 25 MiB, the disposable workspaces (`work/`), tokens, credentials, your environment variables other than the short whitelist above.

## What the fields mean

- `status` is whether the stage's own criterion was met. `task_success` (stages B, C, D, E) is whether the models **solved the task**; it is recorded separately and never changes the status of stage D, which judges the mechanism.
- Generations are recorded twice: `raw_output` is what the model engine produced, `consumed_output` is that text cut at the stop sequences effGen asked for (what effGen actually kept). Text after, for example, a model-written `Observation:` on a new line is discarded and never reaches the next prompt. `generated_tokens` counts what the engine produced, discarded text included.
- `observations` counts generations with a self-written `Observation:` in the raw text and in the consumed text. Only the second can reach an answer.
- `trajectories` says how many of the attempts produced different text (`distinct` of `attempts`); at a low temperature repeated attempts can be identical, so 3 attempts may be fewer than 3 samples.
- Stage D's `coresidency` says whether both models were **measured** on the GPU together (Ollama still listed and almost fully on the GPU while the Hugging Face model was loaded, the worker's parameters all on `cuda`). If it says not established, nothing is claimed.

## Statuses

`pass` the stage's criterion was met. `fail` it was not, or a required check failed. `skip` a stage it depends on did not pass. `unreadable` / `unknown` the stage died before writing a usable record: its logs are still included. A stage stopped with Ctrl-C is recorded as `fail` with `interrupted: true` and the attempts it had finished. None of these ever counts as a pass.

## Redaction

Everything is redacted when it is written (Hugging Face and GitHub tokens, `sk-` keys, bearer tokens, `key=value` secrets, passwords inside URLs, private-key blocks, your home path and username). These are known credential formats; redaction is pattern matching and cannot recognise a secret in a form it does not know (a model's shell can read any file your Linux user can, and a result can be printed into the logs). `package` then scans every text file again, fixes any that still contain something credential-shaped (listed in `re_redacted_files`), and finally scans the finished archive. If a credential still survives, the archive is deleted and nothing is produced. If you ever see `REFUSING to keep the archive`, tell us.

## Before you send it

Please check what is inside: `tar -tzf <archive>.tar.gz`, and open anything you did not expect (extract it and search for passwords or keys if you used this machine for anything private). To verify it arrived intact, compare `sha256sum <archive>.tar.gz` with the checksum printed by `package`.
