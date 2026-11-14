# Runbook: the `rich_3278` Gemma coding-evaluation pilot on a DGX Spark

Audience: a colleague who has not worked on this repository. Follow the steps in order. **Stop at the first failure** (section 10) and send us the artifacts; do not retry, work around a check, or run anything unsandboxed.

Every command block is labelled with the machine it runs on:

- **[DGX]** runs on the DGX Spark (model host).
- **[CONTROLLER]** runs on the controller machine (a separate Linux x86_64 computer, section 2; superseded for the Docker path, see "Choosing your sandbox").
- **[EITHER]** is a read-only command that is only an example of reading a file.

Values in `<ANGLE_BRACKETS>` are placeholders you must supply. Facts marked **Known** come from the repository's source and docs, as of commit `8ba41c5` (check `git log` for anything newer). Facts marked **Observed earlier** were seen on a previous, different experiment (a Qwen3 model, not Gemma) and are given only for comparison; do not assume them.

> **Update 2026-11-13 — the runbook has been used; the Docker sandbox it describes has not.** Three real-model `rich_3278` attempts exist. None produced a non-empty candidate patch and none reached the Docker evaluator's tests; they are real execution attempts, not coding evaluations. Two (2026-11-05) are committed under [`gemma_eval_runs/`](gemma_eval_runs/): `gemma4:e4b` (`ba51`: `attempt_outcome: indeterminate`; one worker timed out; its trace and workspace are missing or zero-byte) and `qwen3.8:27b` (`c25f`: `infrastructure_failure`, a model-call timeout). The third (2026-11-11, `e0cf`, `gemma4:e4b`, in the git-ignored `new_gemma_eval_runs/`) ended `unresolved` with an empty patch after a tool-interface failure (malformed or double-escaped JSON), and its worker prompt contained the reference fix line (since removed). **Two ways of running, documented as they are.** (a) This runbook describes a separate x86_64 *controller* machine that runs the driver and evaluator and reaches the DGX through an SSH tunnel. (b) The three attempts above used the **DGX itself as the controller** with the driver's Docker agent sandbox (`--agent-sandbox docker --agent-image redis:7`), because unprivileged user namespaces are unavailable on the DGX. Path (b) ran only with the **earlier** Docker sandbox form; the current form (narrower mounts, `docker create` then inspect, verify and start, kill and remove in a `finally`) has **never run in full in a container** (system and venv mounts, the Python agent-child; a reduced-fidelity check ran once on x86_64 under Docker Desktop on 2026-11-13, see `experiments/coding_eval/README.md`; the ID-based cleanup has only run against a stubbed `docker`), so DGX-as-controller is **unverified in the new design**. The 2026-11-05 runs used `--context-length 16384 --max-tokens 4096 --temperature 0.1 --seed 0 --max-iterations 8 --agent-timeout-s 1800`; the requested `max-tokens` is not a verified server cap. The driver also has `--tool-calling-mode` (default `react`, recorded in `run.json`); no native or hybrid run has been made against Ollama. Sections that require bubblewrap apply only to the default agent sandbox, and the "do not run the pilot on the DGX" reasoning in section 2 is superseded for the Docker path (see the note there).

---

## 1. Experiment overview

**What smol-harness is.** A small multi-agent coding harness built on effGen 1.3.0. A larger "orchestrator" model delegates work to a smaller "worker" model through one `delegate` tool. Each agent has its own context, model and tool allowlist. Models are served by Ollama (here, on the DGX).

**What this pilot does.** It runs the harness once, on one task, `rich_3278` (a bug fix in the `rich` library, from the Kaggle "Gemma 4 Developer Agent" dataset):

1. The driver unpacks a pinned snapshot of the repository twice: a workspace for the agent and a private pristine copy for the evaluator.
2. The agent (orchestrator plus one worker named `coder`) reads and edits files in its workspace. It has **no shell and cannot run tests**.
3. The driver compares the two trees to produce a candidate patch.
4. An independent evaluator applies the candidate patch plus the official test patch to a fresh pristine copy and runs the task's pinned tests in a disposable Docker container (no network, no mounts).

**What it is meant to validate.** The pipeline end to end with a real model: the sandboxed agent, the connection to the model, candidate collection, the evaluator, and the way outcomes are classified (infrastructure failure, timeout, indeterminate, resolved, unresolved).

**What it cannot establish.**

- It is **one task, one run, one seed**. It is a smoke test, not a benchmark and not evidence of general coding ability.
- The task statement is a pull-request description that **quotes the intended fix**, so a pass is weak evidence of skill.
- The snapshot is not the task's declared base commit (that commit is absent from the snapshot). Results are labelled `snapshot-based`, never "base-commit-exact".
- The prompt is a minimal reconstruction, not the competition's official prompt, and the agent has different tools from the competition agent. The deviations are listed in `experiments/coding_eval/README.md` and written into every `run.json`.
- The requested `--max-tokens` is **not verified to be the limit the server applies** (open item U17 in `docs/new-session.md`).
- A completed agent run does **not** mean the task was solved. Only the evaluator decides `resolved`.

---

## 2. Execution architecture

> **Superseded for the Docker path:** the controller may also be the DGX itself, running the driver with `--agent-sandbox docker`; see "Choosing your sandbox". That path is unverified in the current design.

```
        CONTROLLER machine (Linux x86_64)                          DGX Spark (Linux ARM64)
  ┌─────────────────────────────────────────┐                ┌──────────────────────────────┐
  │ pilot driver  (rich3278_pilot.py)       │                │ Ollama (listens on 127.0.0.1  │
  │ ├─ sandbox: systemd + bubblewrap        │                │   :11434 only)                │
  │ │   + prlimit (default), or Docker      │                │ Gemma model (weights, GPU)    │
  │ │   (--agent-sandbox docker)            │                │ inference                     │
  │ │       │ no network; unix-socket       │                │                               │
  │ │       ▼ bridge to ONE host:port       │   SSH local    │                               │
  │ ├─ ModelBridge ── 127.0.0.1:11434 ──────┼── forwarding ──┼─► 127.0.0.1:11434             │
  │ ├─ evaluator (local Docker container)   │  ssh -N -L ... │                               │
  │ └─ dataset/ (snapshot, test patch,      │                │                               │
  │      reference assets)                  │                │                               │
  └─────────────────────────────────────────┘                └──────────────────────────────┘
```

**DGX Spark** hosts Ollama, stores and serves the chosen Gemma model, and does all inference. Ollama stays bound to the DGX's loopback interface. Nothing else from this repository needs to run on the DGX.

**Controller machine** runs the driver, the agent sandbox, the ModelBridge and the evaluator. It holds the dataset and every evaluation-only asset (test patch, reference patch). It reaches the DGX only through an SSH local port forward.

### Why the pilot does not run on the DGX itself (read this)

> **Superseded for the Docker path:** the controller may also be the DGX itself, running the driver with `--agent-sandbox docker`; see "Choosing your sandbox". That path is unverified in the current design. The reasons below apply to the bubblewrap sandbox and the original laptop.

**Known:** the pilot was built and validated on an **x86_64 laptop** (Linux 6.18 LTS, Docker Desktop, bubblewrap 0.13.0, a systemd user session). The DGX is **ARM64**. Three things make the DGX an unverified place to run the controller, so do not run the pilot there:

1. **The agent sandbox is unverified on the DGX.** It needs unprivileged user namespaces. On the DGX, the earlier shell-sandbox probe did not pass its guarantees: **Observed earlier** (E0, 2026-10-30): the sandbox check failed and the notes record an AppArmor restriction on unprivileged user namespaces (Ubuntu-based DGX OS). The pilot's bubblewrap sandbox would be blocked by the same mechanism. The driver refuses to start without its sandbox and never falls back, so the run would simply be refused. **Superseded for the Docker path:** since 2026-11-06 `--agent-sandbox docker` supplies a container boundary instead; the three attempts ran in the earlier form of it on the DGX, and the full current form has not been run. Points 2 and 3 (the evaluator's architecture-specific Python and an ARM64 reference check) still apply: no run produced a candidate, so the evaluator's tests did not execute on ARM64.
2. **The evaluator's Python is architecture-specific.** The driver's default `--python-dir` is `~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu` (x86_64). That interpreter is copied into the evaluator container, whose image architecture must match.
3. **ARM64 Docker evaluation was never run.** The baseline (16 failed / 7 passed) and gold (23 passed) reference results were reproduced only on the x86_64 laptop.

**If you must use a different controller machine**, it needs everything in section 3.2. What remains **unverified** for any machine other than the original laptop: macOS, Windows or WSL2 as a controller; ARM64 as a controller; Ubuntu 24.04 with default AppArmor settings; a Docker engine other than Docker Desktop. Run the sandbox preflight (section 5) and the reference check (section 6) on that machine before anything else. If either fails, **stop and tell us**. Do not relax system security settings (such as AppArmor or sysctl restrictions) on your own to make them pass.

The previous `experiments/dgx/dgx_diag.py` tool is **not used** in this runbook. It depends on code under `experiments/rtx4070/` (a package for a machine that is no longer used) and is not needed here.

---

## 3. Prerequisites

### 3.1 DGX Spark

| Need | Detail |
|---|---|
| NVIDIA driver and GPU | `nvidia-smi` must show the GPU. **Observed earlier:** GB10 with driver 580.173.02, CUDA 13.0; on this unified-memory machine the memory columns of `nvidia-smi` showed `N/A`. |
| Ollama (ARM64 build), running as a service | **Observed earlier:** Ollama 0.35.1. Any version change must be recorded. |
| The Gemma model installed locally | **Required pre-run decision (section 4.1). We have not chosen a tag.** |
| Memory and disk | Enough free memory for the chosen model plus its 8192-token context (the GPU shares system memory), and enough disk for the model files. **Observed earlier:** 121.7 GiB total memory and 1180 GiB free disk on one DGX. Check yours; we have not measured a Gemma model's footprint. |
| Local Ollama API | `http://127.0.0.1:11434` on the DGX itself. |
| Nobody else using this Ollama during the run | Section 7.5. |

### 3.2 Controller machine

> **Superseded for the Docker path:** the controller may also be the DGX itself, running the driver with `--agent-sandbox docker`; see "Choosing your sandbox". That path is unverified in the current design. The `--python-dir` default below is an x86_64 interpreter; on aarch64 see "Choosing your sandbox".

| Need | Detail |
|---|---|
| OS and architecture | **Validated:** Linux x86_64 (default sandbox) and Linux aarch64 on a DGX Spark (the earlier form of the Docker agent sandbox ran there, 2026-11-05; the current form has not). |
| `uv` and Python 3.12 | The project pins Python `>=3.12,<3.13` (`pyproject.toml`, `.python-version`). `uv` 0.11.29 was used. |
| Project environment | `uv sync --locked` (multi-gigabyte download: it installs PyTorch and NVIDIA wheels because the lock file includes them; the pilot itself does not use a GPU on the controller). |
| `git` | To check out the right revision. |
| `bubblewrap` (`bwrap`) | **Validated:** 0.13.0. Needed only for the default (`--agent-sandbox bwrap`) agent sandbox. |
| `systemd-run` with a user session | `systemd-run --user --scope` must work. The driver passes `XDG_RUNTIME_DIR` and `DBUS_SESSION_BUS_ADDRESS` to it. **Validated:** systemd 262. Default sandbox only. |
| cgroup v2 with `cpu`, `memory` and `pids` delegated to your user | **Validated:** the user manager's `cgroup.controllers` listed `cpu memory pids`. Default sandbox only. |
| `prlimit` (util-linux) | **Validated:** 2.42.4. Default sandbox only. |
| GNU `patch` | **Validated:** 2.8. |
| Unprivileged user namespaces allowed | Section 5 tests this with the real sandbox. Default sandbox only; the Docker path exists for hosts where namespaces are blocked. |
| Docker engine usable by your user | The evaluator runs in a container started with `docker`. **Validated:** Docker 29.8.1 (Docker Desktop, x86_64); the earlier form of the Docker **agent** sandbox also ran on the DGX (aarch64) in 2026-11-05, but the evaluator container did not (empty patches). The full current agent-sandbox form has not run in a container (only the reduced-fidelity check above). |
| A **local** Docker image `redis:7` | Used only because it provides `bash` and `timeout`. **The driver never pulls an image.** `--image` can name another local Debian-based image, but only `redis:7` was validated. The Docker agent sandbox uses `--agent-image` (same default). |
| A standalone CPython 3.12 matching the controller's architecture | Default location `~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu`. On a host with a usable system Python, `--python-dir /usr` mounts `/usr` (and `/lib`, `/lib64`) read-only instead of copying a standalone interpreter. This system-Python mode has not been exercised by a run. See "Choosing your sandbox" for the evaluator's Python on another architecture. |
| The dataset (below) | `dataset/` in the repository root (git-ignored), or `--dataset-dir`. |
| SSH access to the DGX | A user account on the DGX you can log in with. |
| Free disk | The dataset is about 21 GiB (per the note in `.gitignore`). The run itself needs far less: the extracted `rich` tree is small, and a scratch extraction measured about 94 MB with its `.git` directory. As an unmeasured margin, keep a few GiB free in the output directory's filesystem and in Docker's storage. |

### 3.3 The dataset

The repository does **not** distribute the dataset, and we have not created a distribution mechanism. It is the Kaggle "Gemma 4 Developer Agent" competition dataset (the exact Kaggle slug is not recorded in this repository). Obtain it from Kaggle under the competition's terms. You will need to get it from us or from Kaggle with your own access.

The driver expects this layout (from the source, `--dataset-dir` defaults to `<repo>/dataset`):

```
<dataset>/tasks.jsonl                  # one JSON object per task; read for task id rich_3278
<dataset>/snapshots/rich_3278.tgz      # the repository snapshot
<dataset>/wheels/                      # wheel files used by prepare-env
```

**Do not copy `dataset/`, the test patch or the reference patch to the DGX, to a shared location, or into the artifacts you send back.** The agent never sees them; the driver reads them only on the controller.

---

## 4. DGX setup and model verification

All commands in this section run **on the DGX**.

### 4.1 Required pre-run decisions (fill these in; do not guess)

The repository contains **no** fixed Gemma tag for a new run (the 2026-11-05 run used `gemma4:e4b`). Decide the following with us before starting, and write the answers in the handoff record (section 11):

| Decision | Value (fill in) | Notes |
|---|---|---|
| Ollama model tag, exactly as shown in the `NAME` column of `ollama list` | `<MODEL_TAG>` | Passed to `--model`. The driver asks the server for this exact name and refuses to start if it is not listed. A tag whose prefix before `:` is one of `openai`, `openai_compatible`, `openai-compatible`, `openai_compat`, `oai_compatible`, `compatible`, `server`, `vllm_server`, `local_server` is rejected by the harness. A Gemma tag is not affected. |
| Quantization of that tag | `<QUANT>` | Fixed by the tag; recorded as `quantization_level` in `run.json`. |
| Model digest | `<DIGEST>` | Read it in 4.4. The driver **records** it and does not compare it with anything; you do the comparison. |
| Context window | `8192` | Must equal `--context-length` and the server's own `num_ctx` (4.2). |
| Output-token cap in the Modelfile (`num_predict`), if any | `<NUM_PREDICT or "none">` | **Unresolved on our side:** the harness requests `--max-tokens 768`, but which limit Ollama really applies is not verified (U17). The earlier Qwen3 tag set `num_predict 1024`. Record what you set. |
| Is the Gemma base model already installed on the DGX? | yes / no | If **no**, pulling it is a download that we have not decided or approved. Ask us first. |

The 2026-11-05 runs used `gemma4:e4b` (8.0B, Q4_K_M, digest `c6eb396d…`) and `qwen3.8:27b` (27.3B, Q4_K_M, digest `22130167…`), both with `--context-length 16384 --max-tokens 4096`, not the 8192/768 in the examples below. They are data points, not a recommendation.

### 4.2 Create or confirm a tag with a fixed context window

The harness cannot see Ollama's server-side context window. If the server window is smaller than the configured `--context-length`, Ollama silently truncates prompts. So the tag you run must fix `num_ctx` to 8192.

If the installed tag does not already fix it, derive a local tag (no download; `<BASE_TAG>` is a tag already installed):

```bash
# [DGX]
cat > /tmp/Modelfile.gemma-pilot <<'EOF'
FROM <BASE_TAG>
PARAMETER num_ctx 8192
EOF
# add a "PARAMETER num_predict <N>" line only if section 4.1 decided on one
ollama create <MODEL_TAG> -f /tmp/Modelfile.gemma-pilot
```

### 4.3 Hardware, service and model checks

```bash
# [DGX] host, GPU and driver
uname -srm; cat /etc/os-release | head -3
nvidia-smi

# [DGX] Ollama version and service
ollama --version
systemctl status ollama --no-pager

# [DGX] locally available models (copy the exact NAME of the Gemma tag)
ollama list
```

### 4.4 Model identity, metadata and digest

```bash
# [DGX] metadata, parameters and Modelfile of the exact tag
ollama show <MODEL_TAG>
ollama show <MODEL_TAG> --modelfile

# [DGX] the digest, size and quantization as the HTTP API reports them (this is what the driver records)
curl -s http://127.0.0.1:11434/api/tags | python3 -m json.tool

# [DGX] confirm the server's context window for the tag: look for a line "num_ctx    8192"
curl -s http://127.0.0.1:11434/api/show -d '{"model":"<MODEL_TAG>"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('parameters')); print(d.get('details'))"
```

Write down the model's `digest` from `/api/tags`. Confirm `num_ctx 8192` appears. If it does not, return to 4.2.

### 4.5 One small generation, GPU residency and resource use

This is the only inference you do before the pilot. It uses the exact tag and 16 output tokens.

```bash
# [DGX]
curl -s http://127.0.0.1:11434/api/generate -d '{"model":"<MODEL_TAG>","prompt":"Reply with one word.","stream":false,"options":{"num_predict":16}}' | python3 -m json.tool

# [DGX] where the model is loaded: the PROCESSOR column should read 100% GPU; note the CONTEXT column if shown
ollama ps
nvidia-smi
free -h
df -h ~
```

If the model is not fully on the GPU, or the generation fails, stop (section 10).

### 4.6 Keep Ollama on localhost; why we use SSH forwarding

Ollama's API has no authentication and includes management calls (delete, pull, create). It must not be reachable from the network. Check what it listens on:

```bash
# [DGX] the listening address must be 127.0.0.1:11434, not 0.0.0.0 or [::]
ss -ltnp | grep 11434
# [DGX] an OLLAMA_HOST override would show up here
systemctl show ollama --property=Environment
```

If it listens on anything other than loopback, stop and ask the DGX administrator; do not run the experiment against an exposed instance. The controller reaches Ollama through an SSH tunnel (section 7), which keeps the API on the DGX's loopback interface.

---

## 5. Controller-machine setup

All commands in this section run **on the controller machine**, from a clean directory of your choice (for the DGX as controller with the Docker sandbox, see "Choosing your sandbox" and the experiment sequence).

### 5.1 Get the code at the right revision

```bash
# [CONTROLLER]
git clone https://github.com/Ch-Kumar-Kartik/smol-harness.git
cd smol-harness
git checkout <COMMIT_WE_GIVE_YOU>        # the commit we give you; `0f17d7c`, `4067852`, `7ecb2b1`, `4fde05e` and `8ba41c5` are on origin/main (the last two were pushed together with the documentation commit that records them; check `git log origin/main`)
git rev-parse HEAD
git status --short                        # must print nothing (clean tree); record any output
git log --oneline -1 -- experiments/coding_eval/rich3278_pilot.py     # the commit that last changed the driver
sha256sum experiments/coding_eval/rich3278_pilot.py
```

**Known:** the driver's SHA-256 at commit `8ba41c5f3abb0d901f55f379bdecf319867dcd06` is `71b0c8d5b7682851e6872c2dd3e993985c7f84439d113171c67b549c05c5557a`. A different value means a different driver (for example a newer commit); record it and tell us. The driver also records its own hash in `run.json`.

### 5.2 Install the project

```bash
# [CONTROLLER]
uv sync --locked
uv run python -c "import effgen; print(effgen.__version__)"     # expected: 1.3.0
```

### 5.3 Check the sandbox prerequisites

```bash
# [CONTROLLER]
bwrap --version
systemd-run --version | head -1
prlimit --version
patch --version | head -1
stat -fc %T /sys/fs/cgroup                                                        # expected: cgroup2fs
cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/cgroup.controllers   # expected to include: cpu memory pids
systemd-run --user --scope --quiet true && echo "user scope OK"
```

The authoritative check is the driver's own sandbox preflight. It builds the real sandbox, confirms the network namespace contains only a loopback interface, blocks an external connection, and confirms the cgroup limits are applied as configured. It does not touch the dataset, Docker or any model:

```bash
# [CONTROLLER] expected output: sandbox OK
uv run python -c "import sys; sys.path.insert(0,'experiments/coding_eval'); import rich3278_pilot as p; print(p.sandbox_problem() or 'sandbox OK')"
```

If it prints anything else, **stop** (section 10). Do not edit the driver and do not run the agent outside the sandbox.

**If you use `--agent-sandbox docker` instead, this preflight does not apply** (the driver does not preflight Docker either). Verify the engine and the local image yourself, for example `docker run --rm redis:7 true`, and read the `sandbox` block in `run.json` afterwards.

Optional but recommended: run the driver's own tests. They use stand-in servers (no model, no real dataset) and include the real-sandbox tests. The full default suite was `1572 passed, 9 skipped` (the last full run, code as committed in `8ba41c5`; not re-run for documentation edits; the skips are opt-in real-model and real-container tests); the count for this file alone was not re-measured. Any sandbox failure on your machine is a stop.

```bash
# [CONTROLLER] about a minute and a half on the original laptop
uv run pytest tests/test_coding_eval_pilot.py -q
```

### 5.4 Evaluator prerequisites: standalone Python and the Docker image

```bash
# [CONTROLLER] the default --python-dir must exist; it must be a standalone CPython 3.12
ls ~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3
# [CONTROLLER] the local image must exist (the driver never pulls)
docker image inspect redis:7 --format '{{.Id}} {{.Architecture}} {{.Os}}'
docker info --format '{{.OperatingSystem}} {{.Architecture}}'
```

- If the Python path does not exist, `uv python install 3.12` installs uv's managed CPython (a download), or pass `--python-dir <dir>` to every command below, pointing at a standalone CPython 3.12 of the same architecture as the Docker image.
- If the image is missing, the driver will refuse. `docker pull redis:7` downloads a public image; this is a download we have not run for you, so do it only if you are comfortable, then re-check.

### 5.5 Verify the dataset and the `rich_3278` snapshot

```bash
# [CONTROLLER] (run from the repository root; use your --dataset-dir path if it is elsewhere)
test -f dataset/tasks.jsonl && test -f dataset/snapshots/rich_3278.tgz && test -d dataset/wheels && echo "layout OK"
sha256sum dataset/snapshots/rich_3278.tgz
```

**Known:** the pinned value is `bda96e09e9e02fa372b2cf6c5f3f2760f97d292fb531cc631564632940d6a453` (`SNAPSHOT_SHA256` in the driver). The driver refuses to run on any other file. If it differs, stop.

### 5.6 Output directory and space

Use a directory outside the repository and outside the dataset (the driver refuses otherwise). `/tmp` is supported. All of `prepare-env`, `reference-check` and `run` must use the **same** `--out`, because `run` reads the evaluator's Python packages from `<out>/env/site`.

```bash
# [CONTROLLER] bash assumed
export OUT=/tmp/rich-pilot-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT/console"
echo "$OUT"
df -h "$OUT" ~ .
docker system df
```

If `/tmp` is a memory-backed filesystem (`df` shows `tmpfs`), its contents are lost on reboot and the space counts against memory. Copy artifacts out before any reboot or cleanup (section 11). Keep `$OUT` in your shell for the rest of this runbook; if you open a new terminal, set it again to the same path.

---

## 6. Preflight and reference checks (no agent, no model)

These steps prove the evaluator works **before** any model is called. The agent phase can take up to 30 minutes and evaluation happens after it, so a broken evaluator must be found now. Both commands run on the **controller**, from the repository root.

The gold patch and the official test patch are used **only** by the evaluator, on the controller. The agent never receives them.

### 6.1 `prepare-env` (offline)

Installs the test dependencies (`pytest==8.3.4`, `pygments`, `markdown-it-py`) from the dataset's wheels into `<out>/env/site`. It runs `uv --no-config pip install --offline --no-index`, so it uses no network.

```bash
# [CONTROLLER]
uv run python experiments/coding_eval/rich3278_pilot.py prepare-env --out "$OUT" 2>&1 | tee "$OUT/console/prepare-env.txt"
```

It prints JSON with a `packages` list. A non-zero exit or "offline dependency install failed" means stop.

### 6.2 `reference-check` (Docker evaluator)

Runs two evaluations in the disposable container: the baseline (official test patch only) and the gold patch.

```bash
# [CONTROLLER]
uv run python experiments/coding_eval/rich3278_pilot.py reference-check --out "$OUT" 2>&1 | tee "$OUT/console/reference-check.txt"
```

**Expected (Known, reproduced on the original laptop on 2026-11-03 and 2026-11-04):** two JSON lines.

| Label | `outcome` | `resolved` | `counts` |
|---|---|---|---|
| `baseline` | `unresolved_tests_failed` | `false` | `passed` 7, `failed` 16, `errors` 0, `pytest_exit` 1 |
| `gold` | `resolved` | `true` | `passed` 23, `failed` 0, `errors` 0, `pytest_exit` 0 |

**Stop** if either differs, if the command exits non-zero, or if you see `"outcome": "error"`. A different count means the evaluator environment does not reproduce the task and a pilot result would be meaningless. Send us `$OUT/console/` and the `baseline-*`/`gold-*` files in `$OUT`.

---

## 7. SSH tunnel

The tunnel runs on the **controller**, in its **own terminal window**, in the foreground. Keep that terminal open for the whole run.

### 7.1 Start it

```bash
# [CONTROLLER] terminal A. Replace the placeholders; omit "-p <SSH_PORT>" if the DGX uses SSH port 22.
ssh -N -L 11434:127.0.0.1:11434 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -p <SSH_PORT> <DGX_USER>@<DGX_HOST>
```

`-N` forwards without a shell. The first `11434` is the port on the controller; `127.0.0.1:11434` is Ollama as seen from the DGX. The two `-o` options are optional standard OpenSSH options: fail instead of silently running without the forward, and keep the connection alive.

If port 11434 is already in use on the controller (for example a local Ollama), pick another local port, such as `-L 11435:127.0.0.1:11434`, and pass `--base-url http://127.0.0.1:11435/v1` to the run command in section 8, and use that port in the checks below. Do not stop another person's service to free the port.

### 7.2 Verify the endpoint exists locally and belongs to the tunnel

```bash
# [CONTROLLER] terminal B. Expected: the listener belongs to ssh, not to a local ollama
ss -ltnp | grep 11434
```

### 7.3 Verify it reaches the intended Ollama server

```bash
# [CONTROLLER] must report the version you saw in 4.3
curl -s http://127.0.0.1:11434/api/version
# [CONTROLLER] the Gemma tag must be listed, with the same digest you wrote down in 4.4
curl -s http://127.0.0.1:11434/api/tags | python3 -m json.tool
```

If the tag is missing or the digest differs from section 4.4, **stop** (section 10). The driver also performs a metadata-only lookup of `/api/tags` before the agent starts and refuses to run if the tag is not listed, but it does not compare digests.

### 7.4 Close it

Press `Ctrl-C` in terminal A after the run (section 11). Then check that nothing still listens:

```bash
# [CONTROLLER] expected: no output
ss -ltnp | grep 11434
```

### 7.5 Confirm nobody else is using this Ollama

The pilot measures one model on one server. Another user's requests would change timing and could evict the model. Before starting, confirm with the DGX's other users that nobody will use this Ollama instance until you finish, and check:

```bash
# [DGX] before starting: only your model (or nothing) should be listed; no other GPU processes
ollama ps
nvidia-smi
```

---

## 8. Run the pilot (exactly once)

Prerequisites: sections 4.4, 5.3, 5.5 and 6 passed; the tunnel (7.1) is up and 7.3 passed; 7.5 confirmed. Set the model tag you decided in 4.1:

```bash
# [CONTROLLER] same terminal where $OUT is set
export MODEL_TAG="<MODEL_TAG>"
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$OUT/console/run-start-utc.txt"
```

```bash
# [CONTROLLER] bash assumed. This is the only command that calls the model.
uv run python experiments/coding_eval/rich3278_pilot.py run \
  --out "$OUT" \
  --model "$MODEL_TAG" \
  --context-length 8192 \
  --base-url http://127.0.0.1:11434/v1 \
  --max-tokens 768 \
  --temperature 0.1 \
  --seed 0 \
  --max-iterations 8 \
  --agent-timeout-s 1800 \
  --keep-workspaces \
  --allow-model-call 2>&1 | tee "$OUT/console/run.txt"
echo "driver exit status: ${PIPESTATUS[0]}" | tee "$OUT/console/run-exit.txt"
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$OUT/console/run-end-utc.txt"
```

All flags were checked against the driver's `--help` and `build_parser()`. Defaults, **Known:** `--provider` is `ollama` (the only choice), `--base-url` defaults to `http://127.0.0.1:11434/v1`, `--max-iterations` to 8, `--task-id` to `rich_3278`, `--image` to `redis:7`, `--site-dir` to `<out>/env/site`. The per-worker timeout is fixed at 300 s in the driver. `--keep-workspaces` keeps the agent's workspace and the pristine copy under the run directory, which you want for the first run. `--allow-model-call` is required; without it the driver refuses.

**Agent sandbox choice.** The default is the systemd + bubblewrap + prlimit sandbox (section 5.3). Add `--agent-sandbox docker` (optionally `--agent-image <local image>`, default `redis:7`) on a host where unprivileged user namespaces are unavailable. The three real attempts used the **earlier** form of this path; **the full current configuration (system `/usr`, `/lib`, `/lib64` mounts, the venv mount, the Python agent-child) has never run in a container**, and the ID-based cleanup in `4067852` has only run against a stubbed `docker`; a reduced-fidelity check of earlier code (image `redis:7`, `sleep` instead of the Python child, no system or venv mounts) ran once on x86_64 under Docker Desktop on 2026-11-13, and says nothing about arm64, the DGX interpreter location or any real model. In the current form the repository root and `/python` are not mounted; the mounts are `<run>/agent` and the bridge directory (writable) and `src/`, the driver file, the venv, `/usr`, `/lib` and `/lib64` (read-only), as listed by `docker_agent_command`. The driver runs `docker create` (container `smol-agent-<run id>`, `--init`, `--memory-swap` equal to `--memory`, tmpfs `/tmp` and `/data`, `-w /agent`), saves `docker inspect` to `<run>/agent_container_inspect.json`, verifies it, then runs `docker start -a`; the container is killed and removed in a `finally`. Cleanup runs only after this run's `docker create` succeeded and targets the container ID create returned; a stale container with the same name makes create fail instead of being removed. **Accepted risk:** the mounted venv's `rich` package contains the reference fix, readable by code that runs inside the container (the agent's tools cannot read it). **Interpreter handling (commit `4fde05e`):** if the venv's interpreter resolves under `/usr`, `/lib` or `/lib64`, nothing extra is mounted; otherwise the resolved `sys.base_prefix` is mounted read-only at the path the venv uses (which can be an alias directory), with early refusals for unsafe base prefixes (`agent_interpreter_mount`, `rich3278_pilot.py:832`, `:931-936`); this alias-path behaviour is **UNVERIFIED** in a real container, and the no-model `sandbox-check` (X1) exercises it. See "Choosing your sandbox". The evaluator's `--image` is separate and unchanged. `--tool-calling-mode {auto,native,react,hybrid}` (default `react`) is recorded in `run.json`; use the default unless we say otherwise.

- Run it **once**. Do not add retries, loops or a second seed.
- It can take up to the agent timeout (30 minutes) plus evaluation (up to 5 minutes). Do not interrupt it. If you must, press `Ctrl-C` once and record it.
- If you used another local tunnel port, change `--base-url` to match.
- Do not add `--attempts`-style flags; the driver has none.

**Exit status:** 0 if the run completed (`status` is `complete`), 1 if it did not complete (for example `refused`), 2 if the command line itself was refused (for example a missing `--allow-model-call` or a bad `--out`). **Exit 0 does not mean the task was solved.**

On completion the driver prints one JSON line with `run_id`, `status`, `resolved`, `outcome`, `attempt_outcome`, `infrastructure_failure` and `counts_as_coding_failure`. Everything else is in `$OUT/<run_id>/`.

---

## 9. Post-run evidence collection

Set the run directory (the `run_id` is in the JSON line above, and also the only `rich_3278-*` directory in `$OUT`):

```bash
# [CONTROLLER]
export RUN="$OUT/$(ls "$OUT" | grep '^rich_3278-' | head -1)"; echo "$RUN"; ls "$RUN" "$RUN/agent" "$RUN/evaluator"
```

### 9.1 Files

| Path under `$RUN` | What it is |
|---|---|
| `run.json` | Everything about the run: configuration, hashes, provenance, model identity, sandbox, agent phase, candidate, evaluation, classification. **Read this first.** |
| `candidate.patch` | The change the agent made, as a unified diff (protected files are left out and listed in `run.json`). |
| `evaluation.json` | The evaluator's record: patch application, `pytest_exit`, counts, output tail, problems. |
| `evaluator/eval/logs/candidate.stdout.txt`, `candidate.stderr.txt` | Full evaluator container output. |
| `agent/agent_result.json` | The child process's result: the agent's final output, `agent_reported_success`, `termination`, `model_failure`, per-worker records. |
| `agent/history/<id>/trace.jsonl` | One line per event: model calls (finish reason, tokens), tool calls, delegations. |
| `agent/child.stdout.txt`, `agent/child.stderr.txt` | The agent child's console output. |
| `agent/harness.yaml`, `agent/prompt.txt`, `agent/params.json` | The exact rendered configuration, prompt and parameters given to the agent. |
| `agent/workspace/`, `evaluator/pristine/` | Kept by `--keep-workspaces`: the agent's final workspace and the untouched copy. |

### 9.2 Inspection order (do it in this order)

#### Step 1: the classification and the status

A completed agent response does not mean success; the classification says what the outcome means.

```bash
# [CONTROLLER]
python3 - <<'EOF'
import json, os
r = json.load(open(os.path.join(os.environ["RUN"], "run.json")))
print("status:", r.get("status"), "| error:", r.get("error"))
print("resolved (evaluator):", r.get("resolved"))
print("classification:", json.dumps(r.get("classification"), indent=1))
EOF
```

`attempt_outcome` is one of `resolved`, `unresolved`, `timed_out`, `indeterminate`, `infrastructure_failure`. Only `unresolved` has `counts_as_coding_failure: true`. See `experiments/coding_eval/README.md` ("Outcome classes") for the exact rules.

#### Step 2: model identity, sandbox status and provenance

```bash
# [CONTROLLER]
python3 - <<'EOF'
import json, os
r = json.load(open(os.path.join(os.environ["RUN"], "run.json")))
print("model_identity:", json.dumps(r.get("model_identity"), indent=1))
print("sandbox:", json.dumps(r.get("sandbox"), indent=1))
print("basis:", json.dumps(r.get("basis"), indent=1))
p = r.get("provenance") or {}
print("provenance:", {k: p.get(k) for k in ("driver_sha256", "repo_head", "repo_status", "python", "platform", "effgen")})
print("snapshot_sha256:", r.get("snapshot_sha256"))
print("config:", json.dumps(r.get("config"), indent=1))
EOF
```

`model_identity.digest` must equal the digest you recorded in section 4.4. `sandbox.kind` should read `systemd scope + bubblewrap + prlimit` for the default sandbox, or `Docker` with `--agent-sandbox docker`. For the Docker sandbox, `sandbox.limits` holds the Docker flag values, `sandbox.isolation` summarizes the `docker inspect` result, `agent.command` is the `docker create` argv; `agent.start_command` (a separate key next to it) is the `docker start -a <name>` argv; `agent.container` holds `name`, `id`, `inspect_file`, `isolation` and `cleanup`. Records from the 2026-11-05 runs have the older shape.

#### Step 3: agent phase, termination and failure metadata

```bash
# [CONTROLLER]
python3 - <<'EOF'
import json, os
r = json.load(open(os.path.join(os.environ["RUN"], "run.json")))
a = r.get("agent") or {}
print("exit_code:", a.get("exit_code"), "| timed_out:", a.get("timed_out"), "| seconds:", a.get("seconds"))
res = a.get("result") or {}
print("agent_reported_success:", res.get("agent_reported_success"), "| termination:", res.get("termination"), "| error:", res.get("error"))
print("model_failure:", json.dumps(res.get("model_failure"), indent=1))
print("workers:", json.dumps(res.get("workers"), indent=1))
print("final output:", str(res.get("output"))[:1500])
EOF
```

`termination` is one of `done`, `not_possible`, `stuck`, `tool_failed`, `error`. `agent_reported_success: true` only means the orchestrator wrote a final answer. Check `workers[*].status` and `termination` for a worker that did not complete.

#### Step 4: the trace

How the agent actually behaved: did it delegate, which tools did the worker call, did a generation stop on length.

```bash
# [CONTROLLER]
ls "$RUN/agent/history"/*/
head -c 3000 "$RUN/agent/history"/*/trace.jsonl
wc -l "$RUN/agent/history"/*/trace.jsonl
```

#### Step 5: `candidate.patch`, then the evaluator output

```bash
# [CONTROLLER]
cat "$RUN/candidate.patch"
python3 -c "import json,os; e=json.load(open(os.path.join(os.environ['RUN'],'evaluation.json'))); print(e.get('outcome'), e.get('problems')); print(e.get('steps')); r=e.get('result') or {}; print({k:r.get(k) for k in ('passed','failed','errors','pytest_exit','timed_out','oom_killed','exit_code')}); print(r.get('output_tail'))"
```

#### Step 6: the DGX side (after the run, before closing anything)

```bash
# [DGX]
ollama ps
nvidia-smi
journalctl -u ollama --since "<run start time, e.g. '2026-11-07 10:00:00'>" --no-pager | tail -50
```

An empty `candidate.patch` means the agent changed nothing the driver counts as a candidate; read the trace to see why (for example a generation that ended on `length` with no visible output).

---

## 10. Failure handling

**General rules:** stop at the first failure. Do not retry. Do not run anything unsandboxed or bypass a refusal. Keep every artifact (`$OUT` and the console logs). Diagnose the first failure, not the later ones.

| Symptom | Likely meaning | What to do |
|---|---|---|
| Model not listed, or the digest differs from section 4.4 | Wrong tag, a different model behind the name, or the wrong server on the tunnel | Stop. Re-check 4.3 and 7.3. Do not run. Tell us the tag and both digests. |
| Driver prints `refused: ... model identity could not be established` (status `refused`, `attempt_outcome` `infrastructure_failure`) | The endpoint was unreachable or does not list `--model` | Check the tunnel (7.2, 7.3) and the exact tag spelling. Nothing was sent to the model. |
| Connection refused / `ssh` exits / forward fails | The tunnel is down, the local port is in use, or the DGX is unreachable | Restart the tunnel in terminal A and re-run 7.2 and 7.3. If the run was in progress, preserve the artifacts and do **not** start a second run until we agree. |
| `classification.attempt_outcome` is `infrastructure_failure` with a model-call reason (`unreachable`, `timeout`, `transient`, ...) | The model call failed in transport or on the server, not a coding result | Read `agent/agent_result.json` `model_failure` and the trace. Check the tunnel and `journalctl -u ollama` on the DGX. |
| `agent.timed_out` is true (`attempt_outcome` `timed_out`) | The agent exceeded `--agent-timeout-s` (1800 s); partial work is still evaluated | Preserve everything. Read the trace for slow generations. Do not rerun with a bigger timeout without asking us. |
| A single slow model call | Normal for non-streaming generations: the bridge allows long calls | No action unless the run times out. |
| `refused: the agent sandbox is unavailable (...)` | A sandbox prerequisite is missing or blocked (section 5.3) | Stop. Send us the full message. **Do not** change AppArmor or other system security settings, and do not look for a way to run without the sandbox. Use a controller machine where 5.3 passes. With `--agent-sandbox docker` the agent sandbox needs no user namespaces: see "Choosing your sandbox". |
| Evaluator or Docker failure: `evaluation.outcome` `error`, a `docker`-related problem, an outer-timeout message, "no pytest_exit marker", or "OOM-killed" | The evaluator did not run to a verdict; this is **not** a coding result | Check `docker info`, the local image (5.4), and `<out>/env/site`. Re-run `reference-check` (6.2) to see whether the evaluator itself is healthy, but do not re-run the agent. |
| GPU out of memory or "model requires more system memory" (appears in the Ollama log or as an HTTP 500) | The model plus context does not fit | Stop. Record `ollama ps`, `nvidia-smi`, `free -h`, and the journal. Choosing a smaller quantization or model is a decision for us. |
| `classification.attempt_outcome` is `indeterminate` | The cause is not established: an agent error with no recorded category, or a worker that did not complete for an unstated reason | This is neither a pass nor a coding failure. Preserve everything; send `agent_result.json` (`model_failure`, `workers`) and the trace. Do not reclassify it yourself. |
| `attempt_outcome` is `unresolved` | The agent ran and the evaluator ran, but the task was not resolved | This is a genuine (if low-evidence) result. Report it with `candidate.patch` and `evaluation.json`. |
| `resolved: true` | The evaluator's tests passed | Report it, with the caveat in section 1: the task statement quotes the fix and this is one run. |

---

## 11. Cleanup and handoff

### 11.1 Stop everything you started

```bash
# [CONTROLLER] terminal A: press Ctrl-C to close the SSH tunnel, then confirm (expected: no output)
ss -ltnp | grep 11434
# [CONTROLLER] no leftover agent or sandbox processes (expected: no output)
pgrep -fa 'agent-child|bwrap --unshare-all'
# [CONTROLLER] no leftover evaluator containers (expected: no output)
docker ps -a --filter "label=purpose=coding-eval-pilot" -q
# [CONTROLLER] with --agent-sandbox docker: no leftover agent container (expected: no output)
docker ps -a --filter "label=purpose=coding-eval-agent" -q
```

Optionally, if nobody else uses the DGX's Ollama and the team agrees, release the model:

```bash
# [DGX]
ollama stop <MODEL_TAG>
ollama ps
```

Do not delete the model, the derived tag or the `dataset/`.

### 11.2 Record the environment and the run

Create `$OUT/HANDOFF.md` and fill it in. Items marked **(driver)** are also recorded automatically in `run.json`; copy them as a cross-check. The rest is not recorded by the driver, so you must write it.

| Item | Value |
|---|---|
| Repository commit and dirty state (`git rev-parse HEAD`, `git status --short`) | |
| Driver SHA-256 **(driver: `provenance.driver_sha256`)** | |
| Snapshot SHA-256 **(driver: `snapshot_sha256`; pinned `bda96e09...`)** | |
| Model name/tag **(driver: `config.model`)** and digest **(driver: `model_identity.digest`)** | |
| Quantization **(driver: `model_identity.quantization_level`)**; Modelfile contents (`ollama show <MODEL_TAG> --modelfile`) | |
| Ollama version (`ollama --version`) | |
| DGX OS, architecture, kernel (`uname -srm`, `/etc/os-release`) | |
| NVIDIA driver and GPU (`nvidia-smi`) | |
| GPU/memory use: `ollama ps`, `nvidia-smi`, `free -h` before, during (if you can) and after | |
| Controller OS, architecture, `bwrap`, `systemd`, `docker`, `uv` versions | |
| The exact command lines you ran (`prepare-env`, `reference-check`, `run`) and the SSH command with the host and user redacted | |
| Start and end time (UTC): `$OUT/console/run-start-utc.txt`, `run-end-utc.txt` | |
| Driver exit status: `$OUT/console/run-exit.txt` | |
| Output artifact location | |
| Final classification (`attempt_outcome`, `counts_as_coding_failure`) and the `resolved` flag | |
| Any deviation from this runbook, or anything surprising | |

### 11.3 Preserve and send the artifacts

Copy the whole output directory off the machine **before** any reboot or cleanup (it may be on a memory-backed `/tmp`):

```bash
# [CONTROLLER]
tar -C "$(dirname "$OUT")" -czf "$HOME/$(basename "$OUT").tar.gz" "$(basename "$OUT")"
sha256sum "$HOME/$(basename "$OUT").tar.gz"
```

Send us the archive and its hash. **Do not include `dataset/`, the gold patch or the test patch.** The archive contains `candidate.patch`, the evaluator logs, the agent's trace and the kept workspaces; do not edit any of it. Before sending, check that it holds no credentials. The trace records prompts only as lengths and hashes unless you captured them, which this runbook does not do.

### 11.4 Final checklist

- [ ] Pre-run decisions in 4.1 filled in with us (tag, quantization, `num_ctx` 8192, `num_predict`, base model installed).
- [ ] [DGX] GPU, Ollama service, tag, digest, `num_ctx 8192`, one small generation, 100% GPU, Ollama bound to loopback (4.3–4.6).
- [ ] [CONTROLLER] Commit, clean tree, driver hash recorded; `uv sync --locked` done (5.1, 5.2).
- [ ] [CONTROLLER] Sandbox prerequisites and the `sandbox OK` preflight (5.3); optional driver tests green.
- [ ] [CONTROLLER] Python dir and local `redis:7` image present (5.4); dataset layout and snapshot hash verified (5.5).
- [ ] [CONTROLLER] `$OUT` outside the repository with space; the same `$OUT` throughout (5.6).
- [ ] `prepare-env` succeeded; `reference-check` gave baseline 16 failed / 7 passed and gold 23 passed (6).
- [ ] Tunnel up; `/api/version` and `/api/tags` through it match the DGX; nobody else using the DGX Ollama (7).
- [ ] **One** `run` with the exact flags in section 8; console logs saved.
- [ ] Evidence read in the order of 9.2; `HANDOFF.md` filled in (11.2).
- [ ] Tunnel closed; no leftover processes or containers; model released if agreed (11.1).
- [ ] Artifact archive copied off the machine and sent with its SHA-256 (11.3).

---

## Choosing your sandbox

The agent runs in a sandbox that **whoever runs the harness chooses on the command line**; nothing is detected for you. The choice is `--agent-sandbox {bwrap,docker}`, default `bwrap` (`rich3278_pilot.py:1678`). **There is deliberately no unsandboxed mode:** a real run is always sandboxed or refused (`:1545`, `:1549-1552`, `:1560`; `sandbox_problem` never falls back, `:670-671`).

| | `bwrap` (the default) | `docker` |
|---|---|---|
| What the host must provide | Linux (`:672`); `bwrap`, `systemd-run` and `prlimit` on PATH (`:674`); a systemd user session and unprivileged user namespaces (the preflight starts a real sandbox and a cgroup scope, `:670-693`); a network namespace that shows only `lo`; enforced cgroup memory and task limits | A native Docker engine (Docker Desktop is not supported, `experiments/coding_eval/README.md`); a **local** image with `bash` (the container's entrypoint is `bash`, `:959`; the driver never pulls, `--pull never`, `:946`); host `/usr`, `/lib` and, if present, `/lib64` that can be bind-mounted read-only (`:921-929`); the interpreter handling below. It needs no user namespaces, bwrap or systemd: `sandbox_problem` is called only for `bwrap` (`:1549-1550`). |
| Flags to add to `run` | none | `--agent-sandbox docker`, and `--agent-image <local image>` if the image is not `redis:7` (`:1679`) |

**Minimum flags for `run`** (whichever sandbox): `--out <dir>`, `--model <tag>` (both required by the parser, `:1667`, `:1669`) and `--allow-model-call` (the driver refuses without it, `:1715`). Everything else has a default (`--base-url http://127.0.0.1:11434/v1`, `--max-tokens 768`, and so on; section 8). Add `--agent-sandbox docker` for the Docker sandbox.

**The evaluator always needs Docker, whichever agent sandbox you choose.** `main` builds a `DockerRunner` for `run`, `evaluate` and `reference-check` (`:1711`), and `run` evaluates the candidate in it (`:1618`). So you always need: a Docker engine; a local image that provides `bash` and `timeout` (`--image`, default `redis:7`, `:1665`; never pulled); and a Python that matches the machine's architecture, given with `--python-dir` (the default is an **x86_64** uv-managed CPython under `~/.local/share/uv/python/…`, `:1663`; `--python-dir /usr` selects the system-Python mode, which has never been exercised by a run: **UNVERIFIED**). The evaluator's Python and image are separate from the agent's (`--agent-image` is only the agent container's).

**On a host with neither sandbox the driver refuses.** With the default `bwrap` and no usable sandbox, `run` raises "the agent sandbox is unavailable (…); refusing to run the agent as the plain host user" (`:1550-1552`); `run.json` gets `status` `refused` and the `error` text (`:1623`); the driver prints `refused: <error>` on stderr (`:1721-1722`) and exits **1**; the JSON line on stdout is unchanged. With `--agent-sandbox docker` and no usable Docker, `docker create failed: …` is refused the same way. Refusals of the command line itself (a bad `--out`, a missing `--allow-model-call`) print `refused: <error>` and exit **2** (`:1735-1736`).

**Docker sandbox and the Python interpreter (commit `4fde05e`).** The container runs the venv's `/venv/bin/python` with `/usr`, `/lib` and `/lib64` mounted from the host. If the venv's interpreter resolves under `/usr`, `/lib` or `/lib64`, nothing extra is mounted. Otherwise (for example a uv-managed CPython under `~/.local/share/uv/python/…`) the resolved `sys.base_prefix` is mounted **read-only** at the path the venv uses, which can be an alias directory (`agent_interpreter_mount`, `:832`; used by `docker_agent_plan`, `:931-936`). The driver refuses early, before any docker call, when the interpreter is missing or unreachable, or when that base prefix is `/`, the home directory, a parent of home, contains the repository root or a protected directory, or has no python executable. **UNVERIFIED:** the alias-path mount has not run in a real container. The no-model check in X1 (`sandbox-check`) exercises exactly this.

---

## Experiment sequence after commit 7ecb2b1

Experiments X1 to X5, for a colleague running them on the DGX Spark. In this section the DGX is also the controller: the driver runs on the DGX with `--agent-sandbox docker`, against the DGX's native Linux Docker engine, and the model is reached at `http://127.0.0.1:11434/v1` on the same machine (no SSH tunnel, so section 7 is not used). This overrides the "separate controller" advice in section 2 for these experiments only. **Observed earlier** (`provenance` of the e0cf run on the DGX): `platform` `Linux-6.17.0-1032-nvidia-aarch64-with-glibc2.39`, `python` `3.12.3`.

### Common rules

- Run the steps in order. Run each experiment **once**: no retries, no extra seeds.
- **STANDING AUTHORIZATION from the lead.** The whole ladder below is authorized under these conditions; no further go-ahead is needed for each step.
  1. Run X1 and X2 first (they call no model). If either fails, stop and send the results.
  2. If both pass, run X3, then X4. Run X5 only if X3 returned structured `tool_calls` (class 1).
  3. One run per step, no retries, no extra seeds, no edits to the driver; any sandbox refusal, `verify_agent_container` mismatch or leftover container is a stop.
  4. If `sandbox_check.json` shows an `interpreter_mount`, its `source` must be a single CPython install directory; if it is anything broader (for example a directory under home, or `/tmp`), stop and send the file before X4.
  5. Send results after X2 or at any stop, and again after the ladder, not after every step.
- X3, X4 and X5 call a model; X1 and X2 do not. After each step collect the evidence listed for that step and run the leftover-container check below.
- A sandbox refusal, a mount error, a `verify_agent_container` mismatch (the driver prints `agent container isolation not as configured; refusing to start it: …`) or a leftover container is a **stop**. Do not edit the driver; send the full message.
- Run artifacts stay outside git: under `/tmp`, never committed, never copied into the repository. `new_gemma_eval_runs/` stays git-ignored. The driver itself refuses an `--out` inside the repository or the dataset (`check_out_dir`, `rich3278_pilot.py:1187`).
- Do not open or send `dataset/`, the gold patch or the test patch.
- After **every** step, check for leftover containers (expected: no output from either command):

```bash
# [DGX] (the controller)
docker ps -a --filter "label=purpose=coding-eval-agent" -q
docker ps -a --filter "label=purpose=coding-eval-pilot" -q
```

Setup used by every step:

```bash
# [DGX] from the repository root, at the commit the lead names (section 5.1); the tree must be clean
git rev-parse HEAD && git status --short
uv sync --locked
export XOUT=/tmp/rich-x-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$XOUT"; echo "$XOUT"
# the dataset is git-ignored and must already be on this machine (section 5.5); this checks the layout and the pinned hash only
test -f dataset/tasks.jsonl && test -f dataset/snapshots/rich_3278.tgz && test -d dataset/wheels && echo "layout OK"
sha256sum dataset/snapshots/rich_3278.tgz     # must be bda96e09e9e02fa372b2cf6c5f3f2760f97d292fb531cc631564632940d6a453
```

### X1. The full Docker agent sandbox without a model (`sandbox-check`)

**Purpose.** Create the agent container with its full real mount set and the venv's Python, start a fixed diagnostic instead of the agent, and judge what the container shows. No model is involved.

**Known (source).** `sandbox-check` is a subcommand of the driver (parser `rich3278_pilot.py:1690-1692`, dispatch `:1709-1710`). It builds the same plan as a run (`docker_agent_plan`, `:911`) and uses the same create → inspect → `verify_agent_container` → start → kill → rm → gone path (`run_agent_container`, `:1062`); only the container's command differs (`SANDBOX_CHECK_COMMAND`, `:1238`). It needs **no model, no `--allow-model-call`, no `--model` or `--base-url`, no Ollama, no tunnel, no dataset and no snapshot**, and it never calls the model-identity code. Arguments:

- `--out <dir>` (required; the same rules as the other subcommands: it must be outside the repository and the dataset, `check_out_dir`, `:1524-1528`).
- `--agent-image <local image>` (default `redis:7`; a local image with `bash`; never pulled, `:946`).
- `--dataset-dir`, `--python-dir`, `--site-dir`, `--task-id` and `--image` are accepted (they come from the shared parser, `:1661-1665`) but **ignored**: only `--dataset-dir` is used, as a protected path that no mount may contain (`:1321`); it is not read. `--python-dir` does **not** change which interpreter is mounted: that is decided by the venv you run from (`sys.prefix`) and `agent_interpreter_mount` (`:832`, `:931-936`).

**Status.** **UNVERIFIED in a real container:** this has never run against a real Docker engine, only against a stubbed `docker`. It cannot run under Docker Desktop (it does not share `/tmp`, `/lib64` or the host `/usr`, so the system mounts cannot be bind-mounted; `experiments/coding_eval/README.md`).

**Prerequisites.** The Setup above done (`uv sync --locked`, so that `uv run` uses the repository's venv). A native Linux Docker engine. The local image exists: `docker image inspect redis:7 --format '{{.Id}} {{.Architecture}} {{.Os}}'`.

```bash
# [DGX] bash assumed. No model is called.
mkdir -p "$XOUT/x1/console"
uname -m | tee "$XOUT/x1/console/uname-m.txt"
docker info --format '{{.OperatingSystem}} {{.Architecture}}' | tee "$XOUT/x1/console/docker-info.txt"   # Docker Desktop would say "Docker Desktop"
uv run python experiments/coding_eval/rich3278_pilot.py sandbox-check --out "$XOUT/x1" --agent-image redis:7 2>&1 | tee "$XOUT/x1/console/sandbox-check.txt"
echo "sandbox-check exit: ${PIPESTATUS[0]}" | tee "$XOUT/x1/console/sandbox-check-exit.txt"
ls "$XOUT/x1" "$XOUT/x1"/sandbox-check-*
docker ps -a --filter "label=purpose=coding-eval-agent" -q
```

**Files it writes** (in `$XOUT/x1/sandbox-check-<UTC time>-<4 hex>/`, `:1304`):

- `sandbox_check.json` (`:1358`): the host facts, the interpreter mount, every check with expected and observed values, the verify result, the cleanup result and the overall result. It holds no model or dataset information.
- `agent_container_inspect.json` (`:1328`): the real `docker inspect` of the container, saved before it is judged.
- `agent/sandbox_check.json`: the diagnostic's raw observations from inside the container. `agent/sandbox_check_diag.py`, `agent/sandbox_check_plan.json`, `agent/child.stdout.txt` and `agent/child.stderr.txt` are alongside it.

**How to read the result.**

- The driver prints one line per check, starting `PASS  ` or `FAIL  `, then `OVERALL: PASS` or `OVERALL: FAIL` (`:1359-1363`). If the sandbox could not be checked at all (a `verify_agent_container` mismatch, a failed `docker create`, a cleanup that could not be confirmed) it also prints `FAIL  the sandbox could not be checked: <error>` and the overall result is FAIL.
- **Exit code:** 0 only on `OVERALL: PASS`; 1 on any FAIL, a verify refusal or a cleanup failure (`:1364`). Exit 2 with `refused: <error>` is a refusal of the command line (for example an `--out` inside the repository, `:1735-1736`).
- **The checks** (`evaluate_sandbox_check`, `:1255-1299`): the interpreter resolves inside the container; `import sys, encodings, ssl, json` works; `smol_harness`, `effgen` and the driver (imported from `/repo`) import; `/repo/dataset`, `/repo/.git`, `/repo/gemma_eval_runs`, `/repo/new_gemma_eval_runs` and `/repo/docs` do not exist; **`/repo` holds exactly `experiments` and `src`** (`:1280`); the working directory is `/agent`; `/data` is a tmpfs; `/agent` is writable and `/repo`, `/venv` and `/usr` are not; if an interpreter mount was added, its destination exists inside the container as a read-only mount, otherwise "no interpreter mount is needed"; the route table is empty; a TCP connect to `192.0.2.1:9` fails within 3 seconds; the diagnostic produced its result and exited cleanly.
- **A different `/repo` listing, or any other FAIL, is a stop to investigate and report.** Do not edit the driver or the expectations to make it pass; send `sandbox_check.json`, the inspect file and the console log.
- `interpreter_mount` in `sandbox_check.json` is `null` when the venv's interpreter is under `/usr`, `/lib` or `/lib64`, and otherwise holds the `source`, `destination` and `mode` of the one extra read-only mount.

**Record.** `$XOUT/x1/console/*`, `sandbox_check.json`, `agent_container_inspect.json`, `agent/sandbox_check.json`, and from `sandbox_check.json`'s `host` block: `uname_m`, `docker_version`, `interpreter`, `interpreter_realpath`, `sys_prefix` and `sys_base_prefix` (`:1311-1312`, `:1318`); the leftover-container output (expected: none).

**What this does NOT show.** Nothing about arm64 evaluation (X2), about any model or tool behaviour, or about a full agent run: the container runs a diagnostic, not the agent. A PASS shows only that this container, with these mounts and this venv, starts, resolves its Python and imports what the agent needs, and looks as configured from inside. The full configuration it exercises has never run in a real container.

### X2. Evaluator reference check on ARM64 (no model)

**Purpose.** Find out whether the evaluator container reproduces the task's baseline and gold results on the DGX's architecture (aarch64).

**Prerequisites.** Setup above done. `docker image inspect redis:7 --format '{{.Id}} {{.Architecture}} {{.Os}}'` succeeds (the driver never pulls; image check at `rich3278_pilot.py:362`). `uv` is installed.

**Which interpreter.** **Known** (`:1323`): the default `--python-dir` is an x86_64 uv-managed CPython (`~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu`), which cannot be the right interpreter on an aarch64 host. **Known** (`:351`, `:365-372`, `:1310`): `--python-dir /usr` selects the system-Python mode: the evaluator container bind-mounts the host's `/usr`, `/lib` and (if present) `/lib64` read-only and runs `/usr/bin/python3`, and `prepare-env` installs into the site directory with `--python /usr/bin/python3`. **UNVERIFIED:** that mode has never been exercised by a run; the DGX's `/usr/bin/python3` version was not checked against the dataset wheels (the earlier runs' `provenance.python` was `3.12.3`, which may or may not be this interpreter); and the evaluator's behaviour with the host's `/usr` mounted over the `redis:7` image's own is untested. Record `/usr/bin/python3 --version`. If you use a different `--python-dir`, it must be a standalone CPython 3.12 for aarch64 (**UNVERIFIED** that one exists on the machine).

```bash
# [DGX]
/usr/bin/python3 --version | tee "$XOUT/system-python-version.txt"
mkdir -p "$XOUT/x2/console"
uv run python experiments/coding_eval/rich3278_pilot.py prepare-env --python-dir /usr --out "$XOUT/x2" 2>&1 | tee "$XOUT/x2/console/prepare-env.txt"; echo "prepare-env exit: ${PIPESTATUS[0]}" | tee "$XOUT/x2/console/prepare-env-exit.txt"
uv run python experiments/coding_eval/rich3278_pilot.py reference-check --python-dir /usr --out "$XOUT/x2" 2>&1 | tee "$XOUT/x2/console/reference-check.txt"; echo "reference-check exit: ${PIPESTATUS[0]}" | tee "$XOUT/x2/console/reference-check-exit.txt"
ls "$XOUT/x2"
docker ps -a --filter "label=purpose=coding-eval-pilot" -q
```

(Both commands are those of sections 6.1 and 6.2 with `--python-dir /usr` added; `--python-dir` is a flag of every subcommand, `rich3278_pilot.py:1323`. Run `prepare-env` first: `reference-check` reads `<out>/env/site`.)

**Record.** The two exit codes; the two JSON lines printed by `reference-check` (`label`, `outcome`, `resolved`, `counts` with `passed`, `failed`, `errors`, `pytest_exit`); `$XOUT/x2/baseline-*.evaluation.json` and `gold-*.evaluation.json`; the evaluator log files under the `baseline-*` and `gold-*` directories in `$XOUT/x2` (`ls` shows them); the leftover-container output.

**Read it.** **Known** expected values (section 6.2; reproduced on x86_64 only): baseline `unresolved_tests_failed`, `passed` 7, `failed` 16, `errors` 0, `pytest_exit` 1; gold `resolved`, `passed` 23, `failed` 0, `errors` 0, `pytest_exit` 0. Anything else, any non-zero exit, any `"outcome": "error"` or a `refused:` message is a stop (section 10, evaluator row). On aarch64 the match is **UNVERIFIED** until this step is run.

**What this does NOT show.** Anything about the agent sandbox, the model, or any agent-produced patch. A pass shows only that the evaluator container, in the system-Python mode, reproduces the two reference results once on this machine.

### X3. Ollama capability probe (model server only, no harness; one model call)

**Purpose.** See once whether the Ollama endpoint accepts a tool schema and returns a structured tool call for `gemma4:e4b`.

**Prerequisites.** X1 and X2 passed. Ollama is running on this machine (section 4.3). The one request below is the only inference in the step.

```bash
# [DGX]
mkdir -p "$XOUT/x3"
ollama --version 2>&1 | tee "$XOUT/x3/ollama-version.txt"
ollama list | tee "$XOUT/x3/ollama-list.txt"                                  # the gemma4:e4b ID must begin c6eb396dbd59
ollama show gemma4:e4b | tee "$XOUT/x3/ollama-show.txt"                       # note the capabilities list
ollama show gemma4:e4b --modelfile | tee "$XOUT/x3/ollama-modelfile.txt"      # note the template and any num_predict / num_ctx
cat > "$XOUT/x3/request.json" <<'JSON'
{"model": "gemma4:e4b", "temperature": 0, "max_tokens": 256,
 "messages": [{"role": "user", "content": "Call the function record_value with text set to hello."}],
 "tools": [{"type": "function", "function": {"name": "record_value", "description": "Record a value",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}]}
JSON
curl -sS -o "$XOUT/x3/response-body.json" -w '%{http_code}\n' http://127.0.0.1:11434/v1/chat/completions -H 'Content-Type: application/json' -d @"$XOUT/x3/request.json" | tee "$XOUT/x3/http-status.txt"
python3 - <<'PY'
import json, os
b = json.load(open(os.path.join(os.environ["XOUT"], "x3", "response-body.json")))
c = (b.get("choices") or [{}])[0]; m = c.get("message") or {}
print("tool_calls:", json.dumps(m.get("tool_calls"), indent=1))
print("content:", repr(m.get("content")))
print("finish_reason:", c.get("finish_reason"), "| error:", b.get("error"))
PY
docker ps -a --filter "label=purpose=coding-eval-agent" -q
```

(The `ollama` commands are those of section 4.4. The request body, the URL and the `curl` options were specified by the lead for this probe; they are not taken from the driver. **UNVERIFIED:** that this exact request is what the harness's own client sends.)

**Record.** `$XOUT/x3/ollama-version.txt`, `ollama-list.txt` (digest prefix), `ollama-show.txt` (capabilities list), `ollama-modelfile.txt` (template, `num_predict`, `num_ctx`), `http-status.txt`, the full `response-body.json`, and the printed summary.

**Read it.** Classify the response as one of: (1) a structured `message.tool_calls` list; (2) the call appears only as text in `message.content`; (3) neither. Record `finish_reason` and any error text. A non-200 status or an `error` field is recorded as is.

**What this does NOT show.** It is a single sample. It shows only whether the endpoint accepts tool schemas and returns structured calls once. It does not show reliability, and says nothing about `edit_file`.

### X4. One pilot run in the default tool-calling mode (`react`) with `gemma4:e4b`

**Purpose.** See whether the code changes since the third attempt (`rich_3278-20261111T164535Z-e0cf`, an empty patch after four failed `edit_file` calls) change its outcome, with every setting held equal.

**Prerequisites.** X1 read `OVERALL: PASS`, X2 passed and X3 completed (the evaluator needs the packages in `<out>/env/site` and the `--python-dir` that X2 used). The model digest in `ollama list` matches X3 (`c6eb396dbd59…`). The dataset is present (git-ignored). The `interpreter_mount` condition (authorization, item 4) holds.

**Settings.** Taken from the `config` and `provenance` blocks of e0cf's `run.json` (nothing else in that directory was read): `--model gemma4:e4b`, `--context-length 16384`, `--max-tokens 4096`, `--temperature 0.1`, `--seed 0`, `--max-iterations 8`, `--agent-timeout-s 1800`, `--base-url http://127.0.0.1:11434/v1`, `--agent-sandbox docker`, `--agent-image redis:7`, `--keep-workspaces`. The per-worker timeout is fixed in the driver at 300 s (`rich3278_pilot.py:1372`), as in e0cf (`worker_timeout_s` 300). e0cf's record has no `tool_calling_mode`; this run passes `--tool-calling-mode react` explicitly (the default, `:1340`). The only intended differences are code changes.

```bash
# [DGX] bash assumed. This calls the model. Use the same --python-dir as X2.
mkdir -p "$XOUT/x4/console"
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$XOUT/x4/console/run-start-utc.txt"
uv run python experiments/coding_eval/rich3278_pilot.py run \
  --out "$XOUT/x4" \
  --python-dir /usr \
  --site-dir "$XOUT/x2/env/site" \
  --model gemma4:e4b \
  --context-length 16384 \
  --base-url http://127.0.0.1:11434/v1 \
  --max-tokens 4096 \
  --temperature 0.1 \
  --seed 0 \
  --max-iterations 8 \
  --agent-timeout-s 1800 \
  --agent-sandbox docker \
  --agent-image redis:7 \
  --tool-calling-mode react \
  --keep-workspaces \
  --allow-model-call 2>&1 | tee "$XOUT/x4/console/run.txt"
echo "driver exit status: ${PIPESTATUS[0]}" | tee "$XOUT/x4/console/run-exit.txt"
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$XOUT/x4/console/run-end-utc.txt"
docker ps -a --filter "label=purpose=coding-eval-agent" -q
docker ps -a --filter "label=purpose=coding-eval-pilot" -q
```

(`--site-dir` points at X2's packages instead of re-running `prepare-env`; it is a flag of every subcommand, `rich3278_pilot.py:1324`, and `main` uses it in place of `<out>/env/site` (`:1362`). **UNVERIFIED:** a run with `--site-dir` outside its own `--out` has not been exercised. If the driver refuses it, stop; do not re-run `prepare-env` without the lead's say.)

**Record**, in the order of sections 9.2 steps 1 to 5 (set `RUN` first: `export RUN="$XOUT/x4/$(ls "$XOUT/x4" | grep '^rich_3278-' | head -1)"`):

1. `run.json`: `status`, `error`, `resolved`, `classification`.
2. `model_identity` (the digest must equal X3's), `sandbox` (`kind` is `Docker`; `limits`; `isolation`), `config` (confirm `tool_calling_mode` reads `react`), `provenance.driver_sha256`.
3. `agent/agent_result.json`, and `$RUN/agent_container_inspect.json` (the driver's real `docker inspect` of the agent container, `rich3278_pilot.py:1265`).
4. `agent/history/*/trace.jsonl`.
5. `candidate.patch` and `evaluation.json`.
6. The leftover-container output.

**Report separately** (answer each with the evidence):

- Did any `edit_file` call succeed?
- Did the candidate become non-empty?
- Did the evaluator run tests?
- The exact arguments of every `edit_file` call. **Known** (`src/smol_harness/recorder.py:300`): the trace records `tool_call_start` events with `tool` and `args`. Print them with:

```bash
# [DGX]
grep -h '"tool_call_start"' "$RUN"/agent/history/*/trace.jsonl | grep 'edit_file'
```

**Read it** as a pipeline smoke test, not a coding benchmark: the task statement still describes the fix in prose.

**What this does NOT show.** Reliability, since it is one run. Whether the model can do the task. Anything about `native` or `hybrid` mode.

### X5. One pilot run in `hybrid` tool-calling mode (only if X3 returned structured calls)

**Purpose.** The same run as X4 with `--tool-calling-mode hybrid`, to see once what changes in how tool calls are read.

**Prerequisites.** X1 read `OVERALL: PASS`. X4 is done. X3 returned a structured `message.tool_calls` list (class 1).

**Command.** Identical to X4 (same settings, same `--python-dir`, same `--site-dir`) with a separate `--out` and `--tool-calling-mode hybrid`:

```bash
# [DGX] bash assumed. This calls the model.
mkdir -p "$XOUT/x5/console"
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$XOUT/x5/console/run-start-utc.txt"
uv run python experiments/coding_eval/rich3278_pilot.py run \
  --out "$XOUT/x5" \
  --python-dir /usr \
  --site-dir "$XOUT/x2/env/site" \
  --model gemma4:e4b \
  --context-length 16384 \
  --base-url http://127.0.0.1:11434/v1 \
  --max-tokens 4096 \
  --temperature 0.1 \
  --seed 0 \
  --max-iterations 8 \
  --agent-timeout-s 1800 \
  --agent-sandbox docker \
  --agent-image redis:7 \
  --tool-calling-mode hybrid \
  --keep-workspaces \
  --allow-model-call 2>&1 | tee "$XOUT/x5/console/run.txt"
echo "driver exit status: ${PIPESTATUS[0]}" | tee "$XOUT/x5/console/run-exit.txt"
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$XOUT/x5/console/run-end-utc.txt"
docker ps -a --filter "label=purpose=coding-eval-agent" -q
docker ps -a --filter "label=purpose=coding-eval-pilot" -q
```

**Record.** The same evidence as X4 (with `RUN="$XOUT/x5/…"`), plus `config.tool_calling_mode` in `run.json` (it must read `hybrid`; **Known**: the driver stores it in `config` and in `provenance.tool_calling_mode`, `rich3278_pilot.py:1182`, `:1206`) and `tool_calling_mode` in `agent/agent_result.json` (`:754`). For the trace, record whether tool arguments arrived as structured calls or as text. **UNVERIFIED:** which `trace.jsonl` fields show that difference (the `tool_call_start` events carry `tool` and `args`, and the model-call event is `generation_start`; whether either records the raw response form was not checked). Quote the lines you rely on.

**Compare with X4** only by the listed facts (edit calls that succeeded, candidate empty or not, evaluator ran or not, the arguments).

**What this does NOT show.** Reliability: one run each shows nothing about it. It does not show that `hybrid` is better or worse than `react`.

### What to send back

Run artifacts stay outside git; send them directly and never commit them.

- `sandbox_check.json` from X1 (**it contains usernames and host paths**: send it directly, never commit it) and `agent_container_inspect.json` from the same directory.
- The X2 output: `$XOUT/x2/console/`, both exit codes, and the `baseline-*` and `gold-*` evaluation files.
- The X3 files: `$XOUT/x3/` (`ollama-version.txt`, `ollama-list.txt`, `ollama-show.txt`, `ollama-modelfile.txt`, `request.json`, `http-status.txt`, `response-body.json`).
- Each pilot run directory (X4, and X5 if run) **without `dataset/`**: archive `$XOUT` as in section 11.3 (with `$XOUT` in place of `$OUT`) and send it with its SHA-256. Do not include the gold patch or the test patch.
- `ollama --version` and `ollama show gemma4:e4b` (the X3 files hold both).
- The original `rich_3278-20261105T182312Z-ba51` run directory from the DGX (the committed copy holds zero-byte stubs).
- `git diff pyproject.toml uv.lock` from the DGX clone (nothing is expected; send the output even if it is empty).

---

## Appendix: what was and was not verified when this runbook was written

**Verified against the source and `--help` output:** every driver flag and default in sections 6 and 8; the `--out` rules; the `$OUT/env/site` coupling; the run directory layout in 9.1; the outcome classes; the pinned snapshot hash; the driver's SHA-256 at `8ba41c5`; the sandbox preflight one-liner (it printed `sandbox OK` on the original laptop); the controller tool versions listed as "Validated".

**Not verified here:** every DGX command in section 4 (they are standard `nvidia-smi`, `ollama` and `curl` calls that were not run for this document); the SSH tunnel commands; the commands in 5.4 for installing Python or the Docker image (they download); the artifact transfer commands; any model's coding ability. **Since this text was written:** the driver has run for real three times on a DGX Spark (2026-11-05 and 2026-11-11; `gemma4:e4b` twice and `qwen3.8:27b` once, all empty patches; see the update at the top). The DGX Ollama contact and model identity are therefore exercised, and the earlier form of the Docker agent sandbox ran; the full current Docker agent sandbox (a reduced-fidelity check of earlier code ran once on x86_64, 2026-11-13), the evaluator's ARM64 baseline/gold check and any non-empty candidate remain unverified.
