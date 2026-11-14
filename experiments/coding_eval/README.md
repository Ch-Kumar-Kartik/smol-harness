# `rich_3278` single-task evaluation pilot

`rich3278_pilot.py` connects the existing harness to the reference pipeline validated on 2026-11-03 (baseline 16 failed / 7 passed, gold patch 23 passed). It is an
**experiment utility** for ONE task; it is not a benchmark runner, and it is not part of `src/`.

**Colleague runbook:** [`../../instructions.md`](../../instructions.md) has the step-by-step procedure for running this pilot against a model served by a DGX Spark (controller machine plus DGX, SSH tunnel, evidence collection, failure handling).

**Status: three real-model execution attempts exist (two on 2026-11-05, committed under [`../../gemma_eval_runs/`](../../gemma_eval_runs/); one on 2026-11-11, in the git-ignored `new_gemma_eval_runs/`); see [Real runs on record](#real-runs-on-record-2026-11-05-and-2026-11-11). None produced a non-empty candidate patch, none reached the Docker evaluator's tests, and none resolved the task.** They are execution attempts, not coding evaluations. `gemma4:e4b` (ba51) completed the agent phase with an empty patch (`attempt_outcome: indeterminate`), `qwen3.8:27b` (c25f) hit a model-call timeout (`infrastructure_failure`), and a second `gemma4:e4b` run (e0cf) ended `unresolved` with an empty patch after a tool-interface failure. All ran on a DGX Spark in the Docker agent sandbox; the Docker sandbox in its current form (below): the full current configuration (system `/usr`, `/lib`, `/lib64` mounts, the venv mount and the Python agent-child) has never run in a container, and the ID-based cleanup in `4067852` has only run against a stubbed `docker`; a reduced-fidelity check of earlier code ran once on x86_64 under Docker Desktop on 2026-11-13 (details: `experiments/coding_eval/README.md`, Docker agent sandbox). The default tests (`tests/test_coding_eval_pilot.py`) use a fake container, scripted agents and stand-in HTTP servers (not models).

## How a run works

```
snapshot.tgz --(sha256 pinned)--> agent workspace (no .git)      evaluator pristine copy (no .git)
                                        |                                  |
   statement only --> harness (child process, wall-clock limit)            |
                      file_operations + edit_file, no shell                |
                                        |                                  |
                      candidate patch = diff(workspace, pristine) --------+
                                        |
        fresh pristine + candidate (protected files stripped) + OFFICIAL test patch
                                        |
              disposable container: no network, no mounts, caps dropped, limits   -> evaluation.json
```

* The agent phase runs `Harness.run` through its public API (`initialize_runtime`, `load_config`, `build_harness_from_config(..., recorder=...)`) in a **child
  process in its own process group**, so the wall-clock limit can really stop it. One orchestrator (`file_operations`) plus one worker (`coder`, `file_operations` and `edit_file`). `edit_file` (T4.3) replaces one exact occurrence of a text in a file, so the one-line fix does not need a 6,900-character rewrite of `rich/ansi.py`; the worker's system prompt tells it to read first, then edit with a short unique `old_text`.
* The orchestrator is told to delegate the edit to `coder` and is not told to use `edit_file` (it does not hold it); the worker's prompt carries the backslash-escaping guidance (below), and that is the only place it appears.
* The candidate is collected **by comparing the two trees**, never by running `git` in a directory the agent can write (a planted `.git/config` could execute code).
  Symlinks, hard links, special files, non-UTF-8 changes and more than 50 changed files make the run fail closed. A temporary file left behind by `edit_file` (exactly
  `.<name>.edit-<12 lowercase hex>`, only possible if the process is killed mid-edit) is not a candidate file; it is listed in `candidate.edit_temporaries_ignored`.
* Evaluation never trusts the agent: `HarnessResult.success` is recorded as `agent_reported_success` and ignored. `resolved` requires pytest exit 0, no failed or
  errored test, and every test function added by the official test patch to have PASSED (a skipped required test is not a pass).

## Usage

```bash
# 1. once: the test dependencies, installed OFFLINE from the dataset wheels into <out>/env/site (wheels are unpacked, nothing is built or run)
uv run python experiments/coding_eval/rich3278_pilot.py prepare-env --out /tmp/rich-pilot

# 2. no agent, no model: the baseline and the gold patch through the evaluator (expect 16 failed/7 passed, then 23 passed)
uv run python experiments/coding_eval/rich3278_pilot.py reference-check --out /tmp/rich-pilot

# 3. no agent, no model: evaluate any candidate patch file
uv run python experiments/coding_eval/rich3278_pilot.py evaluate --out /tmp/rich-pilot --patch candidate.patch

# 4. a real run: CALLS THE CONFIGURED MODEL, needs explicit permission on the command line
uv run python experiments/coding_eval/rich3278_pilot.py run --out /tmp/rich-pilot --model <ollama tag> --context-length 8192 \
    --max-tokens 768 --temperature 0.1 --seed 0 --agent-timeout-s 1800 --allow-model-call

# 4b. the same run with the Docker agent sandbox (used on the DGX Spark, where unprivileged
#     user namespaces are unavailable); --agent-image must already exist locally
uv run python experiments/coding_eval/rich3278_pilot.py run --out /tmp/rich-pilot --model <ollama tag> --context-length 8192 \
    --max-tokens 768 --temperature 0.1 --seed 0 --agent-timeout-s 1800 \
    --agent-sandbox docker --agent-image redis:7 --allow-model-call
```

`--tool-calling-mode {auto,native,react,hybrid}` (default `react`) selects effGen's tool-calling mode for the agents; it is recorded in `run.json` (`config.tool_calling_mode` and `provenance.tool_calling_mode`). Every real run so far used the default `react`. No `native` or `hybrid` run has ever been made against Ollama; the only native-mode attempt on record is the earlier `LiquidAI/LFM2.5-350M` one (0 of 2).

`--out` must be outside the repository and the dataset. Each run gets `<out>/<run id>/` with `run.json` (configuration, deviations, agent and candidate metadata,
verdict), `candidate.patch`, `evaluation.json`, `agent/` (rendered config, prompt, `child.*.txt`, `agent_result.json`, the recorder's `history/trace.jsonl`) and, at
the end, the temporary workspaces removed (`--keep-workspaces` keeps them). Only paths this run created are removed.

**Explicit settings:** task id, dataset/snapshot (SHA-256 pinned in the code), agent workspace (`<run>/agent/workspace`), model/provider (`--provider ollama`
only, `--model`, `--context-length`, `--base-url`), generation limits (`--max-tokens`, `--temperature`, `--seed`; requested values, see below), agent wall clock
(`--agent-timeout-s`), container limits (`ContainerLimits`: 1 GiB memory without swap, 2 CPUs, 256 processes, 1024 open files, 100 MB per file, 256 MB `/tmp`,
120 s test timeout, 300 s outer timeout), artifact directory (`--out`).

**Tests:** `uv run pytest tests/test_coding_eval_pilot.py -q` (no Docker, no model). The container tests are opt-in: `SMOL_RUN_DOCKER_TESTS=1 uv run pytest
tests/test_coding_eval_pilot.py -q -k real_container` (needs the dataset and a local Docker image; the image is never pulled).

## Real runs on record (2026-11-05 and 2026-11-11)

The two 2026-11-05 runs used the Docker agent sandbox (`--agent-sandbox docker`, image `redis:7`) on a DGX Spark (`aarch64`), with the repository at commit `fda9dfb` and an uncommitted working tree (the recorded `provenance.driver_sha256` and `repo_status` describe it). Generation settings were `--context-length 16384 --max-tokens 4096 --temperature 0.1 --seed 0 --max-iterations 8 --agent-timeout-s 1800`; the requested `max_tokens` is not a verified server cap.

| Run directory | Model | Agent phase | Candidate | `attempt_outcome` |
|---|---|---|---|---|
| `rich_3278-20261105T182312Z-ba51` | `gemma4:e4b` (8.0B, Q4_K_M) | `completed` after 540 s: worker `d1` timed out with 0 iterations, worker `d2` completed (5 iterations, 4 tool calls, 21,581 tokens) | empty (0 bytes, no changed files) | `indeterminate` — the worker timeout's cause is not established; not counted as a coding failure |
| `rich_3278-20261105T183253Z-c25f` | `qwen3.8:27b` (27.3B, Q4_K_M) | `termination: error` after 855 s: the first model call timed out (`RuntimeError`, category `timeout`); no workers | empty (0 bytes, no changed files) | `infrastructure_failure` — transport/server, not a coding result |

Neither run changed a file, so the evaluator short-circuited as `unresolved_empty_patch` and no test ran. It does not establish that the model cannot do the task: in the Gemma run one worker timed out and a second completed without editing. The Qwen run exercises the timeout/classification path only. For `ba51` the committed `agent/history/trace.jsonl` and workspace copy are missing or zero-byte, so its worker behaviour cannot be re-read from the artifacts.

**Third attempt, 2026-11-11 (not committed).** `rich_3278-20261111T164535Z-e0cf` (`gemma4:e4b`, driver sha256 `e4eb6a5b…`) is in `new_gemma_eval_runs/`, which is git-ignored. Its classification is `unresolved` / `unresolved_empty_patch` with `counts_as_coding_failure: true`, but the cause was a **tool-interface failure** (malformed or double-escaped JSON in tool calls), not an assessed coding failure, and that run's worker prompt **still contained the reference fix line**, so it is not a clean attempt either. The prompt has since been changed so that it does not contain that line. Like the other two, it never reached the evaluator's tests.

## Outcome classes (`run.json` → `classification`)

A result is never just `resolved: true/false`. `classification` keeps four things apart and says why:

| Field | Values |
|---|---|
| `agent` | `completed`, `agent_reported_failure`, `timed_out`, `infrastructure_failure`, `indeterminate`, `not_run` |
| `candidate` | `collected`, `empty`, `does_not_apply`, `not_collected` |
| `evaluator` | the evaluator's outcome (`resolved`, `unresolved_tests_failed`, `unresolved_empty_patch`, `unresolved_patch_failed`) or `infrastructure_failure` / `not_run` |
| `attempt_outcome` | `resolved`, `unresolved`, `timed_out`, `indeterminate`, `infrastructure_failure` |

**Precedence** (the first that applies decides `attempt_outcome`; `infrastructure_reasons` and `indeterminate_reasons` say why):

1. **`infrastructure_failure`**: setup refused; the agent left no result (a crash, or the sandbox's memory cap killing it); the harness raised; the model call failed in transport, server or configuration
   (effGen error category `unreachable`, `timeout`, `transient`, `rate_limited`, `auth`, `not_found`, `not_loaded`, `resource_exhausted`, `fatal`); or the evaluator failed: an error, **no `pytest_exit` completion
   marker** (the container or its shell died), a **container OOM kill**, or the **outer evaluator timeout** (the 300 s limit on `docker start`: the process is killed and its output discarded, so no marker can exist).
   Never a coding failure. (Limit: a candidate that really exhausts the container's 1 GiB looks the same as an engine kill and is not scored.) The **inner test timeout** is different: `timeout` inside the script stops pytest after
   120 s, the script still prints `pytest_exit=124`, and the run is judged normally as `unresolved_tests_failed`, a coding failure (a candidate whose tests run too long).
2. **`timed_out`**: the agent exceeded `--agent-timeout-s`; the partial work is still evaluated and recorded.
3. **`indeterminate`**: the cause is not established and the task did not resolve. Either (a) the orchestrator's run ended with `termination == "error"` ("could not be carried out": effGen uses it for provider failures, empty tasks,
   blocked runs, written-out tool calls and schema failures) and the recorded category is missing, `unknown`, `invalid_request` (may be a context overflow) or `empty_response`; or (b) **a worker did not complete for an unstated
   reason**: its `status` is not `completed` and its `termination` is not `stuck` or `tool_failed` (those state a genuine agent failure). effGen reports a worker's timeout and exception with `termination` unset, and a worker
   run that ended in `termination == "error"` carries no error category, so a transport failure inside a worker cannot be told apart. The worker check runs **before** the orchestrator's own category is read: a worker failure followed
   by an orchestrator `truncation` or `reasoning_only` is `indeterminate`, with the category kept in `model_failure`. An orchestrator transport error (rule 1) still wins. Not counted as a coding failure; read `agent/agent_result.json` (`model_failure`, `workers`)
   and the trace.
4. **`resolved` / `unresolved`** from the evaluator alone. `unresolved` is the only coding failure and covers: the agent finished or genuinely failed (`stuck`, `tool_failed`, or a model-side `truncation`, `reasoning_only` or `refusal`
   with every worker accounted for, whose category stays in `model_failure`), an empty or non-applying candidate, and failing tests including the inner test timeout. A task that resolves is `resolved` whatever the agent or a worker reported.

`HarnessResult.success` is recorded as `agent_reported_success` and never decides `resolved`. A reasoning-only or truncated generation is **not** called infrastructure: it is a model-side failure with its category recorded.
`model_failure` (`stop_reason`, `reason`, `category`, `type`, `retryable`, message cut to 300 characters) is effGen's own account of a failed orchestrator run; it is None for an answered run. The model endpoint is asked for the model's
identity (`GET /api/tags`, metadata only, no inference) before the agent starts; if it is unreachable or does not list the tag, the run stops there as an infrastructure failure.

## What a run records

`run.json`: the pinned snapshot hash, the declared base commit and the `basis` label (`snapshot-based`: the declared base commit is not in the snapshot, whose own HEAD is
`200f23b2…`; never read a result as a base-commit-exact reproduction), the configuration (provider, tag, context length, **requested** `max_tokens`/temperature/seed), `model_identity`
(Ollama digest, parameter size, quantization), `hashes` (prompt, rendered harness config, worker prompt, task statement), `provenance` (hashes of the driver and `tools.py`, repository
`HEAD` and `git status`, Python/effGen versions, host), container limits, the agent phase (command, exit code, seconds, timeout), the candidate metadata, the evaluation, and the
classification. Per-generation data (finish reason, token counts, seconds, tool calls, delegations) is in `agent/history/*/trace.jsonl`. **Not recorded:** GPU or other
hardware use on the model host, and the server-enforced output limit (docs/new-session.md U17); cost is wall-clock seconds and token counts only (a local model has no price).

## Execution boundary: what is verified, and what is not (2026-11-04)

Verified by building the real configured agents (scripted model, no inference) and probing their tools:

* **What the agent can execute: nothing.** The orchestrator holds `delegate` and `file_operations`; the worker `coder` holds `edit_file` and `file_operations`; there is no shell, no
  code executor and no web tool, and sub-agents are disabled. Repository code runs only in the evaluator container.
* **Host paths:** both tools are confined to `<run>/agent/workspace`. Reads of `../outside.txt`, the sibling `agent/harness.yaml`, `/etc/hostname`, `~/.ssh/id_rsa` (refused by name),
  listing `..`, and writes to `../escape.py` and `/tmp/...` were all refused, and no file was created. The workspace holds no symlink and the tools cannot create one.
* **Gold patch, test patch, `tasks.jsonl`, other tasks, the dataset, credentials:** not in the workspace (content-scanned before the agent starts), not in the config, prompt or command
  line, and not reachable through the tools. Inside the sandbox none of these paths exists and the environment is `PATH`, `HOME=/tmp`, `LANG`, `VIRTUAL_ENV` and two Python flags. One caveat: the project's venv is mounted read-only, and its `rich` is the fixed version, so the fixed line is readable to code running in the sandbox; the agent's tools cannot read it.
* **Evaluator separation:** the pristine copy, the evaluator's logs and the artifacts live beside, not under, the workspace; tests run only in a disposable container (no network, no mounts,
  no capabilities, non-root, 1 GiB, 2 CPUs, 256 processes, 120 s test and 300 s outer timeouts).
* **Limits on the agent:** `--agent-timeout-s` (the child's process group is killed), `--max-iterations` 8 per agent, 300 s per worker (cooperative), at most 4 delegations.

**Process sandbox (added 2026-11-04; independently reviewed the same day, which found and led to the fixes of the bridge timeout and the classification; not a security certification).** On a real run the agent child no longer runs as the plain host user. `run_pilot` refuses (status `refused`, nothing launched) unless `sandbox_problem()`
confirms the layers below; it never falls back to unsandboxed execution. Layers, outermost first:

1. `systemd-run --user --scope`: cgroup `MemoryMax=2G`, `MemorySwapMax=0`, `TasksMax=256`, `CPUQuota=200%` (the check reads back `memory.max` and `pids.max` from the scope).
2. `bwrap --unshare-all --die-with-parent --new-session --clearenv`: new user, pid, ipc, uts, cgroup and **network** namespaces; read-only binds of `/usr`, the venv, the uv interpreter,
   `src/` and the driver script; `/proc`, `/dev` and a `tmpfs` `/tmp`; the **only writable bind is the run's `agent/` directory** (plus the bridge directory); environment starts empty
   (`PATH`, `HOME=/tmp`, `LANG`, `VIRTUAL_ENV`, `PYTHONDONTWRITEBYTECODE`, `PYTHONNOUSERSITE`).
3. `prlimit`: per-file size 100 MiB, 1024 open files, core dumps 0.

**Docker agent sandbox (added 2026-11-06; redesigned in commit `0f17d7c`).** On a host where unprivileged user namespaces are unavailable (the DGX Spark), `--agent-sandbox docker` replaces the three layers above with one container. **The full current configuration (system `/usr`, `/lib`, `/lib64` mounts, the venv mount and the Python agent-child) has never run in a container, and the ID-based cleanup in `4067852` has only run against a stubbed `docker`.** A reduced-fidelity check of the `0f17d7c` code ran on 2026-11-13 on an x86_64 laptop under Docker Desktop (image `redis:7`, `sleep` instead of the Python child, system and venv mounts removed, everything else the driver's real create/inspect/verify/start/kill/remove path): `verify_agent_container` passed against the real `docker inspect`, a live container that ignored SIGTERM was killed, removed and confirmed gone in about 0.11 s, and `/repo` inside the container held only `src/` and the driver file (no dataset, `.git` or run directories). That check used the earlier cleanup-by-name code, and its scratch evidence was deleted afterwards, so it is not reproducible from artifacts. It says nothing about arm64, the DGX interpreter location or any real model. The 2026-11-05 runs used the earlier `docker run --rm` form (repository root and `/python` mounted, `--memory-swap 0`), and the current code is covered only by stubbed-`docker` tests in the default suite and one opt-in real-container test (skipped by default, not yet run).

* **Mounts** (from `docker_agent_command`; the repository root and `/python` are no longer mounted): writable `<run>/agent` → `/agent` and the bridge directory → `/bridge`; read-only `src/` → `/repo/src`, the driver file, the venv (`sys.prefix`) → `/venv`, `/usr`, `/lib` and `/lib64` (if present). `check_mount_sources` refuses any mount whose source is, contains or is inside the dataset, the evaluator directory or the pristine copy.
* **Interpreter handling (commit `4fde05e`).** If the venv's interpreter resolves under `/usr`, `/lib` or `/lib64`, nothing extra is mounted. Otherwise the resolved `sys.base_prefix` is mounted read-only at the path the venv uses, which can be an alias directory (for a uv-managed CPython, the `cpython-3.12-linux-x86_64-gnu` alias of the versioned directory), so that `/venv/bin/python` does not dangle in the container (`agent_interpreter_mount`, `rich3278_pilot.py:832`; added to the plan at `:931-936`, read back by `verify_agent_container` with the other mounts). The driver refuses early, before any docker call, when the interpreter is missing or unreachable, or when the base prefix is `/`, the home directory, a parent of home, equals or contains the repository root or a protected directory, or has no python executable. **UNVERIFIED:** this alias-path behaviour has not run in a real container. The no-model `sandbox-check` subcommand (`:1690`, `:1301`) exercises it through the same plan.
* **Create, inspect, verify, start.** The driver runs `docker create` (named `smol-agent-<run id>`, labelled, `--init`, `--pull never`, `--network none --ipc none --cap-drop ALL --security-opt no-new-privileges --read-only`, `--user <uid>:<gid>`, `--memory 2g --memory-swap 2g --cpus 2 --pids-limit 256`, `nofile`/`fsize`/`core` ulimits, `--tmpfs /tmp` and `--tmpfs /data` to mask the image's volume, `-w /agent`), saves `docker inspect` to `<run>/agent_container_inspect.json`, compares it with the request (`verify_agent_container`), and only then runs `docker start -a`. A mismatch refuses the run.
* **Cleanup.** The container is killed and removed (`docker kill`, `docker rm -f -v`, then a check that it is gone) in a `finally`; if that cannot be shown, the candidate is not collected (`ContainerCleanupError`).
* **Cleanup only touches this run's container.** Kill, remove and the gone-check run only after this run's `docker create` succeeded, and they address the container ID that create printed. If create fails (including a name conflict), nothing is killed, removed or inspected: a stale container with the same name now makes create fail instead of being removed. Known limit: if create errors after the daemon made the container (for example the client died), this run does not clean it up.
* **Docker Desktop is not supported for the Docker agent sandbox.** It does not share `/tmp`, `/lib64` or the host `/usr` with its VM, so the system mounts cannot be bind-mounted. It was seen once to report `Binds` with a `/host_mnt` prefix, which the fail-closed read-back check (`verify_agent_container`) refuses; a later run on the same machine reported plain paths and passed. The prefix is not always reported. Use a native Linux Docker engine.
* The model is reached through the same unix-socket bridge (`/bridge/m.sock`), and the child is still one `agent-child` process. The image must already exist locally (`--agent-image`, default `redis:7`); the driver never pulls.
* **Accepted risk:** the mounted venv contains the `rich` package, whose `rich/ansi.py` already holds the reference fix. The model's tools cannot read it, but code execution inside the container could. It is not mitigated.
* **`run.json` shape:** `agent.command` is the `docker create` argv; `agent.start_command` (a separate key next to it, not nested in it) is the `docker start -a <name>` argv; `agent.container` holds `name`, `id`, `inspect_file`, `isolation` and `cleanup`; `sandbox.limits` holds the Docker flag values (the same dict builds the flags); `sandbox.isolation` summarizes the inspect result. The 2026-11-05 records have the older shape (`agent.command` = the `docker run` argv, `sandbox.limits` from the bubblewrap limits).

**Writable locations inside the sandbox:** `<run>/agent` (the only bind of host storage, which holds the workspace), the bridge directory (`/bridge`), and ephemeral memory-backed `/tmp`, `/dev/shm` and the root tmpfs (charged to the 2 GiB cgroup, gone at exit). Everything else is read-only or absent.

**Network policy.** The sandbox's network namespace has `lo` plus unconfigured kernel tunnel devices and an empty route table: there is no route out (observed in a reduced-fidelity container test on x86_64, not on the DGX). The model endpoint is reached through a unix-socket bridge: a host-side `ModelBridge` forwards one socket to exactly one `host:port` (from
`--base-url`), and an in-sandbox loopback listener on the same port relays into it. Nothing else is routable; there is no DNS. **The bridge is a raw TCP relay, not an HTTP filter:** the child could send any request to that one
host:port (for Ollama that includes its management API, such as `/api/delete`), and the agent's tools cannot send HTTP at all. Timeouts: connecting to the endpoint is bounded at 10 s and the timeout is removed once connected, because a non-streaming generation sends nothing
until it ends (a model call over 10 s was cut off before 2026-11-04); a connection that moves no bytes for 600 s is closed; shutting the bridge down closes every relay in flight.

**Verified by probes (tests `test_escape_probes_inside_the_sandbox`, `..._only_through_the_bridge`, `..._limits_are_enforced...`, `..._wall_clock_kill...`; skipped, not failed, where the tools are
missing):** dataset, docs, repo `.git`, `~/.ssh`, `~/.bashrc`, the evaluator directory, `/etc/passwd`, `/etc/shadow`, `/run`, `/var` are absent; writes to `/usr`, the venv and `src/` fail;
external addresses, a host loopback service, a host abstract unix socket and `docker.sock` are unreachable; capabilities are zero and `NoNewPrivs` is 1; `mount` onto the sandbox's own filesystems fails, though a nested user namespace can still mount a private tmpfs, remount nothing writable and reach no external network; the environment holds
no credentials; a 1 GiB allocation is killed under a 256 MiB cap; threads stop under the task cap; a 4 MiB write fails with EFBIG at a 1 MiB limit; open files stop at the limit; after the
wall-clock kill no child, grandchild or sandbox process remains. An end-to-end run against a stand-in HTTP server (not a model) showed the child reaching the endpoint only via the bridge.

**Assumed, not verified:** the host kernel's user-namespace behavior and bubblewrap are correct (the sandbox shares the host kernel; **no seccomp filter**, so the remaining syscall
surface is the kernel's); the model endpoint named by `--base-url` is trusted; read-only mounts expose the venv, interpreter, `src/` and the driver script to reading; the agent
directory's total disk use is **not capped** (only per-file size and cgroup memory, which covers tmpfs); an OOM kill shows up as an `infrastructure_failure` in the classification, not a
coding failure. The evaluator container is unchanged, so the reference check was not re-run. Still true: the tools themselves can only read and edit files in the workspace, and the
ordinary-character escaping limitation of `edit_file` is unchanged. This is a single-pilot boundary, not a claim of safety against a determined hostile program.

## Running another attempt: preconditions (nothing here is started by the driver without you)

1. The model server must be reachable at `--base-url` (the DGX model is only reachable through an SSH tunnel you open; nothing local serves it) and list the tag.
2. `prepare-env --out <out>` (offline) and then `reference-check --out <out>` must reproduce 16 failed / 7 passed and 23 passed in the container (reproduced on 2026-11-04 with this code).
3. `run ... --allow-model-call` (see Usage). Read `run.json` → `classification` before reading anything else.

## Isolation: what holds and what does not

* **The agent cannot read the gold patch, the test patch or `tasks.jsonl`:** none of them is ever written into the agent workspace or into the config, prompt or
  command line the child gets; a content scan (not an extension check) of the workspace runs before the agent starts, and the sandbox above removes every other host path from the child's view.
  `file_operations` refuses paths outside the workspace and symlink escapes, and the workspace contains no links. The tools' own confinement is path validation (check-then-use races, hard links); the process boundary is the sandbox.
* **Repository code only runs in the evaluator container:** no network, no host mounts, no volumes, no Docker socket, all capabilities dropped, no-new-privileges,
  non-root user, memory/CPU/PID/file limits, a tmpfs `/tmp`, removed after use. The driver refuses to run when `docker inspect` shows anything else, never pulls
  an image, and never falls back to the host. Python, site-packages and the workspace are copied in with `docker cp`.
* **Limits:** the container shares the Docker engine (on the validation host: Docker Desktop, a VM); there is no hard disk quota; the daemon is rootful. A
  hostile patch could forge pytest output (the official JUnit XML check is forgeable the same way). The agent directory has no total size quota (per-file 100 MiB only); put `--out` on a tmpfs such as `/tmp` for a hard
  cap through the cgroup memory limit (measured: a 600 MiB write under a 256 MiB cap is killed). Patch text is split with `str.splitlines`, so a candidate that adds a form feed, NEL or U+2028 character may not round-trip
  (the `rich_3278` tree has none).

## Deviations from the competition setup (also written into every `run.json`)

1. The agent has `file_operations` and `edit_file` (this project's exact-replace tool; the competition agent has `read_file`/`edit_file`/`write_file` of its own) and no shell, test runner or code-graph tools. Compared with the first version of this pilot, the worker gained `edit_file` and a short system prompt; model, context length, `--max-tokens 768`, temperature, seed, iterations and timeouts are unchanged.
2. No `.git` in the agent workspace or the evaluator copy.
3. A minimal reconstruction of the task prompt: header, the problem statement as given, and a short environment note.
4. Protected-file rules (`test_*.py`, `*_test.py`, `.py` under `tests/`/`test/`/`testing/`, `conftest.py`, `pytest.ini`, `pyproject.toml`, `tox.ini`, `setup.cfg`,
   `.pytest.ini`, `sitecustomize.py`, `usercustomize.py`, `_swegemma_stubs.py`, `*.pth`) are reproduced from the competition README's text; the verifier code is not in
   the dataset. A candidate's changes to them are dropped and listed, and the official test files always win.
5. Verdict from pytest's exit code and `-rA` output with the pinned command `python -m pytest <test files> -q -rA -p no:cacheprovider` (the validated runs used `-ra`).
6. The declared `base_commit` `4c5d305b…` is **not** in the snapshot (its HEAD is `200f23b2…`, a merge of PR #3232). Exact base-commit reproducibility is not established.
7. The problem statement is a PR description and quotes the intended fix: this is a pipeline smoke test, not a measure of coding ability.

## Known limits and unverified

* Verified outside the suite (not in the default tests): `prepare-env` and `reference-check` through the real `DockerRunner` and Docker Desktop (baseline 16 failed / 7 passed, gold 23 passed,
  repeated on 2026-11-03 and 2026-11-04; the suite itself only tests the isolation checks and the command construction with a fake `docker`). Also outside the suite: the three real `run`s above
  (agent sandbox in its pre-redesign form, model identity, empty-patch classification). `fetch_model_identity` has been exercised against a real Ollama server by those runs. **Not verified:** the full current Docker agent sandbox configuration in a container (only a reduced-fidelity check ran; see above), a non-empty agent-produced candidate through the evaluator, a task that resolves, an evaluator reference check on
  ARM64, and a real-model tool call that changes a file.
* **Backslashes at the tool-call boundary.** The model reads files as raw text, but a tool call is JSON: a backslash copied as it reads (`\x1b`) is not valid JSON, and
  effGen then falls back to `ast.literal_eval`, which turns it into a real ESC character. `edit_file` refuses an `old_text` that fails to match and holds such a control
  character, and refuses a `new_text` that holds one (other than TAB, LF and CR); its description and refusal messages state facts only and no longer tell the model to double backslashes. The escaping guidance
  (write a backslash as `\u005c`) is now only in the worker prompt. Not guarded: escapes that decode to ordinary characters (for example `\x41` becomes `A`, a single-backslash
  `\t` or `\n` becomes TAB or LF) cannot be told apart from intended text after parsing and are written silently; tests pin this limit. The tool therefore also cannot insert a control character on purpose.
  Whether a real model gets this right is unobserved: the 2026-11-11 Gemma run (e0cf) failed at this interface (malformed or double-escaped JSON) with the earlier wording, and no run has changed a file.
* Generation limits are the values REQUESTED. The server-enforced output limit on Ollama is unresolved (docs/new-session.md U17); do not quote `--max-tokens` as an
  enforced cap.
* The patch splitter treats a removed line that starts with `-- ` directly followed by an added line starting with `++ ` as a file header. Not relevant to the
  rich_3278 inputs; fail-closed handling of such a patch is not implemented.
* Single task, and the two runs on record are single attempts with different models, not repeat-run statistics.
