# New Session Handoff

_Last updated: 2026-10-17, end of the T4.2 session. Everything below was checked against the working tree on that date._

## 1. Project overview

**smol-harness** is a minimal multi-agent coding harness built on **effGen 1.3.0**.

- A larger-LLM **orchestrator** delegates self-contained tasks to smaller-LLM **workers** through one `delegate` tool, which works like a smolagents managed agent.
- Each worker is a fresh `effgen.Agent` with its own context, model, tool allowlist and limits.
- Models come from Ollama (via effGen's OpenAI-compatible adapter) or local Transformers (bf16/fp16, bnb 8-bit, bnb NF4).
- Tools: effGen's workspace-confined `FileOperations` (**implemented, T4.1**, via `tools.build_tools`) plus a sandboxed shell, `ShellTool`, over effGen's `SubprocessSandbox` (**implemented, T4.2; Linux only; trusted local development only; unresolved risk R1, see §3 and §6**).

Details: `docs/architecture.md`. effGen facts: `docs/effgen-api-audit.md`. Plan: `docs/implementation-plan.md`.

## 2. Repository map (verified against the working tree, 2026-10-16)

Source modules, tests and configuration exist; there is **no CLI yet**.

| Path | Purpose |
|---|---|
| `CLAUDE.md`, `AGENTS.md` | Instructions for Claude Code / any contributor. **Both are git-ignored** (`.gitignore` lines 5-6), so changes to them never show in `git status` or commits |
| `docs/new-session.md` | This handoff |
| `docs/implementation-plan.md` | Milestones, tasks, statuses, completed-work log (§4) |
| `docs/architecture.md` | Design and decisions, with the as-built notes |
| `docs/effgen-api-audit.md` | effGen facts (Part 1, incl. §1.12 on local-model tool calling and context budget), harness integration status (Part 2), per-task history (Part 3) |
| `pyproject.toml`, `uv.lock`, `.python-version` | uv project: Python 3.12, `effgen==1.3.0`, `pyyaml`; dev group: `pytest`; markers `ollama`, `gpu`, `real_model` |
| `src/smol_harness/delegation.py` | `WorkerResult`, `WorkerRecord`, `WorkerRegistry`, `DelegateTool`, `DeadlineMiddleware`, `WorkerTimeout`, `bounded`, `validate_limits`. Worker timeout, `max_delegations`, `max_report_chars`; every report has a unique `delegation_id` |
| `src/smol_harness/harness.py` | **Assembly (T5.0):** `build_harness_from_config(config, *, cache=None, tool_calling_mode="react") -> Harness` (tools first, then models via one `ModelCache`, then agents; stated effGen settings), `Harness.close()` / context manager (closes only a cache it created), `HarnessClosedError`, `initialize_runtime()` (the one call an application makes first: installs the effGen loop patch). `build_harness(..., owned_cache=...)`, `Harness.run(task) -> HarnessResult` (fresh orchestrator and record list per run; refuses with `EffgenCompatError` if the patch is missing; a stopped orchestrator is a failed `HarnessResult` in both `raise_on_error` modes, other exceptions propagate with `exc.worker_records`, review F3). **effGen loop patch (F1):** `install_effgen_loop_compat()` (explicit, idempotent, **process-wide**), `effgen_loop_compat_installed()`, `uninstall_effgen_loop_compat_for_tests()`, `EffgenCompatError` |
| `src/smol_harness/config.py` | YAML to frozen dataclasses with validation: `load_config`, `loads_config`, `parse_config`, `ConfigError`, `ModelSpec`, `AgentSpec`, `Limits`, `HarnessConfig`, `hf_placement_problem`, `KNOWN_TOOLS` |
| `src/smol_harness/models.py` | `build_model(ModelSpec)`, `loader_args`, `ModelCache`, `ModelLoadError` (CPU/CUDA placement, quantization mapping, post-load checks) |
| `src/smol_harness/tools.py` | `build_tools(names, workspace)`, `resolve_workspace` (rejects missing/non-directory/inaccessible workspaces, filesystem roots, `$HOME` and its ancestors), `ToolBuildError`, `TOOL_NAMES`, `ShellTool`, `ShellSandboxError`, `require_shell_host`. `file_operations` = effGen `FileOperations` with the kept root read back; `shell` = `ShellTool` (see plan §4 T4.2). Not yet called by `build_harness` |
| `src/smol_harness/__init__.py` | Package docstring only |
| `configs/example.yaml` | Mixed-provider example (Ollama orchestrator, HF NF4 worker), finite limits |
| `tests/conftest.py` | `ScriptedModel` fake effGen model, `scripted_model` fixture, and the `--run-real-models` opt-in option / skip hook |
| `tests/test_scripted_model.py` (15), `test_delegation.py` (22), `test_e2e_delegation.py` (23), `test_limits.py` (77) | M1/M2: fake model, delegation layer, end-to-end orchestrator-worker flow, limits and failure propagation |
| `tests/test_config.py` (201), `test_models.py` (110), `test_hf_device.py` (8) | M3: configuration, model factory and cache, CPU/CUDA placement against effGen's real engine with a locally generated tiny model |
| `tests/tool_probe.py`, `test_tool_probe.py` (7), `test_lfm_parser_compat.py` (13) | Tool-calling measurement helper and its scripted validation; model-free pins of how effGen 1.3.0 reads LFM tool-call formats |
| `tests/test_harness_config.py` (46) | T5.0: configuration mapping per agent, allowlists, model sharing and ownership, cleanup on failure, `close()`/run-after-close, tool failures before model loading, complete scripted flows, limits, the fresh-process start-up sequence, the F8 shell/deadline behaviour |
| `tests/test_orchestrator_failure.py` (28) |
| `tests/test_loop_guards.py` (35) | Review F1/F2 with the real effGen loop and real tools (all pass now), plus the patch contract: scope (marked tools only), idempotent install, loud failure on an incompatible effGen (6 breakages + unknown replacement), `Harness.run` refusing without the patch, no automatic restoration, test-only restoration, no leak between runs, runaway loops still bounded. `conftest.py` installs the patch in a session fixture |
| `tests/test_tools.py` (121) | M4: tool factory, workspace validation and dangerous roots, `file_operations` confinement, the real `SubprocessSandbox` shell (writes, reads, network, sockets, timeout, children, memory, fail-closed, allowlists), scripted harness runs with real tools, effGen regression pins. Sandbox tests are `linux_only`; they fail rather than skip on a Linux host without user namespaces |
| `tests/test_integration_optin.py` (2), `test_import.py` (2) | Guard that real-model tests cannot run without the flag; effGen version/import check |
| `tests/integration/test_liquid_cpu.py` (1), `test_liquid_tools.py` (2) | **Opt-in** real-model tests for `LiquidAI/LFM2.5-350M` (CPU smoke test; native vs ReAct tool calling and context-window check) |

Not created yet: `src/smol_harness/__main__.py` (T5.1), `tests/integration/test_ollama.py` / `test_hf.py` / `test_mixed.py` (T3.3 to T3.5).

## 3. Current project state

Default suite: **722 passed, 3 skipped** (the 3 opt-in real-model tests; about 81 s); `uv lock --check` passes.

**Implemented and verified (with the tests above):**

- **M1, delegation (scripted models):** an orchestrator `Agent` calls a `delegate` tool; each delegation builds a *fresh* worker `Agent` that shares the configured model instance but not its conversation; the parent sees only a JSON report; records are per run; depth is 1.
- **M2, controls:** cooperative per-worker deadline (checked before model/tool calls, never interrupts one in flight), structured failure reports for every worker failure kind (the parent is never ended as `tool_failed` by them), `max_delegations` (refusals still get unique ids), bounded reports (`max_report_chars`, valid JSON, `truncated` flag), effGen's own `max_iterations`.
- **M3 so far:** validated YAML configuration (T3.1); model factory and cache with explicit quantization and placement (T3.2); explicit `device: cuda | cpu` for HF models (T3.2a); `LiquidAI/LFM2.5-350M` loaded and run on CPU (fp32) through `build_model`, and tool-calling compatibility measured (T3.2b, T3.2c).
- **M4 so far (T4.1):** `build_tools(names, workspace)` builds exactly the allowlisted tools (fresh instances; `[]` = none; unknown, web-like, duplicate and `delegate` names refused; `shell` is `ShellTool`). `file_operations` is confined to the validated workspace; `../`, absolute, symlink (file, directory, dangling) escapes are refused for reads and writes. This is **path validation, not a sandbox** (limitations in plan §4 T4.1).
- **M4 (T4.2):** `shell` runs `bash -c <command>` in effGen's `SubprocessSandbox`: starts in the workspace, writes only inside it, no network (DNS included), a timeout that kills every process the command started, `ulimit -v` memory limit (1g default), output tail-truncated, stdin empty; it refuses to build off Linux or without `bash`/`unshare`/`timeout`, and a pre-flight must show the guarantees before the first model command. Details, measurements and limits: plan §4 T4.2. **This is not a complete security boundary:** reads are not confined, Unix sockets outside `/run` are reachable (R1, unresolved), DNS is blocked only through the network namespace plus the hidden `/run`, Windows/WSL2 are unverified. **Accepted scope (2026-10-17): trusted local development and controlled test repositories only. Untrusted repositories and adversarial agent workloads must NOT use it; they need a dedicated, independently isolated container or VM whose controls are verified separately. No seccomp or other extra isolation will be added in the MVP.**

**Liquid AI results (real model, CPU; details in plan §4 and audit §1.12 and Part 3):**

- Loads in about 3 to 5 s, 354M parameters, all on CPU, no CUDA memory or context; a one-word answer takes about 0.2 to 0.3 s. Measured on an actively cooled laptop, **not** the fanless edge device.
- **ReAct tool calling worked 2/2** (the tool really executed and its result returned); **native tool calling worked 0/2** (the model writes its default Pythonic `<|tool_call_start|>[add(a=.., b=..)]<|tool_call_end|>`, which effGen's native parser cannot read, so the tool never ran). **Recommended mode for this model: `tool_calling_mode="react"`.**
- Context: effGen reports 128,000 (from `config.json`); the model card says 32,768. 32,768 is used only as a **conservative planning limit taken from the model card, not a tested maximum**. With `max_context_length=32768` effGen's prompt budget is 20,889 tokens (101,836 without it).

**Not implemented / remaining MVP work:** the CLI (T5.1; next), the toy-repo end-to-end test (T5.2), T4.3 only if needed, real-model runs: Ollama smoke test (T3.3, blocked), bitsandbytes GPU load (T3.4), mixed-provider real delegation (T3.5), YAML fields for generation settings, HF `max_context_length`, shell limits, `duration_s` in reports. (Old list, kept for reference:)

**Explicitly not done (do not assume):** no real multi-agent run with any real model (all delegation tests use scripted models); the shell has only run on this Linux laptop (no Windows or WSL2 run, no real-model run); nothing builds tools from a `HarnessConfig` inside the harness yet (tests call `build_tools` directly); no Ollama server has been contacted; no bitsandbytes load has been run; the Kaggle dataset is unverified; native tool calling for LFM is unsupported.

**Known limitations (by design):** sequential workers only; depth 1; the worker timeout is cooperative; no orchestrator deadline; FP4 unsupported (effGen hard-codes NF4); web search excluded.

## 4. Most recent session

- **Latest (2026-10-17, T5.0 done): a validated `HarnessConfig` now becomes a runnable, closable harness.** `build_harness_from_config(config, *, cache=None, tool_calling_mode="react")`: tools for every agent first (a bad workspace, dangerous root, forbidden or unknown tool, or unavailable shell fails **before any model loads**), then models through one `ModelCache` (equal specs share an instance), then one `AgentConfig` per agent with the effGen settings stated, not defaulted (review F5: `raise_on_error=False`, `enable_memory=False`, `capability_probe=False`, `enable_sub_agents=False`, the tool-calling mode for all agents, Ollama `max_context_length` = `context_length`, agent-side only), workers registered with the configured limits. **Ownership:** a harness closes only a cache it created (each model unloaded once, idempotent); an injected cache stays the caller's; a failed assembly releases what it created and the original error stays the visible one; `run()` after `close()` raises `HarnessClosedError`. **Start-up:** call `initialize_runtime()` first (installs the process-wide effGen loop patch; nothing else installs it, `close()` never uninstalls it, and the builder and `run()` refuse without it); proven in a fresh interpreter with no pytest fixtures. **F8 mitigated:** a shell command in a worker can no longer outlive that worker's deadline (6.4 s -> 2.4 s on the reproduction): `DelegateTool` publishes the active deadline in a `ContextVar` and the shell bounds its effective timeout by the remaining whole seconds; a generation already in flight is still not interrupted, and a worker that overruns in its last call is reported `completed` (documented, tested). Tests: `tests/test_harness_config.py` 46 passed; full **722 passed, 3 skipped**; `uv lock --check` OK. Details: plan §4 (T5.0), review Follow-up 001-E.
- **Uncommitted at the end of this session:** `src/smol_harness/{harness,delegation,tools}.py`, `tests/test_harness_config.py` and the docs. Everything before this (through F3) is committed (`e379ad3`).
- **Earlier (2026-10-17, F3 fixed): all three P1 review findings (F1, F2, F3) are closed.** `Harness.run` now returns a failed `HarnessResult` (real `termination`, `response.stop_reason`, this run's worker records) when effGen stops the orchestrator, whatever `raise_on_error` says; effGen raises `RunStoppedError` only for `outcome == "stopped"`, carrying the same response `raise_on_error=False` would return. Failures that have **no response** (provider errors, guardrail blocks, a model that raises) are *different* exceptions under `True`: they propagate unchanged, with this run's records attached as `exc.worker_records` (plus a note), and under `False` they are failed results (an effGen asymmetry; the CLI must handle both). The response is validated (`success is False`, has `output`); a malformed `RunStoppedError` propagates instead of producing a made-up result. 28 new tests in `tests/test_orchestrator_failure.py`. Tests: full **676 passed, 3 skipped**; `uv lock --check` OK. New P3 finding F13 (a worker's response-less `RunStoppedError` leaves no `WorkerRecord`). Details: review Follow-up 001-D. **Earlier today:** F1 fixed by the explicit process-wide `install_effgen_loop_compat()` (001-C), F2 by self-describing tool results (001-B).
- **Previous session (T4.2), summarized below:**

- **Goal:** T4.2 only: a sandboxed `shell` tool on effGen's sandbox, verified empirically, plus the T4.1 correction (dangerous workspace roots). No models, no Ollama, no CLI, no effGen change.
- **Result:** **Completed against the plan's acceptance criteria** (all 10 requested behaviours verified with the real sandbox, see plan §4 T4.2) **for trusted local development only; the isolation is not complete (R1 unresolved) and the user accepted that scope on 2026-10-17.** Two effGen defects were found and worked around:
  1. **Unix sockets:** effGen's net namespace does not isolate pathname Unix sockets. From inside the sandbox I could *connect* (nothing sent) to the Docker daemon socket (user is in `docker`), the session and system D-Bus, the systemd user manager, gpg-agent; systemd-resolved answered DNS. Fixed for everything under `/run` by an empty read-only tmpfs; **sockets elsewhere (under `$HOME`, host `/tmp` when the workspace is under `/tmp`) remain reachable.**
  2. **Timeout:** effGen's own timeout path kills only the outer `unshare`, leaves descendants running and then hangs until they exit. Fixed by timing out inside the PID namespace with `timeout -k 2 N`.
- **Files:** modified `src/smol_harness/tools.py`, `tests/test_tools.py`; docs: plan, this file, audit (§1.9, Part 2, Part 3), architecture §9 (shell row; the shell is a `BaseTool` over `SubprocessSandbox`, not a `CodeExecutor` subclass). Nothing is committed from this session (the T4.1 work was committed as `6ff4666` before it).
- **Tests:** `uv run pytest tests/test_tools.py -q` → 121 passed (38 s); `uv run pytest -q` → 601 passed, 3 skipped (65 s); `uv lock --check` OK. Mutation check: removing the `/run` mask fails the socket tests. No real-model test run.
- **Safety of the work itself:** every boundary test ran in `pytest` temp directories or throwaway dirs under `/tmp`; socket probes were connect-only; no exploit was attempted; exploration scripts are in `/tmp/fo-scripts` (not in the repo).
- **Problems:** none open; R1 is an accepted, unresolved limitation. (My first exploration script hung on effGen's timeout defect; I killed it and its test `sleep`s.)
- **Decisions:** `BaseTool` over `SubprocessSandbox` (no private attributes); `auto`/`off` can never be selected; pre-flight before the first command; unknown parameters rejected; failing commands return failed results with their output in `error` (effGen shows a model only `error` on failure).
- **Known limitations of this work:** see plan §4 T4.2 "Not enforced". Windows/WSL2 untested; no config fields for memory limit or max timeout yet; `ulimit -u 256` is per user.

### Configuration schema (as built)

```yaml
workspace: ../workspace            # relative to the YAML file; need not exist yet
limits:                            # optional; all finite; shown values are the defaults
  max_delegations: 4               # 1..100
  max_report_chars: 4000           # 346..1_000_000
  worker_timeout_s: 300            # >0, <=86400, cooperative
  max_depth: 1                     # must be 1
orchestrator:                      # name is fixed ("orchestrator")
  model: {...}                     # see below
  tools: [file_operations]         # required; `delegate` is implicit (may be listed, not stored)
  max_iterations: 10               # 1..200
  system_prompt: "..."             # optional
workers:                           # non-empty LIST (duplicate names are detectable)
  - name: coder
    description: "..."             # required
    model: {...}
    tools: [file_operations, shell]  # required, `[]` allowed; no `delegate`, no web tools
    max_iterations: 8              # 1..200
    system_prompt: "..."           # optional
```

Model, by provider (unknown or other-provider keys are rejected):

| provider | keys | quantization |
|---|---|---|
| `ollama` | `model` (tag), `context_length` (**required**, 256..10M), `base_url` (default `http://127.0.0.1:11434/v1`), `quantization` | `server` only (default) |
| `hf` | `model` (repo id/path), `quantization` (**required**), `device` (`cuda` default \| `cpu`), `dtype` (see below) | `none`, `bnb-8bit`, `bnb-4bit-nf4` |

hf placement rules: `device: cuda` → any quantization; `dtype` `bf16` / `fp16` / omitted. `device: cpu` → `quantization: none` only and `dtype` **required** (`fp32` or `bf16`). Everything else is rejected.

## 5. Current task and next steps

### Next task: **T5.1, CLI** (status: Not started; awaiting approval)

- **Objective (plan §2 M5):** `uv run python -m smol_harness --config configs/example.yaml --task "..." --json` prints the final output plus worker records; non-zero exit on orchestrator failure. New file `src/smol_harness/__main__.py` (the module the architecture assigns), plus tests.
- **Start-up sequence the CLI must follow:** `initialize_runtime()` → `load_config(path)` → `with build_harness_from_config(config) as harness:` → `harness.run(task)`. Both `build_harness_from_config` and `Harness.run` raise `EffgenCompatError` if `initialize_runtime()` was not called; nothing installs the patch implicitly and nothing uninstalls it.
- **Exit codes / output contract:** a failed `HarnessResult` (`success=False`, with `termination` and `response.stop_reason`) and a propagated orchestrator exception (provider error, guardrail block, a model that raises; it carries `exc.worker_records`) both mean non-zero. Config errors (`ConfigError`) and tool/model build errors (`ToolBuildError`, `ModelLoadError`, `EffgenCompatError`) are start-up failures, also non-zero, with the message shown.
- **Should also report** the effective per-agent settings (tool-calling mode, `temperature` 0.7 and other effGen defaults that the config cannot set, review F7) so that experiment runs are interpretable.
- **Read only these:** plan T5.1; `src/smol_harness/harness.py` (`initialize_runtime`, `build_harness_from_config`, `HarnessResult`, `Harness.close`), `config.py` (`load_config`, `ConfigError`), `configs/example.yaml`.
- **Verify with:** `uv run pytest -q` (baseline 722 passed, 3 skipped), `uv lock --check`; the CLI itself with scripted/patched models only (no real model unless asked).
- **Do not repeat:** the effGen architecture audit, the sandbox investigation, or the F1/F2/F3 work. **Not part of it:** T5.2, real models, YAML fields for generation settings.

### Afterwards / open

- T5.2 (toy-repo smoke test, trusted repository only). The wiring step is done (T5.0).
- **Blocked:** T3.3 (no Ollama), T3.4 (`bitsandbytes` not installed), T3.5 (needs both).
- **Decisions for the user (nothing started):** approve or reject a parser adapter for LFM's Pythonic call format or an engine change to send a system-role instruction (native mode); run the opt-in tests on the fanless edge device.

## 6. Known issues and warnings

### Known bugs or broken tests

**Open:** none of the P1s. F13 (P3: a worker's response-less `RunStoppedError` leaves no `WorkerRecord`, `DelegateTool._delegate`), and review findings F4 to F12 (see `docs/reviews/opus-review-001.md`). **Process-wide effGen patch (F1):** call `install_effgen_loop_compat()` once before any agent runs; never reload `smol_harness.harness` after installing; never install or restore during a run or from several threads; it is pinned to effGen 1.3.0 and fails loudly otherwise. **`Harness.run` contract (F3):** a stopped orchestrator is a failed result; other orchestrator exceptions propagate with `exc.worker_records`. The suite is green (676 passed, 3 skipped).

### Unverified assumptions

| ID | Assumption | Where it gets verified |
|---|---|---|
| U1 | ~~A scripted ReAct fake drives effGen's loop deterministically~~ **Verified in T1.1** for the happy path; nudges can consume extra turns on error paths, so use `repeat_last` | done |
| U2 | ~~The orchestrator sees the full JSON string from `DelegateTool`~~ **Verified in T1.3** | done |
| U3 | ~~A failed worker does not end the parent as `tool_failed`~~ **Verified** in T1.3 and T2.2 for the failure kinds tested | done |
| U4 | ~~Middleware exceptions stop a worker run~~ **Verified in T2.1** for `before_model_call` / `before_tool_call` (ReAct). Native tool calling, streaming and `run_async` paths are untested | done / open |
| U5 | Tool calling through Ollama `/v1` for the chosen tags; Ollama's `num_ctx` must match the configured `context_length` | T3.3 |
| U6 | `bitsandbytes` installs and works with `torch 2.14.1+cu130` on this GPU (it is not installed) | T3.4 |
| U7 | Orchestrator and worker fit in 8 GB VRAM together | T3.4 / T3.5 |
| U8 | ~~effGen's `SubprocessSandbox` initializes here and 256m is enough for `pytest`~~ **Verified in T4.2** on this host: it works, and a trivial `pytest` fits 256m; default raised to 1g; larger toolchains untested | done / open for big suites |
| U15 | The shell sandbox works on Windows 11 via WSL2 (user namespaces, `unshare`, `timeout`, repo on the WSL filesystem) | when the Windows machine is available |
| U16 | What a hostile command could do through the residual Unix sockets (R1) and unconfined reads | not planned: out of scope, the shell is for trusted use only |
| U9 | ~~`effgen==1.3.0` installable~~ **Verified in T0.1** | done |
| U10 | Kaggle dataset contents (repo snapshots, tests, patches) are **not** assumed | T6.1 |
| U11 | ~~Invalid `delegate` arguments end the parent as `tool_failed`~~ **Answered in T1.3: they do not** | done |
| U12 | LFM2.5-350M with ReAct is reliable beyond 2 seeds of one task with one two-integer tool (multi-step, string arguments, several tools, as a delegated worker) | not planned yet |
| U13 | Behaviour near the 32,768-token card limit (and the 128,000 the config declares); neither was tested beyond about 2,000 characters | only if needed |
| U14 | Timing, RAM and temperatures on the fanless edge device (laptop numbers only so far) | needs the device |

### Technical risks

- The worker timeout is cooperative: a long local generation or tool can overrun by one call.
- effGen's native tool calling cannot read LFM's default call format; always set `tool_calling_mode="react"` explicitly for that model (`auto` was not tested).
- In ReAct the model invents `Observation:` text after its action; effGen discards it and runs the real tool, but a task where the invented value matters could behave differently.
- effGen retries a failing generation 3 times with 0.5 s and 1 s sleeps and no setting to change it (a dead backend costs about 1.5 s per delegation).
- effGen stops a run whose tool returns the same text twice (`repeated_tool_result`); `delegation_id` in every report prevents this for delegation.
- effGen's `BashTool` is not a sandbox; `ddgs` / `duckduckgo-search` are installed through effGen and must never be registered as tools.
- effGen's `FileOperations` silently confines to the cwd (or `EFFGEN_WORKSPACE`) when its allowed directory is missing. Always build it through `tools.build_tools`, never directly. Its confinement is path validation only (check-then-use race; hard links undetected).
- **R1 (unresolved, accepted for trusted local development only; do NOT run untrusted repositories or adversarial agents with the shell, use a dedicated isolated container or VM instead):** pathname Unix sockets outside `/run` (under `$HOME`; host `/tmp` if the workspace is under `/tmp`) are reachable from the shell; reads are unconfined; the shell reads files `file_operations` refuses (`.env`).
- effGen's `SubprocessSandbox` silently runs plain `bash` off Linux or without user namespaces, and its timeout path hangs on surviving children; both are worked around in `ShellTool` (plan §4 T4.2). Never call it directly.
- `file_operations` refuses extensionless files (`Makefile`, `README`, `.gitignore`); a coding worker cannot edit them with it.
- Disk: `/home` had about 7.8 GB free after the 714 MB model download.
- `AgentResponse.tokens_used` for a run (32) differed from the engine's count for the same prompt (2); not investigated.
- `CLAUDE.md` and `AGENTS.md` are git-ignored, so edits to them are not in version control.

## 7. Quick start

All commands below were run during this project (Linux, uv 0.11.29, 2026-10-16). There is **no CLI yet**; the harness is exercised through the tests.

```bash
cd /home/ch_kartik/Projects/Active/smol-harness
uv sync --locked                 # .venv with Python 3.12 + effgen 1.3.0 (about 2 min cold: torch/CUDA wheels)
uv run python -c "import effgen; print(effgen.__version__)"   # -> 1.3.0
uv run pytest -q                 # default suite -> 722 passed, 3 skipped (real-model tests skipped)
uv run pytest tests/test_limits.py -q                       # a single file works the same way
uv run pytest tests/test_tools.py -q                        # tool factory, file_operations, sandboxed shell -> 121 passed (about 40 s)
uv lock --check                  # -> Resolved 157 packages (lockfile consistent)

# Opt-in real-model tests (CPU; network is blocked during load and inference)
uv run pytest tests/integration/test_liquid_cpu.py --run-real-models -s     # first run downloads ~714 MB into ~/.cache/huggingface; writes /tmp/smol-harness-lfm-cpu-smoke.json
uv run pytest tests/integration/test_liquid_tools.py --run-real-models -s   # about 35 s; writes /tmp/smol-harness-lfm-tools.json
```

`SMOL_RUN_REAL_MODELS=1` is an equivalent opt-in. Without the flag these tests are skipped.

Reference clones for source reading (outside the repo, read only; do not run code from them):

```bash
git clone --depth 1 https://github.com/ctrl-gaurav/effGen /tmp/refsrc/effgen/repo
git clone --depth 1 https://github.com/huggingface/smolagents /tmp/refsrc/smolagents/repo
git clone --depth 1 https://github.com/badlogic/pi-mono /tmp/refsrc/pi/repo
```

The installed copy of effGen under `.venv/lib/python3.12/site-packages/effgen/` is the source of truth for line references in the audit.
