# Implementation Plan (living document)

How to use this document:

- Statuses are Not started, In progress, Blocked, Implemented (unverified) and Completed.
- Mark a task **Completed** only after its acceptance criteria were verified by running the listed tests.
- Only one task may be *In progress* unless parallel work is justified here.
- Design rationale lives in `docs/architecture.md`. effGen facts live in `docs/effgen-api-audit.md`.

## 1. MVP definition

### Required MVP features

1. **Delegation (Feature A).** An orchestrator `effgen.Agent` delegates to a worker `effgen.Agent` through a `delegate` tool during its own loop.
   - The worker runs independently and returns a structured `WorkerResult`.
   - The orchestrator continues and finishes.
   - Depth is 1. One worker is required; N configured workers are supported.
   - Child lifecycle is tracked in memory.
2. **Heterogeneous models (Feature B).** Per-agent `ModelSpec` with `ollama` or `hf` providers. Mixed-provider configurations work in one process.
3. **Quantization (Feature C).** One verified quantized configuration per provider:
   - HF: `bnb-4bit-nf4`
   - Ollama: a `q4_K_M` GGUF tag

   `none` (bf16/fp16) and `bnb-8bit` are accepted by config but may remain "Implemented (unverified)" at MVP.
4. **Tools (Feature D).**
   - Workspace-confined `file_operations` and a sandboxed `shell`.
   - Per-agent explicit allowlists.
   - No web tools anywhere.
5. **Contexts (Feature E).** Separate histories, explicit task/context transfer, no history merging, and an in-memory status registry.
6. **Limits (Feature F).** `max_iterations` per agent, per-worker timeout, depth 1, `max_delegations`, and failure reporting.
7. **Entry point.** A minimal CLI driven by one YAML config.

### Optional improvements (allowed if cheap and justified)

- `edit_file` exact-replace tool (T4.3).
- Read-only worker profile using effGen's `ReadOnlyFileOperations`.

### Post-MVP

- Parallel workers.
- Runtime-defined workers.
- `MultiAgentOrchestrator` HIERARCHICAL baseline comparison.
- Benchmark runner on the Kaggle-derived dataset.
- Orchestrator wall-clock limit.
- Everything listed as excluded in `CLAUDE.md`.

## 2. Milestones and tasks

Milestone order follows the brief: the **delegation path first**, using scripted fake models, before any provider complexity.

### M0: Project scaffold

**T0.1 Create the uv project and pin effGen**

- **Status:** Completed (2026-10-16). Verification: see §4.
- **Objective:** A reproducible environment.
- **Modules:** `pyproject.toml`, `src/smol_harness/__init__.py`, `tests/`.
- **Dependencies:** none.
- **Changes:**
  - Python 3.12 via `uv`. The system Python is 3.14; 3.12 is available under uv.
  - Dependencies: `effgen==1.3.0`; if that version is not on PyPI, pin a git URL at commit `37de05a`.
  - Dev dependencies: `pytest`, `pyyaml` (already an effGen dependency).
  - Package name: `smol_harness`.
- **Acceptance:**
  - `uv sync` succeeds.
  - `uv run python -c "import effgen; print(effgen.__version__)"` prints `1.3.0`.
  - `uv run pytest` runs with 0 tests or a trivial import test.
- **Tests:** `tests/test_import.py`.
- **Risks:**
  - effGen's base install is heavy (torch, google SDKs, docker, ddgs).
  - The torch/CUDA wheel must match the driver. Record the actual install command in `docs/new-session.md` Quick Start.

### M1: First executable milestone, the delegation path (scripted models, CPU only). **Status: COMPLETE (T1.1–T1.3 verified)**

> Goal: an effGen orchestrator delegates a task to an independent effGen worker, receives a structured result, and continues execution.

**T1.1 Scripted-model test fixture**

- **Status:** Completed (2026-10-16). Verification: see §4.
- **Objective:** A deterministic effGen `BaseModel` fake that replays scripted ReAct turns.
- **Modules:** `tests/conftest.py`.
- **Dependencies:** T0.1.
- **Changes:** A `ScriptedModel(BaseModel)` modelled on effGen's `tests/unit/test_written_action_runs.py:_Scripted`. It implements `load`, `unload`, `generate`, `generate_stream`, `count_tokens`, `get_context_length`, and records the prompts it receives.
- **Acceptance:** `Agent(AgentConfig(model=ScriptedModel([...]), tools=[<trivial @tool>], tool_calling_mode="react", capability_probe=False, enable_sub_agents=False))` calls the tool once and returns the scripted final answer with `success=True`.
- **Tests:** `tests/test_scripted_model.py`.
- **Risk:** the 1.3.0 loop's nudges and closing request may consume extra scripted turns. Make the fake return its final turn repeatedly once the script is exhausted. Record the exact ReAct text format that works.

**T1.2 `WorkerResult` + `DelegateTool` + `WorkerRegistry`**

- **Status:** Completed (2026-10-16), with scope deviations listed in §4. Verification: see §4.
- **Objective:** The smallest delegation unit.
- **Modules:** `src/smol_harness/delegation.py`.
- **Dependencies:** T1.1.
- **Changes:**
  - A `WorkerResult` dataclass.
  - `WorkerRegistry` (list of `WorkerRecord`).
  - `DelegateTool(BaseTool)`:
    - parameters `worker` (enum), `task`, `context` (optional);
    - builds a fresh worker `Agent` via an injected factory `Callable[[str], Agent]`;
    - wraps the task in a template and calls `run()`;
    - catches all exceptions;
    - returns a truncated JSON string (architecture §5).
- **Acceptance (unit, no orchestrator yet):**
  - Calling `asyncio.run(tool.execute(worker="w", task="t"))` runs the scripted worker.
  - The returned `ToolResult.success` is True, and `output` is JSON with `status="completed"` and the worker's answer.
  - The registry shows `completed`.
  - An unknown worker name yields a clear error string.
- **Tests:** `tests/test_delegation.py`.

**T1.3 End-to-end orchestrator → worker → orchestrator (scripted)**

- **Status:** Completed (2026-10-16). All 7 acceptance points verified; see §4.
- **Objective:** Prove the M1 goal.
- **Modules:** `src/smol_harness/harness.py` (minimal `build_harness` taking in-code specs with model *instances*), `tests/test_e2e_delegation.py`.
- **Dependencies:** T1.2.
- **Changes:**
  - The orchestrator `Agent` holds `DelegateTool`.
  - Script: the orchestrator turn 1 calls `delegate`, then turn 2 gives a Final Answer quoting the worker output.
- **Acceptance:**
  1. The worker ran exactly once; its model received the templated task text.
  2. The orchestrator's model received, on its turn after delegation, a prompt containing the worker's JSON result.
  3. The orchestrator's final `AgentResponse.success` is True, and its output contains the worker answer.
  4. The worker's prompts contain none of the orchestrator's system prompt or earlier turns (isolation).
  5. The orchestrator's prompts contain none of the worker's internal turns.
  6. The worker agent holds no `delegate` tool (depth 1).
  7. `HarnessResult.workers[0].status == "completed"`.
- **Tests:** `uv run pytest tests/test_e2e_delegation.py -q`.

### M2: Execution controls and failure propagation (scripted). **Status: COMPLETE (T2.1–T2.3 verified)**

**T2.1 `DeadlineMiddleware` per-worker timeout**

- **Status:** Completed (2026-10-16). Verification: see §4. (Test names differ from the plan's `test_worker_timeout`; the acceptance is covered by `test_orchestrator_continues_after_worker_timeout_real_clock` and the `test_deadline_exceeded_*` tests.)
- **Modules:** `delegation.py`.
- **Dependencies:** T1.3.
- **Changes:** The middleware raises `WorkerTimeout` in `before_model_call` / `before_tool_call` after the deadline. `DelegateTool` maps it to `status="timeout"`.
- **Acceptance:** A scripted worker whose tool sleeps past the deadline yields `status="timeout"`, and the orchestrator still completes.
- **Tests:** `tests/test_limits.py::test_worker_timeout`.
- **Risk:** confirm that effGen does not swallow middleware exceptions on the path used (audit §1.8 says hooks propagate). If it does, fall back to checking the deadline in a `before_tool_call` short-circuit and `max_iterations`.

**T2.2 Failure propagation**

- **Status:** Completed (2026-10-16). Verification: see §4.
- **Dependencies:** T2.1.
- **Changes:**
  - Map worker `RunStoppedError` (iteration cap), arbitrary exceptions and `BackendUnreachableError` to `status="failed"` with an error string.
  - Verify the orchestrator is **not** ended as `termination="tool_failed"` by a failed worker.
- **Acceptance:** Three scripted scenarios each produce the right status, and the orchestrator completes or answers.
- **Tests:** `tests/test_limits.py`.

**T2.3 `max_iterations` per agent, `max_delegations`, report truncation**

- **Status:** Completed (2026-10-16). Verification: see §4. **M2 complete.**
- **Dependencies:** T2.2.
- **Acceptance:**
  - The worker stops at its own cap.
  - The (N+1)th delegation is refused with a clear message.
  - The report is truncated to `max_report_chars` with a marker.
- **Tests:** `tests/test_limits.py`.

### M3: Configuration and real model backends. **Status: IN PROGRESS (T3.1, T3.2, T3.2a, T3.2b, T3.2c done; T3.3 blocked; T3.4 and T3.5 not started). Not complete.**

Convention since T3.2b: tests that load a real checkpoint carry the `real_model` marker and run only with `--run-real-models` (or `SMOL_RUN_REAL_MODELS=1`); they use a pinned, cached revision and never download in the default suite. New real-model tests (T3.3 to T3.5) should follow it.

**T3.1 `config.py` (YAML → dataclasses + validation)**

- **Status:** Completed (2026-10-16). Verification: see §4.
- **Dependencies:** T1.3.
- **Acceptance:**
  - The example config loads.
  - Invalid configs are rejected with clear errors: unknown provider, `bnb-4bit-fp4`, a quant field on ollama other than `server`, a worker with `delegate`, an unknown tool, duplicate names.
- **Tests:** `tests/test_config.py`.

**T3.2 `models.py` `build_model` + cache**

- **Status:** Completed (2026-10-16). Verification: see §4.
- **Dependencies:** T3.1.
- **Changes:**
  - Map `ModelSpec` to `effgen.models.load_model` kwargs, always passing `quantization_bits` for hf.
  - Cache by spec.
- **Acceptance:** Unit tests monkeypatch `load_model` and assert the exact kwargs for each provider and quant.
- **Tests:** `tests/test_models.py`.

**T3.2a Explicit Hugging Face device selection** (added 2026-10-16, not in the original plan)

- **Status:** Completed. Details and verification: §4 (T3.2a). `device: cuda | cpu` on hf specs; CPU pinned with `device_map="cpu"`; 458 tests at the time.

**T3.2b Liquid AI CPU smoke test** (added 2026-10-16, not in the original plan)

- **Status:** Completed. Details and measurements: §4 (T3.2b). `LiquidAI/LFM2.5-350M` loaded on CPU (fp32) through `build_model` and answered through `model.generate` and `Agent.run`. Opt-in test `tests/integration/test_liquid_cpu.py`.

**T3.2c Liquid AI tool-calling compatibility** (added 2026-10-16, not in the original plan)

- **Status:** Completed. Details, raw outputs and parser table: §4 (T3.2c). ReAct tool calling worked 2/2; native 0/2 (effGen's native parser does not read the model's Pythonic call format). Opt-in test `tests/integration/test_liquid_tools.py`; model-free pins in `tests/test_lfm_parser_compat.py`.

**T3.3 Ollama smoke test**

- **Status:** Blocked (Ollama not installed on this host as of 2026-10-16)
- **Dependencies:** T3.2. Ollama installed and a small tag pulled (e.g. a `q4_K_M` instruct model).
- **Acceptance:** `pytest -m ollama` builds an agent on the tag and answers a one-line task. Record the tool-calling strategy effGen resolved and the `context_length` used.
- **Tests:** `tests/integration/test_ollama.py` (skipped when the server is not reachable; follow the `real_model` opt-in convention above).

**T3.4 HF Transformers smoke test (bnb NF4)**

- **Status:** Not started
- **Dependencies:** T3.2. `bitsandbytes` installed; CUDA torch working.
- **Acceptance:**
  - `pytest -m gpu` loads a ~0.5-1.5B instruct model with `bnb-4bit-nf4` and answers.
  - Record VRAM used.
  - Also verify `quantization: none` does **not** auto-quantize (inspect `model.quantization_bits`).
- **Tests:** `tests/integration/test_hf.py` (follow the `real_model` opt-in convention above; a CPU real-checkpoint load is already covered by T3.2b).

**T3.5 Mixed-provider delegation, real models**

- **Status:** Not started
- **Dependencies:** T3.3, T3.4, T2.3.
- **Acceptance:** An Ollama orchestrator with an HF worker (and the reverse) completes a trivial delegated task end-to-end. Record whether the small model follows the delegate tool contract.
- **Tests:** `tests/integration/test_mixed.py` (manual / marked).

### M4: Workspace tools. **Status: T4.1 and T4.2 done for trusted local development (T4.2 is NOT a complete security boundary; R1 unresolved). T4.3 contingent.**

**T4.1 `tools.py` registry + `file_operations`** (workspace-confined file operations)

- **Status:** **Completed (2026-10-16)**; all acceptance criteria verified by `tests/test_tools.py` (51 passed). Record: §4 (T4.1). The design notes below are kept as written before the task.
- **Objective:** give each agent exactly the tools its config allowlists, confined to the configured workspace. `build_tools(names, workspace)` returns fresh effGen tool instances from a fixed registry; `file_operations` is effGen's `FileOperations(allowed_directories=[workspace])`. Web or network tools can never be built.
- **Dependencies:** T3.1 (done): `config.KNOWN_TOOLS = ("file_operations", "shell")`, `AgentSpec.tools`, `HarnessConfig.workspace` (resolved, not required to exist). T4.2 (sandboxed shell) builds on the same factory.
- **Expected files:** new `src/smol_harness/tools.py`, new `tests/test_tools.py`. No effGen changes. Wiring `HarnessConfig` into `build_harness` is **not** part of this task (separate, unplanned step; see §5).
- **Relevant source (read these only):** `src/smol_harness/config.py` (`KNOWN_TOOLS`, `AgentSpec`, `HarnessConfig`); installed effGen `effgen/tools/builtin/file_ops.py` (`FileOperations`, `_validate_path`, `_is_path_allowed`, `ALLOWED_EXTENSIONS`) and `effgen/tools/builtin/_fs.py` (`is_credential_filename`, `default_workspace`); audit §1.8 and §1.9; architecture §9.
- **Proposed design (not yet decided; confirm while implementing):** `build_tools(names: Sequence[str], workspace: Path) -> list[BaseTool]`; an unknown name raises `ValueError`; `shell` raises a clear "not available until T4.2" error rather than being skipped (fail closed); the workspace must exist and be a directory (config validation deliberately does not check); a new tool instance per agent.
- **Source-level facts to verify by test, not assume** (audit §1.9; nothing here was run): `FileOperations` operations are read, write (whole file), list, search (filename glob only), metadata, convert; paths are resolved and must lie inside the allowed directories (symlink escapes should be refused); credential-looking filenames are refused; an extension allowlist applies to read and write (extensionless files such as `Makefile` should fail); a relative path resolves against the first allowed directory; failures come back as failed tool results the model sees as observations.
- **Acceptance:**
  - Reads and writes inside the workspace work.
  - Paths outside it (`../`, absolute, symlink escape) are refused with a clear error.
  - The per-agent allowlist is honoured (an agent built with `[]` holds no tools).
  - Unknown tools, `delegate`, and anything web/search/browser-like cannot be built; a test asserts that no agent the harness builds holds a tool whose name matches `web|search_engine|url_fetch|http`.
  - Checked through `BaseTool.execute()` using temporary workspaces (CPU only, no network, no real model).
- **Tests:** `uv run pytest tests/test_tools.py -q`, then the full suite (`uv run pytest -q`, baseline 480 passed, 3 skipped).
- **Cautions:** do not use effGen's `BashTool` (not a sandbox); `ddgs`/`duckduckgo-search` are installed through effGen and must never be registered; effGen reads `EFFGEN_WORKSPACE` as a default workspace, so always pass `allowed_directories` explicitly.

**T4.2 Sandboxed `shell` tool**

- **Status:** **Completed for trusted local development only (2026-10-17).** All six acceptance criteria below were met and verified, so none was weakened. What is *not* achieved is a complete isolation boundary, which the criteria did not ask for but the word "sandbox" suggests: host file **reads** are not confined, and host Unix sockets outside `/run` stay reachable (**R1, unresolved**). **Accepted scope (user decision, 2026-10-17): the shell is for trusted local development and controlled test repositories only. Untrusted repositories and adversarial agent workloads must NOT run in this sandbox; they require a dedicated, independently isolated container or VM whose controls are separately verified. No seccomp filter, new backend or other isolation infrastructure will be added in the MVP.** Record: §4 (T4.2). The text below is the plan as written before the task.
- **Dependencies:** T4.1 (done).
- **Changes (as planned; as built it is a `BaseTool` over `SubprocessSandbox`, see §4):** A `CodeExecutor` subclass exposing `command` (and capped `timeout`). It pins bash, network off and the configured memory limit, and requires the subprocess backend. Fail closed if the sandbox does not initialize.
- **Acceptance:**
  - `echo` works.
  - Writing outside the workspace fails.
  - Network access fails (e.g. `curl` / python socket).
  - A sleep exceeding the timeout is killed.
  - The result reports `filesystem_confined=True`.
  - `pytest` inside the workspace runs under the configured memory limit (tune the floor).
- **Tests:** `tests/test_tools.py` (Linux-only marker).

**T4.3 (contingent) `edit_file` exact-replace tool**

- **Status:** Not started
- **Do only if** T5.2 shows whole-file writes failing for small workers.
- **Acceptance:** Unique-match replacement works; zero or multiple matches return clear errors; the path is confined.

### M5: CLI and end-to-end coding smoke test

**T5.0 Assemble a harness from `HarnessConfig`** (inserted 2026-10-17; before T5.1)

- **Status:** Not started. **This is the next executable task.**
- **Objective:** one call turns a validated `HarnessConfig` into a runnable `Harness` that owns its models: `build_harness_from_config(config, *, cache=None, tool_calling_mode="react") -> Harness`, plus `Harness.close()` / context manager. Responsibilities stay separate: `config.py` only parses and validates (unchanged), `models.py` only loads and owns models (`ModelCache`), `tools.py` only builds tools, `delegation.py` untouched, and `harness.py` does the wiring (the module the architecture already assigns it). No new module, no new abstraction.
- **Dependencies:** T3.1 (`HarnessConfig`), T3.2 (`ModelCache`, `build_model`), T4.1 and T4.2 (`build_tools`), M1/M2 (`build_harness`, `Harness`). Not T3.3 to T3.5 (real models).
- **Expected files:** modify `src/smol_harness/harness.py` (add the function, `Harness.close`, `__enter__/__exit__`; keep `build_harness` and its signature working, since 100+ tests use it); create `tests/test_harness_config.py`. No change to `config.py`, `models.py`, `tools.py`, `delegation.py` unless a test proves a defect (then record it). No effGen change.
- **Current interfaces (inspected 2026-10-17):** `build_harness(orchestrator: AgentConfig, workers: Mapping[str, AgentConfig], descriptions, worker_timeout_s, max_delegations, max_report_chars) -> Harness` takes already-built `AgentConfig`s holding model *instances*; `Harness` has no `close`; `ModelCache.get(spec)` builds each distinct frozen `ModelSpec` once and `close()` unloads all (idempotent, `ExceptionGroup` on unload failures, usable as a context manager); `build_tools(names, workspace)` builds fresh tool instances and validates the workspace; `AgentSpec(name, model, tools, max_iterations, description, system_prompt)`; `Limits(max_delegations, max_report_chars, worker_timeout_s, max_depth)`.
- **Design:**
  1. **Validate before loading anything.** For every agent (orchestrator first, then workers in config order) call `build_tools(spec.tools, config.workspace)`. Any `ToolBuildError` (bad workspace, dangerous root, `shell` unavailable off Linux or without `unshare`) aborts assembly before any model is built.
  2. **Then load models** through one `ModelCache` (the injected `cache` if given, else a new one that the `Harness` owns). Agents with equal `ModelSpec` share one instance (the existing cache rule); distinct specs never do.
  3. **Map each `AgentSpec` to an `AgentConfig`:** `name`, `model` (instance), `tools` (the instances from step 1; the orchestrator's `delegate` is still added by `Harness.run`, never here), `max_iterations`, `system_prompt` when set, `tool_calling_mode` (a parameter, default `"react"`, because ReAct is the only mode verified with the small local model; it is not a YAML field), the same fixed flags the e2e tests use (`capability_probe=False`, `enable_memory=False`, `raise_on_error=False`; confirm each against audit §1.1 and §1.4 before keeping it), and for Ollama specs `max_context_length=spec.context_length`. HF specs get no `max_context_length` (no config field; known gap, audit Part 2).
  4. **Register workers** with `description` from `AgentSpec`, and pass `Limits` (`worker_timeout_s`, `max_delegations`, `max_report_chars`) to `Harness`. `max_depth` is fixed at 1 already.
  5. **Ownership and cleanup:** `Harness.close()` unloads the models of a cache it owns, is idempotent, and does nothing to an injected cache (the caller owns it). `Harness` is a context manager. `run()` after `close()` raises `RuntimeError`. If assembly fails after some models loaded, the models loaded so far are unloaded before the exception propagates (no leak), and the original error is the one raised.
  6. **Tool lifetime:** tool instances are shared by all delegations of one agent within a harness (the worker registry builds a fresh `Agent` per delegation around the same `AgentConfig`); they hold no per-run state (`ShellTool` caches only its sandbox proof). Different agents never share a tool instance.
- **Failure behaviour (all raise before any `run`, no partial harness is returned):** `ToolBuildError` for tools or workspace (raised before any model load; `build_model` is never called); `ModelLoadError` (from `ModelCache`) for a model, with the cache cleaned up; `ConfigError` is not this function's concern (already raised by parsing). No retries, no fallbacks, no silently dropped tools or agents.
- **Acceptance:**
  - A harness built from a two-worker config runs a delegated task end to end with scripted models and real tools, and returns the same `HarnessResult` shape as today.
  - Each agent holds exactly its allowlisted tools (orchestrator `{delegate} ∪ allowlist`, workers their own, `[]` means none; no worker holds `delegate`; no web-like tool anywhere).
  - Equal specs share one model instance; different specs do not; `build_model` is called once per distinct spec.
  - `close()` unloads every owned model exactly once, is idempotent, leaves an injected cache open, and `run()` afterwards raises; the context manager closes on normal exit and on an exception inside the block.
  - An invalid workspace, a dangerous root, or an unbuildable `shell` fails before any model is built; a model failure on the second distinct spec unloads the first.
  - `Limits` reach the harness (a config with `max_delegations: 1` refuses the second delegation; a tiny `worker_timeout_s` times out a scripted slow worker).
  - The effGen loop patch is installed explicitly by the application path, never inside `build_harness_from_config` or `Harness.close()`; `close()` leaves it installed (test), and an uninstalled process fails with `EffgenCompatError` at the first `run()`.
  - Default suite stays CPU-only, no network, no real model; baseline 648 passed, 3 skipped, plus the new tests; `uv lock --check` unchanged.
- **Tests (deterministic, `tests/test_harness_config.py`):** patch `smol_harness.models.build_model` to return `ScriptedModel`s keyed by `spec.model` (recording calls and `unload`); configs through `loads_config` with a temporary workspace; cases for each acceptance bullet above, plus tool-instance non-sharing across agents and system-prompt/iteration/context-length mapping. A `shell` agent test is `linux_only`.
- **Out of scope:** the CLI, YAML changes (a `tool_calling_mode`, HF `max_context_length`, shell `memory_limit` / `max_timeout_s` fields are possible later tasks), real models, Ollama/HF loading, any sandbox change, parallel agents.
- **Open questions for the user (defaults chosen above; say if you disagree):** (1) `tool_calling_mode` default `"react"` as a function parameter rather than a YAML field; (2) HF context window stays unset until a config field is approved.

**T5.1 CLI**

- **Status:** Not started
- **Dependencies:** T3.1, T4.2, T5.0.
- **Acceptance:** `uv run python -m smol_harness --config configs/example.yaml --task "..." --json` prints the final output plus worker records. The exit code is non-zero on orchestrator failure.

**T5.2 Toy-repo smoke test**

- **Status:** Not started
- **Dependencies:** T5.1, T3.5. Trusted repositories only (T4.2 scope); never an untrusted repository.
- **Acceptance:** In a temporary git repo with one failing unit test:
  1. the orchestrator delegates "fix the bug";
  2. the worker edits the file;
  3. the shell runs the tests;
  4. they pass;
  5. the orchestrator reports.

  Record the success rate over 3 runs (informational, not a gate).

### M6: Post-MVP preparation (research only, no infrastructure)

**T6.1 Verify the Kaggle dataset structure**

- **Status:** Blocked: needs Kaggle credentials, since the data page and API returned unauthenticated or empty on 2026-10-16.
- **Acceptance:** Document the actual files and fields, and whether repository snapshots, executable tests or ground-truth patches exist, in `docs/new-session.md` §6 and a new `docs/dataset-notes.md`.

## 3. Current active goal

- **Milestone:** M3 is **in progress, not complete**: T3.1, T3.2, T3.2a, T3.2b, T3.2c are done; T3.3 is blocked (Ollama not installed); T3.4 and T3.5 are not started (need `bitsandbytes`/GPU runs and an Ollama server). M4 (workspace tools): T4.1 done; T4.2 next.
- **Current task:** none active. **T4.2 is Completed for trusted local development only** (R1 accepted, unresolved; §4). **Next executable task: T5.0, assemble a harness from `HarnessConfig`** (§2, M5), then T5.1. Nothing is started.
- **Why:** agents have workspace-confined file operations (T4.1) and a shell for trusted development (T4.2); `HarnessConfig`, `ModelCache`, `build_tools` and `build_harness` exist but nothing connects them, so no configured harness can run yet (T5.0), and the CLI (T5.1) needs that.
- **How success is verified:** `uv run pytest tests/test_tools.py -q` (121 passed), then `uv run pytest -q` (601 passed, 3 skipped).
- **First executable development milestone:** **M1 (T1.1 → T1.3): reached 2026-10-16.** `tests/test_e2e_delegation.py` passes.

## 4. Completed work

### T0.1 Create the uv project and pin effGen (2026-10-16)

- **Files:** `pyproject.toml`, `uv.lock`, `.python-version` (`3.12`), `.gitignore`, `src/smol_harness/__init__.py`, `tests/test_import.py`.
- **Implemented:** `requires-python = ">=3.12,<3.13"`; sole runtime dependency `effgen==1.3.0` (it is on PyPI, so no git pin was needed); dev group `pytest>=8`, `pyyaml>=6`; `uv_build` backend; pytest `testpaths` plus registered `ollama` and `gpu` markers.
- **Verification (all run):**
  - `uv sync` → OK (157 packages resolved).
  - `uv run python --version` → `Python 3.12.13`.
  - `uv run python -c "import effgen; print(effgen.__version__)"` → `1.3.0`.
  - `uv run pytest -q` → `2 passed`.
  - Reproducibility: `rm -rf .venv && uv sync --locked`, then the import and pytest checks again → same results. `uv lock --check` OK.
- **Deviations:** none from the plan. Added `.python-version` and `.gitignore` (minimal packaging). The directory is still not a git repo.
- **Findings:** the default install resolved `torch 2.14.1+cu130` (`torch.cuda.is_available()` is True, driver 615.71.09) and `transformers 5.20.0`. `bitsandbytes` is **not** installed (needed from T3.4). `ddgs` and `duckduckgo-search` are installed (never register them as tools).

### T1.1 Scripted-model test fixture (2026-10-16)

- **Files:** `tests/conftest.py` (`ScriptedModel`, `ModelCall`, `ScriptExhaustedError`, `scripted_model` fixture), `tests/test_scripted_model.py` (15 tests).
- **Implemented:** `ScriptedModel(BaseModel)` replays scripted turns, records every call (`model.calls`: prompt, config, kwargs, response), supports `generate` and `generate_stream`, and has `on_exhausted="repeat_last"` (default) or `"raise"`.
- **Verification (all run):** `uv run pytest tests/test_scripted_model.py -q` → 15 passed. `uv run pytest -q` → 17 passed. The tests drive the real `effgen.Agent` loop in ReAct mode with no mocking of effGen.
- **Verified ReAct format** (the acceptance case makes exactly 2 model calls, 1 tool call, `success=True`):
  - tool turn: `Thought: ...\nAction: <tool>\nAction Input: {"a": 2, "b": 3}`
  - answer turn: `Thought: ...\nFinal Answer: <text>`
  - the next prompt carries `Observation: <tool result>`.
- **Deviations:** the fake declares no native tool calling, so no `generate_with_tools` / `supports_*` overrides were needed (upstream's `_Scripted` declares native). Plan §T1.1's "records the prompts" is implemented as `calls`.
- **Debt/limits:** ReAct text only. A native tool-calling script is not covered. Add it only if a later task needs it.

### T1.2 `WorkerResult` + `DelegateTool` + `WorkerRegistry` (2026-10-16)

- **Files:** `src/smol_harness/delegation.py`, `tests/test_delegation.py` (22 tests).
- **Implemented:**
  - `WorkerResult` (worker, status `completed|failed`, output, error, error_type, termination, iterations, tool_calls, tokens_used; `to_json()`).
  - `WorkerRegistry` (`register`, `names`, `description`, `get_config`, `create`; `UnknownWorkerError`). A definition is an `AgentConfig` with an already-loaded `BaseModel`; `create` builds a fresh `Agent` from a per-agent shallow copy (`enable_sub_agents=False`).
  - `DelegateTool(BaseTool)`: params `worker`, `task`, `context?`; returns a JSON string; every worker failure is a structured result.
- **Verification (all run):** `uv run pytest tests/test_delegation.py -q` → 22 passed. `uv run pytest -q` → 39 passed. Mutation check: making the registry cache and reuse one agent fails 3 isolation tests (restored afterwards).
- **Demonstrated, not assumed:**
  - The second delegation's prompt does not contain the first task.
  - A control test shows a *reused* `Agent` does leak the earlier task, so the fresh-agent design is what provides isolation.
  - Agents created for one or several workers share the same model object (`is`).
  - The report holds only the `WorkerResult` fields: no worker thoughts, observations or thread.
- **Deviations from the plan text (per the T1.2 request):**
  - No `WorkerRecord` / lifecycle status tracking, and so no "registry shows `completed`". That belongs to a later task. **Open:** schedule it (M2) or drop it; `docs/architecture.md` §6 still describes it.
  - No `delegation_id`, `duration_s`, `timeout` status or report truncation (T2.1 / T2.3).
  - The registry takes `AgentConfig` definitions, not an injected `Callable[[str], Agent]` factory, so model sharing is checked at registration.
  - An unknown worker returns a structured failure (`error_type="unknown_worker"`), not a bare error string. Malformed arguments (missing, empty or non-string `worker`/`task`) are rejected by effGen's own validation as a failed `ToolResult` (see audit Part 3); a blank task is a structured failure.
- **Debt / limitations:** the orchestrator has not yet been driven through this tool (T1.3). Whether a *failed* `ToolResult` from effGen's argument validation can end a parent run as `tool_failed` is unverified (U3 covers worker failures only).

### T1.3 End-to-end orchestrator → worker → orchestrator (2026-10-16)

- **Files:** `src/smol_harness/harness.py` (new: `Harness`, `HarnessResult`, `build_harness`), `src/smol_harness/delegation.py` (added `WorkerRecord`; `DelegateTool(registry, records=None)` appends one record per delegation), `tests/test_e2e_delegation.py` (22 tests).
- **Implemented:**
  - `build_harness(orchestrator_config, {name: worker_config}, descriptions)`. Inputs are `AgentConfig`s holding model *instances*.
  - `Harness.run(task)` builds a **fresh orchestrator `Agent` and a fresh record list per call**, with `DelegateTool` appended to the orchestrator's tools. Returns `HarnessResult(output, success, termination, response, workers)`.
  - `WorkerRecord(worker, task, context, result)` with `.status`. Terminal states only (`completed` / `failed`), no state machine.
- **Verification (all run):** `uv run pytest tests/test_e2e_delegation.py -q` → 22 passed. `uv run pytest -q` → 61 passed. Mutations (a record list shared across runs; a reused orchestrator agent) make the independence test fail; both were restored.
- **Acceptance (all verified by tests):** (1) the worker ran once and its prompt carries the templated task and context; (2) the orchestrator's next prompt contains the worker's JSON report as the `Observation`; (3) final `success=True`, output contains the worker answer, exactly 2 orchestrator model calls; (4) worker prompts contain none of the orchestrator system prompt, user task or thoughts (and the marker is shown to be present in the parent's own prompt, so the check is not vacuous); (5) orchestrator prompts contain no worker thought or tool observation; (6) the worker holds no `delegate` tool and its prompt never mentions it; (7) `result.workers[0].status == "completed"`.
- **Also verified:** two delegations create two distinct `Agent`s sharing one model object, with no history carried; a failing worker is recorded `failed` and the parent still finishes `done` (U3 confirmed for a worker whose model raises); an unknown worker is recorded `failed`; independent runs do not share records or orchestrator memory.
- **U11 (invalid `delegate` arguments), observed:** see audit Part 3. The parent always gets an error observation and may continue; it was never ended as `tool_failed`; repeated invalid calls end it as `stuck`.
- **Deviations / notes:** `build_harness` takes `AgentConfig`s rather than bare model instances (so tools, prompts and limits are carried). No `delegation_id`/`duration_s`/`timeout`/truncation yet (T2.x). Invalid calls rejected by effGen never reach `_execute`, so they produce **no** `WorkerRecord`.
- **Debt / limitations:** only ReAct text mode is exercised; `Harness.run` propagates orchestrator exceptions when its config has `raise_on_error=True`; worker failure reporting is only as detailed as `WorkerResult`.

### T2.1 Worker deadline enforcement (2026-10-16)

- **Preliminary correction:** added `test_orchestrator_memory_does_not_carry_across_runs` (no delegation involved, so only orchestrator memory can leak). Verified by mutation: with the orchestrator `Agent` deliberately reused, the assertion `"USER-RUN-ONE" not in second_prompt` fails (it is the first failing assertion in that test); harness restored, no refactor.
- **Files:** `src/smol_harness/delegation.py` (`WorkerTimeout`, `DeadlineMiddleware`, `status="timeout"`, `DelegateTool(..., worker_timeout_s, clock)`), `src/smol_harness/harness.py` (`build_harness(..., worker_timeout_s)`, `Harness(..., worker_timeout_s)`), `tests/test_limits.py` (15 tests), `tests/test_e2e_delegation.py` (+1 regression test).
- **Implemented:** a fresh `DeadlineMiddleware` per delegation, passed as per-call middleware (`agent.run(prompt, middleware=[deadline])`). It raises `WorkerTimeout` from `before_model_call` / `before_tool_call` once a monotonic deadline has passed and sets `expired`. `DelegateTool` reports `status="timeout"` (`error_type="WorkerTimeout"`, no output) whenever `deadline.expired` is set, whether effGen raised the exception or folded it into a failed response. `None` timeout = no deadline (M1 behaviour).
- **COOPERATIVE, not a hard wall-clock timeout.** The deadline is checked only before a model call or a tool call. An in-flight generation or tool is never interrupted (shown by a real-clock test: a 0.1 s deadline did not cut a 0.3 s tool short; the delegation took at least 0.3 s). Worst-case overrun is one model call or tool call. No thread/process cancellation.
- **Verification (all run):** `uv run pytest tests/test_limits.py -q` → 15 passed. `uv run pytest -q` → 77 passed. Mutations each fail tests and were restored: no post-run `expired` check (breaks the `raise_on_error=False` path), one deadline shared across delegations (breaks the leak test), middleware never passed to `run()` (8 failures).
- **Acceptance:** a worker whose tool outlives the deadline yields `status="timeout"` (fake clock and real clock), and the orchestrator still completes (`success=True`, `termination="done"`, record `timeout`). Timeout state does not leak between delegations or runs. The timeout result is valid JSON with the common `WorkerResult` schema.
- **U4 answer:** see audit Part 3. Middleware exceptions propagate on the model and tool paths.
- **Deviations / notes:** the timeout is one value for all workers (`Limits.worker_timeout_s` shape), not per worker. `clock` is an injectable parameter on `DelegateTool` for deterministic tests; `Harness` does not expose it. No `duration_s` or `delegation_id` yet. `Harness`/`build_harness` expose `worker_timeout_s`, which is a small addition beyond the plan text so the limit is reachable end to end.
- **Debt / limitations:** the orchestrator has no deadline (post-MVP). `max_delegations` and report truncation remain (T2.3).

### T2.2 Worker failure propagation (2026-10-16)

- **Files:** `src/smol_harness/delegation.py` (`RunStoppedError` unwrapping, `_from_response`, `WorkerResult.delegation_id`), `tests/test_limits.py` (+32 tests, now 47). `harness.py` unchanged.
- **Implemented:**
  - `DelegateTool` handles a raised `RunStoppedError` by reporting its `.response` exactly as the non-raising path does, so `raise_on_error=True` and `False` give the same `WorkerResult` for iteration-cap and worker-tool failures (status `failed`, `error_type="worker_run_failed"`, termination `stuck` / `tool_failed`, iterations and tool calls preserved).
  - `WorkerResult.delegation_id` (`d1`, `d2`, ... per `DelegateTool`, i.e. per run) on every result.
  - A deadline-expired worker still reports `timeout` (checked first); an unrelated failure with a deadline set stays `failed`.
- **Verification (all run):** `uv run pytest tests/test_limits.py -q` → 47 passed. `uv run pytest -q` → 109 passed. Mutations (no `delegation_id`; `RunStoppedError` not unwrapped) each fail tests and were restored.
- **Acceptance:** four worker failure kinds (iteration cap, backend unreachable, unexpected exception, worker tool failing on its own side) × both `raise_on_error` settings give valid JSON with `status="failed"` and meaningful `error` / `error_type` / `termination`; in all 8 cases the orchestrator receives the report, finishes `success=True`, `termination="done"`, never `tool_failed`.
- **Actual effGen exception paths** (exact types, audit Part 3): iteration cap and worker-tool failure raise `RunStoppedError` only under `raise_on_error=True`; `BackendUnreachableError` is raised under **both** settings and is not a `RuntimeError`; an unexpected model exception becomes a plain `RuntimeError("ValueError: boom")` under `True` and a failed response under `False`.
- **Deviation / finding:** `delegation_id` (planned for T2.3-ish, architecture §5) was pulled forward because T2.2's requirement that a recoverable worker failure must not stop the parent failed without it: two identical failure reports from *different* delegations triggered effGen's `repeated_tool_result` guard and ended the parent as `stuck` (`success=False`). The same would have hit two identical *successful* reports.
- **Debt / limitations:**
  - `error_type` differs by path for unexpected exceptions (`worker_run_failed` vs `RuntimeError`) because effGen drops the response when it wraps them.
  - effGen retries a failing generation 3× with 0.5 s / 1.0 s sleeps and no config knob, so a dead backend costs ~1.5 s per delegation (worker deadline cannot interrupt it). Tests no-op `effgen.core.ledger.backoff_sleep` and assert the retries.
  - A parent that retries the *identical* failing delegation is stopped by effGen (`loop_detected`) without the call being re-run.
  - `iterations` is 0 when no response exists (unreachable / wrapped exceptions).

### T2.3 Delegation limits and report truncation (2026-10-16)

- **Files:** `src/smol_harness/delegation.py` (`max_delegations`, `max_report_chars` on `DelegateTool`; `bounded`, `validate_limits`, `MIN_REPORT_CHARS`, `TRUNCATION_MARK`, `ID_FIELD_CAP`; `WorkerResult.truncated`), `src/smol_harness/harness.py` (both limits on `Harness` / `build_harness`, validated at build time), `tests/test_limits.py` (+30 tests, now 77).
- **Configuration behaviour:**
  - `max_delegations`: `None` (default, unlimited) or an int ≥ 1 (bool, float, str rejected). Counts every attempt that reaches the tool, per run (a new `DelegateTool` per `Harness.run`). Attempts above the limit return a structured failure and run nothing. The tool description tells the parent its budget.
  - `max_report_chars`: `None` (default, unbounded) or an int ≥ `MIN_REPORT_CHARS` (= 346: the fixed part of a report at worst-case id/counter widths + 100). Bool, float, str and anything smaller are rejected at construction / `build_harness`.
  - Both are `None` by default, so M1/M2 behaviour is unchanged unless set.
- **Truncation semantics:** measured on the final serialized JSON (`ensure_ascii=False`, so characters). Order of cuts: cap `worker`/`error_type` to 64 chars → cut `output` (longest prefix that fits + `...[truncated]`) → cut `error` the same way → last resort shrink `worker`, then `error_type`. `truncated: true` is set. Never changed: `delegation_id`, `status`, `termination`, `iterations`, `tool_calls`, `tokens_used`. A report that already fits is returned untouched (`truncated: false`). `WorkerRecord.result` keeps the full text (with `truncated` set accurately); only the parent-visible JSON is cut.
- **Verification (all run):** `uv run pytest tests/test_limits.py -q` → 77 passed. `uv run pytest -q` → 139 passed. Five mutations each fail tests and were restored: `>=` instead of `>` (off-by-one), refusal after running a worker, `bounded()` disabled, `DelegateTool`/counters reused across harness runs, size measured on `len(output)` instead of the serialized JSON.
- **Acceptance:** N delegations allowed and N+1 refused, with no worker built or run (also for unknown worker / blank task); unique ids for every attempt including refusals; identical worker outputs remain distinguishable and do not trip effGen's repeated-result guard (also when truncated); large output truncated to the limit, valid JSON, within size (400-case deterministic fuzz over quotes, newlines, control chars, BMP and astral Unicode, oversized worker names, and limits from the minimum up); failure metadata preserved; invalid configs rejected; counters reset per run; the orchestrator continues after a refusal; worker and parent iteration caps are effGen's own and a worker cap stays a structured failure.
- **Deviations / notes:** limits default to unset rather than to a "sensible" number (defaults belong to the config layer, T3.1). A refused attempt is recorded as a failed `WorkerRecord` like other failed attempts, so `HarnessResult.workers` can be longer than `max_delegations`. Calls rejected by effGen's argument validation never reach the tool and are not counted.
- **effGen findings** (audit Part 3): `max_iterations=N` allows N model calls but only N−1 tool dispatches when the model never answers (the last call's action is not dispatched); parent `stop_reason` is `max_iterations_partial` if any tool result was obtained and `max_iterations_exhausted` otherwise; effGen shortens long exception text itself.
- **Debt / limitations:** the 64-char cap applies whenever a report is over its limit, even if only `output` was too long. `duration_s` is still not reported. Truncation is character-based, not token-based.

### T3.1 YAML configuration and validation (2026-10-16)

- **Files:** `src/smol_harness/config.py`, `tests/test_config.py` (179 tests), `configs/example.yaml`; `pyproject.toml` + `uv.lock` (PyYAML moved from the dev group to a runtime dependency, because `config.py` imports it; it was already installed and locked via effGen).
- **Public API:** `load_config(path)`, `loads_config(text, base_dir=None)`, `parse_config(mapping, base_dir=None)` → `HarnessConfig`; `ConfigError(ValueError)`; frozen dataclasses `ModelSpec`, `AgentSpec`, `Limits`, `HarnessConfig`; constants `PROVIDERS`, `HF_QUANTIZATIONS`, `OLLAMA_QUANTIZATIONS`, `KNOWN_TOOLS`, `MAX_SPAWN_DEPTH`.
- **Schema:** see `docs/architecture.md` §8 and `configs/example.yaml`.
- **Validation decisions:** quantization is exact-match only, never normalised or substituted (`None`, `BNB-8BIT`, `awq`, `gptq`, `int8`, `4bit`, `q4_K_M` all rejected; `bnb-4bit-fp4` rejected with the NF4 reason; bitsandbytes values on ollama rejected with "fixed by the model tag; write `server`"; non-string values such as YAML `off`/`null` rejected with a quoting hint). HF `quantization` and Ollama `context_length` are required. Booleans are rejected wherever integers/numbers are expected; limits must be finite (no `null`/unlimited, `inf`, `nan`); `max_depth` must be 1; no unknown keys anywhere (so no invented controls such as `request_timeout`). Duplicate YAML keys are an error (PyYAML would silently keep the last). Tool names: only `file_operations`/`shell`; web/search/browser/fetch-like names get a specific "not available in the MVP" error; `delegate` is rejected for workers and accepted-but-implicit for the orchestrator. Names and model ids: non-empty, no surrounding or embedded whitespace. At least one worker is required.
- **Verification (all run):** `uv run pytest tests/test_config.py -q` → 179 passed. `uv run pytest -q` → 318 passed. `uv lock --check` OK. Ten mutations (bool accepted as int, FP4 allowed, hf quantization defaulted, `delegate` on workers, unknown tools accepted, duplicate names, duplicate YAML keys, bitsandbytes on ollama, depth ≠ 1, optional context_length) each fail tests and were restored; one (`bnb` branch removed) initially survived because the test accepted either message, so the test was tightened and the mutant now fails.
- **Acceptance:** the example loads; unknown provider, `bnb-4bit-fp4`, quantization fields on ollama other than `server`, a worker with `delegate`, unknown tools and duplicate names are all rejected with path-prefixed messages. Also covered: Ollama-only, HF-only, both mixed directions, multiple workers, defaults, malformed YAML, missing required fields, wrong types and ranges. Importing/using the config layer loads neither `torch` nor `transformers`.
- **Deviations from the plan text:** none in scope; two design choices beyond it: `workers` is a YAML list (not a mapping) and two fields are required (HF `quantization`, Ollama `context_length`), see above. `Limits` fields are finite and non-nullable even though the runtime accepts `None`.
- **Debt / limitations:** the workspace is not checked for existence (T4.1 does); `base_url` is only checked for an http(s) prefix; model ids are not checked against Ollama/HF (no network); `hf` models with `dtype` omitted use effGen's automatic bf16/fp16 choice; the example's model pairing is unverified for 8 GB VRAM (U7). Nothing wires `HarnessConfig` into `build_harness` yet.

### T3.2 Model factory and cache (2026-10-16)

- **Files:** `src/smol_harness/models.py`, `tests/test_models.py` (87 tests). No other source changes.
- **effGen APIs used:** `effgen.models.load_model` (the only loader call), `effgen.models.base.BaseModel` / `ModelType`, `effgen.models.openai_compatible_adapter.OpenAICompatibleAdapter` (result check), `BaseModel.unload()`.
- **Public API:** `build_model(spec)`, `loader_args(spec)`, `ModelCache` (`get`, `close`, `__len__`, `__contains__`, context manager), `ModelLoadError`, `HF_QUANTIZATION_BITS`, `OLLAMA_ID_ALIAS_PREFIXES`.
- **Quantization argument mapping:** hf `none` → `quantization_bits=None`; `bnb-8bit` → `8`; `bnb-4bit-nf4` → `4` (always passed, even `None`; effGen's `quantization=` shorthand unused). hf `dtype` `bf16` → `torch.bfloat16`, `fp16` → `torch.float16` (key omitted when unset). Always `engine="transformers"` and `require_gpu=True`. ollama: `provider="openai_compatible"`, `base_url`, `context_length`; no quantization argument (fixed by the tag). FP4 is unreachable.
- **Cache behaviour:** keyed by the whole frozen `ModelSpec`; equal specs share one instance (one load), any differing field (provider, model, quantization, dtype, base_url, context_length) is a separate entry; per-instance, no global state; failed loads leave no entry; `close()` unloads each model once and is idempotent; `get` after `close` raises.
- **Verification (all run):** `uv run pytest tests/test_models.py -q` → 87 passed. `uv run pytest -q` → 405 passed. `uv lock --check` OK. Ten mutations (quantization_bits omitted for `none`, `require_gpu` dropped, cache keyed by model id only, result check removed, silent retry/downgrade after failure, alias guard removed, error cause dropped, error message concealed, `close` not unloading, 8-bit/4-bit swapped, hf kind check skipped) each fail tests and were restored.
- **Acceptance:** exact loader kwargs for each provider and quantization; endpoint and context length propagate; same-spec reuse, different-spec separation, cross-provider independence; loading failures carry the spec and original error (`__cause__`) and are not cached or retried; wrong-kind and substituted-quantization results are refused; invalid specs fail before anything loads; four workers over two distinct specs share model instances through real `WorkerRegistry`/`Agent`s (2 loads for 4 agents); importing the module loads no model and no torch/transformers/bitsandbytes/accelerate (checked in a subprocess).
- **Beyond the listed arguments (flag for review):** `require_gpu=True` for hf; the Ollama alias-prefix guard; the post-load type and quantization check; `ModelLoadError` wrapping (original exception preserved as `__cause__`).
- **Limitations:** no real model has been loaded: the hf path is verified against effGen's real routing with only `TransformersEngine` stubbed, and the real engine's `model_type`/`quantization_bits` attributes are source-level (T3.4). The Ollama server is first contacted on first use (T3.3). `require_gpu=True` rules out CPU-only hosts for hf. The alias set mirrors effGen's private alias table and can drift (a control test re-checks effGen's behaviour). Nothing wires `ModelCache` into `build_harness` yet.

### T3.2a Explicit Hugging Face device selection (2026-10-16)

- **Why:** T3.2 hard-coded `require_gpu=True`, which rules out CPU-only hosts (the planned fanless edge-device smoke test). Not in the original plan; requested as a focused adjustment before real-model inference.
- **Files:** `src/smol_harness/config.py` (`ModelSpec.device`, `HF_DEVICES`, `DEFAULT_HF_DEVICE`, dtype `fp32`, `hf_placement_problem`, `device` key), `src/smol_harness/models.py` (CPU/CUDA kwargs, `_cuda_available` preflight, post-load device check), `configs/example.yaml` (comment only), `tests/test_config.py` (+22 collected tests, now 201), `tests/test_models.py` (+23, now 110; fakes now report a device, `loader` fixture is hermetic about CUDA), `tests/test_hf_device.py` (new, 8 tests).
- **Behaviour:** `device: cuda` (default; omitted == explicit, same cache key): unchanged arguments (`require_gpu=True`, no `device_map`), plus our own CUDA-availability check and a post-load check that the model is on `cuda`. `device: cpu`: `quantization_bits=None`, `require_gpu=False`, `device_map="cpu"`, `use_flash_attention=False`, explicit `torch_dtype`, post-load check that the model is on `cpu`. Rules: cpu allows only `quantization: none` and requires `dtype` in {`fp32`, `bf16`}; cuda rejects `fp32`; unknown devices (`gpu`, `cuda:0`, `mps`, `CPU`) rejected; `device` rejected for ollama (config and dataclass).
- **Verification (all run):** `uv run pytest tests/test_config.py tests/test_models.py tests/test_hf_device.py -q` → 319 passed. `uv run pytest -q` → 458 passed. `uv lock --check` OK. The 266 pre-existing config/model tests stayed green; only the test fakes were updated for device reporting, no assertion was changed. Seven mutations fail tests and were restored: CPU mode that only sets `require_gpu=False` (the real-engine test shows effGen then places the model on CUDA and the device check refuses it), CUDA preflight removed, device ignored in the cache key, post-load device check removed, CPU dtype optional, bitsandbytes allowed on CPU, `require_gpu=True` kept in CPU mode (real effGen raises `GPUPlacementError`).
- **Real-engine checks without weights:** a 2-layer random Llama plus a word-level tokenizer is generated into a temp dir; effGen's real `load_model` / `TransformersEngine` run against it (CPU and, on this GPU host, CUDA; the three CUDA tests are marked `gpu` and skip without CUDA).
- **Correction to T3.2:** its claim that `require_gpu=True` prevents CPU fallback was too broad. It only rejects partial GPU placement; with CUDA absent effGen loads silently on CPU. The CUDA preflight added here covers that case. Audit and architecture are corrected.
- **Debt / limitations:** CPU mode is verified on a tiny random model, not a real checkpoint (that is the planned Liquid AI smoke test; no weights were downloaded). `device: cuda:N` / multi-GPU selection is not supported. `fp32` on CUDA is deliberately unsupported. CPU inference speed, thread settings and memory on the edge device are untested. `use_flash_attention=False` is passed only in CPU mode.

### T3.2b Liquid AI CPU smoke test (2026-10-16)

- **Scope:** one real checkpoint, CPU only, one load and two tiny inferences. No orchestration changes, GPU inference, quantization work or workspace tools. Not in the original plan; requested by the project's senior as an edge-device feasibility check.
- **Model decision:** the target `LiquidAI/LFM2.5-350M` was kept. The model card describes it as a "general-purpose instruction-tuned model" (a separate `LFM2.5-350M-Base` exists), ChatML-like template, 16 layers (10 conv + 6 attention), 32,768-token context, recommended sampling temperature 0.1 / top_k 50 / repetition_penalty 1.05, recommended for "data extraction, structured outputs, and tool use" and "not recommended for knowledge-intensive tasks and programming". No alternative was needed.
- **Pre-download compatibility checks (no weights):** Hub `config.json`: `model_type: lfm2`, `architectures: [Lfm2ForCausalLM]`, saved with transformers 5.0.0rc1; the installed transformers 5.20.0 contains `transformers.models.lfm2`, `AutoConfig` resolved it to `Lfm2Config` and `Lfm2ForCausalLM` exists; no `trust_remote_code` needed.
- **Files:** `tests/integration/test_liquid_cpu.py` (1 test, marker `real_model`), `tests/test_integration_optin.py` (2 tests), `tests/conftest.py` (`--run-real-models` option / `SMOL_RUN_REAL_MODELS=1`, skip hook), `pyproject.toml` (marker `real_model`). No source module changed.
- **Opt-in design:** tests marked `real_model` are skipped with a reason unless `--run-real-models` (or the env var) is given; `-m real_model` alone does not opt in (both checked by subprocess tests). The test resolves the pinned revision through `huggingface_hub.snapshot_download(local_files_only=True)` and only downloads if the snapshot is missing; it then loads from the local snapshot path with the network blocked. It writes its measurements to `/tmp/smol-harness-lfm-cpu-smoke.json` (override with `SMOL_SMOKE_REPORT`) and prints them.
- **Run:** `uv run pytest tests/integration/test_liquid_cpu.py --run-real-models -s` → 1 passed in 6.5 s. Hard assertions: weights load through `build_model`; all parameters fp32 on CPU; no CUDA memory/context; `generate` and a minimal `Agent.run` return real text (`success=True`, `termination="done"`); both answers contain "paris" (a coherence check, since a wrongly loaded model emits junk); the process is not listed by `nvidia-smi`. No scripted output is involved.
- **Results:**

| Item | Value |
|---|---|
| Model / revision | `LiquidAI/LFM2.5-350M` @ `9e6c6ccf47cd318696e137d381a7ded8fe4df09f` (Hub commit, `lastModified` 2026-08-05); 354,483,968 parameters; BF16 safetensors, 714 MB; not gated; license tag `other` (`lfm1.0`) |
| Versions | Python 3.12.13, effGen 1.3.0, transformers 5.20.0, torch 2.14.1+cu130 |
| Spec | `ModelSpec("hf", <local snapshot path>, "none", dtype="fp32", device="cpu")` through `build_model` (kwargs: `quantization_bits=None`, `require_gpu=False`, `device_map="cpu"`, `use_flash_attention=False`, `torch_dtype=float32`) |
| Host | AMD Ryzen 7 7840HS, 16 logical CPUs, torch used 8 threads; actively cooled laptop with a CUDA GPU present (not the fanless edge device) |
| Load time | 3.3 s (test run), 4.9 s (earlier exploratory run), from the page cache after a one-off 60 s download |
| Inference | `model.generate`: 0.31 s, 2 tokens, text `Paris`. `Agent.run`: 0.21 s, `success=True`, `termination=done`, 1 iteration, output `Paris`. Prompt: "What is the capital of France? Answer in one short sentence." Sampling from the model card: temperature 0.1, top_k 50, repetition_penalty 1.05; max tokens 32 (generate) / 48 (agent) |
| Memory | fp32 weights 1352 MiB; process RSS 597 MiB before load, 2171 MiB after, 2194 MiB at the end; process peak (VmHWM) 2549 MiB (the BF16 file is upcast to fp32 while loading) |
| CPU-only evidence | engine `device == "cpu"`; all parameters `cpu`/`float32`; CUDA bytes allocated delta 0, reserved delta 0, no CUDA context created; `nvidia-smi` does not list the process; network blocked during load and inference |
| Temperatures | `acpitz` zone: 59 C before, 62 C after load, 63 C after inference. The only sensor exposed on this laptop. A before/after snapshot of an actively cooled host, **not** evidence of thermal safety on a fanless device. No power data (RAPL not read) |

- **Verification:** `uv run pytest -q` → 460 passed, 1 skipped (the opt-in test, 24 s). `uv lock --check` OK. Loads performed in total: one exploratory run and one test run (no repeated or sustained inference).
- **Limitations / unresolved:**
  - Only a trivial one-word factual answer was tested; reliability of effGen's final-answer format on longer or tool-using prompts is unknown (the card calls the model unsuited to programming and knowledge tasks, so it is a doubtful coding worker; it targets extraction, structured output and tool calling).
  - Native tool calling is untested: effGen reports `supports_tool_calling() == True` (the chat template renders tools), so `tool_calling_mode="auto"` would pick native mode, while the card says the model emits Pythonic `<|tool_call_start|>` calls by default; whether effGen parses them is unverified.
  - effGen reports a 128,000-token context (`max_position_embeddings` in `config.json`) while the card states 32,768; effGen plans its context budget from the former.
  - `causal_conv1d` is not installed, so transformers uses its reference PyTorch conv implementation (correct, slower); this is a CUDA-only kernel so the CPU path would not use it anyway. CPU speed is therefore not representative of an edge SoC.
  - `Agent.run` reported `tokens_used=32` while the same prompt through `model.generate` reported 2; what effGen counts there was not investigated.
  - The load peak (2.5 GiB RSS) is dominated by the fp32 upcast; an edge device with little RAM may need bf16 (not tested on CPU) or a smaller footprint.
  - Temperatures come from one sensor on an actively cooled laptop; no thermal, power or sustained-load conclusions can be drawn for the fanless device.
  - The license tag is `other` (LFM1.0); terms were not reviewed.

### T3.2c Liquid AI tool-calling compatibility (2026-10-16)

- **Question:** can `LiquidAI/LFM2.5-350M` work as a tool-using worker under effGen 1.3.0? Native vs ReAct. No harness or orchestration change, no parser workaround.
- **Files:** `tests/tool_probe.py` (measurement helper), `tests/test_tool_probe.py` (7 scripted tests that validate the measurement), `tests/test_lfm_parser_compat.py` (13 tests pinning how effGen 1.3.0 reads LFM call formats; no model), `tests/integration/test_liquid_tools.py` (2 opt-in real-model tests), `tests/test_integration_optin.py` (guard now covers all of `tests/integration`). No source module changed.
- **Method:** the real model on CPU (`ModelSpec("hf", <snapshot>, "none", dtype="fp32", device="cpu")` via `build_model`), network blocked, fixed seeds, `max_tokens=64`, `max_iterations=3`, `max_context_length=32768`, card sampling (temperature 0.1, top_k 50, repetition_penalty 1.05). One counting tool `add(a: int, b: int) -> int`; every model call is recorded (prompt, raw text) by a spy on `model.generate`. A run counts as **tool-calling success** only if the tool really executed, effGen dispatched exactly those calls, a prompt sent *after* the execution contains the sum, the agent finished `done`, and the answer contains the sum; a correct answer without an executed tool is never a success (checked by scripted tests).
- **Results (opt-in run, 5 agent runs, 34 s total):**

| Mode (2 seeds each, same task `Use the add tool to calculate 4817 + 9235. Then state the result.`) | Model selected the tool | effGen parsed it | Tool executed | Result returned to the agent | Final answer succeeded | Tool-calling success | Model calls / tool calls / termination |
|---|---|---|---|---|---|---|---|
| **native** (`tool_calling_mode="native"`) | 2/2 (only after effGen's nudge) | **0/2** | **0/2** | 0/2 | 0/2 | **0/2** | 3 / 0 / `error`, `stop_reason=written_tool_call` |
| native + card's "Output function calls as JSON." as `system_prompt` (supplementary, 1 seed) | 1/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 3 / 0 / `error`, `written_tool_call` |
| **react** (`tool_calling_mode="react"`) | 2/2 | **2/2** | **2/2** (`add(4817, 9235)`) | 2/2 | 2/2 (`14052`) | **2/2** | 2 / 1 / `done`, `final_answer` |

- **Raw outputs:**

- **native, seed 0 / 1, call 1 (no tool call; prose with a wrong sum):** `'The tool can be used directly. Call the tool with parameters `4817` and `9235`. The final answer is **4817 + 9235 = 12002**.'` / `'... The final answer is 4818.'`
- **native, calls 2 and 3 (the model's real tool call, in its default Pythonic format; effGen could not run it):** `'<|tool_call_start|>[add(a=4817, b=9235)]<|tool_call_end|>'`. effGen's final output: "The model wrote a 'add' tool call into its answer instead of calling the tool, so add never ran and nothing the answer describes was carried out. ..." (`termination=error`, `stop_reason=written_tool_call`).
- **native + JSON system prompt, call 1 / 2 / 3:** `'The add tool is available. Call it and read the result. Final answer: 4817 + 9235 = 11002.'`, then the same Pythonic call twice. The instruction changed nothing.
- **react, seed 0, call 1:** `'Action: add  \nAction Input: {"a": 4817, "b": 9235}  \nObservation: the result is 51002\n\nThought: I now know the final answer.'` (seed 1: `... the result is 50000 ...`). The model invents its own `Observation` after the action; effGen runs the real tool and discards everything after the action. **Call 2:** `'Final Answer: 14052'` (the true sum, read from the real observation in the prompt).

- **Parser compatibility** (deterministic, pinned in `tests/test_lfm_parser_compat.py`):

| Text | native | react | hybrid |
|---|---|---|---|
| `<|tool_call_start|>[add(a=4817, b=9235)]<|tool_call_end|>` (LFM default, Pythonic) | not a call (reads as answer text) | not a call | not a call |
| `<|tool_call_start|>[add(4817, 9235)]<|tool_call_end|>` (positional) | not a call | not a call | not a call |
| `<|tool_call_start|>[{"name": "add", "arguments": {...}}]<|tool_call_end|>` (card's JSON option) | **call** (generic JSON reader) | not a call | **call** |
| bare `{"name": "add", "arguments": {...}}` | call | not a call | call |
| `Action: add` / `Action Input: {...}` | not a call | **call** | **call** |

- **Why native fails:** LFM2.5 *does* select the tool (2/2) but writes it in its default Pythonic format, which effGen's native reader has no branch for (it supports Qwen `<tool_call>` JSON, XML-parameter, Llama `<|python_tag|>`, Mistral `[TOOL_CALLS]`, generic JSON and Gemma formats). `<|tool_call_start|>`/`<|tool_call_end|>` are *non-special* added tokens (ids 10/11), so effGen's special-token strip leaves them in the text; they reach the parser intact. effGen's loop recognises the written call (`written_tool_call`) but never executes it. effGen's own error message advises using a model that calls tools.
- **Why the card's JSON option did not help:** effGen's `TransformersEngine._apply_chat_template` wraps the whole flat prompt, including `AgentConfig.system_prompt`, as **one user message** (`[{"role": "user", "content": prompt}]`), so the instruction never reaches a system turn; the model ignored it in the user turn. Whether a true system-role instruction would produce JSON calls was not tested (it would need an engine change or template kwargs effGen does not offer).
- **Effective context window (verified numbers, see audit):** config `max_position_embeddings=128000` (RoPE default, no scaling), tokenizer `model_max_length` unbounded, model card 32,768. effGen takes the config value, so its `auto` budget is a 101,836-token prompt budget (window 128,000, reserve 8,192, headroom 0.85); with `AgentConfig.max_context_length=32768` effGen's window becomes 32,768 and the prompt budget 20,889 (override verified against the real model). The model's own quality beyond a few hundred tokens was **not** measured; 32,768 is the card's figure, used as a conservative planning limit, not a tested maximum (the config declares 128,000). These runs used prompts of at most about 2,000 characters.
- **Evidence levels.** *Tested at runtime with the real model* (CPU, fp32, seeds 0 and 1, one task): ReAct tool calling 2/2 and native 0/2 with the outcomes above; same seed reproduces the same text; the JSON-system-prompt variant (1 seed) changed nothing; effGen's budget numbers with and without the `max_context_length` override. *Tested without a model* (deterministic, pinned in `tests/test_lfm_parser_compat.py`): which call formats each effGen parser reads, and that the engine wraps the whole prompt as one user message. *Source-level only*: the list of native formats effGen supports; that `<|tool_call_start|>` is not in effGen's delimiter set and is a non-special token; which of the loop's readers produced `written_tool_call` (not traced). *Untested*: native or hybrid mode with a JSON call format from the model; `tool_calling_mode="auto"` for this model (set `react` explicitly); multi-step, multi-argument or string-argument tools; delegation with this model; long-context behaviour; the edge device.
- **Recommendation for future experiments:** use `tool_calling_mode="react"` for LFM2.5-350M (works with no code change) and set `max_context_length=32768`. Do not use `native` or `hybrid` for the default call format. Options that need explicit approval: (a) a small parser adapter for `<|tool_call_start|>[fn(k=v)]<|tool_call_end|>`; (b) an effGen/engine change to deliver a system-role instruction asking for JSON calls (the JSON-list form is already parseable).
- **Verification:** `uv run pytest tests/integration/test_liquid_tools.py --run-real-models -s` → 2 passed in 34 s (its assertions: CPU-only and no CUDA allocation, the requested mode really engaged, effGen's tool-call record equals the real executions, success only with an executed tool, at least one ReAct run succeeds; native outcomes are recorded, not pinned). `uv run pytest -q` → 480 passed, 3 skipped (all real-model tests; opt-in guard verified). `uv lock --check` OK. Total real inference: about 15 short agent runs (10 exploratory, 5 in the test), each at most 3 model calls and 64 new tokens, plus three model loads.
- **Limitations:** two seeds of one trivial task; the model fabricates observations in ReAct (effGen's written-action guard discards them), so a task where the invented value changes the plan could behave differently; no multi-step, multi-argument or string-argument tool, no delegation; arithmetic correctness of the model on its own is poor (it answered 12002, 4818, 11002 without the tool), so reliability depends on the tool result actually being used; the card advises against programming use; CPU timing is from a laptop (4.7–6.1 s per run), not the edge device.

### T4.1 Tool factory and workspace-confined `file_operations` (2026-10-16)

- **Files:** created `src/smol_harness/tools.py` (about 85 lines) and `tests/test_tools.py` (51 tests). No other source changed; no effGen change; `build_harness` is not wired to it (still the unplanned wiring step).
- **API:** `build_tools(names, workspace) -> list[BaseTool]`, `resolve_workspace(workspace) -> Path`, `ToolBuildError(ValueError)`, `TOOL_NAMES == config.KNOWN_TOOLS == ("file_operations", "shell")`.
  - Names are checked first, all-or-nothing: unknown names (including every web/search/browser/fetch name), `delegate` ("the harness adds it to the orchestrator only"), duplicates and a bare string are refused. `[]` builds nothing.
  - The workspace must be a non-empty path that resolves (`strict=True`, so broken symlinks fail) to a directory this process can read, write and search. A symlinked workspace confines to its target. It is validated even for `[]`.
  - `file_operations` is effGen's `FileOperations(allowed_directories=[workspace])`, a fresh instance per call. After construction the factory reads back the root effGen kept (`_allowed_directories`, private) and refuses to return the tool unless it is exactly `[workspace]`.
  - `shell` raises `ToolBuildError` naming T4.2: fail closed, never skipped, never `BashTool`.
- **Surprising effGen behaviour found (runtime, pinned by regression tests; audit §1.9):** `FileOperations` given a missing allowed directory only logs a warning and **confines to the current directory instead** (or to `EFFGEN_WORKSPACE`, which it creates). Without the factory's checks a vanished or mistyped workspace would have silently widened confinement to the harness repository. Also pinned: extensionless files (`Makefile`, `README`, `.gitignore`, `Dockerfile`) are refused for read and write; credential filenames (`.env`) and dotenv-shaped content in any file are refused on read; `EFFGEN_WORKSPACE` does not override an explicit workspace.
- **Confinement verified through `BaseTool.execute()`** (all refused with `outside allowed directories`, and the outside directory checked unchanged): `../` and `sub/../../` traversal, `..` itself, absolute paths outside (including `/etc/hostname`, `list /`), a file symlink, a directory symlink, a dangling symlink whose target is outside (write), `metadata`/`list` on those symlinks; `list -r` and `search` never return symlinked or parent entries (a `../outside/*` glob returns `[]`). Allowed: relative and absolute paths inside, nested writes creating parents, a symlink to a file inside, relative paths resolving against the workspace even when the cwd is elsewhere. A NUL byte or an empty path fails cleanly.
- **Repository discovery:** `list` (optionally recursive) and `search` (filename glob only, optionally recursive) work inside the workspace; there is no content search (that is `grep` through T4.2's shell).
- **Harness check (scripted models, real effGen loop):** orchestrator built with `build_tools(["file_operations"])` plus the harness's `delegate`; worker `coder` with `["file_operations"]`; worker `idle` with `[]`. Recorded tool sets of every `Agent` constructed: orchestrator `{delegate, file_operations}`, coder `{file_operations}`; no name matches `web|search_engine|url_fetch|http`. The worker wrote `notes/out.txt` in the workspace through the real tool; its `../outside/pwned.txt` attempt reached its next prompt as an `outside allowed directories` observation and wrote nothing.
- **Verification:** `uv run pytest tests/test_tools.py -q` → 51 passed in 0.18 s. `uv run pytest -q` → 531 passed, 3 skipped (the 3 opt-in real-model tests) in 23 s. `uv lock --check` → Resolved 157 packages (no dependency change).
- **Security limitations (this is path validation, not a filesystem sandbox):**
  - Check-then-use race: effGen resolves and checks the path, then opens it in a separate step. A concurrent process that swaps a checked directory for a symlink between the two could escape. Nothing in the harness runs concurrently with a tool today, but T4.2's shell can create symlinks (a later `file_operations` call resolves them and refuses escapes; only the race window is open).
  - Hard links and bind mounts inside the workspace that point outside cannot be detected by path checks.
  - No write-size limit (reads are capped at 10 MB by effGen); `list -r` over a large tree (e.g. `.git`) returns everything.
  - The workspace root is not restricted (e.g. `/` or `$HOME` would be accepted if the config says so).
  - Reads of the whole workspace are allowed, including `.git/` contents with allowed extensions.
- **Deviations from the proposed design:** none in substance. Additions: the read-back of the kept root (closes the fallback above), the empty-path and access checks, rejection of duplicates and bare strings. `ToolBuildError` subclasses `ValueError`, as proposed.

### T4.2 Sandboxed `shell` tool (2026-10-17)

- **Files:** modified `src/smol_harness/tools.py` (now about 270 lines) and `tests/test_tools.py` (121 tests, 70 new). No effGen change, no `build_harness` wiring, no config change.
- **Backend selected:** effGen's `SubprocessSandbox`, instantiated directly (never `get_sandbox`, so `EFFGEN_SANDBOX_BACKEND` cannot select `off`, and Docker is never used). **Deviation from the plan:** `ShellTool` is a small `BaseTool` over `SubprocessSandbox` (both public in `effgen`), not a `CodeExecutor` subclass: `CodeExecutor` exposes language, network and memory as model-facing parameters and would need private attributes (`_sandbox`, `_sandbox_config`) to pin them. Reused: `SubprocessSandbox` (all isolation), `SandboxConfig`, `BaseTool`, `ParameterSpec`.
- **API:** `ShellTool(workspace, *, memory_limit="1g", max_timeout_s=300)`; `build_tools(["shell"], workspace)` builds one with defaults (the YAML has no field for them yet). Parameters: `command` (1..10,000 chars, no NUL, not blank) and `timeout` (1..300 s, default 30). Unknown parameters are **rejected** (effGen's `BaseTool` would silently drop them). Result: `exit_code`, `stdout`, `stderr`, `timed_out`, `truncated`, `filesystem_confined`, `network_isolated`, `writable_root`; non-zero exit or timeout is a failed result whose `error` carries the exit code and the last 8,000 characters of each stream (effGen shows a model only `error` for failures, so a failing `pytest` reaches it with its output); stdin is `/dev/null`.
- **Fail closed (three layers):** (1) at build time: not Linux, or `bash`/`unshare`/`timeout` missing → `ToolBuildError` (the message names WSL2 for Windows, not tested); (2) before the first command, a fixed no-op runs through the same path and the sandbox must report write confinement to exactly the workspace, network isolation and a private process table, otherwise `ShellSandboxError` and the model's command never runs (tested: with user namespaces simulated away, effGen quietly runs plain `bash`, and the command's marker file was not created); (3) every result is re-checked and discarded if a guarantee is missing. A failed proof is never cached.
- **Acceptance results (all through `BaseTool.execute()` with the real sandbox on this host; Linux 6.18, unprivileged user namespaces available):**

| # | Requirement | Result |
|---|---|---|
| 1 | harmless command | `echo`, `pwd`, pipes, `&&` work; `exit 3` gives `exit_code=3` and a failed result with the output in `error` |
| 2 | starts in the workspace | `pwd` is the workspace; relative reads/writes hit it; `file_operations` and `shell` see the same files |
| 3 | writes outside refused | 8 write forms (redirect, `mkdir`, `rm`, `mv`, `cp`, `../`, absolute, sibling) → `Read-only file system`; writes through file/directory symlinks refused; `mount -o remount,rw` and a nested `unshare` remount denied; host `/tmp` writes never reach the host |
| 4 | reads outside | **NOT confined** (tested and pinned): sibling directories, `/etc`, symlinks to outside, `.env`, extensionless files all readable; `~/.ssh`, `~/.netrc` etc. masked by effGen's deny-list (tested with a fake `$HOME`). The shell therefore reads what `file_operations` refuses |
| 5 | network | TCP to host loopback and to a public IP, and DNS: all fail; only `lo` in `/proc/net/dev`. See R1 for what is *not* covered |
| 6 | timeout | `sleep` with `timeout=2` returned in under 8 s, `timed_out=True`; a command's own `exit 124` is not reported as a timeout |
| 7 | children | background, `setsid`, `nohup`, `trap '' TERM` loops: none survived (checked in the host `/proc`); backgrounded daemons are also killed when a command exits normally |
| 8 | memory | `ulimit -v` enforced: 512 MiB allocation fails with `MemoryError` at 256m and succeeds at 2g; default 1g; a real `pytest` run (1 pass, 1 fail) works under the default |
| 9 | allowlist | a worker built without `shell` that asks for it runs nothing (marker file absent); the shell has no way to call other agent tools; the only parameters are `command` and `timeout`; `delegate`/web names still unbuildable |
| 10 | sandbox unavailable | off Linux, missing binaries, no namespaces, each missing guarantee, spawn error, a sandbox that never returns: all fail, none run the command |

- **Two effGen defects found and worked around (no effGen change; pinned by regression tests; audit Part 3):**
  - **Timeout does not clean up and can hang.** On its own deadline `SubprocessSandbox.run` kills only the outer `unshare`, leaves every descendant running, and then awaits `proc.wait()`, which asyncio resolves only after all pipes close. A command that leaves a background process makes `run()` block until that process ends (measured: orphaned `sleep`s alive and the call stuck). Workaround: `ShellTool` runs `timeout -k 2 N bash -c <cmd>` as the script inside the PID namespace; when PID 1 exits the kernel kills everything left in it. effGen's own deadline stays as a later backstop, plus a hard `asyncio.wait_for` of ours.
  - **A network namespace does not isolate pathname Unix sockets** (R1 below). Workaround: an empty read-only tmpfs over `/run` before each command.
- **R1: UNRESOLVED SECURITY LIMITATION, accepted for trusted local development (user decision, 2026-10-17).** On this host (a desktop in the `docker` and `wheel` groups) the unmasked sandbox could **connect** to the Docker daemon socket (root-equivalent), the session and system D-Bus, the systemd user manager's private socket, gpg-agent and PipeWire, and systemd-resolved answered DNS (an egress channel through query names). I only tested connectivity; I did not send commands to any of them or try to exploit this. The `/run` mask closes all of those (tested; mutation-checked: removing the mask fails the tests). **Still reachable:** any other pathname Unix socket the user can reach, e.g. on this host `~/.cache/docker-secrets-engine/engine.sock`, editor and agent sockets under `~/` (`~/.codex`, `~/.local/share/zed`, `~/.zeron`), `/etc/pacman.d/gnupg/S.*`, and, when the **workspace is under `/tmp`**, every socket in the host `/tmp` (effGen gives a private `/tmp` only when the workspace is elsewhere; tested both ways). A path deny-list cannot enumerate these. A hostile command (e.g. a prompt-injected agent) could talk to them; a merely buggy small model will not. **Decision:** accepted for the MVP in the trusted scope only; the shell must not run untrusted repositories or adversarial workloads. For those, a dedicated, independently isolated container or VM is required, with its controls verified separately (the harness would run inside it; this also is the route for Windows). **Not doing:** a seccomp filter, new sandbox backends or other isolation infrastructure. Consequences to keep in mind: do not put secrets, `.env` files or other sensitive data in a workspace or anywhere the user can read when running the shell; do not run it on a machine whose user can reach a Docker socket under a non-`/run` path or other sensitive local services; prefer a workspace outside `/tmp`.
- **Unix-level facts measured (for tuning):** `ulimit -u 256` counts the user's *total* processes (148 running here); a user with more than about 256 processes cannot fork inside the sandbox. A trivial command takes about 0.1 to 0.3 s. Output cap 8,000 chars per stream (tail). The default memory floor is 1g (effGen's 256m default also ran a trivial `pytest`, but node/V8 and real toolchains need more; U8 answered only for a trivial suite).
- **Not enforced (documented limits):** reads (R1 and item 4); CPU quota beyond `ulimit -t` (timeout + 5 s of CPU time); disk quota inside the workspace; output volume before truncation (a command can print gigabytes into a pipe until the timeout); process count is per user not per sandbox; `/dev` devices (e.g. `/dev/kvm`, GPU) stay reachable; hard links and bind mounts inside the workspace; kernel/namespace escapes (this is not a VM or a container boundary); `ulimit -v` is address space, not resident memory. After a **hard-deadline** failure (effGen's backstop never returned) processes may still be running; the normal path cannot leave any.
- **Scope and prohibitions:** **Accepted scope (user decision, 2026-10-17): the shell is for trusted local development and controlled test repositories only. Untrusted repositories and adversarial agent workloads must NOT run in this sandbox; they require a dedicated, independently isolated container or VM whose controls are separately verified. No seccomp filter, new backend or other isolation infrastructure will be added in the MVP.** Unresolved at the end of T4.2: no host **read** isolation (any file the user can read is readable); Unix sockets outside `/run` reachable (R1); DNS is blocked only because `/run` (systemd-resolved) is hidden and the network namespace has no interfaces; a resolver reachable through some other socket path would not be covered; kernel or namespace escapes are out of scope of what was tested.
- **Windows 11 (the eventual experiment machine):** **not tested, not supported natively.** Off Linux, effGen's `SubprocessSandbox` returns a plain `bash -c` plan with no isolation (source-level), so `ShellTool` refuses to build (`platform.system() != "Linux"`; tested by monkeypatching, not on Windows). Docker is no alternative: effGen's `DockerSandbox` does not mount the workspace. The practical route is to run the whole harness inside **WSL2** with the repository on the WSL ext4 filesystem (not `/mnt/c`), where the same Linux code path applies. **Windows and WSL2 compatibility are unverified.** WSL2 needs unprivileged user namespaces and `unshare`, `timeout`, `bash`; none of that was verified, and R1 applies there too (a WSL distro normally has no Docker or D-Bus session unless the user installs them, which would make R1 smaller, but this is unmeasured). `resolve_workspace`'s drive-root check is tested with `PureWindowsPath` semantics only.
- **T4.1 correction:** `resolve_workspace` now also rejects a filesystem root (`/`, a drive root or a UNC share root via `path.parent == path`, including a symlink that resolves to `/`) and the home directory or any of its ancestors (so `/home` and `$HOME` are refused; subdirectories are fine). The check runs before the access check so the message is accurate. Tests: `test_filesystem_roots_are_rejected_as_workspaces`, `test_home_directory_and_its_ancestors_are_rejected_but_subdirectories_are_not`, `test_a_symlink_to_the_filesystem_root_is_rejected`, `test_root_detection_follows_windows_drive_semantics`. No other change to the file tool. Not solved (as instructed): check-then-use races.
- **Verification:** `uv run pytest tests/test_tools.py -q` → 121 passed in 38 s (no skips on this host). `uv run pytest -q` → 601 passed, 3 skipped (the 3 opt-in real-model tests) in 65 s. `uv lock --check` → Resolved 157 packages (no dependency change). Leak check after the suite: no stray processes, no temporary directories. Mutation check: with the `/run` mask removed, the socket tests fail. No real model, no network, no download; all boundary tests ran in `pytest` temp directories, and exploration scripts lived in `/tmp/fo-scripts` (outside the repo). Unix-socket probes were connect-only.
- **Unverified:** Windows and WSL2; hosts without unprivileged user namespaces (simulated by flipping effGen's cached probe flags, not on a real such host); Docker/Firecracker backends (deliberately unused); behaviour with a model-chosen real workload beyond a small `pytest` run; memory use of the `ulimit -v` floor for big toolchains.
- **Decisions:** `BaseTool` over `SubprocessSandbox` instead of a `CodeExecutor` subclass; timeout enforced inside the namespace; `/run` hidden by a tiny prelude rather than leaving the gap undocumented; the proof of enforcement runs before the first model command; unknown parameters rejected; failing commands are failed results (the observation carries the output).

### Review 001, F2 remediation (phase A) and F1 blocker (2026-10-17)

- **Files:** modified `src/smol_harness/tools.py` (`_echo`, `_report`, new `WorkspaceFileOperations`), `tests/test_tools.py` (+11 tests, 132 total), `tests/test_loop_guards.py` (14 tests). No effGen change, no change to `harness.py` or `delegation.py`.
- **F2 fixed.** `shell` reports now start with `$ <command>` (long commands: first 200 characters, the size of the rest and a sha256 prefix); a successful `file_operations` write reports `wrote <path> (<bytes> bytes, <n> lines, sha256 <8>)` read back from the file on disk. Structured fields and failure semantics are unchanged. Deliberately **no per-call counter**: it would blind every result-text guard (see the review report, Follow-up 001-B).
- **(Superseded by the F1 fix below.)** F1 was blocked on approval of the process-global adapter; it is now implemented.
- **Verification:** `uv run pytest tests/test_loop_guards.py -q` → 10 passed, 4 xfailed; tools + delegation + e2e + limits → 254 passed; `uv run pytest -q` → 622 passed, 3 skipped, 4 xfailed; `uv lock --check` → 157 packages, unchanged. Mutation check: removing the echo and summary makes three F2 tests fail.

### Review 001, F1 fix: explicit effGen loop compatibility patch (2026-10-17)

- **Files:** modified `src/smol_harness/harness.py` (`install_effgen_loop_compat`, `effgen_loop_compat_installed`, `uninstall_effgen_loop_compat_for_tests`, `EffgenCompatError`, a guard in `Harness.run`), `src/smol_harness/delegation.py` (`STATEFUL_TOOL_MARKER`; `DelegateTool` marked), `src/smol_harness/tools.py` (`ShellTool`, `WorkspaceFileOperations` marked), `tests/conftest.py` (session fixture that installs the patch explicitly; `compat_uninstalled` fixture), `tests/test_loop_guards.py` (35 tests), `tests/test_limits.py` (one expectation updated, one test added). No effGen change.
- **What it is:** a **process-wide, explicit, idempotent** rebinding of `effgen.core.agent_loop.NativeToolLoop` to a marker-keyed subclass; only `shell`, `file_operations` and `delegate` bypass effGen's pre-dispatch exact-repeat replay/decline; every other guard and every unmarked tool keeps effGen's behaviour. It verifies effGen's version (`1.3.0`), the private names and the construction site at install time and raises `EffgenCompatError` (patching nothing) otherwise. `Harness.run` refuses to run, before any model call, if it is not installed. Importing or constructing a harness installs nothing. The harness never uninstalls it. Concurrent or during-run installation/restoration is unsupported.
- **Private-API dependencies / upgrade risk:** see `docs/effgen-api-audit.md`, "F1 compatibility patch". The `==1.3.0` pin and the install-time checks make an effGen change fail loudly.
- **Behaviour change to know:** an orchestrator retrying the identical failing delegation now runs a real worker each time (bounded by `max_delegations` and `max_iterations`); `test_retrying_the_identical_failing_delegation_forever_ends_stuck_not_tool_failed` was updated accordingly (it had encoded the F1 behaviour).
- **Verification:** `uv run pytest tests/test_loop_guards.py -q` → 35 passed; tools + delegation + e2e + limits → 255 passed; `uv run pytest -q` → 648 passed, 3 skipped, 0 xfailed; `uv lock --check` → 157 packages, unchanged. Mutation checks: patch not installed → the four F1 tests fail; marker widened to all objects → four scoping tests fail.
- **Requirements added to later tasks:** T5.0 and T5.1 must call `install_effgen_loop_compat()` explicitly before any agent runs; `Harness.close()` must not uninstall (test it).

### Review 001, F3 fix: stopped orchestrators return results (2026-10-17)

- **Files:** modified `src/smol_harness/harness.py` (`Harness.run`, `_stopped_response`, `_keep_records`); created `tests/test_orchestrator_failure.py` (28 tests). No effGen change; the F1 patch and its installation requirement, delegation, tools and model loading are untouched.
- **Behaviour:** effGen raises `RunStoppedError` only for `outcome == "stopped"` and it carries the same `AgentResponse` that `raise_on_error=False` returns; `Harness.run` now builds the normal failed `HarnessResult` from it (real `termination`, `stop_reason`, this run's worker records) in both modes. The response is validated (`success is False`, has `output`); a malformed error propagates, nothing is invented. Every other exception propagates unchanged with `exc.worker_records` (and a note) so the work already done is not lost.
- **Limitation (effGen's, documented):** under `raise_on_error=True` a non-stopped orchestrator failure (provider error, guardrail block, a model that raises) raises with no response; under `False` it is a failed result. T5.1 must treat both.
- **Verification:** `uv run pytest tests/test_orchestrator_failure.py -q` → 28 passed; delegation, e2e, limits, loop guards, tools → 290 passed; `uv run pytest -q` → 676 passed, 3 skipped; `uv lock --check` → 157 packages, unchanged. Mutation checks: handling disabled → 7 failures, validation removed → 4, records not attached → 13.
- **New finding F13 (P3, not fixed):** `DelegateTool._delegate` fails inside its own `except RunStoppedError` branch when a worker's error has no usable response, and no `WorkerRecord` is created.

## 5. Upcoming work

0. **Before T5.0 (from `docs/reviews/opus-review-001.md`, 2026-10-17):** all three P1 findings are **fixed** (F1, F2, F3; §4). **Commit the F3 work.** T5.0 should absorb F5 and F8 from the review (explicit effGen flags, shell timeout capped at the worker deadline), call `install_effgen_loop_compat()` (and keep `Harness.close()` from uninstalling it), and T5.1 must handle both a failed `HarnessResult` and a propagated orchestrator exception (with `worker_records`). P3 findings F10 to F13 can be taken in the relevant milestone.
1. **T5.0** (next after the above, §2 M5), then **T5.1** (CLI), **T5.2** (toy-repo smoke test, trusted repository only); T4.3 only if T5.2 needs it.
2. ~~Unplanned wiring step~~ now scheduled as T5.0.
3. **Blocked or hardware-dependent:** T3.3 (needs Ollama), T3.4 (needs `bitsandbytes` and a real quantized GPU load), T3.5 (needs T3.3 and T3.4).
4. **Open decisions for the user** (nothing is started): approve or reject a parser adapter for LFM's Pythonic call format or an engine change to send a system-role instruction (native mode); run the opt-in tests on the fanless edge device.

## 6. Changes to scope or architecture

_None yet._ Record the reason, the affected tasks, and the updates made to `architecture.md` / `new-session.md`.
