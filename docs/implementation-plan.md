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

### M3: Configuration and real model backends

**T3.1 `config.py` (YAML → dataclasses + validation)**

- **Status:** Completed (2026-10-16). Verification: see §4.
- **Dependencies:** T1.3.
- **Acceptance:**
  - The example config loads.
  - Invalid configs are rejected with clear errors: unknown provider, `bnb-4bit-fp4`, a quant field on ollama other than `server`, a worker with `delegate`, an unknown tool, duplicate names.
- **Tests:** `tests/test_config.py`.

**T3.2 `models.py` `build_model` + cache**

- **Status:** Not started
- **Dependencies:** T3.1.
- **Changes:**
  - Map `ModelSpec` to `effgen.models.load_model` kwargs, always passing `quantization_bits` for hf.
  - Cache by spec.
- **Acceptance:** Unit tests monkeypatch `load_model` and assert the exact kwargs for each provider and quant.
- **Tests:** `tests/test_models.py`.

**T3.3 Ollama smoke test**

- **Status:** Blocked (Ollama not installed on this host as of 2026-10-16)
- **Dependencies:** T3.2. Ollama installed and a small tag pulled (e.g. a `q4_K_M` instruct model).
- **Acceptance:** `pytest -m ollama` builds an agent on the tag and answers a one-line task. Record the tool-calling strategy effGen resolved and the `context_length` used.
- **Tests:** `tests/integration/test_ollama.py` (skipped when the server is not reachable).

**T3.4 HF Transformers smoke test (bnb NF4)**

- **Status:** Not started
- **Dependencies:** T3.2. `bitsandbytes` installed; CUDA torch working.
- **Acceptance:**
  - `pytest -m gpu` loads a ~0.5-1.5B instruct model with `bnb-4bit-nf4` and answers.
  - Record VRAM used.
  - Also verify `quantization: none` does **not** auto-quantize (inspect `model.quantization_bits`).
- **Tests:** `tests/integration/test_hf.py`.

**T3.5 Mixed-provider delegation, real models**

- **Status:** Not started
- **Dependencies:** T3.3, T3.4, T2.3.
- **Acceptance:** An Ollama orchestrator with an HF worker (and the reverse) completes a trivial delegated task end-to-end. Record whether the small model follows the delegate tool contract.
- **Tests:** `tests/integration/test_mixed.py` (manual / marked).

### M4: Workspace tools

**T4.1 `tools.py` registry + `file_operations`**

- **Status:** Not started
- **Dependencies:** T3.1.
- **Acceptance:**
  - Reads and writes inside the workspace work.
  - Paths outside it (`../`, absolute, symlink escape) are refused.
  - The allowlist is honoured.
  - A test asserts that no agent built by the harness holds a tool whose name matches `web|search_engine|url_fetch|http`.
- **Tests:** `tests/test_tools.py`.

**T4.2 Sandboxed `shell` tool**

- **Status:** Not started
- **Dependencies:** T4.1.
- **Changes:** A `CodeExecutor` subclass exposing `command` (and capped `timeout`). It pins bash, network off and the configured memory limit, and requires the subprocess backend. Fail closed if the sandbox does not initialize.
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

**T5.1 CLI**

- **Status:** Not started
- **Dependencies:** T3.1, T4.2.
- **Acceptance:** `uv run python -m smol_harness --config configs/example.yaml --task "..." --json` prints the final output plus worker records. The exit code is non-zero on orchestrator failure.

**T5.2 Toy-repo smoke test**

- **Status:** Not started
- **Dependencies:** T5.1, T3.5.
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

- **Milestone:** M3 (configuration and real model backends). T3.1 done.
- **Current task:** none active. **Recommended next: T3.2 `models.py` `build_model` + cache.** It maps a `ModelSpec` to `effgen.models.load_model` kwargs (unit-testable by monkeypatching `load_model`, so no GPU or Ollama is needed). Alternative: T4.1 workspace tools (depends only on T3.1). T3.3 stays blocked until Ollama is installed.
- **Why:** the config layer now produces validated specs; nothing yet turns them into effGen models.
- **How success is verified:** `uv run pytest tests/test_models.py -q` (per the T3.2 plan entry).
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

## 5. Upcoming work

M2 and T3.1 are complete. Next: T3.2 (then T3.3 when Ollama exists, T3.4 for HF/GPU), or T4.1 after T3.1. Do M3 before M4 only if Ollama is installed by then; otherwise do M4 first (M4 depends only on T3.1).

## 6. Changes to scope or architecture

_None yet._ Record the reason, the affected tasks, and the updates made to `architecture.md` / `new-session.md`.
