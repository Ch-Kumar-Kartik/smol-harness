# New Session Handoff

_Last updated: 2026-10-16 (after T2.1: worker deadline)._

## 1. Project overview

**smol-harness** is a minimal multi-agent coding harness built on **effGen 1.3.0**.

- A larger-LLM **orchestrator** delegates self-contained tasks to smaller-LLM **workers** through one `delegate` tool, which works like a smolagents managed agent.
- Each worker is a fresh `effgen.Agent` with its own context, model, tool allowlist and limits.
- Models come from Ollama (via effGen's OpenAI-compatible adapter) or local Transformers (bf16/fp16, bnb 8-bit, bnb NF4).
- Tools are effGen's workspace-confined `FileOperations` plus a sandboxed shell built on effGen's `CodeExecutor` subprocess sandbox.

Details: `docs/architecture.md`. effGen facts: `docs/effgen-api-audit.md`. Plan: `docs/implementation-plan.md`.

## 2. Repository map (verified 2026-10-16)

Docs plus the T0.1 project skeleton. No harness logic yet.

| Path | Purpose |
|---|---|
| `CLAUDE.md` | Claude Code instructions (stable rules) |
| `AGENTS.md` | Tool-agnostic repository guidelines |
| `docs/new-session.md` | This handoff |
| `docs/effgen-api-audit.md` | Verified effGen capabilities + integration status table |
| `docs/architecture.md` | Proposed architecture and decisions |
| `docs/implementation-plan.md` | Milestones, tasks, active goal |
| `pyproject.toml`, `uv.lock`, `.python-version` | uv project (Python 3.12, `effgen==1.3.0`, dev: pytest, pyyaml) |
| `src/smol_harness/__init__.py` | Package stub (docstring only) |
| `tests/test_import.py` | Checks effGen is 1.3.0 and the package imports |
| `src/smol_harness/delegation.py` | `WorkerResult`, `WorkerRecord`, `WorkerRegistry`, `DelegateTool`, `DeadlineMiddleware`, `WorkerTimeout`. No `max_delegations` or truncation yet. |
| `src/smol_harness/harness.py` | `build_harness(..., worker_timeout_s)`, `Harness.run(task) -> HarnessResult` (fresh orchestrator + record list per run) |
| `tests/test_limits.py` | 15 tests of the cooperative worker deadline |
| `tests/test_e2e_delegation.py` | 23 end-to-end tests: delegation flow, isolation, records, U11, orchestrator-memory regression |
| `tests/test_delegation.py` | 22 tests of the delegation layer through `BaseTool.execute()` |
| `tests/conftest.py` | `ScriptedModel` deterministic fake effGen `BaseModel` (+ `scripted_model` fixture) |
| `tests/test_scripted_model.py` | 15 tests of the fake, driven through the real effGen `Agent` |

The remaining planned layout (`src/smol_harness/{config,models,tools,delegation,harness,__main__}.py`, `configs/`, `tests/integration/`) is in `docs/architecture.md` §3. **Not created yet.**

## 3. Current implementation state

| Area | State |
|---|---|
| Implemented and verified | T0.1 bootstrap; T1.1 `ScriptedModel`; T1.2 delegation layer; T1.3 end-to-end orchestrator→worker→orchestrator. **M1 complete**; T2.1 cooperative worker deadline (77 tests pass) |
| Partially implemented | Nothing |
| Not yet implemented | T2.2–T2.3 (failure-kind mapping review, `max_delegations`, truncation), `delegation_id`, config (YAML), real models (Ollama/HF), workspace tools, CLI |
| Known limitations (by design) | Sequential workers only; depth 1; **the worker timeout is cooperative** (checked before model/tool calls; an in-flight generation or tool is never interrupted); no orchestrator deadline; FP4 unsupported (effGen hard-codes NF4) |

## 4. Most recent session

- **Goal:** T2.1 (worker deadline) plus the requested orchestrator-memory regression test.
- **Done:**
  - Regression test `test_orchestrator_memory_does_not_carry_across_runs`; mutation-verified, harness restored.
  - `DeadlineMiddleware` / `WorkerTimeout` / `status="timeout"`, wired through `DelegateTool(worker_timeout_s)` and `build_harness(worker_timeout_s)`.
  - 15 limits tests. Details: `docs/implementation-plan.md` §4.
- **Files:** modified `delegation.py`, `harness.py`, `tests/test_e2e_delegation.py`; created `tests/test_limits.py`. Docs: plan, this file, audit, one row in `architecture.md` §10.
- **Tests:** `uv run pytest tests/test_limits.py -q` → 15 passed; `uv run pytest -q` → 77 passed (CPU, ~7 s). Three deadline mutations each fail tests.
- **U4 answered:** middleware exceptions propagate. See audit Part 3.
- **Problems:** none.
- **Unresolved:** U5–U8, U10. The 120 s async-bridge cap (audit T1.2 entry) remains source-level only.

## 5. Current task and next steps

- **Current milestone:** M2 (execution controls). T2.1 done. **Waiting for user approval.**
- **Next task:** T2.2 Failure propagation. Check scope first: the tests already cover a worker whose model raises, a worker that never finishes (`stuck`), an unknown worker, and a worker construction failure (T1.2/T1.3), all ending with the orchestrator completing. What T2.2 still adds is the explicit `BackendUnreachableError` and `RunStoppedError` (iteration cap) cases and a re-check that none ends the parent as `tool_failed`.
- **Then:** T2.3 `max_iterations` per agent, `max_delegations`, report truncation.
- **Relevant files:** `docs/implementation-plan.md` T2.2–T2.3; `src/smol_harness/delegation.py`; `tests/test_limits.py`.
- **Open questions:** `delegation_id` / `duration_s` (architecture §5) are still unbuilt. Decide whether T2.3 adds them.

## 6. Known issues and warnings

### Unverified assumptions

| ID | Assumption | Where it gets verified |
|---|---|---|
| U1 | ~~A scripted ReAct fake drives effGen 1.3.0's loop deterministically~~ **Verified in T1.1** for the happy path (exactly 2 calls). Nudges can consume extra turns on error paths, so use `repeat_last` | done |
| U2 | ~~The orchestrator sees the full JSON string from `DelegateTool`~~ **Verified in T1.3** (it is the `Observation` in the next prompt) | done |
| U3 | ~~A failed worker reported as a successful tool call does not trigger `tool_failed`~~ **Verified in T1.3** for a worker whose model raises (parent ends `done`). Other failure kinds are exercised in T2.2 | done / T2.2 |
| U4 | ~~Middleware exceptions stop the worker run~~ **Verified in T2.1** for `before_model_call` and `before_tool_call` (ReAct, per-call and configured middleware). Other loop paths (native tool calling, streaming, `run_async`) are untested | done / open |
| U5 | Tool calling through Ollama `/v1` for the chosen tags. Ollama's default `num_ctx` must match the configured `context_length` (set `OLLAMA_CONTEXT_LENGTH` or a Modelfile) | T3.3 |
| U6 | `bitsandbytes` installs and works with the resolved `torch 2.14.1+cu130` on this GPU (torch itself sees CUDA; bitsandbytes is not installed yet) | T3.4 |
| U7 | Orchestrator + worker fit in 8 GB VRAM simultaneously | T3.4 / T3.5 |
| U8 | effGen's `SubprocessSandbox` initializes here, and the default 256m memory limit is enough for `pytest` | T4.2 |
| U9 | ~~`effgen==1.3.0` is installable from PyPI~~ **Verified in T0.1** | done |
| U10 | Kaggle dataset content: repo snapshots, tests and patches are **not** assumed. Third-party pages describe per-issue pass/fail validation tests, but this is unconfirmed | T6.1 |
| U11 | ~~Invalid `delegate` arguments end the parent as `tool_failed`~~ **Answered in T1.3: they do not.** Parent always continues; endless repeats end as `stuck` (`loop_detected` / `max_iterations_exhausted`) | done |

### Technical risks

- The worker timeout is cooperative. A long Transformers `generate()` can overrun by one generation.
- Constructing an agent with `capability_probe=True` may issue a probe model call.
- effGen's base install pulls web-search packages (`ddgs`, `duckduckgo-search`). They are installed but must never be registered as tools.
- `effgen.cli.code.*` (gated or read-only tools) is CLI-internal and unstable. Avoid it in the MVP.

### Known bugs or broken tests

None (no code).

## 7. Quick start

Verified on 2026-10-16 (Linux, uv 0.11.29):

```bash
cd /home/ch_kartik/Projects/Active/smol-harness
uv sync --locked                 # creates .venv with Python 3.12 + effgen 1.3.0 (~2 min cold: torch/CUDA wheels)
uv run python -c "import effgen; print(effgen.__version__)"   # -> 1.3.0
uv run pytest -q                 # -> 77 passed
```

Not yet available: running the harness, integration tests, smoke tests. They'll be added after T1.3 and T5.1.

Reference clones for source reading (outside the repo, read only; don't execute code from them):

```bash
git clone --depth 1 https://github.com/ctrl-gaurav/effGen /tmp/refsrc/effgen/repo
git clone --depth 1 https://github.com/huggingface/smolagents /tmp/refsrc/smolagents/repo
git clone --depth 1 https://github.com/badlogic/pi-mono /tmp/refsrc/pi/repo
```
