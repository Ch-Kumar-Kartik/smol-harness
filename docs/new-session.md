# New Session Handoff

_Last updated: 2026-10-16 (after T3.1: YAML configuration)._

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
| `src/smol_harness/delegation.py` | `WorkerResult`, `WorkerRecord`, `WorkerRegistry`, `DelegateTool`, `DeadlineMiddleware`, `WorkerTimeout`, `bounded`, `validate_limits`. Limits: worker timeout, `max_delegations`, `max_report_chars`; every result carries a unique `delegation_id`. |
| `src/smol_harness/harness.py` | `build_harness(..., worker_timeout_s, max_delegations, max_report_chars)`, `Harness.run(task) -> HarnessResult` (fresh orchestrator + record list per run) |
| `tests/test_limits.py` | 77 tests: worker deadline (T2.1), failure propagation (T2.2), delegation limit, report truncation, iteration caps (T2.3) |
| `tests/test_e2e_delegation.py` | 23 end-to-end tests: delegation flow, isolation, records, U11, orchestrator-memory regression |
| `tests/test_delegation.py` | 22 tests of the delegation layer through `BaseTool.execute()` |
| `src/smol_harness/config.py` | YAML → frozen dataclasses with validation (T3.1): `load_config`, `loads_config`, `parse_config`, `ConfigError`, `ModelSpec`, `AgentSpec`, `Limits`, `HarnessConfig` |
| `configs/example.yaml` | Mixed-provider example (Ollama orchestrator, HF NF4 worker) with finite limits |
| `tests/test_config.py` | 179 configuration tests |
| `tests/conftest.py` | `ScriptedModel` deterministic fake effGen `BaseModel` (+ `scripted_model` fixture) |
| `tests/test_scripted_model.py` | 15 tests of the fake, driven through the real effGen `Agent` |

The remaining planned layout (`src/smol_harness/{config,models,tools,delegation,harness,__main__}.py`, `configs/`, `tests/integration/`) is in `docs/architecture.md` §3. **Not created yet.**

## 3. Current implementation state

| Area | State |
|---|---|
| Implemented and verified | T0.1 bootstrap; T1.1 `ScriptedModel`; T1.2 delegation layer; T1.3 end-to-end orchestrator→worker→orchestrator. **M1 complete**; T2.1 cooperative worker deadline; T2.2 failure propagation; T2.3 delegation limit + report truncation. **M2 complete**; T3.1 YAML configuration (318 tests pass) |
| Partially implemented | Nothing |
| Not yet implemented | `duration_s`, model building (T3.2), real models (Ollama/HF, T3.3–T3.5), workspace tools (M4), CLI (T5.1); `HarnessConfig` is not yet wired into `build_harness` |
| Known limitations (by design) | Sequential workers only; depth 1; **the worker timeout is cooperative** (checked before model/tool calls; an in-flight generation or tool is never interrupted); no orchestrator deadline; FP4 unsupported (effGen hard-codes NF4) |

## 4. Most recent session

- **Goal:** T3.1 only: YAML configuration and validation.
- **Done:** `config.py`, `configs/example.yaml`, 179 tests; PyYAML declared as a runtime dependency (was dev-only though already installed through effGen). Details: `docs/implementation-plan.md` §4.
- **Files created:** `src/smol_harness/config.py`, `tests/test_config.py`, `configs/example.yaml`. Modified: `pyproject.toml`, `uv.lock`. Docs: plan, this file, audit, `architecture.md` §4 and §8.
- **Tests:** `uv run pytest tests/test_config.py -q` → 179 passed; `uv run pytest -q` → 318 passed (CPU, ~8 s). Ten mutations fail tests (one survived first, test tightened).
- **Problems:** one wrong test fragment on first run (message wording), fixed in the test.
- **Unresolved:** U5–U8, U10; native tool calling, streaming and `run_async` untested.

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
| `hf` | `model` (repo id/path), `quantization` (**required**), `dtype` (`bf16`/`fp16`, optional) | `none`, `bnb-8bit`, `bnb-4bit-nf4` |

## 5. Current task and next steps

- **Current milestone:** M3. T3.1 done. **Waiting for user approval.**
- **Recommended next task:** T3.2 `models.py`: `build_model(spec: ModelSpec) -> BaseModel` mapping to `effgen.models.load_model` (Ollama: `provider="openai_compatible"`, `base_url`, `context_length`; HF: `engine="transformers"`, `quantization_bits` **always passed** (None/8/4), `torch_dtype`, `device_map="auto"`) plus a cache keyed by the frozen/hashable `ModelSpec` so identical specs share one instance. Test by monkeypatching `load_model` (no GPU/Ollama).
- **Alternative:** T4.1 workspace tools (depends only on T3.1).
- **Still blocked:** T3.3 (Ollama not installed). T3.4 needs `bitsandbytes` (not installed).
- **Relevant files:** `docs/implementation-plan.md` T3.2; `docs/architecture.md` §8; audit §1.5–§1.7; `src/smol_harness/config.py`.
- **Cautions:** HF `dtype: None` means effGen's automatic choice, so pass `torch_dtype=None` rather than inventing one; the Ollama `context_length` goes to the adapter as `context_length=`; do not pass `quantization` strings through to effGen (use `quantization_bits`).

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
uv run pytest -q                 # -> 318 passed
```

Not yet available: running the harness, integration tests, smoke tests. They'll be added after T1.3 and T5.1.

Reference clones for source reading (outside the repo, read only; don't execute code from them):

```bash
git clone --depth 1 https://github.com/ctrl-gaurav/effGen /tmp/refsrc/effgen/repo
git clone --depth 1 https://github.com/huggingface/smolagents /tmp/refsrc/smolagents/repo
git clone --depth 1 https://github.com/badlogic/pi-mono /tmp/refsrc/pi/repo
```
