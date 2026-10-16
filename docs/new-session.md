# New Session Handoff

_Last updated: 2026-10-16 (after T3.2b: Liquid AI CPU smoke test)._

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
| `src/smol_harness/models.py` | `build_model(ModelSpec)`, `loader_args`, `ModelCache`, `ModelLoadError` (T3.2) |
| `tests/test_models.py` | 110 tests: exact loader kwargs, cache, failures, device selection, wiring against effGen's real routing |
| `tests/integration/test_liquid_cpu.py` | **Opt-in** real-model CPU smoke test (`LiquidAI/LFM2.5-350M`, marker `real_model`) |
| `tests/test_integration_optin.py` | 2 tests proving real-model tests are skipped without `--run-real-models` |
| `tests/test_hf_device.py` | 8 tests: CPU/CUDA placement against effGen's real engine with a locally generated tiny model (no download) |
| `tests/test_config.py` | 201 configuration tests |
| `tests/conftest.py` | `ScriptedModel` deterministic fake effGen `BaseModel` (+ `scripted_model` fixture) |
| `tests/test_scripted_model.py` | 15 tests of the fake, driven through the real effGen `Agent` |

The remaining planned layout (`src/smol_harness/{config,models,tools,delegation,harness,__main__}.py`, `configs/`, `tests/integration/`) is in `docs/architecture.md` §3. **Not created yet.**

## 3. Current implementation state

| Area | State |
|---|---|
| Implemented and verified | T0.1 bootstrap; T1.1 `ScriptedModel`; T1.2 delegation layer; T1.3 end-to-end orchestrator→worker→orchestrator. **M1 complete**; T2.1 cooperative worker deadline; T2.2 failure propagation; T2.3 delegation limit + report truncation. **M2 complete**; T3.1 YAML configuration; T3.2 model factory and cache; T3.2a explicit hf device selection; T3.2b Liquid AI CPU smoke test passed with a real checkpoint (460 pass, 1 opt-in test skipped by default) |
| Partially implemented | Nothing |
| Not yet implemented | `duration_s`, real-model smoke tests (T3.3–T3.5), workspace tools (M4), CLI (T5.1); `HarnessConfig` and `ModelCache` are not yet wired into `build_harness` |
| Known limitations (by design) | Sequential workers only; depth 1; **the worker timeout is cooperative** (checked before model/tool calls; an in-flight generation or tool is never interrupted); no orchestrator deadline; FP4 unsupported (effGen hard-codes NF4) |

## 4. Most recent session

- **Goal:** T3.2b only: one real-model, CPU-only smoke test with `LiquidAI/LFM2.5-350M` through `build_model` and effGen.
- **Result: loaded and ran.** All 354M parameters fp32 on CPU, no CUDA memory or context, `model.generate` and a minimal `Agent.run` both answered `Paris` (0.31 s and 0.21 s). Full results table: `docs/implementation-plan.md` §4 (T3.2b).
- **Files:** created `tests/integration/test_liquid_cpu.py`, `tests/test_integration_optin.py`; modified `tests/conftest.py` (opt-in option and skip hook), `pyproject.toml` (marker). No source module changed. Docs: plan, this file, audit.
- **Tests:** default `uv run pytest -q` → 460 passed, 1 skipped; opt-in run → 1 passed; `uv lock --check` OK. 714 MB of weights are now in `~/.cache/huggingface` (`/home` has 7.8 GB free).
- **Key facts:** architecture `lfm2` is native in transformers 5.20.0 (no remote code); effGen reports `supports_tool_calling() == True` and a 128,000-token context (card says 32,768); peak RSS 2.5 GiB because of the fp32 upcast.
- **Problems:** none; my first metadata fetch failed only because `curl` did not follow a redirect.
- **Unresolved:** U5–U8, U10; native tool calling with this model, reliability on longer prompts, edge-device timing/thermals/RAM, bf16 on CPU.

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

- **Current milestone:** M3 partly done (T3.1, T3.2, T3.2a, T3.2b). **Waiting for user approval.**
- **Done since the last handoff:** the CPU smoke test with a real checkpoint (T3.2b). A natural follow-up, if directed: run the same test on the actual fanless edge device (record RAM, timing and temperatures there), or a short effGen tool-calling check with this model (native-vs-ReAct, see audit). Neither is started.
- **Other candidates:** T4.1 workspace tools (depends only on T3.1); the unplanned wiring step `HarnessConfig` → `ModelCache` → `build_harness` (fits with T5.1).
- **Blocked:** T3.3 (Ollama not installed); real GPU load of a bnb-quantized model needs `bitsandbytes` (not installed).
- **Relevant files:** `src/smol_harness/models.py`, `src/smol_harness/config.py` (`hf_placement_problem`), `tests/test_hf_device.py` (a pattern for offline real-engine tests).
- **Cautions:** CPU mode has only been exercised on a tiny random model; expect slow generation and measure memory on the edge device. `device: cuda` fails fast without CUDA by design.

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
uv run pytest -q                 # -> 460 passed, 1 skipped (real-model test)
uv run pytest tests/integration/test_liquid_cpu.py --run-real-models -s   # opt-in; first run downloads ~714 MB, writes /tmp/smol-harness-lfm-cpu-smoke.json
```

Not yet available: running the harness, integration tests, smoke tests. They'll be added after T1.3 and T5.1.

Reference clones for source reading (outside the repo, read only; don't execute code from them):

```bash
git clone --depth 1 https://github.com/ctrl-gaurav/effGen /tmp/refsrc/effgen/repo
git clone --depth 1 https://github.com/huggingface/smolagents /tmp/refsrc/smolagents/repo
git clone --depth 1 https://github.com/badlogic/pi-mono /tmp/refsrc/pi/repo
```
