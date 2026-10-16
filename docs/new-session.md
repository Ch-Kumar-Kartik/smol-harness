# New Session Handoff

_Last updated: 2026-10-16 (after T0.1 bootstrap)._

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

The remaining planned layout (`src/smol_harness/{config,models,tools,delegation,harness,__main__}.py`, `configs/`, `tests/integration/`) is in `docs/architecture.md` §3. **Not created yet.**

## 3. Current implementation state

| Area | State |
|---|---|
| Implemented and verified | T0.1 bootstrap (env, pinned effGen, pytest runs) |
| Partially implemented | Nothing |
| Not yet implemented | M1–M5 in the plan (no delegation, models, tools, config or CLI) |
| Known limitations (by design) | Sequential workers only; depth 1; cooperative timeout cannot pre-empt an in-flight local generation; FP4 unsupported (effGen hard-codes NF4) |

## 4. Most recent session

- **Goal:** T0.1, the project bootstrap. Nothing else.
- **Done:** created the uv project (Python 3.12, `effgen==1.3.0`) with a package stub and an import test. Full results are in `docs/implementation-plan.md` §4.
- **Files created:** `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `src/smol_harness/__init__.py`, `tests/test_import.py`. Docs updated: `implementation-plan.md`, `new-session.md`, `effgen-api-audit.md` (U9 and install facts).
- **Tests:** `uv run pytest -q` → 2 passed. `uv sync --locked` from a deleted `.venv` reproduces the environment.
- **Problems:** none. The first `uv sync` took about 2 minutes (large torch/CUDA wheels).
- **Unresolved:** U1–U8 and U10 in §6. U9 is resolved.
- **Earlier session (2026-10-16):** architecture and audit only; the decisions are in `docs/architecture.md` §11.

## 5. Current task and next steps

- **Current milestone:** M1 (delegation path, scripted models, CPU only). **Not started; waiting for user approval.**
- **Next task:** T1.1, the scripted-model test fixture (`tests/conftest.py`, `tests/test_scripted_model.py`).
- **Relevant files to inspect:** `docs/implementation-plan.md` T1.1; effGen's `tests/unit/test_written_action_runs.py` (`_Scripted` fake; not shipped in the wheel, so use a scratch clone: `git clone --depth 1 https://github.com/ctrl-gaurav/effGen /tmp/refsrc/effgen/repo`, commit `37de05a`); installed `.venv/lib/python3.12/site-packages/effgen/models/base.py` (`BaseModel`) and `core/agent_config.py`.
- **Acceptance criteria:** as in the plan for T1.1.
- **First executable development milestone:** M1. `tests/test_e2e_delegation.py` passes (T1.3).

## 6. Known issues and warnings

### Unverified assumptions

| ID | Assumption | Where it gets verified |
|---|---|---|
| U1 | A scripted ReAct fake drives effGen 1.3.0's loop deterministically (nudges and closing requests may consume extra turns) | T1.1 |
| U2 | The orchestrator sees the full JSON string from `DelegateTool` | T1.3 |
| U3 | A failed worker reported as a successful tool call does not trigger `tool_failed` on the orchestrator | T2.2 |
| U4 | Middleware exceptions stop the worker run on every loop path | T2.1 |
| U5 | Tool calling through Ollama `/v1` for the chosen tags. Ollama's default `num_ctx` must match the configured `context_length` (set `OLLAMA_CONTEXT_LENGTH` or a Modelfile) | T3.3 |
| U6 | `bitsandbytes` installs and works with the resolved `torch 2.14.1+cu130` on this GPU (torch itself sees CUDA; bitsandbytes is not installed yet) | T3.4 |
| U7 | Orchestrator + worker fit in 8 GB VRAM simultaneously | T3.4 / T3.5 |
| U8 | effGen's `SubprocessSandbox` initializes here, and the default 256m memory limit is enough for `pytest` | T4.2 |
| U9 | ~~`effgen==1.3.0` is installable from PyPI~~ **Verified in T0.1** | done |
| U10 | Kaggle dataset content: repo snapshots, tests and patches are **not** assumed. Third-party pages describe per-issue pass/fail validation tests, but this is unconfirmed | T6.1 |

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
uv run pytest -q                 # -> 2 passed
```

Not yet available: running the harness, integration tests, smoke tests. They'll be added after T1.3 and T5.1.

Reference clones for source reading (outside the repo, read only; don't execute code from them):

```bash
git clone --depth 1 https://github.com/ctrl-gaurav/effGen /tmp/refsrc/effgen/repo
git clone --depth 1 https://github.com/huggingface/smolagents /tmp/refsrc/smolagents/repo
git clone --depth 1 https://github.com/badlogic/pi-mono /tmp/refsrc/pi/repo
```
