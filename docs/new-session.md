# New Session Handoff

_Last updated: 2026-10-16, end of the T4.1 session. Everything below was checked against the working tree on that date._

## 1. Project overview

**smol-harness** is a minimal multi-agent coding harness built on **effGen 1.3.0**.

- A larger-LLM **orchestrator** delegates self-contained tasks to smaller-LLM **workers** through one `delegate` tool, which works like a smolagents managed agent.
- Each worker is a fresh `effgen.Agent` with its own context, model, tool allowlist and limits.
- Models come from Ollama (via effGen's OpenAI-compatible adapter) or local Transformers (bf16/fp16, bnb 8-bit, bnb NF4).
- Tools: effGen's workspace-confined `FileOperations` (**implemented, T4.1**, via `tools.build_tools`) plus a sandboxed shell built on effGen's `CodeExecutor` subprocess sandbox (**not implemented**, T4.2).

Details: `docs/architecture.md`. effGen facts: `docs/effgen-api-audit.md`. Plan: `docs/implementation-plan.md`.

## 2. Repository map (verified against the working tree, 2026-10-16)

Source modules, tests and configuration exist; there is **no CLI and no shell tool yet**.

| Path | Purpose |
|---|---|
| `CLAUDE.md`, `AGENTS.md` | Instructions for Claude Code / any contributor. **Both are git-ignored** (`.gitignore` lines 5-6), so changes to them never show in `git status` or commits |
| `docs/new-session.md` | This handoff |
| `docs/implementation-plan.md` | Milestones, tasks, statuses, completed-work log (§4) |
| `docs/architecture.md` | Design and decisions, with the as-built notes |
| `docs/effgen-api-audit.md` | effGen facts (Part 1, incl. §1.12 on local-model tool calling and context budget), harness integration status (Part 2), per-task history (Part 3) |
| `pyproject.toml`, `uv.lock`, `.python-version` | uv project: Python 3.12, `effgen==1.3.0`, `pyyaml`; dev group: `pytest`; markers `ollama`, `gpu`, `real_model` |
| `src/smol_harness/delegation.py` | `WorkerResult`, `WorkerRecord`, `WorkerRegistry`, `DelegateTool`, `DeadlineMiddleware`, `WorkerTimeout`, `bounded`, `validate_limits`. Worker timeout, `max_delegations`, `max_report_chars`; every report has a unique `delegation_id` |
| `src/smol_harness/harness.py` | `build_harness(...)`, `Harness.run(task) -> HarnessResult` (fresh orchestrator and record list per run). **Takes `AgentConfig`s, not a `HarnessConfig`** |
| `src/smol_harness/config.py` | YAML to frozen dataclasses with validation: `load_config`, `loads_config`, `parse_config`, `ConfigError`, `ModelSpec`, `AgentSpec`, `Limits`, `HarnessConfig`, `hf_placement_problem`, `KNOWN_TOOLS` |
| `src/smol_harness/models.py` | `build_model(ModelSpec)`, `loader_args`, `ModelCache`, `ModelLoadError` (CPU/CUDA placement, quantization mapping, post-load checks) |
| `src/smol_harness/tools.py` | `build_tools(names, workspace)`, `resolve_workspace`, `ToolBuildError`, `TOOL_NAMES`. `file_operations` = effGen `FileOperations(allowed_directories=[workspace])` with the kept root read back; `shell` raises a T4.2 error (fail closed). Not yet called by `build_harness` |
| `src/smol_harness/__init__.py` | Package docstring only |
| `configs/example.yaml` | Mixed-provider example (Ollama orchestrator, HF NF4 worker), finite limits |
| `tests/conftest.py` | `ScriptedModel` fake effGen model, `scripted_model` fixture, and the `--run-real-models` opt-in option / skip hook |
| `tests/test_scripted_model.py` (15), `test_delegation.py` (22), `test_e2e_delegation.py` (23), `test_limits.py` (77) | M1/M2: fake model, delegation layer, end-to-end orchestrator-worker flow, limits and failure propagation |
| `tests/test_config.py` (201), `test_models.py` (110), `test_hf_device.py` (8) | M3: configuration, model factory and cache, CPU/CUDA placement against effGen's real engine with a locally generated tiny model |
| `tests/tool_probe.py`, `test_tool_probe.py` (7), `test_lfm_parser_compat.py` (13) | Tool-calling measurement helper and its scripted validation; model-free pins of how effGen 1.3.0 reads LFM tool-call formats |
| `tests/test_tools.py` (51) | M4: tool factory, workspace validation, `file_operations` confinement through `BaseTool.execute()`, a scripted harness run with real tools, effGen regression pins |
| `tests/test_integration_optin.py` (2), `test_import.py` (2) | Guard that real-model tests cannot run without the flag; effGen version/import check |
| `tests/integration/test_liquid_cpu.py` (1), `test_liquid_tools.py` (2) | **Opt-in** real-model tests for `LiquidAI/LFM2.5-350M` (CPU smoke test; native vs ReAct tool calling and context-window check) |

Not created yet: `src/smol_harness/__main__.py` (T5.1), `tests/integration/test_ollama.py` / `test_hf.py` / `test_mixed.py` (T3.3 to T3.5).

## 3. Current project state

Default suite: **531 passed, 3 skipped** (the 3 opt-in real-model tests); `uv lock --check` passes. Test counts above sum to 534 collected.

**Implemented and verified (with the tests above):**

- **M1, delegation (scripted models):** an orchestrator `Agent` calls a `delegate` tool; each delegation builds a *fresh* worker `Agent` that shares the configured model instance but not its conversation; the parent sees only a JSON report; records are per run; depth is 1.
- **M2, controls:** cooperative per-worker deadline (checked before model/tool calls, never interrupts one in flight), structured failure reports for every worker failure kind (the parent is never ended as `tool_failed` by them), `max_delegations` (refusals still get unique ids), bounded reports (`max_report_chars`, valid JSON, `truncated` flag), effGen's own `max_iterations`.
- **M3 so far:** validated YAML configuration (T3.1); model factory and cache with explicit quantization and placement (T3.2); explicit `device: cuda | cpu` for HF models (T3.2a); `LiquidAI/LFM2.5-350M` loaded and run on CPU (fp32) through `build_model`, and tool-calling compatibility measured (T3.2b, T3.2c).
- **M4 so far (T4.1):** `build_tools(names, workspace)` builds exactly the allowlisted tools (fresh instances; `[]` = none; unknown, web-like, duplicate and `delegate` names refused; `shell` fails closed). `file_operations` is confined to the validated workspace; `../`, absolute, symlink (file, directory, dangling) escapes are refused for reads and writes. This is **path validation, not a sandbox** (limitations in plan §4 T4.1).

**Liquid AI results (real model, CPU; details in plan §4 and audit §1.12 and Part 3):**

- Loads in about 3 to 5 s, 354M parameters, all on CPU, no CUDA memory or context; a one-word answer takes about 0.2 to 0.3 s. Measured on an actively cooled laptop, **not** the fanless edge device.
- **ReAct tool calling worked 2/2** (the tool really executed and its result returned); **native tool calling worked 0/2** (the model writes its default Pythonic `<|tool_call_start|>[add(a=.., b=..)]<|tool_call_end|>`, which effGen's native parser cannot read, so the tool never ran). **Recommended mode for this model: `tool_calling_mode="react"`.**
- Context: effGen reports 128,000 (from `config.json`); the model card says 32,768. 32,768 is used only as a **conservative planning limit taken from the model card, not a tested maximum**. With `max_context_length=32768` effGen's prompt budget is 20,889 tokens (101,836 without it).

**Not implemented / remaining MVP work:** T4.2 sandboxed `shell` (T4.3 only if needed), CLI (T5.1), toy-repo end-to-end test (T5.2), wiring `HarnessConfig` + `ModelCache` + tools into `build_harness` (unplanned, see §5), `duration_s` in reports, Ollama smoke test (T3.3, blocked), real bnb-quantized GPU load (T3.4), mixed-provider real delegation (T3.5), a config field for `max_context_length`.

**Explicitly not done (do not assume):** no real multi-agent run with any real model (all delegation tests use scripted models); no shell tool exists; nothing builds tools from a `HarnessConfig` inside the harness yet (tests call `build_tools` directly); no Ollama server has been contacted; no bitsandbytes load has been run; the Kaggle dataset is unverified; native tool calling for LFM is unsupported.

**Known limitations (by design):** sequential workers only; depth 1; the worker timeout is cooperative; no orchestrator deadline; FP4 unsupported (effGen hard-codes NF4); web search excluded.

## 4. Most recent session

- **Goal:** T4.1 only: a tool factory with workspace-confined file operations. No shell (T4.2), no harness wiring, no models.
- **Result:** **Completed and verified.** `src/smol_harness/tools.py` builds effGen's `FileOperations(allowed_directories=[workspace])` per allowlist; escapes via `../`, absolute paths and symlinks (file, directory, dangling) are refused for reads and writes; allowlists, unknown/web names, `delegate` and `shell` (fail closed) behave as specified. Full record: `docs/implementation-plan.md` §4 (T4.1).
- **Files:** created `src/smol_harness/tools.py`, `tests/test_tools.py`. Docs: plan, this file, audit (§1.9, Part 2, Part 3), architecture §9 (one row clarified, no design change). No other source changed. (The T3.2c test files listed in §2 were still uncommitted at the start of this session; nothing was committed in this session.)
- **Tests:** `uv run pytest tests/test_tools.py -q` → 51 passed; `uv run pytest -q` → 531 passed, 3 skipped; `uv lock --check` → OK (157 packages, no dependency change). No real-model test was run.
- **Key fact found:** effGen's `FileOperations` treats a missing allowed directory as a warning and **falls back to `EFFGEN_WORKSPACE` (creating it) or the current directory**. `build_tools` therefore validates the workspace (exists, directory, read/write/search access) and reads back the private `_allowed_directories`, refusing the tool unless it is exactly `[workspace]`. Pinned by regression tests.
- **Other effGen behaviour pinned:** extensionless files (`Makefile`, `README`, `.gitignore`, `Dockerfile`) are refused for read and write; credential filenames and dotenv-shaped content are refused on read; relative paths resolve against the workspace, not the cwd; `list`/`search` (filename glob) work for repository discovery and drop entries outside the workspace.
- **Problems:** none open.
- **Decisions:** reuse `FileOperations` unwrapped (its own checks held for every escape tested); names validated before the workspace, all-or-nothing; the workspace is validated even for `[]`; `delegate` is never built by the factory.
- **Known limitations:** path validation, not a sandbox: check-then-use race between effGen's resolve and open; hard links / bind mounts not detectable; no write-size cap; the workspace root itself is not restricted. The extension allowlist will block a worker from editing a `Makefile` (watch for it in T5.2).

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

### Next task: **T4.2, sandboxed `shell` tool** (status: Not started; awaiting approval)

- **Objective:** replace `_shell` in `src/smol_harness/tools.py` (it currently raises `ToolBuildError` naming T4.2) with a thin `CodeExecutor` subclass that exposes only `command` (and a capped `timeout`), pins `language="bash"`, `network_enabled=False` and a configured memory limit, requires the subprocess (`unshare`) backend, and fails closed if the sandbox does not initialize. Never `BashTool`.
- **Dependencies:** T4.1 (done): reuse `build_tools` / `resolve_workspace`; update `test_shell_fails_closed_until_t42` in `tests/test_tools.py`.
- **Expected files:** `src/smol_harness/tools.py`, `tests/test_tools.py` (Linux-only marker for sandbox tests). No effGen change, no `build_harness` wiring.
- **Read only these:** plan §2 T4.2; audit §1.9 (`code_executor` row: model-facing `network_enabled` / `memory_limit` must be pinned, Docker does not mount the workspace, force `EFFGEN_SANDBOX_BACKEND=subprocess`, 256m default `ulimit -v`); architecture §9; installed effGen `effgen/tools/builtin/code_executor.py` and `effgen/security/sandbox.py`.
- **Acceptance criteria:** see plan §2 T4.2 (`echo` works; writes outside the workspace fail; network fails; an over-long sleep is killed; `filesystem_confined=True` reported; `pytest` in the workspace runs under the memory limit). Note the shell's reads are **not** confined (audit §1.9).
- **Verify with:** `uv run pytest tests/test_tools.py -q`, then `uv run pytest -q` (baseline 531 passed, 3 skipped).
- **Process:** mark T4.2 `In progress` in the plan before starting; update plan, audit (Part 2/3) and this file when done.
- **Do not repeat:** the effGen architecture audit, the Liquid AI experiments, or the T4.1 `FileOperations` checks (plan §4 T4.1, audit Part 3).

### Afterwards / open

- After T4.2 (sandboxed `shell`; the Linux user/mount/net namespace check passed on this host, effGen's own probe has not been run), then the unplanned wiring step (`HarnessConfig` to `build_harness`, including tools, `ModelCache` and `max_context_length`), T5.1, T5.2.
- **Blocked:** T3.3 (no Ollama), T3.4 (`bitsandbytes` not installed), T3.5 (needs both).
- **Decisions for the user (nothing started):** approve or reject a parser adapter for LFM's Pythonic call format or an engine change to send a system-role instruction (native mode); run the opt-in tests on the fanless edge device.

## 6. Known issues and warnings

### Known bugs or broken tests

None known. The default suite is green (531 passed, 3 skipped).

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
| U8 | effGen's `SubprocessSandbox` initializes here and its default 256m memory limit is enough for `pytest` | T4.2 |
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
uv run pytest -q                 # default suite -> 531 passed, 3 skipped (the real-model tests)
uv run pytest tests/test_limits.py -q                       # a single file works the same way
uv run pytest tests/test_tools.py -q                        # T4.1 tool factory + file_operations -> 51 passed
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
