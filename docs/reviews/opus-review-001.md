# Independent review 001: smol-harness on effGen 1.3.0

- **Date:** 2026-10-17
- **Reviewer:** Claude Opus 5.5 (independent of the implementing sessions)
- **Scope:** `src/smol_harness/*` (1,379 lines), `tests/*`, the four project docs, and the effGen 1.3.0 sources they depend on (installed copy in `.venv`).
- **Working tree reviewed:** commit `6ff4666` plus the uncommitted T4.2 changes (`src/smol_harness/tools.py`, `tests/test_tools.py`, docs).
- **Method:** source reading, plus small scripted diagnostics run in throwaway directories (scripts in `/tmp/rev-scripts/`, outside the repo). There was no real model, no network and no sandbox probing beyond what the existing tests do. I modified no source or project document.

Evidence labels used below: **Verified defect** (reproduced by a diagnostic run in this review), **Source-supported risk** (follows from code I read, not reproduced), **Untested hypothesis**, **Optional improvement**.

---

## A. Executive assessment

**Is the architecture viable?** Yes.
- The agent-as-tool design is the right integration with effGen 1.3.0. One `DelegateTool` builds a fresh `effgen.Agent` per delegation around a shared model instance, and the parent receives only a JSON report.
- effGen's built-in sub-agent machinery is unsuitable here (audit §1.3), and the design matches the smolagents managed-agent reference without re-implementing a loop.
- Responsibilities are cleanly split: `config` validates, `models` loads and owns models, `tools` builds tools, `delegation` delegates, `harness` wires.
- I found no duplicated effGen functionality: no custom reasoning loop, provider client, quantization code or tool base class.

**Is the code sufficiently minimal?** Mostly yes.
- There are 1,379 lines of source against the architecture's "≤ ~600" target. The excess is almost entirely validation, fail-closed checks and comments that cite why each workaround exists, not abstraction layers.
- I would not refactor. The one candidate for simplification (the binary-search report shrinking in `delegation.bounded`) is correct, well tested, and costs ~60 lines; leave it.

**Ready for configuration wiring and CLI work?** Not yet. The wiring itself is well specified (T5.0), but this review found that **effGen's loop guards assume every tool is idempotent**. In a coding harness with side-effecting tools (`shell`, `file_operations` writes, `delegate`), this produces **stale results and premature run termination**. The existing tests cannot show it: none of them repeats a side-effecting call. This undermines the core edit → test → edit loop that T5.2 and the dataset experiments rely on, so it should be resolved, or at least decided on, before more is built on top.

**Top three risks**
1. **Stale or suppressed tool results** (F1, F2): effGen replays a recorded result instead of re-running an identical call, and ends a run when two different calls return the same text. Verified for `shell`, `file_operations` and `delegate`.
2. **Orchestrator failures escape `Harness.run` as exceptions** (F3) under effGen's default `raise_on_error=True`, discarding that run's worker records. That breaks the planned CLI contract (structured output plus a non-zero exit code).
3. **The experimental platform is unverified end to end** (F4, F9): Ollama's server-side context window is not controlled by the harness (silent prompt truncation is possible); the shell is Linux-only, so the Windows 11 / RTX 4070 machine needs the whole harness inside WSL2, which has never been exercised; and there has been **no real multi-agent run with any real model** (every delegation test is scripted).

---

## B. Prioritized findings

Findings by severity: **P0: 0, P1: 3, P2: 6, P3: 3** (12 total).

| ID | Severity | Type | Component | Issue | Evidence | Recommended action |
|---|---|---|---|---|---|---|
| F1 | **P1** | Verified defect (effGen behaviour; no opt-out) | effGen loop guard; affects `tools.ShellTool`, `file_operations`, `delegation.DelegateTool` | **An exact repeat of an earlier successful call is not executed; effGen replays the recorded result.** The key is `(tool name, normalized input)`. After a code change, rerunning the same test command shows the **old** output. A rerun of a previously *failing* identical call is declined and the tool withdrawn. An identical re-delegation never runs the worker. | effGen `core/agent_loop.py:3001-3030` ("An exact repeat of a call that already succeeded is answered from the record… A pure computation is idempotent"); `core/agent_tool_loop.py:529-575` (`check_action`, `MAX_REPLAYS_PER_PAIR=2`, `cached_result`). No config flag or tool attribute disables it (searched `core/`, `tools/` for idempotent/replay/cache flags; `max_turns_without_progress=None` turns off only the stall logic, `agent_config.py:372-384`). **Diagnostics:** (E2) `check` passes → file broken via `shell` → identical `check` again: **2 of 3 calls executed, the model saw the stale `PASSED`**. (E1) failing check → fix → identical check: the rerun was **not executed**, the run moved to a closing request. (D) `cat f.py` → rewrite → `cat f.py`: the model was shown the old `x = 1`. (Re-delegation) orchestrator calls `delegate` twice with identical args: **1 worker run, 1 record**; the second observation was "(the same result as the identical call above)". | Decide before T5.0 (needs user approval per CLAUDE.md, since every real fix touches effGen behaviour): (a) **preferred:** a narrowly scoped effGen change (local patch or upstream fix) that skips replay for non-idempotent tools, e.g. by `ToolCategory` (`CODE_EXECUTION`, `FILE_OPERATIONS`, `COMMUNICATION`) or a tool attribute; (b) a harness-level monkeypatch of `cached_result`/`check_action` for those tool names (private API, fragile); (c) accept and document, which effectively rules out T5.2's fix-and-retest loop. In every case, first add the scripted regression tests from §D so the behaviour is pinned. |
| F2 | **P1** | Verified defect (effGen behaviour interacting with our tool outputs) | effGen result-repeat guard; `tools.ShellTool._execute`, effGen `FileOperations` | **Two *different* successful calls that return the same text end the run** (`termination="stuck"`, `stop_reason="repeated_tool_result"`). The key is `(tool, first 500 chars of whitespace-normalized result)`; arguments are not part of it. This hits two common cases: silent shell commands (`mkdir`, `sed -i`, `git add` all render `exit code: 0`), and rewriting the same file (effGen's `write` returns only the path). | effGen `core/agent_tool_loop.py:644-664`, `core/agent_loop.py:3224-3260` (first repeat: tools withdrawn, answer requested; then stop). `ShellTool` success text has no command or call id (`tools.py` `_report`). **Diagnostics:** (A) `true`, `mkdir -p d`, `touch marker.txt` → **2 executed, marker not created**, `stuck/repeated_tool_result`. (B) `write f.py`, `write f.py` (new content), `write g.py` → **2 executed, `g.py` never written**. | Smallest fix, in our code: make each side-effecting tool's result text unique per call, as `DelegateTool` already does with `delegation_id`. For `ShellTool`, echo the command and a per-instance call number in the report. For `file_operations`, a thin wrapper or subclass that appends the operation and a call number to the result (a new small subclass in `tools.py`). This does **not** fix F1 (replay happens before dispatch), so do both together. |
| F3 | **P1** | Verified defect | `harness.Harness.run` (`harness.py:59-73`) | `Harness.run` copies the orchestrator config but leaves effGen's default `raise_on_error=True`. When the orchestrator does not finish (iteration cap, stuck, …), effGen raises `RunStoppedError`: `Harness.run` raises instead of returning a `HarnessResult`, and **the worker records of that run are lost** (they live only in a local list). Every orchestrator test sets `raise_on_error=False`, so this path is untested. `DelegateTool` already handles both modes for workers; the orchestrator does not. | Diagnostic: orchestrator left at the default, it delegates once then keeps delegating until `max_iterations=2` → `Harness.run RAISED RunStoppedError …`; no records reachable (`Harness` keeps none). effGen default: `core/agent_config.py:281` (`raise_on_error: bool = True`). Test grep: `raise_on_error=True` appears only for workers (`tests/test_limits.py`, `tests/test_delegation.py`). | In `Harness.run`, catch `RunStoppedError` and build the `HarnessResult` from `exc.response` (the same unwrapping `DelegateTool._delegate` does), or force `raise_on_error=False` in the `replace(...)` call. Add a test with an orchestrator at the default setting that fails after a successful delegation and still returns `success=False` plus that run's records. Do it before T5.1, whose acceptance needs "non-zero exit code on orchestrator failure" with output. |
| F4 | P2 | Source-supported risk (documented as U5, under-weighted) | `models.loader_args` (Ollama branch, `models.py:85-103`) | `context_length` only tells effGen how big the window is (budget, compaction). Ollama's **server** context (`num_ctx`) is set by the server or Modelfile, and the OpenAI-compatible `/v1` endpoint used here has no per-request `num_ctx`. If the server window is smaller than `context_length`, Ollama truncates the prompt silently (typically the oldest part, i.e. instructions and tool descriptions), and effGen never knows. | `loader_args` passes `context_length` to `load_model(..., provider="openai_compatible")` only; nothing sets or reads the server context. U5 in `docs/new-session.md` §6 lists it as unverified. | In T3.3: require the Ollama model to be created with `PARAMETER num_ctx <context_length>` (or `OLLAMA_CONTEXT_LENGTH`) and check it before a run (e.g. `/api/show` model parameters; `/api/ps` reports the loaded context in recent versions). Fail closed if they differ. Add an opt-in test that sends a prompt near the configured window and checks a marker from its first line survives. |
| F5 | P2 | Integration gap (known, planned as T5.0) | `harness.build_harness`, `Harness` (no `close`) | Nothing turns a `HarnessConfig` into a running harness: `build_harness` takes ready `AgentConfig`s; `Harness` has no `close()`. The T5.0 plan in `docs/implementation-plan.md` is sound but should absorb F3 and F8, and must set every effGen default that matters **explicitly**: `capability_probe=False` (default `True`, `agent_config.py:353`, an extra probe generation on real models), `tool_calling_mode` explicit (default `"auto"`), `raise_on_error`, `enable_memory`. | `harness.py:76-93`; `models.ModelCache` (`models.py:184-227`) is the only owner of models today; T5.0 entry in the plan. | Keep T5.0 as planned. Add: F3's fix (or a forced `raise_on_error=False`), F8's cap, the explicit flags above, and a test that `Harness.close()` is reached when `run()` raises. |
| F6 | P2 | Source-supported risk | `models.build_model` (`models.py:176-181`) | If `_check_result` rejects a model that **did** load (wrong device, wrong quantization, wrong engine), the instance is dropped without `unload()`. On CUDA its weights stay allocated until garbage collection, so a second attempt in the same process can fail with OOM on a 12 GB card. | `model = load_model(...)` then `_check_result(spec, model)` raises; no `try/finally` unload. `ModelCache.get` never sees the instance (`models.py:197-203`). | Wrap `_check_result` so a rejected model is `unload()`ed (best-effort) before re-raising. One unit test with a fake `load_model` whose model fails the check and records `unload`. |
| F7 | P2 | Missing requirement for experiments | `config.AgentSpec` / YAML schema | There are no per-agent generation controls (`temperature`, `seed`, `max_tokens`) and no `tool_calling_mode`. effGen defaults apply silently, so experiment runs are not reproducible or recorded. T3.2c's reproducibility relied on explicit `seed` and `temperature`. | `config.py:70-77` (`AgentSpec` fields); T3.2c record (seed, temperature 0.1 and `tool_calling_mode="react"` were set by the test, not by config). | Before experiments, not before T5.0: add the minimum fields (`tool_calling_mode`, `temperature`, `seed`, `max_tokens`) or fixed harness defaults, and echo the effective values in the run's structured output. |
| F8 | P2 | Source-supported risk | `delegation.DeadlineMiddleware` (`delegation.py:53-85`); `tools.ShellTool` (`tools.py:115`) | The worker deadline is checked only between calls. A single `shell` call may last up to `max_timeout_s=300` s whatever time the worker has left, so a worker with `worker_timeout_s: 60` can run for about 6 minutes. The cooperative nature is documented, but the size of the overrun from the shell is not. | `ShellTool.MAX_TIMEOUT_S = 300`; `build_tools` passes no cap; `DeadlineMiddleware` checks only `before_model_call` and `before_tool_call`. | In T5.0, build each agent's shell with `max_timeout_s = min(300, ceil(limits.worker_timeout_s))`. Mention the bound in the delegate tool's behaviour notes. |
| F9 | P2 | Untested hypothesis (platform) | whole harness on the Windows 11 / RTX 4070 machine | `ShellTool` refuses to build off Linux by design. Native Windows workers can therefore have only `file_operations`, and the T5.2 / dataset loop needs a shell. The working route is the whole harness inside WSL2 (HF on CUDA-in-WSL, Ollama inside WSL or reached from it). Nothing of that has run: WSL2 user namespaces, bitsandbytes under WSL2, Ollama reachability and the co-residency of Ollama and HF models in 12 GB of VRAM are all unverified (U7, U15). | `tools.require_shell_host`; plan §4 T4.2 "Windows 11"; no Windows or WSL test exists. | Treat "harness inside WSL2" as the experiment platform and validate it in one session: the default suite (sandbox tests included), one CUDA HF load with NF4, one Ollama call, and `nvidia-smi` with both models loaded. |
| F10 | P3 | Test quality | `tests/test_tools.py` | (a) `test_an_agent_without_shell_cannot_reach_it_and_one_with_shell_can` ends with `any("not found" … or "available" in str(c.prompt).lower() …)` (`test_tools.py:914`). "available" is almost certainly present in any prompt that lists tools, so that clause cannot fail; the real check is the absent `smuggled.txt`. (b) The effGen timeout pin sleeps a fixed 3.5 s (`test_tools.py:739`) and asserts on wall-clock times, which is fragile on a loaded machine. (c) Two tests assume `/etc/hostname` exists (`test_tools.py:127`, `:572`); some distros and containers lack it. | Lines cited. | (a) Assert on the specific observation effGen produces for an unknown tool, or drop the clause. (b) Widen the margins or poll for survivors instead of sleeping. (c) Use a file the test creates outside the workspace. |
| F11 | P3 | Optional improvement / documentation drift | `docs/architecture.md` §3 | The "≤ ~600 lines" custom-code target is out of date (1,379 lines). The extra size is justified (validation, fail-closed checks), but the target now misleads reviewers. | `wc -l src/smol_harness/*.py`. | Update the target, or note why it was exceeded. No code change. |
| F12 | P3 | Process risk | repository state | T4.2 (security-relevant code: the `/run` mask, the in-namespace timeout, the fail-closed pre-flight) and the R1 decision exist only as **uncommitted** changes. | `git status`: `src/smol_harness/tools.py`, `tests/test_tools.py`, four docs modified since `6ff4666`. | Commit T4.2 before starting T5.0, so the reviewed security behaviour has a fixed reference. |

### Claims I checked and found to hold

These are not findings; they confirm the docs where it matters.

- **Context isolation.** Each delegation builds a new `Agent` from a shallow copy of the worker config (`delegation.py:265-272`). Each run builds a new orchestrator `Agent` (`harness.py:64-66`). effGen's Transformers engine keeps no prompt or KV cache across calls (no such state in `effgen/models/transformers*`), so sharing a model instance does not leak context. Long-term memory needs an explicit `long_term_persist_path` (default `None`, `agent_config.py:291-297`; `core/agent.py:557-571`), so nothing is persisted across agents by default.
- **Closing a worker `Agent` does not unload the shared model** (`core/agent.py:1093-1116` closes memory and runners only).
- **Report bounding** (`delegation.bounded`) keeps valid JSON within `max_report_chars` and never drops `delegation_id`, `status` or the counters; it is extensively tested.
- **`delegation_id`** does prevent F2 for `delegate` (the results differ), but **not** F1 (replay is keyed on input).
- **The model factory** passes `quantization_bits` explicitly for HF, refuses Ollama ids effGen would rewrite, enforces CPU or CUDA placement and checks the result.
- **The tool factory** builds exactly the allowlisted tools; `config.KNOWN_TOOLS` and `tools.TOOL_NAMES` are kept equal by a test; no web tool can be constructed.
- **Shell security behaviour** matches the T4.2 record within its stated scope: writes confined, no IP network, DNS blocked via the hidden `/run`, timeout kills descendants, fail-closed pre-flight. Reads unconfined and R1 (sockets outside `/run`) remain, as documented and accepted for trusted development only. I found no additional critical security issue, and did not probe further.
- **Import cost.** `config.py` imports `delegation` (and so effGen) for two constants. Measured: 0.45 s and no torch import, so it is not worth changing.

---

## C. Requirements coverage

| Original MVP requirement | Status | Verification evidence (actual) | Remaining work |
|---|---|---|---|
| Larger orchestrator delegates to smaller workers | Implemented (`DelegateTool`, `Harness`) | Scripted end-to-end tests (`tests/test_e2e_delegation.py`) | **No real-model delegation run at all.** F1, F3. |
| All agent execution via effGen | Implemented | Source: every agent is `effgen.Agent`; no custom loop | — |
| Workers execute independently, isolated contexts | Implemented | Scripted tests plus source check (above) | — |
| Mixed Ollama + HF backends | Config and cache support it | Offline adapter construction; HF load on CPU (LFM2.5-350M) and a tiny model on CUDA | Ollama never contacted (T3.3 blocked); no mixed run (T3.5); F4 |
| Per-agent quantization | Validated and mapped (`quantization_bits` always explicit) | Mapping unit tests; real CPU fp32 load | No bitsandbytes load ever ran (T3.4; not installed); 12 GB co-residency unmeasured (F9) |
| Tool calling | Works through effGen ReAct | LFM2.5-350M ReAct 2/2 on one trivial single-tool task; native 0/2 | Multi-step use **broken by F1/F2** for side-effecting tools; small-model reliability unmeasured (U12) |
| Repository interaction (file ops, shell) | Implemented (`file_operations`, `ShellTool`) | 121 tests in `tests/test_tools.py` with the real sandbox on Linux | F1/F2 in multi-step loops; Windows/WSL2 (F9); extensionless files refused by `file_operations` |
| Execution limits | Iteration caps (effGen), cooperative worker deadline, `max_delegations`, report bound | Scripted tests (`tests/test_limits.py`) | Shell overrun of the deadline (F8); no orchestrator wall-clock limit (documented) |
| Failure handling | Worker failures become structured reports in both `raise_on_error` modes | Scripted tests | **Orchestrator failure path (F3)** |
| No web search | Enforced in config, factory and tests | Tests assert no web-like tool; shell has no IP network, DNS blocked | R1 residual Unix sockets (accepted, trusted scope only) |
| Load YAML → run → structured output → release resources | Pieces exist (config, cache, tools, harness) | Each unit tested separately | T5.0 wiring (F5), `Harness.close`, T5.1 CLI, JSON output incl. records and `duration_s` |

---

## D. Test coverage gaps

These are only the gaps that would catch a real failure:

1. **Side-effecting tool sequences through the real effGen loop** (scripted model, real tools):
   - rerun the same test command after a fix;
   - rewrite the same file twice;
   - run two different silent shell commands.

   These would have caught F1 and F2. They currently fail, which is the point: add them, marked as known failures until the fix is chosen, so the decision is visible in the suite.
2. **Identical re-delegation after a worker failure.** This detects F1 for `delegate`. The orchestrator would believe it retried when the worker never ran.
3. **Orchestrator under effGen's default `raise_on_error=True`, failing after a successful delegation.** This detects F3 (exception instead of `HarnessResult`, lost records).
4. **`build_model` with a loaded-but-rejected model.** This detects F6 (no `unload()`).
5. **T5.0 assembly tests**, already listed in the plan:
   - tools are built before models;
   - equal specs share one instance;
   - `close()` unloads exactly once and leaves an injected cache alone;
   - a model failure on the second spec unloads the first;
   - `close()` runs when `run()` raises.
6. **One opt-in real-model delegation run on CPU**, with LFM2.5-350M as both orchestrator and worker in ReAct mode, `file_operations` only, and a trivial task. This is the cheapest evidence that the real loop, `DelegateTool` and a real model work together. Today every delegation test is scripted, and the scripted fake cannot reveal prompt-length, parsing or loop-guard interactions with real text.
7. **Ollama window check (opt-in, T3.3).** A prompt near `context_length` with a marker in its first line detects F4's silent truncation.

---

## E. Minimal remediation plan

### 1. Fix before configuration wiring (T5.0)

1. **Commit T4.2** as it stands (F12).
2. **Decide on F1** (user approval needed; options in F1). Add the scripted regression tests from §D items 1 and 2 first, so the chosen fix is measurable.
3. **Fix F2** in `tools.py`:
   - unique result text per `ShellTool` call: the command plus a call number;
   - a thin `file_operations` wrapper that appends the operation and a call number.
4. **Fix F3** in `Harness.run` (unwrap `RunStoppedError`, or force `raise_on_error=False`), with the test from §D item 3.

### 2. Fix during configuration and CLI integration (T5.0, T5.1)

5. T5.0 as planned, plus:
   - explicit `capability_probe`, `tool_calling_mode`, `raise_on_error` and `enable_memory`;
   - the shell timeout capped at the worker deadline (F8);
   - `close()` on every exit path.
6. **F6**: unload a rejected model in `build_model`.
7. The CLI's JSON output includes worker records, the effective per-agent settings and `duration_s`; this starts F7.
8. **F10**: tighten the three test-quality items.

### 3. Defer until GPU or benchmark experiments

9. **F4**: verify Ollama's `num_ctx` against `context_length` (T3.3).
10. **F9**: validate the whole harness inside WSL2 on the Windows / RTX 4070 machine. Include a bitsandbytes NF4 load and the VRAM co-residency of the Ollama and HF models (T3.4, T3.5).
11. **F7**: generation controls in the config; per-run seeds.
12. Dataset access and task loader (T6.1, blocked on credentials), an evaluation runner and per-run artifacts (transcripts, diffs, test results). This is benchmark infrastructure, not an MVP blocker.

I propose no architectural change. The design is sound; the defects are in how effGen's loop guards meet side-effecting tools, and in one missing exception path.

---

## F. Final verdict

**Ready after specific fixes.**

The architecture is viable and appropriately minimal, the module boundaries are clean, and the security posture of the shell is honestly documented within its accepted trusted-development scope. But two verified behaviours would make the next milestones produce misleading results:

1. **F1/F2.** effGen replays stale results for repeated calls and ends runs on coincidentally identical results. A coding worker can therefore report "tests pass" from a stale observation, or stop after two harmless commands. Diagnostics E1, E2, D, A and B and the re-delegation run reproduce this with the real effGen loop and the real tools.
2. **F3.** Under effGen's default settings, an orchestrator that does not finish raises out of `Harness.run` and loses its worker records, which breaks the CLI's structured-output contract.

Neither needs an architectural change. F2 and F3 are small changes in `tools.py` and `harness.py`. F1 needs your decision because every complete fix changes effGen behaviour. After those, together with committing T4.2, configuration wiring (T5.0) and the CLI (T5.1) can proceed on solid ground.
