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

---

# Follow-up 001-A: F1/F2 reproduced, root cause, remediation recommendation (2026-10-17, Sonnet 5.5)

Status: **investigation only. No runtime behaviour was changed.** The only repository change is the new test file `tests/test_loop_guards.py`. Everything below needs your approval before it is implemented, because CLAUDE.md forbids effGen patches and private-API workarounds without it.

## 1. Reproduction (real effGen loop, scripted models, real tools)

`tests/test_loop_guards.py` has 12 tests: 6 reproductions marked `xfail(strict=True)` (they fail today, and will turn into hard failures the day a fix lands, so the marker must then be removed), and 6 pins/controls that pass today. I ran every reproduction with `--runxfail` and confirmed each fails on its **intended assertion** (two of my first drafts failed for the wrong reason: a helper typo and an F1/F2 mix-up. Both were corrected).

| Test | What happens today |
|---|---|
| `f1_rerunning_a_passing_check_after_breaking_the_code…` | 2 of 3 calls executed. The model is shown the stale `PASSED` for code that is now broken |
| `f1_rerunning_a_failed_check_after_the_fix…` | 2 of 3 executed. The identical rerun after the fix is **declined, never executed** |
| `f1_rereading_a_file_after_editing_it…` (`file_operations`) | 2 of 3 executed. The model is shown the old content |
| `f1_identical_redelegation_runs_the_worker_again` | 1 worker run, 1 record; the orchestrator is told "the same result as the identical call above" |
| `f2_silent_shell_commands_do_not_end_the_run` | `true`, `mkdir -p d`, `touch marker.txt`: `d` is created, `marker.txt` never is; `termination="stuck"`, `stop_reason="repeated_tool_result"` |
| `f2_writing_several_files_does_not_end_the_run` | writing `f.py` twice (result text is the path both times) stops the run before `g.py` is written |

Pins (pass today; document effGen behaviour that stays): `pin_a_custom_tool_repeat_is_replayed_from_the_record` (the F1 mechanism with a plain Python tool), `pin_a_tool_that_returns_constant_text_still_ends_the_run`, `pin_result_key_ignores_the_arguments_and_replay_ignores_state` (model-free, on `NativeToolLoop`). Controls (must pass before and after any fix): distinct calls with distinct results all run; a stuck idempotent loop is still stopped; **a stuck harness-tool loop (same shell command, same output, eight times) is still stopped** (`1 <= executed <= 3`, and the cap of 10 iterations is not reached).

## 2. Exact root cause

All three mechanisms live in `NativeToolLoop` (`effgen/core/agent_tool_loop.py`) and its caller in `effgen/core/agent_loop.py`. They assume **every tool is a pure function**. I instrumented them with a spy (throwaway script, not committed):

1. **F1, replay (and decline), before the tool is dispatched.**
   - `check_action` (`agent_tool_loop.py:529-548`) computes `pair = (tool name, normalized input)`. `is_exact_loop = exact_count >= 1 and known and not retry`. The key contains only what the **model** typed, never the state of the world.
   - `agent_loop.py:3000-3030`: `if check.is_exact_loop and not check.is_fuzzy_loop: replay = guards.cached_result(check)`. A recorded result (`record_pair_result`, `agent_loop.py:3176`, only for results that do not start with `"Error executing tool"`) is returned **without running the tool**, up to `MAX_REPLAYS_PER_PAIR = 2` times (`agent_tool_loop.py:555`, `:581-593`; the source comment says "a pure computation is idempotent").
   - If there is nothing recorded (the earlier call failed), `check.is_loop` is true and the call goes to the decline path (`agent_loop.py:3033-3077`): tools withdrawn, answer requested. Spy output for the stale case: `check_action … exact=True` then `cached_result … REPLAYED (tool NOT run)`; for the failed case: `exact=True` then `cached_result … None`.
2. **F2, result repeat, after dispatch.** `result_is_repeat` (`agent_tool_loop.py:648-658`, called at `agent_loop.py:3229`) uses `_result_key = (tool name, whitespace-normalized first 500 chars of the RESULT)` (`:644-646`). **The arguments are not in the key.** The second call that returns the same text is executed, then the run is told to answer from what it has (`ask_for_answer(ASK_AFTER_REPEATED_RESULT)`, tools withdrawn), and the next tool call ends it as `stuck`. Spy output: `result_is_repeat shell key='exit code: 0' -> True` on the second distinct command. A silent shell command renders as `exit code: 0`, and an effGen file write returns only the path.
3. **Not one guard, several.** Neutralising only `result_is_repeat` is **not enough**: with it disabled the same scripted run still stopped after 3 of 7 calls (`stop_reason="loop_detected"`, from the other result-text heuristics: the progress/stall accounting in `observe_result` and/or the drift check; I did not trace which one). With a **different result text per call** the same run completed 7 of 7. All of them test "was this result text already shown?", so what matters is that results reflect what happened.
4. **No supported switch exists.** `AgentConfig` has no option for the replay or result-repeat guards (`max_turns_without_progress=None` affects only the stall logic); `ToolMetadata`/`ToolCategory` are not consulted by `check_action`; a `before_tool_call` middleware runs inside `_execute_tool`, i.e. after the replay decision, so it cannot prevent it. The guards object is built at one site, `agent_loop.py:3387` (`NativeToolLoop(...)`), from the module-level name.

## 3. Options evaluated

| | A. Harness-side tool wrapper / config | B. Minimal effGen patch in a documented fork | C. Scoped compatibility adapter |
|---|---|---|---|
| What | Make our tools' results describe what happened (shell echoes the command; file writes report path, size and a short content hash) | Teach `check_action` to skip exact-loop for tools that declare themselves non-idempotent; put the call's input into the result key | A `NativeToolLoop` subclass whose `check_action` clears `is_exact_loop` for the harness's own tools, installed only while the harness runs |
| Files / functions | `tools.py`: `ShellTool._execute` (+3 lines); a small `FileOperations` subclass overriding `_execute` for `write` (~15 lines) | effGen `agent_tool_loop.py` (`check_action`, `result_is_repeat`/`_result_key`), `agent_loop.py:3229` (pass the input), a tool attribute; fork hosted somewhere; `pyproject.toml` git URL; `uv.lock` | `harness.py`: `HarnessToolLoop` (~10 lines) + a context manager that rebinds `effgen.core.agent_loop.NativeToolLoop` around `Harness.run` (~15 lines) |
| Private effGen API | None (`FileOperations._execute` is the documented override point of a public class; the result dict shape is covered by our T4.1 tests) | n/a (source patch) | Yes: module global `agent_loop.NativeToolLoop`, `NativeToolLoop.check_action`, `LoopCheck.is_exact_loop` |
| Safety protections changed | None removed. Result-keyed guards become state-sensitive: identical call + identical output is still a repeat | Exact-loop guard off for declared tools; result key gains the input: same net effect as A+C but at the source | Pre-dispatch exact-repeat guard off **for our tools only**; everything else inherited (result repeat, stall, drift, `max_iterations`, deadlines) |
| Fixes shell / file_operations / delegate | F2: shell and file_operations yes; `delegate` already has `delegation_id`. **F1: no** (replay happens before dispatch, keyed on the model's input) | Both F1 and F2, all three | F1: all three (by tool name). F2 only together with A |
| Upgrade risk | Low | Highest: rebase on two large, fast-moving files; a fork to host; breaks the `effgen==1.3.0` pin rule | Moderate: three private names; the pin tests in `test_loop_guards.py` fail loudly if effGen changes them |
| Complexity | ~25 lines + tests | ~20 lines of patch + packaging/process overhead | ~25 lines + tests |

Option A alone leaves F1. Option C alone leaves F2 (silent commands still collide). **A + C** fixes both inside our code, with no fork.

## 4. Recommendation

**Implement A + C (both in our code), and file the root cause upstream.** Keep B in reserve if you prefer a fork to a private-API adapter.

1. **A (fixes F2, no private API).**
   - `ShellTool` prefixes its report with `$ <command>` (truncated to 200 characters).
   - A `FileOperations` subclass makes a `write` report `wrote <path> (<n> chars, sha256 <8 hex>)`. A repeat of an identical write is then genuinely the same result, and a changed file is a new one.
   - `delegate` is unchanged (`delegation_id`).
2. **C (fixes F1).**
   - `HarnessToolLoop.check_action` returns the inherited result with `is_exact_loop=False` **only for** `shell`, `file_operations` and `delegate`. Custom tools keep effGen's behaviour (pinned).
   - `Harness.run` installs it for the run and restores the original in a `finally` (workers run inside the same run, so they are covered; the MVP is sequential, so the process-wide rebinding is not racy; document that it is not thread-safe).
3. **Why this is not "disabling loop detection".** After the change, the stuck-loop protections that remain are all active: an identical command with identical output is a repeat on its second execution (control test: ≤ 3 executions, not 8); the drift guard, the stall logic, `max_iterations`, the delegation limit and the worker deadline are untouched; idempotent custom tools keep the replay guard.
4. **Safety delta, honestly.**
   - A repeat of a side-effecting call now **executes** (the point), so a looping model costs one extra real execution before the result-repeat guard fires, instead of a free replay.
   - A residual false positive remains: the *same command with the same output after a different edit* (for example the same failing test trace) is still read as "no progress" and the model is asked to answer. That is effGen's existing behaviour and its price for keeping stuck-loop protection; measure it in T5.2 before adding anything like a workspace fingerprint to the result text.
5. **Decisions I need from you.** (a) Approve a private-API adapter (C), or choose the fork (B). (b) Approve the result-format change in A (it changes what a small model reads; it only adds a short command echo and a write summary).

## 5. Proposed acceptance tests

1. In `tests/test_loop_guards.py`, delete the `F1`/`F2` markers: all six reproductions pass (stale rerun, failed-check rerun, file reread, re-delegation, silent shell commands, repeated file writes).
2. Controls still pass unchanged (distinct calls; stuck idempotent custom tool stopped; **stuck shell loop stopped in ≤ 3 executions**).
3. Pins still pass (custom tools keep replay and constant-text behaviour), proving the adapter is scoped.
4. The adapter is restored after `Harness.run` returns **and** when it raises, so a later plain `Agent` run in the same process sees effGen's own guard (assert `agent_loop.NativeToolLoop is the original`).
5. A worker's nested run is covered: an orchestrator delegates twice to the same worker and the worker re-runs the same command after a change (one harness test).
6. Result-format tests: the echo is truncated at 200 characters; a write result changes with the content and not with the call number; `ShellTool`'s existing 121 tests and T4.1's `file_operations` tests still pass.
7. Full default suite green (today: 607 passed, 3 skipped, 6 xfailed), `uv lock --check` unchanged (no dependency change under A + C).

F3 (orchestrator `RunStoppedError` escaping `Harness.run`) was not investigated in this follow-up; it is independent and unchanged.

---

# Follow-up 001-B: F2 fixed; F1 blocked on a process-global change awaiting approval (2026-10-17, Sonnet 5.5)

**Outcome:** Phase A (result differentiation, which fixes F2) is implemented and verified. Phase B (the `NativeToolLoop` adapter for F1) is **not implemented**: the only way to reach effGen's guard object is rebinding a module-level name, which is a process-global runtime mutation. Per the instruction for that case, I stopped and am reporting the mechanism below. F3, T5.0 and the CLI were not touched. No effGen source was modified.

## 1. Phase A: what changed (`src/smol_harness/tools.py`)

| Tool | Before | After |
|---|---|---|
| `shell` success / failure report | `exit code: 0` / `exit code: 1\nstdout:…` | `$ <command as one line>\nexit code: …\nstdout:…\nstderr:…`. A command over 200 characters keeps its first 200, the count of the rest and `sha256` (first 8 hex digits) of the whole command, so two long heredoc writes that start alike still read differently. The structured fields (`stdout`, `stderr`, `exit_code`, `timed_out`) stay raw and unprefixed |
| `file_operations` `write` result | the bare absolute path | `wrote <path relative to the workspace> (<bytes> bytes, <n> line(s), sha256 <8 hex>)`, **read back from the file on disk** after the write (`WorkspaceFileOperations`, a 15-line subclass of effGen's `FileOperations`) |
| Failures, other operations (`read`, `list`, `search`, `metadata`, `convert`) | unchanged | unchanged (`convert` still returns effGen's own relative path) |

- **Failure semantics are intact.** A non-zero exit or a timeout is still a failed `ToolResult` whose `error` holds the report (the only text effGen shows a model for a failure). A command that prints effGen's own error prefix cannot make a success read as an error or the reverse (tested with `is_error_result`).
- **Deviation, for your decision: no execution counter.** You asked for identifiers that distinguish legitimately separate invocations. I used *content* instead of a counter: the command echo and the write summary. Reason: a per-call counter makes every result unique, which switches off every result-text guard for these tools (repeat, stall, drift), so a model repeating the same command with the same output would be bounded only by `max_iterations`. With the echo and the hash, an identical command with identical output still reads identically and is still caught (control test `test_control_a_stuck_harness_tool_loop_is_still_stopped`), while different commands and changed files read differently. `delegate` keeps its existing `delegation_id`. If you still want a counter for traceability, it can live in the structured fields (not in the text effGen compares).
- **Verified side effects, not text.** Every Phase A test checks the real disk (the directory exists, the file bytes and hash equal the summary, the symlink exists); mutation check below.

### Phase A results

| Check | Result |
|---|---|
| F2 reproductions (`silent_shell_commands…`, `writing_several_files…`), strict xfail removed | pass (they were XPASS(strict) before the markers were removed, i.e. genuinely fixed) |
| New: 7 distinct silent shell commands in a row; 7 distinct file writes | pass (these exercise the repeat, stall and drift heuristics together) |
| Mutation: command echo and write summary disabled | the three F2 tests that depend on them fail; the file-writes-in-a-row test passes (the paths already differ), so it is now named `test_control_…` and is no longer counted as a reproduction |
| 11 new unit tests in `tests/test_tools.py` (121 → 132) | summaries match the files on disk (empty, nested, non-ASCII, JSON re-indented); content changes the text, a call does not; failures and other operations unchanged; long-command hash; timeout and error semantics |
| `uv run pytest tests/test_loop_guards.py -q` | **10 passed, 4 xfailed** (the 4 F1 reproductions) |
| tools + delegation + e2e + limits | **254 passed** |
| `uv run pytest -q` | **622 passed, 3 skipped, 4 xfailed** (66.7 s) |
| `uv lock --check` | Resolved 157 packages (no dependency change) |

## 2. Phase B: investigation

**Private effGen APIs the adapter needs** (all verified present in 1.3.0):

1. `effgen.core.agent_loop.NativeToolLoop`: the **module attribute** that `drive()` looks up when it builds the guards (imported at `agent_loop.py:113`, used at `:3387`). Replaced by name.
2. `effgen.core.agent_tool_loop.NativeToolLoop.check_action(self, action, action_input) -> LoopCheck`: overridden in a subclass.
3. `LoopCheck.is_exact_loop` (a field of a plain dataclass): cleared with `dataclasses.replace`.
4. `self.tools` (the agent's tool mapping, passed in by `drive`): read to decide, per tool instance, whether the repeat is a real call.

**Is per-instance integration possible?** No for the *binding*, yes for the *behaviour*.
- `NativeToolLoop` is constructed at one site, `drive()` (`agent_loop.py:3387`; the comment there: "one construction site for every path"). I searched all of effGen: no other construction, no factory, no `AgentConfig` or tool-metadata option, no `Agent` method that supplies the guards, and no middleware seam (middleware runs after the decision). Agent subclasses cannot intercept, because `drive` is a module function that reads the name from the module.
- So the only lever is `agent_loop.NativeToolLoop = <subclass>`. **Module-level patching is required.**
- *Behaviour* can still be scoped per tool: a throwaway prototype (outside the repo) whose subclass changes `check_action` only for tool instances carrying a marker attribute (`ShellTool`, `WorkspaceFileOperations`, `DelegateTool`) made **all 14 loop tests pass with the xfail markers ignored**, including the pin showing that a plain custom tool is still replayed exactly as upstream does and the stuck-shell-loop control. In that design the subclass is inert for every unmarked tool, whichever agent or library in the process runs it.

**Limitations and risks of any rebinding**
- **Process-global and visible to all threads.** Workers run in executor threads (`DelegateTool._delegate`) inside the orchestrator's run, so they would see it (wanted); any other effGen user in the process would too (inert unless its tools carry the marker, but still a mutation).
- **Install and restore around each run is not re-entrant.** Two overlapping `Harness.run` calls (different threads) would each capture "the original", and the second restore could leave the patch installed or restore the wrong class, unless a lock and a reference count are added. An exception or interrupt between install and restore also needs a `finally`.
- **A permanent, idempotent install** (done once, never undone) removes the race and the leak-between-runs problem (there is nothing to leak: the subclass is stateless and inert unless marked), at the price of a mutation that lives for the whole process.
- **Upgrade risk:** effGen is pinned `==1.3.0`, which shields us. A new version can rename `drive`, `NativeToolLoop`, `check_action` or `LoopCheck`, or move the construction site; the failure mode is loud (the adapter's install would raise on a missing name, and the F1 tests and pins would fail), not silent.
- **Safety effect (unchanged from Follow-up 001-A):** only the pre-dispatch exact-repeat check is disabled, only for the marked tools. The result-repeat, stall, drift, `max_iterations`, delegation-limit and deadline protections remain; with Phase A the result-repeat guard now catches "same command, same output".

**Options for your approval** (nothing is installed):

| | What | Mutation | My view |
|---|---|---|---|
| **P (recommended)** | A marker-keyed subclass installed **once**, explicitly and idempotently (guarded by a lock) when the first `Harness` is built, with `install()` / `uninstall()` functions so tests can verify and reverse it. Marker: a class attribute on our three tool classes | Permanent for the process, inert for unmarked tools | Smallest and race-free; the mutation is explicit, documented and testable |
| Q | The same subclass, installed and restored around each `Harness.run` with a lock and a reference count | Process-global while any run is active | Matches "installation and cleanup" literally but adds concurrency machinery for no extra protection |
| R | A maintained effGen fork (option B of 001-A) | Source patch | Heaviest; breaks the `effgen==1.3.0` pin rule |
| S | Leave F1 open; tell the worker in its prompt to vary repeated commands | None | Not a fix: stale replays are silent, so the model cannot know to vary |

## 3. What is fixed, what is open

- **F2: fixed** (Phase A, verified).
- **F1: open.** Four strict-xfail reproductions remain (`test_f1_rerunning_a_passing_check…`, `…a_failed_check…`, `…rereading_a_file…`, `…identical_redelegation…`). They should be converted to ordinary tests only when the adapter exists and passes them; I did not weaken any of them.
- Phase C items 1 to 4 (F1 behaviours), 8 (unrelated tools keep upstream behaviour) and 10 (installation and cleanup leave no leak) depend on the adapter. Items 5, 6, 7 and 9 are covered now (7: `test_control_a_stuck_harness_tool_loop_is_still_stopped` and the idempotent-tool control; 8 is pinned for the replay today by `test_pin_a_custom_tool_repeat_is_replayed_from_the_record`).

## 4. Decision needed

Approve option P (or Q or R) to unblock F1, and tell me whether you still want an execution counter in the structured fields. Until then F1 remains: an identical repeated `shell`, `file_operations` or `delegate` call is replayed from effGen's record instead of running.

---

# Follow-up 001-C: F1 fixed by an explicit, process-wide compatibility patch (2026-10-17, Sonnet 5.5)

**Outcome: F1 and F2 are both fixed and verified. F3, T5.0 and the CLI were not touched. No effGen source was modified.** Approved design implemented as option P of Follow-up 001-B, with the contract below.

## 1. What was built

| Piece | Where | Behaviour |
|---|---|---|
| `install_effgen_loop_compat()` | `harness.py` | Explicit, idempotent. Rebinds `effgen.core.agent_loop.NativeToolLoop` to a subclass whose `check_action` clears `is_exact_loop` **only** when the tool in question carries `STATEFUL_TOOL_MARKER`. Every other guard is inherited |
| `STATEFUL_TOOL_MARKER = "executes_every_call"` | `delegation.py` | A class attribute set on `ShellTool`, `WorkspaceFileOperations` (the `file_operations` tool) and `DelegateTool`. A test fails if any tool the factory can build lacks it, so a future tool (e.g. `edit_file`) cannot be forgotten |
| `effgen_loop_compat_installed()` | `harness.py` | True only while effGen's loop currently uses the patched class |
| `Harness.run` guard | `harness.py` | Because `delegate` is always marked, `Harness.run` raises `EffgenCompatError` (naming `install_effgen_loop_compat()`) **before any agent or model call** if the patch is not installed. There is no way to run silently with stale replay |
| `uninstall_effgen_loop_compat_for_tests()` | `harness.py` | Restores effGen's own class. Tests only; refuses to overwrite a class it did not install. **Nothing in the harness calls it** (not `run`, and the future `close()` must not) |
| `EffgenCompatError` | `harness.py` | Raised with an actionable message, patching nothing |

**Install-time verification (all must hold or nothing is patched):** `effgen.__version__ == "1.3.0"`; `effgen.core.agent_tool_loop.NativeToolLoop` is a class whose `check_action` has exactly the parameters `(self, action, action_input)`; `LoopCheck` is a dataclass with an `is_exact_loop` field; `agent_loop` has the module-level `NativeToolLoop` name; `agent_loop.drive` still builds the guards with `NativeToolLoop(`; and the current binding is effGen's own class (it refuses to wrap an unknown replacement).

**Installation is separate from construction.** Importing the harness or building one patches nothing (tested in a fresh interpreter). An application, the future CLI, or a test fixture must call the function once before any agent runs; `tests/conftest.py` does so in a session fixture.

## 2. The process-global nature, stated plainly

- It changes effGen for **every agent in the process**; it is inert for any tool without the marker, whichever code runs it.
- **Unsupported:** installing or restoring while any agent is running, or from several threads at once. A lock protects only the module's two bookkeeping variables, not running agents.
- **Private-API dependencies:** `effgen.core.agent_loop.NativeToolLoop` (module attribute, construction site `agent_loop.py:3387`), `effgen.core.agent_tool_loop.NativeToolLoop.check_action`, `LoopCheck.is_exact_loop`, `NativeToolLoop.tools`.
- **Upgrade risk:** effGen is pinned `==1.3.0` and the patch refuses any other version. Any rename or move raises `EffgenCompatError` at install, and the F1 tests and pins in `tests/test_loop_guards.py` fail loudly. Re-verification on an upgrade means re-reading `agent_loop.drive` and `NativeToolLoop.check_action`.

## 3. What changed for the loop protections

Disabled, for marked tools only: effGen's pre-dispatch exact-repeat replay/decline. Kept and tested: the repeated-result guard, stall and drift accounting, `max_iterations`, `max_delegations`, the worker deadline. Measured bounds with the patch (scripted, deterministic):

| Runaway behaviour | Outcome |
|---|---|
| Same shell command, same output, ×8 | stopped after ≤ 3 executions (repeated result) |
| Same failing shell command ×12 | 3 executions, 4 model calls, `stuck` / `loop_detected` (the stall guard: the same error is not progress) |
| Same file read ×12 | 2 executions, 3 model calls, `stuck` / `repeated_tool_result` |
| Same `delegate` call ×20, `max_delegations=3` | 3 real worker runs, then refusals; ≤ 10 orchestrator calls |
| Same failing `delegate` call, no limit | 5 real worker runs, then the orchestrator's iteration cap (`stuck`, never `tool_failed`) |

**One existing expectation changed, deliberately, and you should know.** `tests/test_limits.py::test_retrying_the_identical_failing_delegation_forever_ends_stuck_not_tool_failed` asserted `len(result.workers) == 1` and `stop_reason == "loop_detected"`, i.e. that effGen does not re-dispatch the identical delegation. That is exactly the F1 behaviour, so it could not survive the fix. It now asserts what the approved design implies (5 real worker runs bounded by the orchestrator's iteration cap, `stuck`, never `tool_failed`, no exception), with exact numbers, and a new test covers the delegation limit. The practical cost: an orchestrator that retries the identical failing delegation now spends up to `max_delegations` (default 4) real worker runs instead of one. A `delegate` report is unique per call (`delegation_id`), so no result-text guard can catch that loop; `max_delegations` and `max_iterations` are its bounds.

## 4. Verification (exact)

| Command | Result |
|---|---|
| `uv run pytest tests/test_loop_guards.py -q` | **35 passed** (14 before; the four former xfails are ordinary tests) |
| `uv run pytest tests/test_tools.py tests/test_delegation.py tests/test_e2e_delegation.py tests/test_limits.py -q` | **255 passed** |
| `uv run pytest -q` | **648 passed, 3 skipped, 0 xfailed** (72 s) |
| `uv lock --check` | Resolved 157 packages, unchanged |
| Leak check | no stray processes |

Mutation checks: (1) if the patch is never installed, the four F1 tests fail; (2) if the marker check is widened to every object, 4 tests fail (the unmarked-tool pin, the idempotent stuck-loop control, the model-free scope test, and the upstream-replay test), so the scoping is enforced. An early draft of my "import does not install" test used `importlib.reload`, which poisoned later tests by resetting the module's bookkeeping while effGen still held the patched class; it now runs in a fresh subprocess. That also shows the module-level state is the patch's one fragile point: do not reload `smol_harness.harness` in a process that installed it.

## 5. Contract for the next tasks (added to T5.0 / T5.1)

- The CLI and any application **must call `install_effgen_loop_compat()` before building or running a harness**; `Harness.run` refuses otherwise.
- `Harness.close()` (T5.0) **must not** uninstall; add a test for it.
- Never reload `smol_harness.harness` after install; never run two installs/restores concurrently or during a run.

## 6. Status of the review's P1 findings

- **F1: fixed** (this follow-up). **F2: fixed** (Follow-up 001-B).
- **F3: open** (`Harness.run` lets `RunStoppedError` escape under effGen's default `raise_on_error=True`; the run's worker records are lost). Not touched, as instructed.

---

# Follow-up 001-D: F3 closed; one new P3 finding (2026-10-17, Sonnet 5.5)

**Outcome: F3 is fixed and verified. All three P1 findings (F1, F2, F3) are now closed.** No effGen source was modified; the F1 compatibility patch and its installation requirement are unchanged; no worker, delegation or model-loading code was touched.

## 1. What effGen actually does (verified in source and by running it)

| Question | Answer |
|---|---|
| When does `Agent.run` raise `RunStoppedError`? | Only under `raise_on_error=True`, and only when the run's `outcome == "stopped"` (iteration cap, repeat guards, a tool failing on its own side, an empty answer after tools): `core/agent_generation.py:_reconstruct_error`, the sole construction site |
| What does it carry? | The full `AgentResponse` of the stopped run (`success=False`, `termination`, `stop_reason`, `output`, `partial`); `RunStoppedError.__init__` (`errors.py:373`) copies `output`, `stop_reason` and `partial` from it |
| Is it the same response `raise_on_error=False` returns? | Yes. Measured on four scenarios: identical `success`, `termination`, `stop_reason`, `output` and worker records in both modes |
| What else does `True` raise? | **Different exceptions with no response**: a plain `RuntimeError` for guardrail blocks and some iteration/model failures, and typed provider errors (`BackendUnreachableError`, `ModelAuthError`, …). With `False` the same failures come back as a failed response, so these are **not** equivalent across modes and cannot be normalised without inventing a response |

## 2. The fix (`src/smol_harness/harness.py`, `Harness.run`)

- `except RunStoppedError`: if `exc.response` is usable, it becomes the run's response and the normal `HarnessResult` is built from it (`success=False`, the real `termination`, `response.stop_reason`, this run's worker records).
- "Usable" is checked, not assumed (`_stopped_response`): the response exists, has an `output`, and has `success is False`. Otherwise the exception propagates (nothing invented).
- Any other `Exception` propagates unchanged. Before it does, `_keep_records` attaches this run's worker records as `exc.worker_records` (a list, possibly empty) and, if there are any, a note (`exc.add_note`) saying how many delegations ran. This is the only addition beyond the requested normalisation: without it the records of a run that raised would still be lost. `BaseException`s such as `KeyboardInterrupt` pass through untouched.
- `EffgenCompatError` is still raised first, before any agent exists; the `Harness.run` docstring now states the contract.

**Documented limitation (unchanged by design):** under `raise_on_error=True`, a non-`RunStoppedError` failure of the orchestrator (a provider error, a guardrail block, a model that raises) **raises**, whereas under `False` it returns a failed result (`termination="error"`, `stop_reason="generation_failed"`). A caller that wants results in both cases must use `raise_on_error=False` for the orchestrator, or catch the exception and read `worker_records`. The CLI (T5.1) must handle both: non-zero exit for a failed result, and for a propagated exception.

## 3. Tests (`tests/test_orchestrator_failure.py`, 28 tests, real effGen execution with scripted models)

| Requirement | Test(s) |
|---|---|
| 1, 2, 6: failure under `True` and `False`, same result, real stop reasons | `test_a_stopped_orchestrator_returns_the_same_result_whatever_raise_on_error_says` ×2 scenarios (`max_iterations_partial`, `loop_detected`; `termination="stuck"`); a premise test proves effGen really raises for the same run |
| 3, 4: records after a successful delegation | `test_failure_after_a_successful_delegation_keeps_the_worker_record` ×2 modes |
| 5: records after a failed delegation | `test_failure_after_an_unsuccessful_delegation_keeps_the_failed_record` ×2 (the worker's own `error_type` legitimately differs by mode: existing `DelegateTool` behaviour) |
| Failure with no delegation | `test_failure_with_no_delegation_gives_an_empty_record_list` ×2 |
| 7: no leaks between runs | `test_records_do_not_leak_between_separate_runs` ×2 |
| 8: unrelated exceptions not swallowed | model exception under `True` propagates as `RuntimeError` (not `RunStoppedError`, no `response`) carrying `worker_records` and the note; the same failure under `False` is a failed result (the documented asymmetry); `ZeroDivisionError`, `KeyError`, `ValueError` ×2 modes propagate; `KeyboardInterrupt` untouched |
| 9: missing or malformed response information | a well-formed stub still yields a result; `response` removed → propagates; four malformed responses (no `output`, no `success`, `success=True`, `success=None`) → propagate; records reach a propagating stopped error |
| Compat requirement preserved | `Harness.run` still raises `EffgenCompatError` before any model call |

Mutation checks: handling disabled (the F3 bug) → 7 failures; validation removed → 4 failures (the malformed responses); records not attached → 13 failures.

## 4. Verification (exact)

| Command | Result |
|---|---|
| `uv run pytest tests/test_orchestrator_failure.py -q` | **28 passed** |
| delegation, e2e, limits, loop guards and tools tests | **290 passed** |
| `uv run pytest -q` | **676 passed, 3 skipped** (73 s) |
| `uv lock --check` | Resolved 157 packages, unchanged |

Two mistakes in my first draft of the tests were caught and fixed before the results above: the "records reach a propagating error" test patched `Agent.run` for the worker as well as the orchestrator, and one assertion assumed the worker's `error_type` was the same in both modes.

## 5. New finding

| ID | Severity | Type | Component | Issue | Evidence | Recommended action |
|---|---|---|---|---|---|---|
| F13 | P3 | Verified defect (synthetic trigger; effGen never builds such an error) | `delegation.DelegateTool._delegate` (the `except RunStoppedError` branch) | A worker whose `Agent.run` raises a `RunStoppedError` whose `response` is missing or malformed makes `_from_response(worker, exc.response)` fail inside the handler (`AttributeError`); the delegate tool call fails and **no `WorkerRecord` is created**, so the failure disappears from the run's records | Throwaway script: the worker's `Agent.run` raised `RunStoppedError` with `response=None`; the harness result had `records: []` and the orchestrator carried on to `done` | During the delegation-hardening work: validate the response as `Harness.run` now does and fall back to a failed report built from `str(exc)`. Not fixed here (outside F3's scope; delegation logic was not to be touched) |

## 6. State of the review's findings

- **P1: F1, F2, F3 closed.**
- Open: F4 (Ollama server context vs `context_length`, T3.3), F5 (T5.0 must set effGen flags explicitly), F6 (unload a model `build_model` rejects), F7 (generation settings in config), F8 (shell timeout vs worker deadline, T5.0), F9 (Windows/WSL2 platform), F10 (test-quality nits), F11 (size target in the architecture doc), F12 (commit hygiene: the T4.2/F1/F2 work is now committed as `1d7d2d3`; F3 is not), F13 (new, above).

---

# Follow-up 001-E: F5 and F8 addressed in T5.0 (2026-10-17, Sonnet 5.5)

No P1 finding was reopened. Status of the two findings T5.0 absorbed:

- **F5 (explicit effGen settings): addressed.** `build_harness_from_config` states, for every agent, the settings that matter instead of inheriting effGen defaults: `tool_calling_mode` (builder parameter, default `"react"`), `raise_on_error=False` (effGen's default is `True`), `enable_memory=False` (default `True`), `capability_probe=False` (default `True`: an extra probe generation), `enable_sub_agents=False` (default `True`), `max_iterations`, `system_prompt` (only when configured), and for Ollama the agent-side `max_context_length`. Tested through the `Agent`s the harness actually constructs. Still not configurable, deliberately (F7): `temperature` (effGen 0.7), `seed`, `max_tokens`, sampling parameters.
- **F8 (shell timeout vs worker deadline): mitigated, not eliminated.** Verified first: a worker with a 2 s deadline that started `sleep 6` with `timeout=30` ran for 6.4 s (the command ran to its end; the cooperative check only fires before the next call), and with the default cap the overrun could reach 300 s. Now `DelegateTool` publishes the active worker's `DeadlineMiddleware` through a `ContextVar` (set inside the executor thread's own context copy, so it cannot leak) and `ShellTool` bounds its *effective* timeout by the whole seconds the worker has left (never below 1 s); the model's argument is not modified, and the report says "(all the time the worker had left)". Same scenario now: 2.4 s. **Remaining limits, documented and tested:** the deadline is still cooperative, so a single model generation in flight is never interrupted and, if no further call follows, the worker is reported `completed` although it overran; the effective shell timeout is rounded up to whole seconds with a 1 s floor plus the 2 s `timeout -k` grace; the orchestrator has no deadline and its shell keeps the model's own timeout; other future tools do not read the deadline.
- F4 and F7 remain open for T3.3 and a later config task; the new `max_context_length` is agent-side only and cannot change Ollama's server context.
