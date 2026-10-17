# Independent review 002: RTX 4070 experimental handoff readiness

- **Date:** 2026-10-17
- **Reviewer:** Claude Opus 5.5 (independent of the implementing sessions)
- **Working tree reviewed:** commit `b06bc26` plus the uncommitted changes: `experiments/rtx4070/` (all files), `tests/test_rtx4070_package.py`, `tests/integration/test_liquid_harness.py`, `tests/test_integration_optin.py`, the three modified docs, and an empty, untracked root `README.md`.
- **Scope:** the RTX 4070 package (`rtx4070.py`, `stage_runner.py`, `models.json`, both config templates, README/RESULTS/TROUBLESHOOTING), its 112 tests, and the T5.2b Liquid AI evidence (`tests/integration/test_liquid_harness.py` and its saved report `/tmp/smol-harness-lfm-harness.json`). Where a claim depends on effGen behaviour, I checked the installed effGen 1.3.0 source in `.venv`.
- **Requirements note:** the session preamble says detailed Review 002 requirements would follow it. None were provided, so this review follows the objective stated in the preamble: implementation, testing, reproducibility, hardware assumptions, portability, safety, colleague-facing instructions, and the raw-output versus model-visible-output question.
- **Method:** source reading, the saved Liquid report (read-only), and one model-free diagnostic (a scripted model through effGen's real ReAct loop with the real `file_operations` tool; script in `/tmp/rev002/`, outside the repo). No model was loaded, nothing was downloaded, and no GPU was used. I modified no source, test or project document other than this file.

Evidence labels: **Verified** (reproduced or read directly from saved evidence in this review), **Source-supported risk** (follows from code I read; not reproduced), **Untested hypothesis**, **Optional improvement**.

### Verification I ran

| Command | Result |
|---|---|
| `uv run pytest tests/test_rtx4070_package.py -q` | **112 passed** (9 s) |
| `uv run pytest -q -p no:cacheprovider` | **896 passed, 8 skipped** (119 s) |
| `uv lock --check` | Resolved 157 packages |
| `git status --short` before and after | unchanged (apart from this file) |

The reported local results are confirmed.

---

## A. Executive assessment

**Is the package well built?** Mostly, yes.
- The design is sound: staged experiments with explicit dependencies, one process per stage under a hard time limit, fail-closed GPU placement (no silent CPU fallback), and coding success judged from the repository by an independent pytest run.
- The preflight is honest. The `requires` contract stops a single root cause from cascading into several FAILs, and a missing dependency can never show as PASS.
- Packaging is defensive: it handles partial records, excludes weights, redacts twice, and scans the finished archive.
- Review 001 F4 (Ollama server context) is handled properly for this package. A derived tag fixes `num_ctx` in its Modelfile, `/api/show` checks it before the run, and `/api/ps` checks it at run time.

**Is it ready to send to the colleague?** Not yet. Two problems could waste or endanger the colleague's single run:
1. **R2-01.** Stage B leaves the 7B Ollama model resident in VRAM (Ollama's default keep-alive is 5 minutes). The very next stage's critical `vram_free ≥ 6000 MiB` gate is then evaluated against that occupancy. On the primary profile's own estimates this lands at about 5.6 GiB free, which **blocks C, and therefore D and E**. The printed remedy ("close games, browsers…") points at the wrong cause.
2. **R2-02.** Stage E runs model-issued shell commands on the colleague's personal Windows machine. The only confinement proof (effGen's probe plus `echo sandbox-ok`) never tests the two WSL2-specific escape routes: writes to `/mnt/c`, and Windows interop (`cmd.exe`, `powershell.exe`), which runs outside every Linux namespace. effGen's read-only remounts discard their own errors (`2>/dev/null`). A hole would be invisible, and the README promises "no unrestricted shell".

Both fixes are small and stay inside the package. One design decision is also needed: **R2-05**, stage D gates the coding stage on model task success, not on mechanics.

**Liquid AI evidence (raw versus model-visible):** the reported T5.2b figures stand, but the labelling needs correcting (§C):
- The "result reached the model" instrument in the Liquid test is only **partly** fixed. It is wrong for a *successful* `file_operations` call. It did not affect the reported numbers, because all 7 executed calls failed.
- The recorded `raw_output` is the text **before** effGen's stop-sequence trim. The "self-written `Observation:`" counts, the token totals and the tok/s figures therefore partly describe text the model never saw again.
- The docs contradict each other on this point (plan §4 T5.2b says the bug was fixed; the RTX section says it is latent).

---

## B. Prioritized findings

By severity: **P0: 0, P1: 2, P2: 5, P3: 8** (15 total).

| ID | Sev. | Type | Component | Issue | Evidence | Recommended action |
|---|---|---|---|---|---|---|
| R2-01 | **P1** | Source-supported risk (the outcome depends on unmeasured VRAM numbers) | `rtx4070.py` `chk_vram_free` (:526-537); `stage_runner.stage_b` (:365-419); `run_stage` (:709) | **The free-VRAM gate is evaluated while Ollama still holds stage B's model.** `vram_free` is critical, `needed="BCDE"`, and is re-run at the start of every stage. Stage B never unloads `smol-orch:8k`, and Ollama keeps a model resident for 5 minutes by default. When C starts seconds later, the estimate is: Ollama ≈ 5.5 GiB (`models.json`: 4.4 + 0.5 + 0.6) + Windows ≈ 1 GiB, which leaves ≈ 5.6 GiB of 12 GiB free. That is below the 6000 MiB FAIL threshold, so C fails and D and E are skipped. **For D the gate is ill-defined either way**: D *wants* Ollama resident, but the 6000 MiB threshold is applied regardless of what is already loaded, so D passes or fails depending on whether the keep-alive expired during C. The fix text blames Windows programs. | Read in source; the keep-alive default is Ollama's documented behaviour; `grep keep_alive` finds nothing in the package. TROUBLESHOOTING mentions `ollama stop` only for the "Ollama uses the CPU" case. Not tested: no test models residency across stages. | At the end of stage B (and at the start of C), unload explicitly: `POST /api/generate {"model": tag, "keep_alive": 0}`, then confirm with `/api/ps`. Record `ollama ps` and `nvidia-smi` at the start of every stage. For D, replace the flat 6000 MiB with "free ≥ expected HF need (+ margin) given what is resident", or make it WARN and let a real OOM fail the stage with its diagnosis. Add a test with a stubbed `/api/ps` showing a resident model before C. |
| R2-02 | **P1** | Untested hypothesis (safety; high impact, low likelihood) | `stage_runner.stage_e`; `rtx4070.chk_sandbox` (:442-452) and `PROBE` (:187-194); effGen `security/sandbox.py` `_confine_script` (:309-328), `_probe_confinement` (:884-938) | **The WSL2-specific escape routes are never probed before model-issued commands run on the colleague's PC.** effGen remounts each mount read-only with `mount -o remount,bind,ro … 2>/dev/null`, so a remount that fails is silent. Its probe only checks that a write under the system temp dir fails. The package's preflight only runs `echo`. On WSL2 there are two extra routes: (a) `/mnt/c` (drvfs), whose read-only remount has never been observed; (b) **Windows interop**: an `.exe` started from WSL runs as a Windows process, outside the PID, network and mount namespaces. It is probably blocked because `ShellTool` hides `/run` (the interop socket lives in `/run/WSL`) and effGen strips `WSL_INTEROP`, but that has never been run. | Source lines cited. U15 (WSL2 sandbox) is still open in `docs/new-session.md` §6. The T4.2 verification ran only on this Linux laptop. | Before handoff, extend the `sandbox` check (stage A, so it blocks E) to run, through `ShellTool`, `touch /mnt/c/.smol-probe-$$`, `cmd.exe /c echo x` and `powershell.exe -c echo x`. **All three must fail**, and the check must confirm `/mnt/c/.smol-probe-*` does not exist afterwards. Otherwise FAIL, so E is not run. (A write that succeeds should be removed by the driver.) Keep the README promise only once this holds. |
| R2-03 | P2 | Verified defect (instrument), no effect on the reported figures | `tests/integration/test_liquid_harness.py` :204, :289-290 | **"Result reached the model" uses the raw output for a successful `file_operations` call.** `shown = output["result"] if "result" in output else output`. For a successful `file_operations` call there is no `result` key, so the key compared with the prompt is `str(dict)` (`"{'success': True, 'data': …"`). effGen actually shows `str(output["data"])` (`agent_tool_execution.py:348-351`), so a successful read that did reach the model is counted as *not* reached. Failures use `ToolResult.error`, which is a substring of what effGen shows (`Error executing tool '…': Tool execution failed: <error>`, :314-318, :420), so they are measured correctly. The per-call `"result"` field in the saved JSON is the raw dict repr, not what the model saw. | Saved JSON: all 7 executed `file_operations` calls (A: 2, B3: 5) have `success: false`, and each `result` field starts `{'success': False, 'data': None, …`. The A metric therefore took the `error` branch, so "2/2 reached the model" is valid; the model's next generation discusses the error in both attempts. **Doc contradiction:** plan §4 T5.2b says the bug was "found and fixed"; plan §4 RTX says the Liquid file still has "the same raw-output comparison". Truth: partly fixed. | Reuse `stage_runner.model_visible` (or the same logic) in the Liquid test, and store a `visible` field next to `result`. Correct both plan entries to "partly fixed; reported figures unaffected because no executed call succeeded". Do not trust the metric on a future run with a model that reads successfully until this is fixed. |
| R2-04 | P2 | Verified (evidence interpretation); the mechanism is source-supported | `test_liquid_harness.py` `traced` / `tracing_build` (:216-254); `stage_runner.tracing_build` (:135-156) | **`raw_output` is recorded before effGen's stop-sequence trim, so it is not what entered the conversation.** effGen sends `"\nObservation:"` (plus `\nQuestion:`, …) as stop sequences (diagnostic: both calls of a scripted ReAct run received them). effGen's HF engine cuts at a received stop sequence and reports `finish="stop_sequence"` (`transformers_engine_generation.py:150-162`). In the saved LFM report, **all 22 generations have `finish="stop"`**, and A2, A3 and B3 contain `\n\nObservation: …` *after* a real `Action:`. So the stops were not applied at the engine for this model; effGen most likely trims them locally after the spy (`agent_generation.py:548-552, 619-629`, the "interleaves reasoning" path). The consequences: (a) the invented values after a real action (A2 `abc123`, A3 `This is secret text`) were **never shown back to the model**; the next prompt carried the real error. (b) The "8 of 12 first generations contained a self-written `Observation:`" figure and the 2,096 tokens / 11.5 tok/s include text effGen discarded. (c) Conversely, B and B2's invented observations are **inline** (`… parameters. Observation: …`, no newline), so no stop sequence catches them and effGen accepted the `Final Answer` that followed. That one *is* a model-visible failure. | Saved JSON (`finish` values; raw texts quoted in §C). Diagnostic `/tmp/rev002/stops.py`: with a scripted model the invented `Observation: invented abc123` is absent from the next prompt and the real file content is present. Why stops went local for LFM (template-declared reasoning versus something else) was not verified without the model. | Record what effGen kept, not only the raw text: either the next prompt's tail or the trimmed text (apply the same `apply_stop_sequences` to `raw_output` and store both). Split the figure into "invented observation after an action (discarded by effGen)" and "inline invented observation (reached the answer)". The RTX runner has the same instrument: for Qwen on Ollama and HF the stops will likely be applied by the provider, so `raw_output` there is post-stop. Cross-model comparisons of this metric are therefore not like-for-like unless both are recorded. |
| R2-05 | P2 | Design / documentation mismatch | `stage_runner.stage_d` (:492-508), `delegation_attempts` success (:294-295); README "What each stage proves" row D; `REQUIRES["E"] = ("D",)` | **Stage D passes only on full task success, but the README says it proves a mechanism.** The README says D PASS means "the delegate call really executed the worker and its result reached the orchestrator". The code requires the worker to read `secret.txt`, the code to come back, and CLI `success`. Because E requires D, **the coding experiment runs only if the 1.5B worker gets the path right.** That is exactly the failure LFM showed in A, B3 and B3's retries. B, by contrast, passes on mechanics. The "hinted" delegation variant (a relative-path hint in the task) mitigates this, but it is prompt-fitting, which plan §4 T5.2b advised against. | Source lines cited. | Decide explicitly (user decision). Recommended: D PASS = mechanics (delegate executed, worker executed ≥1 tool, a structured report reached the orchestrator), consistent with B, with `task_success` recorded separately and shown in the summary. Then E runs whenever the plumbing works, and its own oracle judges success. Update the README row to match whichever rule is chosen. |
| R2-06 | P2 | Verified (reproducibility) | Both runners: temperature 0.1 plus a seed per attempt | **"3 attempts" are not 3 independent samples.** In the LFM report, B attempts 2 and 3 (seeds 1 and 2) have byte-identical first generations (63 tokens). The RTX runner uses the same near-greedy setting, so "0/3" or "1/3" can be one or two distinct behaviours. | Saved JSON: distinct first generations: A 3/3, **B 2/3**, B2 3/3, B3 3/3. | Record, per experiment, "distinct trajectories k of n" (a hash of each attempt's generations) next to the success count, in both runners. Don't add attempts to compensate; just report k. |
| R2-07 | P2 | Source-supported risk | `rtx4070.run_stage` (:729-742); `stage_runner.main` (:545-555) | **A stage that hits its hard limit loses its attempt evidence.** The runner writes `result.json` only at the end. On timeout the driver kills the process group and writes a bare FAIL. The per-attempt generations, tool logs and GPU samples exist only in memory (D's and E's `cli_*.json` files do survive). Budgets: D has up to 6 CLI runs (2 variants × 3) with `worker_timeout_s: 900` inside a 2400 s limit; E has 3 × 900 s inside 3000 s. | Source; the timeout values are in the templates and `TIMEOUTS`. | Append each attempt record to `stage_X/attempts.jsonl` (redacted) as it completes, and have the timeout path read it into the FAIL record. Optionally lower `worker_timeout_s` so the attempts fit inside the stage budget. |
| R2-08 | P3 | Verified defect (instrument) | `test_liquid_harness.py` `report` fixture (:145-151) | `process_peaks[*].experiments` is always `[]`. `data["experiments"]` *is* `previous["experiments"]` (same dict object), so the set difference is always empty. The per-experiment peak-RSS attribution claimed in the plan ("3.09 to 3.42 GiB") cannot be reconstructed from the report. | Saved JSON: four `process_peaks` entries, all `"experiments": []`. | Copy before mutating (`dict(previous.get(...))`). |
| R2-09 | P3 | Evidence labelling | plan §4 T5.2b, B3 row | B3 claims "results reached the model", but the B3 test computes no such metric and prompts are not saved. The claim is inferred from the model's next text ("The error indicates the path is invalid…"), which supports it but is not an instrument. | `test_b3_…` record keys; the B3 raw generations in the saved JSON. | Label it "inferred from the model's replies", or add the same visibility check as A once R2-03 is fixed. |
| R2-10 | P3 | Test gap | `tests/test_rtx4070_package.py` :976-994 | `model_visible` and `result_reached_model` are tested only with hand-made `FakeResult`s. Nothing pins them against effGen's real `_execute_tool_once` with the real `ShellTool` / `WorkspaceFileOperations` / `DelegateTool`. Two instruments in this project have already been wrong about exactly this. | My diagnostic shows a scripted-model ReAct run with the real tool needs about 20 lines and no network. | One model-free test per tool (success and failure): run a scripted agent, capture the second prompt, and assert that `model_visible(result)[:40]` is in it. |
| R2-11 | P3 | Safety wording | README step 11, RESULTS "Redaction" | "It contains no passwords, tokens or model files" is stated absolutely. Redaction is pattern-based (HF, GitHub and `sk-` tokens, bearer headers, URL passwords, `key=value`), and the shell's **reads are unconfined** (R1). The archive keeps 400 characters of every tool result and 500 of every generation. A PEM private key or an unlabelled password a model happened to `cat` would pass. | `_REDACTIONS` (:58-65); R1 in `docs/new-session.md` §6. | Add a PEM-block pattern (`-----BEGIN [A-Z ]*PRIVATE KEY-----`) to the redaction and to `secrets_left`, and soften the wording to "known credential formats are removed; please glance at it with `tar -tzf` before sending". |
| R2-12 | P3 | Source-supported risk (reproducibility) | `cmd_install_deps` (:857) | `uv pip install bitsandbytes==0.50.2` runs with no constraints, so its resolver may upgrade or downgrade locked packages (numpy, torch) in the `.venv` without any check. Whether 0.50.2 exists and has CUDA 13 kernels is unverified (already stated). | Source. | Install with constraints taken from the lock (`uv export --frozen --no-hashes > constraints.txt`; `uv pip install -c constraints.txt …`), then diff `uv pip list` before and after and record it under `environment/`. |
| R2-13 | P3 | Documentation drift | `docs/new-session.md` | The handoff doc contradicts itself and the code. The header says "end of the T4.2 session". §2 and §7 say "there is no CLI yet", but the CLI exists. §2 still says `build_tools` is "not yet called by `build_harness`". §3 says 784 passed, §6 says 676 and §7 says 896. The empty, untracked root `README.md` would be committed by a careless `git add -A`. | Read directly. | When the package is committed: refresh the header, §2, §3, §6 and §7, and delete or fill `README.md`. |
| R2-14 | P3 | Colleague-facing message | `chk_vram` (:297-306); `models.json` `gpu_name_contains` | "RTX 4070 Laptop GPU" (8 GB) matches `"4070"`. The < 11000 MiB FAIL then advises "use the fallback profile", but the threshold is profile-independent, so fallback fails the same check. | Source. | Make the threshold per profile (fallback's own estimate is 7.9 GiB), or drop the fallback suggestion from that message. |
| R2-15 | P3 | CLI help | `build_parser` (:1025) | The `--production-defaults` help text says it disables "temperature 0.1", but it also disables `max_tokens 768` and the per-attempt seed (`stage_runner.SAMPLING`, `runner_overrides`). | Source. | Fix the help text. |

---

## C. Raw output versus model-visible output in the Liquid AI evidence

There are two distinct discrepancies.

### C1. Tool results: what the instrument logs versus what effGen shows (R2-03)

What effGen 1.3.0 puts in the observation (`agent_tool_execution.py:296-365, 386, 420`):

| Tool outcome | Text the model sees |
|---|---|
| any failure (`ToolResult.success` false) | `Error executing tool '<name>': Tool execution failed: <ToolResult.error>` |
| `ShellTool` success | `output["result"]` (the `$ cmd` report) |
| `file_operations` success | `str(output["data"])` (file text, listing, or the `wrote …` summary) |
| `DelegateTool` | the JSON string |

| Instrument | Failure path | `file_operations` success | Shell success |
|---|---|---|---|
| Liquid test (`e["error"] or e["result"]`, `result = output.get("result") or output`) | correct (a substring) | **wrong**: `str(dict)` | correct |
| RTX `stage_runner.model_visible` | correct | correct | correct |

In the saved Liquid evidence, every executed call failed, so the reported A mechanics ("2 of 3 attempts executed a call; the result reached the model in both") are correct. The model's follow-up text confirms it: "The error indicates the path is invalid because it's outside the allowed directories". The defect would only bite on a *successful* run, which is exactly the run that matters next. The JSON's `result` field must not be read as "what the model saw".

### C2. Generations: raw engine text versus what effGen kept (R2-04)

Saved A attempt 2, first generation (verbatim, abridged):

```
Action: file_operations | Action Input: {"operation": "read", "path": "/secret/file.txt", "content": "..."}

Observation: The tool returned the access code "abc123".
...
Final Answer: abc123
```

`finish="stop"`, not `"stop_sequence"`: effGen's engine did not cut this text, although effGen requests `"\nObservation:"` as a stop. effGen then executed the real call, and the attempt's second generation reasons about the real error. Its final answer is the error explanation, not `abc123`. So:

- **Model-visible:** the invented observation and the invented `abc123` after a real action were discarded, which is consistent with `docs/new-session.md` §6 ("effGen discards it and runs the real tool").
- **Raw-only metrics:** "8 of 12 first generations (14 of 22) contained a self-written `Observation:`", "2,096 generated tokens" and "11.5 tok/s" measure engine output, part of which never entered the conversation. They are true statements about what the model *generated*, not about what the agent *processed*. With stops enforced at the engine, much of that text would never have been generated.
- **Genuinely model-visible failures:** B and B2 (no `delegate` call, 0/6) and A attempt 1 (no call, invented `abc123`) stand. Their invented observations are inline (`…parameters. Observation: The tool returned the access code. Final Answer: the access code`), so effGen's newline-anchored stop cannot catch them, and the `Final Answer` was accepted. **The CLI exit 0 / `success: true` finding for B is fully supported.**

### C3. What remains true of T5.2b

All headline conclusions survive: A 0/3, B 0/3, B2 0/3, B3 0/3 task successes; the worker half's mechanics work; C was not attempted; and exit status is not task success. What needs correcting is the labelling of the mechanics and observation figures (R2-03, R2-04, R2-08, R2-09), the independence of the attempts (R2-06), and the contradictory plan entries.

---

## D. Package review by area

**Implementation.** Strengths, beyond those in §A: weights never enter the archive (by suffix, plus a 25 MiB cap); `run_meta` and `stage_result` tolerate malformed records; the runner reaches the harness only through its public CLI path, except `_agent_config`, which is used deliberately for runner overrides and labelled. Gaps: R2-01, R2-05, R2-07.

**Testing.** The 112 tests are mock-only and cover the driver's logic well: the `requires` cascade, packaging of partial runs, redaction plus the archive scan, gitignore, and the AST guard against loading models. They cannot cover anything that matters on the hardware, which is correctly stated. Missing, and cheap: the real-effGen visibility pin (R2-10), Ollama residency between stages (R2-01), and the WSL2 escape probe as a unit-tested check with stubbed results (R2-02).

**Reproducibility.**
- Pinned: the Ollama manifest digest (WARN on drift), the HF revision plus the `model.safetensors` size, `uv.lock`, the bnb version, and the git commit and dirty flag in `run.json`. The package-local `.gitignore` keeps `results/` and `generated/` out of the dirty flag (verified with `git check-ignore`).
- Not pinned: the Ollama server version (recorded, not checked); the HF weight *hash* (size only); the bnb install's effect on locked packages (R2-12).
- The runner-applied sampling remains Review 001 F7.

**Hardware assumptions.** All VRAM numbers are estimates and are labelled as such. The primary profile totals 8.9 of 12 GiB, which is plausible for co-residency in D. The flat per-stage free-VRAM gate does not fit a pipeline in which one stage leaves a model resident (R2-01). The WSL2 `nvidia-smi` `memory.used` figure is assumed to include Windows processes (GpuSampler docstring); that is unverified, so the peak-VRAM evidence in D should be read with that caveat. CUDA 13 support for Ada (sm_89) is not in doubt; the driver ≥ 580 requirement is checked.

**Portability.** The OS, WSL2-vs-WSL1, `/dev/dxg`, `/mnt` location, x86_64 and Ubuntu checks are appropriate. The sandbox under WSL2 is the single biggest unknown (U15), and the stage A check proves only that `echo` runs (R2-02). The sandbox inherits `PATH` (effGen `_build_env` keeps it), so the coding task's bare `python -m pytest` resolves to `.venv/bin/python` because `run_stage` puts `.venv/bin` first. That is correct, but it depends on the driver; T5.2a used an absolute interpreter path.

**Safety.** Good: no CPU fallback, no unrestricted shell by construction, disposable workspaces deleted after each attempt and never archived, network blocked in the shell, the HF hub offline during stages. Open: R2-02 (WSL2 routes), R2-11 (wording versus unconfined reads), and R1, which is accepted for trusted use; the stage E task is benign and the repository is disposable.

**Colleague-facing instructions.** Clear, ordered and labelled by window, with sensible guidance to "package anyway". Corrections needed: the stage D row (R2-05); the remedy text for `vram_free` (R2-01); the fallback suggestion on 8 GB cards (R2-14); the redaction promise (R2-11). The maintainer checklist (commit, record id, bundle, check `manifest.json`) is right, and it has not been done yet: the whole package is untracked.

---

## E. Claims checked and found to hold

- The reported results: 112 / 896 + 8 skipped / lock consistent (rerun here).
- `model_visible` matches effGen's observation formatting for the three harness tools (source comparison in §C1).
- The sandbox environment keeps `PATH` and strips everything else sensitive (effGen `_build_env`).
- The preflight cannot print PASS for a check whose prerequisite failed, and a hidden failed root is still listed (`run_checks`).
- No test in `tests/test_rtx4070_package.py` builds a model or calls `from_pretrained` (the AST guard), and the default suite stays CPU-only with no network.
- `results/`, `generated/` and `__pycache__/` are git-ignored inside the package.
- The T5.2b headline numbers match the saved JSON (§C3).

---

## F. Minimal remediation plan

### Before handing the package over

1. **R2-01:** unload the Ollama model at the end of B and confirm with `/api/ps`; rework the D gate; snapshot `ollama ps` per stage.
2. **R2-02:** add the `/mnt/c` write and Windows-interop probes to the `sandbox` check (fail-closed for E).
3. **R2-05:** decide whether D is a mechanics gate or a task gate, and make the README match.
4. **R2-07:** write attempt records incrementally.
5. **R2-11, R2-14, R2-15:** wording and help fixes.
6. Commit the package and the T5.2b files, record the commit id, then build the bundle. Refresh `docs/new-session.md` (R2-13).

### Before relying on the Liquid evidence or re-running it with another model

7. **R2-03, R2-04, R2-08, R2-09:** fix the visibility key, record the trimmed text next to the raw text, fix the RSS attribution, and label B3's inference. Correct the two contradictory plan entries.
8. **R2-06, R2-10:** report distinct trajectories, and add the real-effGen visibility pin.

### Can wait

9. **R2-12:** constrained bnb install, plus a package diff.

---

## G. Final verdict

**Ready for handoff after specific fixes.** The package is carefully engineered and honest about what it has not proven. Its mock-level correctness is confirmed. Two issues should be fixed before a colleague spends an evening on it:

- **R2-01:** a VRAM gate that stage B's own leftover model is likely to trip, ending the run at stage C with a misleading remedy.
- **R2-02:** model-driven shell commands on a personal Windows machine without a check of the two WSL2 routes the Linux-only sandbox verification never saw.

Both are small, package-local changes. The stage D gating rule (R2-05) needs your decision, because it determines whether the coding experiment runs at all.

The T5.2b Liquid conclusions remain valid. The raw-versus-visible discrepancy is real in two places: the tool-result key for successful file operations, and pre-trim generation text. Neither changes a reported success count, but both change how the mechanics and "self-written Observation" figures must be described, and the first would undercount on the next, more capable model.

---

# Remediation status (added 2026-10-17, Sonnet 5.5)

**Everything above this line is the independent review and is unchanged. This section records what was done about it.** Scope of the evidence: mocks, scripted models and the real effGen loop with the real tools **on this Linux laptop only**. Nothing was run on Windows, WSL2 or the RTX 4070; no model was downloaded; no GPU experiment was run; effGen was not modified; nothing was committed. **No Windows/WSL2/RTX 4070 compatibility is claimed or marked verified.**

## Verification of the remediation

| Command | Result |
|---|---|
| `uv run pytest tests/test_rtx4070_package.py -q` | **165 passed** (was 112) |
| `uv run pytest tests/test_evidence_accounting.py -q` (new) | **11 passed** |
| `uv run pytest -q` | **960 passed, 8 skipped** (was 896 + 8) |
| `uv lock --check` | Resolved 157 packages, no dependency change |

Mutation checks (production code broken, intended test must fail, code restored): no release after Ollama stages; no release before C/D/E; `host_isolation` ignoring mounts; stage D requiring task success; `model_visible` returning the raw dict; no stop-sequence cut; PEM pattern removed. Each failed the intended test.

## Status per finding

| ID | Status | What was done | Not verified / residual |
|---|---|---|---|
| R2-01 (P1) | **Fixed locally** | `release_ollama`: for OUR models, `POST /api/generate {"model": tag, "keep_alive": 0}`, then poll `GET /api/ps` until none of ours is listed; never waits for the default keep-alive. Called before stages C, D, E (clean slate, **before** the free-VRAM gate) and after B, D, E, including after a timeout kill or a crash. Records free VRAM before and after and the models released. An HTTP error from `/api/ps`, or a model still listed after the wait, is `verified: false` and **blocks** the next GPU stage with the model's name and `ollama stop` / restart advice; an unreachable server holds no memory and does not block. A model that is not ours is reported, never touched. `vram_free` now names what holds the memory (our model: `ollama stop <name>`; nothing of ours: the Windows-programs advice). New `mixed_vram` check (D, E) from the profile's estimates (our resident model counts as available); stage D then **measures** co-residency (after each Ollama generation made while the Hugging Face model is loaded: `/api/ps` GPU fraction, nvidia-smi, torch allocation; worker parameters all on `cuda`). It is "observed" only if Ollama's model is at least 95 % on the GPU at that moment; otherwise D fails with a resource-limit reason (fallback profile suggested only for the primary profile). Nothing is moved to CPU. | Ollama's behaviour in WSL2 (unload, `/api/ps` fields), the real VRAM figures and whether nvidia-smi under WSL2 includes Windows processes |
| R2-02 (P1) | **Fixed as a refusal gate; stage E is DISABLED on a default WSL2 setup** | New critical check `host_isolation` (stage E only). Refused while any of these exists: a `drvfs`/`9p`/`virtiofs` mount other than `/usr/lib/wsl` and `/mnt/wslg`; WSL interop (binfmt handler enabled or `WSL_INTEROP`); `cmd.exe`/`powershell.exe`/`explorer.exe` on `PATH`; an unreadable `/proc/mounts` on WSL. **Probe results cannot override a known route** (tested). When no route exists, probes run through the real `ShellTool` as a second line: create a marker file in the real Linux home (must fail; the marker is checked from the host and removed), and start `cmd.exe` / `powershell.exe` (attempted only when the driver found no interop route; it never writes to a Windows drive). A missing or `not attempted` probe result is a FAIL. The runner re-checks before any model-run command. The preflight command does not stop stages A to D because of this one blocker; it says stage E will be refused. README "Stage E: isolation requirements" lists the conditions (no routes, a dedicated and disposable distribution because shell **reads** are unconfined, throwaway repository) and the `/etc/wsl.conf` recipe. **Source finding recorded in the audit:** effGen's confinement remounts discard mount errors and its own probe tests only the temp directory, so neither establishes anything about drvfs or interop. | The `/etc/wsl.conf` recipe (including its effect on the GPU), effGen against mounted Windows drives (deliberately never exercised), and whether any default WSL2 colleague will accept stage E being off. Passing the check is evidence for that configuration, not a proof; R1 still applies |
| R2-03 | **Fixed (instrument); not re-run on a real model** | `tests/evidence.py` (`model_visible`, `result_reached_model`, `logged`) replaces the raw-dict comparison in the Liquid test; pinned against effGen's real loop with `file_operations` (success and failure) and `shell` (success and failure). Reported figures were unaffected (7 of 7 executed calls failed). | Not exercised against the real model (needs separate approval) |
| R2-04 | **Fixed (instrument); counts recomputed, not invented** | Generations are now recorded raw **and** consumed (cut at the stop sequences effGen asked for), with `discarded_chars`; token counts are labelled as engine-generated. Recomputed from the SAVED report with effGen's own rule: self-written `Observation:` in **14 of 22** generations raw, **3** after the cut; first generations **8 of 12** raw, **3** after (A 3→1, B 2→2, B2 0→0, B3 3→0). The 3 survivors are inline `Observation:` that no stop sequence catches. Stored texts are truncated at 600 characters (6 of 22 hit it), so these are lower bounds beyond it; consumed tokens cannot be derived and are not claimed. Plan T5.2b has the correction. | Where effGen cut (engine or locally) was not observed (all `finish="stop"`, prompts not stored); the semantics are shown by a scripted run, not by the Liquid model |
| R2-05 | **Fixed (approved decision)** | Stage D PASS = Ollama orchestrator generated, `delegate` executed, Hugging Face worker generated on `cuda`, a worker record came back, the delegation report is in a later orchestrator prompt, co-residency measured. `task_success` recorded separately and shown in `summary.md`; a final answer claiming delegation is not evidence. README row and `RESULTS.md` updated. | |
| R2-06 | **Fixed** | `trajectories {attempts, distinct}` in the Liquid report and every RTX stage; no extra attempts added. | |
| R2-07 | **Fixed** | Each finished attempt is appended (redacted) to `stage_X/attempts.jsonl`; a timeout or crash result states how many were kept and the file goes into the archive (tested). `worker_timeout_s` lowered (mixed 600, Ollama-only 300) and D's limit raised to 3600 s so the worst case fits. | Timeout values are guesses until a real run |
| R2-08 | **Fixed** | Report copy instead of alias in the Liquid test. | The saved report keeps its empty `process_peaks[*].experiments`; documented in the plan |
| R2-09 | **Fixed in code, documented** | B3 now measures worker-result visibility; the plan states the earlier claim was inferred. | Not re-run |
| R2-10 | **Mostly fixed** | Real-loop pins for `file_operations` and `shell`; the runner's copies are pinned equal to the tested helpers. | `delegate`'s visible text is not pinned separately (it is the plain JSON string branch) |
| R2-11 | **Fixed** | PEM private-key blocks (also truncated ones) redacted and detected by the archive scan; README/RESULTS now say known formats are removed, that this is pattern matching, and ask the colleague to inspect the archive. | |
| R2-12 | **Fixed** | `uv pip install --no-deps bitsandbytes==0.50.2` then `uv pip check`; an inconsistent result stops with a restore instruction. | Whether 0.50.2 exists with CUDA 13 kernels is still stage C's question |
| R2-13 | **Partly fixed** | Handoff header, CLI statements, counts and the stale `build_tools` note refreshed; the unrelated root `README.md` (3 bytes, untracked) was **not** deleted: it is the user's file; `new-session.md` and the plan list it | `docs/new-session.md` §2 and §6 contain older prose that was not re-audited line by line |
| R2-14 | **Fixed** | `min_total_vram_mib` per profile (primary 11000, fallback 9000); only a smaller set is ever suggested. | |
| R2-15 | **Fixed** | Help text names `max_tokens` and the seed. | |

## Deferred

- `delegate` visible-text pin (R2-10 remainder): low value, one more test; no separate behaviour to protect.
- Re-running the Liquid experiments to exercise the corrected instruments: needs approval (cost and a model load); the deterministic tests cover the accounting.
- Constrained-resolver alternatives for bitsandbytes (`uv export` constraints): rejected as less reliable than `--no-deps` plus `uv pip check` without a machine to try it on.
- Per-experiment RSS attribution for the already saved report: not recoverable.

## Decisions and risks for the maintainer

1. **Stage E will not run on a default WSL2 colleague machine.** This follows the instruction not to weaken the requirement. The colleague must create a dedicated distribution with the documented `/etc/wsl.conf`, or stage E stays a local-Linux or VM experiment.
2. The `/etc/wsl.conf` recipe disables interop and automounts; its interaction with the WSL GPU libraries is expected to be nil but is **unverified**.
3. The mixed-VRAM threshold and co-residency verdict rest on estimates and on measurements taken on hardware nobody here has.
