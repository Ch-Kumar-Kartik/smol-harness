# Review 003: `windows-fix` integration review (NOT merged)

_2026-10-22. Reviewer: Claude (Opus 5.5). Decision by the user: do not merge; return the branch to its author._

| | |
|---|---|
| Source | `origin/windows-fix` @ `f28ac4deae1a726cea6b12baac8d6e5468aa2fb0` (1 commit, "vibe coded windows compatiblity fix", author `HelperAccount`, 2026-10-18) |
| Target | `main` @ `60ffb1a55883859679343d151921687b37bf6024` (= `origin/main`) |
| Merge base | `60ffb1a` (a fast-forward was possible; it was not done) |
| Diff | 20 files, +1940 / -88. `uv.lock` and `pyproject.toml` unchanged |

## Findings

### P0-1: native Windows silently gets an unconfined shell (fail-closed contract reversed)

`src/smol_harness/tools.py` `_shell_tool_class()` / `_build_shell` (branch lines ~373-395) route `shell` to the new `src/smol_harness/shell_win.py::WindowsShellTool` whenever `platform.system() == "Windows"`. That backend runs PowerShell with **no write confinement, no network isolation and no read confinement** (its own docstring and result fields say so). The only guard is the tool description asking the model to write inside the project. There is no opt-in: an existing config with `tools: [shell]` that failed closed on Windows now runs.

Reproduction (`build_tools(["shell"], ws)` with `platform.system` mocked to `"Windows"` and PowerShell on PATH; nothing is executed):

- `main` 60ffb1a: `ToolBuildError: tool 'shell' needs Linux (found 'Windows') ...` (fail closed, CLI exit 3)
- branch f28ac4d: builds `smol_harness.shell_win.WindowsShellTool`

This violates AGENTS.md ("Never fall back silently ... from the sandboxed shell to an unsandboxed one"), CLAUDE.md (fail closed if the sandbox is unavailable), and the 2026-10-17 accepted scope. That decision accepted the Linux sandbox *with* write and network confinement and the residual risk R1. It did not accept an unconfined shell. The branch docs (`docs/new-session.md`, `docs/architecture.md` §9) present it as "the same trusted-repositories decision", which misstates that record. Impact: Windows only; Linux behaviour is unchanged.

Smallest fix: keep `shell` Linux-only. `_REGISTRY["shell"]` stays the Linux class, which refuses off Linux. An unconfined Windows shell would need an explicit, separately approved opt-in design.

### P0-2: RTX stage E runs model-issued commands unconfined on native Windows

`experiments/rtx4070/stage_runner.py::stage_e_gate` accepts a `host_isolation` WARN on native Windows. `rtx4070.py::isolation()` returns a fixed `native_no_confinement` marker instead of probing, and `chk_host_isolation` turns it into that WARN. This reverses review 002 R2-02 (stage E is refused whenever the Windows host is reachable) for the case where the host is fully reachable.

Smallest fix: on native Windows, `host_isolation` FAILs and stage E is refused. Stages A to D do not use the shell; only stage E's worker gets `[file_operations, shell]`.

### P1-1: new module outside architecture §3, without approval

`src/smol_harness/shell_win.py` (535 lines of ctypes Win32 Job Object code) is a seventh harness module. The hard constraint says no new modules without approval. `rtx4070.py::StageTreeJob` duplicates the same ctypes code.

### P1-2 (plausible, unverified): the Windows kill-tree guarantee contradicts the author's own measurement

The branch's `new-session.md` entry records that uv's `.venv\Scripts\python.exe` trampoline lets grandchildren of the real interpreter break away from all Jobs. The RTX driver works around this (`stage_interpreter`/`stage_boot`), but `WindowsShellTool` does not. A model command such as `python -m pytest` (with `.venv\Scripts` on PATH) is exactly that case. `tests/test_shell_win.py` has no test for it. Not reproducible here (no Windows host).

### P2-1: the fail-closed regression pins were rewritten instead of kept

The following tests no longer pin "Windows refuses `shell`" at the `build_tools` / harness / CLI level:

- `tests/test_tools.py::test_shell_is_refused_off_linux` now calls `_LinuxShellTool` directly and is `linux_only`.
- `tests/test_harness_config.py::test_an_unavailable_shell_fails_before_any_model_is_loaded` and `tests/test_cli.py::test_an_unavailable_shell_is_a_startup_failure_before_any_model_loads` now mock `shutil.which` instead of the platform.

The suite therefore stays green while P0-1 happens.

### P2-2: stray or personal artifacts committed

- `results/preflight-windows-smoke.json` sits at the repo root, which is not ignored (only `experiments/rtx4070/results/` is). It contains the Windows username and machine details.
- `workspace/hello.py` is a scratch file.
- `configs/qwen3-pair.yaml` is untested and undocumented, and it points its shell worker at `../workspace` inside the repo.

### P2-3: Windows `install-deps` installs torch outside the lock

`rtx4070.py::cmd_install_deps` runs `uv pip install torch==2.14.1+cu130 --index-url download.pytorch.org --extra-index-url pypi --index-strategy unsafe-best-match`. The install is unhashed and unlocked, and `unsafe-best-match` mixes both indexes for torch's dependencies. Any later `uv sync`/`uv run` reverts it (documented by the author). This affects the experiment only, not the harness.

### P3

- `experiments/rtx4070/TROUBLESHOOTING.md` has a duplicated `## Sandbox` heading.
- The branch's handoff reports Windows results (730 passed / 289 skipped; `test_shell_win.py` 33 passed) that could not be reproduced here.

### Not findings (checked)

The Linux paths are behaviour-preserving on read and in the tests:

- `ShellTool` resolves to `_LinuxShellTool` via PEP 562, and every consumer works.
- `WorkspaceFileOperations._restore_translated_write` is a no-op off `nt`.
- The `as_posix()` summary is identical on Linux.
- In the RTX driver the POSIX branches are unchanged (`killpg`, `os.pathsep` == `:`).
- The `tempfile.gettempdir()` report paths and the `_euid`/symlink skips are harmless portability changes.

## Tests (Linux, this laptop; `uv run python -m pytest`)

`uv run pytest` fails to spawn here because `.venv/bin/pytest` has a stale shebang (the repo moved from `Projects/Active/`). This is an environment issue, unrelated to the branch.

| Run | main 60ffb1a | branch f28ac4d |
|---|---|---|
| `uv lock --check` | OK (157) | OK (157) |
| full suite | 972 passed, 8 skipped | 991 passed, 43 skipped |
| `tests/test_tools.py` | 135 collected | 135 passed |
| `tests/test_harness_config.py` | | 46 passed |
| `tests/test_e2e_coding.py` | | 11 passed (real sandbox, git) |
| `tests/test_cli.py` | | 48 passed |
| `tests/test_rtx4070_package.py` | | 177 passed (mocks) |
| `tests/test_rtx4070_preflight_windows.py` (new) | | 19 passed, 2 skipped (Windows-only) |
| `tests/test_shell_win.py` (new) | | 33 skipped (Windows-only) |

Skips on the branch: 8 opt-in real-model tests, all 33 Windows-shell tests and 2 Windows-only preflight tests. **None of the Windows backend code was executed in this review.**

## Status

Not merged. `main` is unchanged at `60ffb1a`. The branch is kept as the remote-tracking ref `origin/windows-fix`; no local branch was created, nothing was pushed or deleted. A resubmission should drop P0-1/P0-2 (or bring an approved opt-in design), remove the P2-2 artifacts, restore the P2-1 pins, and keep the Linux-neutral portability changes.

## Follow-up (2026-10-22, after this review)

The decision above stands: `origin/windows-fix` was not merged. Only the Linux-neutral portability fixes were re-applied by hand on `integrate/windows-portability` (independently reviewed: no P0/P1; one P2 in the new symlink test gate, since fixed). The findings above are unchanged; the "no local branch was created" sentence describes the state when this review was written.
