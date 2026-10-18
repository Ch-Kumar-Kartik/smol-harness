"""The RTX package on a NATIVE Windows host: supported target, honestly reported.

These tests monkeypatch (`rtx.IS_WINDOWS`, `platform.machine`, `rtx.PROC_VERSION`, ...), so they run on any host — this
machine IS the Windows target. The package's own Linux mock module (`tests/test_rtx4070_package.py`) stays POSIX-only by
design; everything here pins the native-Windows consequences:

- `os` PASSes for native Windows (the package stopped refusing it); `wsl2` SKIPs (no WSL involved);
- `arch` PASSes for AMD64 (Windows's spelling of x86_64) and FAILS for Windows-on-ARM (the locked stack is x86_64 only);
- `repo_location` PASSes (the /mnt trap is WSL-only) while the WSL /mnt rejection keeps working;
- `ram` is measured, not skipped (GlobalMemoryStatusEx);
- `install-deps` adds exactly one step on native Windows (the CUDA build of the SAME pinned torch) and none elsewhere;
- stage E on native Windows runs only under a DECLARED no-confinement scope (WARN), never a silent pass.

What this does not prove: that the GPU experiment itself succeeds — that is stages B..E on real hardware.
"""

from __future__ import annotations

import contextlib
import io
import os
import platform
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

import pytest

PKG = Path(__file__).resolve().parents[1] / "experiments" / "rtx4070"
sys.path.insert(0, str(PKG))
import rtx4070 as rtx  # noqa: E402

# exactly `install-deps`' early filter (cmd_install_deps)
EARLY_IDS = ("os", "wsl2", "arch", "nvidia_smi", "driver_cuda", "uv", "git", "repo_location", "disk")


def windows_host(monkeypatch, machine: str = "AMD64") -> None:
    """The patch that makes any host look like native Windows for the driver: only IS_WINDOWS matters for host_kind()."""
    monkeypatch.setattr(rtx, "IS_WINDOWS", True)
    monkeypatch.setattr(platform, "machine", lambda: machine)


def run_native_checks(monkeypatch, ids=EARLY_IDS):
    windows_host(monkeypatch)
    ctx = rtx.Ctx("primary")
    return {r["id"]: r for r in rtx.run_checks(ctx, phase="pre", stage="A") if r["id"] in ids}


# ---- native Windows: the host is supported, so the checks say supported things ------------------------------------
def test_native_windows_os_passes_and_names_the_host(monkeypatch):
    r = run_native_checks(monkeypatch, ("os",))["os"]
    assert r["status"] == rtx.PASS and "native Windows" in r["detail"] and "no WSL" in r["detail"]


def test_native_windows_omits_the_wsl_checks(monkeypatch):
    r = run_native_checks(monkeypatch, ("wsl2",))["wsl2"]
    assert r["status"] == rtx.SKIP and "no WSL" in r["detail"]


def test_native_windows_amd64_passes_the_architecture_check(monkeypatch):
    r = run_native_checks(monkeypatch, ("arch",))["arch"]
    assert r["status"] == rtx.PASS and "AMD64" in r["detail"]


def test_windows_on_arm_fails_with_the_x86_64_reason(monkeypatch):
    windows_host(monkeypatch, machine="ARM64")
    entry = next(item for item in rtx.CHECKS if item["id"] == "arch")
    status, detail, fix = entry["fn"](rtx.Ctx("primary"))
    assert status == rtx.FAIL and "ARM64" in detail and "x86_64" in fix


def test_native_windows_repo_location_passes_without_claiming_a_linux_filesystem(monkeypatch):
    r = run_native_checks(monkeypatch, ("repo_location",))["repo_location"]
    assert r["status"] == rtx.PASS and "native Windows path" in r["detail"] and "Linux filesystem" not in r["detail"]


def test_early_environment_checks_cannot_block_a_native_windows_run(monkeypatch):
    # os / wsl2 / arch / repo_location used to be invented failures on Windows (AMD64 "failing" x86_64, a Windows path
    # "failing" the /mnt check): with native Windows supported, NONE of them may block.
    results = run_native_checks(monkeypatch, ("os", "wsl2", "arch", "repo_location"))
    blocked = {r["id"] for r in rtx.blocking(list(results.values())) if r["needed_by"] != "E"}
    assert blocked == set(), blocked


def test_native_windows_ram_is_measured_via_the_windows_api(monkeypatch):
    windows_host(monkeypatch)
    monkeypatch.setattr(rtx, "windows_ram_gib", lambda: (32.0, 24.5))
    entry = next(item for item in rtx.CHECKS if item["id"] == "ram")
    status, detail, _ = entry["fn"](rtx.Ctx("primary"))
    assert status == rtx.PASS and "32.0 GiB" in detail
    monkeypatch.setattr(rtx, "windows_ram_gib", lambda: None)
    status, detail, _ = entry["fn"](rtx.Ctx("primary"))
    assert status == rtx.UNVERIFIED and "could not be queried" in detail


@pytest.mark.skipif(os.name != "nt", reason="reads real Windows memory; runs only where memory exists")
def test_windows_ram_api_returns_real_numbers_on_this_machine():
    got = rtx.windows_ram_gib()
    assert got is not None and got[0] > 0 and 0 < got[1] <= got[0]


def test_the_report_says_pass_for_the_host_and_keeps_the_wsl_line_skipped():
    # print_checks over canned rows (no subprocesses), so this holds on every host.
    rows = [
        {"id": "os", "title": "Running on a supported host (native Windows, or Ubuntu inside WSL2)", "phase": "pre", "critical": True, "needed_by": "ABCDE", "status": rtx.PASS, "detail": "native Windows 11 (AMD64); no WSL involvement", "fix": ""},
        {"id": "wsl2", "title": "WSL2 (not WSL1)", "phase": "pre", "critical": True, "needed_by": "ABCDE", "status": rtx.SKIP, "detail": "not applicable: a native Windows run involves no WSL at all (the WSL-specific checks are skipped)", "fix": ""},
        {"id": "arch", "title": "x86_64 CPU architecture", "phase": "pre", "critical": True, "needed_by": "ABCDE", "status": rtx.PASS, "detail": "AMD64 native Windows", "fix": ""},
        {"id": "venv", "title": "Project environment exists (.venv)", "phase": "pre", "critical": True, "needed_by": "ABCDE", "status": rtx.UNVERIFIED, "detail": "the check needs the environment", "fix": ""},
    ]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rtx.print_checks(rows)
    out = buffer.getvalue()
    assert "[PASS] Running on a supported host" in out and "[skip] WSL2 (not WSL1)" in out
    assert "STOP" not in out and "[????]" in out  # the unverified row is shown, nothing blocks


# ---- the WSL2/native-Linux semantics are unchanged by the native-Windows support ----------------------------------
def test_linux_wsl2_host_still_passes_os_and_wsl_checks(monkeypatch, tmp_path):
    monkeypatch.setattr(rtx, "IS_WINDOWS", False)
    version = tmp_path / "proc-version"
    version.write_text("Linux version 5.15.167.4-microsoft-standard-WSL2")
    monkeypatch.setattr(rtx, "PROC_VERSION", version)
    monkeypatch.setattr(rtx.platform, "system", lambda: "Linux")
    monkeypatch.setattr(rtx.platform, "release", lambda: "5.15.167.4-microsoft-standard-WSL2")
    assert rtx.host_kind() == "wsl2"
    entry = next(item for item in rtx.CHECKS if item["id"] == "os")
    status, detail, _ = entry["fn"](rtx.Ctx("primary"))
    assert status == rtx.PASS and "WSL2" in detail


def test_linux_repo_location_still_rejects_mnt_paths(monkeypatch):
    monkeypatch.setattr(rtx, "IS_WINDOWS", False)  # force the non-native branch
    monkeypatch.setattr(rtx.platform, "system", lambda: "Linux")
    # PurePosixPath, not Path: a Windows `Path("/mnt/c/...")` normalises the leading slash away.
    monkeypatch.setattr(rtx, "REPO", PurePosixPath("/mnt/c/projects/smol-harness"))
    ctx = rtx.Ctx("primary")
    entry = next(item for item in rtx.CHECKS if item["id"] == "repo_location")
    status, detail, fix = entry["fn"](ctx)
    assert status == rtx.FAIL and "/mnt/c" in detail and "Linux home" in fix


def test_native_linux_still_gets_the_untested_warning(monkeypatch, tmp_path):
    monkeypatch.setattr(rtx, "IS_WINDOWS", False)
    monkeypatch.setattr(rtx, "PROC_VERSION", tmp_path / "missing")
    monkeypatch.setattr(rtx.platform, "system", lambda: "Linux")
    assert rtx.host_kind() == "native-linux"
    entry = next(item for item in rtx.CHECKS if item["id"] == "wsl2")
    status, detail, _ = entry["fn"](rtx.Ctx("primary"))
    assert status == rtx.WARN and "untested elsewhere" in detail


def test_a_linux_venv_on_windows_host_is_named_venv_of_the_wrong_kind_not_missing(monkeypatch, tmp_path):
    windows_host(monkeypatch)
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("")
    monkeypatch.setattr(rtx, "VENV_PY", tmp_path / ".venv" / "Scripts" / "python.exe")
    monkeypatch.setattr(rtx, "REPO", tmp_path)
    entry = next(item for item in rtx.CHECKS if item["id"] == "venv")
    status, detail, fix = entry["fn"](rtx.Ctx("primary"))
    assert status == rtx.FAIL and "WSL/Linux one" in detail and "Ubuntu" in fix


def test_a_windows_venv_passes_when_it_is_the_native_environment(monkeypatch, tmp_path):
    windows_host(monkeypatch)
    (tmp_path / ".venv" / "Scripts").mkdir(parents=True)
    (tmp_path / ".venv" / "Scripts" / "python.exe").write_text("")
    monkeypatch.setattr(rtx, "VENV_PY", tmp_path / ".venv" / "Scripts" / "python.exe")
    monkeypatch.setattr(rtx, "REPO", tmp_path)
    entry = next(item for item in rtx.CHECKS if item["id"] == "venv")
    status, detail, fix = entry["fn"](rtx.Ctx("primary"))
    assert status == rtx.PASS and "Scripts" in detail


# ---- install-deps: one extra step on native Windows, exactly the CUDA build of the SAME pinned torch ----------------
def test_install_plan_adds_exactly_the_torch_cuda_step_on_native_windows(monkeypatch):
    windows_host(monkeypatch)
    assert rtx.install_plan(rtx.MODELS["profiles"]["primary"]) == ("sync", "torch_cu130", "bitsandbytes", "check")
    assert rtx.install_plan(rtx.MODELS["profiles"]["fallback"]) == ("sync", "torch_cu130")


def test_install_plan_is_unchanged_on_linux(monkeypatch):
    monkeypatch.setattr(rtx, "IS_WINDOWS", False)
    assert rtx.install_plan(rtx.MODELS["profiles"]["primary"]) == ("sync", "bitsandbytes", "check")
    assert rtx.install_plan(rtx.MODELS["profiles"]["fallback"]) == ("sync",)


def test_current_torch_build_reads_the_full_build_tag(monkeypatch):
    monkeypatch.setattr(rtx, "run", lambda *a, **k: (0, f"{rtx.TORCH_PIN}\n", ""))
    assert rtx.current_torch_build() == rtx.TORCH_PIN
    monkeypatch.setattr(rtx, "run", lambda *a, **k: (1, "", "no torch"))
    assert rtx.current_torch_build() is None


def test_the_windows_torch_pin_is_the_locked_stack_version_with_the_cuda_build():
    assert rtx.TORCH_PIN == "torch==2.14.1+cu130"
    locked = rtx.MODELS["locked_stack"]["torch"]
    assert "2.14.1" in locked and "CUDA 13.0" in locked


# ---- the stage-E gate: declared scope on native Windows, refusals everywhere else ---------------------------------
sys.path.insert(0, str(PKG))
import stage_runner as runner  # noqa: E402


def test_stage_e_gate_accepts_only_the_declared_native_windows_warn():
    ok = {"id": "sandbox", "status": rtx.PASS}
    isolation_warn = {"id": "host_isolation", "status": rtx.WARN}
    isolation_fail = {"id": "host_isolation", "status": rtx.FAIL}
    assert runner.stage_e_gate([ok, isolation_warn], native_windows=True) == []
    # a WARN is NOT a silent pass anywhere else, and FAILs refuse on every host:
    assert runner.stage_e_gate([{"id": "sandbox", "status": rtx.WARN}], native_windows=True) != []
    assert runner.stage_e_gate([ok, isolation_fail], native_windows=True) == [isolation_fail]
    assert runner.stage_e_gate([ok, isolation_warn], native_windows=False) == [isolation_warn]


# ---- the Windows Job Object tree kill (real processes; runs only where jobs exist) ---------------------------------
@pytest.mark.skipif(os.name != "nt", reason="windows job objects")
def test_stage_job_kills_the_whole_tree_including_detached_grandchildren(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    grandchild_pid = tmp_path / "grandchild.pid"
    grandchild_script = tmp_path / "gc.py"
    grandchild_script.write_text("import os, pathlib, time\n"
                                 f"pathlib.Path({str(grandchild_pid)!r}).write_text(str(os.getpid()))\n"
                                 "time.sleep(120)\n")
    child_script = tmp_path / "child.py"
    child_script.write_text("import os, pathlib, subprocess, sys, time\n"
                            f"pathlib.Path({str(child_pid_file)!r}).write_text(str(os.getpid()))\n"
                            "subprocess.Popen([sys.executable, %r], creationflags=0x00000008)  # DETACHED, adversarial\n"
                            "time.sleep(120)\n" % str(grandchild_script))
    # the exact command transform run_stage applies: VENV_PY (a uv trampoline on Windows) is replaced by the real
    # interpreter, because the trampoline's own Job lets every grandchild silently break away and survive the kill
    command = [str(sys.executable), str(child_script)]
    boot = rtx.stage_boot()
    if boot:
        assert rtx.stage_interpreter().lower() != sys.executable.lower(), "expected a uv trampoline here, not the real interpreter"
        command = [rtx.stage_interpreter(), "-c", boot, *command[1:]]
    process, job = rtx._spawn_stage(command, tmp_path, os.environ)
    try:
        deadline = time.time() + 10
        while not grandchild_pid.exists() and time.time() < deadline:
            time.sleep(0.1)
        assert grandchild_pid.exists(), "the detached grandchild never started"
        grandchild = grandchild_pid.read_text().strip()
        job.kill()
        process.wait(timeout=15)
        time.sleep(1.5)
        rc, out, _ = rtx.run(["tasklist", "/FI", f"PID eq {grandchild}", "/NH"], timeout=20)
        assert "INFO:" in out or grandchild not in out, f"the detached grandchild {grandchild} survived the Job kill"
    finally:
        job.close()


# ---- the stage-E prose an operator reads before the gate -----------------------------------------------------------
def test_native_windows_isolation_check_answers_from_the_probe_not_from_wsl_assumptions(monkeypatch):
    windows_host(monkeypatch)
    probe = {"isolation": {"native_no_confinement": {"escaped": True, "what": "native Windows: no confinement by design"}}}
    monkeypatch.setattr(rtx.Ctx, "probe", lambda self: probe)
    entry = next(item for item in rtx.CHECKS if item["id"] == "host_isolation")
    status, detail, _ = entry["fn"](rtx.Ctx("primary"))
    assert status == rtx.WARN and "NO filesystem or network confinement" in detail and "trusted" in detail
    missing = {"isolation_error": "probe did not run"}
    monkeypatch.setattr(rtx.Ctx, "probe", lambda self: missing)
    status, detail, fix = entry["fn"](rtx.Ctx("primary"))
    assert status == rtx.FAIL and "did not run" in detail
