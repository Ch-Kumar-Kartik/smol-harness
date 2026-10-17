"""Local validation of experiments/rtx4070 (the Windows 11 / RTX 4070 handoff package).

WHAT THIS PROVES: that the package's scripts are syntactically valid, parse their arguments, fail clearly and stop dependent stages when a critical
requirement is missing, never substitute CPU for the GPU, render configurations the project's own loader accepts, redact secrets, and write complete,
correct results directories and archives.

WHAT THIS DOES NOT PROVE: anything about the target hardware. WSL2, the NVIDIA driver, Ollama's real behaviour, bitsandbytes on CUDA 13, VRAM use and
real model quality are all UNVERIFIED until the RTX 4070 machine runs the package. Every GPU, Ollama and WSL fact below is a controlled MOCK.
No weights are downloaded, no Ollama is installed or started, no GPU experiment is run.
"""

from __future__ import annotations

import ast
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from collections import namedtuple
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

PKG = Path(__file__).resolve().parents[1] / "experiments" / "rtx4070"
sys.path.insert(0, str(PKG))
import rtx4070 as rtx  # noqa: E402
import stage_runner as sr  # noqa: E402

GOOD_SMI_QUERY = "NVIDIA GeForce RTX 4070, 12282, 900, 11200, 581.42\n"
GOOD_SMI_HEADER = "| NVIDIA-SMI 581.42   Driver Version: 581.42   CUDA Version: 13.0 |\n"
WSL2_VERSION = "Linux version 5.15.167.4-microsoft-standard-WSL2 (root@build) (gcc) #1 SMP"
GOOD_PROBE = {"python": "3.12.13", "effgen": "1.3.0", "transformers": "5.20.0", "smol_harness": "/x/smol_harness/__init__.py", "bitsandbytes": "0.50.2",
              "torch": {"version": "2.14.1", "cuda_build": "13.0", "available": True, "name": "NVIDIA GeForce RTX 4070", "capability": [8, 9], "total_mib": 12282,
                        "free_mib": 11200, "bf16": True, "matmul_ok": True},
              "sandbox": {"success": True, "stdout": "sandbox-ok\n", "error": None},
              "isolation": {"write_outside_workspace": {"escaped": False, "what": "w"}, "cmd_exe": {"escaped": False, "what": "w"}, "powershell_exe": {"escaped": False, "what": "w"}}}
Usage = namedtuple("Usage", "total used free")
ORIG_PROBE = rtx.Ctx.probe  # the real method, kept before the host fixture replaces it


class Host:
    """A controllable fake of everything the checks look at. Mocks only: nothing here is evidence about real hardware."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.smi_query, self.smi_header, self.smi_rc = GOOD_SMI_QUERY, GOOD_SMI_HEADER, 0
        self.free_gib, self.proc_version = 80, WSL2_VERSION
        self.has_uv, self.has_git, self.ollama_process = True, True, True
        self.probe = json.loads(json.dumps(GOOD_PROBE))
        self.ollama_up = True
        spec = rtx.MODELS["profiles"]["primary"]["ollama"]
        self.tags = {spec["base_model"]: {"name": spec["base_model"], "digest": spec["manifest_digest"]}, spec["derived_tag"]: {"name": spec["derived_tag"], "digest": "d" * 64}}
        self.show_params = f"num_ctx {spec['num_ctx']}\nnum_predict {spec['num_predict']}"
        self.hf = ("/cache/snap", "")
        # Ollama residency (review 002, R2-01). `flag` is a file a stub stage runner writes to say "I left our model loaded"; `explicit` lists other loaded models.
        self.flag: Path | None = None
        self.explicit: list = []
        self.release_works, self.ps_status, self.http_log = True, 200, []

    def resident_models(self) -> list:
        tag = rtx.MODELS["profiles"]["primary"]["ollama"]["derived_tag"]
        loaded = [{"name": tag, "model": tag, "size": 5_700_000_000, "size_vram": 5_700_000_000}] if self.flag and self.flag.exists() else []
        return loaded + list(self.explicit)

    def footprint_mib(self) -> int:
        return sum(m["size_vram"] for m in self.resident_models()) // 2**20

    def run(self, cmd, timeout=30, env=None, cwd=None):
        cmd = [str(c) for c in cmd]
        if cmd[0] == "nvidia-smi":
            if not any(c.startswith("--query-gpu") for c in cmd):
                return self.smi_rc, self.smi_header, ""
            name, total, used, free, driver = [p.strip() for p in self.smi_query.strip().split(",")]
            fp = self.footprint_mib()  # what Ollama holds shows up as used memory
            return self.smi_rc, f"{name}, {total}, {int(float(used)) + fp}, {int(float(free)) - fp}, {driver}\n", ""
        if cmd[0] == "uv":
            return (0, "uv 0.11.29\n", "") if self.has_uv else (127, "", "uv: command not found")
        if cmd[0] == "git":
            if not self.has_git:
                return 127, "", "git: command not found"
            if "rev-parse" in cmd and "HEAD" in cmd:
                return 0, "a" * 40 + "\n", ""
            if "--abbrev-ref" in cmd:
                return 0, "main\n", ""
            return 0, "", ""
        if cmd[0] == "pgrep":
            return (0, "123\n", "") if self.ollama_process else (1, "", "")
        if cmd[0] == "ollama":
            return 0, "ollama version is 0.9.0\n", ""
        return 127, "", f"{cmd[0]}: not mocked"

    def http(self, method, path, body=None, timeout=10):
        self.http_log.append((method, path, body))
        if not self.ollama_up:
            return None, None, "Connection refused"
        if path == "/api/generate":  # an explicit unload request (keep_alive 0) releases the model, unless the test says the server ignores it
            if (body or {}).get("keep_alive") == 0 and self.release_works:
                if self.flag and self.flag.exists():
                    self.flag.unlink()
                self.explicit = [m for m in self.explicit if m["name"] != body.get("model")]
            return 200, {"response": "", "done": True}, ""
        if path == "/api/version":
            return 200, {"version": "0.9.0"}, ""
        if path == "/api/tags":
            return 200, {"models": list(self.tags.values())}, ""
        if path == "/api/show":
            if (body or {}).get("model") not in self.tags:
                return 404, None, "HTTP 404: model not found"
            return 200, {"parameters": self.show_params, "details": {"family": "qwen2"}}, ""
        if path == "/api/ps":
            if self.ps_status != 200:
                return self.ps_status, None, f"HTTP {self.ps_status}: not found"
            return 200, {"models": self.resident_models()}, ""
        return 404, None, "HTTP 404"


@pytest.fixture
def host(tmp_path, monkeypatch):
    h = Host(tmp_path)
    (tmp_path / "version").write_text(h.proc_version)
    (tmp_path / "dxg").write_text("")
    (tmp_path / "meminfo").write_text("MemTotal:       16777216 kB\nMemAvailable:   12582912 kB\n")
    (tmp_path / "mounts").write_text("/dev/sdc / ext4 rw 0 0\ntmpfs /run tmpfs rw 0 0\n/usr/lib/wsl/lib /usr/lib/wsl/lib 9p ro 0 0\ntmpfs /mnt/wslg tmpfs rw 0 0\n")
    repo = tmp_path / "home" / "smol-harness"
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").write_text("")
    monkeypatch.setattr(rtx, "PROC_VERSION", tmp_path / "version")
    monkeypatch.setattr(rtx, "DEV_DXG", tmp_path / "dxg")
    monkeypatch.setattr(rtx, "MEMINFO", tmp_path / "meminfo")
    monkeypatch.setattr(rtx, "PROC_MOUNTS", tmp_path / "mounts")
    (tmp_path / "binfmt_misc").mkdir()  # interop off: no WSLInterop* registration in the directory
    monkeypatch.setattr(rtx, "BINFMT_DIR", tmp_path / "binfmt_misc")
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    real_which = shutil.which
    monkeypatch.setattr(rtx.shutil, "which", lambda name, *a, **k: None if name.endswith(".exe") else real_which(name, *a, **k))
    monkeypatch.setattr(rtx, "RELEASE_WAIT_S", 0.2)  # a model that stays loaded must not make a test wait a minute
    h.flag = tmp_path / "results" / "ollama_loaded.flag"
    monkeypatch.setattr(rtx, "REPO", repo)
    monkeypatch.setattr(rtx, "VENV_PY", repo / ".venv" / "bin" / "python")
    monkeypatch.setattr(rtx, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(rtx, "GENERATED", tmp_path / "generated")
    monkeypatch.setattr(rtx, "run", lambda *a, **k: h.run(*a, **k))
    monkeypatch.setattr(rtx, "http_json", lambda *a, **k: h.http(*a, **k))
    monkeypatch.setattr(rtx.Ctx, "probe", lambda self: h.probe)
    monkeypatch.setattr(rtx, "hf_snapshot_path", lambda profile: h.hf)
    monkeypatch.setattr(rtx.platform, "system", lambda: "Linux")
    monkeypatch.setattr(rtx.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(rtx.shutil, "disk_usage", lambda p: Usage(500 * 2**30, 0, int(h.free_gib * 2**30)))
    return h


def apply_text(host: Host) -> None:
    (host.tmp / "version").write_text(host.proc_version)


def by_id(results: list) -> dict:
    return {r["id"]: r for r in results}


def preflight(profile="primary", **kw):
    return by_id(rtx.run_checks(rtx.Ctx(profile), **kw))


# ====================================================================================================== files and static validity
def test_the_package_is_small_and_has_the_files_a_colleague_needs():
    names = sorted(p.name for p in PKG.iterdir() if p.is_file())
    assert {"README.md", "TROUBLESHOOTING.md", "RESULTS.md", "rtx4070.py", "stage_runner.py", "models.json", ".gitignore"} <= set(names)
    assert sorted(p.name for p in (PKG / "configs").iterdir()) == ["mixed.yaml", "ollama_only.yaml"]


@pytest.mark.parametrize("name", ["rtx4070.py", "stage_runner.py"])
def test_scripts_are_valid_python_down_to_3_9(name):
    # Ubuntu 22.04 ships Python 3.10; the driver must run on a bare system Python before any environment exists.
    source = (PKG / name).read_text()
    ast.parse(source, feature_version=(3, 9))
    compile(source, name, "exec")


def test_the_driver_is_standard_library_only():
    tree = ast.parse((PKG / "rtx4070.py").read_text())
    imported = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)} | {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert imported <= set(sys.stdlib_module_names), sorted(imported - set(sys.stdlib_module_names))


def test_model_pins_are_complete_and_consistent():
    models = rtx.MODELS
    assert models["bitsandbytes"]["pin"] and models["locked_stack"]["effgen"] == "1.3.0"
    for name, profile in models["profiles"].items():
        o, h = profile["ollama"], profile["hf"]
        assert len(o["manifest_digest"]) == 64 and len(h["revision"]) == 40 and o["download_bytes"] > 0 and h["safetensors_bytes"] > 1e9
        assert o["derived_tag"].split(":")[0] not in {"server", "openai", "compatible", "local_server"}  # effGen would strip such a prefix (models.py)
        est = profile["estimated_vram_gib"]
        assert abs(sum(v for k, v in est.items() if k not in ("total", "of")) - est["total"]) < 0.05, name  # the estimate adds up
        assert est["total"] < est["of"]


@pytest.mark.parametrize("profile_name", ["primary", "fallback"])
def test_rendered_configs_pass_the_projects_own_loader_and_always_demand_cuda(profile_name, tmp_path, monkeypatch):
    from smol_harness.config import loads_config
    from smol_harness.models import OLLAMA_ID_ALIAS_PREFIXES, loader_args

    ctx = rtx.Ctx(profile_name, tmp_path)
    profile = ctx.profile
    for template, extra in (("ollama_only", {}), ("mixed", {"__HF_MODEL__": "/cache/snap"})):
        path = rtx.render_config(template, tmp_path, ctx, name="t", __WORKSPACE__=str(tmp_path), __WORKER_NAME__="w", __WORKER_DESCRIPTION__="d", __WORKER_TOOLS__="[file_operations]", **extra)
        config = loads_config(path.read_text())
        assert config.orchestrator.model.context_length == profile["ollama"]["num_ctx"] == 8192
        assert config.orchestrator.model.model.split(":")[0].lower() not in OLLAMA_ID_ALIAS_PREFIXES
    worker = config.workers[0].model  # the mixed config
    assert (worker.provider, worker.device, worker.quantization) == ("hf", "cuda", profile["hf"]["quantization"])
    _, kwargs = loader_args(worker)
    assert kwargs["require_gpu"] is True and "device_map" not in kwargs  # the CUDA path of the model factory, never the CPU one
    assert kwargs["quantization_bits"] == (4 if profile_name == "primary" else None)
    # Inspect the PARSED configuration (comments may say "never on CPU"): the worker's device is cuda, and no configuration VALUE anywhere mentions cpu.
    import yaml

    parsed = yaml.safe_load(path.read_text())
    assert parsed["workers"][0]["model"]["device"] == "cuda"

    def scalars(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield str(key)
                yield from scalars(value)
        elif isinstance(node, list):
            for value in node:
                yield from scalars(value)
        else:
            yield str(node)

    assert not [s for s in scalars(parsed) if "cpu" in s.lower()]
    assert yaml.safe_load((PKG / "configs" / "mixed.yaml").read_text().replace("__WORKER_TOOLS__", "[]"))["workers"][0]["model"]["device"] == "cuda"  # fixed in the template, not a placeholder


def test_an_unfilled_placeholder_is_an_error_not_a_broken_config(tmp_path):
    ctx = rtx.Ctx("primary", tmp_path)
    with pytest.raises(ValueError, match="__WORKSPACE__|__WORKER"):
        rtx.render_config("mixed", tmp_path, ctx, __HF_MODEL__="/x")


# ====================================================================================================== argument parsing
def test_help_and_argument_parsing_in_a_real_process():
    ok = subprocess.run([sys.executable, str(PKG / "rtx4070.py"), "--help"], capture_output=True, text=True)
    assert ok.returncode == 0 and "preflight" in ok.stdout and "package" in ok.stdout
    bad = subprocess.run([sys.executable, str(PKG / "rtx4070.py"), "stage", "Z"], capture_output=True, text=True)
    assert bad.returncode == 2 and "invalid choice" in bad.stderr
    none = subprocess.run([sys.executable, str(PKG / "rtx4070.py")], capture_output=True, text=True)
    assert none.returncode == 2


@pytest.mark.parametrize("argv,profile", [(["--profile", "fallback", "preflight"], "fallback"), (["preflight", "--profile", "fallback"], "fallback"), (["preflight"], "primary")])
def test_profile_is_accepted_before_or_after_the_command(argv, profile):
    assert rtx.build_parser().parse_args(argv).profile == profile


@pytest.mark.parametrize("argv", [["stage", "B", "--attempts", "0"], ["stage", "B", "--attempts", "9"], ["preflight", "--phase", "never"], ["prepare", "--profile", "huge"]])
def test_out_of_range_arguments_are_rejected(argv):
    with pytest.raises(SystemExit) as err:
        rtx.build_parser().parse_args(argv)
    assert err.value.code == 2


# ====================================================================================================== preflight: healthy and negative paths
def test_a_healthy_machine_has_no_blocking_failures(host):
    results = rtx.run_checks(rtx.Ctx("primary"), phase="pre")
    assert rtx.blocking(results) == []
    states = by_id(results)
    assert states["wsl2"]["status"] == rtx.PASS and states["gpu_identity"]["status"] == rtx.PASS and states["torch_cuda"]["status"] == rtx.PASS
    assert states["python3"]["status"] == rtx.PASS
    assert states["bitsandbytes"]["status"] == rtx.UNVERIFIED  # imports, but CUDA 13 kernels can only be proven by stage C


@pytest.mark.parametrize("change,check_id,expected_fix", [
    (lambda h, m: m.setattr(rtx.platform, "system", lambda: "Windows"), "os", "Ubuntu"),
    (lambda h, m: (setattr(h, "proc_version", "Linux version 4.4.0-19041-Microsoft"), apply_text(h)), "wsl2", "wsl --set-version"),
    (lambda h, m: m.setattr(rtx, "DEV_DXG", h.tmp / "missing-dxg"), "gpu_node", "wsl --update"),
    (lambda h, m: setattr(h, "smi_rc", 127), "nvidia_smi", "Do NOT install a Linux NVIDIA driver"),
    (lambda h, m: setattr(h, "smi_query", "NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 500, 7600, 581.42\n"), "gpu_vram", "too small"),
    (lambda h, m: setattr(h, "smi_header", "CUDA Version: 12.4"), "driver_cuda", "580"),
    (lambda h, m: setattr(h, "free_gib", 12), "disk", "free"),
    (lambda h, m: (h.tmp / "meminfo").write_text("MemTotal: 4194304 kB\nMemAvailable: 3000000 kB\n"), "ram", ".wslconfig"),
    (lambda h, m: m.setattr(rtx, "REPO", Path("/mnt/c/Users/x/smol-harness")), "repo_location", "Linux home"),
    (lambda h, m: setattr(h, "has_git", False), "git", "apt install"),
])
def test_each_missing_requirement_is_a_blocking_failure_with_an_instruction(host, monkeypatch, change, check_id, expected_fix):
    change(host, monkeypatch)
    state = preflight(phase="pre")[check_id]
    assert state["status"] == rtx.FAIL and state["critical"], state
    assert expected_fix in state["fix"] or expected_fix in state["detail"]


def test_a_gpu_that_is_not_a_4070_is_a_warning_not_a_stop(host):
    host.smi_query = "NVIDIA GeForce RTX 3090, 24576, 900, 23000, 581.42\n"
    state = preflight(phase="pre")
    assert state["gpu_identity"]["status"] == rtx.WARN and state["gpu_vram"]["status"] == rtx.PASS


def test_an_unreadable_driver_version_is_unverified_not_assumed_fine(host):
    host.smi_header = "no cuda line here"
    assert preflight(phase="pre")["driver_cuda"]["status"] == rtx.UNVERIFIED


def test_uv_is_only_required_when_the_environment_is_missing(host):
    host.has_uv = False
    assert preflight(phase="pre")["uv"]["status"] == rtx.WARN  # environment exists: uv is only needed to install
    shutil.rmtree(rtx.REPO / ".venv")
    assert preflight(phase="pre")["uv"]["status"] == rtx.FAIL


# THE CONTRACT: one cause, one failure. A check that cannot say anything true because a check it requires FAILED is UNVERIFIED and names that root;
# it is never a second FAIL and never a PASS. The root stays a blocking FAIL, so such a machine can never get a passing preflight.
DEPENDENCY_CASES = [
    # (how to break the machine, root check, checks that must be reported as blocked by it)
    ("ollama_down", "ollama_server", {"ollama_models", "ollama_ctx"}),
    ("no_nvidia_smi", "nvidia_smi", {"gpu_identity", "gpu_vram", "driver_cuda", "vram_free"}),
    ("no_gpu_node", "gpu_node", {"nvidia_smi", "gpu_identity", "gpu_vram", "driver_cuda", "vram_free"}),  # transitive: the ROOT is named, not the middle link
    ("no_venv", "venv", {"deps", "torch_cuda", "bitsandbytes", "sandbox", "hf_snapshot"}),
    ("wrong_effgen", "deps", {"sandbox"}),
]


def break_machine(host, monkeypatch, how: str) -> None:
    if how == "ollama_down":
        host.ollama_up = False
    elif how == "no_nvidia_smi":
        host.smi_rc = 127
    elif how == "no_gpu_node":
        monkeypatch.setattr(rtx, "DEV_DXG", host.tmp / "missing-dxg")
        host.smi_rc = 127
    elif how == "no_venv":
        shutil.rmtree(rtx.REPO / ".venv")
        host.probe = {"error": "the project environment does not exist"}
        host.hf = (None, "the project environment does not exist")
    elif how == "wrong_effgen":
        host.probe["effgen"] = "1.2.0"


@pytest.mark.parametrize("how,root,blocked", DEPENDENCY_CASES, ids=[c[0] for c in DEPENDENCY_CASES])
def test_a_root_cause_is_reported_once_and_its_dependents_are_unverified_not_failed(host, monkeypatch, how, root, blocked):
    break_machine(host, monkeypatch, how)
    results = rtx.run_checks(rtx.Ctx("primary"))  # both phases
    states = by_id(results)
    assert [r["id"] for r in rtx.blocking(results)] == [root]  # exactly one blocking failure: the cause
    for dependent in blocked:
        state = states[dependent]
        assert state["status"] == rtx.UNVERIFIED and state["blocked_by"] == root and f"'{root}' failed first" in state["detail"], (dependent, state)
    assert not any(r["status"] == rtx.PASS for r in results if r.get("blocked_by"))  # blocked is never "passing"


@pytest.mark.parametrize("how,root", [("ollama_down", "ollama_server"), ("no_nvidia_smi", "nvidia_smi"), ("no_venv", "venv")])
def test_a_preflight_with_a_failed_root_cause_never_passes(host, monkeypatch, capsys, how, root):
    break_machine(host, monkeypatch, how)
    assert rtx.main(["preflight", "--phase", "all"]) == 1
    out = capsys.readouterr().out
    assert "STOP: fix the FAIL items" in out and "not checkable until a failure above is fixed" in out


def test_a_failed_root_is_shown_even_when_the_phase_filter_would_hide_it(host):
    host.ollama_up = False  # ollama_server is a pre-phase check; ask for the post phase only
    results = rtx.run_checks(rtx.Ctx("primary"), phase="post")
    assert results[0]["id"] == "ollama_server" and results[0]["status"] == rtx.FAIL  # the cause is listed first
    assert {r["id"]: r["status"] for r in results}["ollama_models"] == rtx.UNVERIFIED


def test_unrelated_failures_are_still_all_reported(host):
    host.ollama_up, host.free_gib = False, 10  # two independent causes
    assert sorted(r["id"] for r in rtx.blocking(rtx.run_checks(rtx.Ctx("primary")))) == ["disk", "ollama_server"]


def test_a_missing_environment_is_one_clear_failure_not_a_cascade(host):
    break_machine(host, None, "no_venv")
    states = preflight(phase="pre")
    assert states["venv"]["status"] == rtx.FAIL and "install-deps" in states["venv"]["fix"]
    for dependent in ("deps", "torch_cuda", "bitsandbytes", "sandbox"):
        assert states[dependent]["status"] == rtx.UNVERIFIED


def test_wrong_effgen_version_is_blocking(host):
    host.probe["effgen"] = "1.2.0"
    assert preflight(phase="pre")["deps"]["status"] == rtx.FAIL
    assert preflight(phase="pre")["torch_cuda"]["status"] == rtx.PASS  # independent of the effGen version: still checked


def test_torch_without_cuda_is_blocking_and_never_suggests_cpu(host):
    host.probe["torch"] = {"version": "2.14.1", "cuda_build": "13.0", "available": False}
    state = preflight(phase="pre")["torch_cuda"]
    assert state["status"] == rtx.FAIL and "CPU" in state["fix"] and "do not continue" in state["fix"].lower()


def test_a_gpu_that_cannot_compute_is_blocking(host):
    host.probe["torch"]["matmul_ok"] = False
    assert preflight(phase="pre")["torch_cuda"]["status"] == rtx.FAIL


def test_bitsandbytes_is_required_for_primary_and_not_for_fallback(host):
    del host.probe["bitsandbytes"]
    host.probe["bitsandbytes_error"] = "ModuleNotFoundError: No module named 'bitsandbytes'"
    assert preflight("primary", phase="pre")["bitsandbytes"]["status"] == rtx.FAIL
    fallback = preflight("fallback", phase="pre")["bitsandbytes"]
    assert fallback["status"] == rtx.SKIP and "quantize" in fallback["detail"]


def test_a_failing_sandbox_blocks_only_the_coding_stage(host):
    host.probe["sandbox"] = {"success": False, "stdout": "", "error": "unshare: unshare failed: Operation not permitted"}
    state = preflight(phase="pre")["sandbox"]
    assert state["status"] == rtx.FAIL and state["needed_by"] == "E" and "no unrestricted fallback" in state["fix"]


def test_ollama_down_is_blocking_and_a_windows_side_server_is_flagged(host):
    host.ollama_up = False
    down = preflight(phase="pre")["ollama_server"]
    assert down["status"] == rtx.FAIL and "ollama serve" in down["fix"]
    host.ollama_up, host.ollama_process = True, False
    assert preflight(phase="pre")["ollama_server"]["status"] == rtx.WARN


def test_post_download_checks_find_missing_models_wrong_context_and_low_vram(host):
    spec = rtx.MODELS["profiles"]["primary"]["ollama"]
    ok = preflight(phase="post")
    assert ok["ollama_models"]["status"] == rtx.PASS and ok["ollama_ctx"]["status"] == rtx.PASS and ok["hf_snapshot"]["status"] == rtx.PASS
    host.tags.pop(spec["derived_tag"])
    assert preflight(phase="post")["ollama_models"]["status"] == rtx.FAIL
    host.tags[spec["derived_tag"]] = {"name": spec["derived_tag"], "digest": "d" * 64}
    host.show_params = "num_ctx 4096"
    wrong = preflight(phase="post")["ollama_ctx"]
    assert wrong["status"] == rtx.FAIL and "4096" in wrong["detail"] and "8192" in wrong["detail"]
    host.show_params = "temperature 0.2"  # no num_ctx at all: the server would silently use its default
    assert "silently" in preflight(phase="post")["ollama_ctx"]["detail"]
    host.tags[spec["base_model"]]["digest"] = "e" * 64
    assert preflight(phase="post")["ollama_models"]["status"] == rtx.WARN  # moved tag: recorded, not blocking
    host.hf = (None, "Qwen/Qwen2.5-Coder-1.5B-Instruct@2e1fd397ee is not downloaded")
    assert preflight(phase="post")["hf_snapshot"]["status"] == rtx.FAIL
    host.smi_query = "NVIDIA GeForce RTX 4070, 12282, 9000, 3000, 581.42\n"
    assert preflight(phase="post")["vram_free"]["status"] == rtx.FAIL
    host.smi_query = "NVIDIA GeForce RTX 4070, 12282, 3000, 9000, 581.42\n"
    assert preflight(phase="post")["vram_free"]["status"] == rtx.WARN


def test_preflight_command_exit_code_and_message(host, capsys):
    assert rtx.main(["preflight"]) == 0
    host.smi_rc = 127
    assert rtx.main(["preflight"]) == 1
    assert "STOP: fix the FAIL items" in capsys.readouterr().out


def test_a_crashing_check_is_reported_as_unverified_not_hidden(host, monkeypatch):
    monkeypatch.setattr(rtx, "nvidia_smi_gpus", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    state = preflight(phase="pre")["gpu_vram"]
    assert state["status"] == rtx.UNVERIFIED and "boom" in state["detail"]


# ====================================================================================================== stage orchestration (stub stage runner)
STUB = '''
import json, os, pathlib, signal, subprocess, sys, time
stage, run_dir = sys.argv[1], pathlib.Path(sys.argv[sys.argv.index("--run-dir") + 1])
behaviour = json.loads((run_dir.parent / "behaviour.json").read_text()).get(stage, "pass")
(run_dir / "ran_stages.log").open("a").write(stage + "\\n")
(run_dir / ("stage_" + stage + ".pid")).write_text(str(os.getpid()))

def spawn_child_and_wait():
    # a helper process in the stage's group, like a model server or a shell command, and a marker that says "I am running"
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    (run_dir / "child.pid").write_text(str(child.pid))
    (run_dir / "ready.flag").write_text("1")
    time.sleep(600)

if behaviour == "pass":
    out = run_dir / ("stage_" + stage); out.mkdir(exist_ok=True)
    (out / "result.json").write_text(json.dumps({"stage": stage, "status": "pass", "summary": "stub ok", "unverified": ["stub: not measurable"]}))
elif behaviour == "task_fail":
    out = run_dir / ("stage_" + stage); out.mkdir(exist_ok=True)
    (out / "result.json").write_text(json.dumps({"stage": stage, "status": "fail", "reason": "stub task failed"}))
elif behaviour == "crash":
    sys.stderr.write("Traceback ... hf_abcdefghijklmnopqrstuvwxyz0123 CUDA out of memory\\n"); sys.exit(3)
elif behaviour == "hang":
    spawn_child_and_wait()
elif behaviour in ("load", "load_then_hang", "load_then_crash"):
    # a stage that leaves OUR Ollama model on the GPU, like stage B does: the driver must unload it whatever happens next
    (run_dir.parent / "ollama_loaded.flag").write_text("1")
    if behaviour == "load_then_hang":
        spawn_child_and_wait()
    if behaviour == "load_then_crash":
        sys.exit(4)
    out = run_dir / ("stage_" + stage); out.mkdir(exist_ok=True)
    (out / "result.json").write_text(json.dumps({"stage": stage, "status": "pass", "summary": "stub loaded a model"}))
elif behaviour == "attempts_then_hang":
    out = run_dir / ("stage_" + stage); out.mkdir(exist_ok=True)
    (out / "attempts.jsonl").write_text(json.dumps({"kind": "t", "attempt": 1, "token": "hf_abcdefghijklmnopqrstuvwxyz0123"}) + "\\n" + json.dumps({"kind": "t", "attempt": 2}) + "\\n")
    spawn_child_and_wait()
elif behaviour == "load_attempts_ignore_term":
    # worst case for an interrupt: model left loaded, attempts written, and a stage AND a child that both ignore SIGTERM
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (run_dir.parent / "ollama_loaded.flag").write_text("1")
    out = run_dir / ("stage_" + stage); out.mkdir(exist_ok=True)
    (out / "attempts.jsonl").write_text(json.dumps({"kind": "t", "attempt": 1}) + "\\n")
    spawn_child_and_wait()
'''


@pytest.fixture
def stub_runner(host, tmp_path, monkeypatch):
    hdir = tmp_path / "pkg"
    hdir.mkdir()
    (hdir / "stage_runner.py").write_text(STUB)
    monkeypatch.setattr(rtx, "HERE", hdir)
    monkeypatch.setattr(rtx, "VENV_PY", Path(sys.executable))
    monkeypatch.setattr(rtx, "capture_environment", lambda ctx, run_dir: (run_dir / "environment").mkdir(exist_ok=True))
    monkeypatch.setattr(rtx, "TIMEOUTS", {"B": 20, "C": 20, "D": 20, "E": 20})

    def set_behaviour(**kw):
        rtx.RESULTS.mkdir(parents=True, exist_ok=True)
        (rtx.RESULTS / "behaviour.json").write_text(json.dumps(kw))
    set_behaviour()
    return set_behaviour


def run_all(profile="primary"):
    assert rtx.main(["--profile", profile, "run"]) == 0
    run_dir = rtx.RESULTS / (rtx.RESULTS / "latest.txt").read_text().strip()
    return run_dir, {s: (rtx.stage_result(run_dir, s) or {}).get("status") for s in rtx.STAGES}


def ran(run_dir: Path) -> list:
    log = run_dir / "ran_stages.log"
    return log.read_text().split() if log.exists() else []


def test_all_stages_run_in_order_when_everything_passes(stub_runner):
    run_dir, statuses = run_all()
    assert statuses == {s: rtx.PASS for s in rtx.STAGES} and ran(run_dir) == ["B", "C", "D", "E"]  # A is the preflight itself, no runner process
    summary = (run_dir / "summary.md").read_text()
    assert "stub: not measurable" in summary and "never means the models are good at coding" in summary


def test_a_failed_stage_stops_exactly_its_dependents(stub_runner):
    stub_runner(B="task_fail")
    run_dir, statuses = run_all()
    assert statuses == {"A": "pass", "B": "fail", "C": "pass", "D": "skip", "E": "skip"}  # C does not depend on B; D and E do
    assert ran(run_dir) == ["B", "C"]
    assert "stage B did not pass" in rtx.stage_result(run_dir, "D")["reason"] and "stage D did not pass" in rtx.stage_result(run_dir, "E")["reason"]


def test_a_failed_delegation_stage_skips_the_coding_stage(stub_runner):
    stub_runner(D="task_fail")
    run_dir, statuses = run_all()
    assert statuses["D"] == "fail" and statuses["E"] == "skip" and ran(run_dir) == ["B", "C", "D"]


def test_a_blocking_preflight_failure_prevents_the_stage_from_starting(stub_runner, host):
    host.ollama_up = False  # needed by B, D, E but not by C
    run_dir, statuses = run_all()
    assert statuses["B"] == "fail" and "ollama_server" in rtx.stage_result(run_dir, "B")["reason"]
    assert ran(run_dir) == ["C"] and statuses["D"] == "skip"  # B's runner never started; the unaffected stage still ran
    assert rtx.stage_result(run_dir, "B")["preflight_failures"] == ["ollama_server"]


def test_stage_a_failure_skips_everything_after_it(stub_runner, host):
    host.smi_rc = 127
    run_dir, statuses = run_all()
    assert statuses == {"A": "fail", "B": "skip", "C": "skip", "D": "skip", "E": "skip"} and ran(run_dir) == []


def test_a_crashing_stage_is_recorded_with_a_redacted_tail(stub_runner):
    stub_runner(C="crash")
    run_dir, statuses = run_all()
    result = rtx.stage_result(run_dir, "C")
    assert statuses["C"] == "fail" and "crashed (exit 3)" in result["reason"] and "CUDA out of memory" in result["stderr_tail"]
    assert "hf_abcdefghijklmnop" not in json.dumps(result) and "hf_abcdefghijklmnop" not in (run_dir / "stage_C" / "stderr.log").read_text()


def test_a_hanging_stage_is_killed_with_its_children_and_not_retried(stub_runner, monkeypatch):
    stub_runner(B="hang")
    monkeypatch.setattr(rtx, "TIMEOUTS", {"B": 3, "C": 20, "D": 20, "E": 20})
    run_dir, statuses = run_all()
    assert statuses["B"] == "fail" and "timed out after 3s" in rtx.stage_result(run_dir, "B")["reason"]
    assert ran(run_dir).count("B") == 1  # one attempt, no retry
    pid = int((run_dir / "child.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # the grandchild died with the stage's process group


def test_a_single_stage_needs_its_dependencies_to_have_passed(stub_runner):
    assert rtx.main(["stage", "D"]) == 1
    run_dir = rtx.RESULTS / (rtx.RESULTS / "latest.txt").read_text().strip()
    assert rtx.stage_result(run_dir, "D")["status"] == "skip" and ran(run_dir) == []
    for stage in "ABC":
        assert rtx.main(["stage", stage]) == 0
    assert rtx.main(["stage", "D"]) == 0 and ran(run_dir) == ["B", "C", "D"]


def test_reusing_a_run_directory_with_another_profile_is_refused(stub_runner):
    run_all("primary")
    with pytest.raises(SystemExit, match="--new-run"):
        rtx.main(["--profile", "fallback", "stage", "B"])


def test_the_manifest_lists_every_file_with_hashes_and_the_stage_outcomes(stub_runner):
    run_dir, _ = run_all()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["schema"] == 1 and manifest["profile"] == "primary" and set(manifest["stages"]) == set(rtx.STAGES)
    assert manifest["git"]["commit"] == "a" * 40 and manifest["models"]["bitsandbytes_pin"] == rtx.MODELS["bitsandbytes"]["pin"]
    for entry in manifest["files"]:
        assert rtx.sha256(run_dir / entry["path"]) == entry["sha256"] and "work" not in Path(entry["path"]).parts
    assert {"run.json", "summary.md", "stage_B/result.json"} <= {e["path"] for e in manifest["files"]}


# ====================================================================================================== no silent CPU fallback
def test_cpu_or_split_placement_is_always_a_failure():
    quant = {"model_quantization_bits": 4, "linear4bit_modules": 10}
    assert sr.placement_problems({"cuda"}, "bnb-4bit-nf4", quant, "text", 1500, 10) == []
    assert any("CPU" in p for p in sr.placement_problems({"cpu"}, "none", {}, "text", 1500, 10))
    assert any("cpu" in p for p in sr.placement_problems({"cuda", "cpu"}, "none", {}, "text", 1500, 10))  # offloaded layers count too


def test_requested_quantization_that_is_not_in_effect_is_a_failure():
    assert sr.placement_problems({"cuda"}, "bnb-4bit-nf4", {"model_quantization_bits": None, "linear4bit_modules": 0}, "text", 1500, 10)
    assert sr.placement_problems({"cuda"}, "none", {}, "", 1500, 10)  # no text at all
    assert sr.placement_problems({"cuda"}, "none", {}, "text", 1500, 1400)  # a leak: unloading freed nothing


def test_stage_c_refuses_to_run_without_cuda_and_never_loads_a_model(tmp_path, monkeypatch):
    import torch

    import smol_harness.models as models_mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(models_mod, "build_model", lambda spec: pytest.fail("a model must not be built when CUDA is missing"))
    ctx = rtx.Ctx("primary", tmp_path)
    ctx.snapshot = "/cache/snap"
    result = sr.stage_c(ctx, SimpleNamespace(attempts=1, production_defaults=False))
    assert result["status"] == rtx.FAIL and "Refusing to run on CPU" in result["reason"]


def test_a_model_load_failure_is_a_stage_failure_with_the_message_and_no_cpu_retry(tmp_path, monkeypatch):
    import torch

    import smol_harness.models as models_mod

    calls = []

    def failing(spec):
        calls.append(spec)
        raise models_mod.ModelLoadError("failed to load: bitsandbytes CUDA kernels missing (token=hf_secretsecretsecretsecret1234)")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(models_mod, "build_model", failing)
    monkeypatch.setattr(sr, "gpu_mem", lambda: {"torch_allocated_mib": 0, "torch_reserved_mib": 0, "torch_peak_allocated_mib": 0, "smi_used_mib": 0})
    monkeypatch.setattr(rtx, "nvidia_smi_gpus", lambda: ([], ""))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    ctx = rtx.Ctx("primary", tmp_path)
    ctx.snapshot = "/cache/snap"
    result = sr.stage_c(ctx, SimpleNamespace(attempts=1, production_defaults=False))
    assert result["status"] == rtx.FAIL and "Nothing was run on CPU" in result["reason"] and "--profile fallback" in json.dumps(result["evidence"])
    assert len(calls) == 1 and calls[0].device == "cuda" and "hf_secretsecret" not in result["reason"]


def test_the_runner_records_a_crash_as_a_failed_stage_with_a_diagnosis(tmp_path, monkeypatch):
    monkeypatch.setattr(rtx, "hf_snapshot_path", lambda profile: ("/cache/snap", ""))
    monkeypatch.setitem(sr.STAGE_FUNCS, "C", lambda ctx, args: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")))
    assert sr.main(["C", "--run-dir", str(tmp_path)]) == 0
    result = json.loads((tmp_path / "stage_C" / "result.json").read_text())
    assert result["status"] == rtx.FAIL and "CUDA out of memory" in result["reason"] and "traceback_tail" in result


def test_the_runner_stops_when_the_model_snapshot_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(rtx, "hf_snapshot_path", lambda profile: (None, "not downloaded"))
    sr.main(["D", "--run-dir", str(tmp_path)])
    result = json.loads((tmp_path / "stage_D" / "result.json").read_text())
    assert result["status"] == rtx.FAIL and result["reason"] == "not downloaded"


# ====================================================================================================== Ollama stage against a controlled local server
class MockOllama(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, ps_models, chat_script):
        super().__init__(("127.0.0.1", 0), MockHandler)
        self.ps_models, self.chat, self.lock, self.requests = ps_models, list(chat_script), threading.Lock(), []


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, payload, status=200):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.server.requests.append(("GET", self.path))
        spec = rtx.MODELS["profiles"]["primary"]["ollama"]
        if self.path == "/api/version":
            self.send({"version": "0.9.0"})
        elif self.path == "/api/tags":
            self.send({"models": [{"name": spec["derived_tag"], "digest": "d" * 64}]})
        elif self.path == "/api/ps":
            self.send({"models": self.server.ps_models})
        else:
            self.send({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        self.server.requests.append(("POST", self.path))
        if self.path == "/api/generate":
            self.send({"response": "OK", "eval_count": 5, "eval_duration": 100_000_000, "load_duration": 2_000_000_000})
        elif self.path == "/api/show":
            self.send({"parameters": "num_ctx 8192", "details": {}})
        elif self.path == "/v1/chat/completions":
            with self.server.lock:
                content = self.server.chat.pop(0) if self.server.chat else "Thought: done.\nFinal Answer: nothing more"
            self.send({"id": "x", "object": "chat.completion", "created": 0, "model": body.get("model", "m"),
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})
        else:
            self.send({"error": "not found"}, 404)


def react(tool, **args):
    return f"Thought: next.\nAction: {tool}\nAction Input: {json.dumps(args)}"


def ps_entry(fraction=1.0, context=8192):
    tag = rtx.MODELS["profiles"]["primary"]["ollama"]["derived_tag"]
    size = 5_700_000_000
    entry = {"name": tag, "model": tag, "size": size, "size_vram": int(size * fraction), "expires_at": "soon"}
    if context:
        entry["context_length"] = context
    return [entry]


@pytest.fixture
def mock_ollama(monkeypatch, tmp_path):
    started = []

    def start(ps_models, chat_script):
        server = MockOllama(ps_models, chat_script)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(server)
        monkeypatch.setattr(rtx, "OLLAMA_URL", f"http://127.0.0.1:{server.server_address[1]}")
        monkeypatch.setattr(rtx, "nvidia_smi_gpus", lambda: ([{"name": "mock", "total_mib": 12282, "used_mib": 4000, "free_mib": 8000, "driver": "x"}], ""))
        monkeypatch.setattr(rtx, "snapshot_ollama", lambda ctx, run_dir, name="ollama.json": None)
        return server

    yield start
    for server in started:
        server.shutdown()


STAGE_B_SCRIPT = [react("file_operations", operation="read", path="secret.txt"), "Thought: done.\nFinal Answer: The access code is 7391-QX.",
                  react("delegate", worker="reader", task="Read secret.txt and report the access code."), react("file_operations", operation="read", path="secret.txt"),
                  "Thought: done.\nFinal Answer: The code is 7391-QX.", "Thought: done.\nFinal Answer: The reader reports the code 7391-QX."]


def run_stage_b(tmp_path):
    ctx = rtx.Ctx("primary", tmp_path / "run")
    (tmp_path / "run" / "stage_B").mkdir(parents=True)
    return sr.stage_b(ctx, SimpleNamespace(attempts=1, production_defaults=False))


def test_stage_b_against_a_mock_server_records_real_executions_and_gpu_residency(mock_ollama, tmp_path):
    server = mock_ollama(ps_entry(1.0), STAGE_B_SCRIPT)
    result = run_stage_b(tmp_path)
    assert result["status"] == rtx.PASS, result
    b0 = result["evidence"]["b0_raw_inference"]
    assert b0["gpu_fraction"] == 1.0 and b0["tokens_per_second"] == 50.0 and b0["context_matches_config"] is True
    b1, b2 = result["evidence"]["b1_tool_use"], result["evidence"]["b2_delegation_same_ollama_model"]
    assert b1[0]["success"] and b1[0]["valid_call_executed"] and b1[0]["tools_really_executed"][0]["tool"] == "file_operations"
    assert b2[0]["success"] and b2[0]["delegate_calls_executed"] and b2[0]["worker_records"][0]["status"] == "completed" and b2[0]["secret_reached_worker_model"]
    assert result["task_success"] == {"tool_use": True, "delegation": True} and "GPU inference OK" in result["summary"]
    assert ("POST", "/api/generate") in server.requests


def test_stage_b_refuses_a_partly_cpu_ollama_model(mock_ollama, tmp_path):
    mock_ollama(ps_entry(0.5), STAGE_B_SCRIPT)
    result = run_stage_b(tmp_path)
    assert result["status"] == rtx.FAIL and "50% on the GPU" in result["reason"] and "must run on the GPU" in result["reason"]
    assert "b1_tool_use" not in result["evidence"]  # nothing further ran


def test_stage_b_fails_on_a_context_mismatch_and_notes_when_the_server_cannot_say(mock_ollama, tmp_path):
    mock_ollama(ps_entry(1.0, context=4096), STAGE_B_SCRIPT)
    assert "reports context_length 4096" in run_stage_b(tmp_path)["reason"]


def test_stage_b_marks_the_runtime_context_unverified_when_the_server_does_not_report_it(mock_ollama, tmp_path):
    mock_ollama(ps_entry(1.0, context=None), STAGE_B_SCRIPT)
    result = run_stage_b(tmp_path)
    assert result["status"] == rtx.PASS and any("context_length" in u for u in result["unverified"])


def test_stage_b_reports_a_model_that_never_calls_a_tool_as_a_failure_even_though_the_gpu_works(mock_ollama, tmp_path):
    mock_ollama(ps_entry(1.0), ["Thought: easy.\nFinal Answer: abc123"] * 6)  # answers without acting, like the 350M model did
    result = run_stage_b(tmp_path)
    assert result["status"] == rtx.FAIL and "no attempt produced a tool call" in result["reason"]
    assert result["task_success"] == {"tool_use": False, "delegation": False} and result["evidence"]["b1_tool_use"][0]["no_call_attempted"]


def test_stage_b_fails_cleanly_when_the_server_cannot_run_the_model(monkeypatch, tmp_path):
    monkeypatch.setattr(rtx, "http_json", lambda *a, **k: (500, None, "HTTP 500: out of memory"))
    result = run_stage_b(tmp_path)
    assert result["status"] == rtx.FAIL and "could not run" in result["reason"] and "out of memory" in result["reason"]


# ====================================================================================================== install / prepare (commands only, nothing is downloaded)
class Recorder:
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = [], fail_at

    def __call__(self, cmd, **kw):
        self.calls.append([str(c) for c in cmd])
        return 1 if self.fail_at and self.fail_at in " ".join(map(str, cmd)) else 0


def test_prepare_and_install_refuse_to_download_without_confirmation(host, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(rtx.subprocess, "call", rec)
    for command in ("prepare", "install-deps"):
        with pytest.raises(SystemExit, match="Nothing was changed"):
            rtx.main([command])
    assert rec.calls == []


def test_prepare_runs_the_expected_commands_and_writes_the_fixed_context_modelfile(host, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(rtx.subprocess, "call", rec)
    assert rtx.main(["prepare", "--yes"]) == 0
    spec, hf = rtx.MODELS["profiles"]["primary"]["ollama"], rtx.MODELS["profiles"]["primary"]["hf"]
    assert rec.calls[0] == ["ollama", "pull", spec["base_model"]] and rec.calls[1][:3] == ["ollama", "create", spec["derived_tag"]]
    modelfile = (rtx.GENERATED / "Modelfile.primary").read_text()
    assert modelfile == f"FROM {spec['base_model']}\nPARAMETER num_ctx 8192\nPARAMETER num_predict 1024\n"
    assert hf["revision"] in rec.calls[2][-1] and "snapshot_download" in rec.calls[2][-1] and "local_files_only" not in rec.calls[2][-1]


def test_prepare_stops_at_the_first_failing_download(host, monkeypatch):
    rec = Recorder(fail_at="ollama pull")
    monkeypatch.setattr(rtx.subprocess, "call", rec)
    assert rtx.main(["prepare", "--yes"]) == 1 and len(rec.calls) == 1


def test_prepare_needs_a_running_server_and_downloads_nothing_otherwise(host, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(rtx.subprocess, "call", rec)
    host.ollama_up = False
    assert rtx.main(["prepare", "--yes"]) == 1 and rec.calls == []


def test_install_deps_installs_the_pinned_extra_only_for_the_quantized_profile(host, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(rtx.subprocess, "call", rec)
    assert rtx.main(["install-deps", "--yes"]) == 0
    assert rec.calls[0] == ["uv", "sync", "--locked"] and rec.calls[1][-1] == f"bitsandbytes=={rtx.MODELS['bitsandbytes']['pin']}"
    assert "--no-deps" in rec.calls[1] and rec.calls[2][:3] == ["uv", "pip", "check"]  # the resolver may not touch locked packages; consistency is then proven (R2-12)
    rec.calls.clear()
    assert rtx.main(["--profile", "fallback", "install-deps", "--yes"]) == 0
    assert rec.calls == [["uv", "sync", "--locked"]]


def test_install_deps_stops_when_the_sync_fails_and_refuses_on_a_bad_machine(host, monkeypatch):
    rec = Recorder(fail_at="uv sync")
    monkeypatch.setattr(rtx.subprocess, "call", rec)
    assert rtx.main(["install-deps", "--yes"]) == 1 and len(rec.calls) == 1  # no bitsandbytes after a failed sync
    rec.calls.clear()
    host.free_gib = 5
    assert rtx.main(["install-deps", "--yes"]) == 1 and rec.calls == []  # blocked before any download


# ====================================================================================================== redaction and the archive
SECRETS = ["hf_abcdefghijklmnopqrstuvwxyz012345", "ghp_abcdefghijklmnopqrstuvwxyz0123456789", "sk-abcdefghijklmnopqrstuvwxyz", "Bearer abcdef0123456789xyz",
           "http://alice:hunter2pw@host:11434/v1", "OPENAI_API_KEY=sk-live-999999", "password: correct-horse-battery"]


@pytest.mark.parametrize("secret", SECRETS)
def test_credentials_are_redacted(secret):
    cleaned = rtx.redact(f"prefix {secret} suffix")
    for fragment in ("abcdefghijklmnopqrstuvwxyz0", "hunter2pw", "sk-live-999999", "correct-horse", "abcdef0123456789xyz"):
        assert fragment not in cleaned
    assert "***" in cleaned and "prefix" in cleaned and "suffix" in cleaned


def test_environment_values_with_sensitive_names_and_the_home_directory_are_removed(monkeypatch):
    monkeypatch.setenv("MY_PRIVATE_TOKEN", "value-that-must-not-leak-123")
    text = f"saw value-that-must-not-leak-123 in {Path.home()}/projects and /mnt/c/Users/Someone/Documents"
    cleaned = rtx.redact(text)
    assert "value-that-must-not-leak-123" not in cleaned and str(Path.home()) not in cleaned and "Someone" not in cleaned


def test_redact_obj_masks_sensitive_keys_but_keeps_counters():
    cleaned = rtx.redact_obj({"api_key": "abcdefghijkl", "tokens_used": 52, "nested": [{"password": "longenoughpassword"}], "ok": "fine"})
    assert cleaned == {"api_key": "***", "tokens_used": 52, "nested": [{"password": "***"}], "ok": "fine"}


def test_captured_environment_contains_only_whitelisted_variables(host, tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_SECRET_TOKEN", "must-never-be-captured-9999")
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret-value-1234567890")
    rtx.capture_environment(rtx.Ctx("primary", tmp_path), tmp_path)
    captured = "\n".join(p.read_text() for p in (tmp_path / "environment").rglob("*") if p.is_file())
    assert "must-never-be-captured-9999" not in captured and "aws-secret-value" not in captured and "OLLAMA_HOST" in captured


def make_run(host, with_extra=None):
    run_dir = rtx.run_dir_for("primary", new=True)
    rtx.write_json(run_dir / "stage_A" / "result.json", {"stage": "A", "status": "pass", "summary": "ok"})
    rtx.write_text(run_dir / "environment" / "nvidia_smi.txt", "GPU 0: NVIDIA GeForce RTX 4070")
    (run_dir / "work" / "B").mkdir(parents=True)
    (run_dir / "work" / "B" / "leftover.txt").write_text("a disposable workspace file")
    for relative, content in (with_extra or {}).items():
        target = run_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content if isinstance(content, bytes) else content.encode())
    return run_dir


def test_the_archive_contains_exactly_the_manifest_files_and_no_workspaces(host, capsys):
    run_dir = make_run(host)
    assert rtx.main(["package"]) == 0
    archive = rtx.RESULTS / f"{run_dir.name}.tar.gz"
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        manifest = json.loads(tar.extractfile(f"{run_dir.name}/manifest.json").read())
    assert names[0].startswith(run_dir.name) and not any("/work/" in n for n in names)
    assert sorted(n.split("/", 1)[1] for n in names) == sorted([e["path"] for e in manifest["files"]] + ["manifest.json"])
    assert (rtx.RESULTS / f"{run_dir.name}.tar.gz.sha256").read_text().split()[0] == rtx.sha256(archive)
    out = capsys.readouterr().out
    assert "Send this ONE file" in out and archive.name in out and "no passwords, tokens or model files" in out


def open_archive(run_dir):
    archive = rtx.RESULTS / f"{run_dir.name}.tar.gz"
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        blob = "\n".join(tar.extractfile(m).read().decode(errors="replace") for m in tar.getmembers() if m.isfile())
        manifest = json.loads(tar.extractfile(f"{run_dir.name}/manifest.json").read())
    return names, blob, manifest


@pytest.mark.parametrize("extra,reason", [({"weights/model.safetensors": b"x"}, "model weights"), ({"big.json": b"0" * (26 * 1024 * 1024)}, "is this a model file"),
                                          ({"ggml/model.gguf": b"x"}, "model weights")])
def test_packaging_leaves_out_model_files_and_says_so(host, capsys, extra, reason):
    run_dir = make_run(host, extra)
    assert rtx.main(["package"]) == 0  # the diagnostics must still be sent; only the heavy file is dropped
    names, _, manifest = open_archive(run_dir)
    (path,) = extra
    assert not any(n.endswith(path) for n in names)
    assert [e["path"] for e in manifest["excluded_from_archive"]] == [path] and reason in manifest["excluded_from_archive"][0]["reason"]
    assert f"left out {path}" in capsys.readouterr().out


def test_a_credential_a_non_redacting_writer_left_behind_is_removed_before_packaging(host, capsys):
    run_dir = make_run(host, {"stage_B/leak.log": "the token was hf_abcdefghijklmnopqrstuvwxyz012345 sadly"})  # written WITHOUT the redacting writer
    assert rtx.main(["package"]) == 0
    names, blob, manifest = open_archive(run_dir)
    assert "abcdefghijklmnopqrstuvwxyz012345" not in blob and "stage_B/leak.log" in manifest["re_redacted_files"]
    assert "abcdefghijklmnopqrstuvwxyz012345" not in (run_dir / "stage_B" / "leak.log").read_text()  # fixed in place, not only in the copy
    assert "needed extra redaction" in capsys.readouterr().out


def test_packaging_refuses_when_a_credential_survives_the_redaction(host, capsys, monkeypatch):
    run_dir = make_run(host, {"stage_B/leak.log": "the token was hf_abcdefghijklmnopqrstuvwxyz012345 sadly"})
    monkeypatch.setattr(rtx, "re_redact_text_files", lambda _run_dir: [])  # simulate a redaction gap: the final scan of the archive is the last line of defence
    assert rtx.main(["package"]) == 1
    assert "still contains something that looks like a credential" in capsys.readouterr().out and not (rtx.RESULTS / f"{run_dir.name}.tar.gz").exists()


# ---- partial / malformed / failed records must never stop the diagnostics from being packaged
BAD_RESULTS = {
    "missing status": {"stage": "B", "summary": "died half way"},
    "status is not a string": {"stage": "B", "status": None},
    "not an object": ["B"],
}


@pytest.mark.parametrize("label", sorted(BAD_RESULTS))
def test_a_result_without_a_usable_status_is_never_a_pass_and_never_a_crash(host, label):
    run_dir = make_run(host)
    rtx.write_json(run_dir / "stage_B" / "result.json", BAD_RESULTS[label])
    result = rtx.stage_result(run_dir, "B")
    assert result["status"] in ("unknown", "unreadable") and result["status"] != "pass"
    assert rtx.main(["package"]) == 0
    _, _, manifest = open_archive(run_dir)
    assert manifest["stages"]["B"]["status"] in ("unknown", "unreadable")
    summary = (run_dir / "summary.md").read_text()
    assert "| B. " in summary and ("**UNKNOWN**" in summary or "**UNREADABLE**" in summary) and "**PASS** | " not in summary.split("| B. ")[1].split("\n")[0]


def test_a_truncated_result_file_is_unreadable_and_still_packaged(host):
    run_dir = make_run(host)
    (run_dir / "stage_B").mkdir(exist_ok=True)
    (run_dir / "stage_B" / "result.json").write_text('{"stage": "B", "stat')  # killed mid-write
    assert rtx.stage_result(run_dir, "B")["status"] == "unreadable"
    assert rtx.main(["package"]) == 0


@pytest.mark.parametrize("content", [None, "{not json", "[]", '{"profile": 7, "git": "oops"}'])
def test_a_missing_or_malformed_run_record_does_not_break_packaging(host, content):
    run_dir = make_run(host)
    (run_dir / "run.json").unlink(missing_ok=True)
    if content is not None:
        (run_dir / "run.json").write_text(content)
    assert rtx.main(["package"]) == 0 and (rtx.RESULTS / f"{run_dir.name}.tar.gz").exists()


def test_a_failed_stage_with_a_partial_record_is_packaged_with_its_logs_and_redacted(host):
    run_dir = make_run(host)
    rtx.write_json(run_dir / "stage_B" / "result.json", {"stage": "B", "status": "fail", "reason": "CUDA error with Bearer abcdef0123456789xyz"})
    (run_dir / "stage_B" / "stderr.log").write_text("Traceback ...\nHF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz012345\n")
    (run_dir / "stage_C").mkdir()
    (run_dir / "stage_C" / "result.json").write_text("")  # empty file: the runner was killed before writing
    assert rtx.main(["package"]) == 0
    names, blob, manifest = open_archive(run_dir)
    assert f"{run_dir.name}/stage_B/stderr.log" in names and "Traceback" in blob
    assert manifest["stages"]["B"]["status"] == "fail" and manifest["stages"]["C"]["status"] == "unreadable"
    assert "abcdefghijklmnopqrstuvwxyz012345" not in blob and "abcdef0123456789xyz" not in blob


def test_packaging_works_when_the_summary_cannot_be_generated(host, monkeypatch):
    run_dir = make_run(host)
    monkeypatch.setattr(rtx, "write_summary", lambda _d: (_ for _ in ()).throw(RuntimeError("boom")))
    assert rtx.main(["package"]) == 0
    assert "could not be generated" in (run_dir / "summary.md").read_text()


def test_a_later_stage_is_not_started_after_a_stage_with_a_malformed_result(stub_runner, host):
    run_dir = make_run(host)
    rtx.write_json(run_dir / "stage_A" / "result.json", {"stage": "A"})  # no status
    result = rtx.run_stage("B", rtx.Ctx("primary", run_dir))
    assert result["status"] == "skip" and "did not pass" in result["reason"] and ran(run_dir) == []


# ---- the "did the tool result reach the model" metric looks at what the model is shown, not at the tool's raw output
class FakeResult:
    def __init__(self, success, output=None, error=None):
        self.success, self.output, self.error = success, output, error


def test_model_visible_text_follows_what_effgen_shows_the_model():
    assert sr.model_visible(FakeResult(True, {"result": "ran ok", "exit_code": 0})) == "ran ok"
    assert sr.model_visible(FakeResult(True, {"success": True, "data": "file text", "message": "m"})) == "file text"
    assert sr.model_visible(FakeResult(False, None, "no such file")) == "no such file"
    assert sr.model_visible(FakeResult(True, "plain")) == "plain"


def test_a_result_counts_as_reached_only_if_the_visible_text_is_in_a_later_prompt():
    entry = {"visible": "ran ok: 3 passed"}
    assert sr.result_reached_model([entry], ["...Observation: ran ok: 3 passed..."])
    assert not sr.result_reached_model([entry], ["a prompt that never mentions it"])
    assert not sr.result_reached_model([{"visible": ""}], ["anything"])  # nothing visible proves nothing
    assert not sr.result_reached_model([], ["anything"])
    assert not sr.result_reached_model([entry, {"visible": "second call output"}], ["ran ok: 3 passed"])  # EVERY call must have reached it


def test_everything_written_by_the_redacting_writers_survives_the_final_scan(host):
    run_dir = make_run(host)
    rtx.write_text(run_dir / "stage_B" / "stderr.log", "error with hf_abcdefghijklmnopqrstuvwxyz012345 and http://u:pw12345678@h/x")
    rtx.write_json(run_dir / "stage_B" / "result.json", {"reason": "Bearer abcdef0123456789xyz failed", "api_key": "abcdefghijkl"})
    assert rtx.main(["package"]) == 0
    archive = next(rtx.RESULTS.glob("*.tar.gz"))
    with tarfile.open(archive) as tar:
        blob = "\n".join(tar.extractfile(m).read().decode(errors="replace") for m in tar.getmembers() if m.isfile())
    assert "abcdefghijklmnopqrstuvwxyz012345" not in blob and "pw12345678" not in blob and "abcdef0123456789xyz" not in blob


def test_packaging_without_any_run_says_what_to_do(host, capsys):
    assert rtx.main(["package"]) == 1 and "Run `rtx4070.py run` first" in capsys.readouterr().out


def test_status_shows_the_newest_summary(host, capsys):
    make_run(host)
    assert rtx.main(["status"]) == 0 and "| Stage | Status |" in capsys.readouterr().out


def test_the_gitignore_keeps_results_and_generated_files_out_of_git():
    lines = (PKG / ".gitignore").read_text().split()
    assert "results/" in lines and "generated/" in lines
    ignored = subprocess.run(["git", "check-ignore", "-q", "experiments/rtx4070/results/x/manifest.json"], cwd=PKG.parents[1])
    assert ignored.returncode == 0


# ====================================================================================================== review 002 remediation
TAG = rtx.MODELS["profiles"]["primary"]["ollama"]["derived_tag"]


def events(result: dict, when: str) -> dict:
    return next(e for e in result["residency"] if e["when"] == when)


# ---- R2-01: our own Ollama model is released between stages, deterministically and verifiably
def test_stage_b_leaves_nothing_on_the_gpu_for_stage_c(stub_runner, host):
    stub_runner(B="load")
    run_dir, statuses = run_all()
    assert statuses == {s: rtx.PASS for s in rtx.STAGES}
    assert not host.flag.exists()  # gone from the server's own state, not merely requested
    after_b = events(rtx.stage_result(run_dir, "B"), "after stage B")
    assert after_b["verified"] and [m["name"] for m in after_b["resident_before"]] == [TAG] and after_b["still_resident"] == []
    assert after_b["vram_free_after_mib"] > after_b["vram_free_before_mib"]  # free VRAM before and after the operation is recorded
    assert events(rtx.stage_result(run_dir, "C"), "before stage C")["resident_before"] == []
    assert ("POST", "/api/generate", {"model": TAG, "keep_alive": 0}) in host.http_log  # an explicit unload, not a wait for the default keep-alive


def test_without_the_release_the_leftover_model_would_fail_stage_c_and_the_message_names_it(stub_runner, host, monkeypatch):
    stub_runner(B="load")
    monkeypatch.setattr(rtx, "release_ollama", lambda profile, label, wait_s=None: {"when": label, "verified": True, "reason": "", "resident_before": [], "still_resident": []})
    run_dir, statuses = run_all()
    assert statuses["B"] == "pass" and statuses["C"] == "fail"  # this is the defect review 002 found; the release is what prevents it
    reason = rtx.stage_result(run_dir, "C")["reason"]
    assert "vram_free" in reason and TAG in reason and f"ollama stop {TAG}" not in reason  # the cause is named in the detail
    fix = by_id(json.loads((run_dir / "stage_C" / "preflight.json").read_text()))["vram_free"]["fix"]
    assert f"ollama stop {TAG}" in fix and "Windows" not in fix  # our own model is the cause: do not send the colleague to close Windows programs


@pytest.mark.parametrize("behaviour", ["load_then_hang", "load_then_crash"])
def test_the_model_is_released_even_when_the_stage_times_out_or_crashes(stub_runner, host, monkeypatch, behaviour):
    stub_runner(B=behaviour)
    monkeypatch.setattr(rtx, "TIMEOUTS", {"B": 3, "C": 20, "D": 20, "E": 20})
    run_dir, statuses = run_all()
    assert statuses["B"] == "fail" and statuses["C"] == "pass"
    assert not host.flag.exists() and events(rtx.stage_result(run_dir, "B"), "after stage B")["verified"]


def test_a_release_that_cannot_be_verified_blocks_the_next_gpu_stage_with_the_real_cause(stub_runner, host):
    stub_runner(B="load")
    host.release_works = False  # the server ignores the unload request
    run_dir, statuses = run_all()
    assert statuses["C"] == "fail" and statuses["D"] == "skip"
    b = rtx.stage_result(run_dir, "B")
    assert b["status"] == "pass" and any("could not be verified" in u for u in b["unverified"])  # reported, not hidden
    reason = rtx.stage_result(run_dir, "C")["reason"]
    assert "blocked" in reason and "still loaded" in reason and f"ollama stop {TAG}" in reason and "Windows" not in reason
    assert host.flag.exists()  # and nothing pretended otherwise


def test_an_ollama_without_a_readable_ps_cannot_be_verified_and_blocks(stub_runner, host):
    host.ps_status = 404
    event = rtx.release_ollama(rtx.load_profile("primary"), "t")
    assert not event["verified"] and "/api/ps" in event["reason"]


def test_an_unreachable_server_holds_no_gpu_memory_and_does_not_block_stage_c(host):
    host.ollama_up = False
    event = rtx.release_ollama(rtx.load_profile("primary"), "t")
    assert event["verified"] and "not reachable" in event["reason"]


def test_a_model_that_is_not_ours_is_reported_never_unloaded_and_named_in_the_fix(host):
    host.explicit = [{"name": "llama3:8b", "model": "llama3:8b", "size": 4_700_000_000, "size_vram": 4_700_000_000}]
    host.smi_query = "NVIDIA GeForce RTX 4070, 12282, 900, 5000, 581.42\n"
    event = rtx.release_ollama(rtx.load_profile("primary"), "t")
    assert event["verified"] and event["foreign_resident"][0]["name"] == "llama3:8b" and host.explicit  # untouched
    assert not [call for call in host.http_log if call[1] == "/api/generate"]
    state = preflight(phase="post")["vram_free"]
    assert state["status"] == rtx.FAIL and "llama3:8b" in state["detail"] and "ollama stop llama3:8b" in state["fix"]


def test_stage_c_with_nothing_of_ours_loaded_and_little_memory_blames_other_programs(host):
    host.smi_query = "NVIDIA GeForce RTX 4070, 12282, 9000, 3000, 581.42\n"
    state = preflight(phase="post")["vram_free"]
    assert state["status"] == rtx.FAIL and "Windows" in state["fix"] and "nothing of ours" in state["fix"]


# ---- R2-01 (stage D): both models must fit TOGETHER, measured, never assumed
def test_the_mixed_budget_is_checked_before_stage_d_and_suggests_only_a_smaller_set(host):
    ok = preflight(phase="post")["mixed_vram"]
    assert ok["status"] == rtx.PASS and ok["needed_by"] == "DE"
    host.smi_query = "NVIDIA GeForce RTX 4070, 12282, 4782, 7500, 581.42\n"  # passes the plain free-memory gate (warn), cannot hold the primary pair (8089 MiB estimated)
    state = preflight(phase="post")
    assert state["vram_free"]["status"] == rtx.WARN and state["mixed_vram"]["status"] == rtx.FAIL
    assert "--profile fallback" in state["mixed_vram"]["fix"] and "estimate" in state["mixed_vram"]["detail"]
    small = by_id(rtx.run_checks(rtx.Ctx("fallback"), phase="post"))["mixed_vram"]
    assert small["status"] == rtx.PASS  # the smaller pair is estimated at 7065 MiB
    host.smi_query = "NVIDIA GeForce RTX 4070, 12282, 8282, 4000, 581.42\n"
    assert "--profile" not in by_id(rtx.run_checks(rtx.Ctx("fallback"), phase="post"))["mixed_vram"]["fix"]  # nothing smaller exists: no false hope, no larger set


def test_our_own_resident_model_counts_as_available_for_the_mixed_budget(host):
    host.flag.parent.mkdir(parents=True, exist_ok=True)
    host.flag.write_text("1")  # 5436 MiB held by our model: the reported free memory is then 5764 MiB
    state = preflight(phase="post")
    assert state["vram_free"]["status"] == rtx.FAIL and state["mixed_vram"]["status"] == rtx.PASS


def test_stage_d_starts_from_a_released_gpu(stub_runner, host):
    stub_runner(B="load", D="load")
    run_dir, statuses = run_all()
    assert statuses["D"] == "pass" and events(rtx.stage_result(run_dir, "D"), "before stage D")["verified"] and events(rtx.stage_result(run_dir, "D"), "after stage D")["verified"]
    assert not host.flag.exists()


def test_the_verdict_on_simultaneous_residency_rests_on_measurements_only():
    cuda, probe = [["cuda"]], {"ollama_listed": True, "ollama_gpu_fraction": 1.0}
    assert sr.coresidency_verdict([probe], cuda)["observed"]
    nothing = sr.coresidency_verdict([], cuda)
    assert not nothing["observed"] and "NOT established" in nothing["reason"]  # sizes that add up are not evidence
    spilled = sr.coresidency_verdict([{"ollama_listed": True, "ollama_gpu_fraction": 0.62}], cuda, "primary")
    assert not spilled["observed"] and "62%" in spilled["reason"] and "--profile fallback" in spilled["reason"]
    assert "--profile" not in sr.coresidency_verdict([{"ollama_listed": True, "ollama_gpu_fraction": 0.5}], cuda, "fallback")["reason"]
    assert not sr.coresidency_verdict([probe], [["cuda", "cpu"]])["observed"] and not sr.coresidency_verdict([probe], [])["observed"]
    assert not sr.coresidency_verdict([{"ollama_listed": False, "ollama_gpu_fraction": None}], cuda)["observed"]  # Ollama no longer lists its model


def test_the_gpu_is_measured_only_while_the_worker_model_is_loaded(monkeypatch):
    import smol_harness.models as models_mod
    from smol_harness.config import ModelSpec

    taken = []
    monkeypatch.setattr(sr, "coresidency_probe", lambda tag: taken.append(tag) or {"ollama_listed": True, "ollama_gpu_fraction": 1.0})

    class Fake:
        def __init__(self, devices):
            self.model = SimpleNamespace(parameters=lambda: [SimpleNamespace(device=SimpleNamespace(type=d)) for d in devices])
            self.model_name = "fake"

        def generate(self, prompt, config=None, **kw):
            return SimpleNamespace(text="Thought: x", finish_reason="stop", tokens_used=3)

        def unload(self):
            pass

    monkeypatch.setattr(models_mod, "build_model", lambda spec: Fake(["cuda"] if spec.provider == "hf" else []))
    residency = {"hf_loaded": False, "probes": [], "placements": []}
    calls, loads = [], []
    build = sr.tracing_build(loads, calls, residency)
    ollama = build(ModelSpec("ollama", TAG, "server", base_url="http://127.0.0.1:11434/v1", context_length=8192))
    ollama.generate("p")
    assert taken == []  # the worker model is not loaded yet: nothing is co-resident
    hf = build(ModelSpec("hf", "/snap", "bnb-4bit-nf4", dtype="bf16", device="cuda"))
    ollama.generate("p")
    assert taken == [TAG] and residency["placements"] == [["cuda"]] and residency["hf_loaded"]
    hf.unload()
    ollama.generate("p")
    assert taken == [TAG] and not residency["hf_loaded"] and [c["provider"] for c in calls] == ["ollama", "ollama", "ollama"]


# ---- R2-05: stage D passes on mechanics; solving the task is recorded separately
def d_attempt(**overrides) -> dict:
    record = {"orchestrator_generations": 2, "orchestrator_providers": ["ollama"], "delegate_calls_executed": [{"worker": "reader"}], "worker_generations": 2,
              "worker_providers": ["hf"], "worker_records": [{"status": "failed"}], "delegation_report_reached_orchestrator": True,
              "coresidency": {"observed": True, "reason": "ok"}, "success": False, "final_answer": "I delegated and the worker found the code"}
    record.update(overrides)
    record["mechanics_problems"] = sr.delegation_problems(record, True)
    record["mechanics_ok"] = not record["mechanics_problems"]
    return record


def run_stage_d(tmp_path, monkeypatch, attempts):
    monkeypatch.setattr(sr, "delegation_attempts", lambda *a, **k: attempts)
    monkeypatch.setattr(rtx, "nvidia_smi_gpus", lambda: ([], ""))
    monkeypatch.setattr(rtx, "http_json", lambda *a, **k: (200, {"models": []}, ""))
    ctx = rtx.Ctx("primary", tmp_path)
    ctx.snapshot = "/cache/snap"
    return sr.stage_d(ctx, SimpleNamespace(attempts=1, production_defaults=True))


def test_stage_d_passes_when_the_mechanics_are_shown_even_if_the_task_is_not_solved(tmp_path, monkeypatch):
    result = run_stage_d(tmp_path, monkeypatch, [d_attempt()])
    assert result["status"] == rtx.PASS and result["task_success"]["delegated_task_solved"] is False and result["mechanics"]["attempts_with_full_mechanics"] == 1
    assert "recorded separately" in result["summary"]


@pytest.mark.parametrize("override,expected", [
    ({"delegate_calls_executed": []}, "never executed the delegate tool"),  # the answer claims delegation; nothing executed
    ({"worker_generations": 0, "worker_providers": []}, "worker model never generated"),
    ({"orchestrator_generations": 0, "orchestrator_providers": []}, "orchestrator model never generated"),
    ({"worker_providers": ["ollama"]}, "did not run on the Hugging Face model"),
    ({"orchestrator_providers": ["hf"]}, "did not run on Ollama"),
    ({"worker_records": []}, "no worker record"),
    ({"delegation_report_reached_orchestrator": False}, "did not return through effGen"),
    ({"coresidency": {"observed": False, "reason": "no measurement was taken while both models were loaded"}}, "no measurement was taken"),
])
def test_stage_d_fails_when_a_mechanic_is_missing_whatever_the_final_answer_claims(tmp_path, monkeypatch, override, expected):
    result = run_stage_d(tmp_path, monkeypatch, [d_attempt(**override)])
    assert result["status"] == rtx.FAIL and expected in result["reason"] and "Stage E is skipped" in result["reason"]


def test_a_solved_task_is_recorded_without_changing_a_passing_status(tmp_path, monkeypatch):
    result = run_stage_d(tmp_path, monkeypatch, [d_attempt(success=True)])
    assert result["status"] == rtx.PASS and result["task_success"] == {"delegated_task_solved": True, "attempts_solved": 1}


def test_an_out_of_memory_during_stage_d_is_reported_as_a_resource_limit_with_the_fallback(tmp_path, monkeypatch):
    attempt = d_attempt(delegate_calls_executed=[], worker_generations=0, worker_providers=[], resource_limit="CUDA ran out of memory while the two models were loaded together. "
                                                                                                                 "Use the smaller model set: add `--profile fallback` to every command.")
    reason = run_stage_d(tmp_path, monkeypatch, [attempt])["reason"]
    assert "ran out of memory" in reason and "--profile fallback" in reason


def test_the_summary_shows_task_outcomes_apart_from_the_status(host):
    run_dir = make_run(host)
    rtx.write_json(run_dir / "stage_D" / "result.json", {"stage": "D", "status": "pass", "summary": "x", "task_success": {"delegated_task_solved": False}})
    rtx.write_summary(run_dir)
    assert "Task outcomes (separate from PASS/FAIL)" in (run_dir / "summary.md").read_text()


# ---- R2-02: the Windows host must be unreachable before a model may run shell commands
def mounts_with(host, line: str) -> None:
    (host.tmp / "mounts").write_text((host.tmp / "mounts").read_text() + line + "\n")


@pytest.mark.parametrize("arrange,expected", [
    (lambda h, m: mounts_with(h, "C:\\134 /mnt/c 9p rw,noatime,aname=drvfs 0 0"), "Windows drives or host file systems are mounted"),
    (lambda h, m: mounts_with(h, "drvfs /mnt/d drvfs rw 0 0"), "/mnt/d (drvfs)"),
    (lambda h, m: (h.tmp / "binfmt_misc" / "WSLInterop").write_text("enabled\ninterpreter /init\n"), "WSLInterop)"),
    (lambda h, m: (h.tmp / "binfmt_misc" / "WSLInterop-late").write_text("enabled\ninterpreter /init\nflags: PF\n"), "WSLInterop-late"),
    (lambda h, m: (h.tmp / "binfmt_misc" / "WSLInterop-generic").mkdir(), "WSLInterop-generic"),  # exists but cannot be read as a file: fail closed
    (lambda h, m: m.setenv("WSL_INTEROP", "/run/WSL/12_interop"), "interop is enabled"),
    (lambda h, m: m.setattr(rtx.shutil, "which", lambda name, *a, **k: "/mnt/c/Windows/System32/cmd.exe" if name == "cmd.exe" else None), "cmd.exe"),
    (lambda h, m: m.setattr(rtx, "PROC_MOUNTS", h.tmp / "unreadable-mounts"), "/proc/mounts is unreadable"),
], ids=["drive-mount", "second-drive", "binfmt-interop", "binfmt-interop-late", "binfmt-unreadable-registration", "interop-env", "windows-exe-on-path",
        "mounts-unreadable-on-wsl"])
def test_stage_e_is_refused_while_any_route_to_the_windows_host_exists(host, monkeypatch, arrange, expected):
    arrange(host, monkeypatch)
    state = preflight(phase="pre")["host_isolation"]
    assert state["status"] == rtx.FAIL and state["critical"] and state["needed_by"] == "E" and expected in state["detail"]
    assert "wsl.conf" in state["fix"] and "[interop]" not in state["detail"] and "Do not look for a way around it" in state["fix"]


def test_a_probe_that_cannot_escape_does_not_override_an_existing_route(host):
    mounts_with(host, "C:\\134 /mnt/c 9p rw,aname=drvfs 0 0")
    assert host.probe["isolation"]["cmd_exe"]["escaped"] is False  # the probes are happy, and it changes nothing: a probe proves nothing about other routes
    assert preflight(phase="pre")["host_isolation"]["status"] == rtx.FAIL


@pytest.mark.parametrize("mutate,expected", [
    (lambda p: p["isolation"]["write_outside_workspace"].update(escaped=True), "write_outside_workspace"),
    (lambda p: p["isolation"]["cmd_exe"].update(escaped=True), "cmd_exe"),
    (lambda p: p["isolation"]["powershell_exe"].update(escaped=None), "not attempted"),
    (lambda p: p.pop("isolation"), "did not run"),
])
def test_an_escape_or_a_missing_probe_result_keeps_stage_e_refused_even_without_known_routes(host, mutate, expected):
    mutate(host.probe)
    state = preflight(phase="pre")["host_isolation"]
    assert state["status"] == rtx.FAIL and expected in state["detail"]


def test_the_expected_wsl_gpu_mounts_do_not_count_as_a_route_and_a_clean_machine_passes_honestly(host):
    facts = rtx.windows_bridge_facts()
    assert facts["windows_mounts"] == [] and not facts["interop_binfmt_enabled"]  # /usr/lib/wsl (GPU libraries) and /mnt/wslg are expected
    state = preflight(phase="pre")["host_isolation"]
    assert state["status"] == rtx.PASS and "not a proof" in state["detail"]


def test_stage_e_is_refused_by_the_driver_before_any_model_runs_and_stages_a_to_d_are_unaffected(stub_runner, host):
    mounts_with(host, "C:\\134 /mnt/c 9p rw,aname=drvfs 0 0")
    run_dir, statuses = run_all()
    assert statuses == {"A": "pass", "B": "pass", "C": "pass", "D": "pass", "E": "fail"} and ran(run_dir) == ["B", "C", "D"]
    result = rtx.stage_result(run_dir, "E")
    assert "host_isolation" in result["reason"] and "mounted" in result["reason"] and result["preflight_failures"] == ["host_isolation"]


def test_the_preflight_command_does_not_stop_stages_a_to_d_for_a_stage_e_only_blocker(host, capsys):
    mounts_with(host, "C:\\134 /mnt/c 9p rw,aname=drvfs 0 0")
    assert rtx.main(["preflight"]) == 0
    out = capsys.readouterr().out
    assert "Stages A to D can run" in out and "REFUSED" in out and "STOP: fix the FAIL items" not in out
    host.smi_rc = 127  # a blocker for everything is still a stop
    assert rtx.main(["preflight"]) == 1


def test_the_runner_refuses_stage_e_itself_even_when_started_by_hand(tmp_path, monkeypatch):
    failing = [{"id": "host_isolation", "status": rtx.FAIL, "detail": "Windows drives are mounted"}, {"id": "sandbox", "status": rtx.PASS, "detail": ""}]
    monkeypatch.setattr(rtx, "run_checks", lambda ctx, phase=None, stage=None: failing)
    monkeypatch.setattr(sr, "coding_attempts", lambda *a, **k: pytest.fail("no model-run shell command may start"))
    result = sr.stage_e(rtx.Ctx("primary", tmp_path), SimpleNamespace(attempts=1, production_defaults=True))
    assert result["status"] == rtx.FAIL and result["refused_for_safety"] and "Windows drives are mounted" in result["reason"]


def test_the_probe_script_asks_for_windows_programs_only_when_the_driver_found_no_interop_route(host, monkeypatch):
    compile(rtx.PROBE, "probe", "exec")
    seen = {}
    monkeypatch.setattr(rtx.Ctx, "probe", ORIG_PROBE)
    monkeypatch.setattr(rtx, "run", lambda cmd, timeout=30, env=None, cwd=None: (seen.update(env=env), (0, 'PROBE_JSON={"python": "3.12"}\n', ""))[1])
    rtx.Ctx("primary").probe()
    assert seen["env"]["SMOL_PROBE_EXE"] == "1"
    mounts_with(host, "C:\\134 /mnt/c 9p rw,aname=drvfs 0 0")
    rtx.Ctx("primary").probe()
    assert seen["env"]["SMOL_PROBE_EXE"] == "0"  # never start a Windows program when a route exists


def test_the_isolation_probe_runs_in_the_real_sandbox_and_leaves_nothing_behind(tmp_path, monkeypatch):
    # Executes the probe's own code (not a copy) against the real ShellTool, on this Linux host only. No GPU, no network, no Windows.
    import asyncio

    from smol_harness.tools import ShellTool, resolve_workspace

    ws = resolve_workspace(tmp_path)
    if not asyncio.run(ShellTool(ws).execute(command="true")).success:
        pytest.skip("the shell sandbox is unavailable on this host")
    source = rtx.PROBE[rtx.PROBE.index("def isolation():"):rtx.PROBE.index('attempt("isolation", isolation)')]
    for exe in ("0", "1"):
        monkeypatch.setenv("SMOL_PROBE_EXE", exe)
        namespace: dict = {}
        exec("import json, os, sys, tempfile\n" + source, namespace)
        result = namespace["isolation"]()
        assert result["write_outside_workspace"]["escaped"] is False
        if exe == "1":
            assert result["cmd_exe"]["escaped"] is False and result["powershell_exe"]["escaped"] is False  # not found on Linux: no Windows program started
        else:
            assert result["windows_programs"]["escaped"] is None  # "not attempted" is never a pass in the check
    assert not list(Path.home().glob(".smol-isolation-probe-*"))


# ---- R2-07: a stage that is killed keeps the attempts it finished
def test_a_stage_killed_by_its_time_limit_keeps_its_finished_attempts(stub_runner, host, monkeypatch):
    stub_runner(B="attempts_then_hang")
    monkeypatch.setattr(rtx, "TIMEOUTS", {"B": 3, "C": 20, "D": 20, "E": 20})
    run_dir, _ = run_all()
    result = rtx.stage_result(run_dir, "B")
    assert result["status"] == "fail" and result["attempts_recorded"] == 2 and "2 finished attempt(s) were kept" in result["reason"]
    assert rtx.main(["package"]) == 0
    archive = next(rtx.RESULTS.glob("*.tar.gz"))
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        blob = tar.extractfile(next(n for n in names if n.endswith("stage_B/attempts.jsonl"))).read().decode()
    assert blob.count("\n") == 2


def test_the_runner_appends_every_finished_attempt_redacted(tmp_path, monkeypatch):
    monkeypatch.setitem(sr.SINK, "path", tmp_path / "attempts.jsonl")
    sr.keep({"kind": "x", "note": "token hf_abcdefghijklmnopqrstuvwxyz0123 leaked"})
    sr.keep({"kind": "y"})
    lines = (tmp_path / "attempts.jsonl").read_text().splitlines()
    assert len(lines) == 2 and "hf_abcdefghijklmnop" not in lines[0] and json.loads(lines[1])["kind"] == "y"


def test_the_runner_clears_its_sink_after_a_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(rtx, "hf_snapshot_path", lambda profile: ("/cache/snap", ""))
    monkeypatch.setitem(sr.STAGE_FUNCS, "C", lambda ctx, args: {"stage": "C", "status": rtx.PASS})
    sr.main(["C", "--run-dir", str(tmp_path)])
    assert sr.SINK["path"] is None


# ---- R2-11: a private key block a model happened to print is removed; the promise in the guide matches
PEM = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmU\nAAAAB3NzaC1yc2E\n-----END OPENSSH PRIVATE KEY-----"


def test_a_private_key_block_is_redacted_whole_even_when_truncated():
    assert "b3BlbnNzaC1r" not in rtx.redact(f"cat id_rsa:\n{PEM}\nafter") and "after" in rtx.redact(f"{PEM}\nafter")
    assert "AAAAB3Nza" not in rtx.redact(PEM[: PEM.index("AAAAB3") + 8])  # a result cut off before the END line (tool results are truncated)
    assert rtx.secrets_left(PEM) and not rtx.secrets_left(rtx.redact(PEM))


def test_packaging_removes_a_private_key_from_a_file_that_bypassed_the_redacting_writers(host, capsys):
    run_dir = make_run(host)
    (run_dir / "stage_B").mkdir()
    (run_dir / "stage_B" / "stdout.log").write_text(f"output\n{PEM}\n")
    assert rtx.main(["package"]) == 0
    archive = next(rtx.RESULTS.glob("*.tar.gz"))
    with tarfile.open(archive) as tar:
        blob = tar.extractfile(f"{run_dir.name}/stage_B/stdout.log").read().decode()
    assert "b3BlbnNzaC1r" not in blob and "stage_B/stdout.log" in json.loads((run_dir / "manifest.json").read_text())["re_redacted_files"]


def test_the_guide_does_not_promise_more_than_the_redaction_does():
    text = " ".join((PKG / name).read_text() for name in ("README.md", "RESULTS.md"))
    assert "contains no passwords, tokens or model files" not in text
    assert "known credential formats" in text and "tar -tzf" in text


# ---- RR-1 / RR-5: the guide must not break the colleague's WSL or push them into stage E early
def test_the_guide_edits_wsl_conf_and_never_replaces_it_and_keeps_systemd():
    readme = (PKG / "README.md").read_text()
    assert "tee /etc/wsl.conf" not in readme and "> /etc/wsl.conf" not in readme  # the old recipe overwrote the whole file (and with it [boot] systemd=true)
    assert "never replace it" in readme and "systemd=true" in readme and "only the lines you added" in readme
    assert "systemd=true" in (PKG / "TROUBLESHOOTING.md").read_text()  # and the way back when Ollama stopped starting


def test_the_guide_says_stage_e_is_not_part_of_the_first_round_and_does_not_ask_to_fix_its_failure():
    readme = (PKG / "README.md").read_text()
    assert "not part of this first round" in readme and "disabled on the default WSL2 setup and is not part of the initial handoff" in readme
    assert readme.index("## Step 11.") < readme.index("## Stage E (NOT part of the first round)")  # the colleague finishes and sends A to D first
    title = next(c["title"] for c in rtx.CHECKS if c["id"] == "host_isolation")
    exception = '"Windows host is unreachable from the shell sandbox"'
    assert exception.strip('"') in title and exception in readme and "do not change any WSL setting because of it" in readme
    assert "except the stage-E-only one" in readme


def test_the_guide_explains_ctrl_c_and_the_cleanup_after_a_closed_window():
    readme, trouble = (PKG / "README.md").read_text(), (PKG / "TROUBLESHOOTING.md").read_text()
    assert "Ctrl-C" in readme and "exits with code 130" in readme and "once" in readme
    assert "pgrep -af stage_runner.py" in trouble and "ollama stop" in trouble and "130" in trouble


# ---- R2-12 / R2-14 / R2-15
def test_install_deps_stops_when_bitsandbytes_leaves_the_environment_inconsistent(host, monkeypatch, capsys):
    rec = Recorder(fail_at="pip check")
    monkeypatch.setattr(rtx.subprocess, "call", rec)
    assert rtx.main(["install-deps", "--yes"]) == 1
    out = capsys.readouterr().out
    assert "inconsistent" in out and "uv sync --locked" in out and "--profile fallback" in out


def test_the_vram_threshold_belongs_to_the_profile_and_only_a_smaller_set_is_suggested(host):
    host.smi_query = "NVIDIA GeForce RTX 4070 Laptop GPU, 9500, 500, 9000, 581.42\n"
    primary = preflight(phase="pre")["gpu_vram"]
    assert primary["status"] == rtx.FAIL and "--profile fallback" in primary["fix"]
    assert by_id(rtx.run_checks(rtx.Ctx("fallback"), phase="pre"))["gpu_vram"]["status"] == rtx.PASS
    host.smi_query = "NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 500, 7600, 581.42\n"
    assert "too small" in by_id(rtx.run_checks(rtx.Ctx("fallback"), phase="pre"))["gpu_vram"]["fix"]


def test_production_defaults_help_names_everything_it_disables():
    text = subprocess.run([sys.executable, str(PKG / "rtx4070.py"), "run", "--help"], capture_output=True, text=True).stdout
    assert "max_tokens" in text and "seed" in text


# ---- evidence accounting: the stand-alone copies in the runner agree with the tested helpers
def test_the_runner_accounting_agrees_with_the_tested_evidence_helpers():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import evidence as ev

    from effgen.core.agent_loop import DEFAULT_STOP_SEQUENCES

    stops = list(DEFAULT_STOP_SEQUENCES)
    for text in ("a\n\nObservation: b", "x. Observation: y", "plain", "", "\nUser: z"):
        assert sr.consumed_text(text, stops) == ev.consumed_text(text, stops)
    result, config = SimpleNamespace(text="Action: x\nObservation: y", finish_reason="stop", tokens_used=5), SimpleNamespace(stop_sequences=stops)
    assert sr.call_record("hf", "p", config, result, 0.1) == ev.call_record("hf", "p", config, result, 0.1)
    calls = [ev.call_record("hf", "p", config, result, 0.1)]
    assert sr.observation_accounting(calls) == ev.observation_accounting(calls) and sr.trajectory_id(calls) == ev.trajectory_id(calls)
    assert sr.trajectories([{"trajectory": "a"}, {"trajectory": "a"}]) == ev.trajectories([{"trajectory": "a"}, {"trajectory": "a"}]) == {"attempts": 2, "distinct": 1}
    entries = [{"visible": "the file text"}]
    assert sr.result_reached_model(entries, ["... the file text ..."]) == ev.result_reached_model(entries, ["... the file text ..."]) is True


# ====================================================================================================== RR-2: Ctrl-C must not leave a stage running
def alive(pid: int) -> bool:
    """True if the process exists and is not a zombie (a zombie has already exited and only waits to be reaped)."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


def pid_of(run_dir: Path, name: str) -> int:
    return int((run_dir / name).read_text())


def interrupt_when_ready(sig=signal.SIGINT, wait_s: float = 30.0) -> threading.Thread:
    """After the stub stage says it is running (its marker file), send `sig` to THIS process, exactly as a terminal Ctrl-C (or `kill`) would."""
    def deliver():
        deadline = time.time() + wait_s
        while time.time() < deadline:
            latest = rtx.RESULTS / "latest.txt"
            if latest.exists() and (rtx.RESULTS / latest.read_text().strip() / "ready.flag").exists():
                time.sleep(0.2)
                os.kill(os.getpid(), sig)
                return
            time.sleep(0.05)

    thread = threading.Thread(target=deliver, daemon=True)
    thread.start()
    return thread


@pytest.fixture
def real_sigint(monkeypatch):
    """Python's own SIGINT behaviour (KeyboardInterrupt) whatever the test runner installed, restored afterwards."""
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, previous)


@pytest.fixture
def bystander():
    """An unrelated process in its OWN session, like any other program on the machine. An interrupt must never touch it."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=True)
    yield process
    process.kill()
    process.wait()


def latest_run() -> Path:
    return rtx.RESULTS / (rtx.RESULTS / "latest.txt").read_text().strip()


def test_ctrl_c_stops_the_whole_stage_cleans_up_keeps_the_results_and_exits_130(stub_runner, host, real_sigint, bystander, capsys):
    stub_runner(B="load_then_hang")
    thread = interrupt_when_ready()
    assert rtx.main(["run"]) == 130
    thread.join(5)
    run_dir = latest_run()
    assert not alive(pid_of(run_dir, "stage_B.pid")) and not alive(pid_of(run_dir, "child.pid"))  # the stage AND what it started are gone
    assert alive(bystander.pid)  # nothing outside the stage's process group was signalled
    result = rtx.stage_result(run_dir, "B")
    assert result["status"] == "fail" and result["interrupted"] is True and "interrupted by the user" in result["reason"]
    assert not host.flag.exists() and [e for e in result["residency"] if e["when"] == "after stage B"][0]["verified"]  # our Ollama model was unloaded too
    assert ran(run_dir) == ["B"] and rtx.stage_result(run_dir, "C") is None  # nothing after the interrupted stage was started
    assert (run_dir / "summary.md").exists() and (run_dir / "manifest.json").exists()
    assert "Interrupted" in capsys.readouterr().out
    assert rtx.main(["package"]) == 0  # the partial run is packageable


def test_an_interrupted_stage_keeps_the_attempts_it_had_finished(stub_runner, host, real_sigint):
    stub_runner(B="attempts_then_hang")
    assert rtx.main(["stage", "A"]) == 0
    thread = interrupt_when_ready()
    assert rtx.main(["stage", "B"]) == 130
    thread.join(5)
    result = rtx.stage_result(latest_run(), "B")
    assert result["attempts_recorded"] == 2 and "2 finished attempt(s) were kept" in result["reason"]
    assert "hf_abcdefghijklmnop" not in (latest_run() / "stage_B" / "stdout.log").read_text()


def test_a_stage_that_ignores_sigterm_is_killed_after_the_grace_period_together_with_its_children(stub_runner, host, real_sigint, monkeypatch):
    stub_runner(B="load_attempts_ignore_term")
    monkeypatch.setattr(rtx, "INTERRUPT_GRACE_S", 1)
    thread = interrupt_when_ready()
    started = time.time()
    assert rtx.main(["run"]) == 130
    thread.join(5)
    assert time.time() - started < 15  # bounded: the grace period, then SIGKILL
    run_dir = latest_run()
    assert not alive(pid_of(run_dir, "stage_B.pid")) and not alive(pid_of(run_dir, "child.pid"))
    result = rtx.stage_result(run_dir, "B")
    assert "stopped by force" in result["reason"] and result["attempts_recorded"] == 1 and not host.flag.exists()


def test_sigterm_to_the_driver_is_handled_like_ctrl_c_and_the_handler_is_restored(stub_runner, host):
    stub_runner(B="hang")
    assert rtx.main(["stage", "A"]) == 0
    before = signal.getsignal(signal.SIGTERM)
    thread = interrupt_when_ready(signal.SIGTERM)
    assert rtx.main(["stage", "B"]) == 130
    thread.join(5)
    assert not alive(pid_of(latest_run(), "stage_B.pid")) and not alive(pid_of(latest_run(), "child.pid"))
    assert signal.getsignal(signal.SIGTERM) == before  # the driver does not leave its handler behind


def test_an_interrupt_outside_a_stage_exits_130_with_a_message_and_no_traceback(host, monkeypatch, capsys):
    monkeypatch.setattr(rtx, "cmd_status", lambda args: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert rtx.main(["status"]) == 130
    out = capsys.readouterr()
    assert "Interrupted" in out.out and "Traceback" not in out.out + out.err


def test_terminate_group_ends_a_disposable_group_and_only_that_group(bystander):
    leader = subprocess.Popen([sys.executable, "-c", "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)']); time.sleep(600)"],
                              start_new_session=True)
    time.sleep(0.5)
    members = [int(line.split()[0]) for line in subprocess.run(["ps", "-o", "pid=", "-g", str(leader.pid)], capture_output=True, text=True).stdout.splitlines()]
    assert len(members) == 2
    assert rtx.terminate_group(leader, grace_s=5) is False  # exited on SIGTERM: no force needed
    assert not any(alive(pid) for pid in members) and alive(bystander.pid)
    rtx.terminate_group(leader, grace_s=1)  # a group that is already gone is not an error


def test_a_disabled_or_unrelated_binfmt_registration_is_not_a_route(host):
    (host.tmp / "binfmt_misc" / "WSLInterop").write_text("disabled\n")
    (host.tmp / "binfmt_misc" / "python3.12").write_text("enabled\ninterpreter /usr/bin/python3.12\n")
    facts = rtx.windows_bridge_facts()
    assert facts["interop_registrations"] == [] and not facts["interop_binfmt_enabled"]
    assert preflight(phase="pre")["host_isolation"]["status"] == rtx.PASS


def test_this_test_module_never_loads_weights_or_starts_a_real_model():
    # Guard for the promise that local validation does no GPU or model work: the only builders mentioned here are inside strings or guarded stubs.
    tree = ast.parse(Path(__file__).read_text())
    called = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "") for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not called & {"build_model", "from_pretrained", "build_harness_from_config"}
