#!/usr/bin/env python3
"""RTX 4070 / WSL2 experiment driver for smol-harness.  Standard library only: it runs before the project environment exists.

    python3 rtx4070.py preflight            checks that need no downloaded models
    python3 rtx4070.py install-deps         `uv sync --locked` + the one pinned extra (bitsandbytes)
    python3 rtx4070.py prepare              pull the Ollama model, create the fixed-context tag, snapshot the HF model
    python3 rtx4070.py preflight --phase post
    python3 rtx4070.py run                  stages A..E in order, stopping dependents of any failed stage
    python3 rtx4070.py stage B              one stage (uses the newest run directory)
    python3 rtx4070.py package              ONE redacted archive to send back

Principles: a failed critical requirement stops the experiment that needs it, with a message that says what to do; nothing falls back to
CPU or to an unrestricted shell (the harness itself fails closed; this driver only reports); every artifact is redacted before it is written;
model weights never enter the results. Nothing here was run on the target machine when it was written: see README.md, "What is unverified".
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import getpass
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
GENERATED = HERE / "generated"
CONFIGS = HERE / "configs"
MODELS = json.loads((HERE / "models.json").read_text())
VENV_PY = REPO / ".venv" / "bin" / "python"
OLLAMA_URL = os.environ.get("SMOL_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")

STAGES = ("A", "B", "C", "D", "E")
TITLES = {"A": "Environment and GPU validation", "B": "Ollama inference and tool calling", "C": "Hugging Face CUDA load, placement, quantization",
          "D": "Mixed-provider delegation (Ollama orchestrator -> Hugging Face worker)", "E": "Minimal coding task in a disposable repository"}
REQUIRES = {"A": (), "B": ("A",), "C": ("A",), "D": ("A", "B", "C"), "E": ("D",)}
TIMEOUTS = {"B": 1500, "C": 1500, "D": 3600, "E": 3000}  # seconds: hard limit on the stage process; it is killed, not retried
# Stages that start from a GPU on which none of OUR Ollama models is resident, and stages that leave one resident (review 002, R2-01).
CLEAN_SLATE_BEFORE = ("C", "D", "E")
USES_OLLAMA = ("B", "D", "E")
RELEASE_WAIT_S = 60  # how long to wait for Ollama to report a released model gone; never "wait for the keep-alive to expire"
INTERRUPT_GRACE_S = 10  # after Ctrl-C / SIGTERM: time a stage gets to exit after SIGTERM before its process group is killed
EXIT_INTERRUPTED = 130
# module constants so tests can point them elsewhere
PROC_VERSION, MEMINFO, DEV_DXG = Path("/proc/version"), Path("/proc/meminfo"), Path("/dev/dxg")
PROC_MOUNTS, BINFMT_DIR = Path("/proc/mounts"), Path("/proc/sys/fs/binfmt_misc")
MAX_ARCHIVE_FILE_BYTES = 25 * 1024 * 1024
WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt", ".onnx")
PASS, WARN, FAIL, UNVERIFIED, SKIP = "pass", "warn", "fail", "unverified", "skip"

# The redaction policy lives in ONE place, src/smol_harness/redaction.py (also used by the CLI execution recorder). Loaded by path, not imported as
# `smol_harness.redaction`: this driver is standard library only and runs before the project environment (and so the installed package) exists.
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("smol_harness_redaction", REPO / "src" / "smol_harness" / "redaction.py")
_redaction = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_redaction)
_SECRET_NAME, _PEM, _REDACTIONS = _redaction._SECRET_NAME, _redaction._PEM, _redaction._REDACTIONS
redact, redact_obj, redact_record = _redaction.redact, _redaction.redact_obj, _redaction.redact_record


# --------------------------------------------------------------------------------------------------- small helpers
def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redact_obj(data), indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(redact(text), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gib(n: float) -> float:
    return round(n / 2**30, 1)


def run(cmd, timeout: float = 30, env=None, cwd=None):
    """(returncode, stdout, stderr). 127 = not found, -1 = timed out. Never raises."""
    try:
        done = subprocess.run(list(map(str, cmd)), capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)
        return done.returncode, done.stdout, done.stderr
    except FileNotFoundError:
        return 127, "", f"{cmd[0]}: command not found"
    except subprocess.TimeoutExpired:
        return -1, "", f"{cmd[0]}: timed out after {timeout}s"
    except OSError as exc:
        return 126, "", f"{cmd[0]}: {exc}"


def http_json(method: str, path: str, body=None, timeout: float = 10):
    """(status_code or None, parsed_json or None, error text). Talks to the Ollama server only."""
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(OLLAMA_URL + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            return response.status, (json.loads(raw) if raw.strip() else None), ""
    except urllib.error.HTTPError as exc:
        return exc.code, None, f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return None, None, str(getattr(exc, "reason", exc))


def load_profile(name: str) -> dict:
    if name not in MODELS["profiles"]:
        raise SystemExit(f"unknown profile {name!r}; choose from {', '.join(MODELS['profiles'])}")
    return MODELS["profiles"][name]


def git_info() -> dict:
    rc, commit, _ = run(["git", "-C", REPO, "rev-parse", "HEAD"])
    rc2, status, _ = run(["git", "-C", REPO, "status", "--porcelain"])
    _, branch, _ = run(["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"])
    return {"commit": commit.strip() if rc == 0 else None, "branch": branch.strip() or None,
            "dirty": bool(status.strip()) if rc2 == 0 else None, "dirty_files": status.strip().splitlines()[:20] if rc2 == 0 else []}


# --------------------------------------------------------------------------------------------------- Ollama residency (review 002, R2-01)
def ollama_names(profile: dict) -> set:
    spec = profile["ollama"]
    return {spec["derived_tag"], spec["base_model"]}


def ollama_resident() -> tuple:
    """(status, resident model entries, error). status is None when the server cannot be reached at all, else the HTTP status of /api/ps."""
    status, data, err = http_json("GET", "/api/ps", timeout=15)
    if status != 200 or not isinstance(data, dict):
        return status, None, err or f"HTTP {status}"
    return status, [m for m in data.get("models", []) if isinstance(m, dict)], ""


def entry_name(entry: dict) -> str:
    return str(entry.get("name") or entry.get("model") or "")


def vram_free_mib():
    gpus, _ = nvidia_smi_gpus()
    return gpus[0]["free_mib"] if gpus else None


def release_ollama(profile: dict, label: str, wait_s: float | None = None) -> dict:
    """Unload OUR Ollama models from the GPU now (`keep_alive: 0`) and verify with the server's own state (/api/ps) that they are gone.

    Never waits for Ollama's default five-minute keep-alive. Returns an event record: `verified` is True only when /api/ps was readable and listed
    none of our models afterwards, or when the server cannot be reached at all (a stopped server holds no VRAM). An HTTP error from /api/ps, or a model
    that stays listed, is `verified: False` with the reason; callers must then block, not assume. Models that are not ours are reported, never touched.
    """
    wait_s = RELEASE_WAIT_S if wait_s is None else wait_s
    ours = ollama_names(profile)
    event = {"when": label, "t_utc": now(), "vram_free_before_mib": vram_free_mib(), "resident_before": [], "foreign_resident": [], "released": [],
             "still_resident": [], "verified": False, "reason": ""}
    status, resident, err = ollama_resident()
    if status is None:
        event.update(verified=True, reason=f"the Ollama server is not reachable ({err}), so it holds no GPU memory", vram_free_after_mib=vram_free_mib())
        return event
    if resident is None:
        event["reason"] = (f"cannot read which models Ollama has loaded (GET /api/ps: {err}). Without that, release cannot be verified; this Ollama may be too old "
                           f"(update it) or unhealthy (journalctl -u ollama -n 50)")
        event["vram_free_after_mib"] = vram_free_mib()
        return event
    mine = [m for m in resident if entry_name(m) in ours]
    event["resident_before"] = [{"name": entry_name(m), "size_vram_mib": (m.get("size_vram") or 0) // 2**20} for m in mine]
    event["foreign_resident"] = [{"name": entry_name(m), "size_vram_mib": (m.get("size_vram") or 0) // 2**20} for m in resident if entry_name(m) not in ours]
    for m in mine:
        code, _, perr = http_json("POST", "/api/generate", {"model": entry_name(m), "keep_alive": 0}, timeout=60)
        event["released"].append({"name": entry_name(m), "http_status": code, **({"error": perr} if perr else {})})
    deadline = time.time() + wait_s
    while True:
        status, now_resident, err = ollama_resident()
        if now_resident is not None:
            left = [entry_name(m) for m in now_resident if entry_name(m) in ours]
            if not left or time.time() >= deadline:
                break
        elif time.time() >= deadline:
            break
        time.sleep(min(1.0, max(0.05, deadline - time.time())))
    after = vram_free_mib()
    event["vram_free_after_mib"] = after
    if now_resident is None:
        event["reason"] = f"after the unload request GET /api/ps failed ({err}); release cannot be verified"
        return event
    event["still_resident"] = left
    if left:
        event["reason"] = (f"{', '.join(left)} is still loaded in Ollama {wait_s:.0f}s after an explicit unload request (keep_alive 0). "
                           f"Try `ollama stop {left[0]}` and `ollama ps`; if it stays, restart Ollama (`sudo systemctl restart ollama`)")
        return event
    event["verified"] = True
    freed = (after - event["vram_free_before_mib"]) if after is not None and event["vram_free_before_mib"] is not None else None
    expected = sum(m["size_vram_mib"] for m in event["resident_before"])
    if freed is not None and expected and freed < expected // 2:
        event["warning"] = f"Ollama reports the model gone, but free VRAM rose by only {freed} MiB (it held about {expected} MiB); something else may have taken the memory"
    return event


def ollama_footprint(profile: dict) -> int:
    """MiB of VRAM that our Ollama models hold right now, per /api/ps (0 if unreadable or none)."""
    _, resident, _ = ollama_resident()
    return sum((m.get("size_vram") or 0) for m in resident or [] if entry_name(m) in ollama_names(profile)) // 2**20


# --------------------------------------------------------------------------------------------------- Windows host reachability (review 002, R2-02)
def windows_bridge_facts() -> dict:
    """What is observable from inside this Linux about routes to the Windows host. Facts only; the check below decides.

    A drive letter under /mnt (drvfs/9p) is a writable view of a Windows disk; WSL interop lets a Linux process start Windows programs, which then run
    OUTSIDE every Linux namespace the shell sandbox creates. /usr/lib/wsl (the GPU libraries) and /mnt/wslg are expected and not counted.
    """
    facts: dict = {"windows_mounts": [], "interop_binfmt_enabled": False, "interop_registrations": [], "interop_env": bool(os.environ.get("WSL_INTEROP")), "windows_programs_on_path": [], "mounts_unreadable": False}
    try:
        for line in PROC_MOUNTS.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[2] in ("drvfs", "9p", "virtiofs"):
                mountpoint = parts[1].replace("\\040", " ")
                if not mountpoint.startswith(("/usr/lib/wsl", "/mnt/wslg")):
                    facts["windows_mounts"].append(f"{mountpoint} ({parts[2]})")
    except OSError:
        facts["mounts_unreadable"] = True
    # WSL registers its interop handler as `WSLInterop`, and newer versions can add others (`WSLInterop-late`): look at every registration whose name
    # starts with it. Fail closed: a registration that exists but cannot be read counts as enabled.
    try:
        registrations = sorted(BINFMT_DIR.glob("WSLInterop*"))
    except OSError:
        registrations = []
    for registration in registrations:
        try:
            enabled = registration.read_text().lstrip().startswith("enabled")
        except OSError:
            enabled = True
        if enabled:
            facts["interop_registrations"].append(registration.name)
    facts["interop_binfmt_enabled"] = bool(facts["interop_registrations"])
    facts["windows_programs_on_path"] = [name for name in ("cmd.exe", "powershell.exe", "explorer.exe") if shutil.which(name)]
    return facts


def bridges_exposed(facts: dict, wsl: bool) -> list:
    problems = []
    if facts["windows_mounts"]:
        problems.append("Windows drives or host file systems are mounted and writable from Linux: " + ", ".join(facts["windows_mounts"][:5]))
    if facts["interop_binfmt_enabled"] or facts["interop_env"]:
        problems.append("Windows interop is enabled" + (f" ({', '.join(facts['interop_registrations'])})" if facts["interop_registrations"] else "") +
                        ", so Linux commands can start Windows programs (cmd.exe, powershell.exe, any .exe) outside the sandbox")
    if facts["windows_programs_on_path"]:
        problems.append("Windows programs are on PATH: " + ", ".join(facts["windows_programs_on_path"]))
    if wsl and facts["mounts_unreadable"]:
        problems.append("/proc/mounts is unreadable, so Windows drive mounts cannot be ruled out")
    return problems


# --------------------------------------------------------------------------------------------------- the project environment probe
PROBE = r'''
import json, os, sys, tempfile
out = {"python": sys.version.split()[0]}
def attempt(name, fn):
    try:
        out[name] = fn()
    except Exception as exc:
        out[name + "_error"] = type(exc).__name__ + ": " + str(exc)[:300]
attempt("effgen", lambda: __import__("effgen").__version__)
attempt("transformers", lambda: __import__("transformers").__version__)
attempt("smol_harness", lambda: __import__("smol_harness").__file__)
attempt("bitsandbytes", lambda: __import__("bitsandbytes").__version__)
def torch_info():
    import torch
    info = {"version": torch.__version__, "cuda_build": torch.version.cuda, "available": bool(torch.cuda.is_available())}
    if info["available"]:
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        info.update(name=props.name, capability=list(torch.cuda.get_device_capability(0)), total_mib=props.total_memory // 2**20,
                    free_mib=free // 2**20, bf16=bool(torch.cuda.is_bf16_supported()))
        x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16 if info["bf16"] else torch.float16)
        info["matmul_ok"] = bool((x @ x).float().isfinite().all().item())
    return info
attempt("torch", torch_info)
def sandbox():
    import asyncio
    from smol_harness.tools import ShellTool, resolve_workspace
    ws = resolve_workspace(tempfile.mkdtemp(prefix="smol-probe-"))
    result = asyncio.run(ShellTool(ws).execute(command="echo sandbox-ok"))
    import shutil; shutil.rmtree(ws, ignore_errors=True)
    return {"success": bool(result.success), "stdout": (result.output or {}).get("stdout", ""), "error": result.error}
attempt("sandbox", sandbox)
def isolation():
    # Behaviour, not configuration: try the routes out of the sandbox and look at what actually happened. Each probe is harmless by construction (an empty
    # marker file in the Linux home that we remove, `echo` run by a Windows shell). The Windows-program probes run ONLY when the driver established that no
    # interop route exists (SMOL_PROBE_EXE=1); a probe never writes to a Windows drive. A probe that cannot escape proves nothing about other routes.
    import asyncio, pathlib, shutil, uuid
    from smol_harness.tools import ShellTool, resolve_workspace
    ws = resolve_workspace(tempfile.mkdtemp(prefix="smol-probe-"))
    tool = ShellTool(ws)
    marker = pathlib.Path.home() / (".smol-isolation-probe-" + uuid.uuid4().hex)
    def sh(cmd):
        result = asyncio.run(tool.execute(command=cmd))
        return (result.output or {}) if isinstance(result.output, dict) else {}
    res = {}
    sh("touch " + str(marker))
    res["write_outside_workspace"] = {"escaped": marker.exists(), "what": "create an empty file in the real Linux home from inside the sandbox"}
    if marker.exists():
        marker.unlink()
    if os.environ.get("SMOL_PROBE_EXE") == "1":
        for name, cmd in (("cmd_exe", "cmd.exe /c echo smol-win-probe"), ("powershell_exe", "powershell.exe -NoProfile -Command Write-Output smol-win-probe")):
            o = sh(cmd + " 2>&1; true")
            res[name] = {"escaped": "smol-win-probe" in str(o.get("stdout", "")), "what": "start a Windows program from inside the sandbox"}
    else:
        res["windows_programs"] = {"escaped": None, "what": "not attempted: the driver found a Windows interop route, which already blocks stage E"}
    shutil.rmtree(ws, ignore_errors=True)
    return res
attempt("isolation", isolation)
print("PROBE_JSON=" + json.dumps(out))
'''


class Ctx:
    """What the checks share: the profile and one lazily-run probe of the project environment (it takes several seconds)."""

    def __init__(self, profile_name: str = "primary", run_dir: Path | None = None) -> None:
        self.profile_name, self.profile, self.run_dir, self._probe = profile_name, load_profile(profile_name), run_dir, None

    def probe(self) -> dict:
        if self._probe is None:
            if not VENV_PY.exists():
                self._probe = {"error": "the project environment does not exist"}
            else:
                exe_ok = not bridges_exposed(windows_bridge_facts(), "microsoft" in (PROC_VERSION.read_text().lower() if PROC_VERSION.exists() else ""))
                rc, out, err = run([VENV_PY, "-c", PROBE], timeout=240, cwd=REPO, env={**os.environ, "SMOL_PROBE_EXE": "1" if exe_ok else "0"})
                line = next((ln for ln in out.splitlines() if ln.startswith("PROBE_JSON=")), None)
                self._probe = json.loads(line[len("PROBE_JSON="):]) if line else {"error": f"probe failed (exit {rc}): {(err or out)[-400:]}"}
        return self._probe


# --------------------------------------------------------------------------------------------------- checks
CHECKS: list = []


def check(cid: str, title: str, phase: str = "pre", critical: bool = True, needed: str = "ABCDE", requires: tuple = ()):
    """Register a check. `requires` names checks that must not have FAILED first: if one did, this check cannot say anything true, so it is
    reported as UNVERIFIED ("not checked: <root> failed first") instead of a second FAIL. One cause, one failure, one fix to apply."""
    def register(fn):
        CHECKS.append({"id": cid, "title": title, "phase": phase, "critical": critical, "needed_by": needed, "requires": requires, "fn": fn})
        return fn
    return register


def nvidia_smi_gpus() -> tuple[list, str]:
    rc, out, err = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version", "--format=csv,noheader,nounits"])
    if rc != 0:
        return [], (err or out).strip()[:300] or f"exit {rc}"
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            gpus.append({"name": parts[0], "total_mib": int(float(parts[1])), "used_mib": int(float(parts[2])), "free_mib": int(float(parts[3])), "driver": parts[4]})
    return gpus, ""


def nvidia_cuda_version() -> tuple[float | None, str]:
    rc, out, _ = run(["nvidia-smi"])
    match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", out) if rc == 0 else None
    return (float(match.group(1)), match.group(1)) if match else (None, "")


@check("os", "Running on Linux (Ubuntu inside WSL2)")
def chk_os(ctx):
    if platform.system() != "Linux":
        return FAIL, f"this is {platform.system()}", "Open the Ubuntu app (WSL2) and run the commands there. Do not use PowerShell for this script."
    return PASS, f"{platform.system()} {platform.release()}", ""


@check("wsl2", "WSL2 (not WSL1)")
def chk_wsl(ctx):
    version = PROC_VERSION.read_text().lower() if PROC_VERSION.exists() else ""
    if "microsoft" not in version:
        return WARN, "not WSL: native Linux. The package is written for WSL2 and untested elsewhere; continuing is allowed.", ""
    if "wsl2" not in version:
        return FAIL, "this looks like WSL1 (no GPU support)", "In PowerShell: wsl --set-version <DistroName> 2   (list names with: wsl -l -v)"
    return PASS, f"WSL2 kernel {platform.release()}", ""


@check("arch", "x86_64 CPU architecture")
def chk_arch(ctx):
    return (PASS, platform.machine(), "") if platform.machine() == "x86_64" else (FAIL, platform.machine(), "The locked environment is for x86_64.")


@check("gpu_node", "WSL2 GPU device present (/dev/dxg)", needed="ABCDE")
def chk_gpu_node(ctx):
    if "microsoft" not in (PROC_VERSION.read_text().lower() if PROC_VERSION.exists() else ""):
        return SKIP, "not WSL", ""
    if DEV_DXG.exists():
        return PASS, "/dev/dxg exists", ""
    return FAIL, "/dev/dxg is missing: WSL2 does not expose a GPU", "Update the NVIDIA driver on WINDOWS (not in Ubuntu), then in PowerShell: wsl --update ; wsl --shutdown ; reopen Ubuntu."


@check("nvidia_smi", "nvidia-smi works and sees a GPU", requires=("gpu_node",))
def chk_smi(ctx):
    gpus, err = nvidia_smi_gpus()
    if not gpus:
        return FAIL, err or "no GPU listed", "Install/update the NVIDIA driver on Windows (580 or newer), run `wsl --update` in PowerShell, `wsl --shutdown`, reopen Ubuntu. Do NOT install a Linux NVIDIA driver inside Ubuntu."
    return PASS, "; ".join(f"{g['name']}, {g['total_mib']} MiB, driver {g['driver']}" for g in gpus), ""


@check("gpu_identity", "GPU is an RTX 4070", critical=False, requires=("nvidia_smi",))
def chk_identity(ctx):
    gpus, _ = nvidia_smi_gpus()
    if not gpus:
        return UNVERIFIED, "no GPU readable", ""
    name = gpus[0]["name"]
    if MODELS["min_requirements"]["gpu_name_contains"] in name:
        return PASS, name, ""
    return WARN, f"{name}: not an RTX 4070. The run is still recorded, but memory results then describe this card.", ""


def smaller_profiles(ctx, total_mib: int) -> list:
    """Profiles with a smaller VRAM estimate than the current one that this card can hold (only ever a smaller set is suggested, never a larger one)."""
    mine = ctx.profile["estimated_vram_gib"]["total"]
    return [name for name, p in MODELS["profiles"].items() if p["estimated_vram_gib"]["total"] < mine and total_mib >= p["min_total_vram_mib"]]


@check("gpu_vram", "GPU has enough VRAM for this model set", requires=("nvidia_smi",))
def chk_vram(ctx):
    gpus, _ = nvidia_smi_gpus()
    if not gpus:
        return UNVERIFIED, "no GPU readable", ""
    total = gpus[0]["total_mib"]
    need = ctx.profile["min_total_vram_mib"]
    if total < need:
        other = smaller_profiles(ctx, total)
        hint = f"Use the smaller model set: add `--profile {other[0]}` to the commands." if other else "This card is too small for the planned models; stop and tell the maintainer."
        return FAIL, f"{total} MiB total (< {need} needed by the '{ctx.profile_name}' profile)", hint
    return PASS, f"{total} MiB total, {gpus[0]['free_mib']} MiB free now", ""


@check("driver_cuda", "NVIDIA driver supports CUDA 13.0 (needed by the locked PyTorch)", needed="ACDE", requires=("nvidia_smi",))
def chk_driver(ctx):
    need = float(MODELS["locked_stack"]["min_driver_cuda_version"])
    value, text = nvidia_cuda_version()
    if value is None:
        return UNVERIFIED, "could not read 'CUDA Version' from nvidia-smi", "Run nvidia-smi and check the CUDA Version in its header."
    if value < need:
        return FAIL, f"driver offers CUDA {text}, the locked PyTorch needs {need}", MODELS["locked_stack"]["min_driver_hint"] + ". Install the newest Game Ready or Studio driver on Windows, reboot, retry."
    return PASS, f"driver offers CUDA {text}", ""


@check("python3", "System Python 3 is recent enough for this driver", needed="A", critical=False)
def chk_python3(ctx):
    version = platform.python_version()
    if sys.version_info < (3, 9):
        return FAIL, f"Python {version}", "sudo apt install -y python3   (3.9 or newer; the project's own Python 3.12 is installed by uv)"
    return PASS, f"Python {version} (the project environment uses its own Python 3.12 from uv)", ""


@check("uv", "uv (Python package manager) is installed", needed="A")
def chk_uv(ctx):
    rc, out, _ = run(["uv", "--version"])
    if rc == 0:
        return PASS, out.strip(), ""
    if VENV_PY.exists():
        return WARN, "uv not found, but the environment already exists (only needed to install)", ""
    return FAIL, "uv not found", "curl -LsSf https://astral.sh/uv/install.sh | sh   then   source $HOME/.local/bin/env"


@check("git", "git is installed")
def chk_git(ctx):
    rc, out, _ = run(["git", "--version"])
    return (PASS, out.strip(), "") if rc == 0 else (FAIL, "git not found", "sudo apt install -y git")


@check("repo_state", "Repository revision recorded", critical=False)
def chk_repo(ctx):
    info = git_info()
    if not info["commit"]:
        return WARN, "not a git checkout: the revision cannot be recorded", "Clone the repository with git instead of downloading a zip."
    if info["dirty"]:
        return WARN, f"commit {info['commit'][:10]} with local changes ({len(info['dirty_files'])} files): results will say so", "Run `git status` and undo local edits unless the maintainer asked for them."
    return PASS, f"commit {info['commit'][:10]} on {info['branch']}, clean", ""


@check("repo_location", "Repository is on the Linux filesystem (not /mnt/c)")
def chk_location(ctx):
    path = str(REPO)
    if path.startswith("/mnt/"):
        return FAIL, f"{path} is on the Windows drive", "Clone into your Linux home instead: cd ~ && git clone ... (the Windows drive is slow and breaks the sandbox)."
    return PASS, path, ""


@check("disk", "Enough free disk space")
def chk_disk(ctx):
    req = MODELS["min_requirements"]
    free = shutil.disk_usage(REPO).free / 2**30
    if free < req["disk_free_gib_fail_below"]:
        return FAIL, f"{free:.0f} GiB free", f"Need about {req['disk_free_gib_warn_below']} GiB free in the WSL disk (which lives on C:). Free space on C: or move the distro."
    if free < req["disk_free_gib_warn_below"]:
        return WARN, f"{free:.0f} GiB free (comfortable is {req['disk_free_gib_warn_below']})", ""
    return PASS, f"{free:.0f} GiB free", ""


@check("ram", "Enough memory available to Ubuntu")
def chk_ram(ctx):
    req = MODELS["min_requirements"]
    try:
        fields = {ln.split(":")[0]: int(ln.split()[1]) for ln in MEMINFO.read_text().splitlines() if ln.split(":")[0] in ("MemTotal", "MemAvailable")}
    except (OSError, ValueError, IndexError):
        return UNVERIFIED, "could not read /proc/meminfo", ""
    total, avail = fields.get("MemTotal", 0) / 2**20, fields.get("MemAvailable", 0) / 2**20
    if total < req["ram_gib_fail_below"]:
        return FAIL, f"Ubuntu has {total:.1f} GiB of RAM", "Raise the WSL memory limit: create C:\\Users\\<you>\\.wslconfig with [wsl2] memory=12GB, then `wsl --shutdown` in PowerShell."
    if total < req["ram_gib_warn_below"]:
        return WARN, f"Ubuntu has {total:.1f} GiB of RAM ({avail:.1f} available)", "More is better: see .wslconfig in TROUBLESHOOTING.md."
    return PASS, f"{total:.1f} GiB total, {avail:.1f} GiB available", ""


@check("venv", "Project environment exists (.venv)")
def chk_venv(ctx):
    if VENV_PY.exists():
        return PASS, str(VENV_PY), ""
    return FAIL, "no .venv in the repository", "python3 experiments/rtx4070/rtx4070.py install-deps"


@check("deps", "effGen 1.3.0 and the harness import correctly", requires=("venv",))
def chk_deps(ctx):
    probe = ctx.probe()
    if "error" in probe:
        return FAIL, probe["error"], "python3 experiments/rtx4070/rtx4070.py install-deps"
    problems = []
    if probe.get("effgen") != MODELS["locked_stack"]["effgen"]:
        problems.append(f"effgen is {probe.get('effgen', probe.get('effgen_error'))!r}, expected {MODELS['locked_stack']['effgen']}")
    if "smol_harness" not in probe:
        problems.append(f"smol_harness: {probe.get('smol_harness_error')}")
    if problems:
        return FAIL, "; ".join(problems), "python3 experiments/rtx4070/rtx4070.py install-deps"
    return PASS, f"effgen {probe['effgen']}, transformers {probe.get('transformers')}, python {probe['python']}", ""


@check("torch_cuda", "PyTorch sees and can compute on the GPU (no CPU fallback)", needed="ACDE", requires=("venv",))
def chk_torch(ctx):
    probe = ctx.probe()
    if "error" in probe:
        return UNVERIFIED, "cannot check: " + probe["error"], ""
    info = probe.get("torch")
    if not isinstance(info, dict):
        return FAIL, probe.get("torch_error", "torch not importable"), "python3 experiments/rtx4070/rtx4070.py install-deps"
    if not info.get("available"):
        return FAIL, f"torch {info['version']} (CUDA build {info['cuda_build']}) reports no CUDA device", "Check the driver (see the driver_cuda check) and `nvidia-smi`; do not continue on CPU."
    if not info.get("matmul_ok"):
        return FAIL, f"CUDA device {info.get('name')} could not run a matrix multiply", "See TROUBLESHOOTING.md, 'CUDA errors'."
    gpus, _ = nvidia_smi_gpus()
    note = "" if not gpus or gpus[0]["name"] == info["name"] else f" (nvidia-smi says {gpus[0]['name']})"
    return PASS, f"torch {info['version']} (CUDA {info['cuda_build']}), {info['name']}, capability {info['capability']}, bf16={info['bf16']}{note}", ""


@check("bitsandbytes", "bitsandbytes pinned version (NF4 quantization)", needed="CDE", requires=("venv",))
def chk_bnb(ctx):
    if ctx.profile["hf"]["quantization"] == "none":
        return SKIP, "this profile does not quantize", ""
    probe = ctx.probe()
    if "error" in probe:
        return UNVERIFIED, "cannot check: " + probe["error"], ""
    pin = MODELS["bitsandbytes"]["pin"]
    if "bitsandbytes" not in probe:
        return FAIL, probe.get("bitsandbytes_error", "not installed"), f"python3 experiments/rtx4070/rtx4070.py install-deps   (or use --profile fallback, which needs no bitsandbytes)"
    if probe["bitsandbytes"] != pin:
        return WARN, f"version {probe['bitsandbytes']}, pinned {pin}", "Run install-deps again to get the pinned version."
    return UNVERIFIED, f"version {pin} imports; whether its CUDA 13 kernels work on this GPU is only established by stage C", ""


@check("sandbox", "Shell sandbox starts (Linux namespaces)", needed="E", requires=("venv", "deps"))
def chk_sandbox(ctx):
    probe = ctx.probe()
    if "error" in probe:
        return UNVERIFIED, "cannot check: " + probe["error"], ""
    result = probe.get("sandbox")
    if not isinstance(result, dict):
        return FAIL, probe.get("sandbox_error", "sandbox probe did not run"), "See TROUBLESHOOTING.md, 'Sandbox'. Stage E will not run without it; there is no unrestricted fallback."
    if not result.get("success") or "sandbox-ok" not in result.get("stdout", ""):
        return FAIL, str(result.get("error") or result)[:300], "See TROUBLESHOOTING.md, 'Sandbox'. Stage E will not run without it; there is no unrestricted fallback."
    return PASS, "echo ran inside the sandbox", ""


@check("workspace", "Experiment workspaces are disposable and safe", needed="BDE")
def chk_workspace(ctx):
    base = (ctx.run_dir or RESULTS) / "work"
    try:
        base.mkdir(parents=True, exist_ok=True)
        probe = base / ".write-test"
        probe.write_text("x")
        probe.unlink()
    except OSError as exc:
        return FAIL, f"cannot write under {base}: {exc}", "Check permissions of the repository directory."
    resolved = base.resolve()
    if str(resolved).startswith("/mnt/") or resolved in (Path.home(), Path("/")) or resolved in Path.home().parents:
        return FAIL, f"{resolved} is not a safe workspace location", "Use a checkout in your Linux home directory."
    return PASS, f"{resolved} (created per run, never reused, deleted after each attempt)", ""


@check("ollama_server", "Ollama server reachable", needed="BDE")
def chk_ollama(ctx):
    status, data, err = http_json("GET", "/api/version", timeout=5)
    if status != 200 or not isinstance(data, dict):
        return FAIL, f"{OLLAMA_URL}: {err or status}", "Start it: `sudo systemctl start ollama` (or `ollama serve &`). See README step 7."
    rc, _, _ = run(["pgrep", "-x", "ollama"])
    where = "" if rc == 0 else " NOTE: no ollama process is running inside this Ubuntu, so this answer comes from somewhere else (the Windows app?). That setup is not the one this package was written for."
    return (PASS if rc == 0 else WARN), f"Ollama {data.get('version')} at {OLLAMA_URL}.{where}", ""


def tags_by_name() -> tuple[dict, str]:
    status, data, err = http_json("GET", "/api/tags", timeout=10)
    if status != 200 or not isinstance(data, dict):
        return {}, err or f"status {status}"
    return {m.get("name") or m.get("model"): m for m in data.get("models", [])}, ""


@check("ollama_models", "Ollama models are present (base and fixed-context tag)", phase="post", needed="BDE", requires=("ollama_server",))
def chk_ollama_models(ctx):
    tags, err = tags_by_name()
    if err:
        return FAIL, f"cannot list models: {err}", "Check the server (ollama_server check)."
    spec = ctx.profile["ollama"]
    missing = [t for t in (spec["base_model"], spec["derived_tag"]) if t not in tags]
    if missing:
        return FAIL, "missing: " + ", ".join(missing), "python3 experiments/rtx4070/rtx4070.py prepare" + ("" if ctx.profile_name == "primary" else " --profile " + ctx.profile_name)
    digest = (tags[spec["base_model"]].get("digest") or "")
    if digest and digest != spec["manifest_digest"]:
        return WARN, f"{spec['base_model']} has digest {digest[:12]}, pinned {spec['manifest_digest'][:12]}: the tag may have been updated upstream. Results will record this.", ""
    return PASS, f"{spec['base_model']} (digest {digest[:12] or 'unknown'}) and {spec['derived_tag']}", ""


@check("ollama_ctx", "Ollama server-side context window matches the harness setting", phase="post", needed="BDE", requires=("ollama_server",))
def chk_ollama_ctx(ctx):
    spec = ctx.profile["ollama"]
    status, data, err = http_json("POST", "/api/show", {"model": spec["derived_tag"]}, timeout=15)
    if status != 200 or not isinstance(data, dict):
        return FAIL, f"cannot inspect {spec['derived_tag']}: {err or status}", "python3 experiments/rtx4070/rtx4070.py prepare"
    params = str(data.get("parameters", ""))
    match = re.search(r"\bnum_ctx\s+(\d+)", params)
    if not match:
        return FAIL, "the derived model has no num_ctx parameter, so the server would use its own default and silently cut long prompts", "python3 experiments/rtx4070/rtx4070.py prepare"
    if int(match.group(1)) != spec["num_ctx"]:
        return FAIL, f"server num_ctx is {match.group(1)} but the harness is configured for {spec['num_ctx']}", "python3 experiments/rtx4070/rtx4070.py prepare"
    return PASS, f"num_ctx {match.group(1)} fixed in the Modelfile of {spec['derived_tag']}", ""


@check("hf_snapshot", "Hugging Face model snapshot is downloaded at the pinned revision", phase="post", needed="CDE", requires=("venv",))
def chk_hf(ctx):
    path, err = hf_snapshot_path(ctx.profile)
    if not path:
        return FAIL, err, "python3 experiments/rtx4070/rtx4070.py prepare"
    return PASS, path, ""


def vram_cause(ctx, gpus: list) -> tuple:
    """(what is holding the memory, what to do). Our own Ollama model is named first: it is OUR doing and `ollama stop` is the fix, not closing Windows programs."""
    _, resident, _ = ollama_resident()
    ours = [m for m in resident or [] if entry_name(m) in ollama_names(ctx.profile)]
    other = [m for m in resident or [] if entry_name(m) not in ollama_names(ctx.profile)]
    if ours or other:
        names = [entry_name(m) for m in ours + other]
        held = sum((m.get("size_vram") or 0) for m in ours + other) // 2**20
        return (f"Ollama has {', '.join(names)} loaded, holding about {held} MiB. ",
                f"Unload it: `ollama stop {names[0]}` (the experiment does this itself between stages), then run the check again.")
    return ("", "Close games, browsers with GPU acceleration, video apps and other AI tools on Windows (nothing of ours is loaded in Ollama), then retry.")


@check("vram_free", "Enough free VRAM to start", phase="post", needed="BCDE", requires=("nvidia_smi",))
def chk_vram_free(ctx):
    req = MODELS["min_requirements"]
    gpus, err = nvidia_smi_gpus()
    if not gpus:
        return UNVERIFIED, err, ""
    free = gpus[0]["free_mib"]
    who, fix = vram_cause(ctx, gpus)
    if free < req["vram_free_mib_fail_below"]:
        return FAIL, f"only {free} MiB free ({gpus[0]['used_mib']} MiB in use). {who}".strip(), fix
    if free < req["vram_free_mib_warn_below"]:
        return WARN, f"{free} MiB free: tight for the planned models. {who}".strip(), fix
    return PASS, f"{free} MiB free", ""


@check("mixed_vram", "Both models fit in free VRAM together (Ollama + Hugging Face)", phase="post", needed="DE", requires=("nvidia_smi",))
def chk_mixed_vram(ctx):
    """Stages D and E keep BOTH models on the GPU at once. The need is the profile's own estimate (models.json, ESTIMATES, minus the desktop's share, which is
    already part of 'used'); our own Ollama model, if resident, is counted as available because the stage starts from a clean slate. Actual co-residency is
    then MEASURED by stage D; this check only refuses a start that cannot work."""
    est = ctx.profile["estimated_vram_gib"]
    need = int(sum(v for k, v in est.items() if k not in ("total", "of", "windows_desktop_share")) * 1024)
    gpus, err = nvidia_smi_gpus()
    if not gpus:
        return UNVERIFIED, err, ""
    available = gpus[0]["free_mib"] + ollama_footprint(ctx.profile)
    if available < need:
        other = smaller_profiles(ctx, gpus[0]["total_mib"])
        fix = (f"Resource limit: the pair does not fit. Use the smaller set: add `--profile {other[0]}` to every command." if other
               else "Resource limit: not even the smaller pair is expected to fit. Stop and send what you have (python3 experiments/rtx4070/rtx4070.py package).")
        return FAIL, f"{available} MiB can be made available, the '{ctx.profile_name}' pair is estimated to need {need} MiB (an estimate, not a measurement)", fix
    return PASS, f"{available} MiB available for an estimated {need} MiB; the real co-residency is measured by stage D", ""


@check("host_isolation", "Windows host is unreachable from the shell sandbox (stage E only)", needed="E", requires=("sandbox",))
def chk_host_isolation(ctx):
    """Stage E lets a model run shell commands. effGen's sandbox confines writes with read-only remounts whose errors it discards, and its own probe tests
    only the temp directory; neither was ever observed against a Windows drive or WSL interop, which runs Windows programs OUTSIDE every Linux namespace.
    So stage E is refused while any such route exists, regardless of how a probe behaves. When none exists, behaviour probes run as a second line.
    Passing means: no known route to the Windows host AND the probes could not escape. It does NOT prove the sandbox secure (reads stay unconfined, R1)."""
    wsl = "microsoft" in (PROC_VERSION.read_text().lower() if PROC_VERSION.exists() else "")
    problems = bridges_exposed(windows_bridge_facts(), wsl)
    fix = ("Stage E is refused here. It needs an environment where the Windows host cannot be reached at all (README, 'Stage E: isolation requirements'). On WSL2 that is "
           "/etc/wsl.conf with [automount] enabled=false and [interop] enabled=false and appendWindowsPath=false, then `wsl --shutdown` in PowerShell. "
           "Stages A to D do not need this. Do not look for a way around it.")
    if problems:
        return FAIL, "; ".join(problems), fix
    probe = ctx.probe()
    result = probe.get("isolation") if isinstance(probe, dict) else None
    if not isinstance(result, dict):
        return FAIL, (probe.get("isolation_error") if isinstance(probe, dict) else None) or "the isolation probes did not run, so nothing is established", fix
    escaped = [name for name, r in result.items() if isinstance(r, dict) and r.get("escaped")]
    if escaped:
        return FAIL, "the sandbox let a probe through: " + ", ".join(escaped), fix
    not_run = [name for name, r in result.items() if isinstance(r, dict) and r.get("escaped") is None]
    if not_run:
        return FAIL, "probes not attempted: " + ", ".join(not_run), fix
    return PASS, ("no Windows drive mounts, no interop, and the probes (write outside the workspace, start cmd.exe / powershell.exe) could not escape. "
                  "This is evidence for THIS machine's configuration, not a proof; use disposable repositories and a distro that holds no secrets"), ""


def hf_snapshot_path(profile: dict) -> tuple[str | None, str]:
    """Local snapshot of the pinned revision, from the project environment (never downloads)."""
    if not VENV_PY.exists():
        return None, "the project environment does not exist"
    hf = profile["hf"]
    code = ("import json,sys\nfrom huggingface_hub import snapshot_download\n"
            f"p=snapshot_download({hf['repo']!r}, revision={hf['revision']!r}, allow_patterns={hf['allow_patterns']!r}, local_files_only=True)\n"
            f"import os\nprint('SNAP='+p+'|'+str(os.path.getsize(os.path.join(p,'model.safetensors'))))")
    rc, out, err = run([VENV_PY, "-c", code], timeout=60, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    line = next((ln for ln in out.splitlines() if ln.startswith("SNAP=")), None)
    if rc != 0 or not line:
        return None, f"{hf['repo']}@{hf['revision'][:10]} is not downloaded ({(err or out).strip()[-160:]})"
    path, size = line[5:].rsplit("|", 1)
    if int(size) != hf["safetensors_bytes"]:
        return None, f"model.safetensors is {size} bytes, expected {hf['safetensors_bytes']}: incomplete download"
    return path, ""


def run_checks(ctx: Ctx, phase: str | None = None, stage: str | None = None) -> list:
    """Run the selected checks. A check whose `requires` failed is UNVERIFIED and names the root cause (never a second FAIL, never a PASS);
    a failed root that the filter would have hidden (e.g. `--phase post`) is still listed so the cause is always visible."""
    registry = {item["id"]: item for item in CHECKS}
    done: dict = {}

    def evaluate(item: dict) -> dict:
        if item["id"] in done:
            return done[item["id"]]
        root = None
        for needed in item["requires"]:
            dep = evaluate(registry[needed])
            if dep["status"] == FAIL or dep.get("blocked_by"):
                root = dep.get("blocked_by") or dep["id"]
                break
        if root:
            status, detail, fix = UNVERIFIED, f"not checked: '{root}' failed first. Fix that and run again.", ""
        else:
            try:
                status, detail, fix = item["fn"](ctx)
            except Exception as exc:  # noqa: BLE001 - a broken check must be visible, not fatal
                status, detail, fix = UNVERIFIED, f"check crashed: {type(exc).__name__}: {exc}", "Report this to the maintainer."
        done[item["id"]] = {"id": item["id"], "title": item["title"], "phase": item["phase"], "critical": item["critical"], "needed_by": item["needed_by"],
                            "status": status, "detail": redact(str(detail)), "fix": fix, **({"blocked_by": root} if root else {})}
        return done[item["id"]]

    results = [evaluate(item) for item in CHECKS
               if not (phase and item["phase"] != phase) and not (stage and stage not in item["needed_by"])]
    shown = {r["id"] for r in results}
    hidden_failures = [r for r in done.values() if r["id"] not in shown and r["status"] == FAIL]
    return hidden_failures + results


def blocking(results: list) -> list:
    return [r for r in results if r["status"] == FAIL and r["critical"]]


def print_checks(results: list) -> None:
    mark = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL", UNVERIFIED: "????", SKIP: "skip"}
    for r in results:
        print(f"[{mark[r['status']]}] {r['title']}\n        {r['detail']}")
        if r["status"] in (FAIL, WARN) and r["fix"]:
            print(f"        -> {r['fix']}")
    bad, warns = blocking(results), [r for r in results if r["status"] == WARN]
    blocked = [r for r in results if r.get("blocked_by")]
    print(f"\n{len(results)} checks: {len(bad)} blocking failure(s), {len(warns)} warning(s), {len(blocked)} not checkable until a failure above is fixed, "
          f"{sum(r['status'] == UNVERIFIED and not r.get('blocked_by') for r in results)} that can only be verified later.")
    only_e = [r for r in bad if r["needed_by"] == "E"]
    if only_e and len(only_e) == len(bad):
        print("Stages A to D can run. Stage E (model-run shell commands) will be REFUSED until the FAIL item(s) above are resolved; that is intended, not a mistake of yours.")
    elif bad:
        print("STOP: fix the FAIL items above before running experiments. Nothing was started.")


# --------------------------------------------------------------------------------------------------- run directories, environment capture
def run_dir_for(profile_name: str, new: bool) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    latest = RESULTS / "latest.txt"
    if not new and latest.exists():
        existing = RESULTS / latest.read_text().strip()
        if existing.is_dir():
            recorded = json.loads((existing / "run.json").read_text()) if (existing / "run.json").exists() else {}
            if recorded.get("profile", profile_name) != profile_name:
                raise SystemExit(f"the newest run ({existing.name}) used profile {recorded['profile']!r}; add --new-run to start a fresh one with {profile_name!r}")
            return existing
    commit = (git_info()["commit"] or "nocommit")[:8]
    run_id = f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{commit}-{profile_name}"
    directory = RESULTS / run_id
    directory.mkdir(parents=True)
    write_json(directory / "run.json", {"run_id": run_id, "created_utc": now(), "profile": profile_name, "git": git_info()})
    latest.write_text(run_id, encoding="ascii")
    return directory


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return default


def run_meta(run_dir: Path) -> dict:
    """run.json as a dict; anything else (missing, truncated, a list) is treated as 'no metadata'."""
    meta = read_json(run_dir / "run.json", {})
    return meta if isinstance(meta, dict) else {}


def stage_result(run_dir: Path, stage: str) -> dict | None:
    """The stage's result, or None if it never reported. A partial or malformed record is returned with a status that is NEVER pass
    ('unreadable' / 'unknown'), so gating, the summary and the manifest can all rely on `status` being present."""
    path = run_dir / f"stage_{stage}" / "result.json"
    if not path.exists():
        return None
    data = read_json(path, None)
    if not isinstance(data, dict):
        return {"stage": stage, "status": "unreadable", "reason": "result.json could not be read or is not a JSON object"}
    if not isinstance(data.get("status"), str):
        return {**data, "stage": data.get("stage", stage), "status": "unknown", "reason": str(data.get("reason") or "the stage result has no status field")}
    return data


def capture_environment(ctx: Ctx, run_dir: Path) -> None:
    env_dir = run_dir / "environment"
    rc, smi, _ = run(["nvidia-smi"])
    write_text(env_dir / "nvidia_smi.txt", smi if rc == 0 else "nvidia-smi unavailable")
    _, gpu_csv, _ = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,pcie.link.gen.current,power.limit", "--format=csv"])
    write_text(env_dir / "gpu_query.csv", gpu_csv)
    os_release = Path("/etc/os-release").read_text() if Path("/etc/os-release").exists() else ""
    kernel = PROC_VERSION.read_text() if PROC_VERSION.exists() else ""
    write_text(env_dir / "os.txt", f"{os_release}\n{kernel}\n{' '.join(platform.uname())}\nWSL_DISTRO_NAME={os.environ.get('WSL_DISTRO_NAME', '')}\n")
    _, uv_list, _ = run(["uv", "pip", "list", "--python", VENV_PY], timeout=60) if VENV_PY.exists() else (0, "no environment", "")
    write_text(env_dir / "python_packages.txt", uv_list)
    probe = ctx.probe()
    interesting = {k: v for k, v in os.environ.items() if k.startswith(("OLLAMA_", "CUDA_", "SMOL_", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))}
    write_json(env_dir / "versions.json", {"python_driver": platform.python_version(), "uv": run(["uv", "--version"])[1].strip(), "git": run(["git", "--version"])[1].strip(),
                                           "ollama_cli": run(["ollama", "--version"])[1].strip() or run(["ollama", "--version"])[2].strip(),
                                           "project_environment": probe, "environment_variables_of_interest": interesting,
                                           "locked_stack": MODELS["locked_stack"], "bitsandbytes_pin": MODELS["bitsandbytes"]["pin"]})
    write_json(env_dir / "git.json", git_info())
    snapshot_ollama(ctx, run_dir)


def snapshot_ollama(ctx: Ctx, run_dir: Path, name: str = "ollama.json") -> None:
    spec = ctx.profile["ollama"]
    data: dict = {"url": OLLAMA_URL}
    for key, (method, path, body) in {"version": ("GET", "/api/version", None), "tags": ("GET", "/api/tags", None),
                                      "show_derived": ("POST", "/api/show", {"model": spec["derived_tag"]}), "ps": ("GET", "/api/ps", None)}.items():
        status, payload, err = http_json(method, path, body, timeout=15)
        if isinstance(payload, dict) and key == "show_derived":
            payload = {k: payload.get(k) for k in ("parameters", "details", "model_info", "capabilities") if k in payload}
            payload["modelfile_first_lines"] = str((payload or {}).get("parameters", ""))[:400]
        data[key] = payload if status == 200 else {"error": err or status}
    rc, env_text, _ = run(["systemctl", "show", "ollama", "--property=Environment,ActiveState,SubState"], timeout=10)
    data["systemd_unit"] = env_text.strip() if rc == 0 else "systemctl unavailable or no ollama unit"
    write_json(run_dir / "environment" / name, data)


# --------------------------------------------------------------------------------------------------- stages
def set_stage(run_dir: Path, stage: str, status: str, reason: str = "", **extra) -> dict:
    result = {"stage": stage, "title": TITLES[stage], "status": status, "reason": reason, "finished_utc": now(), **extra}
    write_json(run_dir / f"stage_{stage}" / "result.json", result)
    return result


def terminate_group(process, grace_s: float | None = None) -> bool:
    """Stop a stage's whole process group: SIGTERM, then SIGKILL if it is still there after the grace period. Returns True if SIGKILL was needed.

    The stage was started with `start_new_session=True`, so its process group id is its pid and the group holds only the stage and what it started;
    nothing else on the machine is signalled. Never raises: a group that is already gone is the goal.
    """
    grace_s = INTERRUPT_GRACE_S if grace_s is None else grace_s
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.wait(timeout=grace_s)
        # the leader may exit while a child that ignores SIGTERM lives on in the group: the kill below is cheap and ends it
        escalated = False
    except subprocess.TimeoutExpired:
        escalated = True
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    process.wait()
    return escalated


def drain(process, wait_s: float = 5.0) -> tuple:
    """(stdout, stderr) of a stage whose group was just killed. Bounded: a process that left the group (its own session) could still hold the pipes open."""
    try:
        return process.communicate(timeout=wait_s)
    except subprocess.TimeoutExpired:
        for pipe in (process.stdout, process.stderr):
            try:
                pipe.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
        return "", "(the output pipes are still held by a process outside the stage's process group; its output was not collected)"


def _raise_keyboard_interrupt(signum, frame):  # SIGTERM to the driver is handled like Ctrl-C
    raise KeyboardInterrupt


def partial_attempts(stage_dir: Path) -> dict:
    """What a stage that never reported (timeout, crash) still left behind: the runner appends every finished attempt to attempts.jsonl as it goes."""
    path = stage_dir / "attempts.jsonl"
    if not path.exists():
        return {"attempts_recorded": 0}
    lines = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    return {"attempts_recorded": len(lines), "attempts_file": f"{stage_dir.name}/attempts.jsonl"}


def run_stage(stage: str, ctx: Ctx, attempts: int = 3, production_defaults: bool = False) -> dict:
    run_dir = ctx.run_dir
    assert run_dir is not None
    for dep in REQUIRES[stage]:
        got = stage_result(run_dir, dep)
        if not got or got.get("status") != PASS:
            why = f"stage {dep} " + ("has not been run" if not got else f"did not pass (status: {got.get('status')})")
            print(f"[skip] stage {stage}: {why}")
            return set_stage(run_dir, stage, SKIP, f"dependency not met: {why}")
    started = time.time()
    residency: list = []
    if stage in CLEAN_SLATE_BEFORE:
        # Stage B (or a stage D/E that ended) may have left our Ollama model on the GPU. Unload it explicitly and prove it with Ollama's own state BEFORE
        # the free-VRAM gate is evaluated, so that gate measures the machine and not our leftovers (review 002, R2-01).
        event = release_ollama(ctx.profile, f"before stage {stage}")
        residency.append(event)
        if not event["verified"]:
            print(f"[block] stage {stage}: {event['reason']}")
            return set_stage(run_dir, stage, FAIL, "blocked: could not verify that our Ollama model is unloaded from the GPU. " + event["reason"], residency=residency,
                             duration_s=round(time.time() - started, 1))
    results = run_checks(ctx, phase=None, stage=stage)
    write_json(run_dir / f"stage_{stage}" / "preflight.json", results)
    bad = blocking(results)
    if bad:
        print_checks([r for r in results if r["status"] in (FAIL, WARN)])
        return set_stage(run_dir, stage, FAIL, "blocking preflight failure: " + "; ".join(f"{r['id']}: {r['detail']}" for r in bad),
                         preflight_failures=[r["id"] for r in bad], residency=residency, duration_s=round(time.time() - started, 1))
    if stage == "A":
        capture_environment(ctx, run_dir)
        warns = [r["id"] for r in results if r["status"] == WARN]
        return set_stage(run_dir, stage, PASS, "all blocking checks passed", warnings=warns, unverified=[r["id"] + ": " + r["detail"] for r in results if r["status"] == UNVERIFIED],
                         checks=len(results), duration_s=round(time.time() - started, 1))
    command = [str(VENV_PY), str(HERE / "stage_runner.py"), stage, "--run-dir", str(run_dir), "--profile", ctx.profile_name, "--attempts", str(attempts)]
    if production_defaults:
        command.append("--production-defaults")
    env = {**os.environ, "PATH": f"{REPO / '.venv' / 'bin'}:{os.environ.get('PATH', '')}", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PYTHONUNBUFFERED": "1", "SMOL_OLLAMA_URL": OLLAMA_URL}
    stage_dir = run_dir / f"stage_{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run ] stage {stage}: {TITLES[stage]} (limit {TIMEOUTS[stage] // 60} min) ...")
    process = None
    interrupted = escalated = False
    try:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        out, err = process.communicate(timeout=TIMEOUTS[stage])
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)  # the whole process group: no stray model or shell process survives
        out, err = drain(process)
        timed_out = True
    except KeyboardInterrupt:
        # Ctrl-C / SIGTERM reaches only the driver (the stage has its own session), so stop the stage ourselves, keep what it wrote, clean up, exit 130.
        interrupted, timed_out, out, err = True, False, "", ""
        if process is not None:
            escalated = terminate_group(process)
            out, err = drain(process)
    write_text(stage_dir / "stdout.log", out or "")
    write_text(stage_dir / "stderr.log", err or "")
    if stage in USES_OLLAMA:
        # Whatever happened (pass, fail, crash, kill), our model must not stay on the GPU for the next stage or for the colleague's desktop.
        residency.append(release_ollama(ctx.profile, f"after stage {stage}"))
    result_path = stage_dir / "result.json"
    duration = round(time.time() - started, 1)
    kept = partial_attempts(stage_dir)
    if interrupted:
        return set_stage(run_dir, stage, FAIL, f"interrupted by the user (Ctrl-C); the stage was stopped{' by force' if escalated else ''} and {kept['attempts_recorded']} finished attempt(s) were kept",
                         duration_s=duration, stderr_tail=(err or "")[-600:], residency=residency, interrupted=True, **kept)
    if timed_out:
        return set_stage(run_dir, stage, FAIL, f"timed out after {TIMEOUTS[stage]}s and was killed; {kept['attempts_recorded']} finished attempt(s) were kept", duration_s=duration,
                         stderr_tail=(err or "")[-600:], residency=residency, **kept)
    if process.returncode != 0 or not result_path.exists():
        return set_stage(run_dir, stage, FAIL, f"the stage process crashed (exit {process.returncode}) before reporting; {kept['attempts_recorded']} finished attempt(s) were kept",
                         duration_s=duration, stderr_tail=(err or "")[-1200:], residency=residency, **kept)
    result = stage_result(run_dir, stage)
    result["duration_s"] = duration
    if residency:
        result["residency"] = residency
        last = residency[-1]
        if stage in USES_OLLAMA and not last["verified"]:
            result.setdefault("unverified", []).append("release of our Ollama model after this stage could not be verified: " + last["reason"])
    if result["status"] in ("unreadable", "unknown"):
        result["reason"] = f"the stage process exited normally but its result is unusable: {result.get('reason')}"
        result["status"] = FAIL
    write_json(result_path, result)
    return result


def write_summary(run_dir: Path) -> None:
    lines = [f"# Run {run_dir.name}", ""]
    meta = run_meta(run_dir)
    git = meta.get("git") if isinstance(meta.get("git"), dict) else {}
    lines += [f"Profile: **{meta.get('profile', 'unknown')}**  |  commit: `{(git.get('commit') or 'unknown')[:12]}`  |  dirty: {git.get('dirty')}", "",
              "| Stage | Status | Why / what happened |", "|---|---|---|"]
    unverified: list = []
    for stage in STAGES:
        result = stage_result(run_dir, stage)
        if not result:
            lines.append(f"| {stage}. {TITLES[stage]} | not run | |")
            continue
        lines.append(f"| {stage}. {TITLES[stage]} | **{str(result.get('status', 'unknown')).upper()}** | {redact(str(result.get('reason') or result.get('summary') or ''))[:220]} |")
        extra = result.get("unverified")
        unverified += [f"{stage}: {u}" for u in (extra if isinstance(extra, list) else [])]
    outcomes = []
    for stage in STAGES:
        result = stage_result(run_dir, stage)
        if result and "task_success" in result:
            outcomes.append(f"- {stage}: task success recorded separately from the stage status: {json.dumps(result['task_success'])}")
    if outcomes:
        lines += ["", "## Task outcomes (separate from PASS/FAIL)", ""] + outcomes
    lines += ["", "PASS means the stage's own criterion was met (README.md, 'What each stage proves'). It never means the models are good at coding.", ""]
    if unverified:
        lines += ["## Could not be verified on this machine", ""] + [f"- {u}" for u in unverified] + [""]
    write_text(run_dir / "summary.md", "\n".join(lines))


PRIVATE_ARTIFACT_SUFFIXES = (".local.jsonl", ".db", ".sqlite", ".sqlite3", ".db-wal", ".db-shm", ".db-journal")


def exclusion_reason(path: Path) -> str | None:
    """Why a file must never go into an archive (model weights, anything too big to be a diagnostic, or a private local artifact). The same test keeps the redaction pass away
    from these files: they are neither archived nor rewritten."""
    if path.name.lower().endswith(PRIVATE_ARTIFACT_SUFFIXES):
        return "private local artifact (captured prompts or a SQLite history database): never archived, never rewritten"
    if path.suffix.lower() in WEIGHT_SUFFIXES:
        return "looks like model weights"
    if path.stat().st_size > MAX_ARCHIVE_FILE_BYTES:
        return f"larger than {MAX_ARCHIVE_FILE_BYTES // 2**20} MiB (is this a model file?)"
    return None


def write_manifest(run_dir: Path) -> dict:
    """Describe the run from whatever is on disk. Never raises on missing or malformed records: a failed run must still be packageable."""
    files, excluded = [], []
    for path in sorted(run_dir.rglob("*")):
        relative = path.relative_to(run_dir)
        if not path.is_file() or "work" in relative.parts or path.name == "manifest.json":
            continue
        reason = exclusion_reason(path)
        if reason:
            excluded.append({"path": str(relative), "reason": reason})
        else:
            files.append({"path": str(relative), "bytes": path.stat().st_size, "sha256": sha256(path)})
    meta = run_meta(run_dir)
    profile_name = meta.get("profile") if isinstance(meta.get("profile"), str) else None
    results = {s: stage_result(run_dir, s) for s in STAGES}
    manifest = {"schema": 1, "run_id": run_dir.name, "created_utc": meta.get("created_utc"), "packaged_utc": now(), "profile": profile_name,
                "git": meta.get("git") if isinstance(meta.get("git"), dict) else {},
                "models": {"profile": MODELS["profiles"].get(profile_name), "bitsandbytes_pin": MODELS["bitsandbytes"]["pin"]},
                "stages": {s: {k: r.get(k) for k in ("status", "reason", "duration_s")} for s, r in results.items() if r},
                "files": files, "excluded_from_archive": excluded, "re_redacted_files": [],
                "never_collected": ["private local artifacts: captured prompts (*.local.jsonl) and SQLite databases (*.db, *.sqlite*): never archived, never rewritten", "work/ (the disposable workspace folders themselves; file contents and shell output that passed through tool calls ARE in trace.jsonl, attempts.jsonl, cli_*.json and repo_diff files, redacted by pattern only)", "model weights"],
                "redaction": "credentials, tokens, environment values with sensitive names and personal paths are removed before writing"}
    write_json(run_dir / "manifest.json", manifest)
    return manifest


# --------------------------------------------------------------------------------------------------- commands
def confirm(message: str, assume_yes: bool) -> None:
    print(message)
    if assume_yes:
        return
    if not sys.stdin.isatty() or input("Type yes to continue: ").strip().lower() != "yes":
        raise SystemExit("Stopped. Nothing was changed.")


def render_config(template: str, run_dir: Path, ctx: Ctx, name: str | None = None, **values) -> Path:
    spec, hf = ctx.profile["ollama"], ctx.profile["hf"]
    base = {"__OLLAMA_MODEL__": spec["derived_tag"], "__OLLAMA_CTX__": str(spec["num_ctx"]), "__OLLAMA_BASE_URL__": OLLAMA_URL + "/v1",
            "__HF_QUANT__": hf["quantization"], "__HF_DTYPE__": hf["dtype"]}
    base.update(values)
    text = (CONFIGS / f"{template}.yaml").read_text()
    for key, value in base.items():
        text = text.replace(key, value)
    left = re.findall(r"__[A-Z_]+__", text)
    if left:
        raise ValueError(f"unfilled placeholders in {template}.yaml: {sorted(set(left))}")
    target = run_dir / "configs_rendered" / f"{template}-{name or values.get('__WORKER_NAME__', 'x')}.yaml"
    # Written VERBATIM, not through the redacting writer: the harness reads this file back, and redaction rewrites the home directory to `~`, which turned the workspace path into
    # `~/...` and made every run from a checkout under $HOME fail at start-up. What is shared is still redacted: `package` runs the redactor over every yaml before archiving.
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def cmd_preflight(args) -> int:
    ctx = Ctx(args.profile)
    results = run_checks(ctx, phase=None if args.phase == "all" else args.phase)
    print(f"Preflight ({args.phase}) for profile '{args.profile}'  |  repo {REPO}\n")
    print_checks(results)
    if args.json:
        write_json(Path(args.json), results)
    return 1 if [r for r in blocking(results) if r["needed_by"] != "E"] else 0


def cmd_install_deps(args) -> int:
    ctx = Ctx(args.profile)
    early = [r for r in run_checks(ctx, phase="pre", stage="A") if r["id"] in ("os", "wsl2", "arch", "nvidia_smi", "driver_cuda", "uv", "git", "repo_location", "disk")]
    print_checks(early)
    if blocking(early):
        return 1
    needs_bnb = ctx.profile["hf"]["quantization"] != "none"
    confirm(f"\nThis downloads about 6 GB (PyTorch and the NVIDIA runtime libraries) into {REPO / '.venv'}" + (f" and installs bitsandbytes=={MODELS['bitsandbytes']['pin']}" if needs_bnb else "") +
            ". It can take 5 to 20 minutes.", args.yes)
    if subprocess.call(["uv", "sync", "--locked"], cwd=REPO) != 0:
        print("uv sync failed. See TROUBLESHOOTING.md, 'Installing the dependencies'.")
        return 1
    # --no-deps: the resolver must not upgrade or downgrade any locked package (torch, numpy, ...) to suit bitsandbytes. `uv pip check` then proves that
    # everything bitsandbytes needs is already there; if not, say so instead of leaving a silently modified environment (review 002, R2-12).
    if needs_bnb and subprocess.call(["uv", "pip", "install", "--no-deps", "--python", str(VENV_PY), f"bitsandbytes=={MODELS['bitsandbytes']['pin']}"], cwd=REPO) != 0:
        print("bitsandbytes failed to install. You can continue with `--profile fallback`, which needs no bitsandbytes.")
        return 1
    if needs_bnb and subprocess.call(["uv", "pip", "check", "--python", str(VENV_PY)], cwd=REPO) != 0:
        print("bitsandbytes was installed, but the environment is now inconsistent (see above). Run `uv sync --locked` to restore it and use `--profile fallback`.")
        return 1
    print("\nDone. NOTE: do not run `uv sync` again; it removes bitsandbytes. If you do, re-run this command.")
    return 0


def cmd_prepare(args) -> int:
    ctx = Ctx(args.profile)
    spec, hf = ctx.profile["ollama"], ctx.profile["hf"]
    total = (spec["download_bytes"] + hf["download_bytes"]) / 1e9
    confirm(f"\nProfile '{args.profile}' will download:\n  - Ollama {spec['base_model']}  {spec['download_bytes'] / 1e9:.2f} GB\n"
            f"  - Hugging Face {hf['repo']} @ {hf['revision'][:10]}  {hf['download_bytes'] / 1e9:.2f} GB\n  total about {total:.1f} GB, then a tiny derived Ollama model "
            f"'{spec['derived_tag']}' (context {spec['num_ctx']}).", args.yes)
    status, _, err = http_json("GET", "/api/version", timeout=5)
    if status != 200:
        print(f"Ollama is not reachable at {OLLAMA_URL}: {err}\nStart it first (README step 7).")
        return 1
    if subprocess.call(["ollama", "pull", spec["base_model"]]) != 0:
        print("ollama pull failed. See TROUBLESHOOTING.md, 'Downloads'.")
        return 1
    GENERATED.mkdir(exist_ok=True)
    modelfile = GENERATED / f"Modelfile.{args.profile}"
    modelfile.write_text(f"FROM {spec['base_model']}\nPARAMETER num_ctx {spec['num_ctx']}\nPARAMETER num_predict {spec['num_predict']}\n")
    if subprocess.call(["ollama", "create", spec["derived_tag"], "-f", str(modelfile)]) != 0:
        print("ollama create failed.")
        return 1
    if not VENV_PY.exists():
        print("The project environment is missing: run install-deps first, then prepare again to fetch the Hugging Face model.")
        return 1
    code = (f"from huggingface_hub import snapshot_download\nprint(snapshot_download({hf['repo']!r}, revision={hf['revision']!r}, allow_patterns={hf['allow_patterns']!r}))")
    if subprocess.call([str(VENV_PY), "-c", code], cwd=REPO) != 0:
        print("The Hugging Face download failed. See TROUBLESHOOTING.md, 'Downloads'.")
        return 1
    print("\nModels downloaded. Now run: python3 experiments/rtx4070/rtx4070.py preflight --phase post" + ("" if args.profile == "primary" else f" --profile {args.profile}"))
    return 0


def cmd_stage(args) -> int:
    run_dir = run_dir_for(args.profile, args.new_run)
    ctx = Ctx(args.profile, run_dir)
    result = run_stage(args.stage, ctx, args.attempts, args.production_defaults)
    write_summary(run_dir)
    write_manifest(run_dir)
    print(f"stage {args.stage}: {result['status'].upper()}  {result.get('reason') or result.get('summary') or ''}\nresults: {run_dir}")
    if result.get("interrupted"):
        return EXIT_INTERRUPTED
    return 0 if result["status"] == PASS else 1


def cmd_run(args) -> int:
    run_dir = run_dir_for(args.profile, new=True)
    ctx = Ctx(args.profile, run_dir)
    print(f"Run directory: {run_dir}\n")
    for stage in STAGES:
        result = run_stage(stage, ctx, args.attempts, args.production_defaults)
        print(f"== stage {stage} ({TITLES[stage]}): {result['status'].upper()}  {result.get('reason') or result.get('summary') or ''}\n")
        write_summary(run_dir)
        write_manifest(run_dir)
        if result.get("interrupted"):
            print(f"Interrupted. The stage was stopped, our Ollama model was unloaded, and what was recorded is kept in {run_dir}.\n"
                  f"Send it with: python3 experiments/rtx4070/rtx4070.py package")
            return EXIT_INTERRUPTED
    print((run_dir / "summary.md").read_text())
    print(f"Next: python3 experiments/rtx4070/rtx4070.py package   (then send the .tar.gz it prints)")
    return 0


def cmd_status(args) -> int:
    latest = RESULTS / "latest.txt"
    if not latest.exists():
        print("No runs yet.")
        return 0
    run_dir = RESULTS / latest.read_text().strip()
    write_summary(run_dir)
    print((run_dir / "summary.md").read_text())
    return 0


_URL_PASSWORD = re.compile(r"://[^/\s:@]+:(?!\*\*\*)[^/\s@*]+@")  # a real user:password pair; the masked form http://***@host does not match


def secrets_left(text: str) -> list:
    """Anything in `text` that still looks like a credential (the strict patterns only: tokens, bearer headers, URL passwords)."""
    return [m.group(0)[:40] for pattern in (_REDACTIONS[0][0], _REDACTIONS[1][0], _REDACTIONS[2][0], _REDACTIONS[3][0], _URL_PASSWORD) for m in pattern.finditer(text)] + [
        m.group(0)[:40] for m in re.finditer(r"-----BEGIN [A-Z ]*PRIVATE KEY-----(?!\s*\(redacted\))", text)]


TEXT_SUFFIXES = (".json", ".jsonl", ".txt", ".log", ".md", ".yaml", ".yml", ".csv", ".diff")  # .jsonl: attempts.jsonl and trace.jsonl get the same second pass


def re_redact_text_files(run_dir: Path) -> list:
    """Defence in depth: run the redactor over every text file again, in place, so nothing unredacted can reach an archive even if some
    code path wrote a file without it. Returns the files that had to be changed (they are listed in the manifest)."""
    changed = []
    for path in sorted(run_dir.rglob("*")):
        relative = path.relative_to(run_dir)
        if not path.is_file() or "work" in relative.parts or path.suffix.lower() not in TEXT_SUFFIXES or exclusion_reason(path):
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        clean = redact(text)
        if clean != text:
            path.write_text(clean)
            changed.append(str(relative))
    return changed


def cmd_package(args) -> int:
    """One redacted archive. It is built from whatever the run left behind, failed or partial stages included: diagnostics are most needed then.
    Model weights and oversized files are left out (and listed); credentials are removed; only a post-check that still finds one stops it."""
    latest = RESULTS / "latest.txt"
    run_dir = Path(args.run_dir) if args.run_dir else (RESULTS / latest.read_text().strip() if latest.exists() else None)
    if not run_dir or not run_dir.is_dir():
        print("No run directory found. Run `rtx4070.py run` first.")
        return 1
    re_redacted = re_redact_text_files(run_dir)
    try:
        write_summary(run_dir)
    except Exception as exc:  # noqa: BLE001 - a broken summary must never prevent the diagnostics from being packaged
        write_text(run_dir / "summary.md", f"# Run {run_dir.name}\n\nThe summary could not be generated ({type(exc).__name__}: {exc}). See manifest.json and the stage folders.\n")
    manifest = write_manifest(run_dir)
    manifest["re_redacted_files"] = re_redacted
    write_json(run_dir / "manifest.json", manifest)
    archive = RESULTS / f"{run_dir.name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for entry in manifest["files"] + [{"path": "manifest.json"}]:
            tar.add(run_dir / entry["path"], arcname=f"{run_dir.name}/{entry['path']}")
    leftovers = []
    with tarfile.open(archive) as tar:  # post-condition: look inside the archive itself
        for member in tar.getmembers():
            if member.isfile() and Path(member.name).suffix.lower() in TEXT_SUFFIXES:
                found = secrets_left(tar.extractfile(member).read().decode("utf-8", "replace"))
                if found:
                    leftovers.append(f"{member.name}: {found[0]}...")
    if leftovers:
        archive.unlink()
        print("REFUSING to keep the archive: it still contains something that looks like a credential:\n  " + "\n  ".join(leftovers))
        return 1
    digest = sha256(archive)
    write_text(RESULTS / f"{run_dir.name}.tar.gz.sha256", f"{digest}  {archive.name}\n")
    for item in manifest["excluded_from_archive"]:
        print(f"NOTE: left out {item['path']}: {item['reason']}")
    if re_redacted:
        print(f"NOTE: {len(re_redacted)} file(s) needed extra redaction before packaging (listed in manifest.json).")
    print(f"\nArchive: {archive}\nSize:    {archive.stat().st_size / 1024:.0f} KiB\nSHA-256: {digest}\n\nSend this ONE file back.\n"
          f"To open its folder in Windows Explorer, run in Ubuntu:   explorer.exe {archive.parent}\n"
          f"It contains no passwords, tokens or model files.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rtx4070.py", description="Driver for the RTX 4070 / WSL2 experiment package (see README.md).")
    parser.add_argument("--profile", choices=sorted(MODELS["profiles"]), default="primary", help="model set (default: primary; fallback needs no bitsandbytes)")
    common = argparse.ArgumentParser(add_help=False)  # the same option after the command, e.g. `prepare --profile fallback`
    common.add_argument("--profile", choices=sorted(MODELS["profiles"]), default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND", parser_class=argparse.ArgumentParser)
    p = sub.add_parser("preflight", help="check the machine without changing anything", parents=[common])
    p.add_argument("--phase", choices=("pre", "post", "all"), default="pre", help="pre: before models are downloaded; post: after `prepare`")
    p.add_argument("--json", metavar="FILE", help="also write the results as JSON")
    p.set_defaults(func=cmd_preflight)
    for name, fn, text in (("install-deps", cmd_install_deps, "install the locked Python environment"), ("prepare", cmd_prepare, "download and prepare the models")):
        p = sub.add_parser(name, help=text, parents=[common])
        p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
        p.set_defaults(func=fn)
    p = sub.add_parser("stage", help="run one stage (uses the newest run directory)", parents=[common])
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--new-run", action="store_true", help="start a new results directory")
    p.set_defaults(func=cmd_stage)
    p = sub.add_parser("run", help="run stages A to E in order in a new results directory", parents=[common])
    p.set_defaults(func=cmd_run)
    for q in (sub.choices["stage"], sub.choices["run"]):
        q.add_argument("--attempts", type=int, default=3, choices=range(1, 6), metavar="1-5", help="bounded attempts per task (default 3)")
        q.add_argument("--production-defaults", action="store_true", help="do NOT apply the runner's overrides (temperature 0.1, max_tokens 768, a fixed seed per attempt): use the harness defaults")
    p = sub.add_parser("status", help="show the newest run's summary", parents=[common])
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("package", help="create the single redacted archive to send back", parents=[common])
    p.add_argument("--run-dir", help="a specific run directory (default: the newest)")
    p.set_defaults(func=cmd_package)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    in_main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt) if in_main_thread else None
    try:
        return args.func(args)
    except KeyboardInterrupt:  # outside a running stage (a check, a download, the cleanup): there is no stage process to stop
        print("\nInterrupted. Nothing is running any more. If a stage was running, see TROUBLESHOOTING.md, 'Other'.")
        return EXIT_INTERRUPTED
    finally:
        if in_main_thread:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    sys.exit(main())
