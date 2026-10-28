#!/usr/bin/env python
"""DGX Spark (Linux, ARM64) diagnostic validation: E0 runtime and sandbox preflight, E1 real-model file read, E2 real orchestrator-worker delegation.

A THIN ADAPTER, not a framework and not an RTX stage. It reuses the experiment instruments of `experiments/rtx4070/stage_runner.py` unchanged (trace.jsonl, the
per-attempt records, the independent verification, the executed-versus-written action ledger) and replaces only what is wrong for this host:

* the RTX preflight (x86_64-only `arch` check, WSL2 checks, an `nvidia-smi` memory parser that raises on `[N/A]`) is NOT used; E0 below is its DGX counterpart;
* `drv.nvidia_smi_gpus` is replaced for this process by a reader that falls back to system RAM when `nvidia-smi` reports no numbers (unified memory), and says so.

Nothing here downloads a model, relaxes the shell sandbox, or touches production code. E1/E2 never use the `shell` tool (only `file_operations`), so the sandbox result is
recorded by E0 as Linux evidence but does not gate them. Run from the repository root with the project environment:

    uv run python experiments/dgx/dgx_diag.py e0 --out <DIR> --tag <ollama-tag> --num-ctx 8192
    uv run python experiments/dgx/dgx_diag.py e1 --out <DIR> --tag <ollama-tag> --num-ctx 8192
    uv run python experiments/dgx/dgx_diag.py e2 --out <DIR> --tag <ollama-tag> --num-ctx 8192

See experiments/dgx/README.md.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "rtx4070"))
import rtx4070 as drv  # noqa: E402  (stdlib-only driver: constants, redaction, Ollama helpers)
import stage_runner as sr  # noqa: E402  (the trace/verification instruments)

PASS, WARN, FAIL, UNVERIFIED = drv.PASS, drv.WARN, drv.FAIL, drv.UNVERIFIED
MIN_AVAILABLE_GIB, WARN_AVAILABLE_GIB, MIN_DISK_GIB = 12, 24, 20
GPU_FRACTION_MIN = 0.95
GATING = ("gpu", "memory", "project_env", "ollama", "ollama_inference")  # E1/E2 refuse unless each is PASS or WARN. `cuda_driver` is reported but not gating: project_env runs real CUDA
# kernels, which proves more than a parsed header does (and some nvidia-smi builds print no parsable header)
E2_WORKER = {"__WORKER_NAME__": "reader", "__WORKER_DESCRIPTION__": "reads files in the workspace and reports what they say", "__WORKER_TOOLS__": "[file_operations]"}


# ------------------------------------------------------------------------------------------------------------------ unified-memory aware GPU reader
def meminfo_mib() -> dict:
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            fields[key] = int(value.split()[0]) // 1024
    return fields


def gpus_tolerant() -> tuple[list, str]:
    """`drv.nvidia_smi_gpus` for a host whose GPU shares system memory: when nvidia-smi gives no numbers (`[N/A]`, `Not Supported`) the memory figures are system RAM and each
    row carries `memory_source` saying so. The unmodified reader raises ValueError on such output."""
    rc, out, err = drv.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version", "--format=csv,noheader,nounits"])
    if rc != 0:
        return [], (err or out).strip()[:300] or f"exit {rc}"
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            total, used, free = (int(float(p)) for p in parts[1:4])
            source = "nvidia-smi"
        except ValueError:
            mem = meminfo_mib()
            total, free = mem["MemTotal"], mem["MemAvailable"]
            used = total - free
            source = f"system RAM (unified memory; nvidia-smi reported {','.join(parts[1:4])!r})"
        rows.append({"name": parts[0], "total_mib": total, "used_mib": used, "free_mib": free, "driver": parts[4], "memory_source": source})
    return rows, ""


# ------------------------------------------------------------------------------------------------------------------ E0
PROBE_ENV = r'''
import json
out = {}
try:
    import effgen; out["effgen"] = effgen.__version__
    import smol_harness; out["smol_harness_imports"] = True
    import torch
    out.update(torch=torch.__version__, torch_cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available())
    if torch.cuda.is_available():
        out.update(device=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)), arch_list=torch.cuda.get_arch_list())
        for dtype in ("float32", "bfloat16"):
            try:
                a = torch.arange(64, dtype=getattr(torch, dtype), device="cuda").reshape(8, 8)
                out["matmul_" + dtype] = bool(torch.allclose((a @ a).float().cpu(), a.cpu().float() @ a.cpu().float(), rtol=1e-2))
            except Exception as exc:
                out["matmul_" + dtype] = f"{type(exc).__name__}: {str(exc)[:200]}"
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
print("DIAG_JSON=" + json.dumps(out))
'''

PROBE_SANDBOX = r'''
import asyncio, json, os, tempfile
from pathlib import Path
out = {}
try:
    from smol_harness.tools import ShellTool, resolve_workspace
    ws = resolve_workspace(tempfile.mkdtemp(prefix="smol-dgx-probe-"))
    shell = ShellTool(ws)
    def run(command):
        r = asyncio.run(shell.execute(command=command, timeout=20)); o = r.output if isinstance(r.output, dict) else {}
        return {"success": bool(r.success), "exit_code": o.get("exit_code"), "stdout": (o.get("stdout") or "")[-160:], "stderr": (o.get("stderr") or "")[-160:], "error": (r.error or "")[:200]}
    out["echo"] = run("echo sandbox-ok")
    out["write_inside"] = run("echo x > inside.txt && cat inside.txt")
    marker = Path.home() / (".smol-dgx-probe-%d" % os.getpid())
    out["write_outside"] = run("touch %s" % marker)
    out["write_outside"]["marker_exists_on_host"] = marker.exists()
    marker.unlink(missing_ok=True)
    out["loopback_connect"] = run("python3 -c \"import socket; socket.create_connection(('127.0.0.1', 1), timeout=2)\"")
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
print("DIAG_JSON=" + json.dumps(out))
'''


def probe(code: str, timeout: float = 300) -> dict:
    if not drv.VENV_PY.exists():
        return {"error": "the project environment does not exist (uv sync --locked)"}
    rc, out, err = drv.run([drv.VENV_PY, "-c", code], timeout=timeout, cwd=drv.REPO)
    line = next((ln for ln in out.splitlines() if ln.startswith("DIAG_JSON=")), None)
    return json.loads(line[len("DIAG_JSON="):]) if line else {"error": f"probe produced no result (exit {rc}): {(err or out)[-300:]}"}


def chk_host(a):
    release = {}
    if Path("/etc/os-release").exists():
        release = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
    ev = {"system": platform.system(), "machine": platform.machine(), "kernel": platform.release(), "os": release.get("PRETTY_NAME", "").strip('"'),
          "dgx_release_file": Path("/etc/dgx-release").read_text()[:300] if Path("/etc/dgx-release").exists() else None}
    if platform.system() != "Linux":
        return FAIL, f"{platform.system()}: this validation is Linux only", ev
    if platform.machine() != "aarch64":
        return WARN, f"{platform.machine()}: not the DGX Spark target (aarch64); the results describe this host", ev
    return PASS, f"Linux aarch64, kernel {ev['kernel']}", ev


def chk_gpu(a):
    gpus, err = drv.nvidia_smi_gpus()
    _, raw, _ = drv.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version", "--format=csv,noheader"])
    ev = {"gpus": gpus, "nvidia_smi_query_raw": raw.strip()[:400], "error": err or None}
    if not gpus:
        return FAIL, f"nvidia-smi gives no GPU ({err})", ev
    if "GB10" not in gpus[0]["name"]:
        return WARN, f"{gpus[0]['name']}: not a GB10 (DGX Spark)", ev
    return PASS, f"{gpus[0]['name']}, driver {gpus[0]['driver']}, memory from {gpus[0]['memory_source'] if 'memory_source' in gpus[0] else 'nvidia-smi'}", ev


def chk_cuda_driver(a):
    version, text = drv.nvidia_cuda_version()
    if version is None:
        return UNVERIFIED, "could not read 'CUDA Version' from nvidia-smi", {}
    return (PASS if version >= 13.0 else FAIL), f"driver supports CUDA {text} (the locked PyTorch bundles the CUDA 13.0 runtime)", {"cuda_version": text}


def chk_memory(a):
    mem = meminfo_mib()
    avail = mem["MemAvailable"] / 1024
    ev = {"total_gib": round(mem["MemTotal"] / 1024, 1), "available_gib": round(avail, 1)}
    if avail < MIN_AVAILABLE_GIB:
        return FAIL, f"{avail:.1f} GiB available (< {MIN_AVAILABLE_GIB}); the GPU shares this memory", ev
    return (WARN if avail < WARN_AVAILABLE_GIB else PASS), f"{ev['total_gib']} GiB total, {avail:.1f} GiB available", ev


def chk_disk(a):
    free = shutil.disk_usage(drv.REPO).free / 2**30
    return (PASS if free >= MIN_DISK_GIB else WARN), f"{free:.0f} GiB free", {"free_gib": round(free, 1)}


def chk_project_env(a):
    ev = probe(PROBE_ENV)
    if "error" in ev:
        return FAIL, ev["error"], ev
    if ev.get("effgen") != "1.3.0":
        return FAIL, f"effGen {ev.get('effgen')} (1.3.0 is pinned)", ev
    if not ev.get("cuda_available"):
        return FAIL, f"torch {ev.get('torch')} cannot use CUDA (CPU fallback is not accepted)", ev
    bad = [k for k in ("matmul_float32", "matmul_bfloat16") if ev.get(k) is not True]
    if bad:
        return FAIL, "a real CUDA computation failed: " + "; ".join(f"{k}={ev.get(k)}" for k in bad), ev
    sm = "sm_%d%d" % tuple(ev["capability"])
    note = "" if sm in ev.get("arch_list", []) else f" (note: {sm} is not in torch's compiled arch list {ev.get('arch_list')}; it ran through PTX JIT or another route)"
    return PASS, f"torch {ev['torch']} (CUDA {ev['torch_cuda_build']}) computed on {ev['device']} {tuple(ev['capability'])}{note}", ev


def chk_sandbox(a):
    ev = probe(PROBE_SANDBOX, timeout=120)
    sysctl = Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns")
    ev["apparmor_restrict_unprivileged_userns"] = sysctl.read_text().strip() if sysctl.exists() else None
    rc, _, err = drv.run(["unshare", "-Urn", "true"], timeout=10)
    ev["unshare_user_net_namespace"] = {"returncode": rc, "stderr": err.strip()[:200]}
    if "error" in ev:
        return FAIL, f"the shell sandbox does not build or start (it fails closed): {ev['error']}", ev
    ok = (ev["echo"]["success"] and "sandbox-ok" in ev["echo"]["stdout"] and ev["write_inside"]["success"] and not ev["write_outside"]["success"]
          and not ev["write_outside"]["marker_exists_on_host"] and not ev["loopback_connect"]["success"])
    return (PASS if ok else FAIL), ("runs commands; writes only inside the workspace; no network" if ok else "the sandbox guarantees were not all shown (see evidence)"), ev


def tag_digest(name: str, tags) -> str | None:
    return next((m.get("digest") for m in (tags or {}).get("models", []) if name in (m.get("name"), m.get("model"))), None)


def chk_ollama(a):
    status, ver, err = drv.http_json("GET", "/api/version", timeout=5)
    if status != 200 or not isinstance(ver, dict):
        return FAIL, f"{drv.OLLAMA_URL}: {err or status}", {}
    _, tags, _ = drv.http_json("GET", "/api/tags", timeout=15)
    names = sorted(m.get("name", "") for m in (tags or {}).get("models", []))
    ev = {"version": ver.get("version"), "installed_models": names, "digest": tag_digest(a.tag, tags)}
    if ev["digest"] is None:
        return FAIL, (f"Ollama {ver.get('version')} does not have '{a.tag}' (installed: {names or 'none'}). Nothing is downloaded here. If its base model is installed, create the fixed-context "
                      f"tag locally (README, step 3)"), ev
    _, shown, _ = drv.http_json("POST", "/api/show", {"model": a.tag}, timeout=15)
    match = re.search(r"num_ctx\s+(\d+)", str((shown or {}).get("parameters", "")))
    ev.update(details=(shown or {}).get("details"), capabilities=(shown or {}).get("capabilities"), num_ctx_in_modelfile=int(match.group(1)) if match else None)
    if not match or int(match.group(1)) != a.num_ctx:
        return FAIL, f"'{a.tag}' does not fix num_ctx {a.num_ctx} in its Modelfile (found {match.group(1) if match else 'none'}); the harness needs them equal", ev
    return PASS, f"Ollama {ver.get('version')}, '{a.tag}' present (digest {str(ev['digest'])[:12]}), num_ctx {a.num_ctx}", ev


def chk_ollama_inference(a):
    """One tiny real generation, then where Ollama placed the model. A model that is not (almost) wholly on the GPU is a WARN with the numbers, not a guess about unified memory."""
    status, data, err = drv.http_json("POST", "/api/generate", {"model": a.tag, "prompt": "Reply with exactly one word: OK", "stream": False, "options": {"num_predict": 16, "temperature": 0}}, timeout=900)
    if status != 200 or not isinstance(data, dict):
        return FAIL, f"Ollama could not run '{a.tag}': {err or status}", {}
    pstatus, resident, perr = drv.ollama_resident()
    entry = next((m for m in resident or [] if drv.entry_name(m) == a.tag), None)
    eval_s = (data.get("eval_duration") or 0) / 1e9
    ev = {"reply": str(data.get("response", ""))[:80], "load_seconds": round((data.get("load_duration") or 0) / 1e9, 1), "generated_tokens": data.get("eval_count"),
          "tokens_per_second": round(data["eval_count"] / eval_s, 1) if eval_s and data.get("eval_count") else None, "ps_entry": entry, "ps_error": perr or None}
    release = drv.release_ollama({"ollama": {"derived_tag": a.tag, "base_model": a.tag}}, "after E0")
    ev["released"] = {k: release.get(k) for k in ("verified", "reason", "still_resident")}
    if entry is None:
        return WARN, "the model answered but /api/ps did not list it, so its placement is not established", ev
    size, vram = entry.get("size") or 0, entry.get("size_vram") or 0
    ev["gpu_fraction"] = round(vram / size, 3) if size else None
    if not size or vram / size < GPU_FRACTION_MIN:
        return WARN, f"Ollama reports {ev['gpu_fraction']} of the model on the GPU (size_vram/size); placement on this memory architecture is NOT established", ev
    return PASS, f"answered at {ev['tokens_per_second']} tok/s; {ev['gpu_fraction']:.0%} on the GPU per /api/ps", ev


E0_CHECKS = [("host", chk_host), ("gpu", chk_gpu), ("cuda_driver", chk_cuda_driver), ("memory", chk_memory), ("disk", chk_disk), ("project_env", chk_project_env),
             ("sandbox", chk_sandbox), ("ollama", chk_ollama), ("ollama_inference", chk_ollama_inference)]


def run_e0(a, checks=None) -> dict:
    results, failed_ids = [], set()
    for cid, fn in checks or E0_CHECKS:
        if cid == "ollama_inference" and "ollama" in failed_ids:
            status, detail, ev = UNVERIFIED, "not run: the Ollama check failed first", {}
        else:
            try:
                status, detail, ev = fn(a)
            except Exception as exc:  # noqa: BLE001 - a check that crashes is reported as such, never as a pass
                status, detail, ev = UNVERIFIED, f"check crashed: {type(exc).__name__}: {str(exc)[:200]}", {}
        if status == FAIL:
            failed_ids.add(cid)
        results.append({"id": cid, "status": status, "detail": detail, "evidence": ev})
    gating = {r["id"]: r["status"] for r in results if r["id"] in GATING}
    return {"experiment": "E0", "t_utc": drv.now(), "tag": a.tag, "num_ctx": a.num_ctx, "git": drv.git_info(), "results": results,
            "gates_e1_e2": all(gating.get(i) in (PASS, WARN) for i in GATING), "gating": gating,
            "note": "sandbox is recorded as Linux evidence; E1 and E2 use only file_operations, so it does not gate them"}


def cmd_e0(a) -> int:
    drv.nvidia_smi_gpus = gpus_tolerant
    out = Path(a.out)
    (out / "diag_E0").mkdir(parents=True, exist_ok=True)
    report = run_e0(a)
    drv.write_json(out / "diag_E0" / "e0.json", report)
    for r in report["results"]:
        print(f"[{r['status']:^10}] {r['id']:17} {r['detail']}")
    print(f"\nE1/E2 may run: {report['gates_e1_e2']}   (report: {out / 'diag_E0' / 'e0.json'})")
    return 1 if any(r["status"] == FAIL for r in report["results"]) else 0


# ------------------------------------------------------------------------------------------------------------------ E1 / E2
def gate(a) -> str | None:
    path = Path(a.out) / "diag_E0" / "e0.json"
    if not path.exists():
        return f"no E0 report at {path}: run `e0` first"
    report = json.loads(path.read_text())
    if (report.get("tag"), report.get("num_ctx")) != (a.tag, a.num_ctx):
        return f"the E0 report was made for tag {report.get('tag')!r}, num_ctx {report.get('num_ctx')}; E1/E2 must use the same ({a.tag!r}, {a.num_ctx})"
    if not report.get("gates_e1_e2"):
        return "E0 did not pass its gating checks: " + ", ".join(f"{k}={v}" for k, v in report.get("gating", {}).items() if v not in (PASS, WARN))
    return None


def open_run(a, name: str):
    out = Path(a.out)
    stage_dir = out / f"diag_{name}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    if not (out / "run.json").exists():
        drv.write_json(out / "run.json", {"run_id": out.name, "created_utc": drv.now(), "profile": "dgx-diagnostic", "git": drv.git_info(),
                                          "note": "DGX Spark diagnostic (experiments/dgx/dgx_diag.py), not an RTX 4070 stage"})
    sr.SINK["path"] = stage_dir / "attempts.jsonl"
    sr.TRACE.clear()
    sr.TRACE.update(sr.fresh_trace(path=stage_dir / "trace.jsonl", stage=f"DIAG-{name}", run_id=out.name, invocation=uuid.uuid4().hex[:12]))
    ctx = SimpleNamespace(run_dir=out, profile_name="dgx-diagnostic", profile={"ollama": {"derived_tag": a.tag, "base_model": a.tag, "num_ctx": a.num_ctx}, "hf": {"quantization": "none", "dtype": "bf16"}})
    try:
        header = sr.stage_header(ctx, SimpleNamespace(production_defaults=False, attempts=a.attempts))
    except Exception as exc:  # noqa: BLE001 - missing header facts are recorded as missing
        header = {"header_error": f"{type(exc).__name__}: {exc}"[:300]}
    header.update(models={"ollama": {"tag": a.tag, "num_ctx": a.num_ctx, "base_url": drv.OLLAMA_URL + "/v1"}, "hf": None}, diagnostic=f"DGX Spark diagnostic {name}; not an RTX stage; not a benchmark")
    sr.trace("stage_start", **header)
    if getattr(a, "capture_prompts", False):  # explicit opt-in; see README
        sr.enable_prompt_capture(stage_dir / "prompts.local.jsonl")
    return ctx, stage_dir


def close_run(a, name: str, ctx, stage_dir: Path, attempts: list, extra: dict) -> int:
    release = drv.release_ollama(ctx.profile, f"after {name}")
    drv.snapshot_ollama(ctx, Path(a.out), f"ollama_after_{name}.json")
    summary = [{"attempt_id": r["attempt_id"], "mechanics_demonstrated": bool(r.get("mechanics_ok", sr.mechanics_ok(r))), "agent_completed": r["verification"]["agent_completed"],
                "task_verified": r["verification"]["task_verified"], "failed_checks": [c["name"] for c in r["verification"]["checks"] if not c["passed"]],
                **({"context_transfer_exact": r["context_transfer"]["exact"], "transfer_failed_checks": [c["name"] for c in r["context_transfer"]["checks"] if not c["passed"]]}
                   if r.get("context_transfer") else {})} for r in attempts]
    drv.write_json(stage_dir / "result.json", {"experiment": name, "runner_overrides": sr.SAMPLING, "tag": a.tag, "num_ctx": a.num_ctx, "attempts": attempts, "summary": summary,
                                              "ollama_release": release, **extra})
    sr.SINK["path"] = None
    sr.TRACE["path"] = None
    sr.TRACE["prompts_path"] = None
    shutil.rmtree(stage_dir / "work", ignore_errors=True)
    for row in summary:
        print(f"{name} {row['attempt_id']}: mechanics demonstrated={row['mechanics_demonstrated']}  agent completed={row['agent_completed']}  TASK VERIFIED={row['task_verified']}"
              + (f"  CONTEXT TRANSFER EXACT={row['context_transfer_exact']}" if "context_transfer_exact" in row else "")
              + (f"  failed checks: {row['failed_checks']}" if row["failed_checks"] else "") + (f"  transfer failed checks: {row['transfer_failed_checks']}" if row.get("transfer_failed_checks") else ""))
    print(f"evidence: {stage_dir}")
    return 0 if all(r["mechanics_demonstrated"] for r in summary) else 1  # about the harness only; whether the TASK was solved is task_verified


def abandon_run(ctx, name: str) -> None:
    """Ctrl-C or a crash mid-attempt: the finished attempts and every trace line so far are already on disk (`attempt_aborted` marks the interrupted one). What would be left
    behind is OUR Ollama model, resident for Ollama's default five minutes on memory the GPU shares, and the open sinks; both are closed here before the error continues."""
    try:
        drv.release_ollama(ctx.profile, f"after {name} (interrupted)")
    finally:
        sr.SINK["path"] = None
        sr.TRACE["path"] = None
        sr.TRACE["prompts_path"] = None


def cmd_e1(a) -> int:
    from smol_harness.config import ModelSpec
    from smol_harness.harness import initialize_runtime
    import smol_harness.models as models_mod

    refusal = gate(a)
    if refusal:
        print(f"E1 refused: {refusal}")
        return 2
    drv.nvidia_smi_gpus = gpus_tolerant
    ctx, stage_dir = open_run(a, "E1")
    initialize_runtime()
    spec = ModelSpec("ollama", a.tag, "server", base_url=drv.OLLAMA_URL + "/v1", context_length=a.num_ctx)
    model = models_mod.build_model(spec)
    try:
        with sr.runner_overrides(False):
            attempts = sr.tool_use_attempts(model, spec, stage_dir / "work", a.attempts)
    except BaseException:
        abandon_run(ctx, "E1")
        raise
    finally:
        model.unload()
    return close_run(a, "E1", ctx, stage_dir, attempts, {})


def cmd_e2(a) -> int:
    from smol_harness.harness import initialize_runtime

    refusal = gate(a)
    if refusal:
        print(f"E2 refused: {refusal}")
        return 2
    drv.nvidia_smi_gpus = gpus_tolerant
    ctx, stage_dir = open_run(a, "E2")
    initialize_runtime()
    try:
        with sr.runner_overrides(False):
            attempts = sr.delegation_attempts(ctx, stage_dir / "work", "ollama_only", E2_WORKER, sr.DELEGATION_TASKS[:1], a.attempts, "X2")  # CLI outputs land in stage_X/
    except BaseException:
        abandon_run(ctx, "E2")
        raise
    return close_run(a, "E2", ctx, stage_dir, attempts, {})


# ------------------------------------------------------------------------------------------------------------------ E3: context transfer between two workers
# The orchestrator is the only channel: worker A reads a value, its report reaches the orchestrator, the orchestrator must put the value into the delegation to worker B,
# and B writes it. Two separate judgements per attempt: `context_transfer.exact` (was the value carried through the delegation arguments, from A's report to B's prompt,
# with no bypass) and `verification` (did B's result file end up right, checked by the runner on disk). Exact transfer does not imply a correct file, nor the reverse.
E3_VALUE = "TRANSFER-4172"
E3_SOURCE = f"The transfer value is {E3_VALUE}.\n"


def has_token(text, token: str = "TRANSFER-4172") -> bool:
    """True only if `token` appears as a whole token: not glued to more letters, digits, hyphens or underscores on either side. So `TRANSFER-41729`, `XTRANSFER-4172`,
    `TRANSFER-4172-b` and `TRANSFER-4172_x` do NOT contain `TRANSFER-4172`, while `TRANSFER-4172.`, `"TRANSFER-4172"` and `value: TRANSFER-4172` do. Substring matching would call
    an altered value an exact transfer (Opus review of E3, B2)."""
    return re.search(r"(?<![A-Za-z0-9_-])" + re.escape(token) + r"(?![A-Za-z0-9_-])", "" if text is None else str(text)) is not None


E3_TASK = ("Do two steps, in order, each with the delegate tool. Step 1: delegate to the worker 'reader' to read the file source.txt and report the transfer value it contains. "
           "Step 2: delegate to the worker 'writer' to create the file result.txt containing exactly that transfer value on a single line. The writer cannot see this conversation "
           "or the reader's report, so put the value itself in what you give the writer. Then tell me what was written.")
E3_CONFIG = """# DGX E3: one Ollama model plays the orchestrator and two workers. Rendered by dgx_diag.py; never edit the rendered copy by hand.
workspace: {workspace}
limits:
  max_delegations: 3
  max_report_chars: 3000
  worker_timeout_s: 300
orchestrator:
  model: {{provider: ollama, model: {tag}, quantization: server, context_length: {num_ctx}, base_url: {base_url}}}
  tools: []
  max_iterations: 8
  system_prompt: >-
    You coordinate worker agents. Split the user's task into self-contained pieces, delegate them
    with the delegate tool, and answer from the workers' reports.
workers:
  - name: reader
    description: reads files in the workspace and reports what they say
    model: {{provider: ollama, model: {tag}, quantization: server, context_length: {num_ctx}, base_url: {base_url}}}
    tools: [file_operations]
    max_iterations: 8
  - name: writer
    description: creates files in the workspace with the content it is given
    model: {{provider: ollama, model: {tag}, quantization: server, context_length: {num_ctx}, base_url: {base_url}}}
    tools: [file_operations]
    max_iterations: 8
"""


def render_e3_config(ctx, a, workspace: Path, name: str) -> Path:
    """JSON strings are valid YAML scalars, so paths and the tag need no further quoting. Written verbatim (see rtx4070.render_config for why not through the redactor)."""
    text = E3_CONFIG.format(workspace=json.dumps(str(workspace)), tag=json.dumps(a.tag), num_ctx=a.num_ctx, base_url=json.dumps(drv.OLLAMA_URL + "/v1"))
    target = ctx.run_dir / "configs_rendered" / f"e3-{name}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def delegation_report(entry: dict) -> dict | None:
    """The report the ORCHESTRATOR received from a delegate call: the model-visible text of the executed call (`tool_log`, full, bounded only by `max_report_chars`), parsed.
    NOT the CLI's `--json` worker preview (cut to 500 characters), and never reconstructed from the orchestrator's later answer. None for a call that returned no report
    (refused, raised, or a text that is not a report)."""
    try:
        report = json.loads(entry.get("visible") or "")
    except (TypeError, ValueError):
        return None
    return report if isinstance(report, dict) and "status" in report else None


def writer_file_access(entry: dict) -> dict:
    """What one writer file call can tell about access to the source, from the recorded arguments and the model-visible observation.
    `file_operations` (effGen builtin/file_ops.py): `read` and `convert` read a file's content (convert also writes a copy, which a later `read` of the copy would show);
    `search` is a filename glob and `list` / `metadata` return names, sizes and times, never content. So:
      content_access   read or convert that names the source (any argument containing `source.`, so `source.txt` and `source.json`)
      observed_value   the value is in what the call returned to the model, whatever the route (a `read` of a copy, say); a write, or a read of the writer's own result.txt, is not a route
      names_source     any call that names the source, content or not (recorded, not a bypass: search/list/metadata cannot reveal the value)
    """
    args = entry.get("args") or {}
    operation = str(args.get("operation", "")).lower()
    strings = [v for v in args.values() if isinstance(v, str)]
    names_source = any("source." in v.lower() for v in strings)
    own_output = Path(str(args.get("path", ""))).name == "result.txt" or operation == "write"
    return {"operation": operation, "names_source": names_source, "content_access": names_source and operation in ("read", "convert"),
            "observed_value": has_token(entry.get("visible")) and not own_output}


def e3_judgements(log: list, doc, calls: list, workspace: Path, tree_before: dict, tree_after: dict) -> dict:
    """Everything E3 decides, from executions, the model prompts and the filesystem. Pure: also what the scripted tests call. Values are matched as whole tokens (`has_token`)."""
    workers = (doc or {}).get("workers", [])
    delegate_runs = [e for e in log if e["tool"] == "delegate"]
    dispatched = [e for e in delegate_runs if e.get("dispatched")]
    name_of = {e["delegation_id"]: e["args"].get("worker") for e in dispatched}
    previews = {w.get("delegation_id"): w for w in workers}
    delegations = []
    for e in delegate_runs:
        report = delegation_report(e)
        delegations.append({"delegation_id": e["delegation_id"], "worker": e["args"].get("worker"), "task": e["args"].get("task"), "context": e["args"].get("context"),
                            "dispatched": e.get("dispatched"), "tool_success": e["success"], "report_status": (report or {}).get("status"),
                            "report_output": (report or {}).get("output"), "report_truncated": (report or {}).get("truncated"),
                            "report_output_cli_preview": (previews.get(e["delegation_id"]) or {}).get("output")})  # the preview is display only: no check reads it
    worker_tools = []
    for e in log:
        if e["tool"] == "delegate":
            continue
        access = writer_file_access(e) if name_of.get(e["delegation_id"]) == "writer" else {}
        worker_tools.append({"worker": name_of.get(e["delegation_id"]), "delegation_id": e["delegation_id"], "tool": e["tool"], "args": e["args"], "success": e["success"],
                             **({"names_source": access["names_source"], "content_access": access["content_access"], "observed_value": access["observed_value"]} if access else {})})
    first = next((d for d in delegations if d["worker"] == "reader" and d["dispatched"]), None)
    second = next((d for d in delegations if d["worker"] == "writer" and d["dispatched"]), None)
    order_ok = bool(first and second and delegations.index(first) < delegations.index(second))
    carried = [field for field in ("task", "context") if second and has_token(second[field])]
    writer_prompts = [c["prompt"] for c in calls if "You are the worker agent 'writer'" in c["prompt"]]
    writer_input = writer_prompts[:1]  # only the writer's FIRST prompt is what the delegation gave it; later prompts also hold its own earlier output (including a value it wrote itself)
    reader_entries = [e for e in log if e["tool"] != "delegate" and name_of.get(e["delegation_id"]) == "reader"]
    reader_reads = [e for e in reader_entries if e["success"] and "source.txt" in str((e["args"] or {}).get("path", "")) and has_token(e.get("visible"))]
    bypass = [t for t in worker_tools if t["worker"] == "writer" and (t["content_access"] or t["observed_value"])]
    result_path = workspace / "result.txt"
    content = result_path.read_text(encoding="utf-8", errors="replace") if result_path.is_file() else None
    changed = sorted(set(tree_after) ^ set(tree_before) | {p for p in tree_before if p in tree_after and tree_before[p] != tree_after[p]})
    transfer = sr.verdict(True, [
        ("reader_read_source", bool(reader_reads), "worker 'reader' really executed a successful file_operations call on source.txt whose result showed it the value"),
        ("reader_delegated_before_writer", order_ok, "the reader was delegated to first and the writer second (both really dispatched)"),
        ("value_absent_before_the_reader", bool(first) and not has_token(f"{first['task']} {first['context']}"), "the first delegation does not contain the value (it can only come from the reader)"),
        ("reader_report_has_value", bool(first) and has_token(first["report_output"]), "the report the orchestrator received from the reader (the executed delegate call's own result, not the CLI preview) contains the value as a whole token"),
        ("value_in_writer_delegation", bool(carried), f"the value is, as a whole token, in the writer delegation's {' and '.join(carried) if carried else 'task/context (it is in neither)'}"),
        ("value_reached_writer_model", any(has_token(p) for p in writer_input), "the value appears, as a whole token, in the first prompt the writer's model was given (what the delegation handed it, before any output of its own)"),
        ("writer_did_not_read_source", not bypass, "no writer file call read the source (read/convert naming source.*) or returned the value to the writer from any file other than its own result.txt")])
    verification = sr.verdict(bool((doc or {}).get("success") and len(workers) >= 2 and all(w.get("status") == "completed" for w in workers[:2])), [
        ("result_file_exists", content is not None, "result.txt exists in the workspace (the runner looked on disk)"),
        ("result_file_exact", content is not None and content.strip() == E3_VALUE, f"result.txt holds exactly {E3_VALUE!r} (found {None if content is None else content.strip()[:60]!r})"),
        ("source_unchanged", tree_after.get("source.txt") == tree_before.get("source.txt"), "source.txt is byte-identical to what the runner wrote"),
        ("no_other_changes", changed == ["result.txt"], f"the only file added or changed is result.txt (changed: {changed})")])
    return {"delegations": delegations, "worker_tool_calls": worker_tools, "value_carried_in": carried, "writer_read_source": bool(bypass),
            "result_file": {"exists": content is not None, "content": None if content is None else content[:200]},
            "context_transfer": {"exact": transfer["task_verified"], "checks": transfer["checks"]}, "verification": verification}


def context_transfer_attempts(ctx, a, work: Path, n: int) -> list:
    import smol_harness.models as models_mod

    attempts = []
    for i in range(n):
        sr.STATE["seed"] = i
        attempt_id = f"ctx-transfer-a{i + 1}"
        workspace = work / attempt_id
        workspace.mkdir(parents=True)
        (workspace / "source.txt").write_text(E3_SOURCE)
        config = render_e3_config(ctx, a, workspace, f"a{i + 1}")
        loads: list = []
        calls: list = []
        real_build = models_mod.build_model
        models_mod.build_model = sr.tracing_build(loads, calls)
        tree_before = sr.tree_manifest(workspace)
        started = time.perf_counter()
        try:
            with sr.attempt_scope(attempt_id, E3_TASK, "orchestrator", seed=i, variant="context-transfer", config_file=str(config.relative_to(ctx.run_dir)), workspace_files_before=sorted(tree_before)):
                with sr.tool_log() as log:
                    code, doc, stderr = sr.run_cli(config, E3_TASK)
        finally:
            models_mod.build_model = real_build
        sr.save_cli(ctx, "X", attempt_id, code, doc, stderr)
        tree_after = sr.tree_manifest(workspace)
        judged = e3_judgements(log, doc, calls, workspace, tree_before, tree_after)
        worker_calls = [c for c in calls if "You are the worker agent" in c["prompt"]]
        orchestrator_calls = [c for c in calls if "You are the worker agent" not in c["prompt"]]
        delegate_runs = [e for e in log if e["tool"] == "delegate"]
        file_calls = [e for e in log if e["tool"] == "file_operations"]
        record = {"variant": "context-transfer", "attempt": i + 1, "attempt_id": attempt_id, "task": E3_TASK, "expected_value": E3_VALUE, "seed": i, "seconds": round(time.perf_counter() - started, 1),
                  "exit_code": code, "cli_success_flag": (doc or {}).get("success"), "termination": (doc or {}).get("termination"), "stop_reason": (doc or {}).get("stop_reason"),
                  "error": (doc or {}).get("error"), "models_built": [getattr(m, "model_name", "?") for m in loads], "worker_records": (doc or {}).get("workers", []),
                  "final_answer": (doc or {}).get("output") or "", "final_mentions_value": E3_VALUE in ((doc or {}).get("output") or ""), "cli_stderr_tail": drv.redact(stderr)[-500:],
                  "generations": sr.digest(calls), "observations": sr.observation_accounting(calls), "actions": sr.action_accounting(calls, log), "trajectory": sr.trajectory_id(calls),
                  "workspace_changes": sr.tree_changes(workspace, tree_before, tree_after), "orchestrator_generations": len(orchestrator_calls), "worker_generations": len(worker_calls),
                  "worker_context_isolated": bool(worker_calls) and not any("Available workers" in c["prompt"] for c in worker_calls), **judged}
        problems = []
        if len([d for d in judged["delegations"] if d["dispatched"]]) < 2:
            problems.append("fewer than two delegations were dispatched")
        if not file_calls:
            problems.append("no worker really executed a file_operations call")
        if not worker_calls:
            problems.append("no worker model ever generated")
        if delegate_runs and not sr.result_reached_model(delegate_runs, [c["prompt"] for c in orchestrator_calls]):
            problems.append("a delegation report is not in any later orchestrator prompt")
        record["any_call_executed"] = bool(delegate_runs and file_calls)
        record["mechanics_problems"] = problems
        record["mechanics_ok"] = not problems
        record["success"] = judged["verification"]["agent_completed"] and judged["verification"]["task_verified"] and judged["context_transfer"]["exact"]
        attempts.append(record)
        sr.keep({"kind": "context-transfer", **record})
        shutil.rmtree(workspace, ignore_errors=True)
    return attempts


def gate_e3(a) -> str | None:
    """E3 needs everything E1/E2 need (the E0 gate for this tag and context) AND an E2 that demonstrated delegation on this same tag: a context-transfer experiment is meaningless
    on a host where a single delegation has not been shown to work."""
    refusal = gate(a)
    if refusal:
        return refusal
    path = Path(a.out) / "diag_E2" / "result.json"
    if not path.exists():
        return f"no E2 result at {path}: run `e2` first (E3 builds on a delegation that is known to work)"
    report = json.loads(path.read_text())
    if (report.get("tag"), report.get("num_ctx")) != (a.tag, a.num_ctx):
        return f"the E2 result was made for tag {report.get('tag')!r}, num_ctx {report.get('num_ctx')}; E3 must use the same ({a.tag!r}, {a.num_ctx})"
    if not report.get("summary") or not any(row.get("mechanics_demonstrated") for row in report["summary"]):
        return "E2 did not demonstrate the delegation mechanics on this tag; fix that before E3"
    return None


def cmd_e3(a) -> int:
    from smol_harness.harness import initialize_runtime

    refusal = gate_e3(a)
    if refusal:
        print(f"E3 refused: {refusal}")
        return 2
    drv.nvidia_smi_gpus = gpus_tolerant
    ctx, stage_dir = open_run(a, "E3")
    initialize_runtime()
    try:
        with sr.runner_overrides(False):
            attempts = context_transfer_attempts(ctx, a, stage_dir / "work", a.attempts)
    except BaseException:
        abandon_run(ctx, "E3")
        raise
    return close_run(a, "E3", ctx, stage_dir, attempts, {"expected_value": E3_VALUE})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dgx_diag.py", description="DGX Spark diagnostic validation (E0 preflight, E1 file read, E2 delegation, E3 context transfer).")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, text in (("e0", cmd_e0, "runtime and sandbox preflight (read-only apart from one tiny generation)"), ("e1", cmd_e1, "real-model file read, independently verified"),
                           ("e2", cmd_e2, "real orchestrator-worker delegation with real file-tool execution"),
                           ("e3", cmd_e3, "real context transfer: worker A reads a value, the orchestrator passes it to worker B, B writes it (needs a passed E0 and an E2 on the same tag)")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--out", required=True, help="results folder for this validation (use the same one for e0, e1 and e2)")
        p.add_argument("--tag", required=True, help="Ollama model tag that is ALREADY installed (with num_ctx fixed in its Modelfile); nothing is downloaded")
        p.add_argument("--num-ctx", type=int, default=8192)
        if name != "e0":
            p.add_argument("--attempts", type=int, default=1, choices=range(1, 4), metavar="1-3", help="bounded attempts, seeds 0..N-1 (default 1)")
            p.add_argument("--capture-prompts", action="store_true", help="also write every prompt (the adapter input) to a LOCAL-ONLY prompts.local.jsonl (mode 0600; never archived)")
        else:
            p.set_defaults(attempts=1)
        p.set_defaults(func=fn)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
