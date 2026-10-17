#!/usr/bin/env python
"""Runs ONE experiment stage (B, C, D or E) inside the project environment and writes stage_X/result.json.

Started by rtx4070.py with the project's own interpreter (`.venv/bin/python`), in its own process group, under a hard time limit; it is
not meant to be run by hand. Everything here uses the harness through its normal interfaces (`smol-harness run` is `smol_harness.__main__.main`).

What is recorded is what REALLY happened: tool calls are logged by wrapping the real tools' `execute`, model generations by wrapping the real
model, GPU memory by polling nvidia-smi, and the coding task is judged by inspecting the repository and running its tests ourselves. A model's
own answer is never treated as evidence. Nothing here falls back to CPU: a CPU placement is a stage failure.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import rtx4070 as drv  # the stdlib-only driver next to this file (constants, redaction, Ollama helpers)

CODE = "7391-QX"  # known only to a file on disk: it can reach a model only through an executed tool
READ_TASK = "Read the file secret.txt with the file_operations tool and tell me the access code that it contains."
DELEGATION_TASKS = [
    ("plain", "Delegate to the worker 'reader': it must read the file secret.txt and report the access code. Then tell me the code."),
    ("hinted", "Delegate to the worker 'reader': it must read the file secret.txt (use the relative path secret.txt, it is in the workspace) and report the access code. Then tell me the code."),
]
TEST_CMD = "python -m pytest -q -p no:cacheprovider test_calc.py"
CODING_TASK = ("The test test_calc.py fails because add() in calc.py is wrong. Delegate to the worker 'coder': it must read calc.py, "
               f"run the tests with `{TEST_CMD}`, fix add() in calc.py, and run the same command again to confirm. Then report.")
BUGGY = "def add(a, b):\n    return a - b\n"
TEST_FILE = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
SAMPLING = {"temperature": 0.1, "max_tokens": 768}  # runner overrides the YAML cannot express yet (review F7); --production-defaults disables them
STATE = {"seed": 0}
SINK = {"path": None}  # stage_X/attempts.jsonl: every finished attempt is appended at once, so a stage that is killed still leaves its evidence (review 002, R2-07)
GPU_FRACTION_MIN = 0.95  # an Ollama model that is less than this on the GPU has spilled to CPU


def keep(record: dict) -> None:
    """Append one finished attempt (redacted) to the stage's attempts.jsonl. No-op when no sink is set (direct calls in tests)."""
    if SINK["path"] is not None:
        with open(SINK["path"], "a") as handle:
            handle.write(drv.redact(json.dumps(record, default=str)) + "\n")


# ------------------------------------------------------------------------------------------------------------------ evidence accounting
def consumed_text(raw: str, stops) -> str:
    """The part of a generation that effGen keeps: the text up to the EARLIEST stop sequence it asked for (the rule of effgen.models._adapter_utils.
    apply_stop_sequences, pinned by a test). Whatever follows, for example a model-written `\\nObservation:` after a real `Action:`, is discarded and never
    reaches the next prompt or the answer. A provider that applies the stops itself returns text that is already cut, so raw == consumed there."""
    cut = len(raw)
    for sequence in stops or ():
        index = raw.find(sequence) if sequence else -1
        if index != -1:
            cut = min(cut, index)
    return raw[:cut]


def call_record(provider: str, prompt, config, result, seconds: float) -> dict:
    """One model generation: the RAW engine text, the stop sequences effGen requested, and the text effGen kept. Counts of tokens are what the engine
    GENERATED (including discarded text); they are not a measure of what the agent consumed."""
    stops = list(getattr(config, "stop_sequences", None) or [])
    raw = result.text
    kept = consumed_text(raw, stops)
    return {"provider": provider, "prompt": str(prompt), "raw_output": raw, "stop_sequences": stops, "consumed_output": kept, "discarded_chars": len(raw) - len(kept),
            "finish": result.finish_reason, "seconds": round(seconds, 2), "tokens": getattr(result, "tokens_used", None)}


def observation_accounting(calls: list) -> dict:
    """How many generations contain a self-written `Observation:` in the raw text, and how many still contain one after effGen's cut (those are inline,
    not after a newline, so no stop sequence catches them and they can reach the answer)."""
    return {"generations": len(calls), "raw_with_observation": sum("Observation:" in c["raw_output"] for c in calls),
            "consumed_with_observation": sum("Observation:" in c["consumed_output"] for c in calls), "generated_tokens": sum(c["tokens"] or 0 for c in calls)}


def trajectory_id(calls: list) -> str:
    return hashlib.sha256("\x00".join(c["raw_output"] for c in calls).encode()).hexdigest()[:12]


def trajectories(attempts: list) -> dict:
    """Attempts with different seeds can still produce the same text at low temperature; 3 attempts are then fewer than 3 samples."""
    ids = [a.get("trajectory") for a in attempts]
    return {"attempts": len(ids), "distinct": len(set(ids))}


# ------------------------------------------------------------------------------------------------------------------ instruments
class GpuSampler(threading.Thread):
    """Polls nvidia-smi once a second; reports the peak total memory in use (all processes, Windows included)."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop_flag, self.samples = threading.Event(), []

    def run(self) -> None:
        while not self.stop_flag.is_set():
            gpus, _ = drv.nvidia_smi_gpus()
            if gpus:
                self.samples.append(gpus[0]["used_mib"])
            self.stop_flag.wait(1.0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop_flag.set()
        self.join(timeout=5)

    def summary(self) -> dict:
        s = self.samples
        return {"samples": len(s), "start_used_mib": s[0] if s else None, "peak_used_mib": max(s) if s else None, "end_used_mib": s[-1] if s else None}


def gpu_mem() -> dict:
    import torch

    gpus, _ = drv.nvidia_smi_gpus()
    return {"torch_allocated_mib": torch.cuda.memory_allocated() // 2**20, "torch_reserved_mib": torch.cuda.memory_reserved() // 2**20,
            "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() // 2**20, "smi_used_mib": gpus[0]["used_mib"] if gpus else None}


def model_visible(result) -> str:
    """The text effGen puts in front of the model for a tool result (agent_tool_execution.py): the error text for a failed call; for a successful one
    the `result` field (shell, delegate) or the `data` field of a file operation. NOT the tool's raw output dict: a model never sees that."""
    if not getattr(result, "success", False):
        return str(getattr(result, "error", "") or "")
    out = getattr(result, "output", None)
    if isinstance(out, dict):
        if out.get("result") is not None:
            return str(out["result"])
        if "data" in out and "success" in out:
            return str(out["data"]) if out["data"] is not None else str(out.get("message", ""))
    return str(out if out is not None else "")


def result_reached_model(entries: list, prompts: list) -> bool:
    """True only if EVERY executed call's model-visible text appears in some later prompt. Text we cannot see (empty) proves nothing, so it counts as not reached."""
    if not entries:
        return False
    for entry in entries:
        key = entry.get("visible", "").strip()[:40]
        if not key or not any(key in prompt for prompt in prompts):
            return False
    return True


@contextlib.contextmanager
def tool_log():
    """Every tool call that really executed (wrapped at `execute`). A model-written 'Observation:' never appears here."""
    from smol_harness.delegation import DelegateTool
    from smol_harness.tools import ShellTool, WorkspaceFileOperations

    log: list = []
    originals = {cls: cls.execute for cls in (ShellTool, WorkspaceFileOperations, DelegateTool)}

    def wrap(original):
        async def execute(self, **kwargs):
            started = time.perf_counter()
            result = await original(self, **kwargs)
            shown = result.output.get("result") if isinstance(result.output, dict) and "result" in result.output else result.output
            visible = model_visible(result)
            log.append({"tool": self.name, "args": {k: (v if len(str(v)) < 300 else str(v)[:300] + "...") for k, v in kwargs.items()}, "success": result.success,
                        "seconds": round(time.perf_counter() - started, 2), "error": (result.error or "")[:300] or None, "result": str(shown)[:400], "visible": visible[:400]})
            return result
        return execute

    for cls, original in originals.items():
        cls.execute = wrap(original)
    try:
        yield log
    finally:
        for cls, original in originals.items():
            cls.execute = original


def coresidency_probe(tag: str) -> dict:
    """Measured right after an Ollama generation while the Hugging Face model is loaded: what Ollama and nvidia-smi say about the GPU at that moment."""
    gpus, _ = drv.nvidia_smi_gpus()
    _, resident, _ = drv.ollama_resident()
    entry = next((m for m in resident or [] if drv.entry_name(m) == tag), None) or {}
    size, vram = entry.get("size") or 0, entry.get("size_vram") or 0
    try:
        import torch
        allocated = torch.cuda.memory_allocated() // 2**20
    except Exception:  # noqa: BLE001 - a missing torch is visible as None, not a crash of the measurement
        allocated = None
    return {"ollama_listed": bool(entry), "ollama_gpu_fraction": round(vram / size, 3) if size else None, "ollama_size_vram_mib": vram // 2**20,
            "smi_used_mib": gpus[0]["used_mib"] if gpus else None, "smi_free_mib": gpus[0]["free_mib"] if gpus else None, "hf_torch_allocated_mib": allocated}


def coresidency_verdict(probes: list, placements: list, profile_name: str = "primary") -> dict:
    """Simultaneous residency is OBSERVED only when a measurement taken while the Hugging Face model was loaded shows Ollama's model still (almost) fully on
    the GPU AND every Hugging Face parameter on cuda. Otherwise it is reported as not established, with the reason; nothing is assumed from model sizes."""
    hint = " Resource limit: use the smaller model set (add `--profile fallback` to every command)." if profile_name == "primary" else " Resource limit: this card cannot hold even the smaller pair."
    if not placements:
        return {"observed": False, "reason": "the Hugging Face worker model was never loaded, so there was nothing to be resident together with Ollama's model"}
    if any(p != ["cuda"] for p in placements):
        return {"observed": False, "reason": f"the Hugging Face model was not entirely on cuda ({placements}); refusing to count that as GPU residency"}
    if not probes:
        return {"observed": False, "reason": "no measurement was taken while both models were loaded (the orchestrator did not complete a generation while the worker model was resident), "
                                             "so simultaneous residency is NOT established"}
    good = [p for p in probes if p["ollama_listed"] and (p["ollama_gpu_fraction"] or 0) >= GPU_FRACTION_MIN]
    if not good:
        worst = min((p["ollama_gpu_fraction"] for p in probes if p["ollama_gpu_fraction"] is not None), default=None)
        return {"observed": False, "reason": ("while the Hugging Face model was loaded, " + (f"Ollama kept its model only {worst:.0%} on the GPU (the rest spilled to CPU)" if worst is not None
                                                                                          else "Ollama did not list its model as loaded") + ", so the pair does not fit together." + hint)}
    return {"observed": True, "reason": f"{len(good)} of {len(probes)} measurement(s) taken with both models loaded showed Ollama's model >= {GPU_FRACTION_MIN:.0%} on the GPU and the worker on cuda"}


def tracing_build(loads: list, calls: list, residency: dict | None = None):
    """A `build_model` that loads the REAL model, remembers the instance and records each generation (provider, timing, tokens, raw and consumed text).
    With `residency` (stage D) it also records where the Hugging Face model sits and measures the GPU after each Ollama generation made while it is loaded."""
    import smol_harness.models as models_mod

    real = models_mod.build_model

    def build(spec):
        instance = real(spec)
        loads.append(instance)
        if residency is not None and spec.provider == "hf":
            try:
                residency["placements"].append(sorted({p.device.type for p in instance.model.parameters()}))
            except Exception:  # noqa: BLE001 - unknown placement is recorded as such and fails the verdict
                residency["placements"].append(["unknown"])
            residency["hf_loaded"] = True
            real_unload = instance.unload

            def unload(*a, **k):
                residency["hf_loaded"] = False
                return real_unload(*a, **k)

            instance.unload = unload
        original = instance.generate

        def spy(prompt, config=None, **kw):
            started = time.perf_counter()
            result = original(prompt, config, **kw)
            calls.append(call_record(spec.provider, prompt, config, result, time.perf_counter() - started))
            if residency is not None and spec.provider == "ollama" and residency["hf_loaded"]:
                residency["probes"].append(coresidency_probe(spec.model))
            return result

        instance.generate = spy
        return instance

    return build


@contextlib.contextmanager
def runner_overrides(production_defaults: bool):
    """Sampling the YAML cannot express: temperature 0.1, max_tokens 768, a seed per attempt. Labelled in every result."""
    import smol_harness.harness as harness_mod

    if production_defaults:
        yield
        return
    real = harness_mod._agent_config
    harness_mod._agent_config = lambda spec, model, tools, mode: dataclasses.replace(real(spec, model, tools, mode), seed=STATE["seed"], **SAMPLING)
    try:
        yield
    finally:
        harness_mod._agent_config = real


def digest(calls: list, limit: int = 500) -> list:
    return [{"provider": c.get("provider"), "seconds": c["seconds"], "tokens": c["tokens"], "finish": c["finish"], "raw_output": c["raw_output"][:limit],
             "consumed_output": c["consumed_output"][:limit], "discarded_chars": c["discarded_chars"]} for c in calls]


def make_secret_workspace(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "secret.txt").write_text(f"The access code is {CODE}.\n")
    return root


def mechanics_ok(attempt: dict) -> bool:
    return bool(attempt.get("any_call_executed") and attempt.get("executed_result_reached_model"))


def save_cli(ctx, stage: str, name: str, code: int, doc, stderr: str) -> None:
    drv.write_json(ctx.run_dir / f"stage_{stage}" / f"cli_{name}.json", {"exit_code": code, "stdout_json": doc, "stderr": stderr})


def run_cli(config: Path, task: str) -> tuple[int, dict | None, str]:
    """`smol-harness run --config ... --task ... --json` in this process; returns (exit code, parsed JSON, stderr)."""
    import smol_harness.__main__ as cli

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(["run", "--config", str(config), "--task", task, "--json"])
    try:
        doc = json.loads(out.getvalue())
    except ValueError:
        doc = None
    return code, doc, err.getvalue()


# ------------------------------------------------------------------------------------------------------------------ attempts
def tool_use_attempts(model, model_spec, work: Path, n: int) -> list:
    """One agent, one tool (`file_operations read secret.txt`). Success needs an executed call, the secret reaching the model, and the secret in the answer."""
    import smol_harness.harness as harness_mod
    from effgen import Agent
    from smol_harness.config import AgentSpec
    from smol_harness.tools import build_tools

    attempts = []
    with tool_log() as log:
        for i in range(n):
            STATE["seed"] = i
            log.clear()
            ws = make_secret_workspace(work / f"tooluse-{i}")
            config = harness_mod._agent_config(AgentSpec(name="reader", model=model_spec, tools=("file_operations",), max_iterations=4), model,
                                               build_tools(["file_operations"], ws), "react")
            calls: list = []
            original = model.generate

            def spy(prompt, config=None, _calls=calls, _orig=original, **kw):  # effGen passes `config=` by keyword: the name must match
                started = time.perf_counter()
                result = _orig(prompt, config, **kw)
                _calls.append(call_record(model_spec.provider, prompt, config, result, time.perf_counter() - started))
                return result

            model.generate = spy
            started = time.perf_counter()
            try:
                with Agent(config) as agent:
                    response = agent.run(READ_TASK)
            finally:
                del model.generate
            executed = [e for e in log if e["tool"] == "file_operations"]
            read_ok = [e for e in executed if e["success"] and "secret.txt" in str(e["args"].get("path", ""))]
            seen = result_reached_model(executed, [c["prompt"] for c in calls[1:]])
            final = str(response.output)
            record = {"attempt": i + 1, "seed": i, "seconds": round(time.perf_counter() - started, 1), "iterations": response.iterations, "termination": getattr(response, "termination", None),
                      "any_call_executed": bool(executed), "executed_result_reached_model": seen, "valid_call_executed": bool(read_ok), "tools_really_executed": executed,
                      "no_call_attempted": not executed, "final_answer": final[:300], "final_has_code": CODE in final, "generations": digest(calls),
                      "observations": observation_accounting(calls), "trajectory": trajectory_id(calls),
                      "success": bool(read_ok) and any(CODE in c["prompt"] for c in calls) and CODE in final and bool(response.success)}
            attempts.append(record)
            keep({"kind": "tool_use", **record})
            shutil.rmtree(ws, ignore_errors=True)
            if record["success"]:
                break
    return attempts


def delegation_problems(record: dict, mixed: bool) -> list:
    """Why an attempt does NOT demonstrate the delegation mechanics (empty list = it does). This is the stage D criterion; solving the task is separate.
    A final answer that merely SAYS it delegated proves nothing: every item below is established from executions and measurements."""
    problems = []
    if not record["orchestrator_generations"]:
        problems.append("the orchestrator model never generated")
    if mixed and record["orchestrator_providers"] != ["ollama"]:
        problems.append(f"the orchestrator did not run on Ollama (providers seen: {record['orchestrator_providers']})")
    if not record["delegate_calls_executed"]:
        problems.append("the orchestrator never executed the delegate tool (a final answer that mentions delegation is not evidence)")
    if not record["worker_generations"]:
        problems.append("the worker model never generated, so no worker really ran")
    if mixed and record["worker_generations"] and record["worker_providers"] != ["hf"]:
        problems.append(f"the worker did not run on the Hugging Face model (providers seen: {record['worker_providers']})")
    if not record["worker_records"]:
        problems.append("no worker record came back through the harness")
    if record["delegate_calls_executed"] and not record["delegation_report_reached_orchestrator"]:
        problems.append("the delegation report is not in any later orchestrator prompt, so it did not return through effGen")
    if mixed and not record["coresidency"]["observed"]:
        problems.append(record["coresidency"]["reason"])
    return problems


def delegation_attempts(ctx, work: Path, template: str, worker_kwargs: dict, tasks: list, n: int, label: str, mixed: bool = False) -> list:
    """The delegation task through the real CLI. Two separate judgements per attempt: `mechanics_ok` (the orchestrator and the worker really executed, the report
    came back, and with `mixed` both models were measured on the GPU together) and `success` (the task itself: the worker read the file and the code came back)."""
    import smol_harness.models as models_mod

    attempts = []
    for variant, task in tasks:
        for i in range(n):
            STATE["seed"] = i
            ws = make_secret_workspace(work / f"{label}-{variant}-{i}")
            config = drv.render_config(template, ctx.run_dir, ctx, name=f"{label}-{variant}-a{i + 1}", __WORKSPACE__=str(ws), **worker_kwargs)
            loads: list = []
            calls: list = []
            residency = {"hf_loaded": False, "probes": [], "placements": []} if mixed else None
            real_build = models_mod.build_model
            models_mod.build_model = tracing_build(loads, calls, residency)
            before = drv.nvidia_smi_gpus()[0]
            started = time.perf_counter()
            try:
                with tool_log() as log:
                    code, doc, stderr = run_cli(config, task)
            finally:
                models_mod.build_model = real_build
            seconds = round(time.perf_counter() - started, 1)
            save_cli(ctx, label[0], f"{label}-{variant}-a{i + 1}", code, doc, stderr)
            delegate_runs = [e for e in log if e["tool"] == "delegate"]
            worker_reads = [e for e in log if e["tool"] == "file_operations" and e["success"] and "secret.txt" in str(e["args"].get("path", ""))]
            worker_calls = [c for c in calls if "You are the worker agent" in c["prompt"]]
            orchestrator_calls = [c for c in calls if "You are the worker agent" not in c["prompt"]]
            worker_prompts = [c["prompt"] for c in worker_calls]
            workers = (doc or {}).get("workers", [])
            record = {"variant": variant, "attempt": i + 1, "seed": i, "seconds": seconds, "exit_code": code, "cli_success_flag": (doc or {}).get("success"),
                      "termination": (doc or {}).get("termination"), "stop_reason": (doc or {}).get("stop_reason"), "error": (doc or {}).get("error"),
                      "models_built": [getattr(m, "model_name", "?") for m in loads],
                      "delegate_calls_executed": [e["args"] for e in delegate_runs], "worker_tools_really_executed": [e for e in log if e["tool"] != "delegate"],
                      "worker_records": workers, "worker_context_isolated": bool(worker_prompts) and not any("Available workers" in p for p in worker_prompts),
                      "secret_reached_worker_model": any(CODE in p for p in worker_prompts), "final_answer": ((doc or {}).get("output") or "")[:300],
                      "final_has_code": CODE in ((doc or {}).get("output") or ""), "cli_stderr_tail": drv.redact(stderr)[-500:], "generations": digest(calls),
                      "observations": observation_accounting(calls), "trajectory": trajectory_id(calls),
                      "orchestrator_generations": len(orchestrator_calls), "orchestrator_providers": sorted({c["provider"] for c in orchestrator_calls}),
                      "worker_generations": len(worker_calls), "worker_providers": sorted({c["provider"] for c in worker_calls}),
                      "delegation_report_reached_orchestrator": result_reached_model(delegate_runs, [c["prompt"] for c in orchestrator_calls]),
                      "gpu_used_before_mib": before[0]["used_mib"] if before else None}
            if mixed:
                record["coresidency"] = {**coresidency_verdict(residency["probes"], residency["placements"], ctx.profile_name), "measurements": residency["probes"][:8],
                                         "hf_placements": residency["placements"]}
                if "out of memory" in f"{stderr} {(doc or {}).get('error') or ''}".lower():
                    record["resource_limit"] = ("CUDA ran out of memory while the two models were loaded together." +
                                                (" Use the smaller model set: add `--profile fallback` to every command." if ctx.profile_name == "primary" else ""))
            file_calls = [e for e in log if e["tool"] == "file_operations"]
            record["any_call_executed"] = bool(delegate_runs and file_calls)  # the orchestrator really delegated AND the worker really ran a tool
            # what effGen shows a model for an executed call is the result (or its error text); it must appear in a LATER worker prompt
            record["executed_result_reached_model"] = result_reached_model(file_calls, worker_prompts[1:])
            record["mechanics_problems"] = delegation_problems(record, mixed)
            record["mechanics_ok"] = not record["mechanics_problems"]
            record["success"] = bool(delegate_runs and worker_reads and workers and workers[0].get("status") == "completed" and record["secret_reached_worker_model"]
                                     and record["final_has_code"] and (doc or {}).get("success"))
            attempts.append(record)
            keep({"kind": f"delegation-{label}", **record})
            shutil.rmtree(ws, ignore_errors=True)
            if record["success"] and record["mechanics_ok"]:
                return attempts
    return attempts


def git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "HOME": str(repo.parent / "home")}
    done = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args], cwd=repo, env=env, capture_output=True, text=True, timeout=60)
    return done.stdout


def coding_attempts(ctx, work: Path, n: int) -> list:
    """The calc.add task on a fresh disposable Git repository. Success = the repository, inspected and tested HERE, never the answer."""
    import hashlib

    import smol_harness.models as models_mod

    attempts = []
    for i in range(n):
        STATE["seed"] = i
        root = work / f"coding-{i}"
        repo = root / "repo"
        repo.mkdir(parents=True)
        (repo / "calc.py").write_text(BUGGY)
        (repo / "test_calc.py").write_text(TEST_FILE)
        for step in (("init", "-q", "-b", "main"), ("add", "-A"), ("commit", "-q", "-m", "initial: add() is wrong")):
            git(repo, *step)
        head, test_sha = git(repo, "rev-parse", "HEAD").strip(), hashlib.sha256((repo / "test_calc.py").read_bytes()).hexdigest()
        config = drv.render_config("mixed", ctx.run_dir, ctx, name=f"E-a{i + 1}", __WORKSPACE__=str(repo), __WORKER_NAME__="coder",
                                   __WORKER_DESCRIPTION__="reads, edits and tests code in the repository", __WORKER_TOOLS__="[file_operations, shell]",
                                   __HF_MODEL__=ctx.snapshot)
        loads: list = []
        calls: list = []
        real_build = models_mod.build_model
        models_mod.build_model = tracing_build(loads, calls)
        started = time.perf_counter()
        try:
            with tool_log() as log:
                code, doc, stderr = run_cli(config, CODING_TASK)
        finally:
            models_mod.build_model = real_build
        seconds = round(time.perf_counter() - started, 1)
        save_cli(ctx, "E", f"coding-a{i + 1}", code, doc, stderr)
        host = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_calc.py"], cwd=repo, capture_output=True, text=True,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        shell_cmds = [e["args"].get("command") for e in log if e["tool"] == "shell"]
        test_runs = [c for c in shell_cmds if c and "pytest" in c]
        source_changed = (repo / "calc.py").read_text() != BUGGY
        record = {"attempt": i + 1, "seed": i, "seconds": seconds, "exit_code": code, "cli_success_flag": (doc or {}).get("success"), "termination": (doc or {}).get("termination"),
                  "stop_reason": (doc or {}).get("stop_reason"), "final_answer": ((doc or {}).get("output") or "")[:400], "worker_records": (doc or {}).get("workers", []),
                  "tool_executions": log[:], "shell_commands": shell_cmds, "pytest_runs": len(test_runs), "identical_test_command_rerun": len(test_runs) > len(set(test_runs)),
                  "calc_py_after": (repo / "calc.py").read_text(), "test_file_unchanged": hashlib.sha256((repo / "test_calc.py").read_bytes()).hexdigest() == test_sha,
                  "head_unchanged": git(repo, "rev-parse", "HEAD").strip() == head, "git_status": git(repo, "status", "--porcelain").splitlines(),
                  "files_in_repo": sorted(p.name for p in repo.iterdir()), "independent_pytest_returncode": host.returncode,
                  "independent_pytest_tail": host.stdout.strip().splitlines()[-3:], "cli_stderr_tail": drv.redact(stderr)[-500:], "generations": digest(calls),
                  "observations": observation_accounting(calls), "trajectory": trajectory_id(calls)}
        record["success"] = source_changed and record["test_file_unchanged"] and host.returncode == 0
        record["answer_claims_pass_but_repo_fails"] = ("pass" in record["final_answer"].lower()) and host.returncode != 0
        diff = git(repo, "diff", "HEAD")
        drv.write_text(ctx.run_dir / "stage_E" / f"repo_diff_attempt{i + 1}.diff", diff or "(no changes to tracked files)\n")
        attempts.append(record)
        keep({"kind": "coding", **record})
        shutil.rmtree(root, ignore_errors=True)
        if record["success"]:
            break
    return attempts


# ------------------------------------------------------------------------------------------------------------------ stages
def stage_b(ctx, args) -> dict:
    from smol_harness.config import ModelSpec
    from smol_harness.models import build_model

    spec = ctx.profile["ollama"]
    tag = spec["derived_tag"]
    work = ctx.run_dir / "work" / "B"
    result: dict = {"stage": "B", "evidence": {}, "unverified": [], "runner_overrides": None if args.production_defaults else SAMPLING}
    started = time.time()
    status, data, err = drv.http_json("POST", "/api/generate", {"model": tag, "prompt": "Reply with exactly one word: OK", "stream": False,
                                                                 "options": {"num_predict": 16, "temperature": 0}}, timeout=600)
    if status != 200 or not isinstance(data, dict):
        return {**result, "status": drv.FAIL, "reason": f"Ollama could not run {tag}: {err or status}. Check `ollama ps` and `journalctl -u ollama` (TROUBLESHOOTING.md, 'Ollama')."}
    eval_s = (data.get("eval_duration") or 0) / 1e9
    b0 = {"reply": str(data.get("response", ""))[:80], "load_seconds": round((data.get("load_duration") or 0) / 1e9, 1),
          "generated_tokens": data.get("eval_count"), "tokens_per_second": round(data["eval_count"] / eval_s, 1) if eval_s and data.get("eval_count") else None,
          "total_seconds": round(time.time() - started, 1)}
    pstatus, ps, perr = drv.http_json("GET", "/api/ps", timeout=15)
    entries = [m for m in (ps or {}).get("models", []) if tag in (m.get("name"), m.get("model"))] if isinstance(ps, dict) else []
    drv.write_json(ctx.run_dir / "stage_B" / "ollama_ps.json", ps if pstatus == 200 else {"error": perr or pstatus})
    if not entries:
        return {**result, "status": drv.FAIL, "reason": f"{tag} answered but is not listed by /api/ps ({perr or 'no entry'}), so GPU residency could not be established", "evidence": {"b0": b0}}
    entry = entries[0]
    size, vram = entry.get("size") or 0, entry.get("size_vram") or 0
    b0.update(size_mib=size // 2**20, size_vram_mib=vram // 2**20, gpu_fraction=round(vram / size, 3) if size else None, expires_at=entry.get("expires_at"))
    if "context_length" in entry:
        b0["server_context_length"] = entry["context_length"]
        b0["context_matches_config"] = entry["context_length"] == spec["num_ctx"]
    else:
        result["unverified"].append("the running server's /api/ps does not report context_length in this Ollama version; the Modelfile value was verified by the preflight but the effective runtime window was not")
    result["evidence"]["b0_raw_inference"] = b0
    if not size or vram / size < 0.95:
        return {**result, "status": drv.FAIL, "reason": f"Ollama runs {tag} only {vram / size:.0%} on the GPU (the rest on CPU). Refusing to continue: this stage must run on the GPU. See TROUBLESHOOTING.md, 'Ollama uses the CPU'."}
    if b0.get("context_matches_config") is False:
        return {**result, "status": drv.FAIL, "reason": f"the server reports context_length {entry['context_length']}, the harness is configured for {spec['num_ctx']}"}
    drv.snapshot_ollama(ctx, ctx.run_dir, "ollama_after_B.json")

    from smol_harness.harness import initialize_runtime
    initialize_runtime()
    model_spec = ModelSpec("ollama", tag, "server", base_url=drv.OLLAMA_URL + "/v1", context_length=spec["num_ctx"])
    with GpuSampler() as gpu, runner_overrides(args.production_defaults):
        model = build_model(model_spec)
        b1 = tool_use_attempts(model, model_spec, work, args.attempts)
        b2 = delegation_attempts(ctx, work, "ollama_only", {"__WORKER_NAME__": "reader", "__WORKER_DESCRIPTION__": "reads files in the workspace and reports what they say",
                                                              "__WORKER_TOOLS__": "[file_operations]"}, DELEGATION_TASKS[:1], args.attempts, "B2")
    result["gpu"] = gpu.summary()
    result["evidence"].update(b1_tool_use=b1, b2_delegation_same_ollama_model=b2)
    result["trajectories"] = {"tool_use": trajectories(b1), "delegation": trajectories(b2)}
    mech = [a for a in b1 + b2 if mechanics_ok(a)]
    result["task_success"] = {"tool_use": any(a["success"] for a in b1), "delegation": any(a["success"] for a in b2)}
    result["summary"] = (f"GPU inference OK ({b0['tokens_per_second']} tok/s, {b0['gpu_fraction']:.0%} on GPU); tool-call mechanics shown in {len(mech)} attempt(s); "
                         f"task success: tool use {result['task_success']['tool_use']}, delegation {result['task_success']['delegation']}")
    result["status"] = drv.PASS if mech else drv.FAIL
    if not mech:
        result["reason"] = "Ollama runs on the GPU, but no attempt produced a tool call that the harness executed and whose result reached the model"
    return result


def placement_problems(placed, requested_quantization, quant, text, loaded_mib, after_unload_mib) -> list:
    """What makes a Hugging Face load a FAILURE. Pure, so it can be tested without a GPU. A CPU (or split) placement is always one."""
    problems = []
    if set(placed) != {"cuda"}:
        problems.append(f"parameters live on {sorted(placed)}, not only on cuda: refusing to count a CPU or split placement as a GPU result")
    if requested_quantization != "none" and (quant.get("model_quantization_bits") != 4 or not quant.get("linear4bit_modules")):
        problems.append(f"NF4 quantization was requested but is not in effect ({quant})")
    if not str(text).strip():
        problems.append("the model produced no text")
    if after_unload_mib > max(200, loaded_mib // 10):
        problems.append(f"unloading left {after_unload_mib} MiB allocated (of {loaded_mib})")
    return problems


def stage_c(ctx, args) -> dict:
    import torch
    from effgen.models.base import GenerationConfig
    from smol_harness.config import ModelSpec
    from smol_harness.models import ModelLoadError, build_model

    hf = ctx.profile["hf"]
    work = ctx.run_dir / "work" / "C"
    result: dict = {"stage": "C", "evidence": {}, "unverified": [], "runner_overrides": None if args.production_defaults else SAMPLING}
    if not torch.cuda.is_available():
        return {**result, "status": drv.FAIL, "reason": "torch reports no CUDA device. Refusing to run on CPU."}
    spec = ModelSpec("hf", ctx.snapshot, hf["quantization"], dtype=hf["dtype"], device="cuda")
    from smol_harness.harness import initialize_runtime
    initialize_runtime()
    torch.cuda.reset_peak_memory_stats()
    before = gpu_mem()
    with GpuSampler() as gpu:
        started = time.perf_counter()
        try:
            model = build_model(spec)
        except ModelLoadError as exc:
            result["gpu"] = gpu.summary()
            return {**result, "status": drv.FAIL, "reason": f"the model could not be loaded on the GPU: {drv.redact(str(exc))[:500]}. Nothing was run on CPU.",
                    "evidence": {"profile": ctx.profile_name, "quantization": hf["quantization"], "hint": "If this is a bitsandbytes error, retry with --profile fallback"}}
        load_s = round(time.perf_counter() - started, 1)
        placed = {p.device.type for p in model.model.parameters()}
        dtypes = sorted({str(p.dtype).replace("torch.", "") for p in model.model.parameters()})
        quant: dict = {"requested": hf["quantization"], "model_quantization_bits": getattr(model, "quantization_bits", None)}
        if hf["quantization"] != "none":
            import bitsandbytes as bnb
            quant["linear4bit_modules"] = sum(isinstance(m, bnb.nn.Linear4bit) for m in model.model.modules())
        loaded = gpu_mem()
        t0 = time.perf_counter()
        generated = model.generate("Write one short sentence about the sea.", GenerationConfig(max_tokens=64, temperature=0.1))
        gen_s = time.perf_counter() - t0
        tokens = getattr(generated, "tokens_used", None)
        peak = gpu_mem()
        c2 = tool_use_attempts(model, spec, work, args.attempts)
        model.unload()
        gc.collect()
        torch.cuda.empty_cache()
        after = gpu_mem()
    result["gpu"] = gpu.summary()
    result["evidence"] = {"model": hf["repo"], "revision": hf["revision"], "profile": ctx.profile_name, "load_seconds": load_s, "parameter_devices": sorted(placed), "parameter_dtypes": dtypes,
                          "quantization": quant, "vram_mib": {"before_load": before, "after_load": loaded, "after_generation_peak": peak, "after_unload": after},
                          "generation": {"text": generated.text[:200], "tokens": tokens, "seconds": round(gen_s, 2), "tokens_per_second": round(tokens / gen_s, 1) if tokens and gen_s else None},
                          "c2_tool_use": c2}
    result["trajectories"] = {"tool_use": trajectories(c2)}
    result["task_success"] = {"tool_use": any(a["success"] for a in c2)}
    problems = placement_problems(placed, hf["quantization"], quant, generated.text, loaded["torch_allocated_mib"], after["torch_allocated_mib"])
    result["status"] = drv.FAIL if problems else drv.PASS
    result["reason"] = "; ".join(problems)
    result["summary"] = (f"loaded on {sorted(placed)} in {load_s}s, {loaded['torch_allocated_mib']} MiB allocated, "
                         f"{result['evidence']['generation']['tokens_per_second']} tok/s, tool use success: {result['task_success']['tool_use']}")
    return result


def stage_d(ctx, args) -> dict:
    """Mixed-provider integration. PASS = the mechanics were demonstrated by executions and measurements (delegation_problems); whether the models SOLVED the
    delegated task is recorded separately as `task_success` and never changes the status (approved decision, review 002 R2-05). Stage E is the capability test."""
    result: dict = {"stage": "D", "evidence": {}, "unverified": ["peak VRAM is the machine's total as nvidia-smi reports it; whether that includes Windows processes under WSL2 is not established"],
                    "runner_overrides": None if args.production_defaults else SAMPLING}
    work = ctx.run_dir / "work" / "D"
    with GpuSampler() as gpu, runner_overrides(args.production_defaults):
        attempts = delegation_attempts(ctx, work, "mixed", {"__WORKER_NAME__": "reader", "__WORKER_DESCRIPTION__": "reads files in the workspace and reports what they say",
                                                              "__WORKER_TOOLS__": "[file_operations]", "__HF_MODEL__": ctx.snapshot}, DELEGATION_TASKS, args.attempts, "D", mixed=True)
        status, ps, _ = drv.http_json("GET", "/api/ps", timeout=15)
    result["gpu"] = gpu.summary()
    result["evidence"] = {"attempts": attempts, "ollama_ps_after": ps if status == 200 else None, "mixed": {"orchestrator": ctx.profile["ollama"]["derived_tag"], "worker": ctx.profile["hf"]["repo"]}}
    result["trajectories"] = trajectories(attempts)
    good = [a for a in attempts if a["mechanics_ok"]]
    result["mechanics"] = {"attempts": len(attempts), "attempts_with_full_mechanics": len(good)}
    result["task_success"] = {"delegated_task_solved": any(a["success"] for a in attempts), "attempts_solved": sum(a["success"] for a in attempts)}
    result["status"] = drv.PASS if good else drv.FAIL
    result["summary"] = (f"{len(attempts)} attempt(s); delegation mechanics demonstrated in {len(good)}; "
                         f"the delegated task itself solved: {result['task_success']['delegated_task_solved']} (recorded separately)")
    if not good:
        best = min(attempts, key=lambda a: len(a["mechanics_problems"]), default=None)
        limit = next((a["resource_limit"] for a in attempts if a.get("resource_limit")), None)
        result["reason"] = ((limit + " ") if limit else "") + "no attempt demonstrated the mechanics. " + (
            "Closest attempt lacked: " + "; ".join(best["mechanics_problems"]) if best else "No attempt ran.") + " Stage E is skipped."
    return result


def stage_e(ctx, args) -> dict:
    result: dict = {"stage": "E", "evidence": {}, "unverified": [], "runner_overrides": None if args.production_defaults else SAMPLING}
    # Defence in depth: the driver already refuses to start this stage; the runner re-checks, so running it directly cannot skip the safety gate.
    blocked = [r for r in drv.run_checks(ctx, stage="E") if r["id"] in ("sandbox", "host_isolation") and r["status"] != drv.PASS]
    if blocked:
        return {**result, "status": drv.FAIL, "reason": "refused: model-run shell commands need the sandbox and an environment from which the Windows host is unreachable. "
                + "; ".join(f"{r['id']}: {r['detail']}" for r in blocked), "refused_for_safety": True}
    work = ctx.run_dir / "work" / "E"
    with GpuSampler() as gpu, runner_overrides(args.production_defaults):
        attempts = coding_attempts(ctx, work, args.attempts)
    result["gpu"] = gpu.summary()
    result["evidence"] = {"attempts": attempts}
    result["trajectories"] = trajectories(attempts)
    ok = [a for a in attempts if a["success"]]
    result["task_success"] = bool(ok)
    result["status"] = drv.PASS if ok else drv.FAIL
    result["summary"] = (f"{len(attempts)} attempt(s); repository fixed and its tests pass when WE run them: {bool(ok)}; "
                         f"an answer claimed success while the repository failed: {any(a['answer_claims_pass_but_repo_fails'] for a in attempts)}")
    if not ok:
        result["reason"] = "no attempt left a repository whose tests pass (judged by the independent test run, not by the model's answer)"
    return result


STAGE_FUNCS = {"B": stage_b, "C": stage_c, "D": stage_d, "E": stage_e}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run one stage; called by rtx4070.py.")
    parser.add_argument("stage", choices=sorted(STAGE_FUNCS))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--profile", default="primary")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--production-defaults", action="store_true")
    args = parser.parse_args(argv)
    ctx = drv.Ctx(args.profile, Path(args.run_dir))
    snapshot, err = drv.hf_snapshot_path(ctx.profile)
    ctx.snapshot = snapshot or ""
    if args.stage in ("C", "D", "E") and not snapshot:
        result = {"stage": args.stage, "status": drv.FAIL, "reason": err}
    else:
        started = time.time()
        (Path(args.run_dir) / f"stage_{args.stage}").mkdir(parents=True, exist_ok=True)
        SINK["path"] = Path(args.run_dir) / f"stage_{args.stage}" / "attempts.jsonl"
        try:
            result = STAGE_FUNCS[args.stage](ctx, args)
        except Exception as exc:  # noqa: BLE001 - a crash is a result, with its diagnosis
            import traceback
            result = {"stage": args.stage, "status": drv.FAIL, "reason": f"{type(exc).__name__}: {exc}"[:500], "traceback_tail": traceback.format_exc()[-1500:]}
        finally:
            SINK["path"] = None
        result["runner_seconds"] = round(time.time() - started, 1)
    result["title"] = drv.TITLES[args.stage]
    result["finished_utc"] = drv.now()
    shutil.rmtree(Path(args.run_dir) / "work" / args.stage, ignore_errors=True)  # disposable workspaces never leave the machine
    drv.write_json(Path(args.run_dir) / f"stage_{args.stage}" / "result.json", result)
    print(f"stage {args.stage}: {result['status']} {result.get('reason') or result.get('summary') or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
