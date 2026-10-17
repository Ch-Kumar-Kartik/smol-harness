"""T5.2b: can a REAL model (LiquidAI/LFM2.5-350M, CPU, fp32, ReAct) operate the multi-agent harness?

OPT-IN (real model, CPU only): skipped unless `--run-real-models` (or SMOL_RUN_REAL_MODELS=1).

    uv run pytest tests/integration/test_liquid_harness.py --run-real-models -s

This is an INTEGRATION EXPERIMENT, not a coding benchmark. It never downloads anything: the pinned snapshot must already be
cached (tests/integration/test_liquid_cpu.py caches it) or the module skips. It also skips when free RAM is too low
(the load peaks near 2.5 GiB and this host has no swap). Network access is blocked in-process; every workspace is a
disposable temp directory; nothing else is touched. Results are written to /tmp/smol-harness-lfm-harness.json
(override: SMOL_HARNESS_REPORT) and printed.

Three bounded experiments (3 attempts at most each; C stops at its first success), each reported separately:
  A  real-model tool use       one agent, one tool (`file_operations read`): valid call, effGen executes it, result reaches the model
  B  real-model delegation     the CLI: orchestrator -> `delegate` -> a fresh worker -> a tool -> report -> answer (ONE shared
                               model instance for both agents, so this is NOT heterogeneous-model evidence)
  C  real-model coding         the T5.2a task (calc.add) through the CLI; only attempted if A and B each have a success;
                               success = the repository, inspected and tested by this file, not the model's words

EVIDENCE LABELS (review 002). `raw_output` is what the ENGINE generated; `consumed_output` is that text cut at the stop sequences effGen requested,
i.e. what effGen kept (text after e.g. a model-written "\\nObservation:" is discarded and never reaches the next prompt). Token counts are what the engine
generated, discarded text included. A tool's `raw_result` is what the tool returned; `visible` is what effGen showed the model (tests/evidence.py).
Reports written before this change carry only the raw measurements.

Settings the production config cannot express yet (review F7) are applied by this runner only, and labelled as such in the
report: the card's sampling (temperature 0.1, top_k 50, repetition_penalty 1.05), max_tokens, a fixed seed per attempt, and
max_context_length=32768 (the harness applies no context limit to Hugging Face specs). Success of the assertions below means
the experiment was sound; what the model did is RECORDED, never pinned.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from evidence import call_record, digest, logged, observation_accounting, result_reached_model, trajectories, trajectory_id
from test_e2e_coding import BUGGY, FIXED, TEST_FILE, git, host_pytest, processes_left_in, sha

import smol_harness.__main__ as cli
import smol_harness.harness as harness_mod
import smol_harness.models as models_mod
from smol_harness.config import AgentSpec, ModelSpec
from smol_harness.delegation import DelegateTool
from smol_harness.harness import initialize_runtime
from smol_harness.models import ModelCache
from smol_harness.tools import ShellTool, WorkspaceFileOperations, build_tools

pytestmark = pytest.mark.real_model

MODEL_ID = "LiquidAI/LFM2.5-350M"
REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"
FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
         "tokenizer_config.json", "chat_template.jinja", "LICENSE"]
MIN_AVAILABLE_MIB = 3200  # a load peaks near 2.5 GiB (T3.2b); there is no swap to fall back on
ATTEMPTS = 3
CARD_WINDOW = 32768
MAX_TOKENS = 256
SAMPLING = dict(temperature=0.1, top_k=50, repetition_penalty=1.05)  # the model card's recommendation
REPORT_PATH = Path(os.environ.get("SMOL_HARNESS_REPORT", "/tmp/smol-harness-lfm-harness.json"))
CODE = "7391-QX"  # known only to a file on disk: it can reach the model only through an executed tool
TEST_CMD = "python -m pytest -q -p no:cacheprovider test_calc.py"
CODING_TASK = ("The test test_calc.py fails because add() in calc.py is wrong. Delegate to the worker 'coder': it must read calc.py, "
               f"run the tests with `{TEST_CMD}`, fix add() in calc.py, and run the same command again to confirm. Then report.")
DELEGATION_TASK = "Delegate to the worker 'reader': it must read the file secret.txt and report the access code. Then tell me the code."
# The orchestrator prompt of configs/example.yaml, verbatim: the project's own documented baseline (variant B2). Variant B has none.
EXAMPLE_ORCHESTRATOR_PROMPT = ("You coordinate worker agents. Split the user's task into self-contained pieces, delegate them "
                               "with the delegate tool, and answer from the workers' reports.")
DELEGATION_VARIANTS = {"B_delegation": None, "B2_delegation_example_prompt": EXAMPLE_ORCHESTRATOR_PROMPT}
READ_TASK = "Read the file secret.txt with the file_operations tool and tell me the access code that it contains."


# ---- environment -------------------------------------------------------------------------------------------------------------------------
def _available_mib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return 0


def _rss() -> dict[str, int]:
    out = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS", "VmHWM")):
            key, value = line.split(":")
            out[key] = int(value.split()[0]) // 1024
    return out


@pytest.fixture(scope="module")
def snapshot() -> str:
    """The cached snapshot of the pinned revision. NEVER downloads: a missing checkpoint skips the module."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=FILES, local_files_only=True)
    except LocalEntryNotFoundError:
        pytest.skip(f"{MODEL_ID}@{REVISION[:8]} is not cached and this test never downloads (run test_liquid_cpu.py with approval first)")


@pytest.fixture(autouse=True)
def enough_memory(snapshot):
    """Checked before EVERY experiment: each one loads the model, and a second resident copy would not fit (no swap)."""
    if _available_mib() < MIN_AVAILABLE_MIB:
        pytest.skip(f"only {_available_mib()} MiB RAM available (< {MIN_AVAILABLE_MIB}); no swap, refusing to risk the host")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted during a real-model experiment")

    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.fixture(scope="module")
def report(snapshot):
    import effgen
    import transformers

    data: dict = {
        "environment": {
            "model": MODEL_ID, "revision": REVISION, "snapshot": snapshot, "device": "cpu", "dtype": "fp32 (weights stored bf16, upcast on load)",
            "python": platform.python_version(), "effgen": effgen.__version__, "transformers": transformers.__version__,
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "cpu_threads": os.cpu_count(),
            "tool_calling_mode": "react", "available_mib_at_start": _available_mib(), "rss_mib_at_start": _rss(),
            "runner_overrides_not_in_config": {**SAMPLING, "max_tokens": MAX_TOKENS, "max_context_length": CARD_WINDOW, "seed": "per attempt"},
        },
        "experiments": {},
    }
    # Experiments can run in separate processes (`-k test_a`, `-k test_b`, ...) so that each starts from a clean heap: the
    # freed model's memory is not returned to the OS, and the memory guard then refuses a second load. Results merge by key.
    previous = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {"experiments": {}}
    data["experiments"] = dict(previous.get("experiments", {}))  # a COPY: aliasing made the per-process "new experiments" difference always empty (review 002, R2-08)
    yield data
    data["environment"]["rss_mib_at_end"] = _rss()  # VmHWM is the process peak, including every model load
    data.setdefault("process_peaks", []).append({"experiments": sorted(set(data["experiments"]) - set(previous.get("experiments", {}))),
                                                 "VmHWM_mib": _rss()["VmHWM"]})
    data["process_peaks"] = previous.get("process_peaks", []) + data["process_peaks"]
    REPORT_PATH.write_text(json.dumps(data, indent=2, default=str))
    print(f"\n[real_model] report written to {REPORT_PATH}")
    print(json.dumps({k: v for k, v in data["experiments"].items()}, indent=1, default=str)[:6000])


@pytest.fixture(scope="module")
def spec(snapshot) -> ModelSpec:
    return ModelSpec("hf", snapshot, "none", dtype="fp32", device="cpu")  # explicit CPU, no quantization


@pytest.fixture
def shared_cache():
    """One cache for the attempts of one experiment; closed (model unloaded) before the next experiment loads its own copy."""
    cache = ModelCache()
    yield cache
    cache.close()
    gc.collect()


@pytest.fixture(scope="module", autouse=True)
def runtime():
    initialize_runtime()  # as the CLI does


# ---- instruments --------------------------------------------------------------------------------------------------------------------------
class State:
    seed = 0


@pytest.fixture
def card_settings(monkeypatch):
    """The settings the config cannot express (F7), applied to every agent the harness builds. Labelled in the report."""
    real = harness_mod._agent_config

    def with_card_settings(spec_, model, tools, mode):
        return dataclasses.replace(real(spec_, model, tools, mode), max_tokens=MAX_TOKENS, max_context_length=CARD_WINDOW, seed=State.seed, **SAMPLING)

    monkeypatch.setattr(harness_mod, "_agent_config", with_card_settings)


@pytest.fixture
def tool_log(monkeypatch):
    """Every tool call that REALLY executed (wrapped at `execute`), with its arguments, raw result and the text effGen showed the model.
    A model-written 'Observation:' never appears here."""
    log: list[dict] = []
    for cls in (ShellTool, WorkspaceFileOperations, DelegateTool):
        monkeypatch.setattr(cls, "execute", logged(cls.execute, log))
    return log


@contextlib.contextmanager
def traced(model):
    """Record each generation of `model`: its raw output, what effGen kept of it, and its timing."""
    calls: list[dict] = []
    original = model.generate

    def spy(prompt, config=None, **kw):  # effGen passes `config=` by keyword: the name must match
        started = time.perf_counter()
        result = original(prompt, config, **kw)
        calls.append(call_record("hf", prompt, config, result, time.perf_counter() - started))
        return result

    model.generate = spy
    try:
        yield calls
    finally:
        del model.generate  # back to the class method


def tracing_build(loads: list, calls: list[dict]):
    """A replacement for `models_mod.build_model` that loads the REAL model, remembers the instance, and records every generation."""
    real_build = models_mod.build_model

    def build(model_spec):
        instance = real_build(model_spec)
        loads.append(instance)
        original = instance.generate

        def spy(prompt, config=None, **kw):  # effGen passes `config=` by keyword: the name must match
            started = time.perf_counter()
            result = original(prompt, config, **kw)
            calls.append(call_record("hf", prompt, config, result, time.perf_counter() - started))
            return result

        instance.generate = spy
        return instance

    return build


# ===================================================================================================================================
# A. real-model tool use
# ===================================================================================================================================
def test_a_real_model_tool_use(spec, shared_cache, report, tmp_path, card_settings, tool_log):
    from effgen import Agent

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "secret.txt").write_text(f"The access code is {CODE}.\n")
    model = shared_cache.get(spec)
    assert model.device == "cpu" and {p.device.type for p in model.model.parameters()} == {"cpu"}
    cuda_before = torch.cuda.memory_allocated()
    attempts = []
    for attempt in range(ATTEMPTS):
        State.seed = attempt
        tool_log.clear()
        tools = build_tools(["file_operations"], ws)
        agent_spec = AgentSpec(name="reader", model=spec, tools=("file_operations",), max_iterations=4)
        config = harness_mod._agent_config(agent_spec, model, tools, "react")
        started = time.perf_counter()
        with traced(model) as calls, Agent(config) as agent:
            response = agent.run(READ_TASK)
        seconds = round(time.perf_counter() - started, 1)

        executed = [e for e in tool_log if e["tool"] == "file_operations"]
        read_ok = [e for e in executed if e["success"] and "secret.txt" in str(e["args"].get("path", ""))]
        reached_model = any(CODE in c["prompt"] for c in calls)
        # What effGen shows the model for an executed call (tests/evidence.py): the error text of a failed call, the `data` of a successful file operation.
        # The earlier version compared the tool's raw dict for a successful call, which a model never sees (review 002, R2-03).
        result_seen = result_reached_model(executed, [c["prompt"] for c in calls[1:]])
        final = str(response.output)
        record = {
            "attempt": attempt + 1, "seed": attempt, "seconds": seconds, "model_calls": len(calls), "iterations": response.iterations,
            "termination": getattr(response, "termination", None), "stop_reason": getattr(response, "stop_reason", None),
            "effgen_tool_calls": len(response.tool_calls), "tools_really_executed": executed,
            "valid_call_executed": bool(read_ok), "any_call_executed": bool(executed), "executed_result_reached_model": result_seen,
            "secret_reached_model": reached_model and bool(read_ok),
            "model_wrote_its_own_observation_raw": any("Observation:" in c["raw_output"] for c in calls),
            "model_wrote_its_own_observation_after_effgens_cut": any("Observation:" in c["consumed_output"] for c in calls),
            "observations": observation_accounting(calls), "trajectory": trajectory_id(calls),
            "final_answer": final[:300], "final_has_code": CODE in final,
            "success": bool(read_ok) and reached_model and CODE in final and bool(response.success),
            "generations": digest(calls),
        }
        attempts.append(record)
        assert len(response.tool_calls) == len(executed)  # accounting: effGen's record equals the real executions
        if record["success"]:
            break
    assert torch.cuda.memory_allocated() == cuda_before  # CPU only
    report["experiments"]["A_tool_use"] = {
        "attempts_made": len(attempts), "successes": sum(a["success"] for a in attempts), "attempts": attempts, "trajectories": trajectories(attempts),
        "mechanics": {"attempts_with_an_executed_call": sum(a["any_call_executed"] for a in attempts),
                      "attempts_where_the_result_reached_the_model": sum(a["executed_result_reached_model"] for a in attempts),
                      "attempts_that_ended_with_a_final_answer": sum(a["termination"] == "done" for a in attempts)},
        "rss_mib_after": _rss(), "available_mib_after": _available_mib()}


# ===================================================================================================================================
# B. real-model delegation, through the CLI, one shared model instance
# ===================================================================================================================================
def write_config(root: Path, spec: ModelSpec, *, workspace: Path, worker: dict, orchestrator_tools=(), max_iterations=5,
                 orchestrator_prompt: str | None = None) -> Path:
    model = {"provider": "hf", "model": spec.model, "quantization": "none", "device": "cpu", "dtype": "fp32"}
    data = {"workspace": str(workspace), "limits": {"max_delegations": 2, "worker_timeout_s": 900, "max_report_chars": 2000},
            "orchestrator": {"model": model, "tools": list(orchestrator_tools), "max_iterations": max_iterations,
                             **({"system_prompt": orchestrator_prompt} if orchestrator_prompt else {})},
            "workers": [{"model": model, **worker}]}
    path = root / "harness.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.mark.parametrize("label", list(DELEGATION_VARIANTS))
def test_b_real_model_delegation(label, spec, report, tmp_path, card_settings, tool_log, monkeypatch, capsys):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "secret.txt").write_text(f"The access code is {CODE}.\n")
    reader = {"name": "reader", "description": "reads files in the workspace and reports what they say", "tools": ["file_operations"], "max_iterations": 4}
    attempts = []
    for attempt in range(ATTEMPTS):
        State.seed = attempt
        tool_log.clear()
        loads: list = []
        calls: list[dict] = []
        monkeypatch.setattr(models_mod, "build_model", tracing_build(loads, calls))
        agents_built: list[str] = []
        real_agent = harness_mod.Agent

        class Recording(real_agent):
            def __init__(self, config, *a, **kw):
                agents_built.append(config.name)
                super().__init__(config, *a, **kw)

        monkeypatch.setattr(harness_mod, "Agent", Recording)
        monkeypatch.setattr(sys.modules["smol_harness.delegation"], "Agent", Recording)
        config = write_config(tmp_path, spec, workspace=ws, worker=reader, orchestrator_prompt=DELEGATION_VARIANTS[label])
        started = time.perf_counter()
        rss_before = _rss()
        code = cli.main(["run", "--config", str(config), "--task", DELEGATION_TASK, "--json"])
        out, err = capsys.readouterr()
        seconds = round(time.perf_counter() - started, 1)
        monkeypatch.setattr(harness_mod, "Agent", real_agent)
        monkeypatch.setattr(sys.modules["smol_harness.delegation"], "Agent", real_agent)

        doc = json.loads(out)
        instance = loads[0]
        delegate_runs = [e for e in tool_log if e["tool"] == "delegate"]
        worker_tool_runs = [e for e in tool_log if e["tool"] == "file_operations"]
        orch_prompts = [c["prompt"] for c in calls if "Available workers" in c["prompt"]]
        worker_prompts = [c["prompt"] for c in calls if "You are the worker agent" in c["prompt"]]
        isolated = bool(worker_prompts) and not any("Available workers" in p or "delegate" in p.split("Task:")[0] for p in worker_prompts)
        reached_worker_model = any(CODE in p for p in worker_prompts)
        record = {
            "attempt": attempt + 1, "seed": attempt, "seconds": seconds, "exit_code": code,
            "models_loaded_by_the_run": len(loads), "one_shared_instance": len(loads) == 1,
            "unloaded_after_close": instance.model is None and instance._is_loaded is False,
            "agents_built": agents_built, "delegate_calls_executed": [e["args"] for e in delegate_runs],
            "worker_tool_executions": worker_tool_runs, "worker_prompts": len(worker_prompts), "orchestrator_prompts": len(orch_prompts),
            "worker_context_isolated": isolated, "code_reached_worker_model": reached_worker_model,
            "records": doc["workers"], "success_flag": doc["success"], "termination": doc["termination"], "stop_reason": doc["stop_reason"],
            "final_answer": doc["output"][:300], "final_has_code": CODE in doc["output"], "cli_stderr": err[-600:],
            "generations": digest(calls), "observations": observation_accounting(calls), "trajectory": trajectory_id(calls), "rss_mib_after": _rss(),
        }
        record["success"] = bool(delegate_runs and worker_tool_runs and doc["workers"] and doc["workers"][0]["status"] == "completed"
                                 and reached_worker_model and record["final_has_code"] and doc["success"])
        attempts.append(record)
        # soundness of the experiment (not of the model)
        assert record["one_shared_instance"], "the orchestrator and the worker must share ONE model instance"
        assert record["unloaded_after_close"], "closing the harness must unload the model it loaded"
        assert code in (0, 1, 4), code
        assert len(doc["workers"]) == len(delegate_runs) or not delegate_runs
        if record["success"]:
            break
    report["experiments"][label] = {"orchestrator_system_prompt": DELEGATION_VARIANTS[label], "attempts_made": len(attempts), "successes": sum(a["success"] for a in attempts), "attempts": attempts,
                                            "trajectories": trajectories(attempts), "same_model_for_both_agents": True, "heterogeneous": False}


# ===================================================================================================================================
# C. real-model coding task (only after A and B each have evidence)
# ===================================================================================================================================
def build_repo(root: Path) -> SimpleNamespace:
    repo = root / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text(BUGGY)
    (repo / "test_calc.py").write_text(TEST_FILE)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "initial: add() is wrong")
    return SimpleNamespace(repo=repo, head=git(repo, "rev-parse", "HEAD").strip(), test_sha=sha(repo / "test_calc.py"))


def test_c_real_model_coding(spec, report, tmp_path, card_settings, tool_log, monkeypatch, capsys):
    experiments = report["experiments"]
    b_succeeded = any(v.get("successes") for k, v in experiments.items() if k.startswith("B"))
    if not (experiments.get("A_tool_use", {}).get("successes") and b_succeeded):
        experiments["C_coding"] = {"attempts_made": 0, "note": "not attempted: experiments A and B did not both succeed"}
        pytest.skip("experiment C is only attempted after A and B each have a success")
    coder = {"name": "coder", "description": "reads, edits and tests code in the repository", "tools": ["file_operations", "shell"], "max_iterations": 8}
    attempts = []
    for attempt in range(ATTEMPTS):
        State.seed = attempt
        tool_log.clear()
        root = tmp_path / f"attempt{attempt}"
        root.mkdir()
        project = build_repo(root)
        loads: list = []
        calls: list[dict] = []

        monkeypatch.setattr(models_mod, "build_model", tracing_build(loads, calls))
        config = write_config(project.repo.parent, spec, workspace=project.repo, worker=coder, max_iterations=6)
        started = time.perf_counter()
        code = cli.main(["run", "--config", str(config), "--task", CODING_TASK, "--json"])
        out, err = capsys.readouterr()
        seconds = round(time.perf_counter() - started, 1)

        doc = json.loads(out)
        repo = project.repo
        host = host_pytest(repo)  # the repository's tests, run by us, outside the agent and the sandbox
        status = git(repo, "status", "--porcelain").splitlines()
        source_changed = (repo / "calc.py").read_text() != BUGGY
        shell_runs = [e for e in tool_log if e["tool"] == "shell"]
        test_runs = [e for e in shell_runs if "pytest" in str(e["args"].get("command", ""))]
        identical_reruns = len(test_runs) - len({str(e["args"].get("command")) for e in test_runs})
        record = {
            "attempt": attempt + 1, "seed": attempt, "seconds": seconds, "exit_code": code, "success_flag_of_the_cli": doc["success"],
            "termination": doc["termination"], "stop_reason": doc["stop_reason"], "final_answer": doc["output"][:400], "records": doc["workers"],
            "tool_executions": tool_log[:], "shell_commands": [e["args"].get("command") for e in shell_runs], "pytest_runs": len(test_runs),
            "identical_test_command_rerun": identical_reruns > 0, "calc_py_after": (repo / "calc.py").read_text(), "test_file_unchanged": sha(repo / "test_calc.py") == project.test_sha,
            "git_status": status, "git_diff": git(repo, "diff", "HEAD").splitlines(), "head_unchanged": git(repo, "rev-parse", "HEAD").strip() == project.head,
            "files_in_repo": sorted(p.name for p in repo.iterdir()), "independent_pytest_returncode": host.returncode,
            "independent_pytest_tail": host.stdout.strip().splitlines()[-3:], "processes_left": processes_left_in(repo),
            "generations": digest(calls), "observations": observation_accounting(calls), "trajectory": trajectory_id(calls), "rss_mib_after": _rss(),
        }
        # Success is the repository, never the final answer: the source changed, the test file did not, and the tests pass when WE run them.
        record["success"] = source_changed and record["test_file_unchanged"] and host.returncode == 0
        record["model_claim_matches_repo"] = ("pass" in doc["output"].lower()) == (host.returncode == 0)
        attempts.append(record)
        assert record["processes_left"] == []  # nothing survived in the disposable repo
        assert (repo / "test_calc.py").exists() and record["head_unchanged"]
        if record["success"]:
            break
    experiments["C_coding"] = {"attempts_made": len(attempts), "successes": sum(a["success"] for a in attempts), "attempts": attempts, "trajectories": trajectories(attempts),
                               "same_model_for_both_agents": True, "heterogeneous": False, "reference_fix": FIXED}


# ===================================================================================================================================
# B3 (diagnostic, NOT experiment B): a SCRIPTED orchestrator that delegates correctly, with the REAL model as the worker
# ===================================================================================================================================
def test_b3_real_worker_under_a_scripted_orchestrator(spec, report, tmp_path, card_settings, tool_log, monkeypatch, capsys):
    """Isolates the worker half: when the orchestrator does delegate, can a real model run as a fresh worker, use an allowed
    tool and return a WorkerResult? The orchestrator is scripted, so this is NOT evidence that the real model can delegate."""
    from conftest import ScriptedModel

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "secret.txt").write_text(f"The access code is {CODE}.\n")
    worker_task = "Read the file secret.txt with the file_operations tool and report the access code that it contains."
    delegate_turn = f'Thought: delegate.\nAction: delegate\nAction Input: {json.dumps({"worker": "reader", "task": worker_task})}'
    real_model = {"provider": "hf", "model": spec.model, "quantization": "none", "device": "cpu", "dtype": "fp32"}
    data = {"workspace": str(ws), "limits": {"max_delegations": 2, "worker_timeout_s": 900},
            "orchestrator": {"model": {"provider": "ollama", "model": "scripted-orchestrator:1", "context_length": 8192}, "tools": [], "max_iterations": 4},
            "workers": [{"name": "reader", "description": "reads files in the workspace", "tools": ["file_operations"], "max_iterations": 4, "model": real_model}]}
    config = tmp_path / "harness.yaml"
    config.write_text(yaml.safe_dump(data))
    attempts = []
    for attempt in range(ATTEMPTS):
        State.seed = attempt
        tool_log.clear()
        loads: list = []
        calls: list[dict] = []
        real_build = tracing_build(loads, calls)
        monkeypatch.setattr(models_mod, "build_model", lambda s, _real=real_build: ScriptedModel(
            [delegate_turn, "Thought: done.\nFinal Answer: relayed the worker's report"], name="scripted-orchestrator") if s.provider == "ollama" else _real(s))
        started = time.perf_counter()
        code = cli.main(["run", "--config", str(config), "--task", "get the access code", "--json"])
        out, err = capsys.readouterr()
        doc = json.loads(out)
        record_ = doc["workers"][0] if doc["workers"] else {}
        executed = [e for e in tool_log if e["tool"] == "file_operations"]
        read_ok = [e for e in executed if e["success"] and "secret.txt" in str(e["args"].get("path", ""))]
        worker_prompts = [c["prompt"] for c in calls]
        record = {
            "attempt": attempt + 1, "seed": attempt, "seconds": round(time.perf_counter() - started, 1), "exit_code": code, "models_loaded": len(loads),
            "unloaded_after_close": loads[0].model is None, "delegate_executed": [e["args"] for e in tool_log if e["tool"] == "delegate"],
            "worker_record": record_, "worker_tools_really_executed": executed, "valid_call_executed": bool(read_ok),
            "secret_reached_worker_model": any(CODE in p for p in worker_prompts),
            "worker_prompts_free_of_orchestrator_context": bool(worker_prompts) and not any("Available workers" in p or "relayed" in p for p in worker_prompts),
            "worker_report_has_code": CODE in str(record_.get("output", "")), "orchestrator_final": doc["output"], "generations": digest(calls),
            # measured with the same visible-text comparison as experiment A (review 002, R2-09); the earlier B3 claim was only inferred from the model's replies
            "worker_results_reached_the_model": result_reached_model(executed, worker_prompts[1:]),
            "observations": observation_accounting(calls), "trajectory": trajectory_id(calls),
        }
        record["success"] = bool(read_ok) and record["worker_report_has_code"] and record_.get("status") == "completed"
        attempts.append(record)
        assert len(loads) == 1 and record["unloaded_after_close"]
        if record["success"]:
            break
    report["experiments"]["B3_real_worker_scripted_orchestrator"] = {
        "NOT_experiment_B": "the orchestrator is scripted; this only exercises the worker half with the real model",
        "attempts_made": len(attempts), "successes": sum(a["success"] for a in attempts), "attempts": attempts, "trajectories": trajectories(attempts)}
