"""T3.2b: real-model CPU smoke test: LiquidAI/LFM2.5-350M through build_model and effGen, CPU only.

OPT-IN. Skipped unless you pass `--run-real-models` (or set SMOL_RUN_REAL_MODELS=1):

    uv run pytest tests/integration/test_liquid_cpu.py --run-real-models -s

The first opt-in run downloads ~714 MB into the Hugging Face cache, pinned to REVISION; later runs reuse
the cached snapshot (nothing is re-downloaded). The model is then loaded from that local path with the
network blocked, so nothing can be fetched while loading or generating.

What it checks (hard assertions): the weights load via `build_model`; every parameter is fp32 on CPU;
no CUDA memory or context is created; `model.generate` and a minimal `Agent.run` return real text.
What it only records (JSON + printed): versions, load time, latencies, memory, temperatures. Success of
the inference is not evidence of thermal safety; temperatures are a before/after snapshot of this host.
No scripted output is ever substituted: a failing inference fails the test.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import tempfile
import torch

pytestmark = pytest.mark.real_model

MODEL_ID = "LiquidAI/LFM2.5-350M"
REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"  # Hub commit inspected 2026-10-16
FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
         "tokenizer_config.json", "chat_template.jinja", "LICENSE"]
PROMPT = "What is the capital of France? Answer in one short sentence."
# Generation settings recommended on the model card: temperature 0.1, top_k 50, repetition_penalty 1.05.
SAMPLING = dict(temperature=0.1, top_k=50, repetition_penalty=1.05)
MAX_NEW_TOKENS = 32
REPORT_PATH = Path(os.environ.get("SMOL_SMOKE_REPORT", str(Path(tempfile.gettempdir()) / "smol-harness-lfm-cpu-smoke.json")))


def _proc_mb() -> dict[str, int]:
    out = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS", "VmHWM")):
            key, value = line.split(":")
            out[key] = int(value.split()[0]) // 1024
    return out


def _thermals() -> dict[str, float]:
    """Every readable thermal zone in deg C (empty if the host exposes none)."""
    temps = {}
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            temps[f"{zone.name}:{(zone / 'type').read_text().strip()}"] = int((zone / "temp").read_text()) / 1000
        except (OSError, ValueError):
            continue
    return temps


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


@pytest.fixture(scope="module")
def snapshot() -> str:
    """Local snapshot path of the pinned revision. Reuses the cache; downloads only if it is missing."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=FILES, local_files_only=True)
    except LocalEntryNotFoundError:
        print(f"\n[real_model] {MODEL_ID}@{REVISION[:8]} is not cached: downloading ~714 MB (explicit opt-in)")
        return snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=FILES)


@pytest.fixture
def offline(monkeypatch):
    """After the snapshot exists, nothing may touch the network."""
    def refuse(*a, **k):
        raise AssertionError("network access attempted during a real-model load/inference")

    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_lfm25_350m_cpu_smoke(snapshot, offline, capsys):
    import effgen
    import transformers
    from effgen import Agent, AgentConfig
    from effgen.models.base import GenerationConfig

    from smol_harness.config import ModelSpec
    from smol_harness.models import build_model

    spec = ModelSpec("hf", snapshot, "none", dtype="fp32", device="cpu")  # existing schema, explicit CPU
    cuda_ready_before = torch.cuda.is_initialized()
    cuda_alloc_before = torch.cuda.memory_allocated()
    cuda_reserved_before = torch.cuda.memory_reserved()
    mem_before, temps_before = _proc_mb(), _thermals()

    t0 = time.perf_counter()
    model = build_model(spec)
    load_s = time.perf_counter() - t0
    mem_loaded, temps_loaded = _proc_mb(), _thermals()
    try:
        params = list(model.model.parameters())
        weight_bytes = sum(p.numel() * p.element_size() for p in params)

        # --- placement: CPU only, no silent GPU use ------------------------------------------------------
        assert model.device == "cpu"
        assert {p.device.type for p in params} == {"cpu"}
        assert {p.dtype for p in params} == {torch.float32}
        assert torch.cuda.memory_allocated() == cuda_alloc_before
        assert torch.cuda.memory_reserved() == cuda_reserved_before
        assert torch.cuda.is_initialized() == cuda_ready_before

        # --- real inference through effGen's engine --------------------------------------------------------
        t0 = time.perf_counter()
        result = model.generate(PROMPT, GenerationConfig(max_tokens=MAX_NEW_TOKENS, **SAMPLING))
        generate_s = time.perf_counter() - t0
        assert isinstance(result.text, str) and result.text.strip()
        assert "<|" not in result.text  # no leaked special tokens

        # --- minimal Agent.run (no tools, no delegation) ------------------------------------------------------
        config = AgentConfig(
            name="lfm-smoke", model=model, tools=[], system_prompt="You are a concise assistant.",
            max_tokens=48, tool_calling_mode="react", capability_probe=False, enable_sub_agents=False,
            enable_memory=False, max_iterations=2, raise_on_error=False, **SAMPLING,
        )
        t0 = time.perf_counter()
        with Agent(config) as agent:
            response = agent.run(PROMPT)
        agent_s = time.perf_counter() - t0
        assert response.success is True and response.termination == "done", (response.output, response.termination)
        assert response.output.strip() and "<|" not in response.output

        # Coherence check: a model loaded with the wrong dtype/weights produces junk, not the capital of France.
        assert "paris" in result.text.lower() and "paris" in response.output.lower()

        # Still CPU-only after inference.
        assert {p.device.type for p in params} == {"cpu"}
        assert torch.cuda.memory_allocated() == cuda_alloc_before and torch.cuda.is_initialized() == cuda_ready_before

        gpu_listed = None
        if shutil.which("nvidia-smi"):
            apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=30).stdout.split()
            gpu_listed = str(os.getpid()) in apps
            assert gpu_listed is False  # the driver does not list this process as using the GPU

        report = {
            "model": {"id": MODEL_ID, "revision": REVISION, "snapshot": snapshot, "parameters": sum(p.numel() for p in params),
                      "reported_context_length": model.get_context_length()},
            "spec": {"provider": spec.provider, "quantization": spec.quantization, "dtype": spec.dtype, "device": spec.device},
            "versions": {"python": sys.version.split()[0], "effgen": effgen.__version__,
                         "transformers": transformers.__version__, "torch": torch.__version__},
            "host": {"cpu": _cpu_name(), "logical_cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
                     "cuda_available_to_torch": torch.cuda.is_available(),
                     "note": "development laptop with active cooling, NOT the fanless edge device"},
            "timing_s": {"load": round(load_s, 2), "model_generate": round(generate_s, 2), "agent_run": round(agent_s, 2)},
            "generation": {"prompt": PROMPT, "max_new_tokens": MAX_NEW_TOKENS, **SAMPLING, "model_generate_text": result.text,
                           "model_generate_tokens_used": result.tokens_used, "agent_output": response.output,
                           "agent_tokens_used": response.tokens_used, "agent_iterations": response.iterations},
            "memory_mb": {"weights_fp32": weight_bytes // 2**20, "rss_before_load": mem_before["VmRSS"],
                          "rss_after_load": mem_loaded["VmRSS"], "rss_end": _proc_mb()["VmRSS"],
                          "peak_rss_process": _proc_mb()["VmHWM"]},
            "cpu_only": {"parameter_devices": sorted({p.device.type for p in params}), "engine_device": model.device,
                         "cuda_bytes_allocated_delta": torch.cuda.memory_allocated() - cuda_alloc_before,
                         "cuda_context_created": torch.cuda.is_initialized() and not cuda_ready_before,
                         "process_listed_by_nvidia_smi": gpu_listed},
            "temperatures_c": {"before": temps_before, "after_load": temps_loaded, "after_inference": _thermals(),
                               "note": "single before/after snapshots of this host's sensors; says nothing about thermal safety"},
        }
        REPORT_PATH.write_text(json.dumps(report, indent=2))
        with capsys.disabled():
            print("\n[real_model] LFM2.5-350M CPU smoke test\n" + json.dumps(report, indent=2))
    finally:
        model.unload()
