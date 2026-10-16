"""T3.2: ModelSpec -> effGen model (`build_model`) and the spec-keyed `ModelCache`.

Two kinds of tests, neither downloads or runs a model:
* unit tests replace `smol_harness.models.load_model` with a recorder and assert the exact kwargs;
* wiring tests run effGen's REAL routing (`effgen.models.load_model`) with only the weight-loading
  `TransformersEngine` stubbed (hf) or nothing stubbed at all (ollama: constructing the
  OpenAI-compatible adapter makes no network call, and sockets are blocked to prove it).
"""

from __future__ import annotations

import socket
import subprocess
import sys
import textwrap

import pytest
import torch
from conftest import ScriptedModel
from effgen import AgentConfig
from effgen.models.base import ModelType
from effgen.models.errors import BackendUnreachableError
from effgen.models.openai_compatible_adapter import OpenAICompatibleAdapter

from smol_harness import models as models_mod
from smol_harness.config import DEFAULT_OLLAMA_URL, ModelSpec, parse_config
from smol_harness.delegation import WorkerRegistry
from smol_harness.models import (
    HF_QUANTIZATION_BITS,
    OLLAMA_ID_ALIAS_PREFIXES,
    ModelCache,
    ModelLoadError,
    build_model,
    loader_args,
)

OLLAMA_TAG = "qwen2.5-coder:7b-instruct-q4_K_M"
HF_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def ollama(model=OLLAMA_TAG, url=DEFAULT_OLLAMA_URL, ctx=8192, **over) -> ModelSpec:
    return ModelSpec("ollama", model, "server", base_url=url, context_length=ctx, **over)


def hf(quant="bnb-4bit-nf4", dtype=None, model=HF_ID, device=None) -> ModelSpec:
    return ModelSpec("hf", model, quant, dtype=dtype, device=device)


# ---- fakes ---------------------------------------------------------------------------------------
class FakeHF(ScriptedModel):
    """Stands in for the local Transformers engine."""

    def __init__(self, quantization_bits=None, name="fake-hf", device="cuda"):
        super().__init__(["x"], name=name)
        self.model_type = ModelType.TRANSFORMERS
        self.quantization_bits = quantization_bits
        self.device = device  # where effGen reports having placed the weights
        self.unloads = 0

    def unload(self):
        self.unloads += 1


class FakeOllama(OpenAICompatibleAdapter):
    """An OpenAICompatibleAdapter without the constructor (no client)."""

    def __init__(self):  # noqa: D107 - intentionally skips the base initialiser
        self.unloads = 0

    def unload(self):
        self.unloads += 1


class Loader:
    """Replacement for effgen's load_model: records every call, builds a matching fake."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.raises: Exception | None = None
        self.returns = None

    def __call__(self, model_id, **kwargs):
        self.calls.append((model_id, kwargs))
        if self.raises is not None:
            raise self.raises
        if self.returns is not None:
            return self.returns
        if "engine" not in kwargs:
            return FakeOllama()
        # Behaves like effGen: the device is CPU only when it was pinned with device_map="cpu".
        return FakeHF(kwargs["quantization_bits"], device="cpu" if kwargs.get("device_map") == "cpu" else "cuda")


@pytest.fixture
def loader(monkeypatch):
    fake = Loader()
    monkeypatch.setattr(models_mod, "load_model", fake)
    monkeypatch.setattr(models_mod, "_cuda_available", lambda: True)  # hermetic: independent of this host's GPU
    return fake


# ---- 1, 2, 6: exact loader arguments ---------------------------------------------------------------
def test_ollama_loader_arguments_are_exact(loader):
    build_model(ollama())
    assert loader.calls == [(OLLAMA_TAG, {
        "provider": "openai_compatible", "base_url": "http://127.0.0.1:11434/v1", "context_length": 8192,
    })]  # nothing about quantization, engine, dtype or devices: the server owns the weights


def test_hf_loader_arguments_are_exact(loader):
    build_model(hf("bnb-4bit-nf4", "bf16"))
    assert loader.calls == [(HF_ID, {
        "engine": "transformers", "quantization_bits": 4, "require_gpu": True, "torch_dtype": torch.bfloat16,
    })]


@pytest.mark.parametrize("url, ctx", [
    ("http://127.0.0.1:11434/v1", 8192), ("http://gpu-box:11434/v1", 4096),
    ("https://ollama.example.org/v1", 32768), ("http://localhost:9999/v1", 256),
])
def test_ollama_endpoint_and_context_length_propagate(loader, url, ctx):
    build_model(ollama(url=url, ctx=ctx))
    _, kwargs = loader.calls[0]
    assert kwargs["base_url"] == url and kwargs["context_length"] == ctx


# ---- 3-5: explicit quantization mapping --------------------------------------------------------------
@pytest.mark.parametrize("quant, bits", [("none", None), ("bnb-8bit", 8), ("bnb-4bit-nf4", 4)])
def test_hf_quantization_is_always_passed_explicitly(loader, quant, bits):
    build_model(hf(quant))
    _, kwargs = loader.calls[0]
    assert "quantization_bits" in kwargs  # present even for `none`: absent would let effGen choose by VRAM
    assert kwargs["quantization_bits"] == bits and (bits is None) == (kwargs["quantization_bits"] is None)
    assert "quantization" not in kwargs  # effGen's string shorthand is never used
    assert "torch_dtype" not in kwargs  # no dtype configured, none invented


def test_mapping_table_covers_exactly_the_configured_hf_quantizations():
    from smol_harness.config import HF_QUANTIZATIONS

    assert set(HF_QUANTIZATION_BITS) == set(HF_QUANTIZATIONS)
    assert HF_QUANTIZATION_BITS == {"none": None, "bnb-8bit": 8, "bnb-4bit-nf4": 4}


@pytest.mark.parametrize("dtype, expected", [("bf16", torch.bfloat16), ("fp16", torch.float16)])
def test_hf_dtype_maps_to_torch_dtype(loader, dtype, expected):
    build_model(hf("none", dtype))
    assert loader.calls[0][1]["torch_dtype"] is expected


def test_hf_always_requires_the_gpu_so_there_is_no_silent_cpu_fallback(loader):
    for quant in HF_QUANTIZATION_BITS:
        build_model(hf(quant, model=f"org/{quant}"))
    assert all(kwargs["require_gpu"] is True for _, kwargs in loader.calls)


# ---- 7-9: cache ----------------------------------------------------------------------------------------------
def test_same_spec_reuses_one_instance_and_loads_once(loader):
    cache = ModelCache()
    a, b = cache.get(ollama()), cache.get(ollama())  # equal but distinct spec objects
    assert a is b and len(loader.calls) == 1 and len(cache) == 1 and ollama() in cache


@pytest.mark.parametrize("a, b", [
    (hf("none"), hf("bnb-8bit")),
    (hf("bnb-8bit"), hf("bnb-4bit-nf4")),
    (hf("none", None), hf("none", "bf16")),  # None means "effGen decides"; not assumed equal to bf16
    (hf("none", "bf16"), hf("none", "fp16")),
    (hf(model="org/a"), hf(model="org/b")),
    (ollama(url="http://a:1/v1"), ollama(url="http://b:1/v1")),
    (ollama(ctx=4096), ollama(ctx=8192)),
    (ollama(model="llama3:8b"), ollama(model="llama3:70b")),
])
def test_different_specs_get_independent_entries(loader, a, b):
    cache = ModelCache()
    ma, mb = cache.get(a), cache.get(b)
    assert ma is not mb and len(cache) == 2 and len(loader.calls) == 2


def test_cache_does_not_mix_providers_even_for_the_same_model_string(loader):
    cache = ModelCache()
    o, h = cache.get(ModelSpec("ollama", "shared", "server", base_url=DEFAULT_OLLAMA_URL, context_length=4096)), cache.get(hf("none", model="shared"))
    assert isinstance(o, OpenAICompatibleAdapter) and h.model_type == ModelType.TRANSFORMERS
    assert o is not h and len(cache) == 2
    assert [("engine" in kw) for _, kw in loader.calls] == [False, True]
    assert cache.get(hf("none", model="shared")) is h  # and each is still found again


def test_independent_caches_do_not_share_instances(loader):
    a, b = ModelCache(), ModelCache()
    assert a.get(ollama()) is not b.get(ollama()) and len(loader.calls) == 2


# ---- 12: identity reuse across workers --------------------------------------------------------------------------
def test_workers_with_the_same_model_spec_share_one_model_instance(loader):
    config = parse_config({
        "workspace": "ws",
        "orchestrator": {"model": {"provider": "ollama", "model": OLLAMA_TAG, "context_length": 8192}, "tools": []},
        "workers": [
            {"name": n, "description": n, "tools": [], "model": {"provider": "hf", "model": HF_ID, "quantization": "bnb-4bit-nf4"}}
            for n in ("coder", "tester")
        ] + [{"name": "reviewer", "description": "r", "tools": [],
              "model": {"provider": "ollama", "model": OLLAMA_TAG, "context_length": 8192}}],
    })
    with ModelCache() as cache:
        registry = WorkerRegistry()
        orchestrator_model = cache.get(config.orchestrator.model)
        for w in config.workers:
            registry.register(w.name, AgentConfig(name=w.name, model=cache.get(w.model), tools=[], enable_memory=False))
        coder1, coder2, tester, reviewer = (registry.create("coder"), registry.create("coder"),
                                           registry.create("tester"), registry.create("reviewer"))
        assert coder1 is not coder2  # fresh agents (T1.2) ...
        assert coder1.model is coder2.model is tester.model  # ... around ONE hf model
        assert reviewer.model is orchestrator_model  # equal ollama specs share too
        assert len(loader.calls) == 2 and len(cache) == 2  # one hf load + one ollama build for 4 agents
        assert coder1.model is not orchestrator_model


# ---- 10, 11: failures and no silent fallback -----------------------------------------------------------------------
@pytest.mark.parametrize("error", [
    RuntimeError("CUDA out of memory"), ValueError("Unknown provider 'x'"), ImportError("No module named 'bitsandbytes'"),
    BackendUnreachableError("openai_compatible", "m", "connection refused", "http://x/v1"),
])
def test_loading_failures_are_reported_with_the_spec_and_the_original_error(loader, error):
    loader.raises = error
    spec = hf("bnb-8bit", "fp16")
    cache = ModelCache()
    with pytest.raises(ModelLoadError) as exc:
        cache.get(spec)
    msg = str(exc.value)
    assert "provider=hf" in msg and HF_ID in msg and "quantization=bnb-8bit" in msg and "dtype=fp16" in msg
    assert type(error).__name__ in msg and str(error) in msg
    assert exc.value.__cause__ is error  # nothing concealed
    assert len(cache) == 0 and spec not in cache  # a failed load leaves no entry


def test_a_failed_load_can_be_retried_by_the_caller_but_is_never_retried_by_the_factory(loader):
    cache = ModelCache()
    loader.raises = RuntimeError("boom")
    with pytest.raises(ModelLoadError):
        cache.get(hf())
    assert len(loader.calls) == 1  # one attempt, same arguments, no downgrade
    loader.raises = None
    assert cache.get(hf()) is not None and len(loader.calls) == 2


def test_no_fallback_after_failure_to_another_provider_quantization_or_device(loader):
    loader.raises = RuntimeError("CUDA out of memory")
    with pytest.raises(ModelLoadError):
        build_model(hf("none", "bf16"))
    assert loader.calls == [(HF_ID, {"engine": "transformers", "quantization_bits": None, "require_gpu": True,
                                     "torch_dtype": torch.bfloat16})]  # not retried with 8/4-bit, CPU or ollama


@pytest.mark.parametrize("spec, wrong", [
    (hf("bnb-4bit-nf4"), FakeOllama()),  # effGen routed an hf id to an OpenAI-compatible server
    (hf("bnb-4bit-nf4"), ScriptedModel(["x"])),  # some other provider's model (ModelType.OPENAI)
    (ollama(), FakeHF()),  # an ollama spec answered by a local engine
    (ollama(), ScriptedModel(["x"])),
])
def test_a_model_of_the_wrong_kind_is_refused(loader, spec, wrong):
    loader.returns = wrong
    cache = ModelCache()
    with pytest.raises(ModelLoadError, match="effGen returned"):
        cache.get(spec)
    assert len(cache) == 0


@pytest.mark.parametrize("quant, loaded_bits", [("none", 4), ("bnb-8bit", None), ("bnb-4bit-nf4", 8), ("bnb-8bit", 4)])
def test_a_substituted_quantization_is_refused(loader, quant, loaded_bits):
    loader.returns = FakeHF(quantization_bits=loaded_bits)
    with pytest.raises(ModelLoadError, match="Refusing a quantization the config did not ask for"):
        build_model(hf(quant))


@pytest.mark.parametrize("spec", [
    ModelSpec("hf", HF_ID, "bnb-4bit-fp4"),
    ModelSpec("hf", HF_ID, "awq"),
    ModelSpec("hf", HF_ID, "server"),
    ModelSpec("hf", HF_ID, "none", dtype="fp32"),
    ModelSpec("hf", HF_ID, "none", context_length=4096),
    ModelSpec("hf", HF_ID, "none", base_url=DEFAULT_OLLAMA_URL),
    ModelSpec("ollama", OLLAMA_TAG, "bnb-8bit", base_url=DEFAULT_OLLAMA_URL, context_length=4096),
    ModelSpec("ollama", OLLAMA_TAG, "none", base_url=DEFAULT_OLLAMA_URL, context_length=4096),
    ModelSpec("ollama", OLLAMA_TAG, "server", base_url=None, context_length=4096),
    ModelSpec("ollama", OLLAMA_TAG, "server", base_url=DEFAULT_OLLAMA_URL, context_length=None),
    ModelSpec("ollama", OLLAMA_TAG, "server", base_url=DEFAULT_OLLAMA_URL, context_length=4096, dtype="bf16"),
    ModelSpec("openai", "gpt-5-nano", "none"),
])
def test_specs_that_break_the_rules_fail_before_anything_is_loaded(loader, spec):
    with pytest.raises(ValueError):
        build_model(spec)
    assert loader.calls == []


@pytest.mark.parametrize("alias", sorted(OLLAMA_ID_ALIAS_PREFIXES))
def test_ollama_tags_that_effgen_would_rewrite_are_refused(loader, alias):
    with pytest.raises(ValueError, match="strip as an adapter prefix"):
        build_model(ollama(model=f"{alias}:latest"))
    assert loader.calls == []


@pytest.mark.parametrize("tag", ["llama3:8b", OLLAMA_TAG, "openai/gpt-oss:20b", "hf.co/org/model:Q4_K_M", "mistral", "server-tools:1b"])
def test_ordinary_ollama_tags_are_accepted_unchanged(loader, tag):
    build_model(ollama(model=tag))
    assert loader.calls[0][0] == tag


# ---- cache lifecycle / ownership ------------------------------------------------------------------------------------
def test_close_unloads_every_owned_model_once_and_is_idempotent(loader):
    cache = ModelCache()
    a, b = cache.get(hf("none")), cache.get(ollama())
    cache.get(hf("none"))
    cache.close()
    cache.close()
    assert (a.unloads, b.unloads) == (1, 1) and len(cache) == 0
    with pytest.raises(RuntimeError, match="closed"):
        cache.get(hf("none"))


def test_close_keeps_unloading_after_a_failure_and_reports_all_failures(loader):
    cache = ModelCache()
    first, second, third = cache.get(hf("none")), cache.get(hf("bnb-8bit")), cache.get(ollama())

    def bad():
        raise RuntimeError("unload failed")

    first.unload = bad
    with pytest.raises(ExceptionGroup) as exc:
        cache.close()
    assert [str(e) for e in exc.value.exceptions] == ["unload failed"]
    assert second.unloads == 1 and third.unloads == 1


def test_context_manager_closes(loader):
    with ModelCache() as cache:
        m = cache.get(hf("none"))
    assert m.unloads == 1


# ---- 13: importing loads nothing ---------------------------------------------------------------------------------------
def test_importing_the_module_loads_no_model_and_no_heavy_library():
    code = textwrap.dedent("""
        import sys
        import effgen.models as em

        def boom(*a, **k):
            raise AssertionError("load_model was called while importing")

        em.load_model = boom
        import smol_harness.models  # noqa: F401
        heavy = [m for m in ("torch", "transformers", "bitsandbytes", "accelerate") if m in sys.modules]
        assert not heavy, f"heavy libraries imported: {heavy}"
        print("ok")
    """)
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert run.returncode == 0 and "ok" in run.stdout, run.stderr


# =====================================================================================================================
# Wiring against effGen's REAL routing (no `load_model` stub)
# =====================================================================================================================
@pytest.fixture
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")

    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_real_effgen_builds_the_ollama_adapter_offline(no_network):
    model = build_model(ollama(url="http://127.0.0.1:11434/v1", ctx=8192))
    assert type(model) is OpenAICompatibleAdapter
    assert model.model_name == OLLAMA_TAG  # the full tag, colon included
    assert model.base_url == "http://127.0.0.1:11434/v1"
    assert model.get_context_length() == 8192 and model.is_loaded()
    other = build_model(ollama(url="http://gpu-box:11434/v1", ctx=4096))
    assert other.base_url == "http://gpu-box:11434/v1" and other.get_context_length() == 4096
    assert model.get_context_length() == 8192  # independent instances


class StubEngine(ScriptedModel):
    """Replaces only the weight-loading engine class; effGen's routing around it is real."""

    created: list[tuple[str, dict]] = []

    def __init__(self, model_name, **params):
        super().__init__(["x"], name=model_name)
        self.model_type = ModelType.TRANSFORMERS
        self.quantization_bits = params.get("quantization_bits", "<absent>")
        self.device = "cpu" if params.get("device_map") == "cpu" else "cuda"
        StubEngine.created.append((model_name, params))

    def is_loaded(self):
        return True


@pytest.fixture
def stub_engine(monkeypatch):
    import effgen.models.transformers_engine as te

    StubEngine.created = []
    monkeypatch.setattr(te, "TransformersEngine", StubEngine)
    return StubEngine


@pytest.mark.parametrize("quant, bits, dtype", [
    ("none", None, None), ("none", None, "bf16"), ("bnb-8bit", 8, "fp16"), ("bnb-4bit-nf4", 4, "bf16"), ("bnb-4bit-nf4", 4, None),
])
def test_real_routing_delivers_exactly_the_requested_engine_parameters(stub_engine, monkeypatch, quant, bits, dtype):
    monkeypatch.setattr(models_mod, "_cuda_available", lambda: True)
    model = build_model(hf(quant, dtype))
    (name, params), = stub_engine.created
    assert name == HF_ID and model.model_type == ModelType.TRANSFORMERS
    assert params["quantization_bits"] is bits and params["require_gpu"] is True
    assert ("torch_dtype" in params) == (dtype is not None)
    if dtype:
        assert params["torch_dtype"] is {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    assert "quantization" not in params and "load_in_4bit" not in params


def test_control_effgen_would_pick_quantization_itself_if_it_were_not_passed(stub_engine, monkeypatch):
    # The hazard `quantization_bits` is always passed to avoid (audit §1.7): with the key absent,
    # effGen's VRAM heuristic decides. Pin that heuristic to "4-bit" (what it picks on this 8 GB GPU).
    from effgen.models import load_model
    from effgen.models.model_loader import ModelLoader

    monkeypatch.setattr(models_mod, "_cuda_available", lambda: True)
    monkeypatch.setattr(ModelLoader, "_auto_select_quantization_bits", lambda self: 4)
    load_model(HF_ID, engine="transformers", require_gpu=True)  # what we do NOT do
    build_model(hf("none"))
    absent_case, ours = stub_engine.created
    assert absent_case[1]["quantization_bits"] == 4  # silently decided by effGen
    assert ours[1]["quantization_bits"] is None  # explicit `none` survives the heuristic


@pytest.mark.parametrize("alias", sorted(OLLAMA_ID_ALIAS_PREFIXES))
def test_control_effgen_really_rewrites_alias_prefixed_ids(no_network, alias):
    # Why the guard exists: effGen drops the prefix and asks the server for a different model.
    from effgen.models import load_model

    m = load_model(f"{alias}:latest", provider="openai_compatible", base_url=DEFAULT_OLLAMA_URL, context_length=4096)
    assert m.model_name == "latest"
    with pytest.raises(ValueError):
        build_model(ollama(model=f"{alias}:latest"))


def test_real_effgen_leaves_ordinary_ollama_tags_alone(no_network):
    for tag in ("llama3:8b", OLLAMA_TAG, "mistral"):
        assert build_model(ollama(model=tag)).model_name == tag


# =====================================================================================================================
# T3.2a: explicit hf device selection
# =====================================================================================================================
def test_cpu_loader_arguments_are_exact(loader):
    build_model(hf("none", "fp32", device="cpu"))
    assert loader.calls == [(HF_ID, {
        "engine": "transformers", "quantization_bits": None, "require_gpu": False, "device_map": "cpu",
        "use_flash_attention": False, "torch_dtype": torch.float32,
    })]


def test_cpu_bf16_loader_arguments(loader):
    build_model(hf("none", "bf16", device="cpu"))
    kwargs = loader.calls[0][1]
    assert kwargs["torch_dtype"] is torch.bfloat16 and kwargs["device_map"] == "cpu" and kwargs["require_gpu"] is False


def test_cuda_loader_arguments_are_unchanged_and_explicit_cuda_equals_the_default(loader):
    build_model(hf("bnb-8bit", "fp16", device="cuda"))
    build_model(hf("bnb-8bit", "fp16"))
    first, second = loader.calls
    assert first == second == (HF_ID, {"engine": "transformers", "quantization_bits": 8, "require_gpu": True,
                                       "torch_dtype": torch.float16})
    assert "device_map" not in first[1]  # effGen's own CUDA placement; nothing pinned


def test_cpu_mode_never_asks_effgen_for_the_gpu_or_a_quantization(loader):
    build_model(hf("none", "fp32", device="cpu"))
    kwargs = loader.calls[0][1]
    assert kwargs["require_gpu"] is False and kwargs["device_map"] == "cpu" and kwargs["quantization_bits"] is None
    assert "device" not in kwargs and "quantization" not in kwargs


def test_cpu_mode_refuses_a_model_effgen_placed_on_cuda(loader):
    loader.returns = FakeHF(device="cuda")
    with pytest.raises(ModelLoadError, match=r"placed .* on 'cuda', expected 'cpu'"):
        build_model(hf("none", "fp32", device="cpu"))


@pytest.mark.parametrize("placed", ["cpu", "mixed", None])
def test_cuda_mode_refuses_a_model_that_is_not_fully_on_cuda(loader, placed):
    loader.returns = FakeHF(quantization_bits=4, device=placed)
    with pytest.raises(ModelLoadError, match="Refusing to run on a different device"):
        build_model(hf("bnb-4bit-nf4"))


def test_cuda_unavailable_fails_fast_without_falling_back(loader, monkeypatch):
    monkeypatch.setattr(models_mod, "_cuda_available", lambda: False)
    for quant in ("none", "bnb-8bit", "bnb-4bit-nf4"):
        with pytest.raises(ModelLoadError, match="CUDA as unavailable.*Not falling back to CPU"):
            build_model(hf(quant, device="cuda"))
    with pytest.raises(ModelLoadError):
        ModelCache().get(hf())  # device omitted means cuda
    assert loader.calls == []  # nothing was loaded, nothing retried on CPU


def test_cpu_mode_does_not_need_cuda(loader, monkeypatch):
    monkeypatch.setattr(models_mod, "_cuda_available", lambda: False)
    assert build_model(hf("none", "fp32", device="cpu")).device == "cpu"  # the fanless edge device case
    assert len(loader.calls) == 1


def test_ollama_never_consults_cuda(loader, monkeypatch):
    monkeypatch.setattr(models_mod, "_cuda_available", lambda: (_ for _ in ()).throw(AssertionError("asked")))
    build_model(ollama())
    assert loader.calls[0][1] == {"provider": "openai_compatible", "base_url": DEFAULT_OLLAMA_URL, "context_length": 8192}


def test_cache_is_device_sensitive(loader):
    cache = ModelCache()
    cpu, cuda = cache.get(hf("none", "bf16", device="cpu")), cache.get(hf("none", "bf16", device="cuda"))
    assert cpu is not cuda and cpu.device == "cpu" and cuda.device == "cuda" and len(cache) == 2
    assert cache.get(hf("none", "bf16")) is cuda  # omitted device == explicit cuda: same entry, no extra load
    assert cache.get(hf("none", "bf16", device="cpu")) is cpu and len(loader.calls) == 2


@pytest.mark.parametrize("spec", [
    hf("none", "fp32", device="gpu"),
    hf("none", "fp32", device="cuda:0"),
    hf("none", "bf16", device="mps"),
    hf("bnb-8bit", "bf16", device="cpu"),
    hf("bnb-4bit-nf4", "bf16", device="cpu"),
    hf("none", None, device="cpu"),  # CPU needs an explicit dtype
    hf("none", "fp16", device="cpu"),
    hf("none", "fp32", device="cuda"),
    hf("none", "fp32"),  # device omitted = cuda
    hf("none", "int4", device="cpu"),
])
def test_invalid_device_specs_fail_before_anything_is_loaded(loader, spec):
    with pytest.raises(ValueError):
        build_model(spec)
    assert loader.calls == []


def test_real_routing_delivers_cpu_pinning_to_the_engine(stub_engine):
    model = build_model(hf("none", "fp32", device="cpu"))
    (name, params), = stub_engine.created
    # effGen only fills device_map in when it is absent ("auto" on a CUDA host); our pin must survive routing.
    assert params["device_map"] == "cpu" and params["require_gpu"] is False
    assert params["torch_dtype"] is torch.float32 and params["quantization_bits"] is None
    assert params["use_flash_attention"] is False and model.device == "cpu"
