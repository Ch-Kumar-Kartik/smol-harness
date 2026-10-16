"""T3.2a: device placement against effGen's REAL Transformers engine, with a tiny random model.

The model is generated locally into a temp dir (a 2-layer Llama with a word-level tokenizer, a few
hundred KB): no weights are downloaded and no network is used. These tests show what effGen really
does with `device_map`, `require_gpu` and dtype, which is what `build_model`'s placement rules rest on.
"""

from __future__ import annotations

import pytest
import torch
from effgen.models import load_model
from effgen.models.base import GenerationConfig

from smol_harness import models as models_mod
from smol_harness.config import ModelSpec
from smol_harness.models import ModelLoadError, build_model

pytestmark = pytest.mark.filterwarnings("ignore:.*NVIDIA GPU detected.*")
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("tiny-llama")
    names = ["<pad>", "<eos>", "<unk>", "hello", "world", "user", "assistant"] + [f"w{i}" for i in range(57)]
    tok = Tokenizer(models.WordLevel({t: i for i, t in enumerate(names)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    fast.chat_template = ("{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}"
                          "{% if add_generation_prompt %}assistant {% endif %}")
    fast.save_pretrained(d)
    LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        max_position_embeddings=128, eos_token_id=1, pad_token_id=0)).save_pretrained(d)
    return str(d)


def spec(path, **kw) -> ModelSpec:
    return ModelSpec("hf", path, kw.pop("quantization", "none"), **kw)


def params_of(model):
    return list(model.model.parameters())


def test_cpu_mode_puts_every_parameter_on_cpu_and_never_touches_cuda_memory(tiny_model_dir):
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    model = build_model(spec(tiny_model_dir, dtype="fp32", device="cpu"))
    try:
        assert model.device == "cpu" and all(p.device.type == "cpu" for p in params_of(model))
        assert all(p.dtype == torch.float32 for p in params_of(model))
        assert (torch.cuda.memory_allocated() if torch.cuda.is_available() else 0) == before  # no CUDA bytes
        assert isinstance(model.generate("hello world", GenerationConfig(max_tokens=3)).text, str)  # and it runs
        assert (torch.cuda.memory_allocated() if torch.cuda.is_available() else 0) == before  # not even for inference
    finally:
        model.unload()


def test_cpu_mode_honours_an_explicit_bf16(tiny_model_dir):
    model = build_model(spec(tiny_model_dir, dtype="bf16", device="cpu"))
    try:
        assert model.device == "cpu" and all(p.dtype == torch.bfloat16 for p in params_of(model))
    finally:
        model.unload()


def test_cpu_mode_works_on_a_host_without_cuda(tiny_model_dir, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # the fanless edge device
    monkeypatch.setattr(models_mod, "_cuda_available", lambda: False)
    model = build_model(spec(tiny_model_dir, dtype="fp32", device="cpu"))
    try:
        assert model.device == "cpu" and params_of(model)[0].device.type == "cpu"
    finally:
        model.unload()


def test_cuda_mode_fails_fast_on_a_host_without_cuda(tiny_model_dir, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ModelLoadError, match="CUDA as unavailable.*Not falling back to CPU"):
        build_model(spec(tiny_model_dir, dtype="bf16", device="cuda"))


def test_control_effgen_require_gpu_does_not_fail_when_cuda_is_absent(tiny_model_dir, monkeypatch):
    # Why build_model checks CUDA itself: effGen's require_gpu only rejects partial GPU placement.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = load_model(tiny_model_dir, engine="transformers", quantization_bits=None, require_gpu=True,
                       torch_dtype=torch.float32)
    try:
        assert model.device == "cpu"  # silently on CPU, no exception
    finally:
        model.unload()


@cuda_only
@pytest.mark.gpu
def test_cuda_mode_really_places_the_model_on_the_gpu(tiny_model_dir):
    before = torch.cuda.memory_allocated()
    model = build_model(spec(tiny_model_dir, dtype="fp16", device="cuda"))
    try:
        assert model.device == "cuda" and all(p.device.type == "cuda" for p in params_of(model))
        assert torch.cuda.memory_allocated() > before
        assert isinstance(model.generate("hello world", GenerationConfig(max_tokens=3)).text, str)
    finally:
        model.unload()


@cuda_only
@pytest.mark.gpu
def test_control_effgen_uses_cuda_on_a_cuda_host_unless_pinned_to_cpu(tiny_model_dir):
    # Without device_map="cpu" the same load lands on the GPU: pinning is what makes CPU mode real.
    unpinned = load_model(tiny_model_dir, engine="transformers", quantization_bits=None, torch_dtype=torch.float32)
    pinned = load_model(tiny_model_dir, engine="transformers", quantization_bits=None, torch_dtype=torch.float32,
                        require_gpu=False, device_map="cpu", use_flash_attention=False)
    try:
        assert unpinned.device == "cuda" and params_of(unpinned)[0].device.type == "cuda"
        assert pinned.device == "cpu" and params_of(pinned)[0].device.type == "cpu"
    finally:
        unpinned.unload()
        pinned.unload()


@cuda_only
@pytest.mark.gpu
def test_control_effgen_picks_the_cpu_dtype_from_the_host_so_cpu_mode_requires_one(tiny_model_dir, monkeypatch):
    kw = dict(engine="transformers", quantization_bits=None, require_gpu=False, device_map="cpu", use_flash_attention=False)
    on_cuda_host = load_model(tiny_model_dir, **kw)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    on_cpu_host = load_model(tiny_model_dir, **kw)
    try:
        assert params_of(on_cuda_host)[0].dtype == torch.bfloat16  # same CPU placement, different dtype ...
        assert params_of(on_cpu_host)[0].dtype == torch.float32  # ... decided by whether CUDA exists
    finally:
        on_cuda_host.unload()
        on_cpu_host.unload()
