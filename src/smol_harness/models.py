"""ModelSpec -> effGen model instance, plus a spec-keyed cache (docs/architecture.md §8).

All loading goes through `effgen.models.load_model`; nothing here runs inference or
re-implements a provider. The rules encode effGen 1.3.0 behaviour verified in
docs/effgen-api-audit.md §1.5-§1.7 and T3.2:

* Nothing is guessed. effGen picks 4/8-bit from free VRAM when `quantization_bits` is
  absent, so it is ALWAYS passed for hf (`None` means no quantization).
* Nothing is silently substituted or retried: one load attempt with exactly the requested
  arguments, then the result is checked to be the kind of model that was asked for, quantized
  as asked, and placed on the device that was asked for.
* Device is explicit (`device: cuda | cpu`). effGen's engine has no device parameter: it uses CUDA
  whenever `torch.cuda.is_available()`. CPU is therefore enforced with `device_map="cpu"` (the
  supported placement control), `use_flash_attention=False` and an explicit dtype, and CUDA is
  enforced by our own availability check because effGen's `require_gpu=True` does NOT fail when
  CUDA is absent (it only rejects partial GPU placement; verified T3.2a).
* Importing this module loads no model and imports no torch.

Ownership: a `ModelCache` owns the models it builds. Agents only borrow them (closing an
effGen `Agent` does not unload its model, audit §1.4), so one instance can serve the
orchestrator and any number of workers. Whoever creates the cache calls `close()`.
"""

from __future__ import annotations

from typing import Any

from effgen.models import load_model
from effgen.models.base import BaseModel, ModelType
from effgen.models.openai_compatible_adapter import OpenAICompatibleAdapter

from smol_harness.config import HF_QUANTIZATIONS, OLLAMA_QUANTIZATIONS, ModelSpec, hf_placement_problem

# config quantization -> effGen `quantization_bits` (audit §1.7). Exhaustive on purpose: no default.
HF_QUANTIZATION_BITS: dict[str, int | None] = {"none": None, "bnb-8bit": 8, "bnb-4bit-nf4": 4}
_TORCH_DTYPE_NAMES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}

# effGen strips a leading "<alias>:" from the model id when it names the adapter being used
# (model_loader_routing.py, "duplicate prefix" handling), so an Ollama tag such as
# "server:latest" would silently be requested as "latest" (verified T3.2). Refuse such ids.
OLLAMA_ID_ALIAS_PREFIXES = frozenset({
    "openai", "openai_compatible", "openai-compatible", "openai_compat", "oai_compatible",
    "compatible", "server", "vllm_server", "local_server",
})


class ModelLoadError(RuntimeError):
    """A model could not be built as specified. The original error is `__cause__`."""


def _describe(spec: ModelSpec) -> str:
    bits = [f"provider={spec.provider}", f"model={spec.model!r}", f"quantization={spec.quantization}"]
    if spec.base_url:
        bits.append(f"base_url={spec.base_url}")
    if spec.context_length is not None:
        bits.append(f"context_length={spec.context_length}")
    if spec.dtype:
        bits.append(f"dtype={spec.dtype}")
    if spec.device:
        bits.append(f"device={spec.device}")
    return "model spec (" + ", ".join(bits) + ")"


def _cuda_available() -> bool:
    """Whether torch can use CUDA here (lazy import; tests replace this function)."""
    import torch

    return bool(torch.cuda.is_available())


def _torch_dtype(name: str, spec: ModelSpec) -> Any:
    try:
        import torch  # lazy: importing this module must stay cheap
    except ImportError as exc:
        raise ModelLoadError(f"cannot load {_describe(spec)}: torch is not importable ({exc})") from exc
    return getattr(torch, _TORCH_DTYPE_NAMES[name])


def loader_args(spec: ModelSpec) -> tuple[str, dict[str, Any]]:
    """The exact `(model_id, kwargs)` for `effgen.models.load_model`. Pure apart from lazily importing torch.

    Raises `ValueError` for a spec the config layer would have rejected (these are caller bugs,
    not load failures), so a hand-built `ModelSpec` cannot slip a guess past the rules.
    """
    if spec.provider == "ollama":
        if spec.quantization not in OLLAMA_QUANTIZATIONS:
            raise ValueError(f"ollama quantization must be one of {OLLAMA_QUANTIZATIONS} (it is fixed by the "
                             f"model tag), got {spec.quantization!r}")
        if not spec.base_url or spec.context_length is None:
            raise ValueError("an ollama spec needs base_url and context_length (effGen cannot ask the server "
                             "and would silently assume 32768)")
        if spec.dtype is not None:
            raise ValueError("dtype does not apply to ollama models")
        prefix = spec.model.split(":", 1)[0].strip().lower() if ":" in spec.model else ""
        if prefix in OLLAMA_ID_ALIAS_PREFIXES:
            raise ValueError(f"ollama model id {spec.model!r} starts with {prefix + ':'!r}, which effGen would "
                             "strip as an adapter prefix and request a different model. Use a different tag")
        # No `quantization`, `engine` or torch arguments: the server owns the weights and their format.
        return spec.model, {
            "provider": "openai_compatible",
            "base_url": spec.base_url,
            "context_length": spec.context_length,
        }

    if spec.provider == "hf":
        if spec.quantization not in HF_QUANTIZATION_BITS:
            raise ValueError(f"hf quantization must be one of {HF_QUANTIZATIONS}, got {spec.quantization!r} "
                             "(bnb-4bit-fp4 is not supported: effGen hard-codes NF4)")
        if spec.base_url is not None or spec.context_length is not None:
            raise ValueError("base_url / context_length do not apply to hf models")
        problem = hf_placement_problem(spec.device, spec.quantization, spec.dtype)
        if problem:
            raise ValueError(f"unsupported hf placement: {problem}")
        kwargs: dict[str, Any] = {
            "engine": "transformers",  # a bare "hf" provider would mean the REMOTE Inference API
            "quantization_bits": HF_QUANTIZATION_BITS[spec.quantization],  # always present, even None
        }
        if spec.device == "cuda":
            # Partial GPU placement / CPU offload makes effGen raise. Absent CUDA is checked in build_model.
            kwargs["require_gpu"] = True
            if spec.dtype is not None:
                kwargs["torch_dtype"] = _torch_dtype(spec.dtype, spec)
        else:
            # CPU-only placement. effGen has no device argument and picks CUDA whenever it is available,
            # so the placement is pinned with device_map="cpu" (honoured over its own `.to(cuda)` step),
            # attention is not allowed to try flash-attn, and the dtype is explicit (placement rules
            # guarantee it is set). require_gpu=False because effGen would otherwise reject the CPU
            # placement it was explicitly asked for.
            kwargs.update(
                require_gpu=False, device_map="cpu", use_flash_attention=False,
                torch_dtype=_torch_dtype(spec.dtype, spec),
            )
        return spec.model, kwargs

    raise ValueError(f"unknown provider {spec.provider!r}; supported: ollama, hf")


def _check_result(spec: ModelSpec, model: Any) -> None:
    """Refuse a model that is not what the spec asked for (e.g. effGen routed the id elsewhere)."""
    if spec.provider == "ollama":
        if not isinstance(model, OpenAICompatibleAdapter):
            raise ModelLoadError(f"effGen returned {type(model).__name__} for {_describe(spec)}; expected an "
                                 "OpenAICompatibleAdapter. Refusing to continue with a different provider")
        return
    if not isinstance(model, BaseModel) or getattr(model, "model_type", None) != ModelType.TRANSFORMERS:
        raise ModelLoadError(f"effGen returned {type(model).__name__} for {_describe(spec)}; expected the local "
                             "Transformers engine. The id may have been routed to another provider or engine")
    placed = getattr(model, "device", None)
    if placed != spec.device:
        raise ModelLoadError(f"effGen placed {_describe(spec)} on {placed!r}, expected {spec.device!r}. "
                             "Refusing to run on a different device than the config asked for")
    expected = HF_QUANTIZATION_BITS[spec.quantization]
    actual = getattr(model, "quantization_bits", "<missing>")
    if actual != expected:
        raise ModelLoadError(f"effGen loaded {_describe(spec)} with quantization_bits={actual!r}, expected "
                             f"{expected!r}. Refusing a quantization the config did not ask for")


def build_model(spec: ModelSpec) -> BaseModel:
    """Load the model `spec` describes, once, with exactly the requested settings.

    hf: this really loads weights (on the GPU, or on CPU when `device: cpu`; bitsandbytes for the bnb options). ollama: builds the
    OpenAI-compatible client; the server is first contacted when the model is used (T3.3).

    Raises:
        ValueError: the spec breaks the rules above (nothing was loaded).
        ModelLoadError: effGen failed or returned the wrong kind of model; the effGen error is
            `__cause__` and its text is in the message. Never retried, never downgraded.
    """
    model_id, kwargs = loader_args(spec)
    if spec.provider == "hf" and spec.device == "cuda" and not _cuda_available():
        # effGen would load on CPU here without complaint even with require_gpu=True.
        raise ModelLoadError(
            f"cannot load {_describe(spec)}: device is cuda but torch reports CUDA as unavailable. Not "
            "falling back to CPU. Fix the CUDA install, or set `device: cpu` (quantization: none, explicit dtype)")
    try:
        model = load_model(model_id, **kwargs)
    except Exception as exc:  # noqa: BLE001 - re-raised with the spec attached, cause preserved
        raise ModelLoadError(f"failed to load {_describe(spec)}: {type(exc).__name__}: {exc}") from exc
    _check_result(spec, model)
    return model


class ModelCache:
    """Builds each distinct `ModelSpec` once and hands out the same instance afterwards.

    The key is the whole frozen spec, so specs that differ in provider, model, quantization,
    dtype, endpoint or context length never share an instance. Failed loads are not cached.
    Scope is explicit: create one per harness and `close()` it (or use it as a context manager).
    Not thread-safe; MVP workers run sequentially.
    """

    def __init__(self) -> None:
        self._models: dict[ModelSpec, BaseModel] = {}
        self._closed = False

    def get(self, spec: ModelSpec) -> BaseModel:
        if self._closed:
            raise RuntimeError("ModelCache is closed")
        model = self._models.get(spec)
        if model is None:
            model = self._models[spec] = build_model(spec)  # an exception leaves no entry behind
        return model

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, spec: object) -> bool:
        return spec in self._models

    def close(self) -> None:
        """Unload every model this cache built. Idempotent. Attempts all, then reports failures together."""
        models, self._models, self._closed = list(self._models.values()), {}, True
        errors: list[Exception] = []
        for model in models:
            try:
                model.unload()
            except Exception as exc:  # noqa: BLE001 - keep unloading the rest
                errors.append(exc)
        if errors:
            raise ExceptionGroup("failed to unload model(s)", errors)

    def __enter__(self) -> ModelCache:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
