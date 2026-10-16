"""T3.1: YAML -> validated dataclasses. Pure validation: no model, network or tool is touched."""

from __future__ import annotations

import copy
import dataclasses
import math
from pathlib import Path

import pytest

from smol_harness.config import (
    DEFAULT_MAX_DELEGATIONS,
    DEFAULT_MAX_REPORT_CHARS,
    DEFAULT_ORCHESTRATOR_ITERATIONS,
    DEFAULT_OLLAMA_URL,
    DEFAULT_WORKER_ITERATIONS,
    DEFAULT_WORKER_TIMEOUT_S,
    KNOWN_TOOLS,
    ConfigError,
    HarnessConfig,
    Limits,
    ModelSpec,
    load_config,
    loads_config,
    parse_config,
)
from smol_harness.delegation import MIN_REPORT_CHARS

EXAMPLE = Path(__file__).parent.parent / "configs" / "example.yaml"

OLLAMA = {"provider": "ollama", "model": "qwen2.5-coder:7b-instruct-q4_K_M", "context_length": 8192}
HF_NF4 = {"provider": "hf", "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "quantization": "bnb-4bit-nf4"}


def worker(name="coder", model=None, **over):
    w = {"name": name, "description": f"{name} worker", "model": copy.deepcopy(model or HF_NF4), "tools": ["file_operations"]}
    w.update(over)
    return w


def cfg(**over):
    """A minimal valid config as a dict; keyword args replace top-level keys."""
    data = {
        "workspace": "ws",
        "orchestrator": {"model": copy.deepcopy(OLLAMA), "tools": []},
        "workers": [worker()],
    }
    data.update(over)
    return data


def bad(data, fragment, base=None):
    with pytest.raises(ConfigError) as exc:
        parse_config(data, base)
    assert fragment in str(exc.value), str(exc.value)
    return str(exc.value)


# 1. example ----------------------------------------------------------------------------------------
def test_example_config_loads():
    c = load_config(EXAMPLE)
    assert isinstance(c, HarnessConfig)
    assert c.workspace == (EXAMPLE.parent / "../workspace").resolve()  # relative to the file, absolute
    assert c.orchestrator.name == "orchestrator" and c.orchestrator.model.provider == "ollama"
    assert c.orchestrator.model.quantization == "server" and c.orchestrator.model.context_length == 8192
    assert [w.name for w in c.workers] == ["coder"]
    w = c.workers[0]
    assert (w.model.provider, w.model.quantization, w.model.dtype) == ("hf", "bnb-4bit-nf4", "bf16")
    assert w.tools == ("file_operations", "shell") and w.max_iterations == 6 and c.orchestrator.max_iterations == 8
    assert c.limits == Limits(max_delegations=4, max_report_chars=4000, worker_timeout_s=300.0, max_depth=1)


def test_example_limits_are_all_finite_and_conservative():
    lim = load_config(EXAMPLE).limits
    assert lim.max_delegations <= 10 and lim.max_report_chars <= 20_000 and 0 < lim.worker_timeout_s <= 3600


# 2-5. provider combinations ------------------------------------------------------------------------
def test_ollama_only():
    c = parse_config(cfg(workers=[worker(model=OLLAMA)]))
    assert c.orchestrator.model == ModelSpec("ollama", OLLAMA["model"], "server", DEFAULT_OLLAMA_URL, 8192, None)
    assert c.workers[0].model.provider == "ollama"


@pytest.mark.parametrize("quant, dtype", [("none", "bf16"), ("none", "fp16"), ("none", None), ("bnb-8bit", None), ("bnb-4bit-nf4", "bf16")])
def test_hf_only(quant, dtype):
    hf = {"provider": "hf", "model": "org/model", "quantization": quant, **({"dtype": dtype} if dtype else {})}
    c = parse_config(cfg(orchestrator={"model": hf, "tools": []}, workers=[worker(model=hf)]))
    assert c.orchestrator.model == ModelSpec("hf", "org/model", quant, None, None, dtype)
    assert c.workers[0].model.quantization == quant  # exactly as written, never substituted


def test_mixed_providers_both_directions():
    a = parse_config(cfg())  # ollama orchestrator, hf worker
    b = parse_config(cfg(orchestrator={"model": HF_NF4, "tools": []}, workers=[worker(model=OLLAMA)]))
    assert (a.orchestrator.model.provider, a.workers[0].model.provider) == ("ollama", "hf")
    assert (b.orchestrator.model.provider, b.workers[0].model.provider) == ("hf", "ollama")


def test_multiple_workers_keep_order_and_independent_settings():
    c = parse_config(cfg(workers=[
        worker("coder", tools=["file_operations", "shell"], max_iterations=5),
        worker("reviewer", model=OLLAMA, tools=["file_operations"], system_prompt="Review only."),
        worker("planner", model={**HF_NF4, "quantization": "bnb-8bit"}, tools=[]),
    ]))
    assert [w.name for w in c.workers] == ["coder", "reviewer", "planner"]
    assert [w.model.provider for w in c.workers] == ["hf", "ollama", "hf"]
    assert [w.tools for w in c.workers] == [("file_operations", "shell"), ("file_operations",), ()]
    assert [w.max_iterations for w in c.workers] == [5, DEFAULT_WORKER_ITERATIONS, DEFAULT_WORKER_ITERATIONS]
    assert c.workers[1].system_prompt == "Review only." and c.workers[0].system_prompt is None


# 6. providers -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("provider", ["openai", "Ollama", "HF", "", "vllm", None, 3, ["hf"]])
def test_unknown_provider_rejected(provider):
    msg = bad(cfg(workers=[worker(model={"provider": provider, "model": "m"})]), "workers[0].model.provider")
    assert "unknown provider" in msg and "ollama, hf" in msg


def test_provider_required():
    bad(cfg(workers=[worker(model={"model": "m"})]), "missing required key 'provider'")


# 7. quantization ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("quant, fragment", [
    ("bnb-4bit-fp4", "hard-codes NF4"),  # effGen has no FP4 path (audit §1.7)
    ("server", "only applies to ollama"),
    ("awq", "not a supported hf quantization"),
    ("gptq", "not a supported hf quantization"),
    ("int8", "not a supported hf quantization"),
    ("4bit", "not a supported hf quantization"),
    ("q4_K_M", "not a supported hf quantization"),
    ("None", "not a supported hf quantization"),  # case matters: no normalisation
    ("BNB-8BIT", "not a supported hf quantization"),
    ("fp16", "not a supported hf quantization"),  # a dtype, not a quantization
    (4, "written as a string"), (True, "written as a string"), (False, "written as a string"), (None, "written as a string"),
])
def test_unsupported_hf_quantization_rejected(quant, fragment):
    bad(cfg(workers=[worker(model={**HF_NF4, "quantization": quant})]), fragment)


@pytest.mark.parametrize("quant", ["bnb-8bit", "bnb-4bit-nf4", "bnb-4bit-fp4", "none"])
def test_bitsandbytes_options_rejected_for_ollama(quant):
    msg = bad(cfg(workers=[worker(model={**OLLAMA, "quantization": quant})]), "workers[0].model.quantization")
    assert "Transformers/bitsandbytes option and cannot be used with ollama" in msg and "write `server`" in msg


@pytest.mark.parametrize("quant", ["q4_K_M", "Server", "bf16", 8])
def test_other_ollama_quantization_values_rejected(quant):
    bad(cfg(workers=[worker(model={**OLLAMA, "quantization": quant})]), "model.quantization")


def test_hf_quantization_must_be_explicit():
    # Omitted, effGen would auto-pick 4/8-bit from free VRAM: a silent substitution.
    bad(cfg(workers=[worker(model={"provider": "hf", "model": "m"})]), "missing required key 'quantization'")


def test_ollama_quantization_defaults_to_server():
    assert parse_config(cfg()).orchestrator.model.quantization == "server"


# provider-specific fields -------------------------------------------------------------------------------
@pytest.mark.parametrize("model, key", [
    ({**OLLAMA, "dtype": "bf16"}, "dtype"),
    ({**HF_NF4, "context_length": 4096}, "context_length"),
    ({**HF_NF4, "base_url": "http://x/v1"}, "base_url"),
    ({**OLLAMA, "request_timeout": 30}, "request_timeout"),  # no extra controls are invented
    ({**HF_NF4, "device_map": "auto"}, "device_map"),
])
def test_fields_from_the_wrong_provider_or_unknown_are_rejected(model, key):
    bad(cfg(workers=[worker(model=model)]), f"unknown key {key!r}")


def test_ollama_requires_context_length():
    bad(cfg(workers=[worker(model={"provider": "ollama", "model": "m:q4"})]), "missing required key 'context_length'")


@pytest.mark.parametrize("value", [0, 255, -5, True, 4096.0, "4096", None])
def test_context_length_validated(value):
    bad(cfg(workers=[worker(model={**OLLAMA, "context_length": value})]), "context_length")


@pytest.mark.parametrize("url", ["127.0.0.1:11434", "ftp://x/v1", "http://a b/v1", "", 5])
def test_base_url_validated(url):
    bad(cfg(workers=[worker(model={**OLLAMA, "base_url": url})]), "base_url")


@pytest.mark.parametrize("dtype", ["float16", "BF16", "fp32", 16, None])
def test_hf_dtype_validated(dtype):
    bad(cfg(workers=[worker(model={**HF_NF4, "dtype": dtype})]), "dtype")


# 8. duplicate names --------------------------------------------------------------------------------------
def test_duplicate_worker_names_rejected():
    msg = bad(cfg(workers=[worker("a"), worker("b"), worker("a")]), "duplicate worker name 'a'")
    assert msg.startswith("workers[2].name")


def test_duplicate_yaml_keys_rejected_instead_of_silently_overridden():
    text = "workspace: a\nworkspace: b\n"
    with pytest.raises(ConfigError, match="duplicate key 'workspace'"):
        loads_config(text)
    nested = EXAMPLE.read_text().replace("  max_depth: 1 ", "  max_delegations: 9\n  max_depth: 1 ", 1)
    assert nested != EXAMPLE.read_text()
    with pytest.raises(ConfigError, match="duplicate key 'max_delegations'"):
        loads_config(nested, EXAMPLE.parent)


# 9-10. tools -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tools, fragment", [
    (["git"], "unknown tool 'git'"),
    (["File_Operations"], "unknown tool"),
    (["file_operations", "nope"], "workers[0].tools[1]"),
    ("file_operations", "must be a list"),
    (None, "must be a list"),
    ({"file_operations": True}, "must be a list"),
    ([5], "tool name must be a string"),
    ([""], "must not be empty"),
    (["file_operations", "file_operations"], "duplicate tool"),
    (["shell ", ], "whitespace"),
])
def test_invalid_tool_allowlists_rejected(tools, fragment):
    bad(cfg(workers=[worker(tools=tools)]), fragment)


@pytest.mark.parametrize("name", ["web_search", "browser", "url_fetch", "http_get", "duckduckgo", "wikipedia", "agentic_search", "web_fetch", "news"])
def test_web_and_browser_tools_can_never_be_enabled(name):
    for data in (cfg(workers=[worker(tools=[name])]), cfg(orchestrator={"model": OLLAMA, "tools": [name]})):
        msg = bad(data, "web/search/browser tool")
        assert "not available in the MVP" in msg
    assert not any(k in KNOWN_TOOLS for k in ("web_search", "browser"))


def test_missing_tools_key_rejected_allowlists_are_explicit():
    w = worker()
    del w["tools"]
    bad(cfg(workers=[w]), "missing required key 'tools'")
    bad(cfg(orchestrator={"model": OLLAMA}), "orchestrator: missing required key 'tools'")


def test_worker_with_delegate_rejected():
    msg = bad(cfg(workers=[worker(tools=["file_operations", "delegate"])]), "workers cannot hold the 'delegate' tool")
    assert msg.startswith("workers[0].tools[1]") and "depth" in msg


def test_orchestrator_delegate_is_implicit_so_listing_it_is_accepted_and_not_stored():
    c = parse_config(cfg(orchestrator={"model": OLLAMA, "tools": ["delegate", "file_operations"]}))
    assert c.orchestrator.tools == ("file_operations",)


def test_empty_allowlists_are_allowed():
    c = parse_config(cfg(workers=[worker(tools=[])]))
    assert c.workers[0].tools == () and c.orchestrator.tools == ()


# 11. limits ---------------------------------------------------------------------------------------------------
def with_limits(**limits):
    return cfg(limits=limits)


@pytest.mark.parametrize("key, value", [
    ("max_delegations", 0), ("max_delegations", -1), ("max_delegations", True), ("max_delegations", 1.5),
    ("max_delegations", "3"), ("max_delegations", None), ("max_delegations", 101),
    ("max_report_chars", MIN_REPORT_CHARS - 1), ("max_report_chars", 0), ("max_report_chars", True),
    ("max_report_chars", 4000.0), ("max_report_chars", "4000"), ("max_report_chars", None), ("max_report_chars", 10**7),
    ("worker_timeout_s", 0), ("worker_timeout_s", -1), ("worker_timeout_s", True), ("worker_timeout_s", "5"),
    ("worker_timeout_s", None), ("worker_timeout_s", math.inf), ("worker_timeout_s", math.nan), ("worker_timeout_s", 86_401),
])
def test_invalid_limits_rejected(key, value):
    bad(with_limits(**{key: value}), f"limits.{key}")


@pytest.mark.parametrize("depth", [0, 2, 3, True, "1", None, 1.5])
def test_spawn_depth_is_fixed_at_one(depth):
    bad(with_limits(max_depth=depth), "fixed at 1")


def test_boundary_limits_accepted():
    c = parse_config(with_limits(max_delegations=1, max_report_chars=MIN_REPORT_CHARS, worker_timeout_s=0.5, max_depth=1))
    assert c.limits == Limits(1, MIN_REPORT_CHARS, 0.5, 1)
    assert isinstance(parse_config(with_limits(worker_timeout_s=30)).limits.worker_timeout_s, float)


def test_unknown_limit_keys_rejected_no_invented_controls():
    for key in ("max_tokens", "max_total_seconds", "parallel_workers", "max_retries"):
        bad(with_limits(**{key: 1}), f"unknown key {key!r}")


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8", None, 201])
@pytest.mark.parametrize("where", ["orchestrator", "worker"])
def test_invalid_max_iterations_rejected(where, value):
    if where == "orchestrator":
        data = cfg(orchestrator={"model": OLLAMA, "tools": [], "max_iterations": value})
        bad(data, "orchestrator.max_iterations")
    else:
        bad(cfg(workers=[worker(max_iterations=value)]), "workers[0].max_iterations")


def test_limits_section_must_be_a_mapping():
    bad(cfg(limits=[1, 2]), "limits: must be a mapping")


# 12. defaults ----------------------------------------------------------------------------------------------------
def test_defaults_when_optional_keys_are_omitted():
    c = parse_config(cfg())
    assert c.limits == Limits(DEFAULT_MAX_DELEGATIONS, DEFAULT_MAX_REPORT_CHARS, DEFAULT_WORKER_TIMEOUT_S, 1)
    assert c.orchestrator.max_iterations == DEFAULT_ORCHESTRATOR_ITERATIONS
    assert c.workers[0].max_iterations == DEFAULT_WORKER_ITERATIONS
    assert c.orchestrator.system_prompt is None and c.orchestrator.description == "" and c.orchestrator.name == "orchestrator"
    assert c.orchestrator.model.base_url == DEFAULT_OLLAMA_URL
    assert c.workers[0].model.dtype is None


def test_default_limits_are_finite_and_valid():
    lim = Limits()
    assert lim.max_delegations >= 1 and lim.max_report_chars >= MIN_REPORT_CHARS
    assert 0 < lim.worker_timeout_s < math.inf and lim.max_depth == 1
    assert parse_config(cfg(limits={})).limits == lim


def test_workspace_resolution(tmp_path):
    assert parse_config(cfg(workspace="repo"), tmp_path).workspace == (tmp_path / "repo").resolve()
    assert parse_config(cfg(workspace=str(tmp_path / "abs")), Path("/elsewhere")).workspace == tmp_path / "abs"
    assert not (tmp_path / "repo").exists()  # validation never touches the filesystem


def test_results_are_frozen_and_hashable():
    c = parse_config(cfg())
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.orchestrator.model.quantization = "none"  # type: ignore[misc]
    assert hash(c.workers[0].model) == hash(parse_config(cfg()).workers[0].model)  # usable as a cache key (T3.2)


# 13. malformed YAML -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["workspace: [unclosed", "a: b: c", "\tworkspace: x", "key: 'unterminated", "- - ]"])
def test_malformed_yaml_rejected(text):
    with pytest.raises(ConfigError, match="invalid YAML"):
        loads_config(text)


@pytest.mark.parametrize("text, fragment", [("", "config: must be a mapping"), ("- a\n- b\n", "must be a mapping"), ("just text", "must be a mapping"), ("null", "must be a mapping")])
def test_non_mapping_top_level_rejected(text, fragment):
    with pytest.raises(ConfigError, match=fragment):
        loads_config(text)


def test_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="cannot read config file"):
        load_config(tmp_path / "nope.yaml")


def test_unsafe_yaml_tags_are_not_executed():
    with pytest.raises(ConfigError):
        loads_config("workspace: !!python/object/apply:os.getcwd []")


def test_yaml_scalars_that_look_like_other_types_are_caught():
    text = EXAMPLE.read_text().replace("quantization: bnb-4bit-nf4", "quantization: off")  # YAML 1.1: off == False
    with pytest.raises(ConfigError, match="written as a string"):
        loads_config(text, EXAMPLE.parent)


# 14. missing required fields -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["workspace", "orchestrator", "workers"])
def test_missing_top_level_keys(key):
    data = cfg()
    del data[key]
    bad(data, f"missing required key {key!r}")


@pytest.mark.parametrize("key", ["name", "description", "model", "tools"])
def test_missing_worker_keys(key):
    w = worker()
    del w[key]
    bad(cfg(workers=[w]), f"workers[0]: missing required key {key!r}")


def test_missing_model_keys():
    bad(cfg(workers=[worker(model={"provider": "hf", "quantization": "none"})]), "missing required key 'model'")
    bad(cfg(orchestrator={"tools": []}), "orchestrator: missing required key 'model'")


def test_workers_must_be_a_non_empty_list():
    for value in ([], None, {"coder": worker()}, "coder"):
        bad(cfg(workers=value), "workers: must be a non-empty list")


def test_error_paths_point_at_the_offending_entry():
    data = cfg(workers=[worker("ok"), worker("bad", model={**HF_NF4, "quantization": "bnb-4bit-fp4"})])
    assert bad(data, "workers[1].model.quantization").startswith("workers[1].model.quantization")


@pytest.mark.parametrize("name", ["", "  ", " pad", "has space", 5, None])
def test_invalid_worker_names_rejected(name):
    bad(cfg(workers=[worker(name)]), "workers[0].name")


@pytest.mark.parametrize("model_id", ["", "   ", "a b", " m", 7, None])
def test_invalid_model_ids_rejected(model_id):
    bad(cfg(workers=[worker(model={**HF_NF4, "model": model_id})]), "workers[0].model.model")
    bad(cfg(orchestrator={"model": {**OLLAMA, "model": model_id}, "tools": []}), "orchestrator.model.model")


@pytest.mark.parametrize("field", ["description", "system_prompt"])
def test_empty_text_fields_rejected(field):
    bad(cfg(workers=[worker(**{field: "  "})]), field)


def test_unknown_keys_rejected_at_every_level():
    bad(cfg(extra=1), "unknown key 'extra'")
    bad(cfg(workers=[worker(temperature=0.2)]), "unknown key 'temperature'")
    bad(cfg(orchestrator={"model": OLLAMA, "tools": [], "name": "boss"}), "unknown key 'name'")  # fixed name
    bad(cfg(orchestrator={"model": OLLAMA, "tools": [], "description": "x"}), "unknown key 'description'")


def test_config_error_is_a_value_error():
    assert issubclass(ConfigError, ValueError)


# ---- T3.2a: hf device selection -----------------------------------------------------------------------------------------
def hf_model(**over):
    return {**HF_NF4, **over}


def test_hf_device_defaults_to_cuda_and_explicit_cuda_is_the_same_spec():
    omitted = parse_config(cfg()).workers[0].model
    explicit = parse_config(cfg(workers=[worker(model=hf_model(device="cuda"))])).workers[0].model
    assert omitted.device == "cuda" and omitted == explicit and hash(omitted) == hash(explicit)
    assert parse_config(cfg()).orchestrator.model.device is None  # ollama never has a device


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_hf_cpu_accepts_nonquantized_with_explicit_dtype(dtype):
    m = parse_config(cfg(workers=[worker(model=hf_model(quantization="none", device="cpu", dtype=dtype))])).workers[0].model
    assert (m.device, m.quantization, m.dtype) == ("cpu", "none", dtype)


def test_device_is_part_of_the_spec_identity():
    cpu = ModelSpec("hf", "m", "none", dtype="bf16", device="cpu")
    cuda = ModelSpec("hf", "m", "none", dtype="bf16", device="cuda")
    assert cpu != cuda and len({cpu, cuda}) == 2


@pytest.mark.parametrize("override, fragment, where", [
    ({"quantization": "bnb-8bit", "device": "cpu", "dtype": "fp32"}, "needs a CUDA GPU", "device"),
    ({"quantization": "bnb-4bit-nf4", "device": "cpu", "dtype": "bf16"}, "needs a CUDA GPU", "device"),
    ({"quantization": "none", "device": "cpu"}, "requires an explicit dtype", "dtype"),  # host-dependent otherwise
    ({"quantization": "none", "device": "cpu", "dtype": "fp16"}, "not supported on cpu", "dtype"),
    ({"quantization": "none", "device": "cuda", "dtype": "fp32"}, "only supported with device: cpu", "dtype"),
    ({"quantization": "none", "dtype": "fp32"}, "only supported with device: cpu", "dtype"),  # device omitted = cuda
])
def test_unsupported_device_combinations_are_rejected(override, fragment, where):
    msg = bad(cfg(workers=[worker(model=hf_model(**override))]), fragment)
    assert msg.startswith(f"workers[0].model.{where}")


@pytest.mark.parametrize("device", ["gpu", "CPU", "CUDA", "cuda:0", "mps", "", True, 0, None, ["cpu"]])
def test_invalid_device_values_are_rejected(device):
    bad(cfg(workers=[worker(model=hf_model(device=device))]), "workers[0].model.device")


def test_device_is_rejected_for_ollama_in_config_and_in_the_dataclass():
    bad(cfg(workers=[worker(model={**OLLAMA, "device": "cpu"})]), "unknown key 'device'")
    with pytest.raises(ValueError, match="does not apply to provider 'ollama'"):
        ModelSpec("ollama", "m:1", "server", base_url="http://x/v1", context_length=4096, device="cpu")


def test_fp32_is_only_a_valid_dtype_in_cpu_mode_and_other_dtype_values_stay_invalid():
    parse_config(cfg(workers=[worker(model=hf_model(quantization="none", device="cpu", dtype="fp32"))]))
    for dtype in ("float32", "FP32", "int8"):
        bad(cfg(workers=[worker(model=hf_model(quantization="none", device="cpu", dtype=dtype))]), "dtype")
