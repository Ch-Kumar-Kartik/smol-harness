"""YAML -> validated, frozen dataclasses (docs/architecture.md §4, §8, §9).

Validation only: nothing here loads a model, touches the network or builds a tool.
Rules are the ones verified against effGen 1.3.0 (docs/effgen-api-audit.md §1.7):
quantization is never guessed or substituted, and every value must be spelled
exactly as documented. The first problem found is reported with its dotted path
(e.g. `workers[1].model.quantization`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from smol_harness.delegation import DELEGATE_TOOL_NAME, MIN_REPORT_CHARS

MAX_SPAWN_DEPTH = 1  # fixed for the MVP: only the orchestrator delegates

PROVIDERS = ("ollama", "hf")
HF_QUANTIZATIONS = ("none", "bnb-8bit", "bnb-4bit-nf4")
OLLAMA_QUANTIZATIONS = ("server",)
HF_DTYPES = ("bf16", "fp16")
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/v1"  # Ollama's OpenAI-compatible endpoint (audit §1.5)

# Tools the harness can build (T4.x). Anything else is rejected, whatever the YAML says.
KNOWN_TOOLS = ("file_operations", "shell")
_WEB_TOOL_HINT = re.compile(r"web|search|browser|http|fetch|url|crawl|scrape|duckduckgo|ddgs|wiki|news", re.I)

DEFAULT_ORCHESTRATOR_ITERATIONS = 10
DEFAULT_WORKER_ITERATIONS = 8
MAX_ITERATIONS_CEILING = 200
DEFAULT_MAX_DELEGATIONS = 4
MAX_DELEGATIONS_CEILING = 100
DEFAULT_MAX_REPORT_CHARS = 4000
MAX_REPORT_CHARS_CEILING = 1_000_000
DEFAULT_WORKER_TIMEOUT_S = 300.0
MAX_WORKER_TIMEOUT_S = 86_400.0


class ConfigError(ValueError):
    """The configuration is malformed. The message starts with the offending path."""


@dataclass(frozen=True)
class ModelSpec:
    provider: str  # "ollama" | "hf"
    model: str  # Ollama tag (e.g. "qwen2.5-coder:7b-instruct-q4_K_M") or HF repo id / path
    quantization: str  # ollama: "server" (chosen by the tag) | hf: "none" | "bnb-8bit" | "bnb-4bit-nf4"
    base_url: str | None = None  # ollama only
    context_length: int | None = None  # ollama only (effGen cannot ask the server)
    dtype: str | None = None  # hf only: "bf16" | "fp16"; None = effGen's automatic choice


@dataclass(frozen=True)
class AgentSpec:
    name: str
    model: ModelSpec
    tools: tuple[str, ...]  # explicit allowlist; for the orchestrator, `delegate` is implicit and not listed
    max_iterations: int
    description: str = ""
    system_prompt: str | None = None


@dataclass(frozen=True)
class Limits:
    max_delegations: int = DEFAULT_MAX_DELEGATIONS
    max_report_chars: int = DEFAULT_MAX_REPORT_CHARS
    worker_timeout_s: float = DEFAULT_WORKER_TIMEOUT_S
    max_depth: int = MAX_SPAWN_DEPTH


@dataclass(frozen=True)
class HarnessConfig:
    workspace: Path
    orchestrator: AgentSpec
    workers: tuple[AgentSpec, ...]
    limits: Limits


# ---- small validators ------------------------------------------------------------------------------
def _fail(path: str, msg: str) -> Any:
    raise ConfigError(f"{path}: {msg}" if path else msg)


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, f"must be a mapping, got {type(value).__name__}")
    for key in value:
        if not isinstance(key, str):
            _fail(path, f"keys must be strings, got {key!r}")
    return value


def _keys(m: dict[str, Any], path: str, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> None:
    for key in required:
        if key not in m:
            _fail(path, f"missing required key {key!r}")
    for key in m:
        if key not in required and key not in optional:
            allowed = ", ".join(sorted((*required, *optional)))
            _fail(path, f"unknown key {key!r}. Allowed keys: {allowed}")


def _text(value: Any, path: str, what: str, *, no_space: bool = False) -> str:
    if not isinstance(value, str):
        _fail(path, f"{what} must be a string, got {type(value).__name__} ({value!r}). Quote it in YAML if needed")
    if not value.strip():
        _fail(path, f"{what} must not be empty")
    if value != value.strip():
        _fail(path, f"{what} {value!r} has leading or trailing whitespace")
    if no_space and re.search(r"\s", value):
        _fail(path, f"{what} {value!r} must not contain whitespace")
    return value


def _int(value: Any, path: str, lo: int, hi: int) -> int:
    # bool is an int subclass: `max_iterations: true` must not become 1.
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, f"must be an integer, got {type(value).__name__} ({value!r})")
    if not lo <= value <= hi:
        _fail(path, f"must be between {lo} and {hi}, got {value}")
    return value


def _seconds(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, f"must be a number of seconds, got {type(value).__name__} ({value!r})")
    if not math.isfinite(value) or not 0 < value <= MAX_WORKER_TIMEOUT_S:
        _fail(path, f"must be a finite number > 0 and <= {MAX_WORKER_TIMEOUT_S:g}, got {value!r}")
    return float(value)


# ---- sections ------------------------------------------------------------------------------------------
def _quantization(value: Any, provider: str, path: str) -> str:
    if not isinstance(value, str):
        _fail(path, f"must be one of {_allowed(provider)} written as a string, got {type(value).__name__} "
                    f"({value!r}). Note YAML turns unquoted off/no/null into non-strings")
    if provider == "ollama":
        if value in OLLAMA_QUANTIZATIONS:
            return value
        if value in HF_QUANTIZATIONS or value.startswith("bnb"):
            _fail(path, f"{value!r} is a Transformers/bitsandbytes option and cannot be used with ollama. "
                        "Ollama's quantization is fixed by the model tag; write `server`")
        _fail(path, f"{value!r} is not valid for ollama. Allowed: server (the quantization is chosen by the "
                    "Ollama model tag, e.g. ...-q4_K_M; it cannot be set here)")
    if value in HF_QUANTIZATIONS:
        return value
    if value == "bnb-4bit-fp4":
        _fail(path, "'bnb-4bit-fp4' is not supported: effGen 1.3.0 hard-codes NF4 for 4-bit loading. "
                    "Use 'bnb-4bit-nf4'")
    if value == "server":
        _fail(path, "'server' only applies to ollama. For hf use none, bnb-8bit or bnb-4bit-nf4")
    _fail(path, f"{value!r} is not a supported hf quantization. Allowed: {', '.join(HF_QUANTIZATIONS)} "
                "(exact spelling; nothing is substituted)")
    raise AssertionError  # pragma: no cover


def _allowed(provider: str) -> str:
    return ", ".join(OLLAMA_QUANTIZATIONS if provider == "ollama" else HF_QUANTIZATIONS)


def _model(value: Any, path: str) -> ModelSpec:
    m = _mapping(value, path)
    if "provider" not in m:
        _fail(path, f"missing required key 'provider' (one of: {', '.join(PROVIDERS)})")
    provider = m["provider"]
    if provider not in PROVIDERS or not isinstance(provider, str):
        _fail(_join(path, "provider"), f"unknown provider {provider!r}. Supported: {', '.join(PROVIDERS)}")

    if provider == "ollama":
        # context_length is required: effGen cannot ask Ollama for its window and would silently
        # assume 32768 (audit §1.5), which can be larger than the server really serves.
        _keys(m, path, ("provider", "model", "context_length"), ("quantization", "base_url"))
        quant = _quantization(m.get("quantization", "server"), provider, _join(path, "quantization"))
        url = m.get("base_url", DEFAULT_OLLAMA_URL)
        url_path = _join(path, "base_url")
        _text(url, url_path, "base_url", no_space=True)
        if not url.startswith(("http://", "https://")):
            _fail(url_path, f"must start with http:// or https://, got {url!r}")
        return ModelSpec(
            provider, _text(m["model"], _join(path, "model"), "model id", no_space=True), quant, base_url=url,
            context_length=_int(m["context_length"], _join(path, "context_length"), 256, 10_000_000),
        )

    # hf: quantization is required. If it were omitted effGen would pick 4/8-bit from free VRAM
    # on its own (audit §1.7), i.e. a silent substitution.
    _keys(m, path, ("provider", "model", "quantization"), ("dtype",))
    quant = _quantization(m["quantization"], provider, _join(path, "quantization"))
    dtype = None
    if "dtype" in m:
        dtype = m["dtype"]
        if dtype not in HF_DTYPES or not isinstance(dtype, str):
            _fail(_join(path, "dtype"), f"must be one of {', '.join(HF_DTYPES)}, got {dtype!r}")
    return ModelSpec(provider, _text(m["model"], _join(path, "model"), "model id", no_space=True), quant, dtype=dtype)


def _tools(value: Any, path: str, *, orchestrator: bool) -> tuple[str, ...]:
    if not isinstance(value, list):
        _fail(path, f"must be a list of tool names (use [] for none), got {type(value).__name__}")
    seen: list[str] = []
    for i, name in enumerate(value):
        item = f"{path}[{i}]"
        _text(name, item, "tool name", no_space=True)
        if name == DELEGATE_TOOL_NAME:
            if not orchestrator:
                _fail(item, f"workers cannot hold the {DELEGATE_TOOL_NAME!r} tool: delegation depth is "
                            f"fixed at {MAX_SPAWN_DEPTH} (only the orchestrator delegates)")
            continue  # implicit for the orchestrator: accepted, not stored
        if name not in KNOWN_TOOLS:
            if _WEB_TOOL_HINT.search(name):
                _fail(item, f"{name!r} looks like a web/search/browser tool; those are not available in the MVP "
                            "and cannot be enabled from configuration")
            _fail(item, f"unknown tool {name!r}. Available: {', '.join(KNOWN_TOOLS)}")
        if name in seen:
            _fail(item, f"duplicate tool {name!r}")
        seen.append(name)
    return tuple(seen)


def _agent(value: Any, path: str, *, name: str | None, default_iterations: int) -> AgentSpec:
    m = _mapping(value, path)
    is_orch = name is not None
    required = ("model", "tools") if is_orch else ("name", "description", "model", "tools")
    _keys(m, path, required, ("system_prompt", "max_iterations"))
    agent_name = name if is_orch else _text(m["name"], _join(path, "name"), "name", no_space=True)
    prompt = m.get("system_prompt")
    if "system_prompt" in m:
        prompt = _text(prompt, _join(path, "system_prompt"), "system_prompt")
    return AgentSpec(
        name=agent_name,
        model=_model(m["model"], _join(path, "model")),
        tools=_tools(m["tools"], _join(path, "tools"), orchestrator=is_orch),
        max_iterations=_int(m.get("max_iterations", default_iterations), _join(path, "max_iterations"),
                            1, MAX_ITERATIONS_CEILING),
        description="" if is_orch else _text(m["description"], _join(path, "description"), "description"),
        system_prompt=prompt,
    )


def _limits(value: Any, path: str) -> Limits:
    m = _mapping(value, path)
    # Only the controls that exist. No unlimited (null) values in the MVP: every limit is finite.
    _keys(m, path, (), ("max_delegations", "max_report_chars", "worker_timeout_s", "max_depth"))
    depth = m.get("max_depth", MAX_SPAWN_DEPTH)
    if isinstance(depth, bool) or depth != MAX_SPAWN_DEPTH:
        _fail(_join(path, "max_depth"), f"is fixed at {MAX_SPAWN_DEPTH} in the MVP (only the orchestrator "
                                         f"delegates), got {depth!r}")
    return Limits(
        max_delegations=_int(m.get("max_delegations", DEFAULT_MAX_DELEGATIONS),
                             _join(path, "max_delegations"), 1, MAX_DELEGATIONS_CEILING),
        max_report_chars=_int(m.get("max_report_chars", DEFAULT_MAX_REPORT_CHARS),
                              _join(path, "max_report_chars"), MIN_REPORT_CHARS, MAX_REPORT_CHARS_CEILING),
        worker_timeout_s=_seconds(m.get("worker_timeout_s", DEFAULT_WORKER_TIMEOUT_S), _join(path, "worker_timeout_s")),
        max_depth=MAX_SPAWN_DEPTH,
    )


# ---- public API ------------------------------------------------------------------------------------------
def parse_config(data: Any, base_dir: Path | None = None) -> HarnessConfig:
    """Validate an already-parsed mapping. A relative `workspace` is resolved against `base_dir` (default: cwd).

    The workspace is not required to exist yet; the tool factory checks that (T4.1).
    """
    m = _mapping(data, "config")
    _keys(m, "", ("workspace", "orchestrator", "workers"), ("limits",))
    ws = Path(_text(m["workspace"], "workspace", "workspace"))
    workspace = (ws if ws.is_absolute() else (base_dir or Path.cwd()) / ws).resolve()

    raw_workers = m["workers"]
    if not isinstance(raw_workers, list) or not raw_workers:
        _fail("workers", "must be a non-empty list of worker definitions, each with a `name`")
    workers = tuple(_agent(w, f"workers[{i}]", name=None, default_iterations=DEFAULT_WORKER_ITERATIONS)
                    for i, w in enumerate(raw_workers))
    names = [w.name for w in workers]
    for i, n in enumerate(names):
        if n in names[:i]:
            _fail(f"workers[{i}].name", f"duplicate worker name {n!r}; worker names must be unique")
    return HarnessConfig(
        workspace=workspace,
        orchestrator=_agent(m["orchestrator"], "orchestrator", name="orchestrator",
                            default_iterations=DEFAULT_ORCHESTRATOR_ITERATIONS),
        workers=workers,
        limits=_limits(m["limits"], "limits") if "limits" in m else Limits(),
    )


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys (PyYAML silently keeps the last one)."""


def _no_duplicate_keys(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False) -> Any:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in seen
            seen.add(key)
        except TypeError:  # unhashable key: SafeLoader reports it itself
            continue
        if duplicate:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys)


def loads_config(text: str, base_dir: Path | None = None) -> HarnessConfig:
    """Parse and validate YAML text."""
    try:
        data = yaml.load(text, Loader=_StrictLoader)  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}") from exc
    return parse_config(data, base_dir)


def load_config(path: str | Path) -> HarnessConfig:
    """Read, parse and validate a YAML file; a relative `workspace` is relative to the file's directory."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {str(p)!r}: {exc.strerror or exc}") from exc
    return loads_config(text, p.resolve().parent)
