# effGen API Audit and Integration Status

This document has three parts:

1. Verified effGen capabilities, i.e. what the framework provides.
2. Harness integration status, i.e. what *we* have built on top of it.
3. Integration history.

Keep the parts separate. Never mark a harness capability "Tested" just because effGen supports it.

## Audit baseline

| Item | Value |
|---|---|
| effGen version inspected | `1.3.0` (`effgen/__init__.py: __version__`) |
| effGen commit inspected | `37de05a6b4ed01c90ee38b9c254a162927ca2ac2` (2026-10-01), `github.com/ctrl-gaurav/effGen` |
| smolagents commit inspected | `96f33faaf028479119ec8d34507b47694cf14e34` (2026-10-06) |
| pi-mono commit inspected | `4954d760504c11df9b2cbdc040346107b6a8153d` (2026-10-15) |
| Audit date | 2026-10-16 |
| Method | Static source reading of shallow clones. **No effGen code was executed during this audit.** All "Verification evidence" below is source-level unless stated otherwise. |

To re-inspect, clone into a scratch directory outside this repo (e.g. `/tmp/refsrc/effgen/repo`), then `git checkout 37de05a`. Read the code; do not run it from inside the clone.

---

## Part 1: Verified effGen capabilities

### 1.1 Agent initialization and execution

- **Files:** `effgen/core/agent.py` (`Agent`), `effgen/core/agent_config.py` (`AgentConfig`, `AgentMode`), `effgen/core/agent_orchestration.py` (`AgentOrchestrationMixin.run`), `effgen/core/agent_response.py` (`AgentResponse`).
- **API:** `Agent(AgentConfig(name=..., model=<str | BaseModel>, tools=[BaseTool...], system_prompt=..., max_iterations=10, ...))`, then `agent.run(task, mode=None, context=None, **run_kwargs) -> AgentResponse`.
- **Behavior:**
  - A string `model` is loaded at construction through `ModelLoader.load_model(...)`, with `config.provider`, `config.base_url` and `config.api_key` forwarded. A `BaseModel` instance is used as-is, which allows sharing.
  - `config.tools` must be `BaseTool` instances. Bare name strings raise `TypeError`.
  - `self.tools = {tool.name: tool}`. **This is the agent's whole tool allowlist.** There is no implicit global registry injection.
  - `raise_on_error=True` (the default) makes a stopped run raise `RunStoppedError` (`effgen/errors.py`). With `False`, you inspect `response.success`, `response.metadata["reason"]` and `response.termination`.
  - `run()` kwargs are limited to `_RUN_KWARGS` (`agent_config.py:454`), which includes `max_iterations`, `max_tokens`, `middleware` and `session`. There is **no** `timeout` run kwarg.
- **AgentResponse fields used by us:** `output`, `success`, `iterations`, `tool_calls`, `tokens_used`, `execution_time`, `stop_reason`, `termination` (`done | not_possible | stuck | tool_failed | error`, 1.3.0), `metadata["thread"]` (`AgentThread`).
- **Limitations:**
  - Default `AgentConfig.enable_sub_agents=True` builds a `SubAgentRouter` and a `SubAgentManager` at construction. They are inert under the default `mode=AgentMode.SINGLE`. We set it to `False` anyway.
  - `capability_probe=True` (the default) may measure a local or URL-served model's tool behaviour once and cache the result (`Agent._capability_probe_for`). This adds a model call at construction. **Unverified** whether that matters for Ollama latency.

### 1.2 Reasoning and tool-calling loop

- **Files:**
  - `effgen/core/agent_loop.py` (≈3.9k lines, the main loop)
  - `agent_tool_loop.py`, `agent_react.py`, `agent_native_tools.py`
  - `tool_calling.py` (`get_strategy`)
  - `agent_tool_execution.py` (`_execute_tool`, `_execute_tool_once`)
- **Behavior:**
  - ReAct-style loop of Thought, Action, Observation, then Final Answer.
  - `tool_calling_mode="auto"|"native"|"react"|"hybrid"`. Auto resolves from the model's declared or probed capability.
  - Every dispatch path (ReAct `agent_loop.py:3160`, native `agent_native_tools.py:145,368`) goes through `Agent._execute_tool`.
  - Tool failures are returned to the model as `"Error executing tool '<name>': ..."` observations, not raised.
- **1.3.0 loop changes we must account for:**
  - `max_turns_without_progress=2`: after 2 turns with no new tool result, the next turn offers no tools and asks for the answer.
  - A "closing request" asks the model for an answer before the run stops.
  - A tool that keeps failing on its own side ends the run as `termination="tool_failed"`.
- **Tool output stringification** (`agent_tool_execution.py:300-365`):
  - A `ToolResult` whose `output` is a dict is reduced to selected keys: `result`, `stdout`, `stderr`, `error`, `output`, `data`, `message`.
  - **Returning a dict from a tool can therefore hide fields from the model.** To show the model a structured result, return a JSON **string**.
  - `BaseTool.execute` treats a dict with `success: False` as a tool failure (`base_tool.py:_inner_status`).

### 1.3 Subagent spawning and orchestration

effGen has **three** relevant mechanisms. None of them gives model-initiated, mid-loop delegation to a worker on a different model.

**(a) `SubAgentManager`** (`effgen/core/sub_agent_manager.py`)

- **Trigger:** `Agent.run(mode=AgentMode.AUTO | SUB_AGENTS)` with `enable_sub_agents=True`. `SubAgentRouter` (`router.py`) and `DecompositionEngine` (`decomposition_engine.py`) split the task **up front**. Each subtask then runs via `execute_sequential`, `execute_parallel` or `execute_hybrid`.
- **Child construction:** `_run_real_sub_agent` (lines ~650-780) builds a fresh child `Agent` per subtask.
  - The child **reuses the parent's model instance** (`model=parent.model`) and **the parent's full tool set**.
  - The child gets `enable_sub_agents=False`, `enable_memory=False`, and inherits the context budget, answer style, etc.
  - The child is closed afterwards.
- **Results:** `SubAgentResult` (`success`, `result`, `error`, `execution_time`, `tokens_used`, `tool_calls`, `thread`). A `DelegationStep` is appended to the parent thread. Exceptions become failed results.
- **Limitations, verified from source:**
  - `SubAgentConfig.model` (default `"inherit"`) and `SubAgentConfig.tools` are **ignored**. The child always uses the parent's model and tools.
  - `SubAgentConfig.timeout` is declared but **never enforced** (no reference outside the dataclass).
  - Specializations hard-code tool names that include `web_search`.
  - Decomposition is router- and heuristic-driven, not a tool the model chooses mid-run.
  - Results are synthesized by `synthesize_results` / `Agent.synthesize`, not fed back into a continuing parent loop.
  - **Conclusion:** unsuitable for "different model per worker" and "parent continues after child". **Not used.**

**(b) `MultiAgentOrchestrator`** (`effgen/core/orchestrator.py`)

- **Patterns:** `TeamConfig(pattern=OrchestrationPattern.SEQUENTIAL | PARALLEL | HIERARCHICAL | COLLABORATIVE | COMPETITIVE | PIPELINE, agents=[Agent...], manager_agent=Agent)`.
- **HIERARCHICAL** (`_execute_hierarchical`, lines 852-997) runs in three fixed phases:
  1. The manager writes a numbered `"<worker>: <subtask>"` list, which is parsed from text.
  2. Each worker runs `agent.run(subtask, mode=AgentMode.AUTO)`.
  3. The manager synthesizes.
- **Strengths:** workers are independent `Agent`s, so **different models per agent are supported**. `TeamResponse.thread` keeps each child's `AgentThread`.
- **Limitations, verified:**
  - No mid-loop delegation, so the manager cannot observe a result and decide what to do next.
  - Subtask parsing is regex/heuristic.
  - Workers run in `AgentMode.AUTO`, which can trigger their own decomposition unless `enable_sub_agents=False`.
  - `TeamConfig.timeout` is declared but **never enforced** (no reference in `orchestrator.py`).
  - **Conclusion:** possible baseline for comparison later. **Not used for MVP.**

**(c) Nested `Agent.run` from inside a tool (agent-as-tool)**

- **Evidence that effGen designs for this:**
  - `agent.py:133-139`: `_run_scope_var` is keyed by `id(agent)` "so that a run on one agent that calls another agent's run (a tool that is itself an agent) leaves each reading its own conversation".
  - `agent.py:886-890`: a nested run gets its own `ExecutionTracker`.
  - `tools/function_tool.py:207-224`: a sync `FunctionTool` runs in a copied `contextvars` context so "an agent the function starts is then counted as a child of the run that called the tool", in the same ledger.
- **Gaps:**
  - There is no shipped `AgentTool` or `as_tool()` helper; we write the wrapper.
  - `Agent.short_term_memory` **persists across `run()` calls** (`agent_orchestration.py:574-577` stores each successful turn, and the loop includes it as history). A reused worker would therefore see earlier delegations, so isolation needs a fresh `Agent` per delegation or `reset_memory()`.
- **Conclusion: this is the mechanism the harness uses.**

### 1.4 Context isolation and session management

- **Context:** `AgentThread` / `Step` / `DelegationStep` (`core/thread.py`), `ThreadProjection` with `NoParentContext` (default), `ParentTask`, `ParentAnswers` and `LastCycles` (`core/thread_projection.py`), and the internal `_prior_steps` kwarg (`agent_loop.py:329`).
- **Sessions:** `Session` (`core/session.py`) is a persisted conversation under `~/.effgen/sessions/`. `run(session=...)` scopes one call to a session's history (`agent.py:_open_run_conversation`).
- **Isolation facts:**
  - Each `Agent` owns its own `short_term_memory`.
  - A nested run on a *different* agent reads its own scope (1.3c).
  - A parent never automatically receives a child's history; it sees only what the tool returns.
  - `close()` does **not** unload the model (`agent.py:1093-1116`), so building and closing per-delegation `Agent`s around a shared model is safe.
- **Limitation:** `_prior_steps` is private (leading underscore). The harness passes context as explicit task text instead.

### 1.5 Ollama model loading

- **Files:** `effgen/models/openai_compatible_adapter.py` (`OpenAICompatibleAdapter`), `models/model_loader.py` (`load_model`), `models/model_loader_cloud.py:_load_openai_compatible_model`, `docs/models/openai-compatible.md`.
- **API:** `load_model("<ollama tag>", provider="openai_compatible", base_url="http://127.0.0.1:11434/v1", context_length=<int>, timeout=<s>)`. Alternatively, `AgentConfig(model="<tag>", base_url=...)`.
- **There is no native Ollama adapter.** Ollama is reached via its OpenAI-compatible `/v1` endpoint. The API key defaults to a placeholder.
- **Defaults:**
  - `context_length` defaults to 32768 with a warning. It must match the server's real window.
  - Adapter `timeout=60` s and `max_retries=3`. Extra kwargs pass through `load_model` to the adapter (`model_loader_cloud.py:152-162`).
- **Errors:** an unreachable server raises `BackendUnreachableError` regardless of `raise_on_error`.
- **Unverified:** whether tool calling via Ollama's `/v1` works natively for the chosen tags; `auto` mode probes and falls back to ReAct text. Also, Ollama's own default `num_ctx` is smaller than 32768 unless configured (`OLLAMA_CONTEXT_LENGTH` or a Modelfile), which is a risk.

### 1.6 Hugging Face (local) model loading

- **Files:**
  - `models/model_loader_local.py` (`_load_huggingface_model`, `_load_with_transformers`)
  - `models/transformers_engine.py` (`TransformersEngine.__init__`)
  - `models/transformers_engine_loading.py`
  - `models/model_loader_capacity.py`
- **API:** `load_model("<hf repo id or path>", engine="transformers", quantization_bits=None|4|8, torch_dtype=torch.bfloat16|torch.float16, device_map="auto", require_gpu=False, trust_remote_code=False)`. The `quantization="4bit"|"8bit"` shorthand also works.
- **Prefix trap:** an `"hf:<repo>"` prefix means the *remote* HF Inference API. For local loading, use a bare id with `engine="transformers"`.
- **Engine selection:** the default engine is Transformers (vLLM, MLX and GGUF are opt-in).
- **Concurrency:** the Transformers engine serializes calls on one weights instance ("calls take turns", `docs/architecture/overview.md`).
- **Tool calling:** `supports_tool_calling()` is True only if the chat template renders tools. Otherwise the ReAct text protocol is used.

### 1.7 Quantization

| Format | Backend | effGen support | How | Notes |
|---|---|---|---|---|
| BF16 / FP16 (no quant) | Transformers | Yes | `quantization_bits=None` **passed explicitly**, plus `torch_dtype` | **Trap:** if `quantization_bits` is *absent*, `_auto_select_quantization_bits` picks 4-bit when free VRAM < 16 GB and 8-bit when < 32 GB (`model_loader_capacity.py:69-96`). On this machine's 8 GB GPU, auto means 4-bit. `torch_dtype=None` gives bf16 if supported, else fp16 (CUDA) or fp32 (CPU). |
| bnb 8-bit (LLM.int8) | Transformers | Yes, quantized while loading | `quantization_bits=8` → `BitsAndBytesConfig(load_in_8bit=True)` | Requires `bitsandbytes`, which is only in effGen's `dev`/`all` extras, not the base install. |
| bnb 4-bit NF4 | Transformers | Yes, quantized while loading | `quantization_bits=4` → NF4, double-quant, bf16 compute (`transformers_engine_loading.py:272-296`) | Settings are hard-coded. |
| bnb 4-bit FP4 | Transformers | **No** | — | `bnb_4bit_quant_type` is hard-coded to `"nf4"`. Not exposed by the effGen API. Out of MVP scope. |
| AWQ / GPTQ (pre-quantized) | Transformers | Partial | `quantization="awq"/"gptq"` only checks the import; the checkpoint carries the config | Already-quantized checkpoint. Unverified. |
| GGUF via Ollama | Ollama server | Yes, selected by the server | Pick a quantized **tag** (e.g. `...:7b-instruct-q4_K_M`) | effGen neither knows nor controls the quant; the harness records the tag only. |
| GGUF in-process | llama-cpp-python | Yes (optional) | `load_model("x.gguf")` → `GGUFEngine` | `gguf` extra. Not used in MVP. |

### 1.8 Tool registration and execution

- **Base classes:** `BaseTool` (`tools/base_tool.py`), which declares `ToolMetadata(name, description, category, parameters=[ParameterSpec...], timeout_seconds=30, requires_approval)` and implements `async _execute(**kw)`. `BaseTool.execute()` handles normalization, coercion, validation and error wrapping, and returns a `ToolResult`.
- **`@tool` / `Tool.from_function`** (`tools/function_tool.py`): derives the schema from type hints and the docstring. A sync function runs in an executor with the caller's context copied.
- **Hidden parameters:** `ParameterSpec(model_facing=False)` hides a parameter from the model-facing schema.
- **Timeouts:** `ToolMetadata.timeout_seconds` is **not enforced** by the core loop (no reference in `effgen/core`). Tools enforce their own timeouts (e.g. `BashTool`).
- **Restriction:** the agent can only call tools in `config.tools`. `ToolFallbackChain` (`tools/fallback.py`) only falls back to tools that are also held (`agent_tool_execution.py:154`).
- **Middleware** (`core/middleware.py`): `AgentMiddleware.before_run/after_run/before_model_call/after_model_call/before_tool_call/after_tool_call`.
  - A `before_*` hook may short-circuit.
  - **A hook that raises propagates and stops the run.**
  - Configure with `AgentConfig(middleware=[...])` or per call with `run(middleware=[...])`.

### 1.9 Reusable coding tools and workspace safety

| Tool | File | Capabilities | Safety | Gaps for us |
|---|---|---|---|---|
| `file_operations` (`FileOperations(allowed_directories=[...])`) | `tools/builtin/file_ops.py` | read, write (whole file), list, search (**filename glob only**), metadata, convert | Path containment to allowed dirs (resolved, traversal-safe); credential-filename deny list; extension allowlist for read/write | No in-place edit or patch. No content search. Extensionless files (e.g. `Makefile`) fail the extension allowlist. |
| `bash` (`BashTool(working_directory=..., timeout=30, allowed_commands=...)`) | `tools/builtin/bash_tool.py` | shell commands | Block list, dangerous-pattern regexes, env stripping, timeout with process-group kill, output cap | **Explicitly not a sandbox** ("Run a shell command with the calling user's full privileges"). The cwd is not a boundary. |
| `code_executor` (`CodeExecutor(workdir=...)`) | `tools/builtin/code_executor.py`, `security/sandbox.py` | python, javascript, bash, sh | `SubprocessSandbox` uses `unshare` user/mount/net/pid namespaces: writes confined to `workdir`, private `/tmp`, network off, `ulimit` memory, timeout. `DockerSandbox` uses `--read-only --network=none`. | Docker does **not** mount the workspace, so it is useless for repo work. Force `EFFGEN_SANDBOX_BACKEND=subprocess`. `network_enabled` and `memory_limit` are **model-facing** parameters that must be pinned. The default 256m `ulimit -v` may break test runners. Reads are not confined. |
| Gated coding tools | `effgen/cli/code/tools.py` (`build_code_tools`, `build_review_tools`, `ReadOnlyFileOperations`, `PermissionGate`) | Permission-gated variants plus a read-only file tool | Modes: plan / ask / auto-edit / yes | CLI-internal module, not in `effgen.__all__`, so API stability is not promised. Candidate for read-only workers later. |
| `web_search`, `agentic_search`, `url_fetch`, ... | `tools/builtin/*` | — | — | **Never register.** Note the `duckduckgo-search` and `ddgs` packages are base dependencies of effGen; installed but unused. |

The user-namespace prerequisite was checked on this host on 2026-10-16: `unshare --map-root-user --mount --net true` succeeded. effGen's own probe was not run.

### 1.10 Execution limits

| Limit | effGen mechanism | Enforced? |
|---|---|---|
| Max agent steps | `AgentConfig.max_iterations` (default 10); per-call `run(max_iterations=...)` | Yes (loop) |
| Stall guard | `max_turns_without_progress` (default 2), repeat guards, closing request | Yes |
| Context budget | `context_budget="auto"`, `compaction` | Yes (compaction is out of MVP scope, but the defaults remain active) |
| Output tokens | `max_tokens` | Yes |
| Wall-clock run timeout | none on `run()` | **No.** Build from middleware. |
| Sub-agent depth | `max_sub_agent_depth=3` (applies to SubAgentManager only) | Not relevant to agent-as-tool. We enforce depth 1 structurally. |
| Team / sub-agent timeouts | `TeamConfig.timeout`, `SubAgentConfig.timeout` | **No** (declared, unused) |
| Shell timeout | `BashTool(timeout=)`, `CodeExecutor` `timeout` | Yes |
| HTTP request timeout | `OpenAICompatibleAdapter(timeout=60)` | Yes |
| Generic timeout helper | `reliability/timeouts.py:with_timeout` (SIGALRM on the main thread, async-exception injection otherwise) | Best-effort. Cannot interrupt a blocking CUDA `generate`. |

### 1.11 Error handling and tracing

- **Errors:**
  - `RunStoppedError` (has `.response`, `.partial`, `.stop_reason`) when `raise_on_error=True`.
  - `BackendUnreachableError`, `BudgetExceededError` and `ContextBudgetExceededError` (`models/errors.py`).
  - Tool errors become observations.
- **Tracing:** `AgentResponse.execution_trace`, `execution_tree`, `metadata["thread"]` (`AgentThread`, serializable via `to_dict()`), `ExecutionTracker` events (`core/execution_tracker*.py`), and a run ledger (`core/ledger.py`).
- These are enough for MVP lifecycle reporting. No custom observability is needed.

---

## Part 2: Harness integration status

| Capability | effGen support | Our implementation | Verification | Notes |
|---|---|---|---|---|
| Agent execution | Verified (source) | Not started | Not tested | `Agent` + `AgentConfig`; see 1.1 |
| Subagent spawning | Partial: agent-as-tool supported; built-in managers unsuitable | Not started | Not tested | Custom `delegate` tool (see architecture.md) |
| Context isolation | Verified (source) | Not started | Not tested | Fresh worker `Agent` per delegation around a shared model |
| Ollama loading | Verified (source), via `openai_compatible` | Not started | Not tested | Ollama not installed on this host |
| Hugging Face loading | Verified (source) | Not started | Not tested | Must pass `quantization_bits` explicitly |
| Mixed providers | Plausible (independent `BaseModel` instances per agent) | Not started | Not tested | Unverified on a real run |
| Quantization | Verified (source): bf16/fp16, bnb 8-bit, bnb NF4, Ollama GGUF tags | Not started | Not tested | FP4 not supported by effGen API |
| Tool execution | Verified (source) | Not started | Not tested | `FileOperations` + sandboxed shell |
| Execution limits | Partial: steps yes, wall clock no | Not started | Not tested | Deadline middleware needed |

Status values: Not started, In progress, Implemented (unverified), Tested, Blocked.

## Part 3: Integration history

### 2026-10-16 – T0.1 Project bootstrap
- Added: uv project with `effgen==1.3.0` pinned. Verified install facts (first runtime evidence; everything else in Part 1 is still source-level):
  - `effgen==1.3.0` is on PyPI.
  - On Python 3.12.13 the default install resolves `torch 2.14.1+cu130` (CUDA available here) and `transformers 5.20.0`.
  - `bitsandbytes` is not installed by default. It is needed from T3.4.
  - `ddgs` and `duckduckgo-search` are installed. Never register web tools.
  - `import effgen` works and `effgen.__version__ == "1.3.0"`.
- effGen APIs used: `effgen.__version__` only.
- Files: `pyproject.toml`, `uv.lock`, `tests/test_import.py`.
- Tests: `uv run pytest -q` → 2 passed; `uv sync --locked` reproduced the environment from a deleted `.venv`.
- Limitations: none found.

(Further entries follow this format.)

_Format for later entries:_ Add one entry per completed task in `docs/implementation-plan.md`, using this format:

```
### <date> – <Task ID> <title>
- Added: ...
- effGen APIs used: ...
- Files: ...
- Tests: <command> → <result>
- Limitations / workarounds: ...
- Decisions: ...
```
