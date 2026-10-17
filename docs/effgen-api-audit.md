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
| Method | Sections 1.1 to 1.11 are from static reading of shallow clones; **no effGen code was executed when they were written**, so their evidence is source-level unless a line says otherwise. Runtime findings made since (against the installed `effgen==1.3.0`, `transformers 5.20.0`, `torch 2.14.1+cu130`) are in Section 1.12, the corrections noted in 1.6, and the Part 3 history. Where they differ from an earlier line, the later runtime finding wins. |

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
- **Tool calling:** `supports_tool_calling()` is True only if the chat template renders tools. Otherwise the ReAct text protocol is used. (For a model whose template renders tools, such as LFM2.5, this is `True` yet the native parser may not read the model's call format; see 1.12.)
- **Device and `require_gpu` (runtime-verified in T3.2a; supersedes the signature line above):** the engine has **no device parameter**: `load()` uses CUDA whenever `torch.cuda.is_available()` and a `device` kwarg is dropped. `device_map="cpu"` (with `require_gpu=False`, `use_flash_attention=False` and an explicit `torch_dtype`) pins CPU placement on a CUDA host. `require_gpu=True` only rejects *partial* GPU placement and does **not** fail when CUDA is unavailable. With `torch_dtype` omitted, a CPU-pinned load is bf16 on a CUDA host and fp32 without one. Details and evidence: Part 3, T3.2a.

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
| `file_operations` (`FileOperations(allowed_directories=[...])`) | `tools/builtin/file_ops.py` | read, write (whole file), list, search (**filename glob only**), metadata, convert | Path containment to allowed dirs (resolved, traversal- and symlink-safe; **runtime-verified T4.1**); credential filename and content deny list; extension allowlist for read/write | No in-place edit or patch. No content search. Extensionless files (e.g. `Makefile`) fail the extension allowlist (verified). **A missing allowed dir is silently replaced by `EFFGEN_WORKSPACE` or the cwd** (verified; see Part 3, T4.1). Check-then-use race; hard links not detected. |
| `bash` (`BashTool(working_directory=..., timeout=30, allowed_commands=...)`) | `tools/builtin/bash_tool.py` | shell commands | Block list, dangerous-pattern regexes, env stripping, timeout with process-group kill, output cap | **Explicitly not a sandbox** ("Run a shell command with the calling user's full privileges"). The cwd is not a boundary. |
| `code_executor` (`CodeExecutor(workdir=...)`) / `SubprocessSandbox` | `tools/builtin/code_executor.py`, `security/sandbox.py` | python, javascript, bash, sh | `SubprocessSandbox` uses `unshare` user/mount/net/pid namespaces. **Runtime-verified (T4.2):** writes confined to `workdir` (rest read-only, remount denied), private `/tmp` when `workdir` is not under `/tmp`, network off, credential stores masked, `ulimit -v` memory limit enforced, private PID namespace. `DockerSandbox` uses `--read-only --network=none`. | Docker does **not** mount the workspace, so it is useless for repo work. **Off Linux, or without user namespaces, `SubprocessSandbox` silently runs plain `bash`** (result flags say so only afterwards). **Reads are not confined.** **Pathname Unix sockets (Docker, D-Bus, systemd, gpg-agent, resolved) stay reachable through the net namespace** (measured). **Its timeout path kills only the outer `unshare`, leaves descendants running and then blocks until their pipes close** (measured). `ulimit -u 256` is per user. `CodeExecutor` exposes `network_enabled` and `memory_limit` to the model. We use `SubprocessSandbox` directly (see Part 3, T4.2). |
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

### 1.12 Local-model tool calling and context budget (runtime-verified in T3.2c unless marked source-level)

Framework behaviour only; what the harness did with it is in Part 2 and Part 3.

- **Tool-calling modes.** `AgentConfig.tool_calling_mode` = `"native"` | `"react"` | `"hybrid"` | `"auto"` (`effgen/core/tool_calling.py`: `get_strategy` line 1938, `ReActStrategy` 1306, `NativeFunctionCallingStrategy` 1580, `HybridStrategy` 1893). For a local Transformers model, native mode passes `tools=[...]` to `generate`, and the engine renders them through the tokenizer's chat template (`models/transformers_engine_sampling.py:_apply_chat_template`, line 218). ReAct mode sends a flat prompt without `tools`.
- **Native parser formats** (source-level list, `NativeFunctionCallingStrategy.parse_response`, line 1596): Gemma channel calls, Qwen `<tool_call>{json}</tool_call>`, an XML-parameter dialect, `<function=NAME>`, Llama `<|python_tag|>`, Mistral `[TOOL_CALLS]`, and a generic JSON object reader (`_json_tool_call`, line 37). **Runtime/model-free check on LFM2.5's formats** (`tests/test_lfm_parser_compat.py`): the default Pythonic `<|tool_call_start|>[add(a=1, b=2)]<|tool_call_end|>` is **not** read by native or hybrid; the JSON-list variant `<|tool_call_start|>[{"name": "add", "arguments": {...}}]<|tool_call_end|>` **is** read (native, hybrid); the ReAct reader reads only `Action:` / `Action Input:`.
- **Special-token handling.** On the native path the engine decodes with special tokens kept and strips the tokenizer's `all_special_tokens` except effGen's delimiter set (`models/transformers_engine_support.py`: `_TOOL_CALL_DELIMITERS` line 72, `_strip_special_tokens_keep_tool_calls` line 82). LFM2.5's `<|tool_call_start|>` / `<|tool_call_end|>` (ids 10, 11) are *non-special* added tokens and are not in that delimiter set, so they stay in the text (observed in the raw model output).
- **Model-generated call versus executed call (observed).** A tool call written by the model is only an executed call when effGen parses it and dispatches it. effGen records dispatches in `AgentResponse.tool_calls` (a `ToolCall` with `name`, `arguments`, `result`, `duration`, `iteration`). In native mode with LFM2.5 the model wrote a call on the second and third turn, effGen's loop recognised it as a written call (`find_written_tool_call`, `core/agent_runtime.py:756`, used at `core/agent_loop.py:1102`) and stopped with `stop_reason="written_tool_call"`, `termination="error"`, `tool_calls == []`, and the tool never ran. Which reader inside that check matched was not traced. In ReAct mode the model also wrote an invented `Observation:` after its `Action:`; effGen ran the real tool and discarded the text after the action, and the real result reached the next prompt (the written-action guard; `core/agent_runtime.py`).
- **Chat-template wrapping.** `_apply_chat_template` builds `messages = [{"role": "user", "content": prompt}]` (line 228): the whole flat prompt, including `AgentConfig.system_prompt`, is a single user turn. effGen's Transformers engine therefore cannot send a system-role message (pinned by a model-free test). `chat_template_kwargs` exists on the engine but was not used.
- **Declared context window.** `TransformersEngine._get_max_length` (`models/transformers_engine_loading.py`, line 299) returns the first of `max_position_embeddings`, `n_positions`, `seq_length` found in the model config, so LFM2.5-350M reports 128,000 (`config.json`; RoPE `default`, `rope_theta` 1e6, no scaling; tokenizer `model_max_length` is unbounded). The model card states 32,768.
- **Context budget.** `resolve_context_budget` (`core/thread_budget.py`, line 419) uses `window_override or _declared_window(model)` (the override is `AgentConfig.max_context_length`, passed at `core/agent_loop.py:3405`); with `context_budget="auto"` the prompt budget is `int((window - reserve) * DEFAULT_HEADROOM)` with `DEFAULT_HEADROOM = 0.85` (line 70) and `reserve = min(max(output_tokens, 8192), window // 4)`. Verified numbers for this model: window 128,000 gives a budget of 101,836; with `max_context_length=32768` the window is 32,768 and the budget 20,889 (reserve 8,192). `resolve_context_budget` is a module-level function, not in `effgen.__all__`.
- **Reproducibility.** On CPU with fp32, the same `seed` (and `temperature=0.1`) reproduced identical model text and outcomes across repeated runs.
- **Unresolved (not verified):**
  - **Native tool calling for LFM2.5's default format.** It fails today. A fix needs an approved parser adapter or a way to deliver a system-role "output JSON calls" instruction (the model card documents the JSON option); neither exists in effGen 1.3.0 and neither was attempted.
  - **`tool_calling_mode="auto"`** for this model was not tested (the template renders tools, so `supports_tool_calling()` is `True`); set `react` explicitly.
  - **Long context.** Whether the model behaves well anywhere near 32,768 or 128,000 tokens was not measured (prompts were at most about 2,000 characters). 32,768 is the card's figure and is used only as a conservative planning limit.
  - Multi-step, multi-argument and string-argument tool use; any delegation with this model; the fanless edge device.
  - Why `Agent.run` reported `tokens_used=32` for a prompt that `generate` reported as 2 tokens.

---

## Part 2: Harness integration status

| Capability | effGen support | Our implementation | Verification | Notes |
|---|---|---|---|---|
| Agent execution | Verified (source + runtime) | Implemented (orchestrator + worker `Agent`s via `Harness`) | Tested with scripted models (T1.1–T1.3) | `Agent` + `AgentConfig`; see 1.1 |
| Subagent spawning | Partial: agent-as-tool supported; built-in managers unsuitable | Implemented (T1.2–T1.3) | Tested end-to-end with scripted models (T1.3); real models pending | Custom `delegate` tool (see architecture.md) |
| Context isolation | Verified (source + runtime control test) | Implemented for workers (fresh `Agent` per delegation) | Tested at unit level (T1.2) | Fresh worker `Agent` per delegation around a shared model |
| Ollama loading | Verified (source + offline construction), via `openai_compatible` | `build_model` implemented (T3.2) | Adapter construction tested offline; server contact pending T3.3 | Ollama not installed on this host |
| Hugging Face loading | Verified (source + routing; real engine on a tiny local model for CPU and CUDA) | `build_model` with explicit `device` implemented (T3.2, T3.2a) | Argument flow and real placement tested on a tiny model; a real checkpoint (LFM2.5-350M) loaded and ran on CPU (T3.2b); real GPU/quantized checkpoint pending T3.4 | Must pass `quantization_bits` explicitly |
| Mixed providers | Plausible (independent `BaseModel` instances per agent) | Config + `ModelCache` support it (T3.1, T3.2) | Cache/identity tested with fakes; real mixed run pending T3.5 | Unverified on a real run |
| Quantization | Verified (source): bf16/fp16, bnb 8-bit, bnb NF4, Ollama GGUF tags | Config validation (T3.1) and `quantization_bits` mapping (T3.2) implemented | Mapping tested; real quantized load pending T3.4 | FP4 not supported by effGen API |
| Tool execution: dispatch of model-written calls | Verified at runtime: ReAct dispatch with a real small model; native with that model fails (1.12) | Used as is (effGen loop); no harness code | ReAct 2/2 and native 0/2 on `LiquidAI/LFM2.5-350M` (T3.2c, opt-in); scripted tests for the rest | Recommended mode for that model: `react` |
| Tool execution: workspace tools (`file_operations`, `shell`) | `FileOperations` and `SubprocessSandbox` verified at runtime (T4.1, T4.2) | `build_tools` with `file_operations` (T4.1) and `shell` = `ShellTool`, a `BaseTool` over `SubprocessSandbox` (T4.2) | `file_operations` Tested (T4.1); `shell` Tested with the real sandbox on Linux (121 tests in `tests/test_tools.py`); Windows/WSL2 not tested | effGen silently widens a missing allowed dir to the cwd (factory guards it). SubprocessSandbox timeout/cleanup defect and Unix-socket gap are worked around (Part 3, T4.2). **Scope: trusted local development only (user decision 2026-10-17); not a complete boundary.** Unresolved R1: Unix sockets outside `/run`, no read isolation; Windows/WSL2 unverified. Not yet wired into `build_harness` from config **Loop guards: effGen replays identical repeat calls (assumes pure tools); fixed by the explicit process-wide F1 compatibility patch, which `Harness.run` requires (Part 3).** |
| Device selection (CPU / CUDA) | Verified at runtime on a tiny model and a real checkpoint (1.6, T3.2a, T3.2b) | Implemented (T3.2a): `device` on hf specs, CPU pinning, CUDA preflight | Tested; CPU run on a real 350M checkpoint; CUDA run on a tiny model only | `require_gpu` does not fail without CUDA |
| Context window / budget | Verified at runtime for LFM2.5 (1.12) | Not implemented: the harness does not set `max_context_length` (no config field yet) | Budget numbers verified; long-context behaviour untested | Card says 32,768; config declares 128,000 **T5.0:** Ollama specs pass `max_context_length = context_length` (agent-side, explicit; not the server's `num_ctx`); HF still unset. |
| Execution limits | Partial: steps yes, wall clock no (hooks verified) | Worker deadline (cooperative, T2.1), failure propagation (T2.2), `max_delegations` and report truncation (T2.3) implemented; iteration caps are effGen's | Tested with scripted models (T2.1–T2.3) | Deadline middleware needed **T5.0 (F8):** a worker's shell command is bounded by the worker's remaining deadline; a generation in flight is still not interrupted. |

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

### 2026-10-16 – T1.1 Scripted-model test fixture
- Added: `ScriptedModel` fake `BaseModel` for deterministic CPU tests.
- effGen APIs used: `effgen.models.base.BaseModel` (`generate`, `generate_stream`, `count_tokens`, `get_context_length`, `load`, `unload`; `GenerationResult`, `TokenCount`, `ModelType.OPENAI`), `Agent` / `AgentConfig`, `@tool`.
- Files: `tests/conftest.py`, `tests/test_scripted_model.py`.
- Tests: `uv run pytest -q` → 17 passed.
- **Verified at runtime (effGen 1.3.0, ReAct mode with `tool_calling_mode="react"`, `capability_probe=False`, `enable_sub_agents=False`, `enable_memory=False`, `raise_on_error=False`):**
  - A `BaseModel` subclass needs only the six methods above. The abstract set is smaller than the upstream fake's overrides.
  - ReAct text that works: `Thought: ...\nAction: <tool>\nAction Input: {json}`, then `Thought: ...\nFinal Answer: <text>`. The agent makes exactly 2 model calls and 1 tool call.
  - The prompt is a flat string (`str`) and carries `Observation: <result>` on the next call. `tools=` is not passed to `generate` in react mode.
  - Unknown tool → observation `No tool named '<x>' is available. The tools you can use are: <list>. ...`; the nudge consumes an extra model turn.
  - Tool exception → observation `Error executing tool '<name>': Tool execution failed: <msg>`; the run continues.
  - A model exception → `AgentResponse(success=False, termination="error")` with `output` starting `Generation failed: ...`. It is not raised under `raise_on_error=False`.
  - A repeated identical tool call is stopped by effGen: `termination="stuck"`; the repeated call is not re-executed.
  - `AgentResponse.tool_calls` is a `ToolCallList` (not an int); `.iterations`, `.termination` are populated.
- Limitations: native (non-ReAct) tool calling is untested.
- Decisions: `on_exhausted="repeat_last"` default (plan T1.1 risk); `"raise"` for exact-count assertions.

### 2026-10-16 – T1.2 Delegation contract (`WorkerResult`, `WorkerRegistry`, `DelegateTool`)
- Added: agent-as-tool delegation layer with fresh worker `Agent` per call.
- effGen APIs used: `BaseTool`, `ToolMetadata`, `ParameterSpec`, `ToolCategory.COMMUNICATION`, `BaseTool.execute()` → `ToolResult`; `Agent`, `AgentConfig` (`dataclasses.replace`), `Agent.run`, `Agent.close` (via `with agent`); `AgentResponse` (`success`, `output`, `termination`, `iterations`, `tool_call_count`, `tokens_used`).
- Files: `src/smol_harness/delegation.py`, `tests/test_delegation.py`.
- Tests: `uv run pytest -q` → 39 passed.
- **Verified at runtime (effGen 1.3.0):**
  - `BaseTool.execute(**kw)` validates before `_execute`. Missing required, empty (`min_length`), or wrong-type (`5` for a string, `3` for `worker`) arguments return `ToolResult(success=False, error="Parameter validation failed: Parameter 'x' is required. Pass the parameters ...")` and `_execute` never runs. Unknown extra parameters are dropped with a warning.
  - A tool returning a `str` is passed through as the observation text; JSON strings are fine.
  - Reusing one `Agent` across `run()` calls **does** leak the earlier task into the next prompt (control test), confirming §1.4: fresh `Agent` per delegation is required. `Agent(replace(config, tools=list(...)))` with a shared `BaseModel` instance keeps `agent.model is model`, with per-agent `short_term_memory` and `state`.
  - `agent.run()` on a worker with `raise_on_error=True` and a failing model raises `RuntimeError` (message contains the model's error); with `False` it returns `success=False`. Both are caught and reported as `failed`.
  - A worker that repeats a tool call ends `success=False`, `termination="stuck"`.
  - `AgentResponse.tool_call_count` is the plain int (`tool_calls` is a `ToolCallList`).
  - `ToolCategory.COMMUNICATION` → general contract, `ToolUsePolicy.AUTO` (`prompts/tool_contract.py:130-201`). `SYSTEM` and `CODE_EXECUTION` are `REQUIRED`.
- **Source-level only (not exercised):** when `Agent.run` is called under a running event loop (e.g. `run_async`), effGen's coroutine bridge `run_coroutine_sync` (`utils/async_bridge.py`) gives sync-dispatched async tools a **120 s** default timeout. A plain synchronous `agent.run()` uses `asyncio.run` with no cap. Keep the orchestrator on the sync entry point, or revisit before using `run_async`.
- Limitations / workarounds: workers are built with `enable_sub_agents=False` by the registry; `DelegateTool` runs `Agent.run` in an executor with the caller's `contextvars` copied (as effGen's `FunctionTool` does).
- Decisions: structured failure for unknown worker, blank task and any worker failure; effGen's validation handles malformed arguments.

### 2026-10-16 – T1.3 End-to-end orchestrator → worker → orchestrator
- Added: `Harness` / `build_harness` / `HarnessResult`, `WorkerRecord`.
- effGen APIs used: `Agent`/`AgentConfig` (orchestrator with `DelegateTool` in `tools`), `Agent.run`, `AgentResponse` (`success`, `output`, `termination`, `stop_reason`).
- Files: `src/smol_harness/harness.py`, `src/smol_harness/delegation.py`, `tests/test_e2e_delegation.py`.
- Tests: `uv run pytest -q` → 61 passed.
- **Verified at runtime (effGen 1.3.0, ReAct mode, scripted models):**
  - A parent ReAct turn `Action: delegate` / `Action Input: {json}` runs `DelegateTool` through the normal loop; the returned JSON string appears verbatim as `Observation: {...}` in the parent's next prompt. The parent made exactly 2 model calls (delegate turn + final turn).
  - A custom `system_prompt` on the orchestrator appears in its own first ReAct prompt, so isolation checks against it are meaningful.
  - A worker run *inside* the parent's tool call works: the worker is a separate `Agent` with its own prompts, and the parent's prompts never contain the worker's thoughts or observations.
  - Parent tool failure handling (**U11**):
    - Missing required argument (`worker` or `task` absent): effGen does not dispatch; the observation is `[Your tool call could not be run: it did not carry the required argument(s) task. Make the call again as a tool call with its arguments as a JSON object.]`
    - Present but invalid (empty `task`, non-string `task`/`worker`): `Error executing tool 'delegate': Tool execution failed: Parameter validation failed: Parameter 'task' must have length >= 1. Pass the parameters 'delegate' requires: worker (string), task (string).`
    - In both cases the parent continues and can finish (`success=True`; `termination` was `done`, or `not_possible` once after two consecutive missing-argument calls). `_execute` is never reached, so no `WorkerRecord` is created.
    - The same invalid call repeated endlessly ends the parent with `termination="stuck"`, `stop_reason="loop_detected"` after 3 calls. Different invalid calls repeated until the limit end with `stuck` / `max_iterations_exhausted` after `max_iterations` calls. **`tool_failed` was never produced.**
  - A worker whose model raises (parent unaffected): recorded `failed`, parent finishes `done` (**U3** for this failure kind).
  - **Worker nudge:** a worker that holds a tool but answers `Final Answer:` without calling it gets an extra turn: `Observation: You have not run a tool yet. Run the <tool> tool and answer from what it returns, not from what you expect it to return.` The run then continues and makes a second model call (`iterations` 2). Script worker turns accordingly, and expect small workers that skip tools to cost extra calls. Applies to `@tool`-decorated tools (default category).
- Limitations: ReAct text only; native tool-calling path not exercised.
- Decisions: orchestrator `Agent` and record list are created per `Harness.run` because effGen agents keep memory across runs (§1.4).

### 2026-10-16 – T2.1 Worker deadline enforcement (cooperative)
- Added: `DeadlineMiddleware`, `WorkerTimeout`, `WorkerResult.status="timeout"`; `worker_timeout_s` on `DelegateTool` / `Harness`.
- effGen APIs used: `effgen.AgentMiddleware` (`before_model_call`, `before_tool_call`), per-call `Agent.run(task, middleware=[...])`, `AgentResponse.success/termination/metadata["error"]`.
- Files: `src/smol_harness/delegation.py`, `src/smol_harness/harness.py`, `tests/test_limits.py`.
- Tests: `uv run pytest -q` → 77 passed.
- **U4, verified at runtime (effGen 1.3.0, ReAct loop):**
  - An exception raised from `before_model_call` or `before_tool_call` is **not swallowed**; it stops the run, and no further model calls are made afterwards (the loop's closing request does not retry).
  - With `raise_on_error=True`, `run()` re-raises the **original exception object** (`WorkerTimeout`, not wrapped).
  - With `raise_on_error=False`, `run()` returns `success=False`, `termination="error"`, `output="Error: <message>"`, and `metadata["error"]` = `{"type": "WorkerTimeout", "category": "timeout", ...}`. effGen classifies it itself, but the harness does not rely on that: it uses its own `expired` flag.
  - `before_tool_call` raising prevents the tool from running.
  - Per-call middleware (`run(middleware=[...])`) fires both hooks, the same as `AgentConfig(middleware=[...])`.
  - The hooks only run at call boundaries: a tool or generation already running is never interrupted. A tool that sleeps 0.3 s completes under a 0.1 s deadline; the stop happens at the next model-call hook.
- Source-level only: native tool calling, streaming and `run_async` dispatch through the same `_generate` / `_execute_tool` wrappers (`agent_generation.py:_generate`, `agent_tool_execution.py:_execute_tool`), so the hooks should apply there, but this was not run.
- Limitations: **cooperative timeout only**; worst-case overrun is one model call or one tool call. `ToolMetadata.timeout_seconds` remains unenforced by effGen's core loop (§1.8).
- Decisions: detect timeouts by the middleware's `expired` flag, not by parsing effGen's error text or type, so both `raise_on_error` modes work.

### 2026-10-16 – T2.2 Worker failure propagation
- Added: `RunStoppedError` unwrapping in `DelegateTool`, `WorkerResult.delegation_id`.
- effGen APIs used: `RunStoppedError` (`.response`, `.stop_reason`, `.partial`), `BackendUnreachableError`, `AgentResponse.termination/stop_reason/metadata`, `AgentConfig.raise_on_error`.
- Files: `src/smol_harness/delegation.py`, `tests/test_limits.py`.
- Tests: `uv run pytest -q` → 109 passed.
- **Exact worker exception paths, verified at runtime (effGen 1.3.0, scripted models, ReAct):**

  | Failure | `raise_on_error=False` | `raise_on_error=True` |
  |---|---|---|
  | Iteration cap (`max_iterations=2`) | returns `success=False`, `termination="stuck"`, `stop_reason="max_iterations_partial"`, output `Stopped after 2 iterations without a final answer: ...` | raises `effgen.errors.RunStoppedError` (a `RuntimeError`); `.response.termination="stuck"`, `.stop_reason="max_iterations_partial"` |
  | Worker tool failing on its own side (3 attempts) | returns `termination="tool_failed"`, `stop_reason="tool_failed"`, output `The 'flaky' tool failed on its own side after 3 attempts (ConnectionError...` | raises `RunStoppedError` with `.stop_reason="tool_failed"` |
  | Backend unreachable (`BackendUnreachableError` from the model) | **raises `BackendUnreachableError`** (not a `RuntimeError`; message `<provider> did not answer (model='m'): connection refused. Nothing answered at that endpoint...`) | same: raised |
  | Unexpected model exception (`ValueError("boom")`) | returns `success=False`, `termination="error"`, `stop_reason="generation_failed"`, `metadata["error"]["type"]="ValueError"` | raises plain `RuntimeError("ValueError: boom")`; original type only in the text |

- `BackendUnreachableError` therefore reaches `DelegateTool` as an exception in both modes. It is reported as `status="failed"`, `error_type="BackendUnreachableError"`, `termination=None`.
- **Generation retries (source-confirmed, then observed):** for transient or unknown error categories (including unreachable and unexpected exceptions), `_generate_instrumented` makes 3 attempts with sleeps `[0.5, 1.0]` s (hard-coded `max_retries=3`, `backoff_delays=[0.5, 1.0, 2.0]`; no config option). Non-retryable categories (auth, not found, refusal, invalid request) and rate limits fail fast. Test-only: `effgen.core.ledger.backoff_sleep` is replaced by a recorder; the tests assert `attempts == 3` and sleeps `[0.5, 1.0]`.
- **`repeated_tool_result` guard (important):** a run whose tool returns the same result text twice is stopped: `termination="stuck"`, `stop_reason="repeated_tool_result"`, output `'orch' did not write an answer: 'delegate' returned the same result twice...`, `success=False`. Two failed delegations of *different* tasks produced byte-identical JSON and triggered it after 2 of 3 delegations. Fix: `delegation_id` makes each report unique. A parent repeating the *identical call* is stopped differently: `stop_reason="loop_detected"`, and effGen does not re-dispatch the repeated call (1 worker record, 4 parent calls).
- **`tool_failed` for the parent:** never observed from worker failures of any kind (8 combinations) nor from repeated worker failures. It was only produced for a *worker's own* failing tool.
- Limitations: unexpected-exception reports differ by path (`worker_run_failed` vs `RuntimeError`) because effGen drops the response when wrapping; `iterations` is 0 when no response exists.
- Decisions: unwrap `RunStoppedError.response` instead of forcing `raise_on_error=False` on workers (keeps the user's config; the registry does not override it).

### 2026-10-16 – T2.3 Delegation limits and report truncation
- Added: `max_delegations`, `max_report_chars`, `bounded()`, `WorkerResult.truncated`, `validate_limits`.
- effGen APIs used: unchanged (`BaseTool.execute`, `AgentConfig.max_iterations`, `AgentResponse.stop_reason`).
- Files: `src/smol_harness/delegation.py`, `src/smol_harness/harness.py`, `tests/test_limits.py`.
- Tests: `uv run pytest -q` → 139 passed.
- **Verified at runtime (effGen 1.3.0):**
  - `max_iterations=N` allows **N model calls** but, when the model never answers, only **N−1 tool dispatches**: the action in the last allowed call is not executed. Parent with `max_iterations=3` that emits a `delegate` call every turn → 3 model calls, 2 worker runs. A worker with `max_iterations=2` → 2 model calls, 1 tool run, `iterations == 2`, structured failure.
  - Parent `stop_reason` on hitting the cap: `max_iterations_partial` when at least one tool result was obtained (including delegation refusals), `max_iterations_exhausted` when none was (e.g. only invalid-argument calls, T1.3/U11). Both give `termination="stuck"`, `success=False`.
  - effGen shortens long exception text itself: under `raise_on_error=True` a 25 000-char `ValueError` message becomes a `RuntimeError` of ~270 chars ending `… (25000 characters)`; under `False` the failed response's output is ~380 chars. `BackendUnreachableError` text is shortened as well. So worker `error` text is rarely large; `output` (a worker's final answer) is the main report-size risk.
  - Refusals (`delegation_limit_reached`) are successful tool calls with unique ids: a parent that keeps delegating past the limit is *not* stopped by `repeated_tool_result`; it is stopped by its own `max_iterations`.
  - Identical successful worker outputs from different delegations do not trip `repeated_tool_result` (ids differ), even after truncation.
  - Valid-JSON/size guarantees were checked with a 400-case deterministic fuzz (hostile characters, huge fields, limits from `MIN_REPORT_CHARS` up).
- Limitations: truncation is by characters, not tokens; the 64-char identifier cap loses a long registered worker name when a report must be shrunk.
- Decisions: serialize with `ensure_ascii=False` (limits count characters; the parent reads text); keep `WorkerRecord.result` untruncated for debugging; limits default to `None`.

### 2026-10-16 – T3.1 Configuration layer (no effGen integration)
- Added: `src/smol_harness/config.py`; PyYAML declared as a runtime dependency.
- effGen APIs used: none. The validation rules encode earlier findings: HF quantization via `quantization_bits` in {None, 8, 4} with NF4 only (§1.7), VRAM-based auto-quantization when unspecified (§1.7), Ollama via the OpenAI-compatible adapter with an assumed 32768 context window when `context_length` is not given (§1.5), and tools limited to those the harness will build.
- Tests: `uv run pytest -q` → 318 passed.
- **Newly verified:** importing `smol_harness.config` and loading the example config does not import `torch` or `transformers` (checked via `sys.modules`); `effgen` itself is imported lazily and only `Agent`-level modules (including `effgen.models.base`) are loaded.
- Limitations: the rules reflect effGen 1.3.0 source-level findings (§1.5–§1.7). Whether each accepted combination actually loads is verified only in T3.3–T3.5.

### 2026-10-16 – T3.2 Model factory and cache
- Added: `src/smol_harness/models.py` (`build_model`, `ModelCache`).
- effGen APIs used: `effgen.models.load_model(model_id, provider=, base_url=, context_length=, engine=, quantization_bits=, require_gpu=, torch_dtype=)`, `OpenAICompatibleAdapter`, `BaseModel.model_type` / `.unload()`.
- Files: `src/smol_harness/models.py`, `tests/test_models.py`.
- Tests: `uv run pytest -q` → 405 passed.
- **Verified at runtime (effGen 1.3.0, no model downloaded, no network; sockets blocked in the test):**
  - `load_model("qwen2.5-coder:7b-instruct-q4_K_M", provider="openai_compatible", base_url="http://127.0.0.1:11434/v1", context_length=8192)` returns an exact `OpenAICompatibleAdapter` with `model_name` = the full tag (colon kept), `base_url` and `get_context_length() == 8192` as given, `is_loaded() == True`, `_provider == "openai_compatible"`, `model_type == ModelType.OPENAI`. Constructing it makes no network call (`openai` client creation only), so the Ollama server is first contacted on first use.
  - **Silent id rewriting (new hazard):** with `provider="openai_compatible"`, effGen strips a leading `<alias>:` from the model id when the alias names that adapter. Confirmed for all of `openai`, `openai_compatible`, `openai-compatible`, `openai_compat`, `oai_compatible`, `compatible`, `server`, `vllm_server`, `local_server`: e.g. `server:latest` → the endpoint is asked for `latest`. Ordinary tags (`llama3:8b`, `qwen2.5-coder:7b-instruct-q4_K_M`, `mistral`) are untouched. The factory rejects such ids.
  - **Quantization auto-selection (new evidence for §1.7):** called without `quantization_bits`, effGen's loader decided on its own. On this 8 GB machine the engine received `quantization_bits=4`; an explicit `quantization_bits=None` reaches the engine as `None` (checked against effGen's real routing with only `TransformersEngine` replaced by a recording stub, and with the VRAM heuristic pinned to 4 so the test is hardware-independent).
  - hf routing delivers to `TransformersEngine(model_name, **params)`: `quantization_bits`, `torch_dtype` and `require_gpu` exactly as passed, plus `apply_chat_template=True` and `device_map="auto"` added by effGen itself (CUDA present). `model.load()` is called inside `load_model`, so a real hf load happens at that call. The convenience `load_model` builds a fresh `ModelLoader` per call, so there is no hidden cross-call cache.
- **Source-level only (not run):** `TransformersEngine.model_type == ModelType.TRANSFORMERS` and its `quantization_bits` attribute (used by the factory's post-load check); `require_gpu=True` raising instead of CPU/offload fallback (TransformersEngine docstring; **partly wrong, see the T3.2a entry: it does not fail when CUDA is absent**); a bare hf id that equals a cloud-catalog slug could be routed to that provider when credentials exist (`model_loader_routing.py`, catalog routing), which the factory's result-type check would catch. Real-load verification is T3.4.
- Limitations: no real model loaded; alias list mirrors effGen's private table (a control test re-checks effGen's behaviour each run).
- Decisions: always pass `quantization_bits` and `require_gpu=True` for hf; refuse Ollama ids effGen would rewrite; one load attempt, no retry or fallback; wrap failures in `ModelLoadError` with the original as `__cause__`.

### 2026-10-16 – T3.2a Explicit Hugging Face device selection
- Added: `ModelSpec.device` (`cuda` default | `cpu`), shared placement rules (`config.hf_placement_problem`), CPU pinning and CUDA preflight in `models.py`.
- effGen APIs used: `effgen.models.load_model(..., engine="transformers", quantization_bits=, require_gpu=, device_map=, torch_dtype=, use_flash_attention=)` and the resulting engine's `.device`, `.model`, `.unload()`.
- Files: `src/smol_harness/config.py`, `src/smol_harness/models.py`, `tests/test_hf_device.py`, `tests/test_models.py`, `tests/test_config.py`.
- Tests: `uv run pytest -q` → 458 passed.
- **Verified at runtime (effGen 1.3.0, torch 2.14.1+cu130, GPU present; real engine on a locally generated 2-layer random Llama, no download):**
  - `TransformersEngine` has **no device parameter**. `load()` sets `self.device="cuda"` whenever `torch.cuda.is_available()` (`transformers_engine_loading.py:load`), and a `device` kwarg is filtered out (`transformers_engine.py`, `additional_kwargs` skip list). `device_map` is only applied when the engine decided on CUDA (`_assemble_model_kwargs`), and when it is present the later `.to(cuda)` step is skipped.
  - **CPU pinning works:** on a CUDA host, `device_map="cpu"` + `require_gpu=False` + `use_flash_attention=False` + explicit `torch_dtype` → every parameter on CPU, `hf_device_map` all CPU, engine `.device == "cpu"`, `torch.cuda.memory_allocated()` unchanged before, after load and after `generate`, generation works. effGen then logs a "did not fit in GPU memory" warning that is misleading in this case; harmless.
  - **`require_gpu=False` alone does not prevent CUDA placement:** without `device_map="cpu"` the same load lands on `cuda:0` (8.6 MB allocated for the tiny model). `require_gpu=True` together with `device_map="cpu"` raises `GPUPlacementError` ("could not be placed entirely on the GPU"), so CPU mode must set `require_gpu=False`.
  - **`require_gpu=True` does not fail when CUDA is unavailable** (simulated with `torch.cuda.is_available` patched to `False`): the model loads on CPU with no error, because it only compares the intended device with the final placement. This corrects the T3.2 note. `build_model` therefore checks CUDA availability itself for `device: cuda`.
  - **Dtype is host-dependent:** CPU-pinned with `torch_dtype` omitted loads as `bfloat16` on a CUDA host and `float32` when CUDA is unavailable, so the config requires an explicit dtype for `device: cpu`.
  - Default CUDA placement (no `device_map`, `require_gpu=True`): `device == "cuda"`, parameters on `cuda:0`; effGen itself adds `device_map="auto"`.
  - effGen skips its `device_map="auto"` sampling probe when `device_map != "auto"`, so pinned placements run no CUDA forward pass.
- **Source-level only:** bitsandbytes quantization on CPU is rejected by config because bnb needs CUDA (effGen's `BitsAndBytesConfig` is built regardless of device); it was not run. `fp16` on CPU is rejected as poorly supported, not measured.
- Limitations: tiny random model only; no real checkpoint, no CPU performance or memory numbers; `cuda:N` selection unsupported.
- Decisions: default `cuda` (existing configs and fail-fast unchanged); explicit dtype required on CPU; shared rule function; verify the engine's reported device after loading.

### 2026-10-16 – T3.2b Liquid AI CPU smoke test (first real checkpoint)
- Added: opt-in integration test `tests/integration/test_liquid_cpu.py` (no source change).
- effGen APIs used: `build_model` → `effgen.models.load_model(path, engine="transformers", quantization_bits=None, require_gpu=False, device_map="cpu", use_flash_attention=False, torch_dtype=torch.float32)`; `model.generate(prompt, GenerationConfig(max_tokens=, temperature=, top_k=, repetition_penalty=))`; `Agent(AgentConfig(..., tools=[], tool_calling_mode="react", capability_probe=False)).run`.
- Tests: default `uv run pytest -q` → 460 passed, 1 skipped; opt-in run → 1 passed.
- **Verified at runtime (effGen 1.3.0, transformers 5.20.0, torch 2.14.1+cu130, real weights `LiquidAI/LFM2.5-350M@9e6c6ccf`, CPU, fp32):**
  - effGen's `TransformersEngine` loads an `lfm2` checkpoint (`Lfm2ForCausalLM`, hybrid conv/attention) with no `trust_remote_code` and no code changes. A local snapshot path routes as a local model (no catalog or provider lookup) and works with the network blocked.
  - The T3.2a CPU kwargs work on a real checkpoint: all 354,483,968 parameters `cpu`/`float32`, engine `device == "cpu"`, CUDA allocation/reservation delta 0, no CUDA context created even though torch sees a CUDA GPU, process not listed by `nvidia-smi`. A BF16 checkpoint loaded with `torch_dtype=float32` is upcast (weights 1352 MiB; peak process RSS 2549 MiB).
  - `model.generate` returns `GenerationResult(text='Paris', tokens_used=2, finish_reason='stop')` in 0.31 s; a no-tool `Agent.run` returns `success=True`, `termination='done'`, `iterations=1`, output `Paris` in 0.21 s. The agent reported `tokens_used=32` for the same prompt (what is counted was not investigated).
  - `model.supports_tool_calling()` is `True` (the chat template renders tools), and `get_context_length()` is `128000`, read from `max_position_embeddings`; the model card states a 32,768-token context.
  - transformers logs that `causal_conv1d_fn` / `causal_conv1d_update` fall back to the reference PyTorch implementation because the optional CUDA package `causal_conv1d` is not installed (correct output, slower).
- **Source-level / not verified:** whether effGen's native tool-calling path parses LFM2.5's default Pythonic `<|tool_call_start|>…<|tool_call_end|>` calls (the card says JSON calls need a system-prompt request); reliability of effGen's ReAct final-answer format on longer prompts; bf16 on CPU; behaviour and timing on the actual edge device.
- Limitations: one trivial prompt; development laptop (actively cooled, 8 torch threads on a 16-thread Ryzen 7 7840HS) is not representative of an edge SoC; the model card says the model is not recommended for programming or knowledge-intensive tasks, so its suitability as a coding worker is doubtful; license tag `other` (LFM1.0) not reviewed.
- Decisions: keep the model (it is instruction-tuned); pin the Hub revision and load from the local snapshot path with the network blocked; run real-model tests only with explicit opt-in; coherence check on the answer; record measurements, assert only placement and success.

### 2026-10-16 – T3.2c Liquid AI tool-calling compatibility (native vs ReAct) and effective context window
- Added: opt-in experiment `tests/integration/test_liquid_tools.py`, measurement helper `tests/tool_probe.py`, parser-compatibility pins `tests/test_lfm_parser_compat.py`. No source change.
- effGen APIs used: `AgentConfig(tool_calling_mode="native" | "react", max_context_length=, seed=, max_tokens=, ...)`, `Agent.run`, `AgentResponse.tool_calls/termination/stop_reason`, `effgen.core.tool_calling.get_strategy(...).parse_response`, `effgen.core.thread_budget.resolve_context_budget` (module-level, not in `effgen.__all__`), `TransformersEngine._apply_chat_template` (private; pinned in a test).
- Tests: default `uv run pytest -q` → 480 passed, 3 skipped; opt-in tool test → 2 passed.
- **Verified at runtime (effGen 1.3.0, real `LiquidAI/LFM2.5-350M@9e6c6ccf`, CPU fp32, two seeds):**
  - **ReAct works:** model writes `Action: add` / `Action Input: {"a": 4817, "b": 9235}`; effGen parses it, runs the real tool (`ToolCall(name='add', result='14052', ...)`), returns the result in the next prompt, and the agent ends `done` with `14052` (2 model calls, 1 tool call, 4.7 to 5.3 s on CPU). The model also appends an invented `Observation: the result is 51002` (seed 1: `50000`) and `Thought:` after its action; effGen's written-action guard runs the tool and **discards everything after the action**, so the invented value never reaches the answer.
  - **Native does not work:** the model emits `<|tool_call_start|>[add(a=4817, b=9235)]<|tool_call_end|>` (its documented default Pythonic format) but only on the second and third call; its first reply is prose with a wrong sum (`12002`, `4818`). effGen detects the written call and stops: `success=False`, `termination="error"`, `stop_reason="written_tool_call"`, 3 model calls, 0 tool calls, tool never executed. This was identical for both seeds and when the card's "Output function calls as JSON." was used as `system_prompt`.
  - **Parser compatibility** (`get_strategy(...).parse_response`): the native reader has no branch for the Pythonic format (it falls through to "no call", the text reads as an answer); it **does** read the JSON-list variant `<|tool_call_start|>[{"name": "add", "arguments": {...}}]<|tool_call_end|>` and bare JSON via its generic JSON reader; the ReAct reader reads only `Action:`/`Action Input:`; hybrid adds nothing for the Pythonic format. `<|tool_call_start|>` / `<|tool_call_end|>` are *non-special* added tokens (ids 10 and 11), so effGen's special-token strip (`_strip_special_tokens_keep_tool_calls`) leaves them in place and they are not in effGen's `_TOOL_CALL_DELIMITERS`.
  - **Native-mode prompt:** effGen sends one flat string (`system_prompt`, task, a "use the tools" nudge) plus `tools=[...]`; `TransformersEngine._apply_chat_template` wraps it as a single `{"role": "user"}` message. effGen therefore cannot place `AgentConfig.system_prompt` in a system turn, which explains why the model card's system-prompt JSON option had no effect. A scripted model that does not declare tool support makes `native` degrade to a ReAct-style flat prompt with no `tools` argument (so tests of native mode need `supports_tool_calling() -> True`).
  - Same seed reproduces identical text and outcomes (CPU, fp32, temperature 0.1).
- **Effective context window (verified numbers):** `config.json` `max_position_embeddings = 128000` with `rope_parameters = {rope_type: default, rope_theta: 1e6}` (no scaling); `tokenizer_config.json` `model_max_length` = 1e30 (unbounded); model card: 32,768. effGen's `TransformersEngine._get_max_length` returns the first of `max_position_embeddings` / `n_positions` / `seq_length` it finds, so `model.get_context_length() == 128000`. `resolve_context_budget("auto", model=model)` → window 128,000, reserve 8,192, **prompt budget 101,836** (`(window - reserve) * 0.85`, reserve = `min(max(output_tokens, 8192), window // 4)`). With `AgentConfig.max_context_length=32768` the override wins (`window_override or declared`): window 32,768, reserve 8,192, **prompt budget 20,889**. The effective limit the harness should plan against is therefore 32,768 tokens with ~20.9K usable prompt tokens under effGen's defaults. **Not verified:** the model's behaviour at long context (prompts here were at most about 2,000 characters); 32,768 is the model card's figure and is used only as a conservative planning limit, not a tested maximum. Source references for all of this are collected in Section 1.12.
- **Source-level only:** whether a system-role instruction would make LFM emit JSON calls (effGen cannot send one); behaviour on tools with several or string arguments; native mode with a different effGen version.
- Limitations: 2 seeds of one trivial task; model arithmetic is unreliable without the tool; the model card says it is not recommended for programming or knowledge-intensive tasks.
- Decisions: record native outcomes rather than pin them in the real-model test (a future parser would show up in the report), but pin effGen's parser behaviour in a model-free test so an upgrade is noticed; success requires an executed tool; no parser workaround was added (needs approval).

### 2026-10-16 – T4.1 Tool factory and workspace-confined `file_operations`
- Added: `src/smol_harness/tools.py` (`build_tools`, `resolve_workspace`, `ToolBuildError`, `TOOL_NAMES`); `tests/test_tools.py` (51 tests).
- effGen APIs used: `effgen.tools.builtin.file_ops.FileOperations(allowed_directories=[workspace])` (not in `effgen.__all__`; the module path is the stable public location of the builtin), `BaseTool.execute()` / `ToolResult`, `Agent.tools` (a dict; empty for `tools=[]`, no defaults injected). Private read: `FileOperations._allowed_directories`, see below.
- Tests: `uv run pytest tests/test_tools.py -q` → 51 passed; `uv run pytest -q` → 531 passed, 3 skipped; `uv lock --check` OK.
- **Verified at runtime** (all through `BaseTool.execute()`; §1.9 facts that were source-level are now tested):
  - Containment is resolve-then-check (`Path.resolve()` then `allowed_dir in resolved.parents`). Refused with `Operation failed: Path '...' is outside allowed directories: [...]` and `ToolResult.success=False` (effGen maps the inner `{"success": False}` to the outer result): `../` traversal, `sub/../../`, `..`, absolute paths outside, a symlinked file or directory pointing outside, and a **write through a dangling symlink** whose target is outside (resolution follows the dangling link). `list`/`search` filter every yielded entry through the same check, so symlinked and `..` glob results are dropped.
  - A relative path resolves against the first allowed directory, not the cwd. `write` creates missing parents. `EFFGEN_WORKSPACE` does not override explicit `allowed_directories` that exist.
  - Extension allowlist: files with no suffix (`Makefile`, `README`, `.gitignore`, `Dockerfile`) are refused for read and write. Credential filenames (`.env`, ...) are refused; a read whose content looks like dotenv `KEY=VALUE` lines with a credential-ish key, or a PEM private key, is refused in any file.
  - **Surprise: a missing allowed directory silently widens confinement.** `FileOperations._normalize_allowed_directories` drops a non-existent directory with only a log warning; if none remain it confines to `EFFGEN_WORKSPACE` (creating it) or the **current directory**. Pinned by `test_effgen_widens_a_missing_allowed_directory_to_the_cwd` and `test_effgen_falls_back_to_and_creates_effgen_workspace`.
- Workaround (no effGen change): `build_tools` validates the workspace first (exists, directory, `R|W|X` access, non-empty path) and, after constructing `FileOperations`, reads the private `_allowed_directories` and refuses unless it equals `[workspace]`. effGen offers no public accessor; this is a read-only check that fails closed if the attribute ever changes shape. Covers the race where the workspace disappears between validation and construction (tested with and without `EFFGEN_WORKSPACE`).
- Limitations: path validation, not a sandbox. Check-then-use race between effGen's resolve and its `open`; hard links and bind mounts inside the workspace are invisible to path checks; no write-size cap; recursive `list` is unbounded; the workspace root itself is not restricted.
- Decisions: `shell` fails closed with a T4.2 message instead of being skipped; names are validated before the workspace and all-or-nothing; `delegate` is never built by the factory (the harness adds it to the orchestrator); no wrapper subclass around `FileOperations` was needed since its own checks held for every escape tested.

### 2026-10-17 – T4.2 Sandboxed `shell` tool (effGen `SubprocessSandbox`)
- Added: `ShellTool` (a `BaseTool`), `ShellSandboxError`, `require_shell_host`, and the `/run` prelude in `src/smol_harness/tools.py`; `resolve_workspace` also rejects filesystem roots and the home directory and its ancestors (T4.1 correction). Tests: `tests/test_tools.py` (121 total, 70 new).
- effGen APIs used (all in `effgen.security.sandbox.__all__` or `effgen.tools.base_tool`): `SubprocessSandbox().run(code, "bash", SandboxConfig(backend="subprocess", timeout, memory_limit, docker_image, network_enabled=False, workdir))`, `SandboxResult` flags (`filesystem_confined`, `writable_root`, `network_isolated`, `process_table_isolated`, `credential_reads_masked`), `BaseTool.validate_parameters` (overridden), `ParameterSpec(min_value, max_value, min_length, max_length)`. Not used: `get_sandbox`, `CodeExecutor`, `DockerSandbox`, `OffSandbox`, `BashTool`. No private effGen attribute is read.
- **Verified at runtime (source reading plus empirical tests; Linux 6.18, unprivileged user namespaces):**
  - **Backends:** `docker`, `subprocess`, `firecracker` (stub), `off` (explicit and unsafe); `auto` prefers Docker, else subprocess with a warning. A pinned backend is cached per process (`_resolved_backend`).
  - **Subprocess command shape:** `unshare --map-root-user --mount --net --pid --fork --mount-proc bash -c <prologue>`; the prologue masks credential stores, remounts every mount read-only except the workspace (bind), mounts a tmpfs over `/tmp` and `/dev/shm`, `cd`s into the workspace, then `exec unshare --map-root-user --mount bash -s`, which locks the mounts. Code arrives on stdin. Env is reduced to `PATH`, `HOME`, `LANG`, `LC_*`, `TERM`, `USER`, `TZ`, `TMPDIR`, `PYTHON*` (so `PATH` still carries the venv).
  - **Writes:** refused outside the workspace with `Read-only file system` (redirect, `mkdir`, `rm`, `mv`, `cp`, symlinks, `../`); remount attempts denied. **Reads:** unconfined (any file the user can read), credential stores masked by a deny-list only.
  - **Network:** a fresh net namespace with no interfaces: TCP to host loopback and external IPs fail; only `lo` exists. **But not Unix sockets, see next.**
  - **Surprise 1, host sockets:** connect-only probes from inside the sandbox succeeded to the Docker socket (user in the `docker` group: root-equivalent), the session and system D-Bus, `systemd --user` private socket, gpg-agent and PipeWire; `systemd-resolved` answered DNS (so DNS queries can carry data out). A read-only mount does not prevent `connect()` to a socket. Nothing was sent to those services. Workaround: `mount -t tmpfs -o ro,nosuid,nodev,noexec none /run` before every command (the prologue exits 125 if it fails; the pre-flight turns that into `ShellSandboxError`). Verified closed: all of the above plus DNS; mutation-checked. Still reachable: sockets under `$HOME`, `/etc/pacman.d/gnupg`, and the host `/tmp` when the workspace is under `/tmp` (effGen skips the private `/tmp` in that case, `_confine_script(private_tmp=False)`). See plan §4 T4.2, R1.
  - **Surprise 2, timeout:** on its deadline (`config.timeout + 2`) `run()` does `proc.kill()` on the outer `unshare` only; with `unshare --fork` the real command is a child that keeps running, and `await proc.wait()` completes only when every pipe is closed, so any surviving descendant makes `run()` block until it exits (reproduced: `sleep` children alive and the call still stuck when my watchdogs killed it at 14 s and at 100 s). The `ulimit -t` CPU limit does not help for sleeping commands. Workaround: the script runs `timeout -k 2 N bash -c <cmd> </dev/null`; `bash -s` is PID 1 of the namespace, so when `timeout` returns and the script exits the kernel kills every remaining process (verified for `&`, `setsid`, `nohup`, `trap '' TERM`; also for background processes of a command that exits normally). `config.timeout` is set to N+5 so effGen's own deadline is a later backstop.
  - **Memory:** `ulimit -v <kb>` (address space) in the prologue: `MemoryError` for a 512 MiB allocation at 256m, success at 2g. `-t` (CPU seconds) and `-u 256` (processes, **per user**, not per sandbox) are set too.
  - **Silent degradation:** `_build_cmd` returns an unconfined `bash -c "ulimit...; exec bash -s"` plan when `platform.system() != "Linux"` or user namespaces are missing, with only a log warning; and when the write-confinement probe fails it keeps net/PID isolation but sets `filesystem_confined=False`. Because the result arrives after the command ran, `ShellTool` first runs a fixed no-op and requires all guarantees to be reported, then re-checks every result.
  - **Parameters:** `BaseTool.validate_parameters` strips unknown parameters with a log warning (so `network_enabled=True` would be ignored silently); `ShellTool` overrides it to reject them. `ParameterSpec` rejects bools for integers and enforces ranges and lengths. `BaseTool.execute` maps `{"success": False}` to a failed `ToolResult`, and **the agent loop shows a model only `ToolResult.error` for a failure (not `output`)**, so the failing command's output is embedded in `error` (`agent_tool_execution.py` around line 299).
- Not verified: Windows and WSL2; hosts without user namespaces (simulated via effGen's class-level probe flags); Docker and Firecracker backends; cgroup memory limits (none exist here).
- Limitations: see plan §4 T4.2 ("Not enforced"). Reads and non-`/run` Unix sockets are the material ones.
- **Security scope (user decision, 2026-10-17):** effGen's `SubprocessSandbox` plus our `/run` mask is accepted **only for trusted local development and controlled test repositories**. It gives **no host read isolation** and leaves **Unix sockets outside `/run` reachable (R1, unresolved)**; DNS is blocked only by the interface-less network namespace plus the hidden `/run`; **Windows and WSL2 are unverified**. It must **not** run untrusted repositories or adversarial agent workloads: those need a dedicated, independently isolated container or VM with separately verified controls. No seccomp filter or new backend will be added in the MVP. The write-confinement, network, timeout, cleanup and memory results above are verified functional behaviour, not a claim of a complete security boundary.
- Decisions: use `SubprocessSandbox` directly instead of subclassing `CodeExecutor` (no private attributes, no model-facing network/memory/language parameters); never `auto` and never `off`; prove enforcement before the first model command; no effGen patch (needs approval): the two defects are worked around in `tools.py` and pinned by tests that fail if effGen changes.

### 2026-10-17 – Review follow-up: loop guards vs side-effecting tools (investigation only, no integration change)
- Files: `tests/test_loop_guards.py` (12 tests). Full analysis and options: `docs/reviews/opus-review-001.md`, Follow-up 001-A.
- **Verified at runtime (effGen 1.3.0, `NativeToolLoop` in `core/agent_tool_loop.py`, called from `core/agent_loop.py`):**
  - **Replay:** `check_action` keys on `(tool, normalized model input)`; an exact repeat of an earlier *successful* call is answered from `results_by_pair` without running the tool (`agent_loop.py:3001-3030`, up to `MAX_REPLAYS_PER_PAIR=2`); an exact repeat of an earlier *failed* call (result starting `Error executing tool`) is not recorded, so it takes the `is_loop` decline path. No config, tool-metadata or middleware switch exists (middleware runs after the decision).
  - **Result repeat:** `_result_key = (tool, normalized first 500 chars of the result)`, arguments not included. The second call with the same result text is executed, then tools are withdrawn (`ask_for_answer`) and the next call ends the run (`termination="stuck"`, `stop_reason="repeated_tool_result"`). Disabling it alone is not enough: other result-text heuristics (stall/drift; not traced) still ended the run after 3 of 7 calls. Distinct result text per call completed 7 of 7.
  - The guards object is constructed at one site (`agent_loop.py:3387`) from the module-level name `NativeToolLoop`.
- Implication: every tool of this harness is non-idempotent, so effGen's idempotence assumption is wrong for `shell`, `file_operations` writes and `delegate`. Recommended mitigation is pending approval (A: self-describing results; C: scoped guard subclass); until then multi-step edit/test loops can show stale or missing results.

### 2026-10-17 – F2 remediation and the F1 adapter blocker (private-API dependencies recorded)
- Added: `WorkspaceFileOperations` (subclass of effGen's `FileOperations` overriding `_execute` for `write`), `_echo` / `_report` in `ShellTool`. Tests: `tests/test_tools.py` (+11), `tests/test_loop_guards.py` (14).
- effGen APIs used: `FileOperations._execute(**kwargs) -> dict` (the documented override point; result dict `{success, data, message}` is what the agent loop renders, `agent_tool_execution.py` around line 299), `FileOperations._allowed_directories` (private, already read since T4.1).
- **Verified:** after the change, silent shell commands and file writes no longer collide in the result-text guards (7 distinct silent commands and 7 distinct writes completed); failure semantics unchanged (`ToolResult.success=False`, `error` carries the report); `is_error_result` does not misread a success or a failure because of the `$ ` prefix.
- **F1 adapter (was not implemented at this point; implemented later the same day, see "F1 compatibility patch" below): private-API dependencies it would add, and upgrade risk.**
  - `effgen.core.agent_loop.NativeToolLoop` (module attribute rebound; construction site `agent_loop.py:3387`, import `:113`), `NativeToolLoop.check_action` (`agent_tool_loop.py:529`), `LoopCheck.is_exact_loop` (`:148-166`, a plain dataclass), `NativeToolLoop.tools`.
  - No per-instance seam exists (searched all of effGen: one construction site, no factory, no config or tool-metadata option, middleware runs after the decision). Rebinding is process-global and visible to every thread; install/restore around runs is not re-entrant without a lock and a reference count; a permanent, idempotent install of a marker-keyed (inert for unmarked tools) subclass avoids both. A throwaway prototype of that design passed all 14 loop tests.
  - Upgrade risk: effGen is pinned `==1.3.0`; a new version can rename or move any of these names. The adapter's install should raise on a missing name and `tests/test_loop_guards.py` (pins and reproductions) fails loudly.
- Decisions: no per-call counter in tool text (would blind result-repeat, stall and drift guards); content echo and hash instead. Adapter approved and implemented afterwards (below).

### 2026-10-17 – F1 compatibility patch (process-wide `NativeToolLoop` replacement)
- Added: `install_effgen_loop_compat`, `effgen_loop_compat_installed`, `uninstall_effgen_loop_compat_for_tests`, `EffgenCompatError` in `src/smol_harness/harness.py`; `STATEFUL_TOOL_MARKER = "executes_every_call"` in `delegation.py`, set on `DelegateTool`, `ShellTool` and `WorkspaceFileOperations`; a guard in `Harness.run`; `tests/conftest.py` fixtures; `tests/test_loop_guards.py` (35 tests).
- **What it changes in effGen (process-wide, explicit, never undone by the harness):** `effgen.core.agent_loop.NativeToolLoop` is rebound to `MarkerKeyedToolLoop(effgen.core.agent_tool_loop.NativeToolLoop)`, whose `check_action` returns the inherited `LoopCheck` with `is_exact_loop` cleared (`dataclasses.replace`) when `getattr(self.tools.get(action), "executes_every_call", False)` is true. Consequently an exact repeat of a marked tool's call is no longer answered from `results_by_pair` (stale replay, `agent_loop.py:3001-3030`) and no longer declined (`:3033-3077`); it is dispatched. `effgen.core.agent_tool_loop.NativeToolLoop` itself is never modified.
- **Private effGen API relied on (all verified at install time; any mismatch raises `EffgenCompatError` and patches nothing):** `effgen.__version__ == "1.3.0"`; module attribute `effgen.core.agent_loop.NativeToolLoop` (construction site `agent_loop.drive`, ~line 3387, found by `"NativeToolLoop(" in inspect.getsource(drive)`); class `effgen.core.agent_tool_loop.NativeToolLoop` with `check_action(self, action, action_input)` (exact parameter names); dataclass `LoopCheck` with field `is_exact_loop`; the guard's `tools` mapping of tool instances. Also refuses to wrap a replacement made by anything else.
- **Scope verified:** unmarked tools keep effGen's replay (pin test); every other guard is inherited. Measured bounds with the patch are in review Follow-up 001-C §3 (same failing shell command: 3 executions then `loop_detected` by the stall guard; same read: 2 executions then `repeated_tool_result`; identical failing `delegate`: bounded by `max_delegations` and the iteration cap, since each report is unique by `delegation_id`).
- **Installation requirements:** call once, before any agent runs, from application start-up (future CLI) or a test fixture. `Harness.run` raises `EffgenCompatError` (before any model call) if it is missing. Do not reload `smol_harness.harness` after installing (the module's bookkeeping would be reset while effGen still holds the patched class). Installing or restoring during a run, or from several threads, is unsupported (the lock covers only the two bookkeeping variables).
- **Upgrade risk:** low while pinned to `effgen==1.3.0` (the patch refuses any other version). On an upgrade: re-read `agent_loop.drive` and `NativeToolLoop.check_action`/`cached_result`, run `tests/test_loop_guards.py` (the pins and F1 tests fail first), then update `SUPPORTED_EFFGEN_VERSION`.
- Limitations: the patch removes effGen's protection against an identical repeat of a marked call; the remaining bounds are the repeated-result/stall guards (not applicable to `delegate`, whose reports are unique), `max_delegations`, `max_iterations` and the worker deadline.
- Decisions: option P (install once, explicit, marker-keyed) over an install/restore-per-run (race-prone) or a fork; installation kept out of `Harness.__init__`/`run`/`close`; fail loudly instead of falling back to stale replay.

### 2026-10-17 – `RunStoppedError` and `raise_on_error` (verified; review F3)
- Files: `src/smol_harness/harness.py` (`Harness.run`), `tests/test_orchestrator_failure.py` (28 tests). No effGen change.
- **Verified in source and at runtime (effGen 1.3.0):**
  - `Agent.run` raises `RunStoppedError` (`errors.py:373`) only when `raise_on_error=True` **and** the response's `outcome == "stopped"`; the sole construction site is `_reconstruct_error` in `core/agent_generation.py`. The error carries the full `AgentResponse` (`success=False`, `termination`, `stop_reason`, `output`, `partial`) and copies `output`, `stop_reason`, `partial` from it. It is the same response `raise_on_error=False` returns: identical `success`, `termination` (`"stuck"`), `stop_reason` (`max_iterations_partial`, `loop_detected` measured), `output`, in four scenarios.
  - Every other failure under `True` is a **different exception with no response**: plain `RuntimeError` (guardrail block, some iteration/model failures, e.g. `'RuntimeError: model exploded'`) or typed provider errors (`BackendUnreachableError`, `ModelAuthError`, … via `_reconstruct_error`). With `False` the same failures are failed responses (`termination="error"`, `stop_reason="generation_failed"` for a raising model). The two modes are therefore **not equivalent** for non-stopped failures.
- Harness behaviour: stopped runs are normalised to `HarnessResult` using the (validated) carried response; other exceptions propagate with `exc.worker_records`. No response is ever fabricated.
- Related, `DelegateTool` (not changed): its `except RunStoppedError` branch unwraps `exc.response` without validation; a response-less error leaves no `WorkerRecord` (review F13).

### 2026-10-17 – T5.0 configuration-driven assembly: agent settings, context length, deadline visibility (verified)
- Files: `src/smol_harness/harness.py` (builder), `delegation.py` (`DeadlineMiddleware.remaining`, `_active_deadline` ContextVar), `tools.py` (shell bound); `tests/test_harness_config.py` (46 tests). No effGen change.
- **effGen `AgentConfig` defaults relevant to a harness (listed from the dataclass, effGen 1.3.0) and what the builder sets:** `raise_on_error=True` -> `False`; `enable_memory=True` -> `False`; `capability_probe=True` -> `False` (a probe generation); `enable_sub_agents=True` -> `False`; `tool_calling_mode="auto"` -> builder parameter (`"react"`); `max_iterations=10` -> the spec's value; `system_prompt="You are a helpful AI assistant."` -> the spec's when configured. Left at effGen defaults because the config has no field (review F7): `temperature=0.7`, `seed=None`, `max_tokens=None`, `top_p=0.9`, `top_k=50`, penalties, `max_turns_without_progress=2`, `recover_lost_tool_calls=True`, `context_budget="auto"`.
- **`Agent.__init__` rewrites a *default* system prompt in place:** an `AgentConfig` that kept the default string ends up with "You are <name>, a helpful AI assistant. ..." plus tool tips; a configured prompt is kept verbatim (`config.system_prompt` read after construction). Tests therefore record the prompt the harness passed, not the one read back.
- **Context length:** `load_model(..., provider="openai_compatible", context_length=N)` makes the adapter report `N` as its window (`get_context_length() == N`), so `resolve_context_budget` already yields the same budget (8192 -> prompt budget 5222) with `max_context_length=None` and `=8192`; `=4096` halves it (2611). `AgentConfig.max_context_length` is therefore redundant for Ollama but explicit, and **agent-side only**: it cannot change the Ollama server's `num_ctx`.
- **ContextVar visibility (F8):** a `contextvars.ContextVar` set inside the function that `DelegateTool` runs in the executor thread (`partial(ctx.run, ...)`) is visible inside effGen's tool execution for that run, and does not leak to the caller's context; `before_tool_call` middleware could also rewrite the tool input (`agent_tool_execution.py:_execute_tool`), which was deliberately not used (it would edit model-generated arguments).
- **Deadline behaviour measured:** worker deadline 2 s + shell `sleep 6` with requested timeout 30: 6.4 s before the bound, 2.4 s after. A single model generation in flight is never interrupted, and a worker whose last call overran is reported `completed` (no later check runs).
- Closing a worker/orchestrator `Agent` does not unload its model (audit §1.4), so `ModelCache` is the only owner and unloads each distinct model once (tested with a model shared by two workers).

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
