"""T5.0: a validated `HarnessConfig` becomes a runnable, closable `Harness` (and review F5, F8).

Models are scripted and `smol_harness.models.build_model` is the only thing replaced; tools are the real ones, the agents
are effGen's real `Agent`s running the real loop. Configuration is checked by what the constructed agents actually
received (a recording `Agent` subclass), not by peeking at the harness internals.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from conftest import ScriptedModel
from effgen import Agent

import smol_harness.delegation as delegation_mod
import smol_harness.harness as harness_mod
import smol_harness.models as models_mod
from smol_harness.config import HarnessConfig, loads_config
from smol_harness.delegation import DeadlineMiddleware, _active_deadline, remaining_worker_seconds
from smol_harness.harness import (
    EffgenCompatError,
    Harness,
    HarnessClosedError,
    build_harness_from_config,
    effgen_loop_compat_installed,
    initialize_runtime,
)
from smol_harness.models import ModelCache, ModelLoadError
from smol_harness.tools import ShellTool, ToolBuildError

linux_only = pytest.mark.skipif(platform.system() != "Linux", reason="the shell sandbox is Linux-only")
BIG, SMALL, TINY = "big:1", "small:1", "org/tiny"
FINAL = "Thought: done.\nFinal Answer: all done"


def call(tool_name: str, /, **args) -> str:
    return f"Thought: next.\nAction: {tool_name}\nAction Input: {json.dumps(args)}"


def delegate(worker: str, task: str) -> str:
    return call("delegate", worker=worker, task=task)


# ---- fakes: only model loading is replaced -----------------------------------------------------------------------------------
class TrackedModel(ScriptedModel):
    """A scripted model that counts `unload()` and can be told to fail it or to be slow."""

    def __init__(self, turns, *, name, delay: float = 0.0, unload_error: str | None = None):
        super().__init__(turns, name=name)
        self.unloads, self.delay, self.unload_error = 0, delay, unload_error

    def unload(self) -> None:
        self.unloads += 1
        if self.unload_error:
            raise RuntimeError(self.unload_error)

    def generate(self, prompt, config=None, **kw):
        if self.delay:
            time.sleep(self.delay)
        return super().generate(prompt, config, **kw)


class FakeModels:
    def __init__(self) -> None:
        self.scripts: dict[str, list[str]] = {}
        self.loads: list = []  # every ModelSpec build_model was asked for, in order
        self.instances: dict[str, TrackedModel] = {}
        self.failing: set[str] = set()
        self.delay: dict[str, float] = {}
        self.unload_error: dict[str, str] = {}

    def build(self, spec):
        self.loads.append(spec)
        if spec.model in self.failing:
            raise ModelLoadError(f"cannot load {spec.model}")
        model = TrackedModel(self.scripts.get(spec.model, [FINAL]), name=spec.model, delay=self.delay.get(spec.model, 0.0),
                             unload_error=self.unload_error.get(spec.model))
        self.instances[spec.model] = model
        return model


@pytest.fixture
def models(monkeypatch):
    fake = FakeModels()
    monkeypatch.setattr(models_mod, "build_model", fake.build)  # ModelCache.get looks the name up at call time
    return fake


@pytest.fixture
def agents(monkeypatch):
    """Every effGen Agent the harness constructs, with the config it was given and the tools it ended up holding."""
    seen: list[SimpleNamespace] = []

    class RecordingAgent(Agent):
        def __init__(self, config, *a, **kw):
            passed_prompt = config.system_prompt  # Agent.__init__ rewrites a DEFAULT prompt in place; record what we passed
            super().__init__(config, *a, **kw)
            seen.append(SimpleNamespace(name=config.name, config=config, tools=dict(self.tools), passed_prompt=passed_prompt))

    monkeypatch.setattr(harness_mod, "Agent", RecordingAgent)
    monkeypatch.setattr(delegation_mod, "Agent", RecordingAgent)
    return seen


# ---- configuration under test --------------------------------------------------------------------------------------------------
def make_config(tmp_path: Path, **over) -> HarnessConfig:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    data = {
        "workspace": str(ws),
        "limits": {"max_delegations": 4, "max_report_chars": 4000, "worker_timeout_s": 60},
        "orchestrator": {"model": {"provider": "ollama", "model": BIG, "context_length": 8192}, "tools": ["file_operations"],
                         "max_iterations": 7, "system_prompt": "You coordinate."},
        "workers": [
            {"name": "reader", "description": "reads files", "max_iterations": 3, "system_prompt": "You read.",
             "model": {"provider": "ollama", "model": SMALL, "context_length": 4096}, "tools": ["file_operations"]},
            {"name": "writer", "description": "writes files", "max_iterations": 5,  # same model spec as the reader
             "model": {"provider": "ollama", "model": SMALL, "context_length": 4096}, "tools": ["file_operations"]},
            {"name": "thinker", "description": "only thinks", "max_iterations": 2,
             "model": {"provider": "hf", "model": TINY, "quantization": "none", "device": "cpu", "dtype": "fp32"}, "tools": []},
        ],
    }
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return loads_config(yaml.safe_dump(data), tmp_path)


def agent(agents, name):
    (found,) = [a for a in agents if a.name == name][-1:]
    return found


FULL_FLOW = {
    BIG: [delegate("reader", "READ-TASK-MARKER list the workspace"), delegate("writer", "WRITE-TASK-MARKER create hello.txt"), FINAL],
    SMALL: [call("file_operations", operation="list", path="."), "Thought: ok.\nFinal Answer: read report",
            call("file_operations", operation="write", path="hello.txt", content="hi there\n"), "Thought: ok.\nFinal Answer: wrote hello.txt"],
}


# ===============================================================================================================================
# Configuration -> agents (1 to 5, 9 to 12)
# ===============================================================================================================================
def test_the_orchestrator_and_each_worker_are_configured_from_their_own_specs(tmp_path, models, agents):
    models.scripts.update(FULL_FLOW)
    with build_harness_from_config(make_config(tmp_path)) as harness:
        result = harness.run("coordinate")
    assert result.success, result.output
    orch, reader, writer = agent(agents, "orchestrator"), agent(agents, "reader"), agent(agents, "writer")
    assert (orch.config.max_iterations, reader.config.max_iterations, writer.config.max_iterations) == (7, 3, 5)
    assert orch.passed_prompt == "You coordinate." and reader.passed_prompt == "You read."
    assert orch.config.system_prompt == "You coordinate."  # a configured prompt survives Agent.__init__ verbatim
    # no prompt configured for the writer: the harness passes none, so effGen's own default is what it received
    # (effGen then expands that default into "You are writer, a helpful AI assistant. ..." itself)
    assert writer.passed_prompt == "You are a helpful AI assistant."
    assert [r.worker for r in result.workers] == ["reader", "writer"] and all(r.status == "completed" for r in result.workers)


def test_each_agent_holds_exactly_its_allowlist_and_only_the_orchestrator_can_delegate(tmp_path, models, agents):
    models.scripts.update(FULL_FLOW)
    config = make_config(tmp_path)
    with build_harness_from_config(config) as harness:
        harness.run("coordinate")
    assert sorted(agent(agents, "orchestrator").tools) == ["delegate", "file_operations"]
    assert sorted(agent(agents, "reader").tools) == ["file_operations"] and sorted(agent(agents, "writer").tools) == ["file_operations"]
    assert not any("delegate" in a.tools for a in agents if a.name != "orchestrator")
    assert all(a.config.enable_sub_agents is False for a in agents)  # no effGen sub-agents either
    web_like = ("web", "search_engine", "url_fetch", "http", "browser")
    assert not any(any(w in tool for w in web_like) for a in agents for tool in a.tools)


def test_a_thinker_with_an_empty_allowlist_holds_no_tools_and_cannot_delegate(tmp_path, models, agents):
    models.scripts[BIG] = [delegate("thinker", "just think"), FINAL]
    models.scripts[TINY] = ["Thought: ok.\nFinal Answer: thought about it"]
    with build_harness_from_config(make_config(tmp_path)) as harness:
        result = harness.run("coordinate")
    assert result.success and agent(agents, "thinker").tools == {}
    assert result.workers[0].result.output == "thought about it"


def test_a_worker_cannot_reach_delegate_even_by_asking_for_it(tmp_path, models, agents):
    models.scripts[BIG] = [delegate("reader", "READ-TASK-MARKER"), FINAL]
    models.scripts[SMALL] = [call("delegate", worker="writer", task="recurse"), call("file_operations", operation="list", path="."),
                             "Thought: ok.\nFinal Answer: could not delegate"]
    with build_harness_from_config(make_config(tmp_path)) as harness:
        result = harness.run("coordinate")
    assert [r.worker for r in result.workers] == ["reader"]  # the writer never ran
    assert not any(a.name == "writer" for a in agents)


def test_tool_instances_are_not_shared_between_agents(tmp_path, models, agents):
    models.scripts.update(FULL_FLOW)
    with build_harness_from_config(make_config(tmp_path)) as harness:
        harness.run("coordinate")
    tools = [a.tools["file_operations"] for a in agents if "file_operations" in a.tools]
    assert len({id(t) for t in tools}) == 3  # orchestrator, reader, writer: one instance each


def test_tool_calling_mode_defaults_to_react_and_is_applied_to_every_agent(tmp_path, models, agents):
    models.scripts.update(FULL_FLOW)
    with build_harness_from_config(make_config(tmp_path)) as harness:
        harness.run("coordinate")
    assert {a.config.tool_calling_mode for a in agents} == {"react"}


def test_an_explicit_tool_calling_mode_reaches_every_agent(tmp_path, models, agents):
    models.scripts[BIG] = [delegate("thinker", "think"), FINAL]
    with build_harness_from_config(make_config(tmp_path), tool_calling_mode="hybrid") as harness:
        harness.run("coordinate")
    assert {a.config.tool_calling_mode for a in agents} == {"hybrid"} and len(agents) >= 2


def test_an_unknown_tool_calling_mode_is_rejected_before_anything_is_loaded(tmp_path, models):
    with pytest.raises(ValueError, match="tool_calling_mode"):
        build_harness_from_config(make_config(tmp_path), tool_calling_mode="telepathy")
    assert models.loads == []


def test_the_effgen_settings_that_matter_are_stated_not_defaulted(tmp_path, models, agents):
    models.scripts.update(FULL_FLOW)
    with build_harness_from_config(make_config(tmp_path)) as harness:
        harness.run("coordinate")
    for a in agents:
        assert a.config.raise_on_error is False  # effGen's default is True; failed runs become structured results
        assert a.config.enable_memory is False and a.config.capability_probe is False  # no shared state, no probe generation
        assert a.config.tool_calling_mode == "react" and a.config.enable_sub_agents is False
        assert isinstance(a.config.model, ScriptedModel)  # an instance: effGen never loads a model by itself


def test_ollama_context_length_becomes_the_agent_side_limit_and_hf_gets_none(tmp_path, models, agents):
    # AGENT-side prompt budget only. It does not, and cannot, set the Ollama server's own context window (review F4).
    models.scripts[BIG] = [delegate("reader", "READ-TASK-MARKER"), delegate("thinker", "think"), FINAL]
    models.scripts[SMALL] = [call("file_operations", operation="list", path="."), "Thought: ok.\nFinal Answer: r"]
    models.scripts[TINY] = ["Thought: ok.\nFinal Answer: t"]
    with build_harness_from_config(make_config(tmp_path)) as harness:
        harness.run("coordinate")
    assert agent(agents, "orchestrator").config.max_context_length == 8192
    assert agent(agents, "reader").config.max_context_length == 4096
    assert agent(agents, "thinker").config.max_context_length is None  # no hf field yet
    assert models.instances[BIG].get_context_length() == 8192  # the adapter-side value comes from the same spec (models.py)


# ===============================================================================================================================
# Models: one load per distinct spec, sharing, ownership, cleanup (6 to 8, 14 to 17)
# ===============================================================================================================================
def test_one_load_per_distinct_spec_and_equal_specs_share_one_instance(tmp_path, models, agents):
    models.scripts.update(FULL_FLOW)
    with build_harness_from_config(make_config(tmp_path)) as harness:
        harness.run("coordinate")
    assert sorted(spec.model for spec in models.loads) == sorted([BIG, SMALL, TINY])  # three specs, reader+writer share SMALL
    assert agent(agents, "reader").config.model is agent(agents, "writer").config.model is models.instances[SMALL]
    assert agent(agents, "orchestrator").config.model is models.instances[BIG] is not models.instances[SMALL]
    assert models.instances[BIG] is not models.instances[TINY]


def test_workers_that_share_a_model_still_have_independent_contexts(tmp_path, models):
    models.scripts.update(FULL_FLOW)
    with build_harness_from_config(make_config(tmp_path)) as harness:
        harness.run("coordinate")
    shared = models.instances[SMALL]
    first_prompts = [c.prompt for c in shared.calls if "READ-TASK-MARKER" in str(c.prompt)]
    second_prompts = [c.prompt for c in shared.calls if "WRITE-TASK-MARKER" in str(c.prompt)]
    assert first_prompts and second_prompts
    assert not any("WRITE-TASK-MARKER" in str(p) for p in first_prompts)  # the reader never saw the writer's task
    assert not any("READ-TASK-MARKER" in str(p) for p in second_prompts)  # nor the writer the reader's: fresh Agent each time


def test_close_unloads_each_owned_model_exactly_once_even_when_shared(tmp_path, models):
    harness = build_harness_from_config(make_config(tmp_path))
    assert {m.unloads for m in models.instances.values()} == {0}
    harness.close()
    assert {name: m.unloads for name, m in models.instances.items()} == {BIG: 1, SMALL: 1, TINY: 1}  # SMALL serves two workers


def test_repeated_close_is_safe_and_unloads_nothing_twice(tmp_path, models):
    harness = build_harness_from_config(make_config(tmp_path))
    harness.close()
    harness.close()
    with harness:  # entering and leaving a closed harness is also harmless
        pass
    assert {m.unloads for m in models.instances.values()} == {1}


def test_run_after_close_is_rejected_with_an_actionable_error_and_runs_nothing(tmp_path, models):
    harness = build_harness_from_config(make_config(tmp_path))
    harness.close()
    with pytest.raises(HarnessClosedError, match="closed"):
        harness.run("too late")
    assert models.instances[BIG].calls == []


def test_the_context_manager_closes_on_normal_exit_and_when_the_block_raises(tmp_path, models):
    with build_harness_from_config(make_config(tmp_path)):
        pass
    assert {m.unloads for m in models.instances.values()} == {1}
    models.instances.clear()
    with pytest.raises(KeyError, match="from the block"):
        with build_harness_from_config(make_config(tmp_path)):
            raise KeyError("from the block")
    assert {m.unloads for m in models.instances.values()} == {1}


def test_a_failing_unload_does_not_hide_the_blocks_own_exception(tmp_path, models):
    models.unload_error[BIG] = "unload exploded"
    with pytest.raises(KeyError, match="from the block") as err:
        with build_harness_from_config(make_config(tmp_path)):
            raise KeyError("from the block")
    assert any("closing the harness also failed" in n and "unload exploded" in n for n in err.value.__notes__)
    assert models.instances[SMALL].unloads == 1  # the other models were still released


def test_a_failing_unload_surfaces_from_close_once_and_the_harness_stays_closed(tmp_path, models):
    models.unload_error[BIG] = "unload exploded"
    harness = build_harness_from_config(make_config(tmp_path))
    with pytest.raises(ExceptionGroup):
        harness.close()
    harness.close()  # closed already: no retry, no second error
    assert models.instances[BIG].unloads == 1 and models.instances[SMALL].unloads == 1
    with pytest.raises(HarnessClosedError):
        harness.run("x")


def test_a_model_that_fails_to_load_releases_the_models_loaded_before_it(tmp_path, models):
    models.failing.add(TINY)  # the last distinct spec
    with pytest.raises(ModelLoadError, match=TINY):
        build_harness_from_config(make_config(tmp_path))
    assert [s.model for s in models.loads] == [BIG, SMALL, TINY]  # the shared SMALL was loaded once
    assert models.instances[BIG].unloads == 1 and models.instances[SMALL].unloads == 1


def test_the_original_load_error_stays_visible_when_cleanup_also_fails(tmp_path, models):
    models.failing.add(TINY)
    models.unload_error[BIG] = "unload exploded"
    with pytest.raises(ModelLoadError, match=TINY) as err:
        build_harness_from_config(make_config(tmp_path))
    assert any("releasing the models loaded so far also failed" in n for n in err.value.__notes__)
    assert models.instances[SMALL].unloads == 1  # cleanup still reached every model


def test_an_injected_cache_stays_the_callers_after_close(tmp_path, models):
    cache = ModelCache()
    harness = build_harness_from_config(make_config(tmp_path), cache=cache)
    assert len(cache) == 3
    harness.close()
    assert {m.unloads for m in models.instances.values()} == {0} and len(cache) == 3  # untouched, still usable
    assert cache.get(next(iter(cache._models))) in models.instances.values()  # (a cache hit: nothing was reloaded)
    cache.close()  # the caller closes it
    assert {m.unloads for m in models.instances.values()} == {1}


def test_an_injected_cache_is_reused_across_harnesses_and_not_reloaded(tmp_path, models):
    cache = ModelCache()
    first = build_harness_from_config(make_config(tmp_path), cache=cache)
    second = build_harness_from_config(make_config(tmp_path), cache=cache)
    first.close()
    second.close()
    assert len(models.loads) == 3 and len(cache) == 3  # one load per distinct spec across both harnesses
    cache.close()


def test_a_failed_assembly_does_not_close_an_injected_cache(tmp_path, models):
    models.failing.add(TINY)
    cache = ModelCache()
    with pytest.raises(ModelLoadError):
        build_harness_from_config(make_config(tmp_path), cache=cache)
    assert {m.unloads for m in models.instances.values()} == {0}  # the caller decides what to do with what was loaded
    cache.close()
    assert {m.unloads for m in models.instances.values()} == {1}


def test_an_already_closed_injected_cache_fails_assembly_clearly(tmp_path, models):
    cache = ModelCache()
    cache.close()
    with pytest.raises(RuntimeError, match="closed"):
        build_harness_from_config(make_config(tmp_path), cache=cache)
    assert models.loads == []


# ===============================================================================================================================
# Failures before any model is loaded (13) and the compatibility requirement
# ===============================================================================================================================
def test_a_missing_workspace_fails_before_any_model_is_loaded(tmp_path, models):
    config = make_config(tmp_path)
    config = dataclasses.replace(config, workspace=tmp_path / "does-not-exist")
    with pytest.raises(ToolBuildError, match="does not exist"):
        build_harness_from_config(config)
    assert models.loads == []


def test_a_dangerous_workspace_root_fails_before_any_model_is_loaded(tmp_path, models):
    with pytest.raises(ToolBuildError, match="filesystem root"):
        build_harness_from_config(dataclasses.replace(make_config(tmp_path), workspace=Path("/")))
    assert models.loads == []


def test_an_unavailable_shell_fails_before_any_model_is_loaded(tmp_path, models, monkeypatch):
    config = make_config(tmp_path)
    shelled = dataclasses.replace(config.workers[1], tools=("file_operations", "shell"))
    config = dataclasses.replace(config, workers=(config.workers[0], shelled, config.workers[2]))
    # Any shell backend needs its host binary (bash on Linux, powershell.exe/pwsh on Windows); make
    # them all appear missing, whichever platform the test runs on.
    real = shutil.which
    monkeypatch.setattr(shutil, "which",
                        lambda name, *a, **k: None if name in ("bash", "unshare", "timeout", "powershell", "pwsh")
                        else real(name, *a, **k))
    with pytest.raises(ToolBuildError, match="tool 'shell' needs"):
        build_harness_from_config(config)
    assert models.loads == []


def test_a_tool_name_the_registry_does_not_have_is_rejected_not_ignored(tmp_path, models):
    config = make_config(tmp_path)
    bad = dataclasses.replace(config.workers[0], tools=("file_operations", "web_search"))  # bypasses config validation on purpose
    with pytest.raises(ToolBuildError, match="unknown tool 'web_search'"):
        build_harness_from_config(dataclasses.replace(config, workers=(bad, *config.workers[1:])))
    assert models.loads == []


def test_delegate_in_a_worker_allowlist_is_rejected_not_ignored(tmp_path, models):
    config = make_config(tmp_path)
    bad = dataclasses.replace(config.workers[0], tools=("delegate",))
    with pytest.raises(ToolBuildError, match="orchestrator only"):
        build_harness_from_config(dataclasses.replace(config, workers=(bad, *config.workers[1:])))
    assert models.loads == []


def test_building_without_the_compat_patch_is_refused_before_any_model_is_loaded(tmp_path, models, compat_uninstalled):
    with pytest.raises(EffgenCompatError, match=r"initialize_runtime\(\)"):
        build_harness_from_config(make_config(tmp_path))
    assert models.loads == []
    assert not effgen_loop_compat_installed()  # and building did not install it behind our back


def test_closing_a_harness_leaves_the_compat_patch_installed(tmp_path, models):
    assert effgen_loop_compat_installed()
    with build_harness_from_config(make_config(tmp_path)):
        pass
    assert effgen_loop_compat_installed()


def test_initialize_runtime_installs_idempotently(compat_uninstalled):
    assert not effgen_loop_compat_installed()
    initialize_runtime()
    patched = __import__("effgen.core.agent_loop", fromlist=["NativeToolLoop"]).NativeToolLoop
    initialize_runtime()
    assert effgen_loop_compat_installed()
    assert __import__("effgen.core.agent_loop", fromlist=["NativeToolLoop"]).NativeToolLoop is patched


# ===============================================================================================================================
# Limits (9) and whole flows (18, 19)
# ===============================================================================================================================
def test_the_delegation_limit_from_the_config_refuses_the_extra_delegation(tmp_path, models):
    models.scripts[BIG] = [delegate("thinker", "first"), delegate("thinker", "second"), FINAL]
    models.scripts[TINY] = ["Thought: ok.\nFinal Answer: thought"]
    with build_harness_from_config(make_config(tmp_path, limits={"max_delegations": 1})) as harness:
        result = harness.run("coordinate")
    assert [r.status for r in result.workers] == ["completed", "failed"]
    assert result.workers[1].result.error_type == "delegation_limit_reached"
    assert len(models.instances[TINY].calls) == 1  # the refused delegation never reached a worker


def test_the_worker_deadline_from_the_config_stops_a_slow_worker(tmp_path, models):
    models.scripts[BIG] = [delegate("reader", "READ-TASK-MARKER"), FINAL]
    models.scripts[SMALL] = [call("file_operations", operation="list", path="."), "Thought: ok.\nFinal Answer: r"]
    models.delay[SMALL] = 0.6  # each generation takes longer than the 0.3 s deadline: the check before the tool call trips
    with build_harness_from_config(make_config(tmp_path, limits={"worker_timeout_s": 0.3})) as harness:
        result = harness.run("coordinate")
    assert result.workers[0].status == "timeout" and result.workers[0].result.error_type == "WorkerTimeout"


def test_the_report_size_limit_from_the_config_truncates_a_long_report(tmp_path, models):
    models.scripts[BIG] = [delegate("thinker", "think"), FINAL]
    models.scripts[TINY] = ["Thought: ok.\nFinal Answer: " + "x" * 5000]
    with build_harness_from_config(make_config(tmp_path, limits={"max_report_chars": 400})) as harness:
        result = harness.run("coordinate")
    assert result.workers[0].result.truncated is True
    reported = str(models.instances[BIG].calls[1].prompt).rsplit("Observation:", 1)[-1]
    assert 'x' * 3000 not in reported and '"truncated": true' in reported  # what the orchestrator was shown


def test_a_complete_flow_orchestrator_worker_tool_final_result(tmp_path, models):
    models.scripts.update(FULL_FLOW)
    config = make_config(tmp_path)
    with build_harness_from_config(config) as harness:
        result = harness.run("coordinate")
        assert isinstance(harness, Harness)
    assert result.success is True and result.termination == "done" and result.output == "all done"
    assert (config.workspace / "hello.txt").read_text() == "hi there\n"  # the real tool really wrote it, inside the workspace
    assert [(r.worker, r.status, r.result.delegation_id) for r in result.workers] == [("reader", "completed", "d1"), ("writer", "completed", "d2")]
    assert "wrote hello.txt" in result.workers[1].result.output
    assert sorted(p.name for p in config.workspace.iterdir()) == ["hello.txt"]


def test_an_orchestrator_failure_keeps_the_worker_records(tmp_path, models):
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "a.txt").write_text("x\n")
    models.scripts[BIG] = [delegate("thinker", "think")] + [call("file_operations", operation="read", path="a.txt")] * 8
    models.scripts[TINY] = ["Thought: ok.\nFinal Answer: thought"]
    with build_harness_from_config(make_config(tmp_path)) as harness:
        result = harness.run("coordinate")
    assert result.success is False and result.termination == "stuck"
    assert [(r.worker, r.status) for r in result.workers] == [("thinker", "completed")]


def test_two_runs_on_one_harness_keep_their_records_apart(tmp_path, models):
    models.scripts[BIG] = [delegate("thinker", "one"), FINAL, delegate("thinker", "two"), FINAL]
    models.scripts[TINY] = ["Thought: ok.\nFinal Answer: thought"]
    with build_harness_from_config(make_config(tmp_path)) as harness:
        first, second = harness.run("a"), harness.run("b")
    assert [r.task for r in first.workers] == ["one"] and [r.task for r in second.workers] == ["two"]


# ===============================================================================================================================
# 20: the documented start-up sequence, in a fresh process with no pytest fixtures
# ===============================================================================================================================
FRESH_PROCESS = r"""
import json, sys, tempfile, pathlib
sys.path.insert(0, {tests!r})
from conftest import ScriptedModel        # the scripted model only; its session fixtures are pytest's and never run here
import yaml
import smol_harness.models as models_mod
from smol_harness import harness as h
from smol_harness.config import loads_config

loads, unloads = [], []
class Tracked(ScriptedModel):
    def unload(self): unloads.append(self.model_name)
def fake_build(spec):
    loads.append(spec.model)
    return Tracked({scripts!r}[spec.model], name=spec.model)
models_mod.build_model = fake_build

tmp = pathlib.Path(tempfile.mkdtemp())
ws = tmp / "ws"; ws.mkdir()
config = loads_config(yaml.safe_dump({{
    "workspace": str(ws),
    "orchestrator": {{"model": {{"provider": "ollama", "model": "big:1", "context_length": 8192}}, "tools": []}},
    "workers": [{{"name": "writer", "description": "writes", "tools": ["file_operations"],
                 "model": {{"provider": "ollama", "model": "small:1", "context_length": 4096}}}}],
}}), tmp)

out = {{"installed_at_start": h.effgen_loop_compat_installed()}}
try:
    h.build_harness_from_config(config); out["refused_before_init"] = False
except h.EffgenCompatError as exc:
    out["refused_before_init"] = "initialize_runtime()" in str(exc)
out["loads_before_init"] = len(loads)

h.initialize_runtime(); h.initialize_runtime()          # the documented sequence; idempotent
with h.build_harness_from_config(config) as harness:
    result = harness.run("go")
    out.update(success=result.success, output=result.output, records=[(r.worker, r.status) for r in result.workers],
               file=(ws / "hello.txt").read_text(), unloaded_inside_block=list(unloads))
out.update(installed_after_close=h.effgen_loop_compat_installed(), loads=sorted(loads), unloaded=sorted(unloads))
print("RESULT" + json.dumps(out))
"""


def test_the_startup_sequence_works_in_a_fresh_process_without_any_pytest_fixture(tmp_path):
    scripts = {BIG: [delegate("writer", "create hello.txt"), FINAL],
               SMALL: [call("file_operations", operation="write", path="hello.txt", content="from a fresh process\n"),
                       "Thought: ok.\nFinal Answer: wrote it"]}
    repo = Path(__file__).resolve().parents[1]
    code = FRESH_PROCESS.format(tests=str(repo / "tests"), scripts=scripts)
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=tmp_path,
                          env={k: v for k, v in os.environ.items() if k not in ("PYTEST_CURRENT_TEST",)})
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(next(line for line in done.stdout.splitlines() if line.startswith("RESULT"))[len("RESULT"):])
    assert out["installed_at_start"] is False  # nothing in a fresh interpreter installed it
    assert out["refused_before_init"] is True and out["loads_before_init"] == 0  # refused, before any model load
    assert (out["success"], out["output"]) == (True, "all done")
    assert out["records"] == [["writer", "completed"]] and out["file"] == "from a fresh process\n"
    assert out["unloaded_inside_block"] == [] and out["unloaded"] == [BIG, SMALL] and out["loads"] == [BIG, SMALL]
    assert out["installed_after_close"] is True  # closing never uninstalls


# ===============================================================================================================================
# 21 / review F8: a shell command cannot outlive its worker's deadline
# ===============================================================================================================================
def test_deadline_middleware_reports_its_remaining_time():
    now = [100.0]
    deadline = DeadlineMiddleware(10, clock=lambda: now[0])
    assert deadline.remaining() == pytest.approx(10)
    now[0] = 103.5
    assert deadline.remaining() == pytest.approx(6.5)
    now[0] = 111
    assert deadline.remaining() < 0


def test_there_is_no_remaining_time_outside_a_worker_run():
    assert remaining_worker_seconds() is None


@linux_only
def test_a_shell_command_is_bounded_by_the_time_the_worker_has_left(tmp_path):
    import asyncio

    (tmp_path / "ws").mkdir()
    shell = ShellTool(tmp_path / "ws")
    token = _active_deadline.set(DeadlineMiddleware(2))  # a worker with 2 s left
    try:
        start = time.monotonic()
        result = asyncio.run(shell.execute(command="sleep 20; echo never", timeout=60))  # the model asked for 60 s
        elapsed = time.monotonic() - start
    finally:
        _active_deadline.reset(token)
    assert elapsed < 6, elapsed
    assert not result.success and result.output["timed_out"] is True
    assert "timed out after 2s (all the time the worker had left)" in result.error
    assert "never" not in result.output["stdout"]


@linux_only
def test_a_command_that_fits_in_the_workers_time_is_not_limited(tmp_path):
    import asyncio

    (tmp_path / "ws").mkdir()
    shell = ShellTool(tmp_path / "ws")
    token = _active_deadline.set(DeadlineMiddleware(100))
    try:
        result = asyncio.run(shell.execute(command="echo fine", timeout=30))
    finally:
        _active_deadline.reset(token)
    assert result.success and result.output["timed_out"] is False and "worker had left" not in result.output["result"]


@linux_only
def test_the_orchestrator_is_not_bound_by_a_worker_deadline(tmp_path, models):
    # The orchestrator has no deadline (documented): its shell keeps the model's own timeout.
    models.scripts[BIG] = [call("shell", command="sleep 1; echo orchestrator-done", timeout=30), FINAL]
    config = make_config(tmp_path, orchestrator={"tools": ["shell"]}, limits={"worker_timeout_s": 0.2})
    with build_harness_from_config(config) as harness:
        result = harness.run("coordinate")
    assert result.success and "orchestrator-done" in str(models.instances[BIG].calls[1].prompt)


@linux_only
def test_a_worker_started_shell_command_is_killed_with_the_worker_deadline_end_to_end(tmp_path, models):
    config = make_config(tmp_path, limits={"worker_timeout_s": 2},
                         workers=[{"name": "runner", "description": "runs commands", "tools": ["shell"], "max_iterations": 4,
                                   "model": {"provider": "ollama", "model": SMALL, "context_length": 4096}}])
    models.scripts[BIG] = [delegate("runner", "run the long command"), FINAL]
    models.scripts[SMALL] = [call("shell", command="sleep 8; echo finished", timeout=30), "Thought: ok.\nFinal Answer: done"]
    start = time.monotonic()
    with build_harness_from_config(config) as harness:
        result = harness.run("coordinate")
    elapsed = time.monotonic() - start
    assert result.workers[0].status == "timeout"
    assert elapsed < 6, elapsed  # was 6.4 s for this scenario before the bound existed (the command ran to its end)
    assert remaining_worker_seconds() is None  # nothing leaked out of the worker's context


def test_a_generation_already_in_flight_is_neither_interrupted_nor_flagged_which_remains_a_documented_limit(tmp_path, models):
    # The deadline is cooperative: checked BEFORE a model or tool call (delegation.DeadlineMiddleware). A single slow
    # generation runs to its end, and when no further call follows, nothing ever checks the deadline again, so the
    # worker is reported as completed even though it overran. Hard cancellation is out of scope (no threads or kills).
    models.scripts[BIG] = [delegate("reader", "READ-TASK-MARKER"), FINAL]
    models.scripts[SMALL] = ["Thought: ok.\nFinal Answer: slow but finished"]
    models.delay[SMALL] = 1.0
    start = time.monotonic()
    with build_harness_from_config(make_config(tmp_path, limits={"worker_timeout_s": 0.2})) as harness:
        result = harness.run("coordinate")
    assert time.monotonic() - start >= 1.0  # the whole slow generation ran to its end
    assert result.workers[0].status == "completed" and result.workers[0].result.output == "slow but finished"
