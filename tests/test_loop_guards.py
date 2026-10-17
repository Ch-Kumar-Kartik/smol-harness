"""Review finding F1/F2 (docs/reviews/opus-review-001.md): effGen 1.3.0's repeat guards assume idempotent tools.

Reproductions through the REAL effGen agent loop with scripted models; no mocks of the guards.

* F1 (replay): an exact repeat of an earlier SUCCESSFUL call `(tool, normalized input)` is not executed;
  `NativeToolLoop.cached_result` hands back the recorded result (agent_loop.py:3001-3030, agent_tool_loop.py:561-593).
  An exact repeat of an earlier FAILED call is not executed either: it takes the `is_loop` decline path.
* F2 (result repeat): `NativeToolLoop.result_is_repeat` keys on `(tool, whitespace-normalized first 500 chars of
  the RESULT)` and ignores the arguments (agent_tool_loop.py:643-664). The second call that returns the same text
  is executed, then the run is asked for its answer with tools withdrawn (agent_loop.py:3229-3260).
  FIXED in our tools (phase A): shell reports echo the command, file-write reports carry path, size and hash.

F1 is FIXED by `harness.install_effgen_loop_compat` (a process-wide, explicit, marker-keyed patch; the session
fixture in conftest.py installs it like an application would). The F1 tests below were strict xfails before it and
are now ordinary tests. The tests after them check the patch itself: how it installs, how it fails, what it leaves
alone, and that the loop protections we keep still bound a runaway agent.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import effgen
import pytest
from conftest import ScriptedModel
from effgen import Agent, AgentConfig, tool
from effgen.core import agent_loop, agent_tool_loop
from effgen.core.agent_tool_loop import NativeToolLoop

from smol_harness import harness as harness_mod
from smol_harness.delegation import STATEFUL_TOOL_MARKER, DelegateTool, WorkerRegistry
from smol_harness.harness import (
    EffgenCompatError,
    build_harness,
    effgen_loop_compat_installed,
    install_effgen_loop_compat,
    uninstall_effgen_loop_compat_for_tests,
)
from smol_harness.tools import TOOL_NAMES, ShellTool, WorkspaceFileOperations, build_tools

linux_only = pytest.mark.skipif(platform.system() != "Linux", reason="the shell sandbox is Linux-only")

FINAL = "Thought: done.\nFinal Answer: finished"


def call(tool_name: str, /, **args) -> str:
    return f"Thought: next.\nAction: {tool_name}\nAction Input: {json.dumps(args)}"


def run_agent(tools, turns, *, max_iterations: int = 10):
    model = ScriptedModel([*turns, FINAL], name="scripted", on_exhausted="raise")
    config = AgentConfig(name="w", model=model, tools=list(tools), tool_calling_mode="react", capability_probe=False,
                         enable_memory=False, max_iterations=max_iterations, raise_on_error=False)
    with Agent(config) as agent:
        return agent.run("work"), model


def last_observation(model: ScriptedModel) -> str:
    return str(model.calls[-1].prompt).rsplit("Observation:", 1)[-1]


def state_tools():
    """A deterministic, disposable 'repository': one value that `set_state` changes and `read_state` reports."""
    state = {"value": "A", "reads": 0}

    @tool
    def read_state() -> str:
        """Return the current state."""
        state["reads"] += 1
        return f"state={state['value']}"

    @tool
    def set_state(value: str) -> str:
        """Change the state."""
        state["value"] = value
        return f"set to {value}"

    return state, [read_state, set_state]


# ---- F1: identical call, changed state ------------------------------------------------------------------------
def test_pin_a_custom_tool_repeat_is_replayed_from_the_record():
    # effGen's default for ANY tool, kept for tools outside the harness's own set: the second identical read is not
    # executed and the model is shown the recorded (now stale) value. This is the F1 mechanism in its purest form.
    state, tools = state_tools()
    r, model = run_agent(tools, [call("read_state"), call("set_state", value="B"), call("read_state")])
    assert state["reads"] == 1 and [c.name for c in r.tool_calls] == ["read_state", "set_state"]
    assert "state=A" in last_observation(model) and "state=B" not in last_observation(model)


@linux_only
def test_f1_rerunning_a_passing_check_after_breaking_the_code_shows_the_failure(tmp_path):
    ws = tmp_path / "repo"
    ws.mkdir()
    (ws / "f.py").write_text("x = 2\n")
    check = "python3 -c 'import f; assert f.x == 2, f.x' && echo PASSED"
    r, model = run_agent(build_tools(["shell"], ws), [
        call("shell", command=check), call("shell", command="echo 'x = 3' > f.py && echo changed"), call("shell", command=check)])
    assert len(r.tool_calls) == 3
    assert "AssertionError: 3" in last_observation(model)  # the code is broken now: the rerun must say so


@linux_only
def test_f1_rerunning_a_failed_check_after_the_fix_is_executed_and_passes(tmp_path):
    ws = tmp_path / "repo"
    ws.mkdir()
    check = "test -f ok.txt && echo PASSED"
    r, model = run_agent(build_tools(["shell"], ws), [
        call("shell", command=check), call("shell", command="touch ok.txt && echo fixed"), call("shell", command=check)])
    assert len(r.tool_calls) == 3  # today the identical rerun is declined, never executed
    assert "PASSED" in last_observation(model)


def test_f1_rereading_a_file_after_editing_it_shows_the_new_content(tmp_path):
    ws = tmp_path / "repo"
    ws.mkdir()
    (ws / "f.py").write_text("x = 1\n")  # written by the test, so no two tool results are equal (that would be F2)
    read = {"operation": "read", "path": "f.py"}
    fo, = build_tools(["file_operations"], ws)
    r, model = run_agent([fo], [
        call("file_operations", **read), call("file_operations", operation="write", path="f.py", content="x = 2\n"),
        call("file_operations", **read)])
    assert len(r.tool_calls) == 3
    assert "x = 2" in last_observation(model)


def test_f1_identical_redelegation_runs_the_worker_again(tmp_path):
    delegate_args = {"worker": "w", "task": "fix the bug"}
    worker = ScriptedModel(["Thought: x\nFinal Answer: FIRST attempt: could not fix",
                            "Thought: x\nFinal Answer: SECOND attempt: fixed"], name="worker")
    orch = ScriptedModel([call("delegate", **delegate_args), call("delegate", **delegate_args), FINAL],
                         name="orch", on_exhausted="raise")

    def cfg(model, name):
        return AgentConfig(name=name, model=model, tools=[], tool_calling_mode="react", capability_probe=False,
                           enable_memory=False, max_iterations=6, raise_on_error=False)

    result = build_harness(cfg(orch, "orch"), {"w": cfg(worker, "w")}).run("go")
    assert len(result.workers) == 2 and len(worker.calls) == 2  # today: 1 and 1, the second delegate is a replay
    assert "SECOND attempt" in last_observation(orch)


# ---- F2: different calls, identical result text -----------------------------------------------------------------------
def test_pin_a_tool_that_returns_constant_text_still_ends_the_run():
    # effGen behaviour that stays whatever we do (a result-text heuristic, not a bug in our code): two different
    # calls whose results read the same count as a repeat. Our tools must therefore describe what they did
    # (see the F2 tests below); a custom tool that returns a constant "ok" is outside that protection.
    ran = []

    @tool
    def touch(name: str) -> str:
        """Create a marker."""
        ran.append(name)
        return "ok"

    r, _ = run_agent([touch], [call("touch", name="a"), call("touch", name="b"), call("touch", name="c")])
    assert ran == ["a", "b"] and r.termination == "stuck" and r.stop_reason == "repeated_tool_result"


@linux_only
def test_f2_silent_shell_commands_do_not_end_the_run(tmp_path):
    ws = tmp_path / "repo"
    ws.mkdir()
    r, _ = run_agent(build_tools(["shell"], ws), [
        call("shell", command="true"), call("shell", command="mkdir -p d"), call("shell", command="touch marker.txt")])
    assert (ws / "d").is_dir()  # the second command did execute...
    assert (ws / "marker.txt").exists()  # ...and the third must be allowed to
    assert r.termination == "done"


def test_f2_writing_several_files_does_not_end_the_run(tmp_path):
    ws = tmp_path / "repo"
    ws.mkdir()
    fo, = build_tools(["file_operations"], ws)
    r, _ = run_agent([fo], [
        call("file_operations", operation="write", path="f.py", content="x = 1\n"),
        call("file_operations", operation="write", path="f.py", content="x = 2\n"),  # same text back: the path
        call("file_operations", operation="write", path="g.py", content="y = 1\n")])
    assert (ws / "f.py").read_text() == "x = 2\n" and (ws / "g.py").read_text() == "y = 1\n"  # the effects, not the text
    assert r.termination == "done" and len(r.tool_calls) == 3


@linux_only
def test_f2_a_long_run_of_distinct_silent_commands_is_not_cut_off(tmp_path):
    # effGen has more than one result-text heuristic (repeat, stall, drift); a longer sequence exercises all of them.
    ws = tmp_path / "repo"
    ws.mkdir()
    steps = ["true", "mkdir -p a", "mkdir -p b", "touch a/1", "touch b/2", "ln -s a/1 link", "chmod +x a/1"]
    r, _ = run_agent(build_tools(["shell"], ws), [call("shell", command=c) for c in steps], max_iterations=12)
    assert r.termination == "done" and len(r.tool_calls) == len(steps)
    assert (ws / "a/1").exists() and (ws / "b/2").exists() and (ws / "link").is_symlink()


def test_control_a_long_run_of_distinct_file_writes_is_not_cut_off(tmp_path):
    ws = tmp_path / "repo"
    ws.mkdir()
    fo, = build_tools(["file_operations"], ws)
    names = [f"m{i}.py" for i in range(7)]
    r, _ = run_agent([fo], [call("file_operations", operation="write", path=n, content="pass\n") for n in names], max_iterations=12)
    assert r.termination == "done" and len(r.tool_calls) == len(names)  # same content in every file: the PATH differs
    assert all((ws / n).read_text() == "pass\n" for n in names)


# ---- model-free pins of the two conditions (so an effGen upgrade that changes them is noticed) ------------------------------
def test_pin_result_key_ignores_the_arguments_and_replay_ignores_state():
    guards = NativeToolLoop({})
    guards.record_result("shell", "exit code: 0")
    assert guards.result_is_repeat("shell", "exit code:   0")  # whitespace-normalised, and no input in the key
    guards = NativeToolLoop({"shell": object()})
    first = guards.check_action("shell", '{"command": "cat f.py"}')
    guards.record_action(first)
    guards.record_pair_result(first, "x = 1")
    second = guards.check_action("shell", '{"command":"cat f.py"}')  # same call, different JSON spacing
    assert second.is_exact_loop and guards.cached_result(second) == "x = 1"  # replayed, whatever changed on disk
    assert guards.cached_result(second) == "x = 1" and guards.cached_result(second) is None  # MAX_REPLAYS_PER_PAIR = 2


# ---- controls: protections we keep, passing today and after any fix ----------------------------------------------------------
def test_control_distinct_calls_with_distinct_results_all_run():
    state, tools = state_tools()
    r, _ = run_agent(tools, [call("set_state", value="B"), call("read_state"), call("set_state", value="C")])
    assert len(r.tool_calls) == 3 and r.termination == "done"


def test_control_a_stuck_idempotent_loop_is_still_stopped():
    @tool
    def lookup(key: str) -> str:
        """Pure lookup."""
        return f"value of {key}"

    stuck = [call("lookup", key="a")] * 8
    r, model = run_agent([lookup], stuck, max_iterations=10)
    assert len(r.tool_calls) == 1  # one real run, then replays/declines: it neither executes 8 times nor loops forever
    assert len(model.calls) < 10


@linux_only
def test_control_a_stuck_harness_tool_loop_is_still_stopped(tmp_path):
    # The same command with the same output, over and over: neither executed 8 times nor allowed to run to the cap.
    ws = tmp_path / "repo"
    ws.mkdir()
    r, model = run_agent(build_tools(["shell"], ws), [call("shell", command="echo hi")] * 8, max_iterations=10)
    assert 1 <= len(r.tool_calls) <= 3
    assert len(model.calls) < 10


# ======================================================================================================
# The patch itself (harness.install_effgen_loop_compat): contract, scope, failure modes, cleanup.
# ======================================================================================================
def agent_cfg(model, name, tools=(), **over) -> AgentConfig:
    base = dict(name=name, model=model, tools=list(tools), tool_calling_mode="react", capability_probe=False,
                enable_memory=False, max_iterations=6, raise_on_error=False)
    base.update(over)
    return AgentConfig(**base)


# ---- scope: exactly the marked tools, exactly the pre-dispatch exact-repeat check ----------------------------------------
def test_every_tool_the_harness_can_build_is_marked_and_effgens_own_classes_are_not(tmp_path):
    from effgen.tools.builtin.file_ops import FileOperations
    assert STATEFUL_TOOL_MARKER == "executes_every_call"
    built = build_tools(list(TOOL_NAMES), tmp_path) if platform.system() == "Linux" else build_tools(["file_operations"], tmp_path)
    assert built and all(getattr(t, STATEFUL_TOOL_MARKER, False) for t in built)  # a future tool must opt in, not be forgotten
    assert getattr(DelegateTool(WorkerRegistry()), STATEFUL_TOOL_MARKER) is True
    assert not hasattr(FileOperations, STATEFUL_TOOL_MARKER) and issubclass(WorkspaceFileOperations, FileOperations)
    assert not any(hasattr(c, STATEFUL_TOOL_MARKER) for c in (type(state_tools()[1][0]),))  # plain @tool functions


def test_the_patched_check_clears_only_the_exact_repeat_of_marked_tools():
    class Marked:
        executes_every_call = True

    guards = agent_loop.NativeToolLoop({"shell": Marked(), "lookup": object()})
    assert type(guards) is not NativeToolLoop and isinstance(guards, NativeToolLoop)
    for name, replayed in (("shell", False), ("lookup", True)):
        first = guards.check_action(name, '{"a": 1}')
        assert first.is_exact_loop is False  # nothing dispatched yet
        guards.record_action(first)
        again = guards.check_action(name, '{"a":1}')
        assert again.is_exact_loop is replayed  # marked: executes again; unmarked: effGen's decision, untouched
    # every other field of the check is effGen's own
    plain = NativeToolLoop({"shell": Marked()})
    plain.record_action(plain.check_action("shell", "{}"))
    upstream, patched = plain.check_action("shell", "{}"), guards.check_action("shell", '{"a": 1}')
    assert upstream.is_exact_loop is True and dataclasses.replace(upstream, is_exact_loop=False).pair == ("shell", "{}")
    assert patched.stalled_call_count == 0 and patched.is_fuzzy_loop is False


def test_unmarked_tools_keep_upstream_replay_under_the_patch():
    # The pin in the section above runs with the patch installed (session fixture): a plain tool is still replayed.
    assert effgen_loop_compat_installed()
    test_pin_a_custom_tool_repeat_is_replayed_from_the_record()


# ---- installation: explicit, idempotent, loud, and never undone by the harness --------------------------------------------
def test_installation_is_idempotent_and_wraps_effgens_class_exactly_once():
    install_effgen_loop_compat()
    first = agent_loop.NativeToolLoop
    install_effgen_loop_compat()
    install_effgen_loop_compat()
    assert agent_loop.NativeToolLoop is first and effgen_loop_compat_installed()
    assert first.__mro__.count(first) == 1 and first.__mro__[1] is agent_tool_loop.NativeToolLoop  # one layer over effGen's own
    assert agent_tool_loop.NativeToolLoop.__name__ == "NativeToolLoop"  # effGen's class object itself is never modified


def test_importing_the_harness_and_building_one_do_not_install_anything():
    # In a fresh interpreter, so nothing this suite installed can be mistaken for the library's own doing.
    code = (
        "import sys; sys.path.insert(0, 'tests')\n"
        "from conftest import ScriptedModel\n"
        "from effgen import AgentConfig\n"
        "from effgen.core import agent_loop, agent_tool_loop\n"
        "import smol_harness.harness as h, smol_harness.tools, smol_harness.delegation\n"
        "cfg = lambda m, n: AgentConfig(name=n, model=m, tools=[], tool_calling_mode='react', capability_probe=False, enable_memory=False)\n"
        "h.build_harness(cfg(ScriptedModel(['Final Answer: x']), 'o'), {'w': cfg(ScriptedModel(['Final Answer: y']), 'w')})\n"
        "print(h.effgen_loop_compat_installed(), agent_loop.NativeToolLoop is agent_tool_loop.NativeToolLoop)\n")
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
                          env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == "False True"


def test_harness_run_refuses_to_run_without_the_patch_and_runs_nothing(compat_uninstalled):
    orch = ScriptedModel(["Thought: x\nFinal Answer: hi"], name="orch")
    harness = build_harness(agent_cfg(orch, "orch"), {"w": agent_cfg(ScriptedModel(["Final Answer: -"]), "w")})
    with pytest.raises(EffgenCompatError, match=r"install_effgen_loop_compat\(\)"):
        harness.run("task")
    assert orch.calls == []  # not a single model call: the refusal comes before any agent exists


def test_a_marked_tool_replays_without_the_patch_which_is_why_it_exists(compat_uninstalled, tmp_path):
    ws = tmp_path / "repo"
    ws.mkdir()
    (ws / "f.py").write_text("x = 1\n")
    fo, = build_tools(["file_operations"], ws)
    read = {"operation": "read", "path": "f.py"}
    r, model = run_agent([fo], [call("file_operations", **read), call("file_operations", operation="write", path="f.py", content="x = 2\n"),
                                call("file_operations", **read)])
    assert len(r.tool_calls) == 2 and "x = 1" in last_observation(model)  # effGen's own behaviour: stale replay


BREAKAGES = {
    "version": lambda mp: mp.setattr(effgen, "__version__", "9.9.9"),
    "check_action missing": lambda mp: mp.delattr(agent_tool_loop.NativeToolLoop, "check_action"),
    "check_action signature": lambda mp: mp.setattr(agent_tool_loop.NativeToolLoop, "check_action", lambda self, action: None),
    "LoopCheck field": lambda mp: mp.setattr(agent_tool_loop, "LoopCheck", object),
    "agent_loop name": lambda mp: mp.delattr(agent_loop, "NativeToolLoop"),
    "construction site": lambda mp: mp.setattr(agent_loop, "drive", lambda: None),
}


@pytest.mark.parametrize("what", sorted(BREAKAGES))
def test_incompatible_effgen_fails_loudly_and_patches_nothing(compat_uninstalled, monkeypatch, what):
    BREAKAGES[what](monkeypatch)
    before = getattr(agent_loop, "NativeToolLoop", None)
    with pytest.raises(EffgenCompatError, match="Nothing was patched") as err:
        install_effgen_loop_compat()
    assert "stale results" in str(err.value) and "docs/effgen-api-audit.md" in str(err.value)  # actionable, not cryptic
    assert not effgen_loop_compat_installed() and getattr(agent_loop, "NativeToolLoop", None) is before


def test_installation_refuses_to_wrap_an_unknown_replacement(compat_uninstalled, monkeypatch):
    other = type("SomeoneElsesLoop", (agent_tool_loop.NativeToolLoop,), {})
    monkeypatch.setattr(agent_loop, "NativeToolLoop", other)
    with pytest.raises(EffgenCompatError, match="already .*SomeoneElsesLoop"):
        install_effgen_loop_compat()
    assert agent_loop.NativeToolLoop is other and not effgen_loop_compat_installed()


def test_a_run_that_fails_or_finishes_never_restores_effgen_and_a_dropped_harness_does_not_either(tmp_path):
    ws = tmp_path / "repo"
    ws.mkdir()
    orch = ScriptedModel([call("shell", command="echo hi"), FINAL] if platform.system() == "Linux" else [FINAL], name="orch")
    tools = build_tools(["shell"], ws) if platform.system() == "Linux" else []
    harness = build_harness(agent_cfg(orch, "orch", tools), {"w": agent_cfg(ScriptedModel(["Final Answer: -"]), "w")})
    patched = agent_loop.NativeToolLoop
    assert harness.run("go").success and agent_loop.NativeToolLoop is patched
    failing = build_harness(agent_cfg(ScriptedModel(["Thought: x\nAction: nope\nAction Input: {}"] * 3, name="o"), "o", max_iterations=2),
                            {"w": agent_cfg(ScriptedModel(["Final Answer: -"]), "w")})
    failing.run("go")  # does not finish; whatever it returns, it must not touch the patch
    del harness, failing
    assert agent_loop.NativeToolLoop is patched and effgen_loop_compat_installed()
    # (`Harness.close()` does not exist yet; T5.0 adds it and must not uninstall: it is in that task's acceptance list.)


def test_test_only_restoration_returns_effgens_own_class_untouched(compat_uninstalled):
    assert agent_loop.NativeToolLoop is agent_tool_loop.NativeToolLoop
    assert agent_tool_loop.NativeToolLoop.check_action.__qualname__ == "NativeToolLoop.check_action"  # no override left behind
    assert not effgen_loop_compat_installed()
    uninstall_effgen_loop_compat_for_tests()  # restoring twice is harmless
    assert agent_loop.NativeToolLoop is agent_tool_loop.NativeToolLoop


def test_restoration_refuses_to_overwrite_a_class_it_did_not_install(monkeypatch):
    other = type("SomeoneElsesLoop", (agent_loop.NativeToolLoop,), {})
    monkeypatch.setattr(agent_loop, "NativeToolLoop", other)
    with pytest.raises(EffgenCompatError, match="refusing to overwrite"):
        uninstall_effgen_loop_compat_for_tests()
    assert agent_loop.NativeToolLoop is other


# ---- no state leaks between independent runs -----------------------------------------------------------------------------------
@linux_only
def test_sequential_harness_runs_do_not_share_loop_state(tmp_path):
    ws = tmp_path / "repo"
    ws.mkdir()
    (ws / "f.py").write_text("x = 1\n")
    read = call("shell", command="cat f.py")
    orch = ScriptedModel([read, FINAL, read, FINAL], name="orch", on_exhausted="raise")
    harness = build_harness(agent_cfg(orch, "orch", build_tools(["shell"], ws)), {"w": agent_cfg(ScriptedModel(["Final Answer: -"]), "w")})
    first = harness.run("look")
    (ws / "f.py").write_text("x = 2\n")
    second = harness.run("look again")  # the identical call, in a new run, after the file changed
    assert first.success and second.success
    assert "x = 1" in str(orch.calls[1].prompt) and "x = 2" in str(orch.calls[3].prompt)
    assert agent_loop.NativeToolLoop is harness_mod._compat_class  # the same single patched class served both runs


# ---- genuinely unproductive behaviour is still bounded by the protections we kept -------------------------------------------------
@linux_only
def test_an_endlessly_repeated_failing_command_is_still_stopped(tmp_path):
    ws = tmp_path / "repo"
    ws.mkdir()
    r, model = run_agent(build_tools(["shell"], ws), [call("shell", command="python3 -c 'import nope_missing'")] * 12, max_iterations=12)
    assert 1 <= len(r.tool_calls) <= 5  # the stall guard ("the same error again is not progress") ends it long before the cap
    assert len(model.calls) < 12


def test_an_endlessly_repeated_file_read_is_still_stopped(tmp_path):
    ws = tmp_path / "repo"
    ws.mkdir()
    (ws / "f.py").write_text("x = 1\n")
    fo, = build_tools(["file_operations"], ws)
    r, model = run_agent([fo], [call("file_operations", operation="read", path="f.py")] * 12, max_iterations=12)
    assert 1 <= len(r.tool_calls) <= 3  # same call, same content: a repeated result
    assert len(model.calls) < 12


def test_endlessly_repeated_delegation_is_bounded_by_the_delegation_limit():
    worker = ScriptedModel(["Thought: x\nFinal Answer: could not fix"], name="worker")  # repeat_last: it never gets better
    orch = ScriptedModel([call("delegate", worker="w", task="fix it")] * 20, name="orch")
    harness = build_harness(agent_cfg(orch, "orch", max_iterations=10), {"w": agent_cfg(worker, "w")}, max_delegations=3)
    result = harness.run("go")
    ran = [r for r in result.workers if r.result.error_type != "delegation_limit_reached"]
    assert len(ran) == 3 and len(worker.calls) == 3  # the limit, not the loop guard, caps the real worker runs
    assert len(orch.calls) <= 10  # and the iteration cap bounds the orchestrator
