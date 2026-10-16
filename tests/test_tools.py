"""T4.1: tool factory and workspace-confined `file_operations`.

Everything goes through effGen's real `BaseTool.execute()` (and, at the end, its real
agent loop with scripted models) on temporary workspaces. CPU only, no network.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import pytest
from conftest import ScriptedModel
from effgen import AgentConfig
from effgen.tools.base_tool import BaseTool
from effgen.tools.builtin.file_ops import FileOperations

import smol_harness.delegation as delegation_mod
import smol_harness.harness as harness_mod
import smol_harness.tools as tools_mod
from smol_harness.config import KNOWN_TOOLS, loads_config
from smol_harness.harness import build_harness
from smol_harness.tools import TOOL_NAMES, ToolBuildError, build_tools, resolve_workspace

WEB_LIKE = re.compile(r"web|search_engine|url_fetch|http", re.I)


def run(tool: BaseTool, **kw):
    return asyncio.run(tool.execute(**kw))


def data(result):
    assert result.success, result.error
    return result.output["data"]


def refused(result, reason: str = "outside allowed directories") -> str:
    assert not result.success
    assert reason in result.error, result.error
    return result.error


@pytest.fixture
def layout(tmp_path):
    """tmp/ws (the workspace) next to tmp/outside (must stay untouched)."""
    ws, outside = tmp_path / "ws", tmp_path / "outside"
    ws.mkdir()
    outside.mkdir()
    (ws / "a.txt").write_text("inside\n")
    (outside / "secret.txt").write_text("outside secret\n")
    return ws, outside


@pytest.fixture
def fo(layout):
    (tool,) = build_tools(["file_operations"], layout[0])
    return tool


def outside_unchanged(outside: Path) -> None:
    assert sorted(p.name for p in outside.iterdir()) == ["secret.txt"]
    assert (outside / "secret.txt").read_text() == "outside secret\n"


# ---- 1-3: valid reads, writes, nested paths ----------------------------------------------------------
def test_reads_a_workspace_file(fo, layout):
    assert data(run(fo, operation="read", path="a.txt")) == "inside\n"
    assert data(run(fo, operation="read", path=str(layout[0] / "a.txt"))) == "inside\n"  # absolute, inside


def test_writes_a_workspace_file(fo, layout):
    written = data(run(fo, operation="write", path="b.py", content="x = 1\n"))
    assert Path(written) == layout[0] / "b.py"
    assert (layout[0] / "b.py").read_text() == "x = 1\n"
    data(run(fo, operation="write", path="a.txt", content="replaced"))
    assert (layout[0] / "a.txt").read_text() == "replaced"


def test_nested_paths_create_parents_and_read_back(fo, layout):
    data(run(fo, operation="write", path="pkg/sub/mod.py", content="def f():\n    return 1\n"))
    assert (layout[0] / "pkg/sub/mod.py").is_file()
    assert data(run(fo, operation="read", path="pkg/sub/mod.py")).startswith("def f()")
    assert data(run(fo, operation="read", path="pkg/./sub/../sub/mod.py")).startswith("def f()")  # stays inside


def test_relative_paths_resolve_against_the_workspace_not_the_cwd(fo, layout, tmp_path, monkeypatch):
    monkeypatch.chdir(layout[1])
    data(run(fo, operation="write", path="c.txt", content="c"))
    assert (layout[0] / "c.txt").exists()
    outside_unchanged(layout[1])


def test_list_and_search_discover_repository_files(fo, layout):
    data(run(fo, operation="write", path="src/m.py", content="pass\n"))
    listed = {Path(e["path"]).relative_to(layout[0]).as_posix() for e in data(run(fo, operation="list", path=".", recursive=True))}
    assert {"a.txt", "src", "src/m.py"} <= listed
    found = data(run(fo, operation="search", path=".", pattern="*.py", recursive=True))
    assert [Path(p) for p in found] == [layout[0] / "src/m.py"]


# ---- 4-6: escapes are refused, for reads and writes ----------------------------------------------------
@pytest.mark.parametrize("path", ["../outside/secret.txt", "sub/../../outside/secret.txt", "..", "../../x.txt"])
def test_relative_traversal_is_refused(fo, layout, path):
    refused(run(fo, operation="read", path=path))
    refused(run(fo, operation="write", path=path.replace("secret", "new"), content="pwned"))
    outside_unchanged(layout[1])


def test_absolute_path_escapes_are_refused(fo, layout):
    refused(run(fo, operation="read", path=str(layout[1] / "secret.txt")))
    refused(run(fo, operation="write", path=str(layout[1] / "new.txt"), content="pwned"))
    refused(run(fo, operation="write", path=str(layout[1] / "secret.txt"), content="pwned"))
    refused(run(fo, operation="read", path="/etc/hostname"))
    refused(run(fo, operation="list", path="/"))
    refused(run(fo, operation="metadata", path=str(layout[1])))
    outside_unchanged(layout[1])


def test_symlink_escapes_are_refused_for_reads_and_writes(fo, layout):
    ws, outside = layout
    (ws / "file_link.txt").symlink_to(outside / "secret.txt")
    (ws / "dir_link").symlink_to(outside, target_is_directory=True)
    (ws / "dangling.txt").symlink_to(outside / "created.txt")  # target does not exist yet
    for path in ("file_link.txt", "dir_link/secret.txt"):
        refused(run(fo, operation="read", path=path))
        refused(run(fo, operation="metadata", path=path))
    refused(run(fo, operation="write", path="file_link.txt", content="pwned"))
    refused(run(fo, operation="write", path="dir_link/new.txt", content="pwned"))
    refused(run(fo, operation="write", path="dangling.txt", content="pwned"))
    refused(run(fo, operation="list", path="dir_link"))
    outside_unchanged(outside)


def test_listing_and_search_do_not_leak_symlinked_or_parent_entries(fo, layout):
    ws, outside = layout
    (ws / "dir_link").symlink_to(outside, target_is_directory=True)
    listed = [e["path"] for e in data(run(fo, operation="list", path=".", recursive=True))]
    assert all(Path(p).resolve().is_relative_to(ws) for p in listed) and not any("dir_link" in p for p in listed)
    assert data(run(fo, operation="search", path=".", pattern="../outside/*")) == []
    assert data(run(fo, operation="search", path=".", pattern="**/secret.txt", recursive=True)) == []


def test_symlink_inside_the_workspace_is_followed(fo, layout):
    (layout[0] / "alias.txt").symlink_to(layout[0] / "a.txt")
    assert data(run(fo, operation="read", path="alias.txt")) == "inside\n"


def test_a_symlinked_workspace_confines_to_its_target(layout, tmp_path):
    link = tmp_path / "ws_link"
    link.symlink_to(layout[0], target_is_directory=True)
    (tool,) = build_tools(["file_operations"], link)
    assert data(run(tool, operation="read", path="a.txt")) == "inside\n"
    refused(run(tool, operation="read", path="../outside/secret.txt"))


def test_malformed_paths_fail_cleanly(fo):
    assert not run(fo, operation="read", path="a\x00.txt").success
    assert not run(fo, operation="write", path="", content="x").success


# ---- 7-8: workspace validation --------------------------------------------------------------------------
def test_missing_workspace_is_rejected(tmp_path):
    with pytest.raises(ToolBuildError, match="does not exist"):
        build_tools(["file_operations"], tmp_path / "missing")


def test_workspace_that_is_a_regular_file_is_rejected(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(ToolBuildError, match="not a directory"):
        build_tools(["file_operations"], f)


def test_empty_and_broken_symlink_workspaces_are_rejected(tmp_path):
    with pytest.raises(ToolBuildError, match="non-empty"):
        resolve_workspace("")
    (tmp_path / "broken").symlink_to(tmp_path / "nowhere")
    with pytest.raises(ToolBuildError, match="does not exist"):
        resolve_workspace(tmp_path / "broken")


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses permission bits")
def test_inaccessible_workspace_is_rejected(tmp_path):
    ws = tmp_path / "locked"
    ws.mkdir()
    ws.chmod(0o500)  # no write
    try:
        with pytest.raises(ToolBuildError, match="not readable, writable"):
            build_tools(["file_operations"], ws)
    finally:
        ws.chmod(0o700)


def test_workspace_is_validated_even_for_an_empty_allowlist(tmp_path):
    with pytest.raises(ToolBuildError):
        build_tools([], tmp_path / "missing")


def test_resolve_workspace_returns_the_resolved_directory(layout):
    assert resolve_workspace(str(layout[0] / "." / ".." / "ws")) == layout[0].resolve()


# ---- 9-12: allowlists, unknown and forbidden tools --------------------------------------------------------
def test_allowlist_is_exact(layout):
    assert build_tools([], layout[0]) == []
    tools = build_tools(["file_operations"], layout[0])
    assert [t.name for t in tools] == ["file_operations"] and isinstance(tools[0], FileOperations)


def test_each_call_returns_fresh_instances(layout):
    (a,), (b,) = build_tools(["file_operations"], layout[0]), build_tools(["file_operations"], layout[0])
    assert a is not b


@pytest.mark.parametrize("name", ["bash", "code_executor", "edit_file", "FILE_OPERATIONS", "", "web_search",
                                  "url_fetch", "http_request", "browser", "agentic_search", "wikipedia"])
def test_unknown_and_web_tools_are_rejected(layout, name):
    with pytest.raises(ToolBuildError, match="unknown tool"):
        build_tools(["file_operations", name], layout[0])


def test_delegate_is_never_built(layout):
    with pytest.raises(ToolBuildError, match="orchestrator only"):
        build_tools(["delegate"], layout[0])


def test_duplicates_and_bare_strings_are_rejected(layout):
    with pytest.raises(ToolBuildError, match="duplicate"):
        build_tools(["file_operations", "file_operations"], layout[0])
    with pytest.raises(ToolBuildError, match="list of names"):
        build_tools("file_operations", layout[0])


def test_shell_fails_closed_until_t42(layout):
    with pytest.raises(ToolBuildError, match="T4.2"):
        build_tools(["shell"], layout[0])
    with pytest.raises(ToolBuildError, match="T4.2"):
        build_tools(["file_operations", "shell"], layout[0])


def test_names_are_checked_before_the_workspace(tmp_path):
    with pytest.raises(ToolBuildError, match="unknown tool"):
        build_tools(["web_search"], tmp_path / "missing")


def test_registry_matches_config_and_holds_no_web_tool():
    assert TOOL_NAMES == KNOWN_TOOLS
    assert not any(WEB_LIKE.search(n) for n in TOOL_NAMES)


CONFIG = """
workspace: ws
orchestrator:
  model: {provider: ollama, model: "qwen2.5:7b", context_length: 8192}
  tools: [file_operations, delegate]
  max_iterations: 6
workers:
  - name: reader
    description: reads files
    model: {provider: ollama, model: "qwen2.5:1.5b", context_length: 4096}
    tools: [file_operations]
    max_iterations: 4
  - name: thinker
    description: no tools
    model: {provider: ollama, model: "qwen2.5:1.5b", context_length: 4096}
    tools: []
    max_iterations: 4
"""


def test_orchestrator_and_workers_get_their_own_allowlists(layout):
    config = loads_config(CONFIG, base_dir=layout[0].parent)
    assert config.workspace == layout[0]
    built = {spec.name: [t.name for t in build_tools(spec.tools, config.workspace)]
             for spec in (config.orchestrator, *config.workers)}
    assert built == {"orchestrator": ["file_operations"], "reader": ["file_operations"], "thinker": []}


# ---- the harness: real effGen loop, scripted models -------------------------------------------------------
def cfg(model, name, tools) -> AgentConfig:
    return AgentConfig(name=name, model=model, tools=list(tools), tool_calling_mode="react", capability_probe=False,
                       enable_memory=False, max_iterations=6, raise_on_error=False)


def test_harness_agents_hold_exactly_their_tools_and_no_web_tool(layout, monkeypatch):
    ws, outside = layout
    held: dict[str, list[str]] = {}

    class RecordingAgent(harness_mod.Agent):
        def __init__(self, config, *a, **kw):
            super().__init__(config, *a, **kw)
            held[config.name] = sorted(self.tools)

    monkeypatch.setattr(harness_mod, "Agent", RecordingAgent)
    monkeypatch.setattr(delegation_mod, "Agent", RecordingAgent)

    write = json.dumps({"operation": "write", "path": "notes/out.txt", "content": "from worker"})
    escape = json.dumps({"operation": "write", "path": "../outside/pwned.txt", "content": "x"})
    worker = ScriptedModel([
        f"Thought: write.\nAction: file_operations\nAction Input: {write}",
        f"Thought: try escaping.\nAction: file_operations\nAction Input: {escape}",
        "Thought: done.\nFinal Answer: wrote notes/out.txt",
    ], name="worker-model")
    orch = ScriptedModel([
        'Thought: delegate.\nAction: delegate\nAction Input: {"worker": "coder", "task": "write the note"}',
        "Thought: done.\nFinal Answer: ok",
    ], name="orch-model")
    harness = build_harness(
        cfg(orch, "orch", build_tools(["file_operations"], ws)),
        {"coder": cfg(worker, "coder", build_tools(["file_operations"], ws)),
         "idle": cfg(ScriptedModel(["Final Answer: -"]), "idle", build_tools([], ws))},
        {"coder": "writes files", "idle": "unused"},
    )
    result = harness.run("write a note")

    assert result.success, result.output
    assert held == {"orch": ["delegate", "file_operations"], "coder": ["file_operations"]}
    assert not any(WEB_LIKE.search(n) for names in held.values() for n in names)
    assert (ws / "notes/out.txt").read_text() == "from worker"
    assert [(r.worker, r.status) for r in result.workers] == [("coder", "completed")]
    # The escape attempt really ran and came back to the worker as a refusal observation.
    assert len(worker.calls) == 3 and "outside allowed directories" in str(worker.calls[2].prompt)
    outside_unchanged(outside)


# ---- regression pins for surprising effGen 1.3.0 behaviour (audit §1.9) -----------------------------------
def test_effgen_widens_a_missing_allowed_directory_to_the_cwd(tmp_path, monkeypatch):
    # Why build_tools validates the workspace and reads back the kept root.
    monkeypatch.delenv("EFFGEN_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    assert FileOperations(allowed_directories=[str(tmp_path / "missing")])._allowed_directories == [tmp_path.resolve()]


def test_effgen_falls_back_to_and_creates_effgen_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFGEN_WORKSPACE", str(tmp_path / "env_ws"))
    tool = FileOperations(allowed_directories=[str(tmp_path / "missing")])
    assert tool._allowed_directories == [(tmp_path / "env_ws").resolve()] and (tmp_path / "env_ws").is_dir()


@pytest.mark.parametrize("env", [None, "env_ws"])
def test_a_workspace_vanishing_after_validation_fails_closed(tmp_path, monkeypatch, env):
    if env:
        monkeypatch.setenv("EFFGEN_WORKSPACE", str(tmp_path / env))
    else:
        monkeypatch.delenv("EFFGEN_WORKSPACE", raising=False)
    monkeypatch.setattr(tools_mod, "resolve_workspace", lambda ws: tmp_path / "gone")  # simulates the race
    with pytest.raises(ToolBuildError, match="did not confine"):
        build_tools(["file_operations"], tmp_path / "gone")


def test_effgen_workspace_env_does_not_override_an_explicit_workspace(layout, tmp_path, monkeypatch):
    monkeypatch.setenv("EFFGEN_WORKSPACE", str(tmp_path / "env_ws"))
    (tool,) = build_tools(["file_operations"], layout[0])
    data(run(tool, operation="write", path="x.txt", content="x"))
    assert (layout[0] / "x.txt").exists() and not (tmp_path / "env_ws").exists()


@pytest.mark.parametrize("name", ["Makefile", "README", ".gitignore", "Dockerfile"])
def test_effgen_refuses_extensionless_files(fo, layout, name):
    (layout[0] / name).write_text("x\n")
    refused(run(fo, operation="read", path=name), "not allowed for read")
    refused(run(fo, operation="write", path=name, content="y"), "not allowed for write")


def test_effgen_refuses_credential_filenames_and_content(fo, layout):
    (layout[0] / ".env").write_text("A=1\n")
    refused(run(fo, operation="read", path=".env"), "credentials")
    (layout[0] / "settings.txt").write_text("API_KEY=abc\nDB_PASSWORD=xyz\n")
    refused(run(fo, operation="read", path="settings.txt"), "credentials")
