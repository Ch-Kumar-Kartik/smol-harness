"""T4.3: the `edit_file` tool (exact, single-occurrence text replacement), tested through effGen's real `BaseTool.execute()` and,
at the end, through the real harness and effGen agent loop with scripted models. CPU only, no network, no model, no Docker.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml
from effgen.tools.base_tool import BaseTool
from test_harness_config import BIG, FINAL, SMALL, FakeModels, call, delegate

import smol_harness.__main__ as cli
import smol_harness.models as models_mod
from smol_harness.config import KNOWN_TOOLS, loads_config
from smol_harness.delegation import STATEFUL_TOOL_MARKER
from smol_harness.tools import TOOL_NAMES, EditFileTool, ToolBuildError, build_tools

pytestmark = pytest.mark.skipif(platform.system() != "Linux", reason="edit_file needs POSIX openat-style file APIs")
ROOT = Path(__file__).resolve().parents[1]


def run(tool: BaseTool, **kw):
    return asyncio.run(tool.execute(**kw))


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    (root / "pkg").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside original\n")
    return root


@pytest.fixture
def edit(ws):
    tool = build_tools(["edit_file"], ws)[0]

    def go(path, old, new, **kw):
        return run(tool, path=path, old_text=old, new_text=new, **kw)

    return go


def listing(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def assert_refused(result, *words):
    assert result.success is False and "Nothing was changed" in result.error, result
    for w in words:
        assert w in result.error, (w, result.error)


# ---- 1, 2, 10: the localized edit, preservation, bounded output ------------------------------------------------------------------
def test_a_localized_edit_replaces_only_the_target_and_reports_briefly(ws, edit):
    (ws / "pkg" / "a.py").write_text("one\ntwo\nthree\n")
    r = edit("pkg/a.py", "two", "TWO")
    assert r.success and (ws / "pkg" / "a.py").read_text() == "one\nTWO\nthree\n"
    assert r.output["result"].startswith("edited pkg/a.py: replaced 1 occurrence at line 2 (1 -> 1 lines)") and len(r.output["result"]) < 200
    assert r.output["sha256"] in r.output["result"]


def test_unrelated_content_mode_other_files_and_directory_contents_are_preserved(ws, edit):
    body = "keep \tthis  \n\n    indented()\nmarker = 1\nlast line without newline"
    target = ws / "pkg" / "m.py"
    target.write_bytes(body.encode())
    target.chmod(0o640)
    (ws / "pkg" / "other.py").write_text("untouched\n")
    before = listing(ws)
    assert edit("pkg/m.py", "marker = 1", "marker = 22").success
    assert target.read_bytes() == body.replace("marker = 1", "marker = 22").encode()
    assert stat.S_IMODE(target.stat().st_mode) == 0o640 and (ws / "pkg" / "other.py").read_text() == "untouched\n"
    assert listing(ws) == before  # no temporary file is left behind


def test_the_report_is_bounded_even_for_hostile_arguments(ws, edit):
    (ws / "f.txt").write_text("x\n" * 3)
    r = edit("f.txt", "x", "y")
    assert r.error and len(r.error) < 600 and "starting at lines 1, 2, 3" in r.error
    r = edit("a/" * 400 + "f.txt", "x", "y")
    assert r.success is False and len(r.error) < 800


# ---- 3, 4, 5: missing file, missing text, ambiguity ------------------------------------------------------------------------------
def test_a_missing_file_or_directory_is_refused(ws, edit):
    assert_refused(edit("pkg/missing.py", "a", "b"), "does not exist")
    assert_refused(edit("nodir/x.py", "a", "b"), "does not exist")
    assert listing(ws) == ["pkg"]  # nothing was created


def test_text_that_is_absent_leaves_the_file_untouched(ws, edit):
    f = ws / "pkg" / "a.py"
    f.write_text("alpha\n")
    assert_refused(edit("pkg/a.py", "beta", "x"), "was not found", "exactly")
    assert f.read_text() == "alpha\n" and listing(ws) == ["pkg", "pkg/a.py"]


@pytest.mark.parametrize("body,old,count", [("a x a\n", "a", "2 times"), ("aaa\n", "aa", "2 times"), ("l\nl\nl\nl\nl\nl\nl\n", "l\n", "more than 5 times")])
def test_duplicate_and_overlapping_matches_are_ambiguous_and_change_nothing(ws, edit, body, old, count):
    f = ws / "d.txt"
    f.write_text(body)
    assert_refused(edit("d.txt", old, "Z"), count, "add neighbouring lines")
    assert f.read_text() == body


# ---- 6, 7: traversal and absolute paths ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", ["../outside/secret.txt", "pkg/../../outside/secret.txt", "./../outside/secret.txt", "pkg/..", ".."])
def test_path_traversal_is_refused(ws, edit, path):
    assert_refused(edit(path, "outside", "HACKED"), "'..'")
    assert (ws.parent / "outside" / "secret.txt").read_text() == "outside original\n"


@pytest.mark.parametrize("make", [lambda ws: str(ws.parent / "outside" / "secret.txt"), lambda ws: str(ws / "pkg" / "a.py"), lambda ws: "/etc/hostname",
                                  lambda ws: "C:/Windows/x.py", lambda ws: "pkg\\a.py", lambda ws: "\\\\server\\share\\x"])
def test_absolute_paths_and_backslashes_are_refused_even_when_they_point_inside(ws, edit, make):
    (ws / "pkg" / "a.py").write_text("inside\n")
    assert_refused(edit(make(ws), "inside", "X"), "absolute paths")
    assert (ws / "pkg" / "a.py").read_text() == "inside\n"


@pytest.mark.parametrize("path", ["", "   ", ".", "./", "pkg/", "a\x00b"])
def test_empty_or_degenerate_paths_are_refused(ws, edit, path):
    r = edit(path, "a", "b")
    assert r.success is False


# ---- 8, 9: symbolic links --------------------------------------------------------------------------------------------------------
def test_a_symlink_file_is_never_followed_whether_it_points_inside_or_outside(ws, edit):
    (ws / "real.txt").write_text("real\n")
    (ws / "to_inside.txt").symlink_to("real.txt")
    (ws / "to_outside.txt").symlink_to(ws.parent / "outside" / "secret.txt")
    (ws / "dangling.txt").symlink_to("nothing")
    for name in ("to_inside.txt", "to_outside.txt", "dangling.txt"):
        assert_refused(edit(name, "real" if "inside" in name else "outside", "X"), "symbolic link")
    assert (ws / "real.txt").read_text() == "real\n" and (ws.parent / "outside" / "secret.txt").read_text() == "outside original\n"
    assert (ws / "to_inside.txt").is_symlink()  # the link itself was not replaced


def test_a_symlink_in_any_parent_directory_is_never_followed(ws, edit):
    (ws / "real").mkdir()
    (ws / "real" / "f.txt").write_text("inside file\n")
    (ws / "link_in").symlink_to("real")
    (ws / "link_out").symlink_to(ws.parent / "outside")
    (ws / "pkg" / "deep").symlink_to("../real")
    for path, old in (("link_in/f.txt", "inside"), ("link_out/secret.txt", "outside"), ("pkg/deep/f.txt", "inside")):
        assert_refused(edit(path, old, "X"), "symbolic link")
    assert (ws / "real" / "f.txt").read_text() == "inside file\n" and (ws.parent / "outside" / "secret.txt").read_text() == "outside original\n"


def test_the_workspace_may_itself_be_reached_through_a_symlink_but_only_its_resolved_tree_is_edited(tmp_path):
    real = tmp_path / "real_ws"
    real.mkdir()
    (real / "f.txt").write_text("x\n")
    link = tmp_path / "ws_link"
    link.symlink_to(real)
    tool = build_tools(["edit_file"], link)[0]  # resolve_workspace resolves it once, up front
    assert run(tool, path="f.txt", old_text="x", new_text="y").success and (real / "f.txt").read_text() == "y\n"


# ---- special files, encodings, limits --------------------------------------------------------------------------------------------
def test_directories_fifos_and_hard_links_are_refused(ws, edit):
    (ws / "adir").mkdir()
    os.mkfifo(ws / "fifo")
    (ws / "one.txt").write_text("shared\n")
    os.link(ws / "one.txt", ws / "two.txt")
    assert_refused(edit("adir", "a", "b"), "not a regular file")
    assert_refused(edit("fifo", "a", "b"), "not a regular file")  # and the open did not hang
    assert_refused(edit("one.txt", "shared", "X"), "hard links")
    assert (ws / "two.txt").read_text() == "shared\n"


def test_a_read_only_file_is_not_replaced_through_its_directory(ws, edit):
    f = ws / "ro.txt"
    f.write_text("fixed\n")
    f.chmod(0o444)
    assert_refused(edit("ro.txt", "fixed", "changed"), "read-only")
    assert f.read_text() == "fixed\n" and stat.S_IMODE(f.stat().st_mode) == 0o444


def test_binary_and_invalid_utf8_files_are_refused(ws, edit):
    (ws / "bin.dat").write_bytes(b"abc\x00def")
    (ws / "latin.txt").write_bytes("caf\xe9 au lait\n".encode("latin-1"))
    assert_refused(edit("bin.dat", "abc", "X"), "binary")
    assert_refused(edit("latin.txt", "au lait", "X"), "not valid UTF-8")
    assert (ws / "bin.dat").read_bytes() == b"abc\x00def" and (ws / "latin.txt").read_bytes() == "caf\xe9 au lait\n".encode("latin-1")


def test_invalid_text_arguments_are_refused(ws, edit):
    (ws / "a.txt").write_text("abc\n")
    assert edit("a.txt", "a\x00b", "X").success is False
    assert edit("a.txt", "abc", "X\x00").success is False
    assert_refused(edit("a.txt", "abc", "\ud800"), "not valid text")
    assert_refused(edit("a.txt", "abc", "abc"), "identical")
    assert (ws / "a.txt").read_text() == "abc\n"


def test_the_file_size_limit_applies_to_the_file_and_to_the_result(ws, monkeypatch):
    monkeypatch.setattr(EditFileTool, "MAX_FILE_BYTES", 100)
    tool = build_tools(["edit_file"], ws)[0]
    (ws / "big.txt").write_text("x" * 101)
    (ws / "ok.txt").write_text("A" + "y" * 89)
    assert_refused(run(tool, path="big.txt", old_text="x", new_text="z"), "larger than")
    assert_refused(run(tool, path="ok.txt", old_text="A", new_text="z" * 40), "exceed the size limit")
    assert (ws / "ok.txt").read_text() == "A" + "y" * 89


def test_git_internals_and_the_text_limits_are_refused(ws, edit):
    (ws / ".git").mkdir()
    (ws / ".git" / "config").write_text("[core]\n")
    assert_refused(edit(".git/config", "core", "x"), ".git")
    assert (ws / ".git" / "config").read_text() == "[core]\n"
    r = edit("pkg/a.py", "a" * 10_001, "b")
    assert r.success is False and "length <= 10000" in r.error  # effGen's own parameter validation answers first


# ---- 11, 12: empty replacement, line endings, whitespace -------------------------------------------------------------------------
def test_an_empty_replacement_deletes_the_text_but_an_omitted_one_is_an_error(ws, edit):
    f = ws / "a.txt"
    f.write_text("keep DELETE keep\n")
    assert edit("a.txt", " DELETE", "").success and f.read_text() == "keep keep\n"
    tool = build_tools(["edit_file"], ws)[0]
    r = run(tool, path="a.txt", old_text="keep")  # new_text missing: never an accidental deletion
    assert r.success is False and "new_text" in r.error and f.read_text() == "keep keep\n"
    assert run(tool, path="a.txt", old_text="keep", new_text="x", extra=1).success is False  # unknown parameters are rejected, not dropped


def test_crlf_files_keep_their_line_endings_and_no_normalisation_happens(ws, edit):
    f = ws / "w.txt"
    f.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    assert edit("w.txt", "two", "TWO").success and f.read_bytes() == b"one\r\nTWO\r\nthree\r\n"
    r = edit("w.txt", "one\nTWO", "x")  # LF in old_text does not silently match CRLF text
    assert_refused(r, "was not found", "CRLF")
    assert edit("w.txt", "one\r\nTWO", "1\r\n2").success and f.read_bytes() == b"1\r\n2\r\nthree\r\n"
    assert f.read_bytes().count(b"\r\n") == 3 and b"\n" not in f.read_bytes().replace(b"\r\n", b"")


def test_lf_files_bom_trailing_whitespace_tabs_and_a_missing_final_newline_are_preserved(ws, edit):
    f = ws / "u.txt"
    raw = b"\xef\xbb\xbfline one  \n\tindented\nlast"
    f.write_bytes(raw)
    assert edit("u.txt", "last", "last\nadded").success
    assert f.read_bytes() == raw.replace(b"last", b"last\nadded")
    assert edit("u.txt", "line one  ", "line one").success and f.read_bytes().startswith(b"\xef\xbb\xbfline one\n")
    assert edit("u.txt", "\tindented\n", "    indented\n").success and b"    indented\n" in f.read_bytes()


def test_whitespace_must_match_exactly(ws, edit):
    (ws / "s.py").write_text("def f():\n    return 1\n")
    assert_refused(edit("s.py", "def f():\n  return 1", "x"), "was not found")
    assert (ws / "s.py").read_text() == "def f():\n    return 1\n"


# ---- 13: concurrent modification ------------------------------------------------------------------------------------------------
def test_a_change_made_by_someone_else_during_the_edit_is_detected_and_not_overwritten(ws, edit, monkeypatch):
    f = ws / "pkg" / "c.txt"
    f.write_text("original text\n")
    real_fsync, fired = os.fsync, []

    def hostile_fsync(fd):
        if not fired:
            fired.append(1)
            f.write_text("somebody else wrote this first\n")  # between the read and the commit
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", hostile_fsync)
    r = edit("pkg/c.txt", "original", "edited")
    assert_refused(r, "changed by someone else")
    assert f.read_text() == "somebody else wrote this first\n"  # the newer content survives
    assert listing(ws) == ["pkg", "pkg/c.txt"]  # and the temporary file was removed


def test_a_file_swapped_for_an_identical_copy_during_the_edit_is_also_detected(ws, edit, monkeypatch):
    f = ws / "swap.txt"
    f.write_text("same bytes\n")
    real_fsync, fired = os.fsync, []

    def swap(fd):
        if not fired:
            fired.append(1)
            replacement = ws / "swap.new"
            replacement.write_text("same bytes\n")
            os.replace(replacement, f)  # another inode, identical content
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", swap)
    assert_refused(edit("swap.txt", "same", "other"), "changed by someone else")
    assert f.read_text() == "same bytes\n"


def test_a_failed_write_leaves_the_original_and_no_temporary_file(ws, edit, monkeypatch):
    f = ws / "w.txt"
    f.write_text("data\n")
    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError(5, "boom")))
    r = edit("w.txt", "data", "new")
    assert r.success is False and "EIO" in r.error and f.read_text() == "data\n" and listing(ws) == ["pkg", "w.txt"]


# ---- 14: registration and invocation through the real harness ---------------------------------------------------------------------
def test_edit_file_is_registered_known_to_the_config_fresh_per_call_and_marked_stateful(ws):
    assert "edit_file" in TOOL_NAMES and "edit_file" in KNOWN_TOOLS and TOOL_NAMES == KNOWN_TOOLS
    (a,), (b,) = build_tools(["edit_file"], ws), build_tools(["edit_file"], ws)
    assert a is not b and isinstance(a, EditFileTool) and getattr(a, STATEFUL_TOOL_MARKER) is True
    assert [t.name for t in build_tools(["file_operations", "edit_file"], ws)] == ["file_operations", "edit_file"]
    with pytest.raises(ToolBuildError, match="unknown tool"):
        build_tools(["edit_files"], ws)
    cfg = loads_config(f"workspace: {ws}\norchestrator: {{model: {{provider: ollama, model: x:1, context_length: 8192}}, tools: [edit_file]}}\n"
                       "workers: [{name: w, description: edits, model: {provider: ollama, model: x:1, context_length: 8192}, tools: [file_operations, edit_file]}]\n")
    assert cfg.workers[0].tools == ("file_operations", "edit_file")


def test_the_schema_tells_the_model_what_the_tool_does_and_requires(ws):
    meta = build_tools(["edit_file"], ws)[0].metadata
    assert meta.name == "edit_file"
    assert [(p.name, p.required) for p in meta.parameters] == [("path", True), ("old_text", True), ("new_text", True)]
    assert meta.examples == [{"path": "pkg/mod.py", "old_text": "    return a - b", "new_text": "    return a + b"}]  # shown to the model as a ready-made call
    for phrase in ("exactly once", "without rewriting the whole file", "relative to the project directory", "nothing is changed", "empty new_text deletes"):
        assert phrase in meta.description, phrase


def write_config(tmp_path: Path, ws: Path) -> Path:
    model = {"provider": "ollama", "model": BIG, "context_length": 8192}
    path = tmp_path / "harness.yaml"
    path.write_text(yaml.safe_dump({"workspace": str(ws), "orchestrator": {"model": model, "tools": ["file_operations"], "max_iterations": 6},
                                    "workers": [{"name": "editor", "description": "edits files", "model": {**model, "model": SMALL}, "tools": ["file_operations", "edit_file"],
                                                 "max_iterations": 6}]}))
    return path


def test_an_agent_discovers_and_uses_edit_file_through_the_real_harness_and_sees_each_result(tmp_path, ws, monkeypatch, capsys):
    (ws / "pkg" / "mod.py").write_text("def add(a, b):\n    return a - b\n\n\ndef other():\n    return 0\n")
    fake = FakeModels()
    fake.scripts[BIG] = [delegate("editor", "fix add in pkg/mod.py"), FINAL]
    fake.scripts[SMALL] = [
        call("edit_file", path="pkg/mod.py", old_text="return a - b", new_text="return a + b"),  # succeeds
        call("edit_file", path="pkg/mod.py", old_text="return a - b", new_text="return a + b"),  # the text is gone now: a refusal the model can read
        "Thought: done.\nFinal Answer: add now adds",
    ]
    monkeypatch.setattr(models_mod, "build_model", fake.build)
    assert cli.main(["run", "--config", str(write_config(tmp_path, ws)), "--task", "fix add"]) == 0
    capsys.readouterr()
    assert (ws / "pkg" / "mod.py").read_text() == "def add(a, b):\n    return a + b\n\n\ndef other():\n    return 0\n"
    prompts = [str(c.prompt) for c in fake.instances[SMALL].calls]
    assert "edit_file" in prompts[0] and "exactly once" in prompts[0]  # discoverable: the worker's first prompt carries the tool and its contract
    assert "edited pkg/mod.py: replaced 1 occurrence at line 2" in prompts[1]  # the success text reached the model
    assert "old_text was not found" in prompts[2] and "Nothing was changed" in prompts[2]  # and so did the refusal
    assert "edit_file" not in str(fake.instances[BIG].calls[0].prompt).split("Available tools")[-1].split("delegate")[0]  # the orchestrator was not given it


def test_two_different_successful_edits_in_a_row_do_not_trip_the_repeated_result_guard(tmp_path, ws, monkeypatch, capsys):
    (ws / "f.txt").write_text("one two\n")
    fake = FakeModels()
    fake.scripts[BIG] = [delegate("editor", "edit twice"), FINAL]
    fake.scripts[SMALL] = [call("edit_file", path="f.txt", old_text="one", new_text="1"), call("edit_file", path="f.txt", old_text="two", new_text="2"),
                           "Thought: done.\nFinal Answer: edited both"]
    monkeypatch.setattr(models_mod, "build_model", fake.build)
    assert cli.main(["run", "--config", str(write_config(tmp_path, ws)), "--task", "edit"]) == 0
    capsys.readouterr()
    assert (ws / "f.txt").read_text() == "1 2\n"


# ---- backslashes at effGen's input-parsing boundary (review P1-1) -------------------------------------------------------------
# A file read is shown to the model raw, so it sees single backslashes. effGen reads the call's JSON with json.loads and, when that
# fails (`\x` is not a JSON escape), with ast.literal_eval, which turns `\x1b` into a real ESC character. These tests go through the
# real harness and loop and record what the tool actually received.
ANSI_SRC = 're_ansi = re.compile(\n    r"""\n(?:\\x1b\\](.*?)\\x1b\\\\)|\n(?:\\x1b([(@-Z\\\\-_]|\\[[0-?]*[ -/]*[@-~]))\n""",\n    re.VERBOSE,\n)\n# ZZ_FILE_ONLY_TOKEN\n'
NAIVE_INPUT = r'{"path": "rich/ansi.py", "old_text": "(?:\x1b\](.*?)", "new_text": "(?:\x1b[0-?])|\n(?:\x1b\](.*?)"}'  # backslashes copied from the file as they read, in BOTH fields
NAIVE_OLD_INPUT = r'{"path": "rich/ansi.py", "old_text": "(?:\x1b\](.*?)", "new_text": "REPLACEMENT"}'  # only old_text is naive; new_text has no backslash
ESCAPED_INPUT = r'{"path": "rich/ansi.py", "old_text": "(?:\\x1b\\](.*?)", "new_text": "(?:\\x1b[0-?])|\n(?:\\x1b\\](.*?)"}'  # every file backslash doubled
OLD_ON_DISK, NEW_ON_DISK = r"(?:\x1b\](.*?)", "(?:\\x1b[0-?])|\n(?:\\x1b\\](.*?)"


def worker_edit(tmp_path, ws, monkeypatch, capsys, action_input):
    """One delegated edit_file call through the real CLI, harness and effGen loop. Returns (what the tool received, the worker's prompts)."""
    (ws / "rich").mkdir()
    (ws / "rich" / "ansi.py").write_bytes(ANSI_SRC.encode())
    (ws / "rich" / "other.py").write_text("untouched\n")
    fake = FakeModels()
    fake.scripts[BIG] = [delegate("editor", "fix rich/ansi.py"), FINAL]
    fake.scripts[SMALL] = [f"Thought: edit.\nAction: edit_file\nAction Input: {action_input}", "Thought: done.\nFinal Answer: finished"]
    monkeypatch.setattr(models_mod, "build_model", fake.build)
    received, original = [], EditFileTool._edit

    def spy(self, path, old_text, new_text):
        received.append((path, old_text, new_text))
        return original(self, path, old_text, new_text)

    monkeypatch.setattr(EditFileTool, "_edit", spy)
    assert cli.main(["run", "--config", str(write_config(tmp_path, ws)), "--task", "fix"]) == 0
    capsys.readouterr()
    return received, [str(c.prompt) for c in fake.instances[SMALL].calls]


def test_a_naively_copied_backslash_call_is_refused_with_an_explanation_and_changes_nothing(tmp_path, ws, monkeypatch, capsys):
    before = listing(ws)
    received, prompts = worker_edit(tmp_path, ws, monkeypatch, capsys, NAIVE_OLD_INPUT)
    # What the real parser handed to the tool: `\x1b` became a real ESC character (json.loads rejected the input, literal_eval accepted it).
    assert received == [("rich/ansi.py", "(?:\x1b\\](.*?)", "REPLACEMENT")]
    assert (ws / "rich" / "ansi.py").read_bytes() == ANSI_SRC.encode() and (ws / "rich" / "other.py").read_text() == "untouched\n"
    assert listing(ws) == sorted([*before, "rich", "rich/ansi.py", "rich/other.py"]) and not [p for p in listing(ws) if ".edit-" in p]
    observation = prompts[1].rsplit("Observation:", 1)[1]
    assert "old_text was not found" in observation and "Nothing was changed" in observation
    assert "ESC (U+001B)" in observation and "every backslash as two backslashes, in old_text and in new_text" in observation
    assert 'to match or write the text \\x1b send "\\\\x1b"' in observation  # the example the model is shown, literally
    assert "ZZ_FILE_ONLY_TOKEN" not in observation and "re_ansi" not in observation and len(observation) < 900  # no file content, bounded


def test_when_both_fields_are_naive_the_new_text_rule_answers_first(tmp_path, ws, monkeypatch, capsys):
    received, prompts = worker_edit(tmp_path, ws, monkeypatch, capsys, NAIVE_INPUT)
    assert received == [("rich/ansi.py", "(?:\x1b\\](.*?)", "(?:\x1b[0-?])|\n(?:\x1b\\](.*?)")]  # ESC in both fields
    assert (ws / "rich" / "ansi.py").read_bytes() == ANSI_SRC.encode()
    observation = prompts[1].rsplit("Observation:", 1)[1]
    assert "new_text contains the control character(s) ESC" in observation and "old_text was not found" not in observation  # arguments are checked before the file is read


def test_the_doubled_backslash_call_edits_the_file_exactly_through_the_same_path(tmp_path, ws, monkeypatch, capsys):
    received, prompts = worker_edit(tmp_path, ws, monkeypatch, capsys, ESCAPED_INPUT)
    assert received == [("rich/ansi.py", OLD_ON_DISK, NEW_ON_DISK)]  # literal backslashes, no control character
    expected = ANSI_SRC.replace(OLD_ON_DISK, NEW_ON_DISK, 1).encode()
    on_disk = (ws / "rich" / "ansi.py").read_bytes()
    assert on_disk == expected  # byte for byte
    assert b"\\x1b[0-?]" in on_disk and b"\x1b" not in on_disk  # the replacement's backslashes survived as literal backslashes; no ESC byte was written
    assert (ws / "rich" / "other.py").read_text() == "untouched\n" and sorted(listing(ws)) == ["pkg", "rich", "rich/ansi.py", "rich/other.py"]
    assert "edited rich/ansi.py: replaced 1 occurrence at line 3 (1 -> 2 lines)" in prompts[1].rsplit("Observation:", 1)[1]


def test_the_guidance_about_doubling_backslashes_is_in_the_tool_description_and_the_worker_prompt(ws):
    import sys as _sys

    _sys.path.insert(0, str(ROOT / "experiments" / "coding_eval"))
    import rich3278_pilot as pilot

    for text in (build_tools(["edit_file"], ws)[0].metadata.description, pilot.WORKER_PROMPT):
        assert "every backslash as two backslashes, in old_text and in new_text" in text and "\\x1b" in text and "\\\\x1b" in text
        assert "new_text may not contain control characters" in text or "new_text may not contain control" in text


@pytest.mark.parametrize("old,name", [("a\x08b", "BS (U+0008)"), ("a\x0cb", "FF (U+000C)"), ("a\x07b", "BEL (U+0007)"), ("a\x01b", "control (U+0001)")])
def test_the_diagnostic_names_the_control_characters_but_never_quotes_the_file(ws, edit, old, name):
    (ws / "f.txt").write_text("SECRET-FILE-CONTENT plain\n")
    r = edit("f.txt", old, "x")
    assert_refused(r, "was not found", name, "two backslashes")
    assert "SECRET-FILE-CONTENT" not in r.error and len(r.error) < 700


def test_allowed_whitespace_controls_get_no_escape_hint_and_a_nul_gets_one(ws, edit):
    (ws / "f.txt").write_text("plain\n")
    r = edit("f.txt", "tab\there\nnew\r", "x")
    assert "was not found" in r.error and "control character" not in r.error
    r = edit("f.txt", "a\x00b", "x")
    assert "NUL" in r.error and "two backslashes" in r.error


# ---- new_text: the same escaping mistake would be WRITTEN into the file (review follow-up) --------------------------------------------
# For each case below: ACTION INPUT = the exact text after `Action Input:`; RECEIVED = what effGen's parser handed to the tool;
# BYTES = the file afterwards; OUTCOME = success or refusal. Everything goes through the real CLI, harness and effGen dispatch.
NAIVE_NEW_INPUT = r'{"path": "rich/ansi.py", "old_text": "(?:\\x1b\\](.*?)", "new_text": "(?:\x1b[0-?])|\n(?:\\x1b\\](.*?)"}'  # old_text doubled, new_text's \x1b NOT


def test_a_correct_old_text_with_a_naive_new_text_is_refused_before_anything_is_written(tmp_path, ws, monkeypatch, capsys):
    before = listing(ws)
    received, prompts = worker_edit(tmp_path, ws, monkeypatch, capsys, NAIVE_NEW_INPUT)
    # RECEIVED: json.loads rejects the input (`\x` in new_text), ast.literal_eval accepts it; old_text keeps its literal backslashes and new_text holds a real ESC.
    assert received == [("rich/ansi.py", OLD_ON_DISK, "(?:\x1b[0-?])|\n(?:\\x1b\\](.*?)")]
    assert b"\x1b" in received[0][2].encode() and b"\x1b" not in received[0][1].encode()
    # BYTES: the file is exactly what it was; no other file changed or appeared, and no temporary file exists.
    assert (ws / "rich" / "ansi.py").read_bytes() == ANSI_SRC.encode() and (ws / "rich" / "other.py").read_bytes() == b"untouched\n"
    assert listing(ws) == sorted([*before, "rich", "rich/ansi.py", "rich/other.py"]) and not [p for p in listing(ws) if ".edit-" in p]
    # OUTCOME: refused, with a bounded message that names the character and shows the rule, without quoting the file.
    observation = prompts[1].rsplit("Observation:", 1)[1]
    assert "new_text contains the control character(s) ESC (U+001B), which edit_file never writes" in observation and "Nothing was changed" in observation
    assert 'to match or write the text \\x1b send "\\\\x1b"' in observation and "every backslash as two backslashes, in old_text and in new_text" in observation
    assert "ZZ_FILE_ONLY_TOKEN" not in observation and "re_ansi" not in observation and "old_text was not found" not in observation and len(observation) < 900


@pytest.mark.parametrize("json_new_text,name", [(r"a\bz", "BS (U+0008)"), (r"a\fz", "FF (U+000C)"), (r"a\u001bz", "ESC (U+001B)"), (r"a\u0007z", "BEL (U+0007)")])
def test_control_characters_that_valid_json_escapes_produce_are_refused_too(tmp_path, ws, monkeypatch, capsys, json_new_text, name):
    # These are VALID JSON escapes: json.loads itself yields the control character (no literal_eval fallback involved).
    action = '{"path": "rich/ansi.py", "old_text": "(?:\\\\x1b\\\\](.*?)", "new_text": "' + json_new_text + '"}'
    received, prompts = worker_edit(tmp_path, ws, monkeypatch, capsys, action)
    assert received == [("rich/ansi.py", OLD_ON_DISK, json.loads('"' + json_new_text + '"'))] and ord(received[0][2][1]) < 32
    assert (ws / "rich" / "ansi.py").read_bytes() == ANSI_SRC.encode() and not [p for p in listing(ws) if ".edit-" in p]
    observation = prompts[1].rsplit("Observation:", 1)[1]
    assert name in observation and "never writes" in observation and "Nothing was changed" in observation and len(observation) < 900


def test_tab_lf_and_cr_in_new_text_are_written_exactly(tmp_path, ws, monkeypatch, capsys):
    # ACTION INPUT is valid JSON: \t \n \r are real JSON escapes. RECEIVED holds a TAB, an LF, a lone CR and a CRLF. OUTCOME: success.
    action = r'{"path": "rich/ansi.py", "old_text": "(?:\\x1b\\](.*?)", "new_text": "T\tA\nB\rC\r\nD"}'
    received, prompts = worker_edit(tmp_path, ws, monkeypatch, capsys, action)
    assert received == [("rich/ansi.py", OLD_ON_DISK, "T\tA\nB\rC\r\nD")]
    assert (ws / "rich" / "ansi.py").read_bytes() == ANSI_SRC.replace(OLD_ON_DISK, "T\tA\nB\rC\r\nD", 1).encode()  # exact bytes, nothing normalised
    assert "edited rich/ansi.py: replaced 1 occurrence" in prompts[1].rsplit("Observation:", 1)[1]


def test_a_doubled_backslash_before_t_survives_as_two_literal_characters(tmp_path, ws, monkeypatch, capsys):
    action = r'{"path": "rich/ansi.py", "old_text": "(?:\\x1b\\](.*?)", "new_text": "a\\tb\\n"}'  # JSON: backslash, t ... backslash, n
    received, _ = worker_edit(tmp_path, ws, monkeypatch, capsys, action)
    assert received == [("rich/ansi.py", OLD_ON_DISK, "a\\tb\\n")]
    on_disk = (ws / "rich" / "ansi.py").read_bytes()
    assert on_disk == ANSI_SRC.replace(OLD_ON_DISK, "a\\tb\\n", 1).encode() and b"a\\tb\\n" in on_disk and b"a\tb" not in on_disk


# Known limitations, pinned so that they cannot be mistaken for guarantees: escapes that decode to ORDINARY characters are indistinguishable from
# intended text once parsed, so edit_file cannot detect them. Control-character refusal does not eliminate serialization corruption.
def test_known_limitation_an_escape_that_becomes_an_ordinary_character_is_written_silently(tmp_path, ws, monkeypatch, capsys):
    action = r'{"path": "rich/ansi.py", "old_text": "(?:\\x1b\\](.*?)", "new_text": "\x41"}'  # the model meant the four characters \x41
    received, prompts = worker_edit(tmp_path, ws, monkeypatch, capsys, action)
    assert received == [("rich/ansi.py", OLD_ON_DISK, "A")]  # literal_eval turned \x41 into A
    assert (ws / "rich" / "ansi.py").read_bytes() == ANSI_SRC.replace(OLD_ON_DISK, "A", 1).encode()  # and it was written without complaint


def test_known_limitation_a_single_backslash_t_or_n_becomes_whitespace_silently(tmp_path, ws, monkeypatch, capsys):
    action = r'{"path": "rich/ansi.py", "old_text": "(?:\\x1b\\](.*?)", "new_text": "a\tb\nc"}'  # valid JSON: TAB and LF, although a regex may have been meant
    received, _ = worker_edit(tmp_path, ws, monkeypatch, capsys, action)
    assert received == [("rich/ansi.py", OLD_ON_DISK, "a\tb\nc")]
    assert (ws / "rich" / "ansi.py").read_bytes() == ANSI_SRC.replace(OLD_ON_DISK, "a\tb\nc", 1).encode()


@pytest.mark.parametrize("new,name", [("a\x1bb", "ESC (U+001B)"), ("a\x08b", "BS (U+0008)"), ("a\x0cb", "FF (U+000C)"), ("a\x07b", "BEL (U+0007)"), ("a\x0bb", "VT (U+000B)"),
                                      ("a\x01b", "control (U+0001)"), ("\x1b\x08\x07\x0c", "BEL (U+0007), BS (U+0008), FF (U+000C)")])
def test_new_text_control_characters_are_refused_with_a_bounded_diagnostic_and_the_file_is_untouched(ws, edit, new, name):
    (ws / "f.txt").write_text("SECRET-FILE-CONTENT target\n")
    before = listing(ws)
    r = edit("f.txt", "target", new)
    assert_refused(r, "new_text contains the control character(s)", name, "never writes", "in old_text and in new_text")
    assert "SECRET-FILE-CONTENT" not in r.error and len(r.error) < 700
    assert ("ESC (U+001B)" in r.error) == ("ESC (U+001B)" in name)  # at most three characters are named: the fourth of the last case is not
    assert (ws / "f.txt").read_bytes() == b"SECRET-FILE-CONTENT target\n" and listing(ws) == before


def test_the_new_text_check_comes_before_the_filesystem_like_the_other_argument_checks(ws, edit):
    r = edit("no/such/file.txt", "a", "x\x1b")  # the file does not exist, but the argument is refused first
    assert "new_text contains the control character(s) ESC" in r.error and "does not exist" not in r.error
    r = edit("f.txt", "a", "x\x00y")  # NUL keeps its own message
    assert "new_text must be text without NUL characters" in r.error and "two backslashes" in r.error


def test_tab_lf_and_cr_in_new_text_are_accepted_by_the_tool_directly(ws, edit):
    (ws / "f.txt").write_bytes(b"one TARGET two\n")
    assert edit("f.txt", "TARGET", "a\tb\r\nc\rd\n").success
    assert (ws / "f.txt").read_bytes() == b"one a\tb\r\nc\rd\n two\n" and listing(ws) == ["f.txt", "pkg"]


# ---- the real file from the dataset, if it is present -----------------------------------------------------------------------------
@pytest.mark.skipif(not (ROOT / "dataset" / "snapshots" / "rich_3278.tgz").exists() or shutil.which("patch") is None, reason="needs the local dataset and GNU patch")
def test_the_gold_change_to_rich_ansi_py_is_a_small_edit_not_a_rewrite(tmp_path):
    with tarfile.open(ROOT / "dataset" / "snapshots" / "rich_3278.tgz") as tf:
        original = tf.extractfile("./rich/ansi.py").read()
    gold = next(json.loads(line)["patch"] for line in (ROOT / "dataset" / "tasks.jsonl").read_text().splitlines() if '"rich_3278"' in line) + "\n"
    expected, edited = tmp_path / "expected", tmp_path / "edited"
    for d in (expected, edited):
        (d / "rich").mkdir(parents=True)
        (d / "rich" / "ansi.py").write_bytes(original)
    subprocess.run(["patch", "-p1", "--fuzz=0", "--batch"], input=gold, text=True, cwd=expected, check=True, capture_output=True)
    call_args = {"path": "rich/ansi.py", "old_text": 'r"""\n(?:\\x1b\\](.*?)', "new_text": 'r"""\n(?:\\x1b[0-?])|\n(?:\\x1b\\](.*?)'}
    assert run(build_tools(["edit_file"], edited)[0], **call_args).success
    assert (edited / "rich" / "ansi.py").read_bytes() == (expected / "rich" / "ansi.py").read_bytes()  # byte-identical to the reference fix
    assert len(json.dumps(call_args)) < 300 < len(original) / 20  # a few hundred characters instead of a 6,900 character rewrite
