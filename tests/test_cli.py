"""T5.1: the command line (`smol-harness run --config FILE --task TEXT [--json]`).

In-process tests call `main(argv)` with the real harness, real tools and real effGen agents; only `build_model` is replaced
by scripted models (no downloads, no server). The subprocess tests start a fresh interpreter, so nothing the pytest session
installed (conftest's effGen patch fixture) can stand in for the CLI's own initialisation.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from effgen.models.errors import BackendUnreachableError
from test_harness_config import BIG, FINAL, SMALL, TINY, FakeModels, TrackedModel, call, delegate

import smol_harness.__main__ as cli
import smol_harness.models as models_mod
from smol_harness import harness as harness_mod
from smol_harness.harness import Harness, effgen_loop_compat_installed
from smol_harness.models import ModelLoadError

REPO = Path(__file__).resolve().parents[1]
OK_FLOW = {BIG: [delegate("writer", "create hello.txt"), FINAL],
           SMALL: [call("file_operations", operation="write", path="hello.txt", content="hello from the cli\n"),
                   "Thought: ok.\nFinal Answer: wrote hello.txt"]}


class CliModels(FakeModels):
    """FakeModels plus per-model constructors for models that must misbehave."""

    def __init__(self) -> None:
        super().__init__()
        self.custom: dict[str, object] = {}

    def build(self, spec):
        if spec.model in self.custom:
            self.loads.append(spec)
            model = self.custom[spec.model]()
            self.instances[spec.model] = model
            return model
        return super().build(spec)


@pytest.fixture
def models(monkeypatch):
    fake = CliModels()
    monkeypatch.setattr(models_mod, "build_model", fake.build)
    return fake


def write_config(tmp_path: Path, *, base_url: str | None = None, extra_workers=(), **limits) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    orchestrator_model = {"provider": "ollama", "model": BIG, "context_length": 8192}
    if base_url:
        orchestrator_model["base_url"] = base_url
    data = {
        "workspace": str(ws),
        "orchestrator": {"model": orchestrator_model, "tools": ["file_operations"], "max_iterations": 6},
        "workers": [{"name": "writer", "description": "writes files", "tools": ["file_operations"], "max_iterations": 4,
                     "model": {"provider": "ollama", "model": SMALL, "context_length": 4096}}, *extra_workers],
    }
    if limits:
        data["limits"] = limits
    path = tmp_path / "harness.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def cli_run(tmp_path, *extra, task="make hello.txt", config=None, capsys=None):
    argv = ["run", "--config", str(config or write_config(tmp_path)), "--task", task, *extra]
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ---- argument parsing ------------------------------------------------------------------------------------------------------
def test_run_arguments_are_parsed():
    args = cli.build_parser().parse_args(["run", "--config", "c.yaml", "--task", "do it"])
    assert (args.command, args.config, args.task, args.json) == ("run", "c.yaml", "do it", False)
    assert cli.build_parser().parse_args(["run", "--config", "c.yaml", "--task", "t", "--json"]).json is True


@pytest.mark.parametrize("argv", [[], ["run"], ["run", "--config", "c.yaml"], ["run", "--task", "t"], ["frobnicate"],
                                  ["run", "--config", "c.yaml", "--task", "t", "--bogus"]])
def test_usage_errors_exit_2_with_usage_on_stderr_and_nothing_on_stdout(argv, capsys):
    assert cli.main(argv) == cli.EXIT_USAGE == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "usage: smol-harness" in captured.err


@pytest.mark.parametrize("task", ["", "   ", "\n\t"])
def test_a_blank_task_is_a_usage_error_and_loads_nothing(tmp_path, models, capsys, task):
    code, out, err = cli_run(tmp_path, task=task, capsys=capsys)
    assert code == 2 and out == "" and "--task must not be empty" in err
    assert models.loads == []


def test_help_exits_0_on_stdout(capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "run" in out and "Exit codes" in out and "trusted repositories" in out
    assert cli.main(["run", "--help"]) == 0 and "--config" in capsys.readouterr().out


# ---- configuration problems ------------------------------------------------------------------------------------------------------
def test_a_missing_config_file_is_a_clean_startup_failure(tmp_path, models, capsys):
    code, out, err = cli_run(tmp_path, config=tmp_path / "nope.yaml", capsys=capsys)
    assert code == 3 and out == ""
    assert "cannot read config file" in err and "nope.yaml" in err and "Traceback" not in err
    assert models.loads == []


def test_an_invalid_config_names_the_offending_path_and_loads_nothing(tmp_path, models, capsys):
    path = write_config(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["workers"][0]["model"]["quantization"] = "bnb-4bit-fp4"
    data["workers"][0]["model"]["provider"] = "hf"
    path.write_text(yaml.safe_dump(data))
    code, out, err = cli_run(tmp_path, config=path, capsys=capsys)
    assert code == 3 and out == "" and "workers[0].model" in err and "ConfigError" in err and "Traceback" not in err
    assert models.loads == []


def test_malformed_yaml_is_a_clean_startup_failure(tmp_path, models, capsys):
    path = tmp_path / "broken.yaml"
    path.write_text("workspace: [unclosed\n")
    code, out, err = cli_run(tmp_path, config=path, capsys=capsys)
    assert code == 3 and out == "" and "invalid YAML" in err


def test_a_missing_workspace_fails_before_any_model_is_loaded(tmp_path, models, capsys):
    path = write_config(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["workspace"] = str(tmp_path / "does-not-exist")
    path.write_text(yaml.safe_dump(data))
    code, out, err = cli_run(tmp_path, config=path, capsys=capsys)
    assert code == 3 and "does not exist" in err and "ToolBuildError" in err
    assert models.loads == []


# ---- success ------------------------------------------------------------------------------------------------------------------------
def test_a_successful_run_prints_only_the_answer_on_stdout_and_exits_0(tmp_path, models, capsys):
    models.scripts.update(OK_FLOW)
    code, out, err = cli_run(tmp_path, capsys=capsys)
    assert code == 0
    assert out == "all done\n"  # the orchestrator's final answer, nothing else
    assert (tmp_path / "ws" / "hello.txt").read_text() == "hello from the cli\n"  # the real tool ran in the workspace
    assert "delegations: 1" in err and "d1 writer: completed (iterations" in err and "tool calls 1)" in err
    assert "wrote hello.txt" not in err and "hello from the cli" not in err  # reports and file content are not dumped


def test_a_successful_run_releases_every_model_it_loaded_exactly_once(tmp_path, models, capsys):
    models.scripts.update(OK_FLOW)
    cli_run(tmp_path, capsys=capsys)
    assert {name: m.unloads for name, m in models.instances.items()} == {BIG: 1, SMALL: 1}


def test_json_mode_prints_one_document_with_the_answer_and_worker_records(tmp_path, models, capsys):
    models.scripts.update(OK_FLOW)
    code, out, err = cli_run(tmp_path, "--json", capsys=capsys)
    doc = json.loads(out)  # stdout is exactly one JSON document
    assert code == 0 and doc["success"] is True and doc["output"] == "all done" and doc["termination"] == "done"
    assert doc["stop_reason"] == "final_answer" and doc["error"] is None
    (worker,) = doc["workers"]
    assert (worker["worker"], worker["delegation_id"], worker["status"], worker["error"]) == ("writer", "d1", "completed", None)
    assert worker["output"] == "wrote hello.txt" and worker["tool_calls"] == 1 and worker["iterations"] >= 1
    assert "delegations: 1" in err  # the human summary still goes to stderr


# ---- the task fails ------------------------------------------------------------------------------------------------------------------
def stuck_flow(models, tmp_path):
    (tmp_path / "ws").mkdir(exist_ok=True)
    (tmp_path / "ws" / "a.txt").write_text("x\n")
    models.scripts[BIG] = [delegate("writer", "create hello.txt")] + [call("file_operations", operation="read", path="a.txt")] * 8
    models.scripts[SMALL] = OK_FLOW[SMALL]


def test_a_failed_result_exits_1_with_the_diagnosis_on_stderr_and_nothing_on_stdout(tmp_path, models, capsys):
    stuck_flow(models, tmp_path)
    code, out, err = cli_run(tmp_path, capsys=capsys)
    assert code == 1 and out == ""
    assert "the task failed (termination=stuck, stop_reason=" in err
    assert "delegations: 1" in err and "d1 writer: completed" in err  # the worker records survive the failure
    assert {m.unloads for m in models.instances.values()} == {1}  # and the models are released


def test_a_failed_result_in_json_mode_is_a_document_with_success_false(tmp_path, models, capsys):
    stuck_flow(models, tmp_path)
    code, out, err = cli_run(tmp_path, "--json", capsys=capsys)
    doc = json.loads(out)
    assert code == 1 and doc["success"] is False and doc["termination"] == "stuck" and doc["error"] is None
    assert doc["stop_reason"] and [w["status"] for w in doc["workers"]] == ["completed"]


def test_a_failed_worker_is_summarised_with_its_error(tmp_path, models, capsys):
    models.scripts[BIG] = [delegate("writer", "create hello.txt"), FINAL]
    models.custom[SMALL] = lambda: type("Down", (TrackedModel,), {"generate": lambda self, *a, **k: (_ for _ in ()).throw(
        RuntimeError("worker model is down"))})(["x"], name=SMALL)
    code, out, err = cli_run(tmp_path, capsys=capsys)
    assert code == 0 and out == "all done\n"  # a failed worker does not fail the task: the orchestrator decides
    assert "d1 writer: failed" in err and "worker model is down" in err


# ---- exceptions that propagate -------------------------------------------------------------------------------------------------------
class OrchestratorGoesDown(TrackedModel):
    """Delegates once, then the backend disappears (effGen raises this whatever raise_on_error says)."""

    def generate(self, prompt, config=None, **kw):
        if self.calls:
            raise BackendUnreachableError("openai_compatible", BIG, "connection refused", "http://127.0.0.1:11434/v1")
        return super().generate(prompt, config, **kw)


def test_a_propagated_exception_exits_4_and_shows_the_worker_records(tmp_path, models, capsys):
    models.scripts.update(OK_FLOW)
    models.custom[BIG] = lambda: OrchestratorGoesDown([delegate("writer", "create hello.txt")], name=BIG)
    code, out, err = cli_run(tmp_path, capsys=capsys)
    assert code == 4 and out == ""
    assert "the run raised: BackendUnreachableError" in err and "connection refused" in err and "Traceback" not in err
    assert "delegations: 1" in err and "d1 writer: completed" in err  # recovered from exc.worker_records
    assert (tmp_path / "ws" / "hello.txt").exists()  # the work that was done stays done
    assert {m.unloads for m in models.instances.values()} == {1}


def test_a_propagated_exception_in_json_mode_is_a_document_with_the_error_and_records(tmp_path, models, capsys):
    models.scripts.update(OK_FLOW)
    models.custom[BIG] = lambda: OrchestratorGoesDown([delegate("writer", "create hello.txt")], name=BIG)
    code, out, _ = cli_run(tmp_path, "--json", capsys=capsys)
    doc = json.loads(out)
    assert code == 4 and doc["success"] is False and doc["error"]["stage"] == "run"
    assert doc["error"]["type"] == "BackendUnreachableError" and [w["worker"] for w in doc["workers"]] == ["writer"]


def test_an_unexpected_error_is_never_reported_as_success_and_shows_no_traceback(tmp_path, models, monkeypatch, capsys):
    monkeypatch.setattr(Harness, "run", lambda self, task: 1 / 0)
    code, out, err = cli_run(tmp_path, capsys=capsys)
    assert code == 4 and out == "" and "ZeroDivisionError: division by zero" in err and "Traceback" not in err
    assert {m.unloads for m in models.instances.values()} == {1}  # still closed


def test_ctrl_c_exits_130_after_releasing_the_models(tmp_path, models, monkeypatch, capsys):
    def interrupted(self, task):
        raise KeyboardInterrupt

    monkeypatch.setattr(Harness, "run", interrupted)
    code, out, err = cli_run(tmp_path, capsys=capsys)
    assert code == 130 and out == "" and "interrupted" in err
    assert {m.unloads for m in models.instances.values()} == {1}


def test_a_failure_while_releasing_models_is_a_warning_and_does_not_change_the_outcome(tmp_path, models, capsys):
    models.scripts.update(OK_FLOW)
    models.unload_error[BIG] = "unload exploded"
    code, out, err = cli_run(tmp_path, capsys=capsys)
    assert code == 0 and out == "all done\n"  # the task did succeed
    assert "warning: releasing resources failed" in err  # but the problem is not hidden
    assert models.instances[SMALL].unloads == 1


# ---- start-up failures after some work was done ----------------------------------------------------------------------------------------
def test_a_model_that_cannot_load_is_a_startup_failure_and_releases_the_others(tmp_path, models, capsys):
    models.failing.add(TINY)
    hf = {"name": "thinker", "description": "thinks", "tools": [], "max_iterations": 2,
          "model": {"provider": "hf", "model": TINY, "quantization": "none", "device": "cpu", "dtype": "fp32"}}
    code, out, err = cli_run(tmp_path, config=write_config(tmp_path, extra_workers=[hf]), capsys=capsys)
    assert code == 3 and out == "" and "ModelLoadError" in err and TINY in err
    assert {m.unloads for m in models.instances.values()} == {1}  # BIG and SMALL, loaded before the failure, were released


def test_an_unavailable_shell_is_a_startup_failure_before_any_model_loads(tmp_path, models, monkeypatch, capsys):
    shelled = {"name": "runner", "description": "runs", "tools": ["shell"], "max_iterations": 2,
               "model": {"provider": "ollama", "model": SMALL, "context_length": 4096}}
    # Any shell backend needs its host binary (bash on Linux, powershell.exe/pwsh on Windows); make
    # them all appear missing, whichever platform the test runs on.
    real = shutil.which
    monkeypatch.setattr(shutil, "which",
                        lambda name, *a, **k: None if name in ("bash", "powershell", "pwsh") else real(name, *a, **k))
    code, out, err = cli_run(tmp_path, config=write_config(tmp_path, extra_workers=[shelled]), capsys=capsys)
    assert code == 3 and "tool 'shell' needs" in err and models.loads == []


# ---- the compatibility patch is initialised explicitly by the CLI --------------------------------------------------------------------------
def test_the_cli_initialises_the_runtime_before_loading_config_or_models(tmp_path, models, monkeypatch, capsys):
    order: list[str] = []
    real_init, real_load = cli.initialize_runtime, cli.load_config
    monkeypatch.setattr(cli, "initialize_runtime", lambda: (order.append("initialize"), real_init())[1])
    monkeypatch.setattr(cli, "load_config", lambda path: (order.append("load_config"), real_load(path))[1])
    real_build = models.build
    models.build = lambda spec: (order.append("model"), real_build(spec))[1]
    monkeypatch.setattr(models_mod, "build_model", models.build)
    models.scripts.update(OK_FLOW)
    assert cli_run(tmp_path, capsys=capsys)[0] == 0
    assert order[:2] == ["initialize", "load_config"] and order.count("initialize") == 1 and "model" in order[2:]


def test_the_cli_installs_the_patch_itself_and_leaves_it_installed(tmp_path, models, compat_uninstalled, capsys):
    assert not effgen_loop_compat_installed()
    models.scripts.update(OK_FLOW)
    assert cli_run(tmp_path, capsys=capsys)[0] == 0  # would be EffgenCompatError without the CLI's own initialisation
    assert effgen_loop_compat_installed()


def test_an_incompatible_effgen_stops_the_cli_before_the_config_is_read(tmp_path, models, monkeypatch, capsys):
    def refuse():
        raise harness_mod.EffgenCompatError("effgen is '9.9.9', the patch was written for '1.3.0'. Nothing was patched")

    loaded = []
    monkeypatch.setattr(cli, "initialize_runtime", refuse)
    monkeypatch.setattr(cli, "load_config", lambda path: loaded.append(path))
    code, out, err = cli_run(tmp_path, capsys=capsys)
    assert code == 3 and out == "" and "EffgenCompatError" in err and "9.9.9" in err
    assert loaded == [] and models.loads == []


# ---- nothing sensitive is printed ------------------------------------------------------------------------------------------------------------
def test_credentials_in_urls_and_environment_secrets_never_reach_the_output(tmp_path, models, monkeypatch, capsys):
    monkeypatch.setenv("SMOL_TEST_SECRET", "env-secret-value-12345")

    def failing_load(spec):  # the real error text: build_model describes the whole spec, base_url included (models.py)
        raise ModelLoadError(f"failed to load {models_mod._describe(spec)}: ConnectionError: refused (SMOL_TEST_SECRET=env-secret-value-12345)")

    monkeypatch.setattr(models_mod, "build_model", failing_load)
    path = write_config(tmp_path, base_url="http://alice:hunter2-password@127.0.0.1:11434/v1")
    for flags in ((), ("--json",)):
        code, out, err = cli_run(tmp_path, *flags, config=path, capsys=capsys)
        assert code == 3 and "failed to load model spec" in err and "127.0.0.1:11434" in err  # the diagnosis is still useful...
        for text in (out, err):  # ...without the credentials
            assert "hunter2-password" not in text and "env-secret-value-12345" not in text and "alice:" not in text


@pytest.mark.parametrize("raw", ["http://u:pw-123@host/v1", "Authorization: Bearer abcdef0123456789xyz", "api_key=sk-live-999", "password: hunter2",
                                 "token=abc123def456",
                                 "OPENAI_API_KEY=sk-abc123XYZ", "HF_TOKEN=hf_abc12345", "SMOL_TEST_SECRET=env-secret-value-12345", "db_password=hunter2"])
def test_the_scrubber_masks_common_credential_shapes(raw):
    cleaned = cli._scrub(f"failed: {raw} and more")
    assert "***" in cleaned and not any(s in cleaned for s in ("pw-123", "abcdef0123456789xyz", "sk-live-999", "hunter2", "abc123def456", "sk-abc123XYZ",
                                                                  "hf_abc12345", "env-secret-value-12345"))


def test_long_messages_and_worker_reports_are_truncated(tmp_path, models, monkeypatch, capsys):
    models.scripts[BIG] = [delegate("writer", "write a lot"), FINAL]
    models.scripts[SMALL] = ["Thought: ok.\nFinal Answer: " + "L" * 3000]
    code, out, err = cli_run(tmp_path, "--json", capsys=capsys)
    doc = json.loads(out)
    assert code == 0 and len(doc["workers"][0]["output"]) < 700 and "more chars" in doc["workers"][0]["output"]
    assert "L" * 600 not in err
    monkeypatch.setattr(Harness, "run", lambda self, task: (_ for _ in ()).throw(RuntimeError("E" * 5000)))
    code, out, err = cli_run(tmp_path, capsys=capsys)
    assert code == 4 and "E" * 600 not in err and "more chars" in err


# ---- real processes: a fresh interpreter, no pytest fixtures --------------------------------------------------------------------------------
DRIVER = r"""
import json, sys
sys.path.insert(0, {tests!r})
from conftest import ScriptedModel          # only the scripted model; the session fixtures of conftest never run here
import smol_harness.models as models_mod
from smol_harness import harness as h
unloads = []
class Tracked(ScriptedModel):
    def unload(self): unloads.append(self.model_name)
models_mod.build_model = lambda spec: Tracked({scripts!r}[spec.model], name=spec.model)
state = {{"installed_before": h.effgen_loop_compat_installed()}}
from smol_harness.__main__ import main
code = main({argv!r})
state.update(installed_after=h.effgen_loop_compat_installed(), unloaded=sorted(unloads), code=code)
sys.stderr.write("STATE" + json.dumps(state) + "\n")
sys.exit(code)
"""


def run_driver(tmp_path, scripts, argv):
    code = DRIVER.format(tests=str(REPO / "tests"), scripts=scripts, argv=argv)
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=tmp_path)
    state_line = next(line for line in done.stderr.splitlines() if line.startswith("STATE"))
    return done, json.loads(state_line[len("STATE"):])


def test_a_fresh_process_run_initialises_by_itself_and_succeeds(tmp_path):
    config = write_config(tmp_path)
    done, state = run_driver(tmp_path, OK_FLOW, ["run", "--config", str(config), "--task", "make hello.txt"])
    assert state["installed_before"] is False  # nothing installed the patch before the CLI did
    assert done.returncode == 0 and done.stdout == "all done\n", done.stderr[-1500:]
    assert state["installed_after"] is True and state["unloaded"] == [BIG, SMALL]
    assert (tmp_path / "ws" / "hello.txt").read_text() == "hello from the cli\n"


def test_a_fresh_process_failed_task_exits_1(tmp_path):
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "a.txt").write_text("x\n")
    scripts = {BIG: [delegate("writer", "x")] + [call("file_operations", operation="read", path="a.txt")] * 8, SMALL: OK_FLOW[SMALL]}
    done, state = run_driver(tmp_path, scripts, ["run", "--config", str(write_config(tmp_path)), "--task", "go", "--json"])
    assert done.returncode == 1 and state["code"] == 1 and json.loads(done.stdout)["success"] is False
    assert state["unloaded"] == [BIG, SMALL]


def test_python_dash_m_reports_a_missing_config_cleanly(tmp_path):
    done = subprocess.run([sys.executable, "-I", "-m", "smol_harness", "run", "--config", str(tmp_path / "missing.yaml"), "--task", "t"],
                          capture_output=True, text=True, cwd=tmp_path)
    assert done.returncode == 3 and done.stdout == "" and "cannot read config file" in done.stderr and "Traceback" not in done.stderr


def test_python_dash_m_usage_error_exits_2(tmp_path):
    done = subprocess.run([sys.executable, "-I", "-m", "smol_harness", "run", "--task", "t"], capture_output=True, text=True, cwd=tmp_path)
    assert done.returncode == 2 and done.stdout == "" and "--config" in done.stderr


CONSOLE_SCRIPT = Path(sys.executable).parent / "smol-harness"


@pytest.mark.skipif(not CONSOLE_SCRIPT.exists(), reason="the smol-harness console script is installed by `uv sync`")
def test_the_installed_console_script_works(tmp_path):
    done = subprocess.run([str(CONSOLE_SCRIPT), "run", "--config", str(tmp_path / "missing.yaml"), "--task", "t"],
                          capture_output=True, text=True, cwd=tmp_path, env={k: v for k, v in os.environ.items()})
    assert done.returncode == 3 and "cannot read config file" in done.stderr
    helped = subprocess.run([str(CONSOLE_SCRIPT), "--help"], capture_output=True, text=True, cwd=tmp_path)
    assert helped.returncode == 0 and "COMMAND" in helped.stdout


def test_the_project_declares_the_console_script():
    import tomllib

    scripts = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts == {"smol-harness": "smol_harness.__main__:main"}
    assert callable(cli.main)
