"""Real-model tests must be impossible to run by accident (no download, no load)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _pytest(*args: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "SMOL_RUN_REAL_MODELS"}
    env.update(extra_env or {})
    return subprocess.run([sys.executable, "-m", "pytest", *args, "-q", "-p", "no:cacheprovider"],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)


def test_real_model_tests_are_skipped_without_the_flag():
    run = _pytest("tests/integration/test_liquid_cpu.py", "-rs")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "1 skipped" in run.stdout and "--run-real-models" in run.stdout
    assert "passed" not in run.stdout and "failed" not in run.stdout


def test_the_marker_alone_does_not_opt_in():
    run = _pytest("tests/integration/test_liquid_cpu.py", "-m", "real_model")
    assert run.returncode == 0 and "1 skipped" in run.stdout
