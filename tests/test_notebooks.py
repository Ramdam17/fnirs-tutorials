"""Execute every notebook of the table of contents of ``myst.yml``, headless.

Run with ``uv run python -m pytest tests/test_notebooks.py -v``. Each notebook is one
test; a failing notebook does not stop the others from being executed and timed. The
executed copies are written under ``_build/executed/`` at the notebook's path, and the
result of every notebook, a failure included, is recorded in
``build-reports/notebook-timings.json``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from notebook_runner import (
    STATUS_PASSED,
    execute_notebook,
    list_notebooks,
    load_timeouts,
    record_result,
    reset_timings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MYST_PATH = REPO_ROOT / "myst.yml"
CONFIG_PATH = REPO_ROOT / "config" / "notebooks.yaml"
EXECUTED_DIR = REPO_ROOT / "_build" / "executed"
TIMINGS_PATH = REPO_ROOT / "build-reports" / "notebook-timings.json"

# Evaluated at import, so a wrong table of contents or configuration stops the
# collection with a clear error instead of skipping a notebook.
NOTEBOOKS = list_notebooks(MYST_PATH)
TIMEOUTS = load_timeouts(CONFIG_PATH, NOTEBOOKS)


@pytest.fixture(scope="session", autouse=True)
def _empty_timings() -> None:
    """Empty the timings file once, so it never holds the result of an earlier run."""
    reset_timings(TIMINGS_PATH)


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_notebook_executes(notebook: str) -> None:
    """The notebook runs without error inside its timeout."""
    result = execute_notebook(
        REPO_ROOT / notebook,
        EXECUTED_DIR / notebook,
        TIMEOUTS.timeout_for(notebook),
    )
    record_result(TIMINGS_PATH, notebook, result)
    assert result.status == STATUS_PASSED, (
        f"{notebook}: {result.status} after {result.wall_time_s:.1f} s "
        f"(budget {result.timeout_s:g} s): {result.error}"
    )
