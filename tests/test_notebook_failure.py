"""Tests of the notebook runner: failure fixtures and the parts that need no kernel.

The two fixtures run in a real kernel and write to temporary folders, never to
``_build/executed/`` or ``build-reports/``. The other tests use temporary
``myst.yml`` and configuration files, never the real ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_notebook
from notebook_runner import (
    MIN_CELL_TIMEOUT_S,
    STATUS_ERROR,
    STATUS_PASSED,
    STATUS_TIMEOUT,
    NotebookResult,
    execute_notebook,
    list_notebooks,
    load_timeouts,
    record_result,
    remaining_cell_timeout,
    reset_timings,
)

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_TIMEOUT_S = 2.0
# The sleeping fixture sleeps 60 s; far below that proves the timeout cut it short.
SLEEPING_MAX_WALL_S = 30.0


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- Failure fixtures, run in a real kernel -------------------------------------------


def test_failing_notebook_is_reported_as_error(tmp_path: Path) -> None:
    """A notebook that raises is reported as ``error`` and recorded in the timings."""
    result = execute_notebook(
        FIXTURES / "failing.ipynb",
        tmp_path / "out" / "failing.ipynb",
        FIXTURE_TIMEOUT_S,
    )
    timings = tmp_path / "timings.json"
    record_result(timings, "tests/fixtures/failing.ipynb", result)

    assert result.status == STATUS_ERROR
    assert result.error is not None and "RuntimeError" in result.error
    recorded = json.loads(timings.read_text(encoding="utf-8"))
    entry = recorded["tests/fixtures/failing.ipynb"]
    assert entry["status"] == STATUS_ERROR
    assert entry["wall_time_s"] == pytest.approx(result.wall_time_s)
    assert entry["timeout_s"] == FIXTURE_TIMEOUT_S
    assert (tmp_path / "out" / "failing.ipynb").is_file()


def test_sleeping_notebook_is_reported_as_timeout(tmp_path: Path) -> None:
    """A notebook that overruns a two-second budget is a ``timeout``, and recorded."""
    result = execute_notebook(
        FIXTURES / "sleeping.ipynb",
        tmp_path / "out" / "sleeping.ipynb",
        FIXTURE_TIMEOUT_S,
    )
    timings = tmp_path / "timings.json"
    record_result(timings, "tests/fixtures/sleeping.ipynb", result)

    assert result.status == STATUS_TIMEOUT
    assert result.wall_time_s < SLEEPING_MAX_WALL_S
    entry = json.loads(timings.read_text(encoding="utf-8"))[
        "tests/fixtures/sleeping.ipynb"
    ]
    assert entry["status"] == STATUS_TIMEOUT
    assert entry["timeout_s"] == FIXTURE_TIMEOUT_S


def test_source_notebook_is_never_written(tmp_path: Path) -> None:
    """Executing a fixture leaves the source file byte for byte unchanged."""
    source = FIXTURES / "failing.ipynb"
    before = source.read_bytes()
    execute_notebook(source, tmp_path / "failing.ipynb", FIXTURE_TIMEOUT_S)
    assert source.read_bytes() == before


def _notebook(path: Path, sources: list[str]) -> Path:
    """Write a notebook with one code cell per source string."""
    notebook = new_notebook(cells=[new_code_cell(source) for source in sources])
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)
    return path


def test_total_wall_time_over_budget_is_a_timeout(tmp_path: Path) -> None:
    """Cells that each fit their own limit but together overrun the budget time out.

    Each cell sleeps 0.6 s and always receives at least one second, so no cell times
    out; the four of them, with the kernel start-up, exceed a budget of two seconds.
    """
    source = _notebook(
        tmp_path / "slow_total.ipynb", ["import time\ntime.sleep(0.6)"] * 4
    )
    result = execute_notebook(source, tmp_path / "out" / "slow_total.ipynb", 2.0)

    assert result.status == STATUS_TIMEOUT
    assert result.error is not None and "exceeds the budget" in result.error


def test_kernel_runs_in_the_notebooks_own_folder(tmp_path: Path) -> None:
    """The working directory of the kernel is the folder of the notebook."""
    folder = tmp_path / "nested" / "folder"
    source = _notebook(folder / "where.ipynb", ["import os\nprint(os.getcwd())"])
    executed = tmp_path / "out" / "where.ipynb"
    result = execute_notebook(source, executed, 60.0)

    assert result.status == STATUS_PASSED
    output = nbformat.read(executed, as_version=4).cells[0].outputs[0].text
    assert Path(output.strip()).resolve() == folder.resolve()


# --- Remaining time, pure -----------------------------------------------------------


def test_remaining_time_is_budget_minus_elapsed() -> None:
    assert remaining_cell_timeout(300.0, 100.0) == pytest.approx(200.0)


@pytest.mark.parametrize("elapsed", [2.0, 2.5, 1000.0])
def test_remaining_time_is_never_zero_or_negative(elapsed: float) -> None:
    """nbclient reads zero or less as no limit, so the floor must stay positive."""
    assert remaining_cell_timeout(2.0, elapsed) >= MIN_CELL_TIMEOUT_S > 0


# --- Table of contents ----------------------------------------------------------------


def _project(tmp_path: Path, toc: str, notebooks: list[str]) -> Path:
    for name in notebooks:
        _write(tmp_path / name, "{}")
    return _write(tmp_path / "myst.yml", f"version: 1\nproject:\n  toc:\n{toc}")


def test_toc_lists_notebooks_and_ignores_other_files(tmp_path: Path) -> None:
    myst = _project(
        tmp_path,
        "    - file: a.md\n    - file: b.ipynb\n    - url: https://example.org\n",
        ["b.ipynb", "a.md"],
    )
    assert list_notebooks(myst) == ["b.ipynb"]


def test_toc_finds_nested_notebooks(tmp_path: Path) -> None:
    toc = (
        "    - file: a.md\n"
        "      children:\n"
        "        - file: sub/one.ipynb\n"
        "        - title: Group\n"
        "          children:\n"
        "            - file: sub/deeper/two.ipynb\n"
    )
    myst = _project(tmp_path, toc, ["a.md", "sub/one.ipynb", "sub/deeper/two.ipynb"])
    assert list_notebooks(myst) == ["sub/one.ipynb", "sub/deeper/two.ipynb"]


def test_toc_refuses_a_pattern_entry(tmp_path: Path) -> None:
    myst = _project(tmp_path, "    - file: a.md\n    - pattern: '*.ipynb'\n", ["a.md"])
    with pytest.raises(ValueError, match="unsupported"):
        list_notebooks(myst)


def test_toc_refuses_a_pattern_nested_in_children(tmp_path: Path) -> None:
    toc = "    - file: a.md\n      children:\n        - pattern: 'x/*.ipynb'\n"
    myst = _project(tmp_path, toc, ["a.md"])
    with pytest.raises(ValueError, match="unsupported"):
        list_notebooks(myst)


def test_toc_refuses_a_missing_notebook(tmp_path: Path) -> None:
    myst = _project(tmp_path, "    - file: gone.ipynb\n", [])
    with pytest.raises(ValueError, match="does not exist"):
        list_notebooks(myst)


def test_toc_refuses_a_file_entry_without_extension(tmp_path: Path) -> None:
    myst = _project(tmp_path, "    - file: page\n", [])
    with pytest.raises(ValueError, match="no extension"):
        list_notebooks(myst)


def test_toc_missing_is_an_error(tmp_path: Path) -> None:
    myst = _write(tmp_path / "myst.yml", "version: 1\nproject:\n  title: x\n")
    with pytest.raises(ValueError, match="no project.toc"):
        list_notebooks(myst)


# --- Timeout configuration -----------------------------------------------------------


def test_default_timeout_applies_without_an_entry(tmp_path: Path) -> None:
    config = _write(tmp_path / "c.yaml", "default_timeout_s: 300\ntimeouts: {}\n")
    assert load_timeouts(config, ["a.ipynb"]).timeout_for("a.ipynb") == 300.0


def test_own_timeout_overrides_the_default(tmp_path: Path) -> None:
    config = _write(
        tmp_path / "c.yaml",
        "default_timeout_s: 300\ntimeouts:\n  a.ipynb: 42\n",
    )
    loaded = load_timeouts(config, ["a.ipynb", "b.ipynb"])
    assert loaded.timeout_for("a.ipynb") == 42.0
    assert loaded.timeout_for("b.ipynb") == 300.0


def test_unknown_path_under_timeouts_is_refused(tmp_path: Path) -> None:
    config = _write(
        tmp_path / "c.yaml",
        "default_timeout_s: 300\ntimeouts:\n  typo.ipynb: 42\n",
    )
    with pytest.raises(ValueError, match="typo.ipynb"):
        load_timeouts(config, ["a.ipynb"])


@pytest.mark.parametrize("bad", ["0", "-5", "abc"])
def test_non_positive_default_is_refused(tmp_path: Path, bad: str) -> None:
    config = _write(tmp_path / "c.yaml", f"default_timeout_s: {bad}\ntimeouts: {{}}\n")
    with pytest.raises(ValueError, match="default_timeout_s"):
        load_timeouts(config, [])


def test_real_configuration_matches_the_real_table_of_contents() -> None:
    """The committed configuration only names notebooks of the committed book."""
    root = Path(__file__).resolve().parents[1]
    notebooks = list_notebooks(root / "myst.yml")
    load_timeouts(root / "config" / "notebooks.yaml", notebooks)


# --- Timings file ---------------------------------------------------------------------


def test_timings_are_written_for_a_pass_and_for_a_failure(tmp_path: Path) -> None:
    timings = tmp_path / "reports" / "timings.json"
    record_result(timings, "a.ipynb", NotebookResult(STATUS_PASSED, 1.5, 300.0))
    record_result(timings, "b.ipynb", NotebookResult(STATUS_ERROR, 0.5, 300.0, "boom"))
    recorded = json.loads(timings.read_text(encoding="utf-8"))
    assert recorded["a.ipynb"] == {
        "status": STATUS_PASSED,
        "wall_time_s": 1.5,
        "timeout_s": 300.0,
    }
    assert recorded["b.ipynb"]["status"] == STATUS_ERROR
    assert recorded["b.ipynb"]["error"] == "boom"


def test_reset_empties_the_timings_file(tmp_path: Path) -> None:
    timings = tmp_path / "timings.json"
    record_result(timings, "a.ipynb", NotebookResult(STATUS_PASSED, 1.0, 300.0))
    reset_timings(timings)
    assert json.loads(timings.read_text(encoding="utf-8")) == {}
