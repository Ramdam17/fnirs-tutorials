"""Tests of the notebook runner: failure fixtures and the parts that need no kernel.

The two fixtures run in a real kernel and write to temporary folders, never to
``_build/executed/`` or ``build-reports/``. The other tests use temporary
``myst.yml`` and configuration files, never the real ones.
"""

from __future__ import annotations

import json
import subprocess
import sys
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
    remove_executed_notebooks,
    reset_timings,
)

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_TIMEOUT_S = 2.0
# Budget of the fixtures that must not time out: kernel start-up counts against the
# budget, so a tight one could leave the raising cell with the floor on a slow runner.
FIXTURE_ERROR_TIMEOUT_S = 60.0
# The sleeping fixture sleeps 60 s; far below that proves the timeout cut it short.
SLEEPING_MAX_WALL_S = 30.0
# Heuristic bound, not derived from a measurement: the child pytest of the command
# tests takes a few seconds, so three minutes only catches a hang before the job limit
# of a CI runner does.
COMMAND_SUBPROCESS_TIMEOUT_S = 180.0
CHILD_PREFIX = "[child] "
STALE_EXECUTED_COPY = Path("_build") / "executed" / "book" / "old.ipynb"


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
        FIXTURE_ERROR_TIMEOUT_S,
    )
    timings = tmp_path / "timings.json"
    record_result(timings, "tests/fixtures/failing.ipynb", result)

    assert result.status == STATUS_ERROR
    assert result.error is not None and "RuntimeError" in result.error
    recorded = json.loads(timings.read_text(encoding="utf-8"))
    entry = recorded["tests/fixtures/failing.ipynb"]
    assert entry["status"] == STATUS_ERROR
    assert entry["wall_time_s"] == pytest.approx(result.wall_time_s)
    assert entry["timeout_s"] == FIXTURE_ERROR_TIMEOUT_S
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
    execute_notebook(source, tmp_path / "failing.ipynb", FIXTURE_ERROR_TIMEOUT_S)
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

    The floor of the time given to a cell is raised to 30 s and each cell sleeps 1 s,
    so every cell has a margin of 29 s and cannot time out, however slow the machine.
    The four cells take at least 4 s, which exceeds the budget of 2 s for certain.
    """
    source = _notebook(
        tmp_path / "slow_total.ipynb", ["import time\ntime.sleep(1.0)"] * 4
    )
    result = execute_notebook(
        source, tmp_path / "out" / "slow_total.ipynb", 2.0, min_cell_timeout_s=30.0
    )

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
    with pytest.raises(ValueError, match="explicit extension"):
        list_notebooks(myst)


def test_toc_refuses_a_stem_that_only_looks_like_it_has_an_extension(
    tmp_path: Path,
) -> None:
    """``06.1_intro`` has the suffix ``.1_intro``; MyST would resolve it on disk."""
    myst = _project(tmp_path, "    - file: book/m06/06.1_intro\n", [])
    with pytest.raises(ValueError, match="explicit extension"):
        list_notebooks(myst)


def test_toc_treats_an_upper_case_extension_as_a_notebook(tmp_path: Path) -> None:
    myst = _project(tmp_path, "    - file: Upper.IPYNB\n", ["Upper.IPYNB"])
    assert list_notebooks(myst) == ["Upper.IPYNB"]


def test_toc_refuses_a_markdown_page_with_a_kernelspec(tmp_path: Path) -> None:
    """A MyST build executes such a page; this runner would not."""
    page = "---\nkernelspec:\n  name: python3\n---\n\n# Title\n"
    myst = _project(tmp_path, "    - file: executable.md\n", [])
    _write(tmp_path / "executable.md", page)
    with pytest.raises(ValueError, match="kernelspec"):
        list_notebooks(myst)


_CODE_CELL = "```{code-cell} python\nprint(1)\n```\n"
_REFUSED_PAGES = {
    # row: .md with a top-level kernelspec (also without any executable node)
    "top-level kernelspec, no executable node": (
        "---\nkernelspec:\n  name: python3\n---\n\n# Title\n",
        "kernelspec",
    ),
    "top-level kernelspec with a code cell": (
        "---\nkernelspec:\n  name: python3\n---\n\n" + _CODE_CELL,
        "kernelspec",
    ),
    # row: .md with jupyter.kernelspec
    "nested jupyter.kernelspec": (
        "---\njupyter:\n  kernelspec:\n    name: python3\n---\n\n# Title\n",
        "kernelspec",
    ),
    # row: .md with {code-cell} or {eval} and no kernelspec
    "code cell without kernelspec": ("# Title\n\n" + _CODE_CELL, "code-cell"),
    "colon-fenced code cell": ("# T\n\n:::{code-cell} python\n1\n:::\n", "code-cell"),
    "eval role without kernelspec": ("The value is {eval}`1 + 1`.\n", "eval"),
    # skip semantics settled in the bundle (lines 207880 to 207886)
    "skip false wins over skip_execution true": (
        "---\nkernelspec:\n  name: p\nexecute:\n  skip: false\nskip_execution: true\n---\n",
        "kernelspec",
    ),
    "skip as a string is refused on purpose": (
        "---\nkernelspec:\n  name: p\nexecute:\n  skip: 'true'\n---\n",
        "kernelspec",
    ),
}
_ACCEPTED_PAGES = {
    "jupytext key without kernelspec": "---\njupytext:\n  formats: md\n---\n\n# T\n",
    "null kernelspec": "---\nkernelspec: null\n---\n\n# Title\n",
    "null nested kernelspec": "---\njupyter:\n  kernelspec: null\n---\n\n# T\n",
    "kernelspec with execute.skip true": (
        "---\nkernelspec:\n  name: p\nexecute:\n  skip: true\n---\n"
    ),
    "kernelspec with skip_execution true": (
        "---\nkernelspec:\n  name: p\nskip_execution: true\n---\n"
    ),
    "code cell with execute.skip true": (
        "---\nexecute:\n  skip: true\n---\n\n" + _CODE_CELL
    ),
    "skip_execution true with a null execute.skip": (
        "---\nkernelspec:\n  name: p\nexecute:\n  skip: null\nskip_execution: true\n---\n"
    ),
    "plain page": "# Title\n\nSome text.\n",
}


@pytest.mark.parametrize(
    "name",
    [pytest.param(key, id=key.replace(" ", "-")) for key in sorted(_REFUSED_PAGES)],
)
def test_toc_refuses_a_markdown_page_meant_for_execution(
    tmp_path: Path, name: str
) -> None:
    """Each sign of a page meant for execution is refused, naming entry and page."""
    page, sign = _REFUSED_PAGES[name]
    myst = _project(tmp_path, "    - file: page.md\n", [])
    _write(tmp_path / "page.md", page)
    with pytest.raises(ValueError, match="page.md") as raised:
        list_notebooks(myst)
    assert sign in str(raised.value)
    assert "execute: {skip: true}" in str(raised.value)


@pytest.mark.parametrize(
    "name",
    [pytest.param(key, id=key.replace(" ", "-")) for key in sorted(_ACCEPTED_PAGES)],
)
def test_toc_accepts_a_markdown_page_not_meant_for_execution(
    tmp_path: Path, name: str
) -> None:
    myst = _project(tmp_path, "    - file: page.md\n", [])
    _write(tmp_path / "page.md", _ACCEPTED_PAGES[name])
    assert list_notebooks(myst) == []


def test_toc_refuses_a_myst_json_entry(tmp_path: Path) -> None:
    """The runner cannot tell whether MyST executes it, and this was not measured."""
    myst = _project(tmp_path, "    - file: page.myst.json\n", ["page.myst.json"])
    with pytest.raises(ValueError, match="not measured"):
        list_notebooks(myst)


def test_toc_accepts_a_tex_entry(tmp_path: Path) -> None:
    myst = _project(tmp_path, "    - file: paper.tex\n", ["paper.tex"])
    assert list_notebooks(myst) == []


def test_toc_accepts_a_markdown_page_with_other_front_matter(tmp_path: Path) -> None:
    myst = _project(tmp_path, "    - file: plain.md\n", [])
    _write(tmp_path / "plain.md", "---\ntitle: Plain\n---\n\n# Title\n")
    assert list_notebooks(myst) == []


def test_toc_names_a_markdown_page_with_invalid_front_matter(tmp_path: Path) -> None:
    """A leading ``---`` block that is not YAML is refused, naming the page."""
    myst = _project(tmp_path, "    - file: broken.md\n", [])
    _write(tmp_path / "broken.md", "---\nNote: a: b\n---\n\n# Title\n")
    with pytest.raises(ValueError, match="broken.md") as raised:
        list_notebooks(myst)
    assert "not valid YAML" in str(raised.value)
    assert raised.value.__cause__ is not None


def test_toc_names_a_markdown_page_that_is_not_utf8(tmp_path: Path) -> None:
    myst = _project(tmp_path, "    - file: latin.md\n", [])
    (tmp_path / "latin.md").write_bytes(
        "---\ntitle: caf\u00e9\n---\n".encode("latin-1")
    )
    with pytest.raises(ValueError, match="latin.md") as raised:
        list_notebooks(myst)
    assert "not UTF-8" in str(raised.value)
    assert isinstance(raised.value.__cause__, UnicodeDecodeError)


def test_toc_accepts_prose_between_two_horizontal_rules(tmp_path: Path) -> None:
    """A block that parses as plain text, not a mapping, is ordinary markdown."""
    myst = _project(tmp_path, "    - file: rules.md\n", [])
    _write(tmp_path / "rules.md", "---\nSome prose\n\n---\n\n# Title\n")
    assert list_notebooks(myst) == []


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


@pytest.mark.parametrize("bad", [".nan", ".inf", "-.inf"])
def test_non_finite_default_is_refused(tmp_path: Path, bad: str) -> None:
    config = _write(tmp_path / "c.yaml", f"default_timeout_s: {bad}\ntimeouts: {{}}\n")
    with pytest.raises(ValueError, match="default_timeout_s"):
        load_timeouts(config, [])


@pytest.mark.parametrize("bad", [".nan", ".inf"])
def test_non_finite_notebook_timeout_is_refused(tmp_path: Path, bad: str) -> None:
    config = _write(
        tmp_path / "c.yaml", f"default_timeout_s: 300\ntimeouts:\n  a.ipynb: {bad}\n"
    )
    with pytest.raises(ValueError, match="a.ipynb"):
        load_timeouts(config, ["a.ipynb"])


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


def test_removal_deletes_executed_notebooks_and_keeps_other_files(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "executed"
    _write(folder / "book" / "a.ipynb", "{}")
    _write(folder / "b.ipynb", "{}")
    _write(folder / "book" / "page.md", "# page")
    _write(folder / "_build" / "templates" / "theme.js", "x")

    removed = remove_executed_notebooks(folder)

    assert len(removed) == 2
    assert not list(folder.rglob("*.ipynb"))
    assert (folder / "book" / "page.md").is_file()
    assert (folder / "_build" / "templates" / "theme.js").is_file()
    assert remove_executed_notebooks(tmp_path / "absent") == []


def test_reset_empties_the_timings_file(tmp_path: Path) -> None:
    timings = tmp_path / "timings.json"
    record_result(timings, "a.ipynb", NotebookResult(STATUS_PASSED, 1.0, 300.0))
    reset_timings(timings)
    assert json.loads(timings.read_text(encoding="utf-8")) == {}


# --- The command itself, in a subprocess ----------------------------------------------


def echo_child_output(*streams: str | bytes | None) -> None:
    """Write the output of the child process, every line prefixed with ``[child] ``.

    Parameters
    ----------
    *streams : str or bytes or None
        Captured streams. ``subprocess.TimeoutExpired`` carries ``bytes`` or ``None``
        on POSIX even when ``text=True`` was asked (``subprocess.py``, lines 1253 to
        1256 of Python 3.12), so all three types are accepted; bytes are decoded with
        ``errors="replace"``.
    """
    for stream in streams:
        if stream is None:
            continue
        text = (
            stream.decode("utf-8", errors="replace")
            if isinstance(stream, bytes)
            else stream
        )
        for line in text.splitlines():
            sys.stdout.write(f"{CHILD_PREFIX}{line}\n")


def _run_command_in_temporary_repository(
    tmp_path: Path, fixture: str, timeouts: str
) -> tuple[int, dict[str, dict[str, object]]]:
    """Run ``pytest tests/test_notebooks.py`` in a repository that lists one fixture.

    A stale notebook is planted under ``_build/executed/`` before the run; the command
    must remove it, which pins the call made by the session fixture.

    The repository holds a ``myst.yml`` whose table of contents lists a copy of the
    fixture, a ``config/notebooks.yaml``, and copies of the real ``test_notebooks.py``
    and ``notebook_runner.py``. ``test_notebooks.py`` finds its repository from its own
    location, so the copy runs against this temporary one.

    Returns
    -------
    tuple
        The exit code of the command and the content of its timings file.
    """
    repo = tmp_path / "repo"
    notebook = f"book/{fixture}"
    _write(repo / "myst.yml", f"version: 1\nproject:\n  toc:\n    - file: {notebook}\n")
    _write(
        repo / "config" / "notebooks.yaml",
        f"default_timeout_s: 300\ntimeouts:\n{timeouts}",
    )
    (repo / "book").mkdir(parents=True)
    (repo / notebook).write_bytes((FIXTURES / fixture).read_bytes())
    (repo / "tests").mkdir()
    for name in ("test_notebooks.py", "notebook_runner.py"):
        (repo / "tests" / name).write_bytes((FIXTURES.parent / name).read_bytes())

    stale = repo / STALE_EXECUTED_COPY
    stale.parent.mkdir(parents=True)
    stale.write_bytes((FIXTURES / "failing.ipynb").read_bytes())

    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_notebooks.py",
                "-p",
                "no:cacheprovider",
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
            timeout=COMMAND_SUBPROCESS_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as expired:
        # The exception carries what the child wrote before it was killed.
        echo_child_output(expired.stdout, expired.stderr)
        raise
    echo_child_output(completed.stdout, completed.stderr)
    assert not stale.exists(), "the command left a stale executed copy"
    timings = json.loads(
        (repo / "build-reports" / "notebook-timings.json").read_text(encoding="utf-8")
    )
    return completed.returncode, timings


@pytest.mark.parametrize(
    "stream", ["one\ntwo\n", b"one\ntwo\n", None], ids=["str", "bytes", "none"]
)
def test_echo_child_output_prefixes_every_line(
    stream: str | bytes | None, capsys: pytest.CaptureFixture[str]
) -> None:
    """The echo accepts ``str``, ``bytes`` and ``None`` and prefixes each line."""
    echo_child_output(stream, b"bad \xff byte\n")
    lines = capsys.readouterr().out.splitlines()
    expected = [] if stream is None else [f"{CHILD_PREFIX}one", f"{CHILD_PREFIX}two"]
    assert lines == [*expected, f"{CHILD_PREFIX}bad \ufffd byte"]


def test_command_exits_non_zero_when_a_notebook_raises(tmp_path: Path) -> None:
    """The documented command fails, and records an ``error``, for a failing notebook."""
    code, timings = _run_command_in_temporary_repository(
        tmp_path, "failing.ipynb", " {}\n"
    )

    assert code != 0
    assert timings["book/failing.ipynb"]["status"] == STATUS_ERROR


def test_command_exits_non_zero_when_a_notebook_overruns(tmp_path: Path) -> None:
    """The documented command fails, and records a ``timeout``, under a 2 s timeout."""
    code, timings = _run_command_in_temporary_repository(
        tmp_path, "sleeping.ipynb", "  book/sleeping.ipynb: 2\n"
    )

    assert code != 0
    assert timings["book/sleeping.ipynb"]["status"] == STATUS_TIMEOUT
