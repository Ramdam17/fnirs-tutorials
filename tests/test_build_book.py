"""Tests of ``scripts/build_book.py``: the staging, the check of the built pages, the exits.

None of these tests needs the network or a real book build. The build step of the script
is a parameter (``runner``): the tests give it a stand-in that writes page files into
``_build/site/content/`` of the staging folder, as MyST does, and returns an exit code.

The page files under ``tests/fixtures/built_pages/`` are real files written by MyST, not
written by hand: jupyter-book 2.1.7 (myst-cli 1.11.0) built them on 2026-11-11 in one
scratch project that holds one page per case, from the sources named by the file names.
``tests/fixtures/built_pages/make_page_files.py`` makes them again; run it after an
upgrade of Jupyter Book. For example
``code_cell.json`` comes from a markdown page that holds one ``{code-cell}`` and
``notebook_smoke.json`` from the executed smoke notebook of the book. They are small, so
none is trimmed. Each keeps the ``location`` MyST gave it, ``/book/<stem>.md`` or
``/book/notebook_smoke.ipynb``, so a test project must hold the page at that path.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_book.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "built_pages"

_spec = importlib.util.spec_from_file_location("build_book", SCRIPT)
assert _spec is not None and _spec.loader is not None
build_book = importlib.util.module_from_spec(_spec)
# A dataclass of the module looks itself up in sys.modules while it is being defined.
sys.modules["build_book"] = build_book
_spec.loader.exec_module(build_book)

NOTEBOOK = "book/notebook_smoke.ipynb"
RUN_NOTEBOOKS = "uv run python -m pytest tests/test_notebooks.py -v"

ACCEPTED = [
    "notebook_smoke",
    "no_code",
    "plain_fence",
    "literal_block",
    "skipped",
    "legacy_skip",
    "string_skip",
]
# stem -> the sign the message must name.
REFUSED = {
    "code_cell": "block of kind notebook-code",
    "eval_only": "inlineExpression",
    "include_case": "block of kind notebook-code",
    "blockquote": "block of kind notebook-code",
    "list_item": "block of kind notebook-code",
    "admonition": "block of kind notebook-code",
    "tab_item": "block of kind notebook-code",
}


def load_fixture(stem: str) -> dict[str, Any]:
    """Read one real page file."""
    return json.loads((FIXTURES / f"{stem}.json").read_text(encoding="utf-8"))


def page_path(stem: str) -> str:
    """Path of the source of a fixture, from the location MyST recorded."""
    return load_fixture(stem)["location"].lstrip("/")


class FakeBuild:
    """Stand-in for ``run_jupyter_book``: writes page files as MyST does.

    Attributes
    ----------
    calls : list of Path
        Staging folder of each call.
    content_before : list of list of str
        Names of the page files that were in ``_build/site/content/`` when the build
        started, one list per call.
    """

    def __init__(self, stems: list[str], code: int = 0) -> None:
        self.stems = stems
        self.code = code
        self.calls: list[Path] = []
        self.content_before: list[list[str]] = []

    def __call__(self, stage_dir: Path) -> int:
        self.calls.append(stage_dir)
        content = stage_dir / "_build" / "site" / "content"
        self.content_before.append(
            sorted(p.name for p in content.glob("*.json")) if content.is_dir() else []
        )
        content.mkdir(parents=True, exist_ok=True)
        for stem in self.stems:
            shutil.copy2(FIXTURES / f"{stem}.json", content / f"{stem}.json")
        return self.code


def make_project(
    tmp_path: Path,
    stems: list[str],
    texts: dict[str, str] | None = None,
    executed: bool = True,
    status: str = "passed",
) -> Path:
    """Write a repository-like folder for the pages of the given fixtures.

    Notebooks get a source and, when ``executed`` is true, an executed copy that is newer
    and an entry with the given ``status`` in the timings file of the notebook command.
    Other pages get the text in ``texts`` (by stem) or a title.
    """
    root = tmp_path / "repo"
    texts = texts or {}
    toc = []
    timings: dict[str, Any] = {}
    for stem in stems:
        path = page_path(stem)
        toc.append({"file": path})
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.endswith(".ipynb"):
            target.write_text('{"source": true}', encoding="utf-8")
            if executed:
                copy = root / "_build" / "executed" / path
                copy.parent.mkdir(parents=True, exist_ok=True)
                copy.write_text('{"executed": true}', encoding="utf-8")
                stamp = target.stat().st_mtime + 100
                os.utime(copy, (stamp, stamp))
                timings[path] = {
                    "status": status,
                    "wall_time_s": 1.0,
                    "timeout_s": 300.0,
                }
        else:
            target.write_text(texts.get(stem, "# A page\n"), encoding="utf-8")
    document = {"version": 1, "project": {"toc": toc}}
    (root / "myst.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    if timings:
        write_timings(root, timings)
    return root


def write_timings(root: Path, timings: dict[str, Any]) -> None:
    """Write the timings file of the notebook command."""
    path = root / "build-reports" / "notebook-timings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timings), encoding="utf-8")


def run_main(root: Path, runner: Any) -> int:
    """Call the entry point on a test project."""
    return build_book.main(["--root", str(root)], runner=runner)


# --------------------------------------------------------------------------------------
# The check of the built pages, on real page files
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("stem", ACCEPTED)
def test_check_accepts_the_pages_that_hold_nothing_to_execute(stem: str) -> None:
    """A notebook, a page without code, a page that only shows code, a skipped page."""
    page = load_fixture(stem)
    assert build_book.check_built_pages({page["location"]: page}, [], []) == []


@pytest.mark.parametrize("stem", sorted(REFUSED))
def test_check_refuses_a_page_that_holds_code_to_execute(stem: str) -> None:
    """The message names the page and the kind of node; the exit code is its own."""
    page = load_fixture(stem)
    problems = build_book.check_built_pages({page["location"]: page}, [], [])
    assert len(problems) == 1
    assert problems[0].exit_code == build_book.EXIT_PAGE_HOLDS_CODE
    message = str(problems[0])
    assert f"page {page['location']!r} is not a notebook" in message
    assert REFUSED[stem] in message
    assert "execute: {skip: true}" in message


@pytest.mark.parametrize("skip", [False, "false", 0, None])
def test_check_refuses_a_page_whose_recorded_skip_is_not_the_boolean_true(
    skip: Any,
) -> None:
    """Only ``execute.skip: true`` excludes a page; the page is a real one, one key changed."""
    page = load_fixture("skipped")
    page["frontmatter"]["execute"] = {"skip": skip}
    problems = build_book.check_built_pages({page["location"]: page}, [], [])
    assert len(problems) == 1
    assert problems[0].exit_code == build_book.EXIT_PAGE_HOLDS_CODE


def test_check_names_both_signs_of_a_page_that_holds_both() -> None:
    """A tree with a notebook-code block and an inlineExpression names both."""
    tree = {
        "type": "root",
        "children": [
            {"type": "block", "kind": "notebook-code", "children": []},
            {"type": "paragraph", "children": [{"type": "inlineExpression"}]},
        ],
    }
    assert build_book.find_code_to_execute(tree) == [
        "block of kind notebook-code",
        "inlineExpression",
    ]


def test_check_refuses_a_missing_notebook_page_and_names_the_notebook() -> None:
    """A notebook of the table of contents with no page file is its own failure."""
    page = load_fixture("no_code")
    problems = build_book.check_built_pages(
        {page["location"]: page},
        [page_path("no_code"), NOTEBOOK],
        [NOTEBOOK],
    )
    assert len(problems) == 1
    assert problems[0].exit_code == build_book.EXIT_PAGE_NOT_BUILT
    assert f"notebook {NOTEBOOK!r} of the table of contents has no built page" in str(
        problems[0]
    )


def test_check_refuses_a_missing_markdown_page_and_calls_it_a_page() -> None:
    """A markdown page with no page file is refused too, named as a page."""
    problems = build_book.check_built_pages({}, ["book/no_code.md"], [])
    assert len(problems) == 1
    assert "page 'book/no_code.md' of the table of contents has no built page" in str(
        problems[0]
    )


# --------------------------------------------------------------------------------------
# The entry point: exit codes and messages, with a stand-in for the build
# --------------------------------------------------------------------------------------


def test_success_exits_zero_and_builds_once_in_the_executed_folder(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    root = make_project(tmp_path, ["notebook_smoke", "no_code", "plain_fence"])
    build = FakeBuild(["notebook_smoke", "no_code", "plain_fence"])
    assert run_main(root, build) == 0
    assert build.calls == [(root / "_build" / "executed").resolve()]
    assert "Book built and checked: 3 pages" in caplog.text


def test_a_markdown_page_with_code_exits_with_its_code_and_names_the_page(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "code_cell"])
    build = FakeBuild(["notebook_smoke", "code_cell"])
    assert run_main(root, build) == build_book.EXIT_PAGE_HOLDS_CODE
    assert "page '/book/code_cell.md' is not a notebook and holds code" in caplog.text
    assert "Book not built (exit 10)" in caplog.text


def test_every_problem_is_logged_and_the_first_one_sets_the_exit_code(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "code_cell", "eval_only"])
    build = FakeBuild(["code_cell", "eval_only"])
    assert run_main(root, build) == build_book.EXIT_PAGE_HOLDS_CODE
    assert "page '/book/code_cell.md'" in caplog.text
    assert "page '/book/eval_only.md'" in caplog.text
    assert f"notebook {NOTEBOOK!r} of the table of contents has no built page" in (
        caplog.text
    )


def test_a_notebook_without_a_built_page_exits_with_its_code(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    build = FakeBuild(["no_code"])
    assert run_main(root, build) == build_book.EXIT_PAGE_NOT_BUILT
    assert f"notebook {NOTEBOOK!r} of the table of contents has no built page" in (
        caplog.text
    )


def test_a_failed_build_passes_its_exit_code_on(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    build = FakeBuild(["notebook_smoke", "no_code"], code=7)
    assert run_main(root, build) == 7
    assert "the book build exited with code 7" in caplog.text
    assert "the book is not built" in caplog.text
    assert "runs with --strict" in caplog.text
    assert "an error that MyST reports on a page stops it" in caplog.text
    assert "about a DOI or another network lookup" in caplog.text
    assert "can come from outside the repository" in caplog.text
    assert "run the build a second time" in caplog.text


def test_no_executed_folder_asks_for_the_notebook_command(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"], executed=False)
    build = FakeBuild(["notebook_smoke", "no_code"])
    assert run_main(root, build) == build_book.EXIT_NOT_EXECUTED
    assert "does not exist: no notebook was executed" in caplog.text
    assert RUN_NOTEBOOKS in caplog.text
    assert build.calls == []


def test_a_missing_executed_copy_names_the_notebook(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    (root / "_build" / "executed" / NOTEBOOK).unlink()
    build = FakeBuild(["notebook_smoke", "no_code"])
    assert run_main(root, build) == build_book.EXIT_NOT_EXECUTED
    assert f"notebook {NOTEBOOK!r} has no executed copy" in caplog.text
    assert RUN_NOTEBOOKS in caplog.text
    assert build.calls == []


def test_an_executed_copy_older_than_its_source_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    source = root / NOTEBOOK
    stamp = (root / "_build" / "executed" / NOTEBOOK).stat().st_mtime + 500
    os.utime(source, (stamp, stamp))
    build = FakeBuild(["notebook_smoke", "no_code"])
    assert run_main(root, build) == build_book.EXIT_NOT_EXECUTED
    assert f"the executed copy of notebook {NOTEBOOK!r} is older than its source" in (
        caplog.text
    )
    assert build.calls == []


def test_a_table_of_contents_the_notebook_reader_refuses_exits_with_its_code(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["no_code"])
    document = {"version": 1, "project": {"toc": [{"pattern": "book/*.ipynb"}]}}
    (root / "myst.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    build = FakeBuild([])
    assert run_main(root, build) == build_book.EXIT_BAD_TOC
    assert "unsupported table-of-contents key(s) ['pattern']" in caplog.text
    assert build.calls == []


def test_a_build_that_wrote_no_page_exits_with_its_code(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["no_code"])
    assert run_main(root, FakeBuild([])) == build_book.EXIT_NO_CONTENT
    assert "the build wrote no page file" in caplog.text


def test_a_file_that_is_not_a_page_file_exits_with_its_code(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["no_code"])

    def build(stage_dir: Path) -> int:
        content = stage_dir / "_build" / "site" / "content"
        content.mkdir(parents=True)
        (content / "broken.json").write_text("{not json", encoding="utf-8")
        return 0

    assert run_main(root, build) == build_book.EXIT_UNREADABLE_PAGE
    assert "broken.json is not a page file written by MyST" in caplog.text


def test_the_base_url_of_the_environment_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("BASE_URL", "/some-repository")
    root = make_project(tmp_path, ["no_code"])
    assert run_main(root, FakeBuild(["no_code"])) == 0
    assert "BASE_URL is /some-repository" in caplog.text


def test_the_script_exits_with_the_code_of_the_failure(tmp_path: Path) -> None:
    """The ``__main__`` entry point returns the exit code to the shell.

    No executed folder exists, so the script stops before any build; this needs neither
    the network nor a real build.
    """
    root = make_project(tmp_path, ["notebook_smoke", "no_code"], executed=False)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == build_book.EXIT_NOT_EXECUTED
    assert RUN_NOTEBOOKS in result.stderr


# --------------------------------------------------------------------------------------
# What the script reads: only the build it just ran
# --------------------------------------------------------------------------------------


def test_a_page_file_left_by_an_earlier_build_is_not_read(tmp_path: Path) -> None:
    """A stale page that holds code, absent from this build, changes nothing."""
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    stale = root / "_build" / "executed" / "_build" / "site" / "content"
    stale.mkdir(parents=True)
    shutil.copy2(FIXTURES / "code_cell.json", stale / "code_cell.json")
    build = FakeBuild(["notebook_smoke", "no_code"])
    assert run_main(root, build) == 0
    assert build.content_before == [[]]
    assert not (stale / "code_cell.json").exists()


def test_the_html_of_an_earlier_build_is_removed(tmp_path: Path) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    html = root / "_build" / "executed" / "_build" / "html"
    html.mkdir(parents=True)
    (html / "old.html").write_text("old", encoding="utf-8")
    assert run_main(root, FakeBuild(["notebook_smoke", "no_code"])) == 0
    assert not html.exists()


def test_the_downloaded_theme_is_kept(tmp_path: Path) -> None:
    """The theme takes tens of seconds to fetch; only the site and the HTML go."""
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    theme = root / "_build" / "executed" / "_build" / "templates" / "site" / "theme.txt"
    theme.parent.mkdir(parents=True)
    theme.write_text("theme", encoding="utf-8")
    assert run_main(root, FakeBuild(["notebook_smoke", "no_code"])) == 0
    assert theme.read_text(encoding="utf-8") == "theme"


# --------------------------------------------------------------------------------------
# Staging
# --------------------------------------------------------------------------------------


def test_the_staging_folder_holds_the_config_the_pages_and_the_executed_notebook(
    tmp_path: Path,
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    staged = build_book.stage_book(root, root / "_build" / "executed")
    stage = staged.stage_dir
    assert (stage / "myst.yml").read_text() == (root / "myst.yml").read_text()
    assert (stage / "book" / "no_code.md").is_file()
    # The executed copy is not replaced by the source of the notebook.
    assert (stage / NOTEBOOK).read_text() == '{"executed": true}'
    assert staged.notebooks == [NOTEBOOK]
    assert staged.toc_files == [NOTEBOOK, "book/no_code.md"]


def test_a_nested_entry_of_the_table_of_contents_is_staged(tmp_path: Path) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    nested = {
        "version": 1,
        "project": {
            "toc": [
                {"file": NOTEBOOK, "children": [{"file": "book/no_code.md"}]},
            ]
        },
    }
    (root / "myst.yml").write_text(yaml.safe_dump(nested), encoding="utf-8")
    staged = build_book.stage_book(root, root / "_build" / "executed")
    assert staged.toc_files == [NOTEBOOK, "book/no_code.md"]
    assert (staged.stage_dir / "book" / "no_code.md").is_file()


def test_files_a_page_references_are_staged_and_data_is_not(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    text = (
        "# Page\n\n![figure](figs/a.png)\n\n"
        "```{include} parts/intro.md\n```\n\n"
        "![secret](../data/fresh/secret.png)\n\n"
        "[the notebook](notebook_smoke.ipynb)\n\n"
        "![remote](https://example.org/x.png)\n"
    )
    root = make_project(tmp_path, ["notebook_smoke", "no_code"], {"no_code": text})
    (root / "book" / "figs").mkdir()
    (root / "book" / "figs" / "a.png").write_bytes(b"png")
    (root / "book" / "parts").mkdir()
    (root / "book" / "parts" / "intro.md").write_text("![inner](b.png)\n")
    (root / "book" / "parts" / "b.png").write_bytes(b"png")
    (root / "data" / "fresh").mkdir(parents=True)
    (root / "data" / "fresh" / "secret.png").write_bytes(b"secret")
    staged = build_book.stage_book(root, root / "_build" / "executed")
    stage = staged.stage_dir
    assert (stage / "book" / "figs" / "a.png").is_file()
    assert (stage / "book" / "parts" / "intro.md").is_file()
    assert (stage / "book" / "parts" / "b.png").is_file()
    assert not (stage / "data").exists()
    assert "is under data/ and is not staged" in caplog.text
    assert (stage / NOTEBOOK).read_text() == '{"executed": true}'


# --------------------------------------------------------------------------------------
# The build step
# --------------------------------------------------------------------------------------


def test_the_build_command_is_strict_has_no_execute_flag_and_shows_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        seen["command"] = command
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 3)

    monkeypatch.setattr(build_book.subprocess, "run", fake_run)
    assert build_book.run_jupyter_book(tmp_path) == 3
    assert seen["command"] == [
        sys.executable,
        "-m",
        "jupyter_book",
        "build",
        "--html",
        "--strict",
    ]
    assert seen["kwargs"]["cwd"] == tmp_path
    # No redirection of the output and no changed environment (BASE_URL is inherited).
    assert not {"stdout", "stderr", "capture_output", "env"} & set(seen["kwargs"])


# --------------------------------------------------------------------------------------
# What the notebook command recorded, and the helper package
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["error", "timeout"])
def test_a_notebook_recorded_as_failed_is_refused_and_named(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, status: str
) -> None:
    """The executed copy of a failed notebook shows the traceback; it is never built."""
    root = make_project(tmp_path, ["notebook_smoke", "no_code"], status=status)
    build = FakeBuild(["notebook_smoke", "no_code"])
    assert run_main(root, build) == build_book.EXIT_NOT_EXECUTED
    assert (
        f"notebook {NOTEBOOK!r} was recorded with status {status!r}, not 'passed'"
        in (caplog.text)
    )
    assert RUN_NOTEBOOKS in caplog.text
    assert build.calls == []


def test_a_notebook_without_an_entry_in_the_timings_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    write_timings(root, {"book/other.ipynb": {"status": "passed"}})
    build = FakeBuild(["notebook_smoke", "no_code"])
    assert run_main(root, build) == build_book.EXIT_NOT_EXECUTED
    assert f"notebook {NOTEBOOK!r} has no entry in" in caplog.text
    assert "notebook-timings.json" in caplog.text
    assert build.calls == []


def test_a_missing_timings_file_has_its_own_message(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    (root / "build-reports" / "notebook-timings.json").unlink()
    build = FakeBuild(["notebook_smoke", "no_code"])
    assert run_main(root, build) == build_book.EXIT_NOT_EXECUTED
    assert "notebook-timings.json does not exist" in caplog.text
    assert "did not record the outcome of the notebooks" in caplog.text
    assert RUN_NOTEBOOKS in caplog.text
    assert build.calls == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [("{not json", "cannot be read"), ("[1, 2]", "is not a JSON object")],
)
def test_an_unreadable_timings_file_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, text: str, expected: str
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    (root / "build-reports" / "notebook-timings.json").write_text(text)
    assert run_main(root, FakeBuild([])) == build_book.EXIT_NOT_EXECUTED
    assert expected in caplog.text


def test_a_project_without_notebooks_needs_no_timings(tmp_path: Path) -> None:
    root = make_project(tmp_path, ["no_code"])
    assert not (root / "build-reports").exists()
    assert run_main(root, FakeBuild(["no_code"])) == 0


def test_a_copy_older_than_the_helper_package_is_refused_and_names_the_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    helper = root / "src" / "package" / "style.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("X = 1\n", encoding="utf-8")
    stamp = (root / "_build" / "executed" / NOTEBOOK).stat().st_mtime + 500
    os.utime(helper, (stamp, stamp))
    build = FakeBuild(["notebook_smoke", "no_code"])
    assert run_main(root, build) == build_book.EXIT_NOT_EXECUTED
    assert "is older than src/package/style.py of the helper package" in caplog.text
    assert RUN_NOTEBOOKS in caplog.text
    assert build.calls == []


def test_a_newer_bytecode_file_of_the_helper_package_is_not_a_change(
    tmp_path: Path,
) -> None:
    """Importing the package writes ``__pycache__`` files; they are not sources."""
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    cached = root / "src" / "package" / "__pycache__" / "style.cpython-312.pyc"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"pyc")
    stamp = (root / "_build" / "executed" / NOTEBOOK).stat().st_mtime + 500
    os.utime(cached, (stamp, stamp))
    assert run_main(root, FakeBuild(["notebook_smoke", "no_code"])) == 0


def test_a_helper_package_older_than_the_copy_is_accepted(tmp_path: Path) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    helper = root / "src" / "package" / "style.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("X = 1\n", encoding="utf-8")
    stamp = (root / "_build" / "executed" / NOTEBOOK).stat().st_mtime - 500
    os.utime(helper, (stamp, stamp))
    assert run_main(root, FakeBuild(["notebook_smoke", "no_code"])) == 0


# --------------------------------------------------------------------------------------
# The staging folder and the files that cannot be read or copied
# --------------------------------------------------------------------------------------


def test_a_staging_folder_that_is_the_root_is_refused_before_anything_is_removed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["no_code"])
    local_site = root / "_build" / "site" / "content"
    local_site.mkdir(parents=True)
    (local_site / "page.json").write_text("{}", encoding="utf-8")
    local_html = root / "_build" / "html"
    local_html.mkdir(parents=True)
    build = FakeBuild(["no_code"])
    code = build_book.main(
        ["--root", str(root), "--executed-dir", str(root)], runner=build
    )
    assert code == build_book.EXIT_BAD_EXECUTED_DIR
    assert "is the repository root" in caplog.text
    assert str(root) in caplog.text
    assert (local_site / "page.json").is_file()
    assert local_html.is_dir()
    assert build.calls == []


def test_a_staging_folder_that_holds_the_root_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["no_code"])
    build = FakeBuild(["no_code"])
    code = build_book.main(
        ["--root", str(root), "--executed-dir", str(tmp_path)], runner=build
    )
    assert code == build_book.EXIT_BAD_EXECUTED_DIR
    assert "or one of its parents" in caplog.text
    assert build.calls == []
    assert not (tmp_path / "myst.yml").exists()


def test_a_missing_myst_yml_exits_with_a_named_code(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["no_code"])
    (root / "myst.yml").unlink()
    build = FakeBuild([])
    assert run_main(root, build) == build_book.EXIT_FILE_ERROR
    assert f"cannot read {root / 'myst.yml'}" in caplog.text
    assert build.calls == []


def test_a_myst_yml_that_is_not_yaml_exits_with_the_table_of_contents_code(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["no_code"])
    (root / "myst.yml").write_text("project: [unclosed", encoding="utf-8")
    assert run_main(root, FakeBuild([])) == build_book.EXIT_BAD_TOC
    assert "is not valid YAML" in caplog.text


def test_a_file_that_cannot_be_copied_exits_with_a_named_code(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A file where a staging folder is needed makes the copy fail with an OSError."""
    root = make_project(tmp_path, ["no_code"])
    executed = root / "_build" / "executed"
    executed.mkdir(parents=True)
    (executed / "book").write_text("in the way", encoding="utf-8")
    build = FakeBuild(["no_code"])
    assert run_main(root, build) == build_book.EXIT_FILE_ERROR
    assert "cannot stage" in caplog.text
    assert "to " + str(executed / "book" / "no_code.md") in caplog.text
    assert build.calls == []


# --------------------------------------------------------------------------------------
# Staging: what pages, notebooks and myst.yml reference
# --------------------------------------------------------------------------------------

DIRECTIVE_TARGETS = {
    "include": "parts/target.md",
    "literalinclude": "parts/target.py",
    "image": "parts/target.png",
    "figure": "parts/target.png",
    "download": "parts/target.zip",
}


@pytest.mark.parametrize("directive", sorted(DIRECTIVE_TARGETS))
def test_the_file_each_directive_names_is_staged(
    tmp_path: Path, directive: str
) -> None:
    target = DIRECTIVE_TARGETS[directive]
    root = make_project(
        tmp_path,
        ["notebook_smoke", "no_code"],
        {"no_code": f"# Page\n\n```{{{directive}}} {target}\n```\n"},
    )
    (root / "book" / "parts").mkdir()
    (root / "book" / target).write_bytes(b"content")
    staged = build_book.stage_book(root, root / "_build" / "executed")
    assert (staged.stage_dir / "book" / target).read_bytes() == b"content"


def test_a_root_relative_reference_is_resolved_against_the_root(
    tmp_path: Path,
) -> None:
    root = make_project(
        tmp_path,
        ["notebook_smoke", "no_code"],
        {"no_code": "# Page\n\n![fig](/assets/fig.png)\n"},
    )
    (root / "assets").mkdir()
    (root / "assets" / "fig.png").write_bytes(b"png")
    staged = build_book.stage_book(root, root / "_build" / "executed")
    assert (staged.stage_dir / "assets" / "fig.png").is_file()


def test_a_fragment_and_a_query_are_dropped_from_a_reference(tmp_path: Path) -> None:
    text = "# Page\n\n[a](figs/a.html#section)\n\n![b](figs/b.png?raw=1)\n"
    root = make_project(tmp_path, ["notebook_smoke", "no_code"], {"no_code": text})
    (root / "book" / "figs").mkdir()
    (root / "book" / "figs" / "a.html").write_text("a", encoding="utf-8")
    (root / "book" / "figs" / "b.png").write_bytes(b"b")
    staged = build_book.stage_book(root, root / "_build" / "executed")
    assert (staged.stage_dir / "book" / "figs" / "a.html").is_file()
    assert (staged.stage_dir / "book" / "figs" / "b.png").is_file()


def test_a_reference_outside_the_repository_is_not_staged_and_does_not_fail(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(
        tmp_path,
        ["notebook_smoke", "no_code"],
        {"no_code": "# Page\n\n![out](../../outside.png)\n"},
    )
    (tmp_path / "outside.png").write_bytes(b"outside")
    staged = build_book.stage_book(root, root / "_build" / "executed")
    assert "points outside the repository" in caplog.text
    assert not list(staged.stage_dir.rglob("outside.png"))


def test_a_file_referenced_by_two_pages_is_staged_once(tmp_path: Path) -> None:
    text = "# Page\n\n![fig](fig.png)\n"
    root = make_project(
        tmp_path,
        ["notebook_smoke", "no_code", "plain_fence"],
        {"no_code": text, "plain_fence": text},
    )
    (root / "book" / "fig.png").write_bytes(b"png")
    staged = build_book.stage_book(root, root / "_build" / "executed")
    assert staged.copied.count(Path("book/fig.png")) == 1


def _notebook_with_markdown(markdown: str) -> str:
    """Return the text of a notebook whose first cell is the given markdown."""
    return json.dumps(
        {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": markdown.splitlines(True),
                },
                {"cell_type": "code", "metadata": {}, "source": ["![nope](code.png)"]},
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    )


def test_the_files_the_markdown_cells_of_a_notebook_reference_are_staged(
    tmp_path: Path,
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    source = root / NOTEBOOK
    source.write_text(
        _notebook_with_markdown(
            "A figure: ![fig](figs/n.png)\n\n```{include} parts/n.md\n```\n"
        ),
        encoding="utf-8",
    )
    stamp = (root / "_build" / "executed" / NOTEBOOK).stat().st_mtime - 100
    os.utime(source, (stamp, stamp))
    for name in ("figs/n.png", "parts/n.md", "code.png"):
        path = root / "book" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"x")
    staged = build_book.stage_book(root, root / "_build" / "executed")
    assert (staged.stage_dir / "book" / "figs" / "n.png").is_file()
    assert (staged.stage_dir / "book" / "parts" / "n.md").is_file()
    # Code cells are not searched, and the executed copy is not replaced by the source.
    assert not (staged.stage_dir / "book" / "code.png").exists()
    assert (staged.stage_dir / NOTEBOOK).read_text() == '{"executed": true}'


def test_a_notebook_source_that_is_not_json_is_skipped_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    source = root / NOTEBOOK
    source.write_text("not json", encoding="utf-8")
    stamp = (root / "_build" / "executed" / NOTEBOOK).stat().st_mtime - 100
    os.utime(source, (stamp, stamp))
    build_book.stage_book(root, root / "_build" / "executed")
    assert "is not a notebook file; its references are not staged" in caplog.text


def test_the_local_files_myst_yml_names_are_staged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    document = yaml.safe_load((root / "myst.yml").read_text(encoding="utf-8"))
    document["project"]["bibliography"] = ["refs/library.bib", "refs/absent.bib"]
    document["project"]["title"] = "Not a file"
    document["site"] = {"options": {"logo": "assets/logo.png"}}
    document["project"]["thumbnail"] = "data/fresh/secret.png"
    (root / "myst.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    for name in ("refs/library.bib", "assets/logo.png", "data/fresh/secret.png"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    staged = build_book.stage_book(root, root / "_build" / "executed")
    stage = staged.stage_dir
    assert (stage / "refs" / "library.bib").is_file()
    assert (stage / "assets" / "logo.png").is_file()
    assert not (stage / "refs" / "absent.bib").exists()
    assert not (stage / "data").exists()
    assert "is under data/ and is not staged" in caplog.text


def test_a_myst_yml_that_is_not_a_mapping_exits_with_the_table_of_contents_code(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["no_code"])
    (root / "myst.yml").write_text("- a\n- list\n", encoding="utf-8")
    assert run_main(root, FakeBuild([])) == build_book.EXIT_BAD_TOC
    assert "is not a mapping at its top level" in caplog.text


def test_the_references_of_a_markdown_file_named_in_myst_yml_are_followed(
    tmp_path: Path,
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    document = yaml.safe_load((root / "myst.yml").read_text(encoding="utf-8"))
    document["project"]["abstract"] = "parts/abstract.md"
    (root / "myst.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    (root / "parts").mkdir()
    (root / "parts" / "abstract.md").write_text("![x](pic.png)\n", encoding="utf-8")
    (root / "parts" / "pic.png").write_bytes(b"png")
    staged = build_book.stage_book(root, root / "_build" / "executed")
    assert (staged.stage_dir / "parts" / "abstract.md").is_file()
    assert (staged.stage_dir / "parts" / "pic.png").is_file()


def test_the_script_exits_with_the_code_of_a_refused_staging_folder(
    tmp_path: Path,
) -> None:
    """The ``__main__`` entry point returns the staging-folder code to the shell."""
    root = make_project(tmp_path, ["no_code"])
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--executed-dir", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == build_book.EXIT_BAD_EXECUTED_DIR
    assert "is the repository root" in result.stderr


# --------------------------------------------------------------------------------------
# A string of myst.yml that the file system refuses
# --------------------------------------------------------------------------------------


def test_a_description_too_long_for_a_file_name_is_not_a_file_and_the_book_builds(
    tmp_path: Path,
) -> None:
    """Every string of ``myst.yml`` is tried as a path; prose longer than a name allows."""
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    document = yaml.safe_load((root / "myst.yml").read_text(encoding="utf-8"))
    plain = build_book.find_config_files(document, root / "myst.yml", root)
    document["project"]["description"] = "a" * 300 + "/x"
    (root / "myst.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    assert build_book.find_config_files(document, root / "myst.yml", root) == plain
    build = FakeBuild(["notebook_smoke", "no_code"])
    assert run_main(root, build) == 0
    assert len(build.calls) == 1


# --------------------------------------------------------------------------------------
# The script that makes the page files refuses a scratch folder in the repository
# --------------------------------------------------------------------------------------

_maker_spec = importlib.util.spec_from_file_location(
    "make_page_files", FIXTURES / "make_page_files.py"
)
assert _maker_spec is not None and _maker_spec.loader is not None
make_page_files = importlib.util.module_from_spec(_maker_spec)
sys.modules["make_page_files"] = make_page_files
_maker_spec.loader.exec_module(make_page_files)


def test_the_maker_of_the_page_files_refuses_the_repository_root_and_what_is_inside(
    tmp_path: Path,
) -> None:
    assert make_page_files.is_inside_repository(REPO_ROOT)
    assert make_page_files.is_inside_repository(REPO_ROOT / "docs" / "scratch")
    assert not make_page_files.is_inside_repository(tmp_path)


@pytest.mark.parametrize("scratch", [REPO_ROOT, REPO_ROOT / "scratch"])
def test_the_maker_of_the_page_files_stops_before_writing_in_the_repository(
    scratch: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A notebook that does not exist makes the maker stop before it writes anything,
    # also when the guard under test is broken, so this test never touches the repository.
    missing = tmp_path / "no-such-notebook.ipynb"
    with pytest.raises(SystemExit) as stop:
        make_page_files.main(
            [
                "--scratch",
                str(scratch),
                "--notebook",
                str(missing),
                "--output",
                str(tmp_path),
            ]
        )
    assert stop.value.code == 2
    assert "--scratch must be outside the repository tree" in capsys.readouterr().err
    assert not (scratch / "project").exists()


# --------------------------------------------------------------------------------------
# The rule for a frame: <base>/interactive/<path>, published by the build
# --------------------------------------------------------------------------------------

FRAME_SOURCE = "frames/live.html"
ILLUSTRATION = "hrf.html"


def frame_page(stem: str, src: str) -> dict[str, Any]:
    """A real page file with one frame whose source is replaced."""
    page = load_fixture(stem)
    frame = page["mdast"]["children"][0]["children"][0]
    assert frame["type"] == "iframe"
    frame["src"] = src
    return page


def html_with_frame(src: str) -> str:
    """The HTML of a built page that holds one frame, as the theme writes it."""
    return (
        '<div class="relative inline-block"><iframe title="t" '
        f'src="{src}" style="width:100%"></iframe></div>'
    )


def publish(html_dir: Path, slug: str, src: str, files: tuple[str, ...]) -> None:
    """Write a built site: the page of ``slug`` with one frame, and the published files."""
    page = html_dir / slug / "index.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(html_with_frame(src), encoding="utf-8")
    for name in files:
        target = html_dir / "interactive" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("<!doctype html>", encoding="utf-8")


def check_one(
    tmp_path: Path,
    src: str,
    base_url: str,
    files: tuple[str, ...] = (ILLUSTRATION, "sub/two.html"),
    html_src: str | None = None,
) -> list[Any]:
    """Check one frame page against a built site that holds ``files``.

    The HTML of the page holds ``html_src``, which is ``src`` unless the test says
    otherwise.
    """
    html_dir = tmp_path / "_build" / "html"
    page = frame_page("frame_relative", src)
    publish(html_dir, page["slug"], src if html_src is None else html_src, files)
    return build_book.check_frames(page["location"], page, base_url, html_dir)


@pytest.mark.parametrize(
    ("base_url", "src"),
    [
        ("", f"/interactive/{ILLUSTRATION}"),
        ("", "/interactive/sub/two.html"),
        ("/fnirs-tutorials", f"/fnirs-tutorials/interactive/{ILLUSTRATION}"),
        ("/fnirs-tutorials", "/fnirs-tutorials/interactive/sub/two.html"),
    ],
)
def test_a_frame_under_the_base_url_and_interactive_that_is_published_is_accepted(
    tmp_path: Path, base_url: str, src: str
) -> None:
    assert check_one(tmp_path, src, base_url) == []


@pytest.mark.parametrize("base_url", ["", "/fnirs-tutorials"])
@pytest.mark.parametrize(
    ("src_template", "reason"),
    [
        ("frames/live.html", "it is a relative path"),
        ("interactive/{f}", "it is a relative path"),
        ("https://example.org/live.html", "it has a scheme"),
        ("http://example.org/live.html", "it has a scheme"),
        ("data:text/html,x", "it has a scheme"),
        ("//example.org/live.html", "it names another host"),
        ("/frames/live.html", "it is a root path outside '/interactive/'"),
        ("{b}/frames/live.html", "it is a root path outside '/interactive/'"),
        ("{b}/interactive/missing.html", "the build did not publish it"),
        ("{b}/interactive/sub", "the build did not publish it"),
        ("{b}/interactive/../{f}", "leaves the folder"),
        ("{b}/interactive/sub/../{f}", "leaves the folder"),
        ("{b}/interactive/./{f}", "leaves the folder"),
        ("{b}/interactive//{f}", "leaves the folder"),
        ("{b}/interactive/", "empty"),
        ("{b}/interactive/{f}?x=1", "a query, a fragment"),
        ("{b}/interactive/{f}#top", "a query, a fragment"),
        ("{b}/interactive/%2e%2e/{f}", "a query, a fragment, a percent sign"),
        ("{b}/interactive/sub\\two.html", "a backslash"),
    ],
)
def test_a_refused_frame_is_named_with_its_reason_and_what_is_accepted(
    tmp_path: Path, base_url: str, src_template: str, reason: str
) -> None:
    src = src_template.format(b=base_url, f=ILLUSTRATION)
    problems = check_one(tmp_path, src, base_url)
    assert len(problems) == 1
    assert problems[0].exit_code == build_book.EXIT_FRAME == 18
    message = str(problems[0])
    assert "page '/book/frame_relative.md'" in message
    assert f"source is {src!r}" in message
    assert reason in message
    assert f"accepted only when its source is {base_url}/interactive/<path>" in message


def test_a_source_without_the_base_url_is_refused_when_the_build_has_one(
    tmp_path: Path,
) -> None:
    problems = check_one(tmp_path, f"/interactive/{ILLUSTRATION}", "/fnirs-tutorials")
    assert len(problems) == 1
    assert "does not start with the base URL '/fnirs-tutorials'" in str(problems[0])


def test_a_source_with_a_base_url_is_refused_when_the_build_has_none(
    tmp_path: Path,
) -> None:
    src = f"/fnirs-tutorials/interactive/{ILLUSTRATION}"
    problems = check_one(tmp_path, src, "")
    assert len(problems) == 1
    assert "root path outside" in str(problems[0])


def test_the_html_that_is_deployed_must_hold_the_same_source(tmp_path: Path) -> None:
    src = f"/fnirs-tutorials/interactive/{ILLUSTRATION}"
    problems = check_one(
        tmp_path, src, "/fnirs-tutorials", html_src=f"/interactive/{ILLUSTRATION}"
    )
    # Two defects, one per direction: the page file's frame is not in the HTML, and the
    # HTML's frame (which misses the base URL) is not in the page file.
    assert len(problems) == 2
    assert all(p.exit_code == build_book.EXIT_FRAME for p in problems)
    assert any(
        "HTML page that is deployed does not hold an iframe" in str(p) for p in problems
    )
    assert any("does not start with the base URL" in str(p) for p in problems)


def test_a_page_without_a_built_html_file_is_refused(tmp_path: Path) -> None:
    html_dir = tmp_path / "_build" / "html"
    (html_dir / "interactive").mkdir(parents=True)
    (html_dir / "interactive" / ILLUSTRATION).write_text("x", encoding="utf-8")
    page = frame_page("frame_relative", f"/interactive/{ILLUSTRATION}")
    problems = build_book.check_frames(page["location"], page, "", html_dir)
    assert len(problems) == 1
    assert "HTML page that is deployed does not hold an iframe" in str(problems[0])


def test_the_html_of_the_index_page_is_read_at_the_root_of_the_site(
    tmp_path: Path,
) -> None:
    html_dir = tmp_path / "_build" / "html"
    src = f"/interactive/{ILLUSTRATION}"
    (html_dir / "interactive").mkdir(parents=True)
    (html_dir / "interactive" / ILLUSTRATION).write_text("x", encoding="utf-8")
    (html_dir / "index.html").write_text(html_with_frame(src), encoding="utf-8")
    page = frame_page("frame_relative", src)
    page["slug"] = "index"
    assert build_book.check_frames(page["location"], page, "", html_dir) == []


def test_without_a_built_site_every_frame_is_refused() -> None:
    page = frame_page("frame_relative", f"/interactive/{ILLUSTRATION}")
    problems = build_book.check_frames(page["location"], page, "", None)
    assert len(problems) == 1
    assert "no built site was given" in str(problems[0])


def test_every_frame_of_a_page_is_reported(tmp_path: Path) -> None:
    html_dir = tmp_path / "_build" / "html"
    good = f"/interactive/{ILLUSTRATION}"
    publish(html_dir, "frame-relative", good, (ILLUSTRATION,))
    page = frame_page("frame_relative", good)
    children = page["mdast"]["children"][0]["children"]
    children.append(dict(children[0], src="/other.html"))
    children.append(dict(children[0], src="https://example.org/x"))
    problems = build_book.check_frames(page["location"], page, "", html_dir)
    assert len(problems) == 2


def test_a_frame_nested_in_another_node_is_found() -> None:
    page = load_fixture("frame_relative")
    frame = page["mdast"]["children"][0]["children"].pop()
    page["mdast"]["children"][0]["children"].append(
        {"type": "admonition", "children": [{"type": "paragraph", "children": [frame]}]}
    )
    assert build_book.find_frame_sources(page["mdast"]) == [FRAME_SOURCE]


def test_a_node_that_only_looks_like_a_frame_is_not_one() -> None:
    tree = {"type": "root", "children": [{"type": "image", "src": "frames/live.html"}]}
    assert build_book.find_frame_sources(tree) == []
    assert build_book.find_frame_sources({"type": "iframe", "src": 3}) == []


def test_the_fixture_pages_hold_one_frame_each_with_the_source_as_written() -> None:
    sources = {
        stem: build_book.find_frame_sources(load_fixture(stem)["mdast"])
        for stem in ("frame_relative", "frame_root_relative", "frame_scheme", "no_code")
    }
    assert sources == {
        "frame_relative": [FRAME_SOURCE],
        "frame_root_relative": ["/frames/live.html"],
        "frame_scheme": ["https://example.org/live.html"],
        "no_code": [],
    }


@pytest.mark.parametrize(
    "stem", ["frame_relative", "frame_root_relative", "frame_scheme"]
)
def test_the_real_page_files_with_a_frame_outside_the_rule_are_refused(
    stem: str,
) -> None:
    page = load_fixture(stem)
    problems = build_book.check_built_pages({page["location"]: page}, [], [])
    assert [p.exit_code for p in problems] == [build_book.EXIT_FRAME]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ""),
        ("", ""),
        ("/fnirs-tutorials", "/fnirs-tutorials"),
        ("/fnirs-tutorials/", "/fnirs-tutorials"),
        ("/fnirs-tutorials//", "/fnirs-tutorials"),
        # Spaces are not stripped: MyST and the plugin receive them, so the script must
        # see them to refuse the value.
        ("  /fnirs-tutorials//  ", "  /fnirs-tutorials//  "),
        ("  ", "  "),
        ("/", ""),
    ],
)
def test_the_base_url_is_normalised(raw: str | None, expected: str) -> None:
    assert build_book.normalize_base_url(raw) == expected


# --------------------------------------------------------------------------------------
# The rule for a frame, through the entry point
# --------------------------------------------------------------------------------------


def set_frame_build(
    root: Path, src: str, files: tuple[str, ...] = (ILLUSTRATION,)
) -> Any:
    """A stand-in build that also writes the HTML site the frame points into."""
    page = frame_page("frame_relative", src)
    content_stem = "frame_relative"

    class Build(FakeBuild):
        def __call__(self, stage_dir: Path) -> int:
            code = super().__call__(stage_dir)
            # The page file of the fixture holds the source of the fixture; replace it
            # with the source under test, as the plugin would have done.
            path = stage_dir / "_build" / "site" / "content" / f"{content_stem}.json"
            path.write_text(json.dumps(page), encoding="utf-8")
            publish(stage_dir / "_build" / "html", page["slug"], src, files)
            return code

    return Build(["notebook_smoke", content_stem])


def test_a_published_frame_under_the_base_url_lets_the_book_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BASE_URL", "/fnirs-tutorials/")
    root = make_project(tmp_path, ["notebook_smoke", "frame_relative"])
    build = set_frame_build(root, f"/fnirs-tutorials/interactive/{ILLUSTRATION}")
    assert run_main(root, build) == 0


def test_a_published_frame_lets_the_book_build_when_no_base_url_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BASE_URL", raising=False)
    root = make_project(tmp_path, ["notebook_smoke", "frame_relative"])
    build = set_frame_build(root, f"/interactive/{ILLUSTRATION}")
    assert run_main(root, build) == 0


def test_a_frame_that_misses_the_base_url_stops_the_script_with_exit_18(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("BASE_URL", "/fnirs-tutorials")
    root = make_project(tmp_path, ["notebook_smoke", "frame_relative"])
    build = set_frame_build(root, f"/interactive/{ILLUSTRATION}")
    assert run_main(root, build) == build_book.EXIT_FRAME == 18
    assert "Book not built (exit 18)" in caplog.text
    assert "page '/book/frame_relative.md' holds a frame whose source is" in caplog.text


def test_a_frame_whose_target_is_not_published_stops_the_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("BASE_URL", raising=False)
    root = make_project(tmp_path, ["notebook_smoke", "frame_relative"])
    build = set_frame_build(root, f"/interactive/{ILLUSTRATION}", files=("other.html",))
    assert run_main(root, build) == build_book.EXIT_FRAME
    assert "the build did not publish it" in caplog.text


def test_a_frame_with_an_https_source_stops_the_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("BASE_URL", raising=False)
    root = make_project(tmp_path, ["notebook_smoke", "frame_scheme"])
    build = FakeBuild(["notebook_smoke", "frame_scheme"])
    assert run_main(root, build) == build_book.EXIT_FRAME
    assert "https://example.org/live.html" in caplog.text


def test_a_book_without_a_frame_still_builds(tmp_path: Path) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    assert run_main(root, FakeBuild(["notebook_smoke", "no_code"])) == 0


# --------------------------------------------------------------------------------------
# Staging the static folder, the plugin and the style file
# --------------------------------------------------------------------------------------


def add_to_myst(root: Path, project: dict[str, Any], site: dict[str, Any]) -> None:
    """Merge entries into the ``project`` and ``site`` blocks of ``myst.yml``."""
    path = root / "myst.yml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["project"].update(project)
    if site:
        document["site"] = site
    path.write_text(yaml.safe_dump(document), encoding="utf-8")


def write_files(root: Path, names: list[str]) -> None:
    """Write small files under ``root``."""
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")


def test_the_static_folder_the_plugin_and_the_style_file_are_staged_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    add_to_myst(
        root,
        {"static_files": ["interactive"], "plugins": ["plugins/p.mjs"]},
        {"options": {"style": "book/frames.css"}},
    )
    write_files(
        root,
        [
            "interactive/a.html",
            "interactive/sub/b.js",
            "plugins/p.mjs",
            "book/frames.css",
        ],
    )
    staged = build_book.stage_book(root, root / "_build" / "executed")
    for name in (
        "interactive/a.html",
        "interactive/sub/b.js",
        "plugins/p.mjs",
        "book/frames.css",
    ):
        assert (staged.stage_dir / name).is_file(), name
        assert f"Staged {name}" in caplog.text


def test_a_hidden_file_and_a_cache_folder_of_a_static_folder_are_not_staged(
    tmp_path: Path,
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    add_to_myst(root, {"static_files": ["interactive"]}, {})
    write_files(
        root,
        [
            "interactive/a.html",
            "interactive/.DS_Store",
            "interactive/.hidden/c.html",
            "interactive/__pycache__/d.pyc",
        ],
    )
    staged = build_book.stage_book(root, root / "_build" / "executed")
    assert (staged.stage_dir / "interactive" / "a.html").is_file()
    assert not (staged.stage_dir / "interactive" / ".DS_Store").exists()
    assert not (staged.stage_dir / "interactive" / ".hidden").exists()
    assert not (staged.stage_dir / "interactive" / "__pycache__").exists()


@pytest.mark.parametrize(
    ("entry", "fragment"),
    [
        ("absent", "names no file or folder"),
        ("../outside", "points outside the repository"),
        ("data", "never staged"),
        ("data/fresh", "never staged"),
        ("_build", "never staged"),
        (".", "the repository itself"),
    ],
)
def test_a_static_entry_that_cannot_be_staged_is_a_build_error(
    tmp_path: Path, entry: str, fragment: str
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    add_to_myst(root, {"static_files": [entry]}, {})
    (root / "data" / "fresh").mkdir(parents=True)
    with pytest.raises(build_book.BookError) as stop:
        build_book.stage_book(root, root / "_build" / "executed")
    assert stop.value.exit_code == build_book.EXIT_FILE_ERROR
    assert fragment in str(stop.value)


def test_static_files_that_is_not_a_list_of_paths_is_a_build_error(
    tmp_path: Path,
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    add_to_myst(root, {"static_files": "interactive"}, {})
    with pytest.raises(build_book.BookError) as stop:
        build_book.stage_book(root, root / "_build" / "executed")
    assert stop.value.exit_code == build_book.EXIT_FILE_ERROR
    assert "must be a list of paths" in str(stop.value)


def test_a_static_folder_that_cannot_be_copied_exits_with_a_named_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    add_to_myst(root, {"static_files": ["interactive"]}, {})
    write_files(root, ["interactive/a.html"])
    real_copy = shutil.copy2

    def refuse(source: Any, destination: Any) -> Any:
        if str(source).endswith("a.html"):
            raise PermissionError(13, "Permission denied", str(destination))
        return real_copy(source, destination)

    monkeypatch.setattr(shutil, "copy2", refuse)
    assert run_main(root, FakeBuild(["notebook_smoke", "no_code"])) == (
        build_book.EXIT_FILE_ERROR
    )
    assert "cannot stage interactive/a.html" in caplog.text


def test_rglob_does_not_list_the_files_behind_a_link_to_a_folder(
    tmp_path: Path,
) -> None:
    """Measured on this Python: a link to a folder is listed, but not entered.

    The folder is therefore silently left out of what is staged, which is why the
    refusal of a link must not rely on the files that ``rglob`` returns.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("x", encoding="utf-8")
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "linked").symlink_to(outside, target_is_directory=True)
    listed = sorted(p.name for p in folder.rglob("*"))
    assert listed == ["linked"]
    assert [p for p in folder.rglob("*") if p.is_file()] == []


@pytest.mark.parametrize(
    "target",
    ["data/fresh/secret.csv", "book/page.md", "../outside-repo.txt"],
)
def test_a_link_to_a_file_out_of_its_static_folder_is_refused(
    tmp_path: Path, target: str
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    add_to_myst(root, {"static_files": ["interactive"]}, {})
    write_files(root, ["interactive/a.html", "data/fresh/secret.csv", "book/page.md"])
    (tmp_path / "outside-repo.txt").write_text("x", encoding="utf-8")
    link = root / "interactive" / "link.txt"
    link.symlink_to((root / target).resolve())
    with pytest.raises(build_book.BookError) as stop:
        build_book.stage_book(root, root / "_build" / "executed")
    assert stop.value.exit_code == build_book.EXIT_FILE_ERROR
    assert "link.txt" in str(stop.value)
    assert Path(target).name in str(stop.value)
    assert not (root / "_build" / "executed" / "data").exists()


@pytest.mark.parametrize("target", ["data/fresh", "book", "../outside-dir"])
def test_a_link_to_a_folder_out_of_its_static_folder_is_refused(
    tmp_path: Path, target: str
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    add_to_myst(root, {"static_files": ["interactive"]}, {})
    write_files(root, ["interactive/a.html", "data/fresh/secret.csv"])
    (tmp_path / "outside-dir").mkdir()
    (root / "interactive" / "linked").symlink_to(
        (root / target).resolve(), target_is_directory=True
    )
    with pytest.raises(build_book.BookError) as stop:
        build_book.stage_book(root, root / "_build" / "executed")
    assert stop.value.exit_code == build_book.EXIT_FILE_ERROR
    assert "linked" in str(stop.value)


def test_a_link_that_stays_inside_its_static_folder_is_accepted(
    tmp_path: Path,
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    add_to_myst(root, {"static_files": ["interactive"]}, {})
    write_files(root, ["interactive/a.html"])
    (root / "interactive" / "alias.html").symlink_to(root / "interactive" / "a.html")
    staged = build_book.stage_book(root, root / "_build" / "executed")
    assert Path("interactive/a.html") in staged.copied


def test_a_staged_path_that_cannot_be_placed_under_the_root_is_a_book_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    (tmp_path / "elsewhere.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        build_book, "find_static_files", lambda *_args: [tmp_path / "elsewhere.txt"]
    )
    with pytest.raises(build_book.BookError) as stop:
        build_book.stage_book(root, root / "_build" / "executed")
    assert stop.value.exit_code == build_book.EXIT_FILE_ERROR
    assert "elsewhere.txt" in str(stop.value)


@pytest.mark.parametrize(
    "entry", ["book", "book/no_code.md", "book/notebook_smoke.ipynb"]
)
def test_a_static_entry_that_holds_a_page_of_the_book_is_refused_before_any_removal(
    tmp_path: Path, entry: str
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    add_to_myst(root, {"static_files": [entry]}, {})
    executed = root / "_build" / "executed"
    notebook_copies = sorted(executed.rglob("*.ipynb"))
    assert notebook_copies
    with pytest.raises(build_book.BookError) as stop:
        build_book.stage_book(root, executed)
    assert stop.value.exit_code == build_book.EXIT_FILE_ERROR
    message = str(stop.value)
    assert repr(entry) in message
    assert "book/" in message and ("no_code.md" in message or ".ipynb" in message)
    assert sorted(executed.rglob("*.ipynb")) == notebook_copies
    assert not (executed / "myst.yml").exists()


def test_a_static_entry_that_is_the_staging_folder_or_a_parent_of_it_is_refused(
    tmp_path: Path,
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    executed = root / "out" / "stage"
    shutil.copytree(root / "_build" / "executed", executed)
    for entry in ("out", "out/stage"):
        add_to_myst(root, {"static_files": [entry]}, {})
        with pytest.raises(build_book.BookError) as stop:
            build_book.stage_book(root, executed)
        assert stop.value.exit_code == build_book.EXIT_FILE_ERROR
        assert repr(entry) in str(stop.value)
        assert "staging folder" in str(stop.value)
        assert sorted(executed.rglob("*.ipynb"))


def test_the_target_of_a_frame_is_staged_only_through_the_static_folder(
    tmp_path: Path,
) -> None:
    root = make_project(
        tmp_path,
        ["notebook_smoke", "frame_relative"],
        texts={"frame_relative": ":::{iframe} /interactive/a.html\n:::\n"},
    )
    write_files(root, ["interactive/a.html"])
    staged = build_book.stage_book(root, root / "_build" / "executed")
    assert not (staged.stage_dir / "interactive" / "a.html").exists()
    add_to_myst(root, {"static_files": ["interactive"]}, {})
    staged = build_book.stage_book(root, root / "_build" / "executed")
    assert (staged.stage_dir / "interactive" / "a.html").is_file()


# --------------------------------------------------------------------------------------
# The deployed HTML is read for every frame it holds
# --------------------------------------------------------------------------------------


def write_html(html_dir: Path, slug: str, body: str) -> None:
    """Write the deployed HTML of a page whose body is given."""
    page = html_dir / slug / "index.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(f"<div>{body}</div>", encoding="utf-8")


def page_without_frame() -> dict[str, Any]:
    """A real page file that holds no frame."""
    return load_fixture("no_code")


@pytest.mark.parametrize(
    "tag",
    [
        '<iframe src="https://example.org/x" title="t"></iframe>',
        '<iframe title="t" src="/other.html"></iframe>',
        '<iframe src="/interactive/hrf.html"></iframe>',
    ],
)
def test_a_frame_in_the_html_that_the_page_file_does_not_hold_is_refused(
    tmp_path: Path, tag: str
) -> None:
    html_dir = tmp_path / "_build" / "html"
    (html_dir / "interactive").mkdir(parents=True)
    (html_dir / "interactive" / ILLUSTRATION).write_text("x", encoding="utf-8")
    page = page_without_frame()
    write_html(html_dir, page["slug"], tag)
    problems = build_book.check_frames(page["location"], page, "", html_dir)
    assert len(problems) == 1
    assert problems[0].exit_code == build_book.EXIT_FRAME
    assert page["location"] in str(problems[0])


@pytest.mark.parametrize(
    "tag",
    [
        "<iframe src='/interactive/hrf.html'></iframe>",
        "<iframe src=/interactive/hrf.html></iframe>",
    ],
)
def test_a_frame_whose_src_is_not_written_in_double_quotes_is_refused(
    tmp_path: Path, tag: str
) -> None:
    html_dir = tmp_path / "_build" / "html"
    publish(html_dir, "frame-relative", f"/interactive/{ILLUSTRATION}", (ILLUSTRATION,))
    page = frame_page("frame_relative", f"/interactive/{ILLUSTRATION}")
    write_html(html_dir, page["slug"], tag)
    problems = build_book.check_frames(page["location"], page, "", html_dir)
    assert problems
    assert all(p.exit_code == build_book.EXIT_FRAME for p in problems)


def test_a_double_quoted_src_is_read_whatever_the_case_of_the_tag(
    tmp_path: Path,
) -> None:
    html_dir = tmp_path / "_build" / "html"
    publish(html_dir, "frame-relative", f"/interactive/{ILLUSTRATION}", (ILLUSTRATION,))
    page = frame_page("frame_relative", f"/interactive/{ILLUSTRATION}")
    write_html(
        html_dir, page["slug"], f'<IFRAME SRC = "/interactive/{ILLUSTRATION}"></IFRAME>'
    )
    assert build_book.check_frames(page["location"], page, "", html_dir) == []


@pytest.mark.parametrize(
    "tag",
    [
        "<iframe srcdoc=\"<script src='https://cdn.example.org/x.js'></script>\"></iframe>",
        "<iframe srcdoc='<p>x</p>'></iframe>",
        "<iframe SRCDOC=x></iframe>",
        '<iframe src="/interactive/hrf.html" srcdoc="<p>x</p>"></iframe>',
    ],
)
def test_an_iframe_with_srcdoc_is_refused(tmp_path: Path, tag: str) -> None:
    html_dir = tmp_path / "_build" / "html"
    (html_dir / "interactive").mkdir(parents=True)
    (html_dir / "interactive" / ILLUSTRATION).write_text("x", encoding="utf-8")
    page = page_without_frame()
    write_html(html_dir, page["slug"], tag)
    problems = build_book.check_frames(page["location"], page, "", html_dir)
    assert problems
    assert all(p.exit_code == build_book.EXIT_FRAME for p in problems)
    assert any("srcdoc" in str(p) for p in problems)


@pytest.mark.parametrize(
    "body",
    [
        '<iframe src="https://example.org/x" title=don\'t></iframe>',
        '<iframe title="a src="https://example.org/x"></iframe>',
        '<p>text</p><IFRAME src="https://example.org/x',
        '<iframe src="/a"></iframe><iframe src="/b>',
    ],
)
def test_a_page_with_more_iframe_tags_than_the_reader_parsed_is_refused(
    tmp_path: Path, body: str
) -> None:
    html_dir = tmp_path / "_build" / "html"
    html_dir.mkdir(parents=True)
    page = page_without_frame()
    write_html(html_dir, page["slug"], body)
    problems = build_book.check_frames(page["location"], page, "", html_dir)
    assert len(problems) >= 1
    assert all(p.exit_code == build_book.EXIT_FRAME for p in problems)
    assert any("occurs" in str(p) and "parsed" in str(p) for p in problems)


def test_a_sound_frame_is_counted_once_and_accepted(tmp_path: Path) -> None:
    html_dir = tmp_path / "_build" / "html"
    publish(html_dir, "frame-relative", f"/interactive/{ILLUSTRATION}", (ILLUSTRATION,))
    page = frame_page("frame_relative", f"/interactive/{ILLUSTRATION}")
    assert build_book.check_frames(page["location"], page, "", html_dir) == []


def test_a_page_without_frame_and_without_html_is_not_a_frame_error(
    tmp_path: Path,
) -> None:
    page = page_without_frame()
    html_dir = tmp_path / "_build" / "html"
    html_dir.mkdir(parents=True)
    assert build_book.check_frames(page["location"], page, "", html_dir) == []


def test_a_page_without_frame_whose_html_holds_none_is_accepted(tmp_path: Path) -> None:
    page = page_without_frame()
    html_dir = tmp_path / "_build" / "html"
    write_html(html_dir, page["slug"], "<p>no frame</p>")
    assert build_book.check_frames(page["location"], page, "", html_dir) == []


def test_a_raw_frame_in_the_html_stops_the_script_with_exit_18(
    tmp_path: Path,
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])

    class Build(FakeBuild):
        def __call__(self, stage_dir: Path) -> int:
            code = super().__call__(stage_dir)
            write_html(
                stage_dir / "_build" / "html",
                load_fixture("no_code")["slug"],
                '<iframe src="https://example.org/x"></iframe>',
            )
            return code

    assert run_main(root, Build(["notebook_smoke", "no_code"])) == build_book.EXIT_FRAME


# --------------------------------------------------------------------------------------
# The target of a frame exists under its exact name
# --------------------------------------------------------------------------------------


def test_a_frame_target_whose_case_differs_from_the_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On the Mac ``is_file`` ignores case; here it is made to, so Linux checks it too."""
    real = build_book.is_file_or_absent

    def ignoring_case(path: Path) -> bool:
        if real(path):
            return True
        parent = path.parent
        return parent.is_dir() and any(
            p.name.lower() == path.name.lower() and p.is_file()
            for p in parent.iterdir()
        )

    monkeypatch.setattr(build_book, "is_file_or_absent", ignoring_case)
    problems = check_one(
        tmp_path,
        "/interactive/HRF.html",
        "",
        files=(ILLUSTRATION,),
    )
    assert len(problems) == 1
    assert "does not exist" in str(problems[0])


def test_a_frame_target_whose_folder_differs_in_case_is_refused(
    tmp_path: Path,
) -> None:
    problems = check_one(
        tmp_path, "/interactive/SUB/two.html", "", files=("sub/two.html",)
    )
    assert len(problems) == 1


# --------------------------------------------------------------------------------------
# The base URL is checked before the build
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "fnirs-tutorials",
        "https://example.org/book",
        "//evil.example/x",
        "/a b",
        "/a?x=1",
    ],
)
def test_a_base_url_that_is_not_a_root_path_is_refused_before_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("BASE_URL", raw)
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    build = FakeBuild(["notebook_smoke", "no_code"])
    assert run_main(root, build) == build_book.EXIT_BAD_BASE_URL
    assert build.calls == []
    assert not (root / "_build" / "executed" / "myst.yml").exists()


def test_the_refusal_of_a_base_url_gives_the_value_and_the_accepted_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BASE_URL", "fnirs-tutorials")
    with pytest.raises(build_book.BookError) as stop:
        build_book.read_base_url()
    assert "'fnirs-tutorials'" in str(stop.value)
    assert "/fnirs-tutorials" in str(stop.value)


@pytest.mark.parametrize(
    "raw", ["  ", " ", " /fnirs-tutorials", "/fnirs-tutorials ", "\t/a", "/a\n"]
)
def test_a_base_url_with_a_space_is_refused_as_myst_and_the_plugin_read_it(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """Measured on a real build: MyST fails on ' /x' and on '  ', and the plugin keeps them."""
    monkeypatch.setenv("BASE_URL", raw)
    with pytest.raises(build_book.BookError) as stop:
        build_book.read_base_url()
    assert stop.value.exit_code == build_book.EXIT_BAD_BASE_URL
    assert repr(raw) in str(stop.value)


@pytest.mark.parametrize("raw", [None, "", "/", "/fnirs-tutorials", "/a/b/"])
def test_a_base_url_that_is_empty_or_a_root_path_is_read(
    monkeypatch: pytest.MonkeyPatch, raw: str | None
) -> None:
    if raw is None:
        monkeypatch.delenv("BASE_URL", raising=False)
    else:
        monkeypatch.setenv("BASE_URL", raw)
    assert build_book.read_base_url() == build_book.normalize_base_url(raw)


# --------------------------------------------------------------------------------------
# The staged copy of a static entry is cleared before the copy
# --------------------------------------------------------------------------------------


def test_a_deleted_illustration_does_not_stay_in_the_staging_folder(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    add_to_myst(root, {"static_files": ["interactive"]}, {})
    write_files(root, ["interactive/a.html", "interactive/gone.html"])
    executed = root / "_build" / "executed"
    build_book.stage_book(root, executed)
    assert (executed / "interactive" / "gone.html").is_file()
    (root / "interactive" / "gone.html").unlink()
    caplog.clear()
    build_book.stage_book(root, executed)
    assert not (executed / "interactive" / "gone.html").exists()
    assert (executed / "interactive" / "a.html").is_file()
    assert f"Removed {executed / 'interactive'} from an earlier build" in caplog.text


def test_the_removal_of_a_static_copy_cannot_leave_the_staging_folder(
    tmp_path: Path,
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    add_to_myst(root, {"static_files": ["interactive"]}, {})
    write_files(root, ["interactive/a.html"])
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("x", encoding="utf-8")
    executed = root / "_build" / "executed"
    executed.mkdir(parents=True, exist_ok=True)
    (executed / "interactive").symlink_to(victim, target_is_directory=True)
    build_book.stage_book(root, executed)
    assert (victim / "keep.txt").is_file()
    assert not (victim / "a.html").exists()


# --------------------------------------------------------------------------------------
# File probes on names that come from the content of the book
# --------------------------------------------------------------------------------------

LONG_NAME = "a" * 300


def test_the_newest_helper_file_is_found_under_a_folder_named_pycache(
    tmp_path: Path,
) -> None:
    """Only the parts below ``src/`` count: the checkout may sit under any folder."""
    root = tmp_path / "__pycache__" / "repo"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "a.py").write_text("x", encoding="utf-8")
    (root / "src" / "pkg" / "__pycache__").mkdir()
    (root / "src" / "pkg" / "__pycache__" / "a.pyc").write_text("x", encoding="utf-8")
    assert build_book._newest_helper_file(root) == root / "src" / "pkg" / "a.py"


def test_a_name_the_file_system_refuses_is_not_a_file(tmp_path: Path) -> None:
    assert build_book.is_file_or_absent(tmp_path / LONG_NAME) is False
    assert build_book.is_file_or_absent(tmp_path / "missing.md") is False
    present = tmp_path / "present.md"
    present.write_text("x", encoding="utf-8")
    assert build_book.is_file_or_absent(present) is True
    assert build_book.is_file_or_absent(tmp_path) is False


def test_an_entry_of_the_table_of_contents_the_file_system_refuses_is_reported_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    names = ["book/no_code.md", f"book/{LONG_NAME}.md"]
    monkeypatch.setattr(build_book, "read_toc_files", lambda _path: (names, []))
    with caplog.at_level(logging.WARNING):
        staged = build_book.stage_book(root, root / "_build" / "executed")
    assert (
        f"Page book/{LONG_NAME}.md of the table of contents does not exist"
        in caplog.text
    )
    assert Path("book/no_code.md") in staged.copied


def test_an_entry_of_the_table_of_contents_the_notebook_reader_cannot_test_exits_17(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    document = yaml.safe_load((root / "myst.yml").read_text(encoding="utf-8"))
    document["project"]["toc"].append({"file": f"book/{LONG_NAME}.md"})
    (root / "myst.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    code = run_main(root, FakeBuild(["notebook_smoke", "no_code"]))
    assert code == build_book.EXIT_FILE_ERROR == 17
    assert "Book not built (exit 17)" in caplog.text
    assert "OSError on " in caplog.text
    assert LONG_NAME in caplog.text
    assert "File name too long" in caplog.text


def raise_from_staging(monkeypatch: pytest.MonkeyPatch, error: OSError) -> None:
    def refuse(*_args: Any) -> None:
        raise error

    monkeypatch.setattr(build_book, "stage_book", refuse)


def test_a_file_error_the_script_did_not_guard_exits_17_with_the_whole_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    raise_from_staging(monkeypatch, OSError(13, "Permission denied", "/some/file.md"))
    assert run_main(root, FakeBuild([])) == build_book.EXIT_FILE_ERROR
    assert (
        "PermissionError on /some/file.md: [Errno 13] Permission denied" in caplog.text
    )
    assert "Traceback (most recent call last)" in caplog.text


def test_a_file_error_without_a_file_name_does_not_print_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    raise_from_staging(monkeypatch, OSError("disk is full"))
    assert run_main(root, FakeBuild([])) == build_book.EXIT_FILE_ERROR
    assert "OSError on no named file: disk is full" in caplog.text
    assert "None" not in caplog.text


def test_a_file_error_with_two_file_names_prints_both(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    raise_from_staging(
        monkeypatch, OSError(18, "Invalid cross-device link", "/a/from", None, "/b/to")
    )
    assert run_main(root, FakeBuild([])) == build_book.EXIT_FILE_ERROR
    assert "OSError on /a/from and /b/to:" in caplog.text


# --------------------------------------------------------------------------------------
# The workflow
# --------------------------------------------------------------------------------------

WORKFLOW = REPO_ROOT / ".github" / "workflows" / "book.yml"


def load_workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_only_a_push_to_main_deploys() -> None:
    deploy = load_workflow()["jobs"]["deploy"]
    assert (
        deploy["if"] == "github.ref == 'refs/heads/main' && github.event_name == 'push'"
    )
    assert deploy["needs"] == "build"


def test_the_runs_are_grouped_by_event_and_ref_and_main_is_never_cancelled() -> None:
    concurrency = load_workflow()["concurrency"]
    assert concurrency["group"] == "book-${{ github.event_name }}-${{ github.ref }}"
    assert concurrency["cancel-in-progress"] == "${{ github.ref != 'refs/heads/main' }}"
    assert load_workflow()["jobs"]["deploy"]["concurrency"] == {
        "group": "pages",
        "cancel-in-progress": False,
    }


def test_the_deploy_job_keeps_its_two_permissions_and_has_no_token_step() -> None:
    deploy = load_workflow()["jobs"]["deploy"]
    assert deploy["permissions"] == {"pages": "write", "id-token": "write"}
    check = deploy["steps"][0]
    assert "token" not in json.dumps(check).lower()


def test_the_stale_run_step_comes_before_the_deploy_step_and_gates_it() -> None:
    steps = load_workflow()["jobs"]["deploy"]["steps"]
    ids = [step.get("id") for step in steps]
    assert ids == ["head", "deployment"]
    check, deploy = steps
    assert deploy["if"] == "steps.head.outputs.current == 'true'"
    assert deploy["uses"].startswith("actions/deploy-pages@")
    script = check["run"]
    assert 'git ls-remote "$REMOTE" refs/heads/main' in script
    assert check["env"]["REMOTE"] == "${{ github.server_url }}/${{ github.repository }}"
    assert "set -euo pipefail" in script
    # An empty answer fails the job, and a different head only sets the output.
    assert 'if [ -z "$head" ]; then' in script and "exit 1" in script
    assert 'if [ "$head" = "${GITHUB_SHA}" ]' in script
    assert script.count("current=false") == 1
    # The first attempt deploys at once, before git ls-remote is asked.
    first = script.index('if [ "${GITHUB_RUN_ATTEMPT}" = "1" ]; then')
    assert first < script.index("git ls-remote")
    assert script.count("current=true") == 2
    assert (
        script.index("current=true")
        < script.index("exit 0")
        < script.index("git ls-remote")
    )


# --------------------------------------------------------------------------------------
# One identity for a path: the spelling the disk gives it
# --------------------------------------------------------------------------------------


def _fold_to_disk(path: Path | str) -> Path | None:
    """Return the real path that ``path`` names on a disk that ignores case, or None."""
    wanted = Path(path)
    current = Path(wanted.anchor)
    for part in wanted.parts[1:]:
        try:
            names = os.listdir(current)
        except OSError:
            return None
        if part in names:
            current = current / part
            continue
        folded = [name for name in names if name.casefold() == part.casefold()]
        if not folded:
            return None
        current = current / folded[0]
    return current


@pytest.fixture
def case_blind_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the file system behave as APFS does by default: it ignores case.

    The Mac of the lab has such a disk and the Linux runner has not, so a test that
    depends on it has to mean the same on both. ``Path.resolve`` keeps the spelling it
    was given on such a disk (measured on APFS), so it is left alone; the probes that
    answer "does this exist" and "is this the same file" ignore case.
    """
    real_is_file = Path.is_file
    real_is_dir = Path.is_dir
    real_exists = Path.exists
    real_samefile = os.path.samefile

    def probe(real: Any) -> Any:
        def wrapped(self: Path) -> bool:
            folded = _fold_to_disk(self)
            return folded is not None and real(folded)

        return wrapped

    def samefile(first: Any, second: Any) -> bool:
        one, two = _fold_to_disk(first), _fold_to_disk(second)
        return one is not None and two is not None and real_samefile(one, two)

    monkeypatch.setattr(Path, "is_file", probe(real_is_file))
    monkeypatch.setattr(Path, "is_dir", probe(real_is_dir))
    monkeypatch.setattr(Path, "exists", probe(real_exists))
    monkeypatch.setattr(os.path, "samefile", samefile)


@pytest.fixture
def case_sensitive_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make two spellings that differ in case two different files, as on Linux."""
    real_samefile = os.path.samefile

    def samefile(first: Any, second: Any) -> bool:
        return str(first) == str(second) and real_samefile(first, second)

    monkeypatch.setattr(os.path, "samefile", samefile)


def test_a_path_spelled_as_the_disk_names_it_is_returned_as_it_is(
    tmp_path: Path,
) -> None:
    write_files(tmp_path, ["book/page.md"])
    path = tmp_path.resolve() / "book" / "page.md"
    assert build_book.on_disk(path) == path


@pytest.mark.usefixtures("case_blind_disk")
def test_on_a_disk_that_ignores_case_each_part_takes_the_spelling_of_its_listing(
    tmp_path: Path,
) -> None:
    write_files(tmp_path, ["book/Sub/page.md"])
    asked = tmp_path.resolve() / "BOOK" / "sub" / "PAGE.md"
    assert build_book.on_disk(asked) == tmp_path.resolve() / "book" / "Sub" / "page.md"


@pytest.mark.usefixtures("case_blind_disk")
def test_a_part_that_does_not_exist_is_kept_as_given(tmp_path: Path) -> None:
    write_files(tmp_path, ["book/page.md"])
    asked = tmp_path.resolve() / "BOOK" / "Absent" / "Deeper.md"
    assert (
        build_book.on_disk(asked)
        == tmp_path.resolve() / "book" / "Absent" / "Deeper.md"
    )


@pytest.mark.usefixtures("case_sensitive_disk")
def test_on_a_case_sensitive_disk_a_path_that_differs_in_case_is_not_changed(
    tmp_path: Path,
) -> None:
    write_files(tmp_path, ["book/page.md"])
    asked = tmp_path.resolve() / "BOOK" / "Page.md"
    assert build_book.on_disk(asked) == asked


@pytest.mark.usefixtures("case_sensitive_disk")
def test_on_a_case_sensitive_disk_two_names_that_differ_in_case_stay_two_files(
    tmp_path: Path,
) -> None:
    write_files(tmp_path, ["dir/a.txt"])
    if (tmp_path / "dir" / "A.TXT").exists():
        pytest.skip("this disk ignores case, so two such files cannot be written")
    write_files(tmp_path, ["dir/A.TXT"])
    for name in ("a.txt", "A.TXT"):
        path = tmp_path.resolve() / "dir" / name
        assert build_book.on_disk(path) == path


def test_a_link_is_followed_and_its_target_is_the_answer(tmp_path: Path) -> None:
    write_files(tmp_path, ["real/target.txt"])
    link = tmp_path / "alias.txt"
    link.symlink_to(tmp_path / "real" / "target.txt")
    assert build_book.on_disk(link) == tmp_path.resolve() / "real" / "target.txt"


@pytest.mark.usefixtures("case_blind_disk")
def test_a_path_that_cannot_be_listed_is_kept_as_given(tmp_path: Path) -> None:
    write_files(tmp_path, ["file.txt"])
    asked = tmp_path.resolve() / "FILE.txt" / "below"
    assert build_book.on_disk(asked) == tmp_path.resolve() / "file.txt" / "below"


# --------------------------------------------------------------------------------------
# Every path that enters the script is checked once, and a spelling the disk does not
# give is refused as the case-sensitive host would
# --------------------------------------------------------------------------------------


def snapshot(folder: Path) -> dict[str, tuple[Any, ...]]:
    """Describe every entry under ``folder`` without following a link.

    A folder is ``("dir",)``, a link is ``("link", text of the link)``, and a file is
    ``("file", bytes, st_mtime_ns)``, so that a rewrite with the same bytes, a link that
    is pointed elsewhere and a folder that is replaced by a link are all seen.
    """
    if not os.path.lexists(folder):
        return {}
    found: dict[str, tuple[Any, ...]] = {}
    for current, folders, files in os.walk(folder, followlinks=False):
        for name in sorted(folders + files):
            path = Path(current) / name
            key = path.relative_to(folder).as_posix()
            if path.is_symlink():
                found[key] = ("link", os.readlink(path))
            elif path.is_dir():
                found[key] = ("dir",)
            else:
                found[key] = ("file", path.read_bytes(), path.stat().st_mtime_ns)
    return found


def project_with_stale_stage(tmp_path: Path, static: list[str] | None = None) -> Path:
    """Make a project whose staging folder holds what an earlier build left."""
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    write_files(root, ["interactive/a.html", "interactive/Vendor/d3.js"])
    if static is not None:
        add_to_myst(root, {"static_files": static}, {})
    write_files(
        root / "_build" / "executed",
        [
            "_build/site/content/stale.json",
            "_build/html/stale.html",
            "interactive/old.html",
        ],
    )
    return root


def refuse(root: Path, executed: Path) -> build_book.BookError:
    """Run the whole build on a project and return the refusal."""
    runner = FakeBuild([])
    with pytest.raises(build_book.BookError) as stop:
        build_book.build_book(root, executed, runner)
    assert runner.calls == []
    return stop.value


@pytest.mark.usefixtures("case_blind_disk")
@pytest.mark.parametrize(
    "entry", ["Interactive", "INTERACTIVE/Vendor", "interactive/VENDOR"]
)
def test_a_static_entry_spelled_with_another_case_is_refused(
    tmp_path: Path, entry: str
) -> None:
    root = project_with_stale_stage(tmp_path, [entry])
    error = refuse(root, root / "_build" / "executed")
    assert error.exit_code == build_book.EXIT_FILE_ERROR
    assert repr(entry) in str(error)
    assert "spelled" in str(error)


@pytest.mark.usefixtures("case_blind_disk")
def test_a_static_entry_under_data_in_another_case_is_refused(tmp_path: Path) -> None:
    root = project_with_stale_stage(tmp_path, ["Data/fresh"])
    write_files(root, ["data/fresh/small.csv"])
    error = refuse(root, root / "_build" / "executed")
    assert error.exit_code == build_book.EXIT_FILE_ERROR
    assert "Data/fresh" in str(error)
    assert not (root / "_build" / "executed" / "data").exists()


@pytest.mark.usefixtures("case_blind_disk")
def test_a_page_of_the_table_of_contents_spelled_with_another_case_is_refused(
    tmp_path: Path,
) -> None:
    root = project_with_stale_stage(tmp_path)
    path = root / "myst.yml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("book/no_code.md", "Book/No_Code.md"),
        encoding="utf-8",
    )
    error = refuse(root, root / "_build" / "executed")
    assert error.exit_code == build_book.EXIT_BAD_TOC
    assert "Book/No_Code.md" in str(error)
    assert "book/no_code.md" in str(error)


@pytest.mark.usefixtures("case_blind_disk")
@pytest.mark.parametrize("reference", ["/Images/FIG.png", "../Images/fig.png"])
def test_a_reference_of_a_page_spelled_with_another_case_is_refused(
    tmp_path: Path, reference: str
) -> None:
    root = make_project(
        tmp_path,
        ["notebook_smoke", "no_code"],
        texts={"no_code": f"![figure]({reference})\n"},
    )
    write_files(root, ["images/fig.png"])
    executed = root / "_build" / "executed"
    before = snapshot(executed)
    with pytest.raises(build_book.BookError) as stop:
        build_book.stage_book(root, executed)
    assert stop.value.exit_code == build_book.EXIT_FILE_ERROR
    message = str(stop.value)
    assert "no_code.md" in message
    assert repr(reference) in message
    assert "images/fig.png" in message
    assert snapshot(executed) == before


@pytest.mark.usefixtures("case_blind_disk")
def test_a_reference_of_a_notebook_cell_spelled_with_another_case_is_refused(
    tmp_path: Path,
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    (root / NOTEBOOK).write_text(
        _notebook_with_markdown("![figure](/Images/FIG.png)"), encoding="utf-8"
    )
    write_files(root, ["images/fig.png"])
    stamp = (root / NOTEBOOK).stat().st_mtime + 100
    os.utime(root / "_build" / "executed" / NOTEBOOK, (stamp, stamp))
    with pytest.raises(build_book.BookError) as stop:
        build_book.stage_book(root, root / "_build" / "executed")
    assert stop.value.exit_code == build_book.EXIT_FILE_ERROR
    assert "notebook_smoke.ipynb" in str(stop.value)


@pytest.mark.usefixtures("case_blind_disk")
def test_one_file_referenced_with_two_spellings_is_refused_for_the_second(
    tmp_path: Path,
) -> None:
    root = make_project(
        tmp_path,
        ["notebook_smoke", "no_code", "plain_fence"],
        texts={
            "no_code": "![dot](img/dot.png)\n",
            "plain_fence": "![dot](Img/DOT.png)\n",
        },
    )
    write_files(root, ["book/img/dot.png"])
    with pytest.raises(build_book.BookError) as stop:
        build_book.stage_book(root, root / "_build" / "executed")
    assert stop.value.exit_code == build_book.EXIT_FILE_ERROR
    assert "plain_fence.md" in str(stop.value)
    assert "'Img/DOT.png'" in str(stop.value)


@pytest.mark.usefixtures("case_blind_disk")
def test_a_string_of_myst_yml_that_names_a_file_in_another_case_is_ignored(
    tmp_path: Path,
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    write_files(root, ["plugins/p.mjs"])
    add_to_myst(root, {"plugins": ["Plugins/P.MJS"]}, {})
    staged = build_book.stage_book(root, root / "_build" / "executed")
    assert not [c for c in staged.copied if c.as_posix().casefold() == "plugins/p.mjs"]


@pytest.mark.usefixtures("case_blind_disk")
def test_a_link_whose_target_is_spelled_with_another_case_is_refused(
    tmp_path: Path,
) -> None:
    root = project_with_stale_stage(tmp_path, ["interactive"])
    (root / "interactive" / "alias.js").symlink_to(
        root / "interactive" / "VENDOR" / "D3.JS"
    )
    error = refuse(root, root / "_build" / "executed")
    assert error.exit_code == build_book.EXIT_FILE_ERROR
    assert "alias.js" in str(error)
    assert "spells" in str(error)


@pytest.mark.usefixtures("case_blind_disk")
def test_a_staging_folder_that_is_the_root_in_another_case_is_refused(
    tmp_path: Path,
) -> None:
    root = project_with_stale_stage(tmp_path)
    spelled = root.parent / "REPO"
    assert build_book.on_disk(spelled) == root
    runner = FakeBuild([])
    code = build_book.main(
        ["--root", str(root), "--executed-dir", str(spelled)], runner=runner
    )
    assert code == build_book.EXIT_BAD_EXECUTED_DIR
    assert runner.calls == []


# --------------------------------------------------------------------------------------
# The plan lists the operations on the staging folder, and two invariants hold on them
# --------------------------------------------------------------------------------------


def folder_outside(tmp_path: Path) -> Path:
    """Make a folder beside the repository that no build may write to or remove from."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    return outside


def test_a_folder_of_the_staging_folder_that_is_a_link_out_of_it_is_refused(
    tmp_path: Path,
) -> None:
    root = project_with_stale_stage(tmp_path, ["interactive"])
    write_files(root, ["plugins/p.mjs"])
    add_to_myst(root, {"plugins": ["plugins/p.mjs"]}, {})
    outside = folder_outside(tmp_path)
    executed = root / "_build" / "executed"
    (executed / "plugins").symlink_to(outside, target_is_directory=True)
    before, beyond = snapshot(executed), snapshot(outside)
    error = refuse(root, executed)
    assert error.exit_code == build_book.EXIT_BAD_EXECUTED_DIR
    assert "a link" in str(error) and "plugins" in str(error)
    assert snapshot(executed) == before
    assert snapshot(outside) == beyond


def test_a_build_folder_that_is_a_link_is_not_followed_by_the_removal(
    tmp_path: Path,
) -> None:
    root = project_with_stale_stage(tmp_path)
    executed = root / "_build" / "executed"
    outside = folder_outside(tmp_path)
    shutil.copytree(executed / "_build", outside / "moved")
    shutil.rmtree(executed / "_build")
    (executed / "_build").symlink_to(outside / "moved", target_is_directory=True)
    before, beyond = snapshot(executed), snapshot(outside)
    error = refuse(root, executed)
    assert error.exit_code == build_book.EXIT_BAD_EXECUTED_DIR
    assert snapshot(executed) == before
    assert snapshot(outside) == beyond


def test_a_destination_that_is_a_link_is_not_written_through(tmp_path: Path) -> None:
    root = project_with_stale_stage(tmp_path)
    outside = folder_outside(tmp_path)
    executed = root / "_build" / "executed"
    (executed / "myst.yml").symlink_to(outside / "keep.txt")
    before, beyond = snapshot(executed), snapshot(outside)
    error = refuse(root, executed)
    assert error.exit_code == build_book.EXIT_BAD_EXECUTED_DIR
    assert "is a link" in str(error)
    assert snapshot(executed) == before
    assert snapshot(outside) == beyond


def link_the_notebook_of_the_book(root: Path) -> None:
    """Make the notebook of the table of contents a link to a notebook elsewhere."""
    elsewhere = root / "notebooks"
    elsewhere.mkdir()
    (root / NOTEBOOK).replace(elsewhere / "notebook_smoke.ipynb")
    (root / NOTEBOOK).symlink_to(elsewhere / "notebook_smoke.ipynb")


@pytest.mark.parametrize("other_checks", ["kept", "removed"])
def test_a_static_entry_that_holds_a_linked_notebook_is_refused_by_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, other_checks: str
) -> None:
    root = make_project(tmp_path, ["notebook_smoke"])
    add_to_myst(root, {"static_files": ["book"]}, {})
    link_the_notebook_of_the_book(root)
    executed = root / "_build" / "executed"
    write_files(executed, ["_build/site/content/stale.json"])
    if other_checks == "removed":
        # The invariant must guard even if no other check stood before it.
        monkeypatch.setattr(build_book, "check_static_entries", lambda *_args: None)
        monkeypatch.setattr(build_book, "_refuse_links_out_of", lambda *_args: None)
    before = snapshot(executed)
    assert (executed / NOTEBOOK).is_file()
    error = refuse(root, executed)
    if other_checks == "removed":
        assert error.exit_code == build_book.EXIT_BAD_EXECUTED_DIR
        assert "notebook command" in str(error)
    else:
        assert error.exit_code == build_book.EXIT_FILE_ERROR
    assert snapshot(executed) == before


def test_what_the_script_does_not_own_is_listed_from_the_disk(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    write_files(
        stage, ["_build/templates/t.txt", "_build/other/o.txt", "_build/site/s"]
    )
    owned = build_book.owned_by_others(stage, ["book/nb.ipynb"])
    assert stage / "book" / "nb.ipynb" in owned
    assert stage / "_build" / "templates" in owned
    assert stage / "_build" / "other" in owned
    assert stage / "_build" / "site" not in owned
    assert stage / "_build" / "html" not in owned


@pytest.mark.parametrize(
    ("kind", "path", "allowed"),
    [
        ("remove", "_build/site", True),
        ("remove", "_build/html", True),
        ("remove", "_build", False),
        ("remove", "_build/templates", False),
        ("remove", "book", False),
        ("remove", "book/nb.ipynb", False),
        ("remove", "interactive", True),
        ("copy", "book/nb.ipynb", False),
        ("copy", "_build/templates/x.js", False),
        ("copy", "book/page.md", True),
    ],
)
def test_the_operations_that_touch_what_the_script_does_not_own_are_refused(
    tmp_path: Path, kind: str, path: str, allowed: bool
) -> None:
    stage = tmp_path / "stage"
    write_files(stage, ["_build/templates/t.txt", "book/nb.ipynb", "interactive/a"])
    owned = build_book.owned_by_others(stage, ["book/nb.ipynb"])
    source = tmp_path / "source.txt"
    source.write_text("x", encoding="utf-8")
    operation = build_book.StageOperation(
        kind, stage / path, source if kind == "copy" else None
    )
    if allowed:
        build_book.check_stage_operations(stage, [operation], owned)
        return
    with pytest.raises(build_book.BookError) as stop:
        build_book.check_stage_operations(stage, [operation], owned)
    assert stop.value.exit_code == build_book.EXIT_BAD_EXECUTED_DIR


def test_the_plan_lists_every_operation_below_the_staging_folder_and_changes_nothing(
    tmp_path: Path,
) -> None:
    root = project_with_stale_stage(tmp_path, ["interactive"])
    executed = root / "_build" / "executed"
    outside = folder_outside(tmp_path)
    before, beyond = snapshot(executed), snapshot(outside)
    plan = build_book.plan_stage(root, executed)
    assert snapshot(executed) == before
    removals = [op.path for op in plan.operations if op.kind == "remove"]
    assert removals == [
        executed / "_build" / "site",
        executed / "_build" / "html",
        executed / "interactive",
    ]
    assert plan.operations[: len(removals)] == [
        op for op in plan.operations if op.kind == "remove"
    ]
    owned = build_book.owned_by_others(executed, plan.notebooks)
    for op in plan.operations:
        assert executed in op.path.parents
        assert not any(op.path == held or held in op.path.parents for held in owned)
        between = [a for a in op.path.parents if executed in a.parents]
        assert not any(folder.is_symlink() for folder in between)
        assert not (op.kind == "copy" and op.path.is_symlink())
    build_book.execute_stage(plan)
    assert snapshot(outside) == beyond


@pytest.mark.usefixtures("case_blind_disk")
def test_a_string_of_myst_yml_in_another_case_naming_a_staged_file_is_refused(
    tmp_path: Path,
) -> None:
    root = project_with_stale_stage(tmp_path)
    add_to_myst(root, {"description": "Book/NO_CODE.md"}, {})
    executed = root / "_build" / "executed"
    before = snapshot(executed)
    error = refuse(root, executed)
    assert error.exit_code == build_book.EXIT_FILE_ERROR
    assert "'Book/NO_CODE.md'" in str(error)
    assert "book/no_code.md" in str(error)
    assert "worded otherwise" in str(error)
    assert snapshot(executed) == before


@pytest.mark.usefixtures("case_blind_disk")
def test_a_string_of_myst_yml_in_another_case_that_nothing_stages_is_only_warned_of(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])
    write_files(root, ["notes/readme.txt"])
    add_to_myst(root, {"description": "Notes/README.TXT"}, {})
    build_book.stage_book(root, root / "_build" / "executed")
    assert "Notes/README.TXT" in caplog.text


Scenario = tuple[Path, list[Path], int, str]
"""The staging folder, the folders beside it that nothing may touch, the exit code and a
fragment of the message the refusal must have."""


def scenario_static(entry: str, code: int, fragment: str) -> Any:
    def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Scenario:
        root = project_with_stale_stage(tmp_path, [entry])
        write_files(root, ["data/fresh/small.csv"])
        return root / "_build" / "executed", [], code, fragment

    return setup


def scenario_link_to_file(target: str) -> Any:
    def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Scenario:
        root = project_with_stale_stage(tmp_path, ["interactive"])
        write_files(root, ["data/fresh/secret.csv", "book/page.md"])
        (tmp_path / "outside-repo.txt").write_text("x", encoding="utf-8")
        (root / "interactive" / "link.txt").symlink_to((root / target).resolve())
        return root / "_build" / "executed", [], build_book.EXIT_FILE_ERROR, "link.txt"

    return setup


def scenario_link_to_folder(target: str) -> Any:
    def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Scenario:
        root = project_with_stale_stage(tmp_path, ["interactive"])
        write_files(root, ["data/fresh/secret.csv"])
        (tmp_path / "outside-dir").mkdir()
        (root / "interactive" / "linked").symlink_to(
            (root / target).resolve(), target_is_directory=True
        )
        return root / "_build" / "executed", [], build_book.EXIT_FILE_ERROR, "linked"

    return setup


def scenario_entry_in_staging(entry: str) -> Any:
    def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Scenario:
        root = project_with_stale_stage(tmp_path, [entry])
        executed = root / "out" / "stage"
        shutil.copytree(root / "_build" / "executed", executed)
        return executed, [], build_book.EXIT_FILE_ERROR, "staging folder"

    return setup


def scenario_not_placeable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Scenario:
    root = project_with_stale_stage(tmp_path)
    (tmp_path / "elsewhere.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        build_book, "find_static_files", lambda *_args: [tmp_path / "elsewhere.txt"]
    )
    return (
        root / "_build" / "executed",
        [],
        build_book.EXIT_FILE_ERROR,
        "elsewhere.txt",
    )


def scenario_stale_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Scenario:
    root = project_with_stale_stage(tmp_path)
    os.utime(root / "_build" / "executed" / NOTEBOOK, (1.0, 1.0))
    return (
        root / "_build" / "executed",
        [],
        build_book.EXIT_NOT_EXECUTED,
        "older than its source",
    )


def scenario_failed_notebook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Scenario:
    root = project_with_stale_stage(tmp_path)
    write_timings(root, {NOTEBOOK: {"status": "error"}})
    return root / "_build" / "executed", [], build_book.EXIT_NOT_EXECUTED, "'error'"


def scenario_no_timing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Scenario:
    root = project_with_stale_stage(tmp_path)
    write_timings(root, {"book/other.ipynb": {"status": "passed"}})
    return (
        root / "_build" / "executed",
        [],
        build_book.EXIT_NOT_EXECUTED,
        "has no entry",
    )


def scenario_no_executed_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Scenario:
    root = project_with_stale_stage(tmp_path)
    (root / "_build" / "executed" / NOTEBOOK).unlink()
    return (
        root / "_build" / "executed",
        [],
        build_book.EXIT_NOT_EXECUTED,
        "no executed copy",
    )


def scenario_bad_toc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Scenario:
    root = project_with_stale_stage(tmp_path)
    document = {"version": 1, "project": {"toc": [{"pattern": "book/*.ipynb"}]}}
    (root / "myst.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    return root / "_build" / "executed", [], build_book.EXIT_BAD_TOC, "pattern"


def scenario_not_a_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Scenario:
    root = project_with_stale_stage(tmp_path)
    add_to_myst(root, {"static_files": "interactive"}, {})
    return (
        root / "_build" / "executed",
        [],
        build_book.EXIT_FILE_ERROR,
        "must be a list of paths",
    )


def scenario_link_in_another_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Scenario:
    root = project_with_stale_stage(tmp_path, ["interactive"])
    (root / "interactive" / "alias.js").symlink_to(
        root / "interactive" / "VENDOR" / "D3.JS"
    )
    return root / "_build" / "executed", [], build_book.EXIT_FILE_ERROR, "spells"


def scenario_toc_page_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Scenario:
    root = project_with_stale_stage(tmp_path)
    path = root / "myst.yml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("book/no_code.md", "Book/No_Code.md"),
        encoding="utf-8",
    )
    return root / "_build" / "executed", [], build_book.EXIT_BAD_TOC, "Book/No_Code.md"


def scenario_executed_is_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Scenario:
    root = project_with_stale_stage(tmp_path)
    return root, [], build_book.EXIT_BAD_EXECUTED_DIR, "repository root"


def scenario_reference_in_another_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Scenario:
    root = project_with_stale_stage(tmp_path)
    (root / "book" / "no_code.md").write_text(
        "![d](/Images/FIG.png)\n", encoding="utf-8"
    )
    write_files(root, ["images/fig.png"])
    return (
        root / "_build" / "executed",
        [],
        build_book.EXIT_FILE_ERROR,
        "'/Images/FIG.png'",
    )


def scenario_string_in_another_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Scenario:
    root = project_with_stale_stage(tmp_path)
    add_to_myst(root, {"description": "Book/NO_CODE.md"}, {})
    return (
        root / "_build" / "executed",
        [],
        build_book.EXIT_FILE_ERROR,
        "worded otherwise",
    )


def scenario_linked_notebook_in_an_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Scenario:
    root = make_project(tmp_path, ["notebook_smoke"])
    add_to_myst(root, {"static_files": ["book"]}, {})
    link_the_notebook_of_the_book(root)
    write_files(root / "_build" / "executed", ["_build/site/content/stale.json"])
    monkeypatch.setattr(build_book, "check_static_entries", lambda *_args: None)
    monkeypatch.setattr(build_book, "_refuse_links_out_of", lambda *_args: None)
    return (
        root / "_build" / "executed",
        [],
        build_book.EXIT_BAD_EXECUTED_DIR,
        "notebook command",
    )


def scenario_linked_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Scenario:
    root = project_with_stale_stage(tmp_path, ["interactive"])
    write_files(root, ["plugins/p.mjs"])
    add_to_myst(root, {"plugins": ["plugins/p.mjs"]}, {})
    outside = folder_outside(tmp_path)
    executed = root / "_build" / "executed"
    (executed / "plugins").symlink_to(outside, target_is_directory=True)
    return executed, [outside], build_book.EXIT_BAD_EXECUTED_DIR, "a link"


def scenario_linked_build_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Scenario:
    root = project_with_stale_stage(tmp_path)
    executed = root / "_build" / "executed"
    outside = folder_outside(tmp_path)
    shutil.copytree(executed / "_build", outside / "moved")
    shutil.rmtree(executed / "_build")
    (executed / "_build").symlink_to(outside / "moved", target_is_directory=True)
    return executed, [outside], build_book.EXIT_BAD_EXECUTED_DIR, "a link"


def scenario_linked_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Scenario:
    root = project_with_stale_stage(tmp_path)
    outside = folder_outside(tmp_path)
    executed = root / "_build" / "executed"
    (executed / "myst.yml").symlink_to(outside / "keep.txt")
    return executed, [outside], build_book.EXIT_BAD_EXECUTED_DIR, "is a link"


FILE = build_book.EXIT_FILE_ERROR
REFUSING_SCENARIOS = {
    "entry absent": scenario_static("absent", FILE, "names no file"),
    "entry outside": scenario_static("../outside", FILE, "points outside"),
    "entry data": scenario_static("data", FILE, "never staged"),
    "entry data/fresh": scenario_static("data/fresh", FILE, "never staged"),
    "entry _build": scenario_static("_build", FILE, "never staged"),
    "entry root": scenario_static(".", FILE, "the repository itself"),
    "entry in another case": scenario_static(
        "Interactive", FILE, "spelled differently"
    ),
    "entry Data/fresh": scenario_static("Data/fresh", FILE, "spelled differently"),
    "entry holds the book folder": scenario_static("book", FILE, "holds the page"),
    "entry is a page": scenario_static("book/no_code.md", FILE, "holds the page"),
    "entry is a notebook": scenario_static(
        "book/notebook_smoke.ipynb", FILE, "holds the page"
    ),
    "entry is the staging folder": scenario_entry_in_staging("out/stage"),
    "entry is its parent": scenario_entry_in_staging("out"),
    "link to a data file": scenario_link_to_file("data/fresh/secret.csv"),
    "link to a page": scenario_link_to_file("book/page.md"),
    "link to a file outside": scenario_link_to_file("../outside-repo.txt"),
    "link to the data folder": scenario_link_to_folder("data/fresh"),
    "link to the book folder": scenario_link_to_folder("book"),
    "link to a folder outside": scenario_link_to_folder("../outside-dir"),
    "link target in another case": scenario_link_in_another_case,
    "static_files is not a list": scenario_not_a_list,
    "staged path outside the root": scenario_not_placeable,
    "table of contents refused": scenario_bad_toc,
    "page of the toc in another case": scenario_toc_page_case,
    "copy older than its notebook": scenario_stale_copy,
    "notebook recorded as failed": scenario_failed_notebook,
    "notebook without a timing": scenario_no_timing,
    "missing executed copy": scenario_no_executed_copy,
    "staging folder is the root": scenario_executed_is_root,
    "reference of a page in another case": scenario_reference_in_another_case,
    "string of myst.yml in another case": scenario_string_in_another_case,
    "linked notebook in a static entry": scenario_linked_notebook_in_an_entry,
    "folder of the staging folder is a link": scenario_linked_folder,
    "_build of the staging folder is a link": scenario_linked_build_folder,
    "destination is a link": scenario_linked_destination,
}


@pytest.mark.usefixtures("case_blind_disk")
@pytest.mark.parametrize("name", sorted(REFUSING_SCENARIOS))
def test_a_refused_build_leaves_the_staging_folder_as_it_found_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.delenv("BASE_URL", raising=False)
    executed, beside, code, fragment = REFUSING_SCENARIOS[name](tmp_path, monkeypatch)
    root = tmp_path / "repo"
    watched = [executed, *beside]
    before = [snapshot(folder) for folder in watched]
    assert before[0], "the scenario must leave something in the staging folder"
    error = refuse(root, executed)
    assert (error.exit_code, fragment in str(error)) == (code, True), str(error)
    assert [snapshot(folder) for folder in watched] == before
