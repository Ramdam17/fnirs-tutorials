"""Tests of ``scripts/build_book.py``: the staging, the check of the built pages, the exits.

None of these tests needs the network or a real book build. The build step of the script
is a parameter (``runner``): the tests give it a stand-in that writes page files into
``_build/site/content/`` of the staging folder, as MyST does, and returns an exit code.

The page files under ``tests/fixtures/built_pages/`` are real files written by MyST, not
written by hand: jupyter-book 2.1.7 (myst-cli 1.11.0) built them on 2026-11-11 in one
scratch project, made by ``docs/measures/s01-book-foundations-built-page-check-records.py
--export-fixtures``, from the sources named by the file names, for example
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
) -> Path:
    """Write a repository-like folder for the pages of the given fixtures.

    Notebooks get a source and, when ``executed`` is true, an executed copy that is newer.
    Other pages get the text in ``texts`` (by stem) or a title.
    """
    root = tmp_path / "repo"
    texts = texts or {}
    toc = []
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
        else:
            target.write_text(texts.get(stem, "# A page\n"), encoding="utf-8")
    document = {"version": 1, "project": {"toc": toc}}
    (root / "myst.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    return root


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
        {page["location"]: page}, [page_path("no_code"), NOTEBOOK], [NOTEBOOK]
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


def test_the_build_command_has_no_execute_flag_and_shows_its_output(
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
    ]
    assert seen["kwargs"]["cwd"] == tmp_path
    # No redirection of the output and no changed environment (BASE_URL is inherited).
    assert not {"stdout", "stderr", "capture_output", "env"} & set(seen["kwargs"])
