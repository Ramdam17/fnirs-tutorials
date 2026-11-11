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
# A frame must point to another site over https
# --------------------------------------------------------------------------------------

FRAME_SOURCE = "frames/live.html"


def frame_page(stem: str, src: str) -> dict[str, Any]:
    """A real page file with one frame whose source is replaced."""
    page = load_fixture(stem)
    frame = page["mdast"]["children"][0]["children"][0]
    assert frame["type"] == "iframe"
    frame["src"] = src
    return page


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
    ("src", "expected"),
    [
        ("https://example.org/live.html", True),
        ("HTTPS://Example.org/live.html", True),
        ("  \t\nhttps://example.org/live.html", True),
        ("//example.org/live.html", True),
        ("", False),
        ("#top", False),
        ("?x=1", False),
        ("frames/live.html", False),
        ("./frames/live.html", False),
        ("../frames/live.html", False),
        ("/frames/live.html", False),
        ("/fnirs-tutorials/frames/live.html", False),
        ("my%20demo.html", False),
        ("demo/", False),
        ("demo", False),
        ("http://example.org/live.html", False),
        ("https:live.html", False),
        ("https:///live.html", False),
        ("///live.html", False),
        ("//", False),
        ("data:text/html,<p>x</p>", False),
        ("about:blank", False),
        ("javascript:alert(1)", False),
        ("C:\\frames\\live.html", False),
        ("\\\\example.org\\live.html", False),
    ],
)
def test_only_an_https_url_or_a_protocol_relative_one_is_an_external_frame(
    src: str, expected: bool
) -> None:
    assert build_book.frame_is_external(src) is expected


@pytest.mark.parametrize("stem", ["frame_relative", "frame_root_relative"])
def test_a_local_frame_is_refused_and_named(stem: str) -> None:
    page = load_fixture(stem)
    problems = build_book.check_built_pages({page["location"]: page}, [], [])
    assert len(problems) == 1
    assert problems[0].exit_code == build_book.EXIT_FRAME_LOCAL == 18
    message = str(problems[0])
    source = build_book.find_frame_sources(page["mdast"])[0]
    assert f"page {page['location']!r}" in message
    assert f"source {source!r}" in message
    assert "neither an https URL nor one that starts with //" in message
    assert "does not publish the target of a frame" in message
    assert "an empty frame" in message
    assert "blocks an http source on a site served over HTTPS" in message
    assert "Give the frame an https URL" in message


def test_a_frame_whose_source_is_an_https_url_is_accepted() -> None:
    page = load_fixture("frame_scheme")
    assert build_book.check_built_pages({page["location"]: page}, [], []) == []


def test_an_http_frame_is_refused() -> None:
    page = frame_page("frame_scheme", "http://example.org/live.html")
    problems = build_book.check_built_pages({page["location"]: page}, [], [])
    assert [p.exit_code for p in problems] == [build_book.EXIT_FRAME_LOCAL]


def test_every_refused_frame_of_a_page_is_reported() -> None:
    page = load_fixture("frame_relative")
    frame = page["mdast"]["children"][0]["children"][0]
    page["mdast"]["children"][0]["children"].append(dict(frame, src="/other.html"))
    problems = build_book.check_built_pages({page["location"]: page}, [], [])
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


def test_a_local_frame_stops_the_script_with_its_own_exit_code(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "frame_relative"])
    build = FakeBuild(["notebook_smoke", "frame_relative"])
    assert run_main(root, build) == build_book.EXIT_FRAME_LOCAL == 18
    assert "Book not built (exit 18)" in caplog.text
    assert "page '/book/frame_relative.md' holds a frame whose source" in caplog.text


def test_a_frame_with_an_https_source_lets_the_script_succeed(tmp_path: Path) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "frame_scheme"])
    assert run_main(root, FakeBuild(["notebook_smoke", "frame_scheme"])) == 0


def test_the_target_of_a_frame_is_not_staged(tmp_path: Path) -> None:
    root = make_project(
        tmp_path,
        ["notebook_smoke", "frame_relative"],
        texts={"frame_relative": ":::{iframe} frames/live.html\n:::\n"},
    )
    (root / "book" / "frames").mkdir()
    (root / "book" / "frames" / "live.html").write_text("x", encoding="utf-8")
    staged = build_book.stage_book(root, root / "_build" / "executed")
    assert not (staged.stage_dir / "book" / "frames" / "live.html").exists()
    assert (staged.stage_dir / "book" / "frame_relative.md").is_file()


# --------------------------------------------------------------------------------------
# File probes on names that come from the content of the book
# --------------------------------------------------------------------------------------

LONG_NAME = "a" * 300


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


def test_a_file_error_the_script_did_not_guard_exits_17_with_the_whole_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_project(tmp_path, ["notebook_smoke", "no_code"])

    def refuse(*_args: Any) -> None:
        raise OSError(13, "Permission denied", "/some/file.md")

    monkeypatch.setattr(build_book, "stage_book", refuse)
    assert run_main(root, FakeBuild([])) == build_book.EXIT_FILE_ERROR
    assert (
        "PermissionError on /some/file.md: [Errno 13] Permission denied" in caplog.text
    )


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
    assert script.count("current=true") == 1 and script.count("current=false") == 1
