"""Build the book from the executed notebooks, then check the pages MyST built.

The notebooks are executed first, by ``tests/test_notebooks.py``, which writes the
executed copies under ``_build/executed/`` at the notebooks' own relative paths. This
script stages ``myst.yml``, the pages of the table of contents that are not notebooks and
the local files that those pages, the markdown cells of the notebooks and ``myst.yml``
reference next to those copies, runs
``jupyter book build --html --strict`` in that folder without ``--execute``, and reads the page
files that MyST wrote under ``_build/site/content/``. It exits non-zero when

* the build exits non-zero (the exit code of the build is passed on), which under
  ``--strict`` includes a missing image or a missing ``{include}``;
* a page whose ``location`` does not end with ``.ipynb`` holds a ``block`` of kind
  ``notebook-code`` or an ``inlineExpression``, which MyST would leave without output
  because nothing executes it here, unless the page's own front matter says
  ``execute: {skip: true}``;
* a file of the table of contents has no built page;
* the executed copy of a notebook is missing, older than its source or than the helper
  package under ``src/``, or the timings file of the notebook command does not record
  the notebook as ``passed``.

The check reads MyST's own parse, so it sees a ``{code-cell}`` brought in by an
``{include}`` or opened inside a blockquote, a list item, an admonition or a tab, which
the textual guard of ``tests/notebook_runner.py`` lets through.

When to use
-----------
Run it after the notebook command, from any folder::

    uv run python -m pytest tests/test_notebooks.py -v
    BASE_URL=/fnirs-tutorials uv run python scripts/build_book.py

The base URL of the site is not known here: the caller passes it in the environment
variable ``BASE_URL`` (https://mystmd.org/guide/deployment-github-pages and
https://mystmd.org/guide/deployment), and the child process inherits it. The built site
is ``_build/executed/_build/html``.

Examples
--------
>>> main(["--root", "."])  # doctest: +SKIP
0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("build_book")

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
EXECUTED_SUBDIR = Path("_build") / "executed"
# MyST writes one JSON file per page here, relative to the folder it was run in.
SITE_SUBDIR = Path("_build") / "site"
HTML_SUBDIR = Path("_build") / "html"
CONTENT_SUBDIR = SITE_SUBDIR / "content"
NOTEBOOK_COMMAND = "uv run python -m pytest tests/test_notebooks.py -v"
# Written by tests/test_notebooks.py: one entry per notebook, with its ``status``. It is
# emptied at the start of each run of the notebook command, so it carries the outcome of
# the run that wrote the executed copies.
TIMINGS_SUBPATH = Path("build-reports") / "notebook-timings.json"
STATUS_PASSED = "passed"
# The helper package the notebooks import; a change there makes executed outputs stale.
HELPER_SUBDIR = Path("src")

# Exit codes of this script. The exit code of a failed build is passed on unchanged, so
# the codes of the script's own failures sit well above the values a build uses.
EXIT_OK = 0
EXIT_PAGE_HOLDS_CODE = 10
EXIT_PAGE_NOT_BUILT = 11
EXIT_NOT_EXECUTED = 12
EXIT_BAD_TOC = 13
EXIT_UNREADABLE_PAGE = 14
EXIT_NO_CONTENT = 15
EXIT_BAD_EXECUTED_DIR = 16
EXIT_FILE_ERROR = 17

NOTEBOOK_SUFFIX = ".ipynb"
# Folders of the repository that a staged page must never drag into the staging folder:
# the data are large and not ours to publish, and ``_build`` holds the staging folder.
NEVER_STAGED = ("data", "_build")

# Local files that a page references, found by text. This is a heuristic: it reads the
# target of a markdown link or image and the first argument of the directives that take a
# path, in the markdown pages and in the markdown cells of the notebooks, and every string
# of myst.yml that names an existing file. A reference it misses is not staged, and MyST
# then warns that the file is missing (the build is run with --strict for that reason).
_LINK_TARGET = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_DIRECTIVE_PATH = re.compile(
    r"^[ \t>]*(?:`{3,}|~{3,}|:{3,})\{(include|literalinclude|image|figure|download|iframe)\}"
    r"[ \t]+(\S+)",
    re.MULTILINE,
)
_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")

Runner = Callable[[Path], int]


class BookError(Exception):
    """A failure of the build or of the check, with the exit code to return.

    Attributes
    ----------
    exit_code : int
        Exit code of the script for this failure.
    """

    def __init__(self, exit_code: int, message: str) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass
class StagedBook:
    """What :func:`stage_book` put in the staging folder.

    Attributes
    ----------
    stage_dir : Path
        The staging folder, which is also the folder of the executed copies.
    toc_files : list of str
        Every ``file:`` entry of the table of contents, POSIX form, in order.
    notebooks : list of str
        The entries of ``toc_files`` that are notebooks.
    copied : list of Path
        Files copied into the staging folder, relative to it, ``myst.yml`` included.
    """

    stage_dir: Path
    toc_files: list[str]
    notebooks: list[str]
    copied: list[Path] = field(default_factory=list)


def _load_toc_reader() -> Any:
    """Import ``tests/notebook_runner.py``, the one reader of the table of contents.

    It is imported on demand, from the tests folder, so that this script and the notebook
    command can never disagree about which entries are notebooks. Importing it executes
    nothing but its definitions.
    """
    tests_dir = str(DEFAULT_ROOT / "tests")
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    import notebook_runner

    return notebook_runner


def _collect_files(entries: Any, found: list[str]) -> None:
    """Append every ``file`` entry of a table of contents, at any depth of ``children``.

    The structure was validated by ``notebook_runner.list_notebooks`` before this runs.
    """
    for entry in entries:
        if entry.get("file") is not None:
            found.append(Path(str(entry["file"])).as_posix())
        if "children" in entry:
            _collect_files(entry["children"], found)


def load_myst_document(myst_path: Path) -> dict[str, Any]:
    """Read ``myst.yml`` as a mapping.

    Raises
    ------
    BookError
        With ``EXIT_FILE_ERROR`` when the file cannot be read, and with ``EXIT_BAD_TOC``
        when it is not YAML or not a mapping. The message names the file.
    """
    import yaml

    try:
        text = myst_path.read_text(encoding="utf-8")
    except OSError as error:
        raise BookError(
            EXIT_FILE_ERROR, f"cannot read {myst_path}: {type(error).__name__}: {error}"
        ) from error
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise BookError(
            EXIT_BAD_TOC, f"{myst_path} is not valid YAML: {error}"
        ) from error
    if not isinstance(document, dict):
        raise BookError(EXIT_BAD_TOC, f"{myst_path} is not a mapping at its top level.")
    return document


def read_toc_files(myst_path: Path) -> tuple[list[str], list[str]]:
    """Read the files of the table of contents and the notebooks among them.

    Parameters
    ----------
    myst_path : Path
        Path of ``myst.yml``.

    Returns
    -------
    tuple
        ``(toc_files, notebooks)``, both in table-of-contents order, POSIX form.

    Raises
    ------
    BookError
        With ``EXIT_FILE_ERROR`` when ``myst.yml`` cannot be read, and with
        ``EXIT_BAD_TOC`` when the table of contents is refused by the reader of the
        notebook command.
    """
    document = load_myst_document(myst_path)
    runner = _load_toc_reader()
    try:
        notebooks = runner.list_notebooks(myst_path)
    except (ValueError, TypeError) as error:
        raise BookError(EXIT_BAD_TOC, f"{myst_path}: {error}") from error
    toc_files: list[str] = []
    _collect_files(document["project"]["toc"], toc_files)
    return toc_files, notebooks


def _newest_helper_file(root: Path) -> Path | None:
    """Return the most recently modified file of the helper package, or None."""
    helper_dir = root / HELPER_SUBDIR
    files = [
        path
        for path in helper_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    return max(files, key=lambda path: path.stat().st_mtime, default=None)


def _read_timings(root: Path) -> dict[str, Any]:
    """Read the timings file of the notebook command.

    Raises
    ------
    BookError
        With ``EXIT_NOT_EXECUTED`` when the file is missing, or is not a JSON object.
    """
    path = root / TIMINGS_SUBPATH
    if not path.is_file():
        raise BookError(
            EXIT_NOT_EXECUTED,
            f"{path} does not exist: the notebook command did not record the outcome "
            f"of the notebooks. Run the notebook command first: {NOTEBOOK_COMMAND}",
        )
    try:
        timings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BookError(
            EXIT_NOT_EXECUTED,
            f"{path} cannot be read ({type(error).__name__}: {error}). "
            f"Run the notebook command again: {NOTEBOOK_COMMAND}",
        ) from error
    if not isinstance(timings, dict):
        raise BookError(
            EXIT_NOT_EXECUTED,
            f"{path} is not a JSON object. Run the notebook command again: "
            f"{NOTEBOOK_COMMAND}",
        )
    return timings


def check_executed_copies(root: Path, executed_dir: Path, notebooks: list[str]) -> None:
    """Fail unless every notebook was executed successfully and its copy is fresh.

    A book built from an unexecuted notebook would publish code with no output, and a
    book built from a notebook that failed would publish its traceback; nothing in the
    page would say so. For every notebook of the table of contents, three things must
    hold: the timings file of the notebook command records it with the status
    ``passed``; its executed copy exists; and the copy is at least as new as the
    notebook's source and as every file of the helper package under ``src/``, since a
    change there makes the outputs stale.

    Modification times are a conservative heuristic. They err toward refusing a sound
    copy when a source file is touched or rewritten without a change of content, for
    example after switching to another branch and back, and the message then asks for
    another run of the notebook command. They err toward accepting a stale copy only if
    a clock or a tool sets a source's time in the past of the copy's, for example
    ``cp -p`` or an archive extracted with its original times; the timings file does not
    cover that case.

    Raises
    ------
    BookError
        With ``EXIT_NOT_EXECUTED``, naming the notebook and the command to run.
    """
    if not notebooks:
        return
    if not executed_dir.is_dir():
        raise BookError(
            EXIT_NOT_EXECUTED,
            f"{executed_dir} does not exist: no notebook was executed. "
            f"Run the notebook command first: {NOTEBOOK_COMMAND}",
        )
    timings = _read_timings(root)
    newest_helper = _newest_helper_file(root)
    for notebook in notebooks:
        entry = timings.get(notebook)
        if not isinstance(entry, dict):
            raise BookError(
                EXIT_NOT_EXECUTED,
                f"notebook {notebook!r} has no entry in {root / TIMINGS_SUBPATH}: it "
                f"was not executed by the last run. Run the notebook command first: "
                f"{NOTEBOOK_COMMAND}",
            )
        status = entry.get("status")
        if status != STATUS_PASSED:
            raise BookError(
                EXIT_NOT_EXECUTED,
                f"notebook {notebook!r} was recorded with status {status!r}, not "
                f"{STATUS_PASSED!r}, by the last run; its executed copy would show the "
                f"failure. Fix the notebook and run the notebook command again: "
                f"{NOTEBOOK_COMMAND}",
            )
        copy = executed_dir / notebook
        if not copy.is_file():
            raise BookError(
                EXIT_NOT_EXECUTED,
                f"notebook {notebook!r} has no executed copy at {copy}. "
                f"Run the notebook command first: {NOTEBOOK_COMMAND}",
            )
        copy_time = copy.stat().st_mtime
        if copy_time < (root / notebook).stat().st_mtime:
            raise BookError(
                EXIT_NOT_EXECUTED,
                f"the executed copy of notebook {notebook!r} is older than its source. "
                f"Run the notebook command again: {NOTEBOOK_COMMAND}",
            )
        if newest_helper is not None and copy_time < newest_helper.stat().st_mtime:
            raise BookError(
                EXIT_NOT_EXECUTED,
                f"the executed copy of notebook {notebook!r} is older than "
                f"{newest_helper.relative_to(root).as_posix()} of the helper package, "
                f"which the notebooks import. Run the notebook command again: "
                f"{NOTEBOOK_COMMAND}",
            )


def _local_target(raw: str, page: Path, root: Path) -> Path | None:
    """Resolve a reference found in ``page`` to a file of ``root``, or None.

    A fragment (``#...``) and a query (``?...``) are dropped; a URL with a scheme is not
    local; a path that starts with ``/`` is relative to ``root``; any other path is
    relative to the folder of ``page``.
    """
    target = raw.split("#", 1)[0].split("?", 1)[0]
    if not target or _URL_SCHEME.match(target):
        return None
    base = root if target.startswith("/") else page.parent
    try:
        candidate = (base / target.lstrip("/")).resolve()
        candidate.relative_to(root.resolve())
    except ValueError:
        logger.warning("%s: reference %r points outside the repository", page, raw)
        return None
    return candidate if candidate.is_file() else None


def _page_text(path: Path) -> str:
    """Return the text to search for references: a page, or the markdown cells of a notebook."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() != NOTEBOOK_SUFFIX:
        return raw
    try:
        cells = json.loads(raw).get("cells", [])
    except (ValueError, AttributeError):
        logger.warning("%s is not a notebook file; its references are not staged", path)
        return ""
    parts = []
    for cell in cells:
        if isinstance(cell, dict) and cell.get("cell_type") == "markdown":
            source = cell.get("source", "")
            parts.append("".join(source) if isinstance(source, list) else str(source))
    return "\n\n".join(parts)


def _is_stageable(target: Path, raw: str, current: Path, root: Path) -> bool:
    """Return True for a referenced file that may be copied into the staging folder.

    A reference into ``data/`` or ``_build/`` is dropped with a warning, and a
    notebook is dropped silently: its executed copy is already in the staging folder
    and the source must not replace it.
    """
    relative = target.relative_to(root.resolve())
    if relative.parts[0] in NEVER_STAGED:
        logger.warning(
            "%s: reference %r is under %s/ and is not staged",
            current,
            raw,
            relative.parts[0],
        )
        return False
    return target.suffix.lower() != NOTEBOOK_SUFFIX


def find_referenced_files(page: Path, root: Path) -> list[Path]:
    """List the local files a page or notebook references, included pages followed.

    Parameters
    ----------
    page : Path
        Markdown page, or notebook whose markdown cells are searched.
    root : Path
        Repository root; references are resolved against the page's folder, or against
        the root when they start with ``/``.

    Returns
    -------
    list of Path
        Absolute paths of existing files under ``root``, each once. A reference to a
        notebook is never returned, so that the executed copy is not overwritten by its
        source; a reference into ``data/`` or ``_build/`` is dropped with a warning.
    """
    found: list[Path] = []
    pending = [page.resolve()]
    seen = set(pending)
    while pending:
        current = pending.pop()
        try:
            text = _page_text(current)
        except (UnicodeDecodeError, OSError):
            continue
        raws = _LINK_TARGET.findall(text) + [
            m[1] for m in _DIRECTIVE_PATH.findall(text)
        ]
        for raw in raws:
            target = _local_target(raw, current, root)
            if target is None or target in seen:
                continue
            seen.add(target)
            if not _is_stageable(target, raw, current, root):
                continue
            found.append(target)
            if target.suffix.lower() == ".md":
                pending.append(target)
    return found


def _strings_of(node: Any) -> list[str]:
    """Return every string value of a loaded YAML document, at any depth."""
    strings: list[str] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            strings.append(current)
        elif isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return strings


def find_config_files(
    document: dict[str, Any], myst_path: Path, root: Path
) -> list[Path]:
    """List the local files that ``myst.yml`` names, such as a bibliography or a logo.

    Parameters
    ----------
    document : dict
        The loaded ``myst.yml``.
    myst_path : Path
        Path of ``myst.yml``; its folder is the base of the paths it holds.
    root : Path
        Repository root.

    Returns
    -------
    list of Path
        Absolute paths of the existing files under ``root`` that a string value of the
        file names, the pages of the table of contents included (they are staged once).
        This reads every string value rather than a list of keys, so that a key MyST adds
        later is covered; a string that is not an existing file is ignored.
    """
    found: list[Path] = []
    for raw in _strings_of(document):
        target = _local_target(raw, myst_path, root)
        if target is None or target == myst_path.resolve() or target in found:
            continue
        if _is_stageable(target, raw, myst_path, root):
            found.append(target)
    return found


def remove_stale_site(stage_dir: Path) -> list[Path]:
    """Remove the site and the HTML of an earlier build from the staging folder.

    MyST does not clear ``_build/site/content/``, so a page written by an earlier build
    would still be there after this one and would be read as if it belonged to it. The
    downloaded theme, ``_build/templates``, is kept: it is slow to fetch.

    Returns
    -------
    list of Path
        The folders removed.
    """
    removed = []
    for sub in (SITE_SUBDIR, HTML_SUBDIR):
        path = stage_dir / sub
        if path.exists():
            shutil.rmtree(path)
            removed.append(path)
    return removed


def check_executed_dir(root: Path, executed_dir: Path) -> None:
    """Refuse a staging folder that is the repository root or one of its ancestors.

    The staging folder is written to and its ``_build/site`` and ``_build/html`` are
    removed before every build, so it must be a folder of its own: the root would lose
    the ordinary local build of the book, and an ancestor would be filled with staged
    files. Both paths must be resolved.

    Raises
    ------
    BookError
        With ``EXIT_BAD_EXECUTED_DIR``, before anything is removed or copied.
    """
    if executed_dir == root or executed_dir in root.parents:
        raise BookError(
            EXIT_BAD_EXECUTED_DIR,
            f"the folder of the executed notebooks, {executed_dir}, is the repository "
            f"root {root} or one of its parents. It is the staging folder of the build "
            "and its _build/site and _build/html are removed, so it must be a separate "
            f"folder, by default {root / EXECUTED_SUBDIR}.",
        )


def stage_book(root: Path, executed_dir: Path) -> StagedBook:
    """Stage ``myst.yml``, the non-notebook pages and their references.

    The references of the notebooks (their markdown cells) and the files that
    ``myst.yml`` names are staged too; the notebooks themselves are not, since the
    staging folder holds their executed copies.

    Parameters
    ----------
    root : Path
        Repository root, holding ``myst.yml``.
    executed_dir : Path
        Folder of the executed copies; it becomes the staging folder.

    Returns
    -------
    StagedBook
        The files staged.

    Raises
    ------
    BookError
        If the table of contents is refused, a notebook has no fresh executed copy, or
        a file cannot be read or copied.
    """
    myst_path = root / "myst.yml"
    toc_files, notebooks = read_toc_files(myst_path)
    check_executed_copies(root, executed_dir, notebooks)
    staged = StagedBook(
        stage_dir=executed_dir, toc_files=toc_files, notebooks=notebooks
    )
    for removed in remove_stale_site(executed_dir):
        logger.info("Removed %s from an earlier build", removed)
    sources = [myst_path]
    for name in toc_files:
        page = root / name
        if name in notebooks:
            sources.extend(find_referenced_files(page, root))
            continue
        if not page.is_file():
            # MyST reports a missing page itself; the check then finds no built page.
            logger.warning("Page %s of the table of contents does not exist", name)
            continue
        sources.append(page)
        sources.extend(find_referenced_files(page, root))
    for config_file in find_config_files(
        load_myst_document(myst_path), myst_path, root
    ):
        sources.append(config_file)
        if config_file.suffix.lower() == ".md":
            sources.extend(find_referenced_files(config_file, root))
    copied: set[Path] = set()
    for source in sources:
        resolved = source.resolve()
        if resolved in copied:
            continue
        copied.add(resolved)
        relative = resolved.relative_to(root.resolve())
        destination = executed_dir / relative
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        except OSError as error:
            raise BookError(
                EXIT_FILE_ERROR,
                f"cannot stage {relative.as_posix()} to {destination}: "
                f"{type(error).__name__}: {error}",
            ) from error
        staged.copied.append(relative)
        logger.info("Staged %s", relative.as_posix())
    return staged


def run_jupyter_book(stage_dir: Path) -> int:
    """Run ``jupyter book build --html --strict`` in the staging folder, without ``--execute``.

    ``--strict`` makes the build exit non-zero on a warning. Measured with jupyter-book
    2.1.7 (myst-cli 1.11.0): without it a missing image and a missing ``{include}`` are
    reported and the build exits 0, with it both exit 1, and the book itself builds with
    exit 0. It does not catch a missing local page named by an ``{iframe}`` (no message
    at all) or a missing ``bibliography`` file of ``myst.yml`` (a warning that exit 0
    survives), which is why :func:`stage_book` stages those files.

    The command is run through the interpreter of this script, so that it uses the same
    environment. Its output goes to the console unchanged. The environment, ``BASE_URL``
    included, is inherited.

    Returns
    -------
    int
        Exit code of the build.
    """
    command = [sys.executable, "-m", "jupyter_book", "build", "--html", "--strict"]
    logger.info("Running %s in %s", " ".join(command), stage_dir)
    return subprocess.run(command, cwd=stage_dir, check=False).returncode


def find_code_to_execute(mdast: Any) -> list[str]:
    """Name the kinds of executable node that a MyST syntax tree holds.

    Parameters
    ----------
    mdast : Any
        The ``mdast`` value of a page file, a tree of dicts and lists.

    Returns
    -------
    list of str
        ``"block of kind notebook-code"`` and ``"inlineExpression"``, each once, in
        that order, for those present anywhere in the tree.
    """
    has_block = False
    has_inline = False
    stack = [mdast]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("type") == "block" and node.get("kind") == "notebook-code":
                has_block = True
            if node.get("type") == "inlineExpression":
                has_inline = True
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    signs = []
    if has_block:
        signs.append("block of kind notebook-code")
    if has_inline:
        signs.append("inlineExpression")
    return signs


def _skipped_by_author(page: dict[str, Any]) -> bool:
    """Return True when the page's recorded front matter says ``execute.skip: true``.

    MyST records the deprecated ``skip_execution`` and the string ``'true'`` as
    ``execute: {skip: true}`` too, so one key is read. An ``execute: {skip: true}`` set
    under ``project:`` in ``myst.yml`` is not copied into the page's recorded front
    matter, so it never excuses a page here. Both facts were measured with jupyter-book
    2.1.7 (myst-cli 1.11.0).
    """
    execute = page.get("frontmatter", {}).get("execute")
    return isinstance(execute, dict) and execute.get("skip") is True


def read_built_pages(content_dir: Path) -> dict[str, dict[str, Any]]:
    """Read every page file under ``content_dir``, keyed by the page's ``location``.

    Raises
    ------
    BookError
        With ``EXIT_NO_CONTENT`` when the folder does not exist or holds no page file,
        and with ``EXIT_UNREADABLE_PAGE`` for a file that is not JSON with a ``location``
        and an ``mdast``.
    """
    files = sorted(content_dir.rglob("*.json")) if content_dir.is_dir() else []
    if not files:
        raise BookError(
            EXIT_NO_CONTENT,
            f"the build wrote no page file under {content_dir}; the pages cannot be "
            "checked, so the book is not accepted.",
        )
    pages: dict[str, dict[str, Any]] = {}
    for path in files:
        try:
            page = json.loads(path.read_text(encoding="utf-8"))
            location = page["location"]
            page["mdast"]
        except (ValueError, KeyError, TypeError) as error:
            raise BookError(
                EXIT_UNREADABLE_PAGE,
                f"{path} is not a page file written by MyST ({type(error).__name__}: {error}).",
            ) from error
        pages[location] = page
    return pages


def check_built_pages(
    pages: dict[str, dict[str, Any]], toc_files: list[str], notebooks: list[str]
) -> list[BookError]:
    """Check the pages MyST built against the table of contents.

    Parameters
    ----------
    pages : dict
        Page files by ``location``, as returned by :func:`read_built_pages`.
    toc_files : list of str
        ``file:`` entries of the table of contents.
    notebooks : list of str
        The entries of ``toc_files`` that are notebooks.

    Returns
    -------
    list of BookError
        One per problem, empty when the pages are accepted: a page that is not a
        notebook and holds code to execute, and a file of the table of contents with
        no built page.
    """
    problems: list[BookError] = []
    for location, page in sorted(pages.items()):
        if location.lower().endswith(NOTEBOOK_SUFFIX):
            continue
        signs = find_code_to_execute(page["mdast"])
        if not signs:
            continue
        if _skipped_by_author(page):
            logger.info(
                "Page %s holds %s and is excluded by execute.skip", location, signs
            )
            continue
        problems.append(
            BookError(
                EXIT_PAGE_HOLDS_CODE,
                f"page {location!r} is not a notebook and holds code to execute "
                f"({' and '.join(signs)}); nothing executes it in this build, so it "
                "would be published with no output. Convert it to a notebook, or set "
                "'execute: {skip: true}' in its front matter.",
            )
        )
    for name in toc_files:
        if f"/{name}" not in pages:
            kind = "notebook" if name in notebooks else "page"
            problems.append(
                BookError(
                    EXIT_PAGE_NOT_BUILT,
                    f"{kind} {name!r} of the table of contents has no built page "
                    "among the page files of this build.",
                )
            )
    return problems


def build_book(root: Path, executed_dir: Path, runner: Runner) -> StagedBook:
    """Stage, build and check; return what was staged when everything passes.

    Parameters
    ----------
    root : Path
        Repository root.
    executed_dir : Path
        Folder of the executed copies, used as the staging folder.
    runner : callable
        Takes the staging folder, builds the book there and returns the exit code. It
        is :func:`run_jupyter_book` for a real build and a stand-in in the tests.

    Raises
    ------
    BookError
        For the first failure, after logging every problem of the check.
    """
    check_executed_dir(root, executed_dir)
    staged = stage_book(root, executed_dir)
    logger.info("BASE_URL is %s", os.environ.get("BASE_URL", "not set"))
    code = runner(staged.stage_dir)
    if code != 0:
        raise BookError(
            code,
            f"the book build exited with code {code} in {staged.stage_dir}; "
            "the book is not built. The build runs with --strict, so a warning above "
            "fails it too.",
        )
    pages = read_built_pages(staged.stage_dir / CONTENT_SUBDIR)
    logger.info("Checking %d built pages: %s", len(pages), ", ".join(sorted(pages)))
    problems = check_built_pages(pages, staged.toc_files, staged.notebooks)
    for problem in problems:
        logger.error("%s", problem)
    if problems:
        raise problems[0]
    return staged


def main(argv: Sequence[str] | None = None, runner: Runner = run_jupyter_book) -> int:
    """Command-line entry point.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments; ``sys.argv[1:]`` when omitted.
    runner : callable, optional
        The build step (see :func:`build_book`); replaced in the tests.

    Returns
    -------
    int
        0 when the book is built and its pages are accepted, otherwise the exit code of
        the failure: that of the build when it failed, or one of the ``EXIT_*`` values.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="repository root, holding myst.yml (default: the repository of this script)",
    )
    parser.add_argument(
        "--executed-dir",
        type=Path,
        default=None,
        help="folder of the executed notebooks (default: <root>/_build/executed)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    root = args.root.resolve()
    executed_dir = (args.executed_dir or root / EXECUTED_SUBDIR).resolve()
    try:
        staged = build_book(root, executed_dir, runner)
    except BookError as error:
        logger.error("Book not built (exit %d): %s", error.exit_code, error)
        return error.exit_code
    logger.info(
        "Book built and checked: %d pages, site in %s",
        len(staged.toc_files),
        staged.stage_dir / HTML_SUBDIR,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
