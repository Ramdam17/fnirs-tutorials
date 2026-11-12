"""Build the book from the executed notebooks, then check the pages MyST built.

The notebooks are executed first, by ``tests/test_notebooks.py``, which writes the
executed copies under ``_build/executed/`` at the notebooks' own relative paths. This
script stages ``myst.yml``, the pages of the table of contents that are not notebooks, the
local files that those pages, the markdown cells of the notebooks and ``myst.yml``
reference, and the files of the folders that ``project.static_files`` names (the
illustrations of ``interactive/``) next to those copies, runs
``jupyter book build --html --strict`` in that folder without ``--execute``, and reads the page
files that MyST wrote under ``_build/site/content/``. It exits non-zero when

* ``BASE_URL`` is set to something that is not a root path such as ``/fnirs-tutorials``
  (``EXIT_BAD_BASE_URL``, 19, before anything is staged or built: a value without its
  leading slash would make the frame check accept a relative source);
* a file or link of a static folder cannot be staged, for example a symbolic link that
  points out of its folder (``EXIT_FILE_ERROR``, 17);
* the build exits non-zero (the exit code of the build is passed on), which under
  ``--strict`` includes a missing image or a missing ``{include}``;
* a page whose ``location`` does not end with ``.ipynb`` holds a ``block`` of kind
  ``notebook-code`` or an ``inlineExpression``, which MyST would leave without output
  because nothing executes it here, unless the page's own front matter says
  ``execute: {skip: true}``;
* a file of the table of contents has no built page;
* a built page holds a frame (an ``{iframe}`` in its page file, or any ``<iframe>`` in its
  deployed HTML) whose source is not the base URL of the build followed by
  ``/interactive/<path>``, with ``<path>`` a file that the build published under
  ``_build/html/interactive/`` under that exact name, or the page file and the HTML do
  not hold the same frames (``EXIT_FRAME``, 18; see :func:`check_frames`);
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
import html
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("build_book")

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
EXECUTED_SUBDIR = Path("_build") / "executed"
# The only place a frame may point to: MyST copies the folders named by
# ``project.static_files`` into the built site under their own name.
INTERACTIVE_SUBDIR = "interactive"
INTERACTIVE_PREFIX = f"/{INTERACTIVE_SUBDIR}/"
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
EXIT_FRAME = 18
EXIT_BAD_BASE_URL = 19

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
    r"^[ \t>]*(?:`{3,}|~{3,}|:{3,})\{(include|literalinclude|image|figure|download)\}"
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
        if path.is_file() and "__pycache__" not in path.relative_to(helper_dir).parts
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


def _name_key(name: str) -> str:
    """Return a key under which two spellings of one name that a disk may merge are equal."""
    return unicodedata.normalize("NFC", name).casefold()


def spell_as_on_disk(path: Path) -> Path:
    """Return ``path`` with each existing part spelled as its parent folder lists it.

    The default file system of a Mac (APFS) ignores case, so ``Path.resolve`` and
    ``is_file`` accept ``Data/fresh`` for ``data/fresh`` and keep the spelling they were
    given (measured on APFS, Python 3.12.12). Two such spellings then differ as strings
    while naming one file, which makes every comparison of paths by string unsound, and a
    case-sensitive host (the Linux runner, GitHub Pages) tells them apart. Listing the
    parent folder gives the spelling the disk stores. A part that the listing holds
    exactly is kept without a further test; a part that only differs from an entry by
    case or by Unicode normalisation is replaced by that entry when ``os.path.samefile``
    says they are one file. On a case-sensitive disk that test fails, so nothing changes.

    The part that does not exist, and everything after it, is kept as given, and so is a
    part whose folder cannot be listed.

    Parameters
    ----------
    path : Path
        Absolute path. Links are not followed here; see :func:`on_disk`.

    Returns
    -------
    Path
        The same path, spelled as the disk names it for as much of it as exists.
    """
    parts = path.parts
    if not path.is_absolute() or len(parts) < 2:
        return path
    current = Path(parts[0])
    for index, part in enumerate(parts[1:], start=1):
        try:
            names = os.listdir(current)
        except OSError:
            return current.joinpath(*parts[index:])
        if part in names:
            current = current / part
            continue
        key = _name_key(part)
        spelled = None
        for name in names:
            if _name_key(name) != key:
                continue
            try:
                if os.path.samefile(current / part, current / name):
                    spelled = name
                    break
            except OSError:
                continue
        if spelled is None:
            return current.joinpath(*parts[index:])
        current = current / spelled
    return current


def on_disk(path: Path) -> Path:
    """Return the identity of a path: links resolved, each part spelled as the disk spells it.

    This is the one place that decides what a path is. Every path that enters the script
    from outside goes through it once, at the boundary: the repository root, the staging
    folder, each entry of the table of contents, each static entry, each file that a page
    or ``myst.yml`` references, and each file that a listing finds. Past that boundary two
    paths are the same file exactly when they are equal as strings, so the comparisons of
    the script stay plain comparisons.

    When to use
    -----------
    On every path that comes from the command line, ``myst.yml``, a page, or a folder
    listing, before it is compared, tested for containment or turned into a destination.

    Examples
    --------
    >>> on_disk(Path("DATA/Fresh"))  # doctest: +SKIP
    PosixPath('/repo/data/fresh')
    """
    return spell_as_on_disk(path.resolve())


def is_file_or_absent(path: Path) -> bool:
    """Return True when ``path`` is a file, and False when it is not or cannot be tested.

    Python 3.12 lets ``Path.is_file()`` raise for a name the file system refuses (for
    example ENAMETOOLONG for a segment of more than 255 bytes). Use this only for a name
    that comes from the content of the book, a reference or an entry of the table of
    contents, which can be any string: such a name is not a file. Do not use it for a
    path the script builds itself, where an error must be seen.
    """
    try:
        return path.is_file()
    except OSError:
        return False


class _Misspelt(Exception):
    """A reference names an existing file with a spelling the disk does not store."""

    def __init__(self, candidate: Path, spelled: Path) -> None:
        super().__init__(f"{candidate} is spelled {spelled} on the disk")
        self.candidate = candidate
        self.spelled = spelled


def _local_target(raw: str, page: Path, root: Path) -> Path | None:
    """Resolve a reference found in ``page`` to a file of ``root``, or None.

    A fragment (``#...``) and a query (``?...``) are dropped; a URL with a scheme is not
    local; a path that starts with ``/`` is relative to ``root``; any other path is
    relative to the folder of ``page``.

    Raises
    ------
    _Misspelt
        When the file exists but the disk stores its name with another spelling. A host
        that tells case apart would not find it, and the caller decides what that costs.
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
    # Every string of myst.yml comes here, prose included, so the name can be anything.
    if not is_file_or_absent(candidate):
        return None
    spelled = spell_as_on_disk(candidate)
    if spelled != candidate:
        raise _Misspelt(candidate, spelled)
    return candidate


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
            try:
                target = _local_target(raw, current, root)
            except _Misspelt as error:
                raise BookError(
                    EXIT_FILE_ERROR,
                    f"{current}: the reference {raw!r} is spelled differently from the "
                    f"file the disk names, {error.spelled}. This disk ignores case; a "
                    "case-sensitive host (the Linux runner, GitHub Pages) would not "
                    "find it, while MyST finds it here. Write it as the disk spells it.",
                ) from None
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
    document: dict[str, Any],
    myst_path: Path,
    root: Path,
    misspelt: list[tuple[str, Path, Path]] | None = None,
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
    misspelt : list, optional
        Receives ``(string, name as written, name the disk stores)`` for each string that
        names an existing file with a spelling the disk does not store. Such a string is
        warned about and not staged, which is what a case-sensitive host does with it.

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
        try:
            target = _local_target(raw, myst_path, root)
        except _Misspelt as error:
            logger.warning(
                "%s: the string %r names a file the disk spells %s; a case-sensitive "
                "host would not find it, so it is not staged",
                myst_path,
                raw,
                error.spelled,
            )
            if misspelt is not None:
                misspelt.append((raw, error.candidate, error.spelled))
            continue
        if target is None or target == myst_path.resolve() or target in found:
            continue
        if _is_stageable(target, raw, myst_path, root):
            found.append(target)
    return found


def _refuse_links_out_of(folder: Path, myst_path: Path) -> None:
    """Refuse a symbolic link of a static folder whose target lies outside the folder.

    ``rglob`` lists a link to a folder without entering it (measured by
    ``test_rglob_does_not_list_the_files_behind_a_link_to_a_folder``), so the files
    behind such a link would be left out without a word; and a link to a file would be
    staged at the place its target resolves to, which for a link into ``data/`` copies
    a data file into the staging folder. Every entry that is a link is therefore
    resolved, whatever it points to, and refused when the result is not under ``folder``.
    ``folder`` must be resolved.

    Raises
    ------
    BookError
        With ``EXIT_FILE_ERROR``, naming the link and its target.
    """
    for entry in sorted(folder.rglob("*")):
        if not entry.is_symlink():
            continue
        target = entry.resolve()
        spelled = spell_as_on_disk(target)
        if spelled != target:
            raise BookError(
                EXIT_FILE_ERROR,
                f"{entry} in a static folder of {myst_path} is a link to {target}, "
                f"which the disk spells {spelled}. A host that tells case apart would "
                "find no file there; write the target as the disk spells it.",
            )
        if target != folder and folder not in target.parents:
            raise BookError(
                EXIT_FILE_ERROR,
                f"{entry} in a static folder of {myst_path} is a link to {target}, "
                f"which is not under {folder}. A static folder publishes only the "
                "files that lie in it; replace the link by a copy of the file.",
            )


def static_entry_paths(
    document: dict[str, Any], myst_path: Path, root: Path
) -> list[Path]:
    """Return the resolved path of each entry of ``project.static_files``.

    Raises
    ------
    BookError
        With ``EXIT_FILE_ERROR`` when ``static_files`` is not a list of strings, when an
        entry does not exist, or when it lies outside the repository or under ``data/``
        or ``_build/``.
    """
    project = document.get("project")
    entries = project.get("static_files", []) if isinstance(project, dict) else []
    if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
        raise BookError(
            EXIT_FILE_ERROR,
            f"{myst_path}: project.static_files must be a list of paths, got {entries!r}.",
        )
    paths: list[Path] = []
    for entry in entries:
        path = (myst_path.parent / entry).resolve()
        spelled = spell_as_on_disk(path)
        if spelled != path:
            raise BookError(
                EXIT_FILE_ERROR,
                f"static_files entry {entry!r} of {myst_path} is spelled differently "
                f"from the path the disk names, {spelled}. This disk ignores case; the "
                "Linux runner and GitHub Pages do not, and there the entry names no "
                "file. Write it as the disk spells it.",
            )
        try:
            relative = path.relative_to(root.resolve())
        except ValueError:
            raise BookError(
                EXIT_FILE_ERROR,
                f"static_files entry {entry!r} points outside the repository {root}.",
            ) from None
        if not relative.parts or relative.parts[0] in NEVER_STAGED:
            raise BookError(
                EXIT_FILE_ERROR,
                f"static_files entry {entry!r} is the repository itself or under "
                f"{'/ or '.join(NEVER_STAGED)}/, which are never staged.",
            )
        if not (path.is_file() or path.is_dir()):
            raise BookError(
                EXIT_FILE_ERROR,
                f"static_files entry {entry!r} of {myst_path} names no file or folder "
                f"({path}); a frame that points into it would have no target.",
            )
        paths.append(path)
    return paths


def check_static_entries(
    root: Path,
    executed_dir: Path,
    entries: list[Path],
    toc_files: list[str],
    myst_path: Path,
) -> None:
    """Refuse a static entry that overlaps what the staging folder holds for another reason.

    The staging folder holds the executed notebooks, the staged pages and the site, which
    this script writes or removes for their own sake. A static entry is copied from the
    repository and its staged copy is removed before each build, so it must hold none of
    them: an entry that contains a page or a notebook of the table of contents would have
    the executed copy of that notebook removed and the source copied over it, and an entry
    that is the staging folder or one of its parents would remove the staging folder.

    Parameters
    ----------
    root, executed_dir : Path
        Repository root and staging folder, both resolved.
    entries : list of Path
        Resolved entries, as returned by :func:`static_entry_paths`.
    toc_files : list of str
        ``file:`` entries of the table of contents, relative to ``root``.
    myst_path : Path
        ``myst.yml``, for the message.

    Raises
    ------
    BookError
        With ``EXIT_FILE_ERROR``, naming the entry and the page or folder it overlaps, before
        anything is removed or copied.
    """
    stage = executed_dir.resolve()
    for entry in entries:
        name = entry.relative_to(root.resolve()).as_posix()
        if entry == stage or entry in stage.parents or stage in entry.parents:
            raise BookError(
                EXIT_FILE_ERROR,
                f"static_files entry {name!r} of {myst_path} is the staging folder "
                f"{executed_dir} or one of its parents or children; the staged copy of an "
                "entry is removed before each build, so it would remove the staging "
                "folder.",
            )
        for page in toc_files:
            target = (root / page).resolve()
            if target == entry or entry in target.parents:
                raise BookError(
                    EXIT_FILE_ERROR,
                    f"static_files entry {name!r} of {myst_path} holds the page {page!r} "
                    "of the table of contents. A static entry is copied as it is and its "
                    "staged copy is removed before each build, which would remove the "
                    "executed copy of a notebook; name a folder that holds no page.",
                )


def find_static_files(
    document: dict[str, Any], myst_path: Path, root: Path
) -> list[Path]:
    """List the files of the folders and files that ``project.static_files`` names.

    MyST copies each entry into the built site, a folder recursively under its own name
    (https://mystmd.org/guide/website-downloads), so what is staged here is what the site
    publishes. A file or folder whose name starts with a dot is left out, so that
    ``.DS_Store`` or a hidden editor file is never published, and so is a
    ``__pycache__`` folder.

    Parameters
    ----------
    document : dict
        The loaded ``myst.yml``.
    myst_path : Path
        Path of ``myst.yml``; its folder is the base of the entries.
    root : Path
        Repository root.

    Returns
    -------
    list of Path
        Absolute paths of the files to stage, in sorted order, each once.

    Raises
    ------
    BookError
        With ``EXIT_FILE_ERROR`` when ``static_files`` is not a list of strings, when an
        entry does not exist, when it lies outside the repository or under ``data/`` or
        ``_build/``, or when a symbolic link inside a folder points out of that folder.
        A static folder that is missing would publish a frame target that does not
        exist, and MyST would not say so.
    """
    found: list[Path] = []
    for path in static_entry_paths(document, myst_path, root):
        if path.is_file():
            candidates = [path]
        else:
            _refuse_links_out_of(path, myst_path)
            candidates = sorted(p for p in path.rglob("*") if p.is_file())
        for candidate in candidates:
            parts = candidate.relative_to(path if path.is_dir() else path.parent).parts
            if any(part.startswith(".") or part == "__pycache__" for part in parts):
                logger.info("Not staged (hidden or cache file): %s", candidate)
                continue
            spelled = on_disk(candidate)
            if spelled not in found:
                found.append(spelled)
    return found


def check_misspelt_strings(
    root: Path,
    executed_dir: Path,
    misspelt: list[tuple[str, Path, Path]],
    writes: list[StageOperation],
    myst_path: Path,
) -> None:
    """Refuse a string of ``myst.yml`` whose file reaches the staging folder another way.

    A string of ``myst.yml`` that names an existing file with another spelling is left
    unstaged with a warning: MyST then finds nothing on either kind of disk, and the
    string may be prose that is not meant as a path. When the same file is written by
    the plan, or the staging folder already holds it (a page, an executed copy), MyST
    finds it on a disk that ignores case and not on one that tells case apart, so the
    string is refused.

    Raises
    ------
    BookError
        With ``EXIT_FILE_ERROR``, naming the string and the spelling of the disk.
    """
    for raw, _candidate, spelled in misspelt:
        place = executed_dir / spelled.relative_to(root.resolve())
        if os.path.lexists(place) or any(op.path == place for op in writes):
            raise BookError(
                EXIT_FILE_ERROR,
                f"{myst_path}: the string {raw!r} is spelled differently from the file "
                f"the disk names, {spelled}, and the build stages that file anyway. MyST "
                "would find it on this disk and not on a host that tells case apart. "
                "Write it as the disk spells it; a string that is not meant as a path "
                "has to be worded otherwise.",
            )


def check_toc_spelling(root: Path, toc_files: list[str], myst_path: Path) -> None:
    """Refuse an entry of the table of contents that the disk spells differently.

    On a disk that ignores case the notebook command and MyST find ``Book/Page.md`` for
    ``book/page.md``; the Linux runner does not, and the book would not build there.
    The entry is compared with :func:`spell_as_on_disk` of the path it names; an entry
    that names no file is left to MyST, which reports it.

    Raises
    ------
    BookError
        With ``EXIT_BAD_TOC``, naming the entry and the spelling of the disk.
    """
    for name in toc_files:
        try:
            target = (root / name).resolve()
        except OSError:
            continue
        spelled = spell_as_on_disk(target)
        if spelled != target:
            raise BookError(
                EXIT_BAD_TOC,
                f"the table of contents of {myst_path} names {name!r}, which the disk "
                f"spells {spelled.relative_to(root.resolve()).as_posix()!r}. This disk "
                "ignores case; the Linux runner and GitHub Pages do not, and there the "
                "page would not be found. Write the entry as the disk spells it.",
            )


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


REMOVE = "remove"
COPY = "copy"


@dataclass(frozen=True)
class StageOperation:
    """One operation on the staging folder: a removal or a write, with its exact path.

    Attributes
    ----------
    kind : str
        ``REMOVE`` or ``COPY``.
    path : Path
        Exact path below the staging folder, as the script will use it, links not
        resolved. A removal whose path is a link unlinks the link and never follows it.
        A copy writes a file at this path and creates the folders above it.
    source : Path or None
        For a copy, the file of the repository to read.
    """

    kind: str
    path: Path
    source: Path | None = None


@dataclass
class StagePlan:
    """The operations a build performs on the staging folder, every refusal already made.

    Attributes
    ----------
    root, executed_dir : Path
        Repository root and staging folder.
    toc_files, notebooks : list of str
        Files of the table of contents, and the notebooks among them.
    operations : list of StageOperation
        Every removal, then every write, in the order of execution. The execution touches
        the staging folder through these and no other path.
    """

    root: Path
    executed_dir: Path
    toc_files: list[str]
    notebooks: list[str]
    operations: list[StageOperation]

    @property
    def copies(self) -> list[StageOperation]:
        """The writes of the plan, in order."""
        return [op for op in self.operations if op.kind == COPY]


def owned_by_others(executed_dir: Path, notebooks: list[str]) -> list[Path]:
    """List what the staging folder holds that this script neither writes nor removes.

    The executed copy of each notebook of the table of contents, at ``executed_dir / name``
    as the table of contents writes it (the notebook command writes it there and removes
    it itself); and everything MyST keeps in ``_build`` apart from the site and the HTML
    this script clears: the downloaded theme, ``_build/templates``, which is slow to fetch,
    and any other entry that the folder holds. The listing is read from the disk, since
    MyST can add an entry.

    Returns
    -------
    list of Path
        Exact paths below ``executed_dir``.
    """
    owned = [executed_dir / name for name in notebooks]
    build_dir = executed_dir / HTML_SUBDIR.parent
    keep_apart = {SITE_SUBDIR.name, HTML_SUBDIR.name}
    try:
        names = os.listdir(build_dir)
    except OSError:
        names = []
    owned.extend(build_dir / name for name in names if name not in keep_apart)
    owned.append(build_dir / "templates")
    return owned


def check_stage_operations(
    executed_dir: Path, operations: list[StageOperation], owned: list[Path]
) -> None:
    """Refuse an operation that leaves the staging folder or touches what is not ours.

    Two invariants, checked on each operation whatever input produced it. Containment: no
    folder between the staging folder and the path is a link, so the operation happens
    inside the staging folder as the disk sees it; a path to write is not a link either.
    A link that an earlier removal of the plan removes does not count, since it is gone
    when the operation runs. A path to remove that is itself a link is allowed: it is
    unlinked and never followed. Ownership: no removal is a path of ``owned`` or one of
    its ancestors, and no write replaces one or lies under one.

    Raises
    ------
    BookError
        With ``EXIT_BAD_EXECUTED_DIR``, naming the operation and the reason, before
        anything is removed or written.
    """
    removed: list[Path] = []

    def gone(path: Path) -> bool:
        return any(r == path or r in path.parents for r in removed)

    for op in operations:
        try:
            relative = op.path.relative_to(executed_dir)
        except ValueError:
            relative = None
        if relative is None or not relative.parts or ".." in relative.parts:
            raise BookError(
                EXIT_BAD_EXECUTED_DIR,
                f"{op.kind} of {op.path} is not a path below the staging folder "
                f"{executed_dir}; nothing was changed.",
            )
        current = executed_dir
        for part in relative.parts[:-1]:
            current = current / part
            if current.is_symlink() and not gone(current):
                raise BookError(
                    EXIT_BAD_EXECUTED_DIR,
                    f"the {op.kind} of {op.path} would go through {current}, a link of "
                    f"the staging folder {executed_dir}, and leave it; nothing was "
                    "changed. Remove the link.",
                )
        if op.kind == COPY and op.path.is_symlink() and not gone(op.path):
            raise BookError(
                EXIT_BAD_EXECUTED_DIR,
                f"{op.path} in the staging folder {executed_dir} is a link, and the "
                "copy would write through it; nothing was changed. Remove the link.",
            )
        for held in owned:
            hits = (
                held == op.path or op.path in held.parents
                if op.kind == REMOVE
                else held == op.path or held in op.path.parents
            )
            if hits:
                raise BookError(
                    EXIT_BAD_EXECUTED_DIR,
                    f"the {op.kind} of {op.path} would reach {held}, which the notebook "
                    "command or MyST keeps in the staging folder and this script does "
                    "not own; nothing was changed.",
                )
        if op.kind == REMOVE:
            removed.append(op.path)


def plan_stage(root: Path, executed_dir: Path) -> StagePlan:
    """Read ``myst.yml``, the pages and the folders, and list what is done to the staging folder.

    This function writes and removes nothing. It refuses, before the execution touches
    anything:

    * a table of contents that the notebook reader refuses, or an entry that the disk
      spells differently;
    * a notebook without a fresh executed copy;
    * a static entry that is not a list of existing paths in the repository, that is
      spelled differently, lies under ``data/`` or ``_build/``, holds a page of the table
      of contents, or is the staging folder or one of its parents or children;
    * a link of a static folder that leads out of it or is spelled differently;
    * a reference of a page or of a notebook that the disk spells differently, and a
      string of ``myst.yml`` that does so when the file it names is staged anyway;
    * a source that does not resolve under the repository;
    * an operation on the staging folder that goes through a link or that touches what
      the script does not own (:func:`check_stage_operations`).

    The staging folder is owned as follows. The executed copies and ``_build/templates``
    belong to the notebook command and to MyST; ``_build/site``, ``_build/html`` and the
    staged copy of each static entry are the script's, and the plan lists their removal.
    ``root`` and ``executed_dir`` must have gone through :func:`on_disk`.

    Parameters
    ----------
    root : Path
        Repository root, holding ``myst.yml``.
    executed_dir : Path
        Folder of the executed copies; it becomes the staging folder.

    Returns
    -------
    StagePlan
        The operations, removals first.

    Raises
    ------
    BookError
        For any refusal above.
    """
    myst_path = root / "myst.yml"
    toc_files, notebooks = read_toc_files(myst_path)
    check_toc_spelling(root, toc_files, myst_path)
    check_executed_copies(root, executed_dir, notebooks)
    document = load_myst_document(myst_path)
    static_entries = static_entry_paths(document, myst_path, root)
    check_static_entries(root, executed_dir, static_entries, toc_files, myst_path)
    sources = [myst_path]
    for name in toc_files:
        page = root / name
        if name in notebooks:
            sources.extend(find_referenced_files(page, root))
            continue
        if not is_file_or_absent(page):
            # MyST reports a missing page itself; the check then finds no built page.
            logger.warning("Page %s of the table of contents does not exist", name)
            continue
        sources.append(page)
        sources.extend(find_referenced_files(page, root))
    misspelt: list[tuple[str, Path, Path]] = []
    for config_file in find_config_files(document, myst_path, root, misspelt):
        sources.append(config_file)
        if config_file.suffix.lower() == ".md":
            sources.extend(find_referenced_files(config_file, root))
    sources.extend(find_static_files(document, myst_path, root))
    writes: list[StageOperation] = []
    seen: set[Path] = set()
    for source in sources:
        resolved = source.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            relative = resolved.relative_to(root.resolve())
        except ValueError:
            raise BookError(
                EXIT_FILE_ERROR,
                f"cannot stage {source}: it resolves to {resolved}, which is not "
                f"under the repository {root}.",
            ) from None
        writes.append(StageOperation(COPY, executed_dir / relative, source))
    check_misspelt_strings(root, executed_dir, misspelt, writes, myst_path)
    removals = [
        StageOperation(REMOVE, executed_dir / sub)
        for sub in (SITE_SUBDIR, HTML_SUBDIR)
        if os.path.lexists(executed_dir / sub)
    ]
    for entry in static_entries:
        copy = executed_dir / entry.relative_to(root.resolve())
        if os.path.lexists(copy):
            removals.append(StageOperation(REMOVE, copy))
    operations = removals + writes
    check_stage_operations(
        executed_dir, operations, owned_by_others(executed_dir, notebooks)
    )
    return StagePlan(root, executed_dir, toc_files, notebooks, operations)


def _remove(path: Path) -> bool:
    """Remove a file, a link (never followed) or a folder; return False if it is absent."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        return False
    return True


def execute_stage(plan: StagePlan) -> StagedBook:
    """Run the operations of the plan in order, and compute no path of its own.

    Every refusal was made by :func:`plan_stage`, so this raises only for what the disk
    does: a file that cannot be removed, read or written.

    Returns
    -------
    StagedBook
        The files staged.
    """
    staged = StagedBook(
        stage_dir=plan.executed_dir,
        toc_files=plan.toc_files,
        notebooks=plan.notebooks,
    )
    for op in plan.operations:
        relative = op.path.relative_to(plan.executed_dir)
        if op.kind == REMOVE:
            if _remove(op.path):
                logger.info("Removed %s from an earlier build", op.path)
            continue
        assert op.source is not None
        try:
            op.path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(op.source, op.path)
        except OSError as error:
            raise BookError(
                EXIT_FILE_ERROR,
                f"cannot stage {relative.as_posix()} to {op.path}: "
                f"{type(error).__name__}: {error}",
            ) from error
        staged.copied.append(relative)
        logger.info("Staged %s", relative.as_posix())
    return staged


def stage_book(root: Path, executed_dir: Path) -> StagedBook:
    """Stage ``myst.yml``, the non-notebook pages and their references.

    The references of the notebooks (their markdown cells), the files that ``myst.yml``
    names (the plugin and the style file among them, found by :func:`find_config_files`)
    and the files of the folders of ``project.static_files`` (found by
    :func:`find_static_files`) are staged too; the notebooks themselves are not, since the
    staging folder holds their executed copies. The target of an ``{iframe}`` is not found
    by reading the page: it is staged because it lies in a static folder, and MyST
    publishes that folder.

    It is :func:`plan_stage`, which refuses without touching the disk, followed by
    :func:`execute_stage`, which removes and copies.

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
        See :func:`plan_stage` and :func:`execute_stage`.
    """
    return execute_stage(plan_stage(root, executed_dir))


def run_jupyter_book(stage_dir: Path) -> int:
    """Run ``jupyter book build --html --strict`` in the staging folder, without ``--execute``.

    ``--strict`` makes the build stop, with a non-zero exit, on an error that MyST reports
    on a page. Measured with jupyter-book 2.1.7 (myst-cli 1.11.0): without it a missing
    image and a missing ``{include}`` are reported and the build exits 0, with it both
    exit 1 ("Site has 2 errors and 0 warning, stopping build"), and the book itself builds
    with exit 0. Two warnings that were measured, the theme's ignored option and a missing
    ``bibliography`` file of ``myst.yml``, still exit 0 under it, and so does a missing
    local page named by an ``{iframe}`` (no message at all), and an entry of the table of
    contents that does not exist; that is why :func:`stage_book` stages the files a page
    references and the static folders, :func:`check_frames` checks that the target of every
    frame was published and :func:`check_built_pages` compares the table of contents with
    the built pages.
    A citation of a DOI that doi.org does not answer is an error of the page too: with
    doi.org unreachable, the same page exits 0 without ``--strict`` and 1 with it, so a
    network failure outside the repository can fail the build.

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
    ``execute: {skip: true}`` too, so one key is read. Written under ``project:`` in
    ``myst.yml``, ``execute: {skip: true}`` is not accepted by MyST 2.1.7: the build answers
    "'config.project' extra key ignored: execute" and the page records no ``execute``, so
    there is nothing to inherit and the key never excuses a page here. Both facts were
    measured with jupyter-book 2.1.7 (myst-cli 1.11.0), and the second only for that
    spelling.
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


def find_frame_sources(mdast: Any) -> list[str]:
    """Return the ``src`` of every ``iframe`` node of a MyST syntax tree.

    Parameters
    ----------
    mdast : Any
        The ``mdast`` value of a page file, a tree of dicts and lists.

    Returns
    -------
    list of str
        The source of each frame as recorded in the built page, in no particular order.
        MyST records the source as written, a relative path unchanged and a root path
        without the base URL; only the plugin of the book adds the base URL, and only to a
        source that starts with ``/interactive/``.
    """
    sources: list[str] = []
    stack = [mdast]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("type") == "iframe" and isinstance(node.get("src"), str):
                sources.append(node["src"])
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return sources


def normalize_base_url(raw: str | None) -> str:
    """Return the base URL of the build without trailing slashes, or ``""`` when unset.

    The one base URL of a build is the environment variable ``BASE_URL``. MyST, the plugin
    ``plugins/iframe-base-url.mjs`` and this script all read that variable and nothing
    else, and none of them strips spaces (measured on a real build: MyST fails on
    ``" /x"`` and on ``"  "``, and the plugin prefixes them as they are), so neither does
    this function. The plugin and this function remove trailing slashes alike; the
    value is judged by :func:`read_base_url` before the build starts.
    """
    return (raw or "").rstrip("/")


# A base URL is a path on the host: it starts with one slash, and holds no space, query,
# fragment or backslash. ``//host`` is refused because ``//host/interactive/x`` would
# pass for a source of the book while naming another host.
_BASE_URL_FORM = re.compile(r"/(?!/)[^\s?#\\]*")


def read_base_url() -> str:
    """Read ``BASE_URL`` from the environment, refuse a malformed one, and normalise it.

    The value is judged exactly as MyST and the plugin receive it. The accepted form is an
    unset or empty variable, or a root path such as ``/fnirs-tutorials`` with optional
    trailing slashes. A value made of spaces, or with a space anywhere, is refused: it is
    not empty for MyST and the plugin, and MyST then fails on it.
    Without its leading slash the value would make the check accept a relative frame
    source (``fnirs-tutorials/interactive/x`` starts with the prefix it builds), which
    depends on the address the page is read at.

    Raises
    ------
    BookError
        With ``EXIT_BAD_BASE_URL``, giving the value and the accepted form.
    """
    raw = os.environ.get("BASE_URL")
    base_url = normalize_base_url(raw)
    if base_url and not _BASE_URL_FORM.fullmatch(base_url):
        raise BookError(
            EXIT_BAD_BASE_URL,
            f"BASE_URL is {raw!r}, which is not a root path. It must be empty or start "
            "with one slash and hold no space, query or fragment, for example "
            "'/fnirs-tutorials'.",
        )
    return base_url


def _has_exact_names(base: Path, segments: list[str]) -> bool:
    """Return True when each segment is the name of an entry as the folder lists it.

    On macOS the default file system ignores case, so ``is_file`` accepts
    ``Banana.html`` for ``banana.html``, which a case-sensitive host, such as GitHub
    Pages, does not serve. The spelling the disk gives, :func:`spell_as_on_disk`, is the
    spelling asked for exactly when the answer is yes. A link along the way is not
    followed, as a frame source names a path of the published site.
    """
    asked = base.joinpath(*segments)
    return spell_as_on_disk(asked) == asked


def _frame_refusal(src: str, base_url: str, interactive_dir: Path | None) -> str | None:
    """Say why a frame source is refused, or return None when it is accepted.

    The accepted shape is ``<base_url>/interactive/<path>``, with ``<path>`` made of plain
    file names, naming a file of ``interactive_dir``, the ``interactive`` folder of the
    built site.
    """
    prefix = f"{base_url}{INTERACTIVE_PREFIX}"
    if not src.startswith(prefix):
        if _URL_SCHEME.match(src):
            return (
                "it has a scheme, and no page loads an illustration from another host"
            )
        if src.startswith("//"):
            return "it names another host"
        if not src.startswith("/"):
            return "it is a relative path, which depends on the address the page is read at"
        if base_url and src.startswith(INTERACTIVE_PREFIX):
            return (
                f"it does not start with the base URL {base_url!r} of this build, so the "
                "browser would ask the host for a file outside the book"
            )
        return f"it is a root path outside {INTERACTIVE_PREFIX!r}"
    path = src[len(prefix) :]
    segments = path.split("/")
    if (
        not path
        or any(s in ("", ".", "..") for s in segments)
        or re.search(r"[?#%\\]", path)
    ):
        return (
            "the part after the folder name is empty, leaves the folder, or holds a "
            "query, a fragment, a percent sign or a backslash"
        )
    if interactive_dir is None:
        return "no built site was given to look for the target in"
    target = (interactive_dir / path).resolve()
    try:
        target.relative_to(interactive_dir.resolve())
    except ValueError:
        return "its target is outside the interactive folder of the built site"
    if not (_has_exact_names(interactive_dir, segments) and is_file_or_absent(target)):
        return (
            f"its target {target} does not exist, under that exact name: the build did not publish it, and "
            "MyST says nothing when a frame target is missing"
        )
    return None


def _html_page_path(html_dir: Path, slug: str) -> Path:
    """Return the HTML file MyST writes for a page slug: ``<slug>/index.html``."""
    return (
        html_dir / "index.html" if slug == "index" else html_dir / slug / "index.html"
    )


# A tag as the browser reads it: quoted values may hold a ``>``.
_IFRAME_TAG = re.compile(r"<iframe\b((?:[^>\"']|\"[^\"]*\"|'[^']*')*)>", re.IGNORECASE)
_ATTRIBUTE = re.compile(
    r"([^\s=/>\"']+)(?:\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>\"']*))?", re.IGNORECASE
)


_WHY_SRCDOC = "this iframe has a srcdoc, a page written inside the tag, which this check does not read"
_WHY_QUOTING = (
    "the HTML page that is deployed writes its src in a form that is not double-quoted, "
    "which this check does not read"
)


def read_html_frames(text: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Return the ``src`` of every ``<iframe>`` of a deployed page, and what cannot be read.

    What counts as a frame: any ``<iframe`` tag of the HTML, whatever the case, written
    anywhere in the page. The theme writes ``src="..."`` with double quotes (measured in
    the HTML of the book: ``<iframe width="100%" height="100%" src="/..." title="..." ...>``),
    and a source is read only in that form, with HTML entities decoded.

    Everything that looks like a frame and cannot be read is returned in the second list,
    as ``(what was found, why it is refused)``, so that the caller refuses it: a change of
    the markup must make the check refuse and never accept. That is a ``src`` in another
    form (single quotes, no quotes), an ``<iframe>`` that has a ``srcdoc`` (its content is
    a page of its own, which no rule of this check looks at), and a page in which the text
    ``<iframe`` occurs more often than the tags this reader parsed (a tag with an
    unbalanced quote, or cut off, which the pattern cannot read). An ``<iframe>`` with no
    ``src`` and no ``srcdoc`` loads nothing and is left out.

    Returns
    -------
    tuple of (list of str, list of (str, str))
        The sources read, and the unreadable frames with their reasons.
    """
    sources: list[str] = []
    unreadable: list[tuple[str, str]] = []
    parsed = 0
    for match in _IFRAME_TAG.finditer(text):
        parsed += 1
        for name, value in _ATTRIBUTE.findall(match.group(1)):
            lowered = name.lower()
            if lowered == "srcdoc":
                unreadable.append((match.group(0), _WHY_SRCDOC))
            elif lowered == "src":
                if value.startswith('"') and value.endswith('"') and len(value) >= 2:
                    sources.append(html.unescape(value[1:-1]))
                else:
                    unreadable.append((match.group(0), _WHY_QUOTING))
    occurrences = len(re.findall(r"<iframe", text, re.IGNORECASE))
    if occurrences > parsed:
        found = f"{occurrences} occurrences of '<iframe'"
        why = (
            f"the text '<iframe' occurs {occurrences} times in the HTML page that is "
            f"deployed but only {parsed} tags were parsed, so a tag this reader cannot "
            "parse (an unbalanced quote, a tag cut off) would be passed over"
        )
        unreadable.append((found, why))
    return sources, unreadable


def check_frames(
    location: str,
    page: dict[str, Any],
    base_url: str = "",
    html_dir: Path | None = None,
) -> list[BookError]:
    """Check the frames of a built page: each must point at a file the build published.

    A frame is accepted when its source is exactly ``<base_url>/interactive/<path>`` and
    ``<path>`` is a file under ``<html_dir>/interactive/``. Nothing else is accepted: a
    relative source (it holds only at one of the two addresses a page is reached at), a
    source with a scheme (https included, since no page needs one), a protocol-relative
    source, a root path outside ``/interactive/``, a source without the base URL when the
    build has one, a path that leaves the folder, and a target the build did not publish
    (MyST says nothing about a missing one, even under ``--strict``).

    Two built files are read, and they must agree in both directions. The rule is applied
    to the ``src`` of every ``iframe`` node in the page file under
    ``_build/site/content/``, which is what the plugin of the book rewrote and what
    :func:`find_frame_sources` reads. The page that is deployed is the HTML under
    ``_build/html/<slug>/index.html``, and it is read for every page, whether or not the
    page file holds a frame: each frame of the page file must be in the HTML with the same
    ``src``, and each ``<iframe>`` of the HTML must be a frame of the page file and pass
    the same rule. A frame that only the HTML holds (a raw ``<iframe>`` written in a
    markdown page, if MyST lets one through) is refused, and so is anything that looks like
    a frame and cannot be read: a ``src`` that is not written in double quotes, a
    ``srcdoc``, and a page in which ``<iframe`` occurs more often than tags were parsed
    (see :func:`read_html_frames`). A page whose HTML is missing
    is refused only when the page file holds a frame; for a page without one, nothing
    in this script reports a missing HTML file, because MyST's ``--strict`` build does and
    the table-of-contents check covers the page file. The source page of the book is not
    read: the plugin leaves it unchanged, so it does not show the base URL.

    Parameters
    ----------
    location : str
        ``location`` of the page, for the message.
    page : dict
        The page file as read by :func:`read_built_pages`.
    base_url : str, optional
        Base URL of the build, without trailing slash; ``""`` when unset.
    html_dir : Path, optional
        ``_build/html`` of the build. When omitted every frame is refused, since its
        target cannot be found.

    Returns
    -------
    list of BookError
        One per refused frame, with ``EXIT_FRAME``; the message names the page, the source
        as found, why it is refused and what is accepted.
    """
    problems: list[BookError] = []
    interactive_dir = html_dir / INTERACTIVE_SUBDIR if html_dir is not None else None
    where = (
        interactive_dir if interactive_dir is not None else "_build/html/interactive"
    )
    accepted = (
        f"A frame is accepted only when its source is {base_url}{INTERACTIVE_PREFIX}<path>, "
        f"with <path> a file under {where}."
    )

    def refuse(src: str, reason: str) -> None:
        problems.append(
            BookError(
                EXIT_FRAME,
                f"page {location!r} holds a frame whose source is {src!r}, which is "
                f"refused because {reason}. {accepted}",
            )
        )

    page_sources = find_frame_sources(page["mdast"])
    html_sources: list[str] = []
    unreadable: list[str] = []
    if html_dir is not None:
        slug = page.get("slug")
        html_file = _html_page_path(html_dir, slug) if isinstance(slug, str) else None
        try:
            text = html_file.read_text(encoding="utf-8") if html_file else ""
        except OSError:
            text = ""
        html_sources, unreadable = read_html_frames(text)
    for src in page_sources:
        reason = _frame_refusal(src, base_url, interactive_dir)
        if reason is None and html_dir is not None and src not in html_sources:
            reason = (
                "the HTML page that is deployed does not hold an iframe with this "
                f"source (it holds {html_sources!r})"
            )
        if reason is not None:
            refuse(src, reason)
    for src in html_sources:
        if src in page_sources:
            continue
        reason = _frame_refusal(src, base_url, interactive_dir)
        refuse(
            src,
            reason
            or "the HTML page that is deployed holds it but the page file does not "
            "record it",
        )
    for found, why in unreadable:
        refuse(found, why)
    return problems


def check_built_pages(
    pages: dict[str, dict[str, Any]],
    toc_files: list[str],
    notebooks: list[str],
    base_url: str = "",
    html_dir: Path | None = None,
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
    base_url : str, optional
        Base URL of the build, without trailing slash (see :func:`normalize_base_url`).
    html_dir : Path, optional
        ``_build/html`` of the build, where the targets of the frames are looked for.

    Returns
    -------
    list of BookError
        One per problem, empty when the pages are accepted: a page that is not a
        notebook and holds code to execute, a frame that :func:`check_frames` refuses,
        and a file of the table of contents with no built page.
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
    for location, page in sorted(pages.items()):
        problems.extend(check_frames(location, page, base_url, html_dir))
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
    base_url = read_base_url()
    check_executed_dir(root, executed_dir)
    staged = stage_book(root, executed_dir)
    logger.info("BASE_URL is %s", os.environ.get("BASE_URL", "not set"))
    code = runner(staged.stage_dir)
    if code != 0:
        raise BookError(
            code,
            f"the book build exited with code {code} in {staged.stage_dir}; "
            "the book is not built. The build runs with --strict, so an error that MyST "
            "reports on a page stops it. A diagnostic about a DOI or another network "
            "lookup can come from outside the repository: run the build a second time "
            "to tell a lookup that failed once from a defect of the book.",
        )
    pages = read_built_pages(staged.stage_dir / CONTENT_SUBDIR)
    logger.info("Checking %d built pages: %s", len(pages), ", ".join(sorted(pages)))
    problems = check_built_pages(
        pages,
        staged.toc_files,
        staged.notebooks,
        base_url,
        staged.stage_dir / HTML_SUBDIR,
    )
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
    root = on_disk(args.root)
    executed_dir = on_disk(args.executed_dir or root / EXECUTED_SUBDIR)
    try:
        staged = build_book(root, executed_dir, runner)
    except BookError as error:
        logger.error("Book not built (exit %d): %s", error.exit_code, error)
        return error.exit_code
    except OSError as error:
        # A file operation the script did not guard: report the whole text of the error,
        # each file name that is set, and the traceback, with its own exit code, which is
        # not the exit code of a failed build.
        names = [name for name in (error.filename, error.filename2) if name is not None]
        logger.exception(
            "Book not built (exit %d): %s on %s: %s",
            EXIT_FILE_ERROR,
            type(error).__name__,
            " and ".join(str(name) for name in names) or "no named file",
            error,  # noqa: TRY401 - the first line must carry the error text
        )
        return EXIT_FILE_ERROR
    logger.info(
        "Book built and checked: %d pages, site in %s",
        len(staged.toc_files),
        staged.stage_dir / HTML_SUBDIR,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
