"""Write the Colab link, the setup cell and the runtime request into every notebook.

Three things are written into each notebook of the table of contents of ``myst.yml``:

* a link, one line of text in the first markdown cell, with a blank line on each side and
  placed after its title line, that opens the notebook on Colab from GitHub at a given
  ref (no image, so that building the book needs no remote file);
* the setup cell, the first code cell, rendered from ``book/_templates/colab_setup.py``;
* the request for the Colab runtime version of ``config/colab.yaml``, in the metadata
  key ``colab.runtime_attributes.runtime_version`` that Colab itself writes.

How the rendering works. The template is valid Python so that ruff and editors read it.
It holds one block between the two comment lines ``# --- values written by
scripts/sync_colab_cells.py (begin) ---`` and ``(end) ---``, made of named constants
(``_REPOSITORY``, ``_REF``, ``_OVERRIDES``, ``_RUNTIME_VERSION``, ``_PYTHON_VERSION``,
``_DATA_DIR``, ``_CACHE_DIR`` and ``_SETUP_DIR``). This script rewrites that block with
the repository, the ref, the lines of ``[tool.uv] override-dependencies`` of
``pyproject.toml`` and the values of ``config/colab.yaml``, each written as a Python
literal with ``json.dumps``, never pasted into code. The rest of the template is copied
as it is. The cell it writes carries the tags ``colab-setup`` and ``hide-input``; the
first one is how this script finds the cell again, so a rerun replaces it and never adds
a second.

The ref of a notebook is the one the notebook already carries (read from its setup cell,
then from its link), or the default ref of the configuration for a notebook that carries
none; ``--ref`` overrides both. The link and the setup cell always name the same ref. The
committed notebooks carry ``main``, so a reader on Colab gets the latest notebook with
the latest package; a numbered release writes its tag into both with ``--ref``. A ref is
one path segment and never holds a slash: a branch whose name holds one is named by its
commit hash. The
script owns one shape of line, the link it writes for the notebook, and no other line of
the cell, an image badge included.

The line that the cell prints to report its duration starts with the prefix
``Setup time in seconds:``, followed by ``install``, ``download`` and ``total`` values.

Exit codes
----------
0: every notebook is in order, or was written.
1: only with ``--check``, at least one notebook differs from what the script writes. A
notebook that the write refuses for its content (no title line or no markdown cell, for
example) is reported here as a difference.
2: a refused ``--ref``, a refused value of the configuration, a notebook that cannot be
read as a notebook (the run stops at it), a notebook that the write refuses, or a ref
that a notebook carries and the script refuses (``--check`` then still reports the other
notebooks before it ends with 2).

A missing or malformed ``config/colab.yaml``, ``pyproject.toml``, template or ``myst.yml``
is not caught: the script then ends with the Python traceback and exit code 1.

When to use
-----------
Run it after editing the template, ``config/colab.yaml`` or the overrides of
``pyproject.toml``, and with ``--check`` in the workflow and before a commit; for a release, with
``--ref <tag>`` before the tag is made.

Examples
--------
    uv run python scripts/sync_colab_cells.py
    uv run python scripts/sync_colab_cells.py --ref v1.0.0
    uv run python scripts/sync_colab_cells.py --check
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("sync_colab_cells")

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_SUBPATH = Path("book") / "_templates" / "colab_setup.py"
CONFIG_SUBPATH = Path("config") / "colab.yaml"
PYPROJECT_SUBPATH = Path("pyproject.toml")
MYST_SUBPATH = Path("myst.yml")

SETUP_TAG = "colab-setup"
HIDE_TAG = "hide-input"
SETUP_TAGS = [SETUP_TAG, HIDE_TAG]
SETUP_CELL_ID = "colab-setup"
LINK_TEXT = "Open this notebook in Google Colab"
COLAB_GITHUB_URL = "https://colab.research.google.com/github"

BLOCK_BEGIN = "# --- values written by scripts/sync_colab_cells.py (begin) ---"
BLOCK_END = "# --- values written by scripts/sync_colab_cells.py (end) ---"

# Characters of a tag, a commit or a branch name without a slash; the ref is written into
# code and into a URL as one path segment, so it never holds a slash.
REF_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
PYTHON_VERSION_PATTERN = re.compile(r"\d+\.\d+")
NOTEBOOK_PATH_PATTERN = re.compile(r"[A-Za-z0-9._/-]+")
CELL_REF_PATTERN = re.compile(r'^_REF = "([^"\n]*)"$', re.MULTILINE)

EXIT_OK = 0
EXIT_DIFFERS = 1
EXIT_BAD_INPUT = 2


class SyncError(Exception):
    """Raised when the input cannot be turned into notebooks the script may write."""


@dataclass(frozen=True)
class Config:
    """The values of ``config/colab.yaml``."""

    repository: str
    default_ref: str
    runtime_version: str
    python_version: str
    data_dir: str
    cache_dir: str
    setup_dir: str


def validate_ref(ref: str) -> str:
    """Return ``ref`` if a tag, a commit or a branch can be named that way.

    Parameters
    ----------
    ref : str
        A git ref written into the link and into the setup cell.

    Returns
    -------
    str
        The same ref.

    Raises
    ------
    SyncError
        If the ref is empty, holds a character other than letters, digits, ``.``, ``_``
        and ``-`` (so no ``/``), starts with ``-``, or holds ``..``.

    Notes
    -----
    When to use: on every ref before it is written anywhere. A ref is one path segment of
    the Colab URL, so a branch whose name holds a slash is named by its commit hash.

    Examples
    --------
    >>> validate_ref("v1.0.0")
    'v1.0.0'
    """
    if not REF_PATTERN.fullmatch(ref) or ref.startswith("-") or ".." in ref:
        raise SyncError(
            f"The ref {ref!r} is refused: a ref is made of letters, digits, '.', '_' "
            "and '-', never a '/', does not start with '-' and does not hold '..'. A branch "
            "whose name holds a slash is named by its commit hash."
        )
    return ref


def load_config(path: Path) -> Config:
    """Read and validate ``config/colab.yaml``.

    Raises
    ------
    SyncError
        If a key is missing or has the wrong type or form.
    """
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise SyncError(f"{path}: expected a mapping.")
    values: dict[str, str] = {}
    for key in (
        "repository",
        "default_ref",
        "colab_runtime_version",
        "python_version",
        "data_dir",
        "cache_dir",
        "setup_dir",
    ):
        value = document.get(key)
        if not isinstance(value, str) or not value:
            raise SyncError(f"{path}: '{key}' must be a non-empty string.")
        values[key] = value
    if not REPOSITORY_PATTERN.fullmatch(values["repository"]):
        raise SyncError(f"{path}: 'repository' must look like <owner>/<name>.")
    if not PYTHON_VERSION_PATTERN.fullmatch(values["python_version"]):
        raise SyncError(f"{path}: 'python_version' must look like \"3.12\".")
    for key in ("data_dir", "cache_dir", "setup_dir"):
        if not values[key].startswith("/"):
            raise SyncError(f"{path}: '{key}' must be an absolute path.")
    validate_ref(values["default_ref"])
    return Config(
        repository=values["repository"],
        default_ref=values["default_ref"],
        runtime_version=values["colab_runtime_version"],
        python_version=values["python_version"],
        data_dir=values["data_dir"],
        cache_dir=values["cache_dir"],
        setup_dir=values["setup_dir"],
    )


def read_overrides(pyproject: Path) -> list[str]:
    """Return the lines of ``[tool.uv] override-dependencies`` of ``pyproject.toml``.

    Raises
    ------
    SyncError
        If the key is missing or is not a list of one-line strings.
    """
    document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    overrides = document.get("tool", {}).get("uv", {}).get("override-dependencies")
    if (
        not isinstance(overrides, list)
        or not overrides
        or not all(
            isinstance(line, str) and line and "\n" not in line for line in overrides
        )
    ):
        raise SyncError(
            f"{pyproject}: [tool.uv] override-dependencies must be a non-empty list "
            "of one-line strings."
        )
    return list(overrides)


def render_cell(template: str, config: Config, ref: str, overrides: list[str]) -> str:
    """Render the setup cell from the template.

    Parameters
    ----------
    template : str
        Text of ``book/_templates/colab_setup.py``.
    config : Config
        Values of ``config/colab.yaml``.
    ref : str
        Git ref the cell installs.
    overrides : list of str
        Lines of ``[tool.uv] override-dependencies`` of ``pyproject.toml``.

    Returns
    -------
    str
        The source of the cell, without a trailing newline.

    Raises
    ------
    SyncError
        If the ref is refused or the template lacks its block of values.

    Notes
    -----
    When to use: to know what a notebook's setup cell must hold.
    """
    validate_ref(ref)
    lines = template.split("\n")
    if lines.count(BLOCK_BEGIN) != 1 or lines.count(BLOCK_END) != 1:
        raise SyncError("The template must hold its block of values exactly once.")
    begin, end = lines.index(BLOCK_BEGIN), lines.index(BLOCK_END)
    if end < begin:
        raise SyncError("The end of the block of values comes before its beginning.")
    values = [
        ("_REPOSITORY", config.repository),
        ("_REF", ref),
        ("_OVERRIDES", overrides),
        ("_RUNTIME_VERSION", config.runtime_version),
        ("_PYTHON_VERSION", config.python_version),
        ("_DATA_DIR", config.data_dir),
        ("_CACHE_DIR", config.cache_dir),
        ("_SETUP_DIR", config.setup_dir),
    ]
    # json.dumps writes a Python literal for strings and lists of strings.
    block = [
        f"{name} = {json.dumps(value, ensure_ascii=False)}" for name, value in values
    ]
    rendered = lines[: begin + 1] + block + lines[end:]
    return "\n".join(rendered).rstrip("\n")


def link_line(config: Config, ref: str, notebook: str) -> str:
    """Return the line of text that opens ``notebook`` on Colab at ``ref``."""
    validate_ref(ref)
    if not NOTEBOOK_PATH_PATTERN.fullmatch(notebook):
        raise SyncError(
            f"The notebook path {notebook!r} holds an unexpected character."
        )
    url = f"{COLAB_GITHUB_URL}/{config.repository}/blob/{ref}/{notebook}"
    return f"[{LINK_TEXT}]({url})"


def owned_line(line: str, notebook: str, config: Config) -> re.Match[str] | None:
    """Match a line of the first markdown cell that this script wrote for ``notebook``.

    The script owns exactly one shape, matched on the whole line: the text link it
    writes, pointing at the Colab URL of this repository for this notebook, at any ref
    (one path segment with no slash, parenthesis or space, so the rest of the target is
    this notebook's path exactly). Any other line, a link to another notebook or an
    image badge included, belongs to the author and is never touched.

    Returns
    -------
    re.Match or None
        The match, whose group ``ref`` is the ref the line names, or None.
    """
    target = (
        re.escape(f"({COLAB_GITHUB_URL}/{config.repository}/blob/")
        + r"(?P<ref>[^/()\s]+)/"
        + re.escape(notebook)
        + r"\)"
    )
    return re.fullmatch(re.escape(f"[{LINK_TEXT}]") + target, line.rstrip("\n"))


def _to_lines(text: str) -> list[str]:
    """Split text on ``\\n`` only, keeping the newline, as a notebook source list does."""
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def _source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def _source_lines(cell: dict[str, Any]) -> list[str]:
    source = cell.get("source", "")
    return _to_lines(source) if isinstance(source, str) else list(source)


def _tags(cell: dict[str, Any]) -> list[str]:
    tags = (cell.get("metadata") or {}).get("tags")
    return list(tags) if isinstance(tags, list) else []


def _is_setup_cell(cell: dict[str, Any]) -> bool:
    return SETUP_TAG in _tags(cell)


def _first_markdown(nb: dict[str, Any]) -> dict[str, Any] | None:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            return cell
    return None


def _title_index(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if line.startswith("# "):
            return index
    return None


def carried_ref(nb: dict[str, Any], notebook: str, config: Config) -> str | None:
    """Return the ref a notebook already carries, or None.

    The setup cell is read first, then the link. The value is returned as written,
    without validation.
    """
    for cell in nb.get("cells", []):
        if _is_setup_cell(cell):
            match = CELL_REF_PATTERN.search(_source_text(cell))
            if match:
                return match.group(1)
            break
    first = _first_markdown(nb)
    if first is not None:
        for line in _source_lines(first):
            match = owned_line(line, notebook, config)
            if match:
                return match.group("ref")
    return None


def dump_notebook(nb: dict[str, Any]) -> str:
    """Return the notebook as nbformat writes it: indent 1, sorted keys, a final newline."""
    return json.dumps(nb, indent=1, ensure_ascii=False, sort_keys=True) + "\n"


def load_notebook(path: Path) -> tuple[dict[str, Any], bool]:
    """Read a notebook; also say whether the file is laid out the way ``dump_notebook`` writes.

    Raises
    ------
    SyncError
        If the file is missing, unreadable, not valid JSON or not a notebook.

    Returns
    -------
    tuple of (dict, bool)
        The notebook, and True if dumping it again gives the file byte for byte.
    """
    try:
        text = path.read_text(encoding="utf-8")
        nb = json.loads(text)
    except (OSError, ValueError) as error:
        raise SyncError(f"{path}: cannot be read as a notebook: {error}") from error
    if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
        raise SyncError(f"{path}: not a notebook (no list of cells).")
    return nb, dump_notebook(nb) == text


def sync_notebook(
    nb: dict[str, Any],
    notebook: str,
    ref: str,
    config: Config,
    template: str,
    overrides: list[str],
) -> dict[str, Any]:
    """Return a copy of ``nb`` with the link, the setup cell and the runtime request.

    Parameters
    ----------
    nb : dict
        The notebook as loaded from JSON.
    notebook : str
        Path of the notebook from the repository root, in POSIX form.
    ref : str
        Git ref written into the link and the cell.
    config : Config
        Values of ``config/colab.yaml``.
    template : str
        Text of the setup cell template.
    overrides : list of str
        Lines of ``[tool.uv] override-dependencies``.

    Returns
    -------
    dict
        The synchronised notebook. Cells other than the first markdown cell and the
        setup cell are the same objects, untouched.

    Raises
    ------
    SyncError
        If the notebook has no markdown cell with a title line, or the cell id is taken.

    Notes
    -----
    When to use: through ``main``; it is separate so that tests build notebooks in memory.
    """
    result = dict(nb)
    cells = list(nb["cells"])

    # Link: the lines this script owns (see owned_line) are removed from the whole first
    # markdown cell, above the title too, and one is written after the title line with a
    # blank line on each side. The blank lines that followed the title are dropped first,
    # so that a cell that already holds the link, or none, comes out the same. Every
    # other line is the author's and is left as it is.
    first = _first_markdown({"cells": cells})
    if first is None:
        raise SyncError(f"{notebook}: no markdown cell to hold the link.")
    lines = [
        line for line in _source_lines(first) if not owned_line(line, notebook, config)
    ]
    title = _title_index(lines)
    if title is None:
        raise SyncError(f"{notebook}: the first markdown cell has no title line.")
    before = lines[:title] + [lines[title].rstrip("\n") + "\n"]
    after = lines[title + 1 :]
    while after and not after[0].strip():
        after.pop(0)
    link = link_line(config, ref, notebook)
    # The last line of a notebook source has no newline.
    link_lines = ["\n", link + "\n", "\n"] if after else ["\n", link]
    new_first = dict(first)
    new_first["source"] = before + link_lines + after
    cells[cells.index(first)] = new_first

    # Setup cell: the tagged cells are replaced by one, placed as the first code cell.
    tagged = [cell for cell in cells if _is_setup_cell(cell)]
    cell_id = SETUP_CELL_ID
    if tagged and isinstance(tagged[0].get("id"), str):
        cell_id = tagged[0]["id"]
    cells = [cell for cell in cells if not _is_setup_cell(cell)]
    if any(cell.get("id") == cell_id for cell in cells):
        raise SyncError(
            f"{notebook}: the cell id {cell_id!r} is taken by another cell."
        )
    setup = {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {"tags": list(SETUP_TAGS)},
        "outputs": [],
        "source": _to_lines(render_cell(template, config, ref, overrides)),
    }
    position = next(
        (i for i, cell in enumerate(cells) if cell.get("cell_type") == "code"),
        len(cells),
    )
    cells.insert(position, setup)
    result["cells"] = cells

    # Runtime request, in the form Colab writes; other keys are left alone.
    metadata = copy.deepcopy(nb.get("metadata") or {})
    colab = metadata.get("colab")
    if not isinstance(colab, dict):
        colab = {}
    colab.setdefault("provenance", [])
    attributes = colab.get("runtime_attributes")
    if not isinstance(attributes, dict):
        attributes = {}
    attributes["runtime_version"] = config.runtime_version
    colab["runtime_attributes"] = attributes
    metadata["colab"] = colab
    result["metadata"] = metadata
    return result


def _first_difference_lines(found: str, expected: str, limit: int = 5) -> list[int]:
    found_lines, expected_lines = found.split("\n"), expected.split("\n")
    differing = [
        number
        for number in range(1, max(len(found_lines), len(expected_lines)) + 1)
        if (found_lines[number - 1 : number] != expected_lines[number - 1 : number])
    ]
    return differing[:limit]


def check_notebook(
    nb: dict[str, Any],
    notebook: str,
    ref: str,
    config: Config,
    template: str,
    overrides: list[str],
) -> list[str]:
    """List what differs between ``nb`` and what ``sync_notebook`` would write for ``ref``.

    Returns
    -------
    list of str
        One sentence per difference; empty when the notebook is what the script writes.

    Notes
    -----
    When to use: through ``main --check``; it names each difference so that the person who
    reads the failure knows what to fix.
    """
    try:
        expected = sync_notebook(nb, notebook, ref, config, template, overrides)
    except SyncError as error:
        return [str(error)]
    problems: list[str] = []
    cells = nb["cells"]

    # Link.
    first = _first_markdown(nb)
    links = [
        line.rstrip("\n")
        for line in (_source_lines(first) if first is not None else [])
        if owned_line(line, notebook, config)
    ]
    wanted = link_line(config, ref, notebook)
    if not links:
        problems.append("the Colab link is missing")
    elif len(links) > 1:
        problems.append(f"{len(links)} Colab links are present, one is expected")
    elif links[0] != wanted:
        problems.append(
            f"the Colab link differs (a different ref than the setup cell): "
            f"expected {wanted!r}, found {links[0]!r}"
        )
    elif _source_lines(first) != _source_lines(_first_markdown(expected)):
        problems.append(
            "the Colab link is not placed after the title line with a blank line on each side"
        )

    # Setup cell.
    tagged = [cell for cell in cells if _is_setup_cell(cell)]
    rendered = render_cell(template, config, ref, overrides)
    if not tagged:
        problems.append(f"the setup cell (tag {SETUP_TAG!r}) is missing")
    else:
        if len(tagged) > 1:
            problems.append(
                f"{len(tagged)} cells carry the tag {SETUP_TAG!r}, one is expected"
            )
        first_code = next((c for c in cells if c.get("cell_type") == "code"), None)
        if first_code is not tagged[0]:
            problems.append("the setup cell is not the first code cell")
        cell = tagged[0]
        if cell.get("cell_type") != "code":
            problems.append("the setup cell is not a code cell")
        text = _source_text(cell)
        if text != rendered:
            numbers = _first_difference_lines(text, rendered)
            problems.append(
                "the setup cell differs from the template rendered for this ref and "
                f"for the overrides of pyproject.toml (first differing lines: {numbers}); "
                "it was edited by hand, or the template, the configuration or the "
                "overrides changed since it was written"
            )
        if _tags(cell) != SETUP_TAGS:
            problems.append(f"the setup cell must carry exactly the tags {SETUP_TAGS}")
        if cell.get("outputs") or cell.get("execution_count") is not None:
            problems.append("the setup cell holds an output or an execution count")

    # Runtime request.
    colab = (nb.get("metadata") or {}).get("colab")
    attributes = colab.get("runtime_attributes") if isinstance(colab, dict) else None
    found_version = (
        attributes.get("runtime_version") if isinstance(attributes, dict) else None
    )
    if found_version is None:
        problems.append(
            "the runtime request (colab.runtime_attributes.runtime_version) is missing"
        )
    elif found_version != config.runtime_version:
        problems.append(
            f"the runtime request is {found_version!r}, the configuration says "
            f"{config.runtime_version!r}"
        )
    elif not isinstance(colab, dict) or colab.get("provenance") is None:
        problems.append(
            "the runtime request lacks the key 'provenance' that Colab writes"
        )

    if not problems and expected != nb:
        problems.append("the notebook differs from what the script writes")
    return problems


def _load_toc_reader() -> Any:
    """Import ``tests/notebook_runner.py``, the one reader of the table of contents."""
    tests_dir = str(DEFAULT_ROOT / "tests")
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    import notebook_runner

    return notebook_runner


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the Colab link, setup cell and runtime request into the notebooks."
    )
    parser.add_argument(
        "--ref",
        help="tag or commit (no slash) for every notebook; by default each notebook keeps the ref it carries",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing and report every notebook; exit 0 when all match, 1 when one differs, 2 when one carries a refused ref or cannot be read",
    )
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT, help="repository root"
    )
    return parser.parse_args(argv)


def run(root: Path, ref: str | None, check: bool) -> int:
    """Synchronise (or check) every notebook of the table of contents of ``root``.

    Returns
    -------
    int
        0 when all is in order, 1 when ``check`` found at least one notebook that differs
        (a notebook the write would refuse for its content counts as one), 2
        (``EXIT_BAD_INPUT``) when ``check`` met a notebook that carries a refused ref.

    Raises
    ------
    SyncError
        If an input is refused; nothing is written then (``main`` turns it into 2).
    """
    config = load_config(root / CONFIG_SUBPATH)
    overrides = read_overrides(root / PYPROJECT_SUBPATH)
    template = (root / TEMPLATE_SUBPATH).read_text(encoding="utf-8")
    if ref is not None:
        validate_ref(ref)
    try:
        notebooks = _load_toc_reader().list_notebooks(root / MYST_SUBPATH)
    except (ValueError, TypeError) as error:
        raise SyncError(f"{root / MYST_SUBPATH}: {error}") from error

    pending: list[tuple[Path, str]] = []
    failed = False
    refused = False
    for notebook in notebooks:
        path = root / notebook
        nb, canonical = load_notebook(path)
        target = ref
        if target is None:
            target = carried_ref(nb, notebook, config) or config.default_ref
            try:
                validate_ref(target)
            except SyncError as error:
                message = (
                    f"{notebook}: the ref it carries is refused: {error} Repair it by "
                    "running the script with `--ref <tag or commit>`."
                )
                if not check:
                    raise SyncError(message) from error
                logger.error("%s", message)
                refused = True
                continue
        if check:
            problems = check_notebook(nb, notebook, target, config, template, overrides)
            if not canonical:
                problems.append(
                    "the JSON layout is not the one nbformat writes (indent 1, sorted keys)"
                )
            for problem in problems:
                logger.error("%s: %s", notebook, problem)
            failed = failed or bool(problems)
            continue
        if not canonical:
            raise SyncError(
                f"{notebook}: the JSON layout is not the one nbformat writes (indent 1, "
                "sorted keys); the script would change lines it does not own."
            )
        synced = sync_notebook(nb, notebook, target, config, template, overrides)
        pending.append((path, dump_notebook(synced)))
        logger.info("%s: ref %s", notebook, target)

    if check:
        if refused:
            return EXIT_BAD_INPUT
        if failed:
            return EXIT_DIFFERS
        logger.info("%d notebook(s) match what the script writes.", len(notebooks))
        return EXIT_OK

    for path, text in pending:
        if path.read_text(encoding="utf-8") == text:
            logger.info("%s: unchanged", path)
        else:
            path.write_text(text, encoding="utf-8")
            logger.info("%s: written", path)
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the exit code."""
    args = _parse_arguments(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return run(args.root, args.ref, args.check)
    except SyncError as error:
        logger.error("%s", error)
        return EXIT_BAD_INPUT


if __name__ == "__main__":
    sys.exit(main())
