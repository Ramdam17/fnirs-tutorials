"""Make the page files of this folder: build one scratch MyST project and copy its output.

When to use
-----------
Run it after an upgrade of Jupyter Book (or of the MyST engine it bundles), then run
``tests/test_build_book.py``. The files here are what MyST writes under
``_build/site/content/`` for one page each, and ``scripts/build_book.py`` decides from them
whether a page holds code that nothing executes. A later MyST may record a page
differently, and these files are the only place where that shows.

The facts the files record, measured with jupyter-book 2.1.7 (myst-cli 1.11.0):

* a ``{code-cell}`` becomes a ``block`` node of kind ``notebook-code`` wherever it is
  written (directly, from an ``{include}``, in a blockquote, a list item, an admonition or
  a tab item), and an ``{eval}`` role becomes an ``inlineExpression`` node;
* a fenced block that only shows the syntax of a ``{code-cell}`` gives no such node;
* ``execute: {skip: true}`` in a page's front matter is recorded in the page file as
  ``frontmatter.execute.skip`` equal to true, and so are the deprecated
  ``skip_execution: true`` and the string ``'true'``.

Usage, from the repository root, after the notebook command has written the executed
smoke notebook::

    uv run python tests/fixtures/built_pages/make_page_files.py --scratch <folder outside the repository>

The script writes the scratch project (the executed smoke notebook and the markdown pages
below), builds it once with ``jupyter book build --html`` without ``--execute``, and
copies each page file, unchanged, to the output folder as ``<source stem>.json``. The
theme is copied from ``_build/executed/_build/templates`` when that folder exists, so that
this build does not download it.
MyST writes random node keys and a content hash into each page file, so a new run
differs from the committed files in those values only; the tests do not read them.

Examples
--------
>>> main(["--scratch", "/tmp/page-files"])  # doctest: +SKIP
0
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

logger = logging.getLogger("make_page_files")

REPO = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
DEFAULT_NOTEBOOK = (
    REPO / "_build" / "executed" / "book" / "smoke" / "smoke_cedalion.ipynb"
)
THEME = (
    "https://github.com/myst-templates/book-theme/archive/"
    "08951401cf4aa6710f040fe34dcf0c1351a08639.zip"
)
CODE_CELL = '```{code-cell} python\nprint("cell")\n```\n'

# stem -> text of the page. The notebook is copied, not written.
PAGES: dict[str, str] = {
    "no_code": "# Prose only\n\nA page with a title and a sentence.\n",
    "plain_fence": (
        "# Plain fence\n\n```python\nprint('shown, not run')\n```\n\n"
        "```{code-block} python\nprint('code-block directive')\n```\n"
    ),
    "code_cell": "# Code cell\n\n" + CODE_CELL,
    "eval_only": "# Eval role\n\nThe value is {eval}`1 + 1`.\n",
    "include_case": "# Include\n\n```{include} included_cell.md\n```\n",
    "blockquote": "# Blockquote\n\n> "
    + CODE_CELL.replace("\n", "\n> ").rstrip("> ")
    + "\n",
    "list_item": (
        "# List item\n\n- item\n\n  ```{code-cell} python\n  print('cell')\n  ```\n"
    ),
    "admonition": "# Admonition\n\n:::{note}\n" + CODE_CELL + ":::\n",
    "tab_item": (
        "# Tab\n\n::::{tab-set}\n:::{tab-item} A\n" + CODE_CELL + ":::\n::::\n"
    ),
    "skipped": "---\nexecute:\n  skip: true\n---\n# Skipped\n\n" + CODE_CELL,
    "legacy_skip": "---\nskip_execution: true\n---\n# Legacy skip\n\n" + CODE_CELL,
    "string_skip": "---\nexecute:\n  skip: 'true'\n---\n# String skip\n\n" + CODE_CELL,
    "literal_block": ("# Literal block\n\n````markdown\n" + CODE_CELL + "````\n"),
}
# Pulled in by include_case; not in the table of contents.
INCLUDED = {"included_cell": CODE_CELL}


def write_project(scratch: Path, notebook: Path) -> dict[str, str]:
    """Write the scratch project; return the source file of each toc entry by stem.

    Parameters
    ----------
    scratch : Path
        Folder outside the repository; ``<scratch>/project`` is (re)created.
    notebook : Path
        Executed smoke notebook, copied to ``book/notebook_smoke.ipynb``.

    Returns
    -------
    dict
        Stem to the path of the source relative to the project, in table-of-contents order.
    """
    project = scratch / "project"
    if project.exists():
        shutil.rmtree(project)
    (project / "book").mkdir(parents=True)
    sources: dict[str, str] = {}
    shutil.copy2(notebook, project / "book" / "notebook_smoke.ipynb")
    sources["notebook_smoke"] = "book/notebook_smoke.ipynb"
    for stem, text in PAGES.items():
        (project / "book" / f"{stem}.md").write_text(text, encoding="utf-8")
        sources[stem] = f"book/{stem}.md"
    for stem, text in INCLUDED.items():
        (project / "book" / f"{stem}.md").write_text(text, encoding="utf-8")
    document = {
        "version": 1,
        "project": {
            "id": "22222222-2222-2222-2222-222222222222",
            "title": "built page files",
            "toc": [{"file": path} for path in sources.values()],
        },
        "site": {"template": THEME},
    }
    (project / "myst.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    theme = REPO / "_build" / "executed" / "_build" / "templates"
    if theme.is_dir():
        shutil.copytree(theme, project / "_build" / "templates")
    return sources


def build(project: Path) -> int:
    """Run ``jupyter book build --html`` in the project; return its exit code."""
    command = [sys.executable, "-m", "jupyter_book", "build", "--html"]
    logger.info("Building %s with: %s", project, " ".join(command))
    code = subprocess.run(command, cwd=project, check=False).returncode
    logger.info("jupyter book build exit code: %d", code)
    return code


def export(project: Path, sources: dict[str, str], output: Path) -> list[Path]:
    """Copy each page file to ``output`` as ``<source stem>.json``, unchanged.

    The page file of a source is found by the ``location`` MyST recorded in it.
    """
    by_location = {f"/{path}": stem for stem, path in sources.items()}
    content = project / "_build" / "site" / "content"
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for page_file in sorted(content.rglob("*.json")):
        location = json.loads(page_file.read_text(encoding="utf-8"))["location"]
        stem = by_location.get(location)
        if stem is None:
            continue
        destination = output / f"{stem}.json"
        shutil.copy2(page_file, destination)
        logger.info("Copied %s to %s", page_file.name, destination)
        written.append(destination)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; return 0 when every page file was written."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if REPO in args.scratch.resolve().parents:
        parser.error("--scratch must be outside the repository tree")
    if not args.notebook.is_file():
        parser.error(f"run the notebook command first: {args.notebook} does not exist")
    sources = write_project(args.scratch, args.notebook)
    project = args.scratch / "project"
    code = build(project)
    if code != 0:
        return code
    written = export(project, sources, args.output)
    if len(written) != len(sources):
        logger.error(
            "Expected %d page files, wrote %d: some source has no page file.",
            len(sources),
            len(written),
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
