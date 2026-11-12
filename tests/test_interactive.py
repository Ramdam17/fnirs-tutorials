"""Tests of the interactive illustrations under ``interactive/``.

Two things are asserted about the real files. Every illustration that a book page embeds
exists, so a typo in a frame source fails here before it fails in a browser. And no
illustration loads anything from another host: a page of the book must work offline, on
Colab and on the deployed site alike, and an illustration that fetched a script from a
content-delivery network would break the day that address changes.

The detector of the second assertion has its own tests, one per kind of load, because an
assertion that a file holds no external load only means something when the detector is
known to find one of each kind.

When to use: run with the rest of the suite,
``uv run python -m pytest tests/test_interactive.py``.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERACTIVE_DIR = REPO_ROOT / "interactive"
SOURCE_PREFIX = "/interactive/"

# The page files are read through the one reader of the table of contents that the build
# script uses, not through a second parser of myst.yml.
_spec = importlib.util.spec_from_file_location(
    "build_book", REPO_ROOT / "scripts" / "build_book.py"
)
assert _spec is not None and _spec.loader is not None
build_book = importlib.util.module_from_spec(_spec)
sys.modules["build_book"] = build_book
_spec.loader.exec_module(build_book)

# Strings that look like an address and are only identifiers: an XML namespace is compared
# with a string, never requested.
NAMESPACES = {
    "http://www.w3.org/2000/svg",
    "http://www.w3.org/1999/xlink",
    "http://www.w3.org/1999/xhtml",
}

_URL = r"(?:https?:|wss?:|ftp:)?//[^\s\"'`<>)\\]+"
_EXTERNAL = re.compile(r"^(?:https?:|wss?:|ftp:)?//", re.IGNORECASE)

# Each pattern captures the address a kind of load asks for. They are tried on the text of
# a file with its comments left in: an address in a comment is read as an address, which
# errs on the side of refusing.
_LOADS: dict[str, re.Pattern[str]] = {
    "script": re.compile(r"<script\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)", re.IGNORECASE),
    "stylesheet": re.compile(
        r"<link\b[^>]*?\bhref\s*=\s*[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE
    ),
    "image": re.compile(
        r"<(?:img|image|source|video|audio|embed|track)\b[^>]*?"
        r"\b(?:src|href|xlink:href|poster)\s*=\s*[\"']([^\"']+)",
        re.IGNORECASE,
    ),
    "srcset": re.compile(r"\bsrcset\s*=\s*[\"']([^\"']+)", re.IGNORECASE),
    "frame": re.compile(
        r"<(?:iframe|object)\b[^>]*?\b(?:src|data)\s*=\s*[\"']([^\"']+)", re.IGNORECASE
    ),
    "css url": re.compile(r"url\(\s*[\"']?([^\"')]+)", re.IGNORECASE),
    "css import": re.compile(r"@import\s+[\"']([^\"']+)", re.IGNORECASE),
    "fetch": re.compile(r"\bfetch\(\s*[\"'`]([^\"'`]+)"),
    "dynamic import": re.compile(r"\bimport\(\s*[\"'`]([^\"'`]+)"),
    "static import": re.compile(r"\bimport\b[^;\n]*?\bfrom\s*[\"']([^\"']+)"),
    "import scripts": re.compile(r"\bimportScripts\(\s*[\"'`]([^\"'`]+)"),
    "worker": re.compile(r"\bnew\s+(?:Shared)?Worker\(\s*[\"'`]([^\"'`]+)"),
    "socket": re.compile(r"\bnew\s+(?:WebSocket|EventSource)\(\s*[\"'`]([^\"'`]+)"),
    "request": re.compile(r"\.open\(\s*[\"'][A-Z]+[\"']\s*,\s*[\"'`]([^\"'`]+)"),
    "beacon": re.compile(r"\bsendBeacon\(\s*[\"'`]([^\"'`]+)"),
}
# A link the reader follows is not a load: it is requested only when clicked.
_ANCHOR = re.compile(r"<a\b[^>]*>", re.IGNORECASE)
_ADDRESS_LITERAL = re.compile(rf"[\"'`(]\s*({_URL})", re.IGNORECASE)


def external_loads(text: str) -> list[tuple[str, str]]:
    """Return the loads of a file that go to another host.

    Parameters
    ----------
    text : str
        Source of an HTML file, with its styles and scripts inline.

    Returns
    -------
    list of (str, str)
        The kind of load and the address it asks for, for each load whose address names
        another host (a scheme, or two slashes). A relative address, a ``data:`` or
        ``blob:`` address, a fragment, a link of an ``<a>`` tag and an XML namespace are
        not external loads. Besides the kinds of load that are named, any quoted address
        literal is reported as ``"address"``, because a script can hold an address in a
        variable and pass it to a function this detector does not know.
    """
    without_links = _ANCHOR.sub("", text)
    found: list[tuple[str, str]] = []
    for kind, pattern in _LOADS.items():
        for match in pattern.finditer(without_links):
            if kind == "stylesheet" and "stylesheet" not in match.group(0).lower():
                continue
            values = match.group(1)
            # A srcset holds several addresses, each followed by a descriptor.
            candidates = (
                [part.strip().split()[0] for part in values.split(",") if part.strip()]
                if kind == "srcset"
                else [values.strip()]
            )
            for address in candidates:
                if _EXTERNAL.match(address) and address not in NAMESPACES:
                    found.append((kind, address))
    for match in _ADDRESS_LITERAL.finditer(without_links):
        address = match.group(1)
        known = {found_address for _, found_address in found}
        if address not in NAMESPACES and address not in known:
            found.append(("address", address))
    return found


# --------------------------------------------------------------------------------------
# The detector, on one example of each kind of load
# --------------------------------------------------------------------------------------

CDN = "https://cdn.example.org/lib.js"


@pytest.mark.parametrize(
    ("kind", "snippet"),
    [
        ("script", f'<script src="{CDN}"></script>'),
        ("script", f"<script src='{CDN}' defer></script>"),
        ("stylesheet", f'<link rel="stylesheet" href="{CDN}">'),
        ("stylesheet", f'<link href="{CDN}" rel="stylesheet">'),
        ("image", f'<img alt="" src="{CDN}">'),
        ("image", f'<image href="{CDN}"/>'),
        ("image", f'<video poster="{CDN}"></video>'),
        ("srcset", f'<img srcset="local.png 1x, {CDN} 2x">'),
        ("frame", f'<iframe src="{CDN}"></iframe>'),
        ("css url", f"<style>a {{ background: url({CDN}); }}</style>"),
        ("css url", f"<style>@font-face {{ src: url('{CDN}'); }}</style>"),
        ("css import", f'<style>@import "{CDN}";</style>'),
        ("fetch", f'<script>fetch("{CDN}");</script>'),
        ("dynamic import", f'<script>import("{CDN}");</script>'),
        ("static import", f'<script type="module">import x from "{CDN}";</script>'),
        ("import scripts", f'<script>importScripts("{CDN}");</script>'),
        ("worker", f'<script>new Worker("{CDN}");</script>'),
        ("socket", '<script>new WebSocket("wss://example.org/s");</script>'),
        ("socket", f'<script>new EventSource("{CDN}");</script>'),
        ("request", f'<script>x.open("GET", "{CDN}");</script>'),
        ("beacon", f'<script>navigator.sendBeacon("{CDN}");</script>'),
        ("script", '<script src="//cdn.example.org/lib.js"></script>'),
    ],
)
def test_the_detector_finds_each_kind_of_external_load(kind: str, snippet: str) -> None:
    kinds = [found_kind for found_kind, _ in external_loads(snippet)]
    assert kind in kinds


def test_the_detector_finds_an_address_kept_in_a_variable() -> None:
    snippet = f'<script>const u = "{CDN}"; load(u);</script>'
    assert external_loads(snippet) == [("address", CDN)]


@pytest.mark.parametrize(
    "snippet",
    [
        '<script src="local.js"></script>',
        '<script src="/interactive/local.js"></script>',
        '<link rel="stylesheet" href="local.css">',
        '<img src="data:image/png;base64,AAAA">',
        '<img srcset="a.png 1x, b.png 2x">',
        "<style>a { background: url(#gradient); }</style>",
        "<style>a { background: url(data:image/svg+xml;base64,AAAA); }</style>",
        '<script>fetch("data.json");</script>',
        '<script>const SVG_NS = "http://www.w3.org/2000/svg";</script>',
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"></svg>',
        '<script>document.createElementNS("http://www.w3.org/2000/svg", "g");</script>',
        '<a href="https://example.org/paper">A paper</a>',
        "<script>// see https://example.org is not a quoted address\n</script>",
    ],
)
def test_the_detector_accepts_what_is_not_an_external_load(snippet: str) -> None:
    assert external_loads(snippet) == []


# --------------------------------------------------------------------------------------
# The real illustrations
# --------------------------------------------------------------------------------------


def illustrations() -> list[Path]:
    """The HTML files of the illustrations, in sorted order."""
    return sorted(INTERACTIVE_DIR.rglob("*.html"))


def page_text(path: Path) -> str:
    """Text of a book page: a markdown file, or the markdown cells of a notebook."""
    if path.suffix.lower() == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        return "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "markdown"
        )
    return path.read_text(encoding="utf-8")


# The {iframe} directive in its colon fence and in its backtick fence.
_IFRAME_DIRECTIVE = re.compile(
    r"^\s*(?::{3,}|`{3,})\s*\{iframe\}\s+(\S+)", re.MULTILINE
)


def embedded_sources() -> list[tuple[str, str]]:
    """Return ``(page, source)`` for each frame under ``/interactive/`` that a page has."""
    toc_files, _ = build_book.read_toc_files(REPO_ROOT / "myst.yml")
    pairs = []
    for name in toc_files:
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        for source in _IFRAME_DIRECTIVE.findall(page_text(path)):
            if source.startswith(SOURCE_PREFIX):
                pairs.append((name, source))
    return pairs


def test_the_directive_pattern_reads_both_fences() -> None:
    text = (
        ":::{iframe} /interactive/a.html\n:::\n\n```{iframe} /interactive/b.html\n```\n"
    )
    assert _IFRAME_DIRECTIVE.findall(text) == [
        "/interactive/a.html",
        "/interactive/b.html",
    ]


def test_a_page_of_the_book_embeds_at_least_one_illustration() -> None:
    assert embedded_sources(), "no page of the table of contents embeds an illustration"


@pytest.mark.parametrize(("page", "source"), embedded_sources())
def test_every_illustration_a_page_embeds_exists(page: str, source: str) -> None:
    relative = source[len(SOURCE_PREFIX) :]
    target = (INTERACTIVE_DIR / relative).resolve()
    assert target.is_relative_to(INTERACTIVE_DIR.resolve()), (
        f"{page}: {source} leaves the folder"
    )
    assert target.is_file(), f"{page} embeds {source}, but {target} does not exist"


@pytest.mark.parametrize("path", illustrations(), ids=lambda p: p.name)
def test_no_illustration_loads_a_resource_from_another_host(path: Path) -> None:
    loads = external_loads(path.read_text(encoding="utf-8"))
    assert loads == [], f"{path.name} loads from another host: {loads}"


def test_there_is_an_illustration_to_check() -> None:
    assert illustrations()
