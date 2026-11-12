"""Tests of the interactive illustrations under ``interactive/``.

Two things are asserted about the real files. Every illustration that a book page embeds
exists, so a typo in a frame source fails here before it fails in a browser. And no
illustration loads anything from another host: a page of the book must work offline, on
Colab and on the deployed site alike, and an illustration that fetched a script from a
content-delivery network would break the day that address changes.

A third assertion covers the one library the illustrations use, D3, which is copied into
``interactive/vendor/``. The file is pinned by its size and its SHA-256, so that a changed
or replaced library fails the suite, and a page may load a script only by a relative
address that resolves to a file of that folder. The vendored file is not scanned by the
detector below. It is third-party code that holds namespace strings and its own address in
a comment, and ``fetch`` calls that run only if a page passes them an address; the scan of
the pages covers that case, since a page that passed an external address would be found
there.

The detector of the second assertion has its own tests, one per kind of load, because an
assertion that a file holds no external load only means something when the detector is
known to find one of each kind.

When to use: run with the rest of the suite,
``uv run python -m pytest tests/test_interactive.py``.
"""

from __future__ import annotations

import hashlib
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


# --------------------------------------------------------------------------------------
# The vendored library and the scripts a page may load
# --------------------------------------------------------------------------------------

VENDOR_DIR = INTERACTIVE_DIR / "vendor"
D3_VERSION = "7.9.2"
# dist/d3.min.js of the npm package d3 at the version above.
D3_SIZE_BYTES = 280_622
D3_SHA256 = "912111098ea99503a6f4967361d2bbdc9dbf58cf9c678c8e27e5b02d1a07f0c1"


def sha256_of(path: Path) -> str:
    """Return the SHA-256 of a file, as lower-case hexadecimal."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def library_problem(path: Path, size: int, digest: str) -> str | None:
    """Say how a file differs from the recorded size and hash, or return None."""
    if not path.is_file():
        return f"{path} does not exist"
    if path.stat().st_size != size:
        return f"{path.name} has {path.stat().st_size} bytes, expected {size}"
    if sha256_of(path) != digest:
        return f"{path.name} has another SHA-256 than the recorded {digest}"
    return None


def test_the_vendored_d3_is_the_recorded_version() -> None:
    problem = library_problem(VENDOR_DIR / "d3.min.js", D3_SIZE_BYTES, D3_SHA256)
    assert problem is None, f"D3 {D3_VERSION}: {problem}"
    header = (VENDOR_DIR / "d3.min.js").read_text(encoding="utf-8").splitlines()[0]
    assert f"v{D3_VERSION}" in header


def test_the_vendored_d3_comes_with_its_licence() -> None:
    licence = VENDOR_DIR / "LICENSE"
    assert licence.is_file()
    assert "Mike Bostock" in licence.read_text(encoding="utf-8")


def test_a_changed_or_missing_library_is_reported(tmp_path: Path) -> None:
    real = (VENDOR_DIR / "d3.min.js").read_bytes()
    same_size = tmp_path / "same_size.js"
    same_size.write_bytes(b"x" + real[1:])
    shorter = tmp_path / "shorter.js"
    shorter.write_bytes(real[:-1])
    for path, fragment in ((same_size, "SHA-256"), (shorter, "bytes")):
        problem = library_problem(path, D3_SIZE_BYTES, D3_SHA256)
        assert problem is not None and fragment in problem
    missing = library_problem(tmp_path / "absent.js", D3_SIZE_BYTES, D3_SHA256)
    assert missing is not None and "does not exist" in missing


def script_problem(page: Path, source: str, vendor_dir: Path) -> str | None:
    """Say why a ``<script src>`` is not allowed, or return None.

    A script is allowed when its source is a relative address that resolves, from the
    folder of the page, to an existing file under ``vendor_dir``.
    """
    if _EXTERNAL.match(source) or re.match(
        r"^[a-z][a-z0-9+.-]*:", source, re.IGNORECASE
    ):
        return "it has a scheme or names another host"
    if source.startswith("/"):
        return "it is an absolute path"
    target = (page.parent / source).resolve()
    if not target.is_relative_to(vendor_dir.resolve()):
        return "it does not resolve to a file under vendor/"
    if not target.is_file():
        return "the file does not exist"
    return None


def script_sources(text: str) -> list[str]:
    """Return the ``src`` of every ``<script>`` tag of a page."""
    return _LOADS["script"].findall(text)


@pytest.fixture
def fake_site(tmp_path: Path) -> tuple[Path, Path]:
    """An ``interactive`` folder with a page, a vendored file and a stray file."""
    folder = tmp_path / "interactive"
    (folder / "vendor").mkdir(parents=True)
    (folder / "vendor" / "lib.js").write_text("x", encoding="utf-8")
    (folder / "other").mkdir()
    (folder / "other" / "lib.js").write_text("x", encoding="utf-8")
    (folder / "secret.js").write_text("x", encoding="utf-8")
    return folder / "page.html", folder / "vendor"


def test_a_relative_script_under_vendor_is_allowed(
    fake_site: tuple[Path, Path],
) -> None:
    page, vendor = fake_site
    assert script_problem(page, "vendor/lib.js", vendor) is None
    assert script_problem(page, "./vendor/lib.js", vendor) is None


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("/interactive/vendor/lib.js", "absolute"),
        ("https://cdn.example.org/lib.js", "scheme"),
        ("//cdn.example.org/lib.js", "scheme"),
        ("data:text/javascript,1", "scheme"),
        ("blob:abc", "scheme"),
        ("other/lib.js", "under vendor/"),
        ("secret.js", "under vendor/"),
        ("vendor/../secret.js", "under vendor/"),
        ("../interactive/secret.js", "under vendor/"),
        ("../../outside.js", "under vendor/"),
        ("vendor/missing.js", "does not exist"),
        ("vendor", "does not exist"),
    ],
)
def test_any_other_script_is_refused_with_its_reason(
    fake_site: tuple[Path, Path], source: str, fragment: str
) -> None:
    page, vendor = fake_site
    problem = script_problem(page, source, vendor)
    assert problem is not None and fragment in problem


def test_the_script_tags_of_a_page_are_all_read() -> None:
    text = (
        '<script src="vendor/a.js"></script><p>x</p>'
        "<script defer src='vendor/b.js'></script><script>inline()</script>"
    )
    assert script_sources(text) == ["vendor/a.js", "vendor/b.js"]


@pytest.mark.parametrize("path", illustrations(), ids=lambda p: p.name)
def test_an_illustration_loads_scripts_only_from_the_vendor_folder(path: Path) -> None:
    for source in script_sources(path.read_text(encoding="utf-8")):
        problem = script_problem(path, source, VENDOR_DIR)
        assert problem is None, f"{path.name} loads {source!r}: {problem}"
