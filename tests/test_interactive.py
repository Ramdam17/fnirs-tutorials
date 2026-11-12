"""Tests of the interactive illustrations under ``interactive/``.

Two things are asserted about the real files. Every illustration that a book page embeds
exists, so a typo in a frame source fails here before it fails in a browser. And no
illustration loads anything from another host: a page of the book must work offline, on
Colab and on the deployed site alike, and an illustration that fetched a script from a
content-delivery network would break the day that address changes.

A third assertion covers the libraries the illustrations use, today D3 only, which are
copied into ``interactive/vendor/``. They are an allow-list (``VENDORED``): each file is
pinned by its version, its size and its SHA-256, so that a changed or replaced library
fails the suite; a page may load a script only by a relative address that resolves to a
file of the list; a file of that folder that is neither in the list nor a licence fails a
test; and the version, size and hash that ``COMPAT.md`` and ``interactive/README.md`` give
must equal the constants. The vendored file is not scanned by the detector below. It is
third-party code that holds namespace strings and its own address in a comment, and
``fetch`` calls that run only if a page passes them an address; the scan of the pages
covers that case, since a page that passed an external address would be found there.

The detector of the second assertion has its own tests, one per kind of load, because an
assertion that a file holds no external load only means something when the detector is
known to find one of each kind.

What a static scan cannot see: the detector reads the text of a file. An address that is
assembled at run time (pieces of text joined, a value read from data, a name built from
user input) is not written in the file, so it is out of its reach, and so is a script
that a vendored library loads by itself. A page that builds an address that way has to be
reviewed by hand.

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
from typing import NamedTuple

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

# The value of an attribute, quoted or not: an optional opening quote, then the address up
# to a space, a quote or the end of the tag.
_VALUE = r"""\s*=\s*["']?\s*([^\s"'>]+)"""

# Each pattern captures the address a kind of load asks for. They are tried on the text of
# a file with its comments left in: an address in a comment is read as an address, which
# errs on the side of refusing.
_LOADS: dict[str, re.Pattern[str]] = {
    "script": re.compile(rf"<script\b[^>]*?\bsrc{_VALUE}", re.IGNORECASE),
    "stylesheet": re.compile(rf"<link\b[^>]*?\bhref{_VALUE}[^>]*>", re.IGNORECASE),
    "image": re.compile(
        r"<(?:img|image|source|video|audio|embed|track)\b[^>]*?"
        rf"\b(?:src|href|xlink:href|poster){_VALUE}",
        re.IGNORECASE,
    ),
    "srcset": re.compile(r"\bsrcset\s*=\s*[\"']?([^\"'>]+)", re.IGNORECASE),
    "frame": re.compile(
        rf"<(?:iframe|object)\b[^>]*?\b(?:src|data){_VALUE}", re.IGNORECASE
    ),
    "base": re.compile(r"(<base\b[^>]*>)", re.IGNORECASE),
    "form": re.compile(
        rf"<(?:form|button|input)\b[^>]*?\b(?:action|formaction){_VALUE}", re.IGNORECASE
    ),
    "meta refresh": re.compile(
        r"(?=<meta\b[^>]*\bhttp-equiv\s*=\s*[\"']?refresh)<meta\b[^>]*?"
        r"\burl\s*=\s*[\"']?([^\"'>\s;]+)",
        re.IGNORECASE,
    ),
    "css url": re.compile(r"url\(\s*[\"']?([^\"')]+)", re.IGNORECASE),
    "css import": re.compile(r"@import\s+[\"']([^\"']+)", re.IGNORECASE),
    "fetch": re.compile(r"\bfetch\(\s*[\"'`]([^\"'`]+)", re.IGNORECASE),
    "dynamic import": re.compile(r"\bimport\(\s*[\"'`]([^\"'`]+)"),
    "static import": re.compile(r"\bimport\b[^;\n]*?\bfrom\s*[\"']([^\"']+)"),
    "import scripts": re.compile(r"\bimportScripts\(\s*[\"'`]([^\"'`]+)"),
    "worker": re.compile(r"\bnew\s+(?:Shared)?Worker\(\s*[\"'`]([^\"'`]+)"),
    "socket": re.compile(r"\bnew\s+(?:WebSocket|EventSource)\(\s*[\"'`]([^\"'`]+)"),
    "request": re.compile(r"\.open\(\s*[\"'][A-Z]+[\"']\s*,\s*[\"'`]([^\"'`]+)"),
    "beacon": re.compile(r"\bsendBeacon\(\s*[\"'`]([^\"'`]+)"),
}
# Kinds that are reported whatever their address, since the tag itself changes what the
# page loads: a <base> moves every relative address, including that of the vendored script.
_ALWAYS_REPORTED = {"base"}
# A link the reader follows is not a load: it is requested only when clicked. Only the
# href of an anchor is left out of the scan, so that what else the tag holds (a ping, an
# onclick) is still read.
_ANCHOR = re.compile(r"<a\b[^>]*>", re.IGNORECASE)
_HREF = re.compile(r"""\bhref\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]*)""", re.IGNORECASE)
# Any address after a quote, a backtick, a parenthesis or an equal sign.
_ADDRESS_LITERAL = re.compile(rf"[\"'`(=]\s*({_URL})", re.IGNORECASE)


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
    without_links = _ANCHOR.sub(lambda tag: _HREF.sub("", tag.group(0)), text)
    found: list[tuple[str, str]] = []
    for kind, pattern in _LOADS.items():
        for match in pattern.finditer(without_links):
            if kind == "stylesheet" and "stylesheet" not in match.group(0).lower():
                continue
            values = match.group(1)
            if kind in _ALWAYS_REPORTED:
                found.append((kind, values))
                continue
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
        # Unquoted values, for each kind of load.
        ("script", f"<script src={CDN}></script>"),
        ("script", f"<script defer src={CDN} async></script>"),
        ("stylesheet", f"<link rel=stylesheet href={CDN}>"),
        ("image", f"<img alt=x src={CDN}>"),
        ("image", f"<video poster={CDN}></video>"),
        ("srcset", f"<img srcset={CDN}>"),
        ("frame", f"<iframe src={CDN}></iframe>"),
        # Other tags and forms that reach another host.
        ("meta refresh", f'<meta http-equiv="refresh" content="0; url={CDN}">'),
        ("meta refresh", f"<meta content='5;URL={CDN}' http-equiv=refresh>"),
        ("meta refresh", f"<meta http-equiv=refresh content=0;url={CDN}>"),
        ("base", f'<base href="{CDN}">'),
        ("base", '<base href="other/">'),
        ("form", f'<form action="{CDN}" method="post"></form>'),
        ("form", f"<form method=post action={CDN}></form>"),
        ("form", f'<button formaction="{CDN}">go</button>'),
        (
            "css url",
            f"<style>@font-face {{ font-family: f; src: url({CDN}); }}</style>",
        ),
        (
            "css url",
            "<style>@font-face { src: url(//fonts.example.org/f.woff2); }</style>",
        ),
        ("script", '<SCRIPT SRC="HTTPS://CDN.EXAMPLE.ORG/LIB.JS"></SCRIPT>'),
        ("fetch", '<script>fetch("HTTPS://example.org/x");</script>'),
        ("address", f"<script>const u = `{CDN}`;</script>"),
        ("address", "<script>const u = `https://example.org/${name}.json`;</script>"),
        # What follows `<a` in the same tag is scanned, only the href is a link.
        ("address", f'<a href="#" onclick="location=\'{CDN}\'">x</a>'),
        ("address", f'<a href="#" ping="{CDN}">x</a>'),
        ("address", f"<a href=# ping={CDN}>x</a>"),
        ("fetch", f'<a onclick="fetch(`{CDN}`)" href="#">x</a>'),
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
        "<a href=https://example.org/paper>A paper</a>",
        "<a class='x' href='https://example.org/paper' target=_blank>A paper</a>",
        '<meta charset="utf-8">',
        '<meta http-equiv="refresh" content="0; url=other.html">',
        '<form action="submit.html"></form>',
        "<script src=vendor/d3.min.js></script>",
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


def embedded_problem(source: str, interactive_dir: Path) -> str | None:
    """Say why the target of an embedded source is not a file of ``interactive_dir``.

    The name must be the one the folder lists, letter for letter: the Mac ignores case
    in its default file system and the Linux runner and the host do not, so a source
    written ``Banana.html`` for ``banana.html`` would pass here and be a 404 once deployed.
    """
    relative = source[len(SOURCE_PREFIX) :]
    target = (interactive_dir / relative).resolve()
    if not target.is_relative_to(interactive_dir.resolve()):
        return "it leaves the folder"
    if not build_book._has_exact_names(interactive_dir, relative.split("/")):
        return "no file has exactly this name"
    if not target.is_file():
        return "the target is not a file"
    return None


@pytest.mark.parametrize(("page", "source"), embedded_sources())
def test_every_illustration_a_page_embeds_exists(page: str, source: str) -> None:
    problem = embedded_problem(source, INTERACTIVE_DIR)
    assert problem is None, f"{page} embeds {source}: {problem}"


@pytest.mark.usefixtures("case_blind_disk")
def test_an_embedded_source_must_match_the_name_of_the_file_letter_for_letter(
    tmp_path: Path,
) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "banana.html").write_text("x", encoding="utf-8")
    assert embedded_problem("/interactive/sub/banana.html", tmp_path) is None
    for wrong in ("/interactive/sub/Banana.html", "/interactive/SUB/banana.html"):
        problem = embedded_problem(wrong, tmp_path)
        assert problem is not None and "exactly" in problem, wrong
    assert embedded_problem("/interactive/sub/absent.html", tmp_path) is not None


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


class Vendored(NamedTuple):
    """The pin of a vendored file: what it is, and what it must weigh and hash to."""

    version: str
    size: int
    sha256: str


# The allow-list. A page may load a script only if it resolves to a file named here, and a
# file under vendor/ that is neither named here nor a licence fails a test, so a second
# library cannot arrive without a pin. d3.min.js is dist/d3.min.js of the npm package d3.
VENDORED: dict[str, Vendored] = {
    "d3.min.js": Vendored(
        "7.9.2",
        280_622,
        "912111098ea99503a6f4967361d2bbdc9dbf58cf9c678c8e27e5b02d1a07f0c1",
    ),
}
LICENCE_NAMES = {"LICENSE"}
D3 = VENDORED["d3.min.js"]
D3_VERSION, D3_SIZE_BYTES, D3_SHA256 = D3


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


def test_every_vendored_file_is_the_recorded_version() -> None:
    for name, pin in VENDORED.items():
        problem = library_problem(VENDOR_DIR / name, pin.size, pin.sha256)
        assert problem is None, f"{name} {pin.version}: {problem}"


def test_the_vendored_d3_header_names_the_recorded_version() -> None:
    header = (VENDOR_DIR / "d3.min.js").read_text(encoding="utf-8").splitlines()[0]
    assert f"v{D3_VERSION}" in header


def vendor_files(vendor_dir: Path = VENDOR_DIR) -> list[str]:
    """Paths under ``vendor_dir`` relative to it, as published; hidden files are left out.

    Only the parts below ``vendor_dir`` are tested for a leading dot: the folders above it
    belong to wherever the repository is checked out, for example a hidden worktree folder.
    """
    return sorted(
        p.relative_to(vendor_dir).as_posix()
        for p in vendor_dir.rglob("*")
        if p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(vendor_dir).parts)
    )


def unpinned(files: list[str]) -> list[str]:
    """The files that are neither in the allow-list nor a licence."""
    return [f for f in files if f not in VENDORED and Path(f).name not in LICENCE_NAMES]


def test_a_file_under_vendor_is_in_the_allow_list_or_a_licence() -> None:
    assert unpinned(vendor_files()) == []


def test_the_vendor_listing_does_not_depend_on_the_folders_above_it(
    tmp_path: Path,
) -> None:
    """A checkout under a hidden folder must still list its vendored files."""
    vendor = tmp_path / ".worktrees" / "agent" / "interactive" / "vendor"
    (vendor / "sub").mkdir(parents=True)
    for name in ("d3.min.js", "lodash.min.js", "LICENSE", ".DS_Store", "sub/x.js"):
        (vendor / name).write_text("x", encoding="utf-8")
    assert vendor_files(vendor) == ["LICENSE", "d3.min.js", "lodash.min.js", "sub/x.js"]
    assert unpinned(vendor_files(vendor)) == ["lodash.min.js", "sub/x.js"]


def test_a_second_library_without_a_pin_is_found() -> None:
    assert unpinned(["d3.min.js", "LICENSE", "lodash.min.js", "sub/x.js"]) == [
        "lodash.min.js",
        "sub/x.js",
    ]


def test_the_allow_list_holds_d3_only() -> None:
    assert list(VENDORED) == ["d3.min.js"]


def _pins_in(text: str) -> tuple[str, int, str]:
    """Read the version, the size in bytes and the SHA-256 that a sentence of a document gives."""
    version = re.search(r"\bD3\b[^.\d]*(\d+\.\d+\.\d+)", text)
    size = re.search(r"([\d,]+) bytes", text)
    digest = re.search(r"\b([0-9a-f]{64})\b", text)
    assert version and size and digest, f"no version, size and SHA-256 in: {text!r}"
    return version.group(1), int(size.group(1).replace(",", "")), digest.group(1)


def _line_with(path: Path, needle: str) -> str:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if needle in ln]
    assert lines, f"{path.name} has no line with {needle!r}"
    return lines[0]


@pytest.mark.parametrize(
    ("document", "needle"),
    [
        (REPO_ROOT / "COMPAT.md", "| D3 "),
        (INTERACTIVE_DIR / "README.md", "drawn with D3"),
    ],
    ids=["COMPAT.md", "README.md"],
)
def test_the_documents_give_the_version_size_and_hash_of_the_constants(
    document: Path, needle: str
) -> None:
    assert _pins_in(_line_with(document, needle)) == tuple(D3)


def test_a_document_that_differs_from_the_constants_is_found() -> None:
    line = f"D3 9.9.9 (1 bytes, SHA-256 `{'0' * 64}`)"
    assert _pins_in(line) != tuple(D3)


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


def script_problem(
    page: Path,
    source: str,
    vendor_dir: Path,
    allowed: dict[str, Vendored] | None = None,
) -> str | None:
    """Say why a ``<script src>`` is not allowed, or return None.

    A script is allowed when its source is a relative address that resolves, from the
    folder of the page, to an existing file of ``allowed`` (by default ``VENDORED``)
    directly under ``vendor_dir``.
    """
    allowed = VENDORED if allowed is None else allowed
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
    if target.relative_to(vendor_dir.resolve()).as_posix() not in allowed:
        return "it is not a file of the allow-list of vendored files"
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


LIB = {"lib.js": Vendored("1", 1, "0" * 64)}


def test_a_relative_script_of_the_allow_list_is_allowed(
    fake_site: tuple[Path, Path],
) -> None:
    page, vendor = fake_site
    assert script_problem(page, "vendor/lib.js", vendor, LIB) is None
    assert script_problem(page, "./vendor/lib.js", vendor, LIB) is None


def test_a_script_under_vendor_that_is_not_in_the_allow_list_is_refused(
    fake_site: tuple[Path, Path],
) -> None:
    page, vendor = fake_site
    (vendor / "unpinned.js").write_text("x", encoding="utf-8")
    problem = script_problem(page, "vendor/unpinned.js", vendor, LIB)
    assert problem is not None and "allow-list" in problem


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
    problem = script_problem(page, source, vendor, LIB)
    assert problem is not None and fragment in problem


def test_the_script_tags_of_a_page_are_all_read() -> None:
    text = (
        '<script src="vendor/a.js"></script><p>x</p>'
        "<script defer src='vendor/b.js'></script><script>inline()</script>"
    )
    assert script_sources(text) == ["vendor/a.js", "vendor/b.js"]


def test_an_unquoted_script_source_is_read_and_refused(
    fake_site: tuple[Path, Path],
) -> None:
    page, vendor = fake_site
    text = (
        "<script src=https://cdn.example.org/lib.js></script>"
        "<script defer src=vendor/lib.js async></script>"
        "<SCRIPT SRC = 'other/lib.js'></SCRIPT>"
    )
    sources = script_sources(text)
    assert sources == [
        "https://cdn.example.org/lib.js",
        "vendor/lib.js",
        "other/lib.js",
    ]
    problems = [script_problem(page, source, vendor, LIB) for source in sources]
    assert problems[0] is not None and "scheme" in problems[0]
    assert problems[1] is None
    assert problems[2] is not None


@pytest.mark.parametrize("path", illustrations(), ids=lambda p: p.name)
def test_an_illustration_loads_scripts_only_from_the_vendor_folder(path: Path) -> None:
    for source in script_sources(path.read_text(encoding="utf-8")):
        problem = script_problem(path, source, VENDOR_DIR)
        assert problem is None, f"{path.name} loads {source!r}: {problem}"


# --------------------------------------------------------------------------------------
# Live regions: a step of a slider must not be announced twice
# --------------------------------------------------------------------------------------

_TAG = re.compile(
    r"<([a-z][a-z0-9-]*)\b((?:[^>\"']|\"[^\"]*\"|'[^']*')*)>", re.IGNORECASE
)


def live_regions(text: str) -> list[str]:
    """Return the tags of a page that announce a change of their content.

    A region is live when it carries ``aria-live`` with a value other than ``off``, a
    role of ``status``, ``alert`` or ``log``, or is an ``<output>`` (whose implicit role
    is ``status``) that does not say ``aria-live="off"``.
    """
    found = []
    for match in _TAG.finditer(text):
        name, attributes = match.group(1).lower(), match.group(2)
        live = re.search(r"""aria-live\s*=\s*["']?(\w+)""", attributes, re.IGNORECASE)
        role = re.search(r"""\brole\s*=\s*["']?(\w+)""", attributes, re.IGNORECASE)
        value = live.group(1).lower() if live else None
        is_live = (
            (value is not None and value != "off")
            or (
                value is None
                and role
                and role.group(1).lower() in {"status", "alert", "log"}
            )
            or (value is None and name == "output")
        )
        if is_live:
            found.append(match.group(0))
    return found


def test_the_live_region_reader_finds_each_kind() -> None:
    assert len(live_regions('<p aria-live="polite"></p>')) == 1
    assert len(live_regions("<p role=status></p>")) == 1
    assert len(live_regions('<output id="v"></output>')) == 1
    assert live_regions('<output id="v" aria-live="off"></output>') == []
    assert live_regions('<p id="readout"></p>') == []


@pytest.mark.parametrize("path", illustrations(), ids=lambda p: p.name)
def test_an_illustration_has_at_most_one_live_region(path: Path) -> None:
    regions = live_regions(path.read_text(encoding="utf-8"))
    assert len(regions) <= 1, f"{path.name} has several live regions: {regions}"


# --------------------------------------------------------------------------------------
# The accessible name of a control must not change when its value does
# --------------------------------------------------------------------------------------

_NATIVE_CONTROLS = {"input", "select", "textarea", "button"}
# The widget and composite roles of WAI-ARIA 1.2, section 5.3.2
# (https://www.w3.org/TR/wai-aria-1.2/#widget_roles); a separator is a widget only when
# it can take the focus, which the reader tests with ``tabindex``.
_WIDGET_ROLES = {
    "button",
    "checkbox",
    "gridcell",
    "link",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "option",
    "progressbar",
    "radio",
    "scrollbar",
    "searchbox",
    "separator",
    "slider",
    "spinbutton",
    "switch",
    "tab",
    "tabpanel",
    "textbox",
    "treeitem",
    "combobox",
    "grid",
    "listbox",
    "menu",
    "menubar",
    "radiogroup",
    "tablist",
    "tree",
    "treegrid",
}
# The roles that ARIA lets take their name from their content, as a button does.
_NAMED_FROM_CONTENT = {
    "button",
    "checkbox",
    "gridcell",
    "link",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "option",
    "radio",
    "switch",
    "tab",
    "treeitem",
}
_VALUE_BUTTONS = {"button", "submit", "reset"}


def _attribute(attributes: str, name: str) -> str | None:
    """Return the value of an attribute written in a tag, or None when it is absent."""
    found = re.search(
        rf"""(?<![\w-]){name}\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))""",
        attributes,
        re.IGNORECASE,
    )
    return next(g for g in found.groups() if g is not None) if found else None


def _content(text: str, tag: re.Match[str]) -> tuple[int, int]:
    """Return the span of the content of an element, up to the first closing tag of its name."""
    name = tag.group(1).lower()
    closing = re.search(rf"</{name}\s*>", text[tag.end() :], re.IGNORECASE)
    if closing is None:
        return tag.end(), tag.end()
    return tag.end(), tag.end() + closing.start()


# The labelable elements of the HTML Standard, form section ("Categories", labelable
# elements, and "The label element", labeled control): only these can be the labeled
# control of a label, through ``for`` or by being
# inside it. An <input type=hidden> is the one exception among the inputs, and it is not
# read here. A <div role="slider"> that a label points at or wraps is not named by it.
_LABELABLE = frozenset(
    {"button", "input", "meter", "output", "progress", "select", "textarea"}
)


def _static_text_problem(content: str) -> str | None:
    """Say why a content cannot be a static name, or return None."""
    if "<" in content:
        return "it holds a tag"
    if not content.strip():
        return "it holds no text"
    return None


def _name_source_problem(text: str, tag: re.Match[str]) -> str | None:
    """Say why an element that gives a name cannot be a static one, or return None.

    The element must not be an ``<output>``, which holds a value that changes at every
    step, and its content must be text without a tag.
    """
    if tag.group(1).lower() == "output":
        return "it is an <output>, whose value changes"
    start, end = _content(text, tag)
    return _static_text_problem(text[start:end])


def _wrapping_label_problem(
    text: str, control: re.Match[str], labels: list[re.Match[str]]
) -> tuple[bool, str | None]:
    """Return whether a ``<label>`` wraps the control, and why its text is not static."""
    for label in labels:
        if _attribute(label.group(2), "for") is not None:
            continue
        start, end = _content(text, label)
        if not start <= control.start() < end:
            continue
        inner = (control.start(), control.end())
        if control.group(1).lower() != "input":
            closing = re.compile(rf"</{control.group(1)}\s*>", re.IGNORECASE).search(
                text, control.end()
            )
            inner = (control.start(), closing.end() if closing else control.end())
        outside = text[start : inner[0]] + text[inner[1] : end]
        return True, _static_text_problem(outside)
    return False, None


def control_name_problems(text: str) -> list[str]:
    """Return what is wrong with the accessible names of the controls of a page.

    The controls are every ``<input>`` (except a hidden one), ``<select>``,
    ``<textarea>`` and ``<button>``, and every element whose ``role`` is a widget role of
    WAI-ARIA 1.2 (``_WIDGET_ROLES``). A control needs a name from one of: a non-empty
    ``aria-label`` or ``title``; a ``<label for="id">`` or a ``<label>`` that wraps it,
    when the control is a labelable element (``_LABELABLE``) and not a ``div`` or an
    SVG group that has a widget role;
    the elements that ``aria-labelledby`` lists; for a button, a widget role that takes
    its name from content and a button-type input, its own content or ``value``. Each
    element or label text that gives the name must be text without a tag and not an
    ``<output>``, so that the name does not change when the value does and the control is
    not named by its own value.

    This reads the HTML as written. A control that a script creates when the page runs is
    out of its reach, and so is the markup inside a script string; the content of an
    element is read up to the first closing tag of its name, so a nested element of the
    same name makes the content look like it holds a tag.
    """
    tags = list(_TAG.finditer(text))
    by_id: dict[str, re.Match[str]] = {}
    for tag in tags:
        identifier = _attribute(tag.group(2), "id")
        if identifier:
            by_id.setdefault(identifier, tag)
    labels = [tag for tag in tags if tag.group(1).lower() == "label"]
    labels_for: dict[str, list[re.Match[str]]] = {}
    for label in labels:
        target = _attribute(label.group(2), "for")
        if target:
            labels_for.setdefault(target, []).append(label)
    problems = []
    for tag in tags:
        name, attributes = tag.group(1).lower(), tag.group(2)
        role = (_attribute(attributes, "role") or "").split()[:1]
        role_name = role[0].lower() if role else ""
        is_widget = role_name in _WIDGET_ROLES and not (
            role_name == "separator" and _attribute(attributes, "tabindex") is None
        )
        if name not in _NATIVE_CONTROLS and not is_widget:
            continue
        kind = (_attribute(attributes, "type") or "").lower()
        if name == "input" and kind == "hidden":
            continue
        named, found = False, []
        if (_attribute(attributes, "aria-label") or "").strip():
            named = True
        if (_attribute(attributes, "title") or "").strip():
            named = True
        if name == "input" and kind in _VALUE_BUTTONS:
            named = named or bool((_attribute(attributes, "value") or "").strip())
        identifier = _attribute(attributes, "id")
        labelable = name in _LABELABLE
        sources = (
            list(labels_for.get(identifier, [])) if identifier and labelable else []
        )
        for reference in (_attribute(attributes, "aria-labelledby") or "").split():
            if reference in by_id:
                sources.append(by_id[reference])
            else:
                found.append(f"is named by {reference!r}, which is absent")
        for source in sources:
            reason = _name_source_problem(text, source)
            if reason:
                found.append(f"is named by {source.group(0)}: {reason}")
            else:
                named = True
        wraps, reason = (
            _wrapping_label_problem(text, tag, labels) if labelable else (False, None)
        )
        if wraps:
            if reason:
                found.append(f"is named by the label that wraps it: {reason}")
            else:
                named = True
        if not named and (name == "button" or role_name in _NAMED_FROM_CONTENT):
            start, end = _content(text, tag)
            reason = _static_text_problem(text[start:end])
            if reason:
                found.append(f"is named by its own content: {reason}")
            else:
                named = True
        if not named and not found:
            found.append("has no name")
        problems.extend(f"{tag.group(0)} {message}" for message in found)
    return problems


NAME_SOUND = {
    "label for": '<label for="a">Length</label><input id="a" type="range">',
    "label for with its output beside": (
        '<label for="a">Length</label>: <output id="v" for="a">1 s</output>'
        '<input id="a" type="range">'
    ),
    "wrapping label": "<label>Length <input type=range></label>",
    "wrapping label around a select": "<label>Pick <select><option>a</option></select></label>",
    "aria-label": '<input type="range" aria-label="Length">',
    "title": '<input type="range" title="Length">',
    "aria-labelledby on static text": (
        '<span id="n">Length</span><select aria-labelledby="n"></select>'
    ),
    "button with text": "<button>Play</button>",
    "button with an aria-label and a changing content": (
        '<button aria-label="Play"><span>0:01</span></button>'
    ),
    "button-type input with a value": '<input type="submit" value="Send">',
    "widget role with aria-label": '<g role="slider" aria-label="Length"></g>',
    "widget role named from content": '<div role="tab">Overview</div>',
    "hidden input": '<input id="a" type="hidden">',
    "role that is not a widget": '<div role="img"></div>',
    "separator that cannot take the focus": '<hr role="separator">',
    "a link without a role is not read": '<a href="#">x</a>',
}
NAME_REFUSED = {
    "no name on an input": ('<input id="a" type="range">', "has no name"),
    "no name on a textarea": ('<textarea id="t"></textarea>', "has no name"),
    "no name on a button": ("<button></button>", "holds no text"),
    "no name on a widget role": ('<g role="slider"></g>', "has no name"),
    "label for a div with a widget role": (
        '<label for="s">Length</label><div id="s" role="slider" tabindex="0"></div>',
        "has no name",
    ),
    "label wrapping a div with a widget role": (
        '<label>Length <div role="slider" tabindex="0"></div></label>',
        "has no name",
    ),
    "empty aria-label": ('<input aria-label=" " type="range">', "has no name"),
    "a focusable separator": (
        '<div role="separator" tabindex="0"></div>',
        "has no name",
    ),
    "named by its own output": (
        '<input id="a" type="range" aria-labelledby="v"><output id="v" for="a">1 s</output>',
        "<output>",
    ),
    "widget role named by its own output": (
        '<g role="slider" aria-labelledby="v"></g><output id="v">1 s</output>',
        "<output>",
    ),
    "a tag inside the label": (
        '<label for="a">Length: <output id="v">1 s</output></label><input id="a">',
        "holds a tag",
    ),
    "an output inside the wrapping label": (
        "<label>Length <output>1 s</output><input></label>",
        "holds a tag",
    ),
    "a button that holds a changing value": (
        "<button>Length <output>1 s</output></button>",
        "holds a tag",
    ),
    "an absent target": ('<input id="a" aria-labelledby="gone">', "absent"),
    "an empty label": ('<label for="a"></label><input id="a">', "holds no text"),
}


@pytest.mark.parametrize("case", sorted(NAME_SOUND))
def test_the_name_reader_accepts_each_sound_source(case: str) -> None:
    assert control_name_problems(NAME_SOUND[case]) == []


@pytest.mark.parametrize("case", sorted(NAME_REFUSED))
def test_the_name_reader_refuses_each_unsound_name(case: str) -> None:
    html, reason = NAME_REFUSED[case]
    problems = control_name_problems(html)
    assert len(problems) == 1, problems
    assert reason in problems[0]


@pytest.mark.parametrize("path", illustrations(), ids=lambda p: p.name)
def test_an_illustration_names_its_controls_with_static_text(path: Path) -> None:
    problems = control_name_problems(path.read_text(encoding="utf-8"))
    assert problems == [], f"{path.name}: {problems}"
