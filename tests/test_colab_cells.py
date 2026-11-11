"""Tests of the Colab link, the setup cell and ``scripts/sync_colab_cells.py``.

None of these tests reaches the network, starts ``uv`` or ends the test process. The
setup cell is compiled from its rendered source and executed in a namespace where
``google.colab`` is absent, which defines its functions and runs nothing; the tests then
replace the functions that start a process, fetch a file, wait or end the kernel by
stand-ins and call the cell's own functions. Only the first group of tests reads the
notebooks of the book; the others build their notebooks in ``tmp_path``.
"""

from __future__ import annotations

import copy
import difflib
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tomllib
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import pytest
import yaml
from packaging.specifiers import SpecifierSet

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "sync_colab_cells.py"

_spec = importlib.util.spec_from_file_location("sync_colab_cells", SCRIPT)
assert _spec is not None and _spec.loader is not None
sync = importlib.util.module_from_spec(_spec)
sys.modules["sync_colab_cells"] = sync
_spec.loader.exec_module(sync)

CONFIG = sync.load_config(REPO_ROOT / "config" / "colab.yaml")
OVERRIDES = sync.read_overrides(REPO_ROOT / "pyproject.toml")
TEMPLATE = (REPO_ROOT / sync.TEMPLATE_SUBPATH).read_text(encoding="utf-8")
BOOK_NOTEBOOKS = sync._load_toc_reader().list_notebooks(REPO_ROOT / "myst.yml")


def _render(ref: str) -> str:
    return sync.render_cell(TEMPLATE, CONFIG, ref, OVERRIDES)


# ---------------------------------------------------------------------------
# The configuration and the notebooks of the book.
# ---------------------------------------------------------------------------


def test_python_version_is_accepted_by_requires_python() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    specifier = SpecifierSet(pyproject["project"]["requires-python"])
    assert specifier.contains(f"{CONFIG.python_version}.0")


def test_runtime_version_is_a_string_in_the_file() -> None:
    document = yaml.safe_load((REPO_ROOT / "config" / "colab.yaml").read_text("utf-8"))
    assert isinstance(document["colab_runtime_version"], str)
    assert isinstance(document["python_version"], str)


def test_template_ships_the_values_it_would_be_rendered_with() -> None:
    assert _render(CONFIG.default_ref) == TEMPLATE.rstrip("\n")


@pytest.mark.parametrize("notebook", BOOK_NOTEBOOKS)
def test_book_notebook_has_badge_cell_and_runtime_request(notebook: str) -> None:
    nb = json.loads((REPO_ROOT / notebook).read_text(encoding="utf-8"))
    cells = nb["cells"]

    first_markdown = next(c for c in cells if c["cell_type"] == "markdown")
    badge = re.compile(
        re.escape(f"[![Open In Colab]({sync.BADGE_IMAGE})]")
        + re.escape(
            f"(https://colab.research.google.com/github/{CONFIG.repository}/blob/"
        )
        + r"(?P<ref>.+)/"
        + re.escape(notebook)
        + r"\)"
    )
    badges = [
        m
        for line in first_markdown["source"]
        if (m := badge.fullmatch(line.rstrip("\n")))
    ]
    assert len(badges) == 1
    ref = badges[0].group("ref")
    sync.validate_ref(ref)

    first_code = next(c for c in cells if c["cell_type"] == "code")
    assert "".join(first_code["source"]) == _render(ref)
    assert first_code["metadata"]["tags"] == ["colab-setup", "hide-input"]
    assert first_code["outputs"] == [] and first_code["execution_count"] is None
    assert (
        sum("colab-setup" in c.get("metadata", {}).get("tags", []) for c in cells) == 1
    )

    attributes = nb["metadata"]["colab"]["runtime_attributes"]
    assert attributes["runtime_version"] == CONFIG.runtime_version


def test_check_passes_on_the_notebooks_of_the_book() -> None:
    assert sync.main(["--check"]) == 0


# ---------------------------------------------------------------------------
# The setup cell, executed without Colab, without network and without a kill.
# ---------------------------------------------------------------------------


def _fresh_namespace() -> dict[str, Any]:
    """Execute the rendered cell outside Colab: it defines its functions, nothing more."""
    namespace: dict[str, Any] = {"__name__": "colab_cell"}
    # The source is the template of this repository, rendered by the script under test.
    exec(compile(_render("main"), "<setup cell>", "exec"), namespace)  # noqa: S102
    return namespace


class Cell:
    """The rendered cell executed in a namespace, with stand-ins for its side effects."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delitem(sys.modules, "google.colab", raising=False)
        for name in ("FNIRS_DATA_DIR", "FNIRS_CACHE_DIR"):
            monkeypatch.setenv(name, "unset-by-test")
            monkeypatch.delenv(name)
        self.calls: list[tuple[Any, ...]] = []
        self.setup_dir = tmp_path / "setup"
        self.ns = _fresh_namespace()
        self.ns["_SETUP_DIR"] = str(self.setup_dir)
        self.ns["_DATA_DIR"] = str(tmp_path / "data")
        self.ns["_CACHE_DIR"] = str(tmp_path / "cache")
        self.ns["_running_python"] = lambda: CONFIG.python_version
        self.ns["_shutil"] = SimpleNamespace(which=lambda name: f"/bin/{name}")
        self.ns["_kernel_has_old_numpy"] = lambda: False
        clock = iter([0.0, 66.0, 91.0])
        self.ns["_time"] = SimpleNamespace(
            sleep=lambda seconds: self.calls.append(("sleep", seconds)),
            monotonic=lambda: next(clock),
        )
        self.ns["_run"] = self._run
        self.ns["_fetch"] = self._fetch
        self.ns["_kill_kernel"] = self._kill
        self.fail_on: str | None = None

    @property
    def marker(self) -> Path:
        return self.setup_dir / "installed.json"

    def _run(self, command: list[str]) -> None:
        kind = "install" if command[0] == "uv" else "download"
        self.calls.append((kind, command))
        if self.fail_on == kind:
            raise RuntimeError(f"stand-in failure of {kind}")

    def _fetch(self, url: str, destination: Path) -> None:
        self.calls.append(("fetch", url, destination))
        if self.fail_on == "fetch":
            raise RuntimeError("stand-in failure of fetch")
        destination.write_text("# stand-in", encoding="utf-8")

    def _kill(self) -> None:
        self.calls.append(("kill", self.marker.exists()))

    def kinds(self) -> list[str]:
        return [call[0] for call in self.calls]

    def setup(self) -> None:
        self.ns["_setup"]()


@pytest.fixture
def cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Cell:
    return Cell(tmp_path, monkeypatch)


def test_outside_colab_the_cell_does_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the cell must not do this outside Colab")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(os, "kill", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.chdir(tmp_path)
    environment = dict(os.environ)
    namespace = _fresh_namespace()

    assert capsys.readouterr().out == ""
    assert dict(os.environ) == environment
    assert list(tmp_path.iterdir()) == []
    public = [name for name in namespace if not name.startswith("_")]
    assert public == []


def test_colab_is_detected_by_the_google_colab_module(
    cell: Cell, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cell.ns["_on_colab"]() is False
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace())
    assert cell.ns["_on_colab"]() is True


def test_another_python_version_is_refused_with_the_runtime_version(
    cell: Cell,
) -> None:
    cell.ns["_running_python"] = lambda: "3.13"
    with pytest.raises(RuntimeError) as error:
        cell.setup()
    message = str(error.value)
    assert "3.13" in message and CONFIG.python_version in message
    assert CONFIG.runtime_version in message
    assert "Change runtime type" in message and "Runtime version" in message
    assert cell.calls == [] and not cell.marker.exists()


def test_fresh_setup_runs_every_step_in_order(
    cell: Cell, capsys: pytest.CaptureFixture[str]
) -> None:
    cell.setup()

    assert cell.kinds() == ["install", "fetch", "download", "sleep", "kill"]
    install, fetch, download, sleep, kill = cell.calls
    overrides_file = cell.setup_dir / "overrides.txt"
    assert install[1] == [
        "uv",
        "pip",
        "install",
        "--system",
        "--override",
        str(overrides_file),
        f"fnirs-tutorials @ git+https://github.com/{CONFIG.repository}@main",
    ]
    assert overrides_file.read_text("utf-8").splitlines() == OVERRIDES
    script = cell.setup_dir / "download_data.py"
    assert fetch[1] == (
        f"https://raw.githubusercontent.com/{CONFIG.repository}/main/"
        "scripts/download_data.py"
    )
    assert fetch[2] == script
    assert download[1] == [
        sys.executable,
        "-u",
        str(script),
        "--data-dir",
        cell.ns["_DATA_DIR"],
    ]
    assert sleep == ("sleep", 2)
    assert kill == ("kill", True), "the marker exists when the kernel is ended"
    assert json.loads(cell.marker.read_text("utf-8")) == {
        "ref": "main",
        "overrides": OVERRIDES,
    }
    assert os.environ["FNIRS_DATA_DIR"] == cell.ns["_DATA_DIR"]
    assert os.environ["FNIRS_CACHE_DIR"] == cell.ns["_CACHE_DIR"]

    printed = capsys.readouterr().out
    assert "Setup time in seconds: install 66.0, download 25.0, total 91.0" in printed
    for phrase in (
        "The environment is installed",
        "restarts once, on purpose",
        "Your session crashed for an unknown reason",
        '"Runtime", "Run all"',
        "about a second",
    ):
        assert phrase in printed
    assert "!" not in printed


@pytest.mark.parametrize("step", ["install", "fetch", "download"])
def test_a_failed_step_leaves_no_marker_and_ends_no_kernel(
    cell: Cell, step: str
) -> None:
    cell.fail_on = step
    with pytest.raises(RuntimeError):
        cell.setup()
    assert "kill" not in cell.kinds() and "sleep" not in cell.kinds()
    assert not cell.marker.exists()


def test_a_missing_uv_is_a_clear_error(cell: Cell) -> None:
    cell.ns["_shutil"] = SimpleNamespace(which=lambda name: None)
    with pytest.raises(RuntimeError, match="uv"):
        cell.setup()
    assert cell.calls == [] and not cell.marker.exists()


def test_second_run_with_the_marker_installs_nothing(
    cell: Cell, capsys: pytest.CaptureFixture[str]
) -> None:
    cell.setup()
    capsys.readouterr()
    cell.calls.clear()
    os.environ.pop("FNIRS_DATA_DIR")
    os.environ.pop("FNIRS_CACHE_DIR")

    cell.setup()

    assert cell.calls == []
    assert os.environ["FNIRS_DATA_DIR"] == cell.ns["_DATA_DIR"]
    assert os.environ["FNIRS_CACHE_DIR"] == cell.ns["_CACHE_DIR"]
    assert capsys.readouterr().out.strip() == "The environment is ready."


@pytest.mark.parametrize(
    "marker",
    [
        {"ref": "other", "overrides": OVERRIDES},
        {"ref": "main", "overrides": ["scipy==0.0"]},
        "not json",
    ],
)
def test_a_marker_for_another_ref_or_overrides_does_not_count(
    cell: Cell, marker: object
) -> None:
    cell.setup_dir.mkdir()
    text = marker if isinstance(marker, str) else json.dumps(marker)
    cell.marker.write_text(text, encoding="utf-8")
    cell.setup()
    assert cell.kinds()[0] == "install"


def test_a_stale_numpy_in_the_kernel_ends_the_session_once(
    cell: Cell, capsys: pytest.CaptureFixture[str]
) -> None:
    cell.setup()
    cell.calls.clear()
    capsys.readouterr()
    del cell.ns["_kernel_has_old_numpy"]
    cell.ns["_kernel_has_old_numpy"] = lambda: True

    cell.setup()

    assert cell.kinds() == ["sleep", "kill"]
    assert "The environment is installed" in capsys.readouterr().out


def test_the_numpy_comparison_uses_the_loaded_and_the_installed_version(
    cell: Cell, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = _fresh_namespace()
    namespace["_metadata"] = SimpleNamespace(version=lambda name: "2.2.6")
    check = namespace["_kernel_has_old_numpy"]
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(__version__="2.2.6"))
    assert check() is False
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(__version__="2.0.2"))
    assert check() is True
    monkeypatch.delitem(sys.modules, "numpy")
    assert check() is False


def test_the_kernel_is_ended_with_signal_nine(cell: Cell) -> None:
    sent: list[tuple[int, int]] = []
    cell.ns["_os"] = SimpleNamespace(
        getpid=lambda: 4242, kill=lambda pid, sig: sent.append((pid, sig))
    )
    fresh = _fresh_namespace()
    fresh["_os"] = cell.ns["_os"]
    fresh["_kill_kernel"]()
    assert sent == [(4242, 9)]


def test_command_output_is_shown_line_by_line_and_failures_raise(
    cell: Cell, capsys: pytest.CaptureFixture[str]
) -> None:
    fresh = _fresh_namespace()
    code = "import sys; print('one'); print('two', file=sys.stderr); sys.exit(3)"
    with pytest.raises(RuntimeError) as error:
        fresh["_run"]([sys.executable, "-c", code])
    out = capsys.readouterr().out
    assert "one\n" in out and "two\n" in out
    assert "exit code 3" in str(error.value) and "-c" in str(error.value)


def test_fetch_writes_the_bytes_of_the_response(cell: Cell, tmp_path: Path) -> None:
    class Response:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return b"payload"

    seen: list[str] = []
    fresh = _fresh_namespace()
    fresh["_request"] = SimpleNamespace(
        urlopen=lambda url, timeout: seen.append(url) or Response()
    )
    target = tmp_path / "fetched.py"
    fresh["_fetch"]("https://example.invalid/x.py", target)
    assert seen == ["https://example.invalid/x.py"]
    assert target.read_bytes() == b"payload"


# ---------------------------------------------------------------------------
# The script, on notebooks built in tmp_path.
# ---------------------------------------------------------------------------


def _notebook() -> dict[str, Any]:
    def cell(kind: str, cell_id: str, source: str, **extra: Any) -> dict[str, Any]:
        result: dict[str, Any] = {
            "cell_type": kind,
            "id": cell_id,
            "metadata": {},
            "source": sync._to_lines(source),
        }
        if kind == "code":
            result["execution_count"] = extra.get("execution_count")
            result["outputs"] = extra.get("outputs", [])
        return result

    return {
        "cells": [
            cell("markdown", "aaaa0001", "# Cédalion title\n\nSome text."),
            cell("markdown", "aaaa0002", "## Hands-on"),
            cell(
                "code",
                "aaaa0003",
                "import numpy as np\nprint('é')",
                execution_count=3,
                outputs=[{"name": "stdout", "output_type": "stream", "text": ["é\n"]}],
            ),
            cell("code", "aaaa0004", "x = 1"),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOK_PATHS = ["book/a.ipynb", "book/sub/b.ipynb"]


def _make_project(root: Path) -> Path:
    (root / "config").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "config" / "colab.yaml", root / "config" / "colab.yaml")
    shutil.copy(REPO_ROOT / "pyproject.toml", root / "pyproject.toml")
    (root / sync.TEMPLATE_SUBPATH).parent.mkdir(parents=True)
    shutil.copy(REPO_ROOT / sync.TEMPLATE_SUBPATH, root / sync.TEMPLATE_SUBPATH)
    toc = [{"file": path} for path in NOTEBOOK_PATHS]
    (root / "myst.yml").write_text(
        yaml.safe_dump({"project": {"toc": toc}}), encoding="utf-8"
    )
    for path in NOTEBOOK_PATHS:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(sync.dump_notebook(_notebook()), encoding="utf-8")
    return root


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return _make_project(tmp_path / "project")


def _read(root: Path, path: str) -> dict[str, Any]:
    return json.loads((root / path).read_text(encoding="utf-8"))


def _write(root: Path, path: str, nb: dict[str, Any]) -> None:
    (root / path).write_text(sync.dump_notebook(nb), encoding="utf-8")


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p: (root / p).read_bytes() for p in NOTEBOOK_PATHS}


def test_sync_writes_badge_cell_and_runtime_request(project: Path) -> None:
    assert sync.main(["--root", str(project), "--ref", "abc1234"]) == 0
    nb = _read(project, "book/a.ipynb")

    first = "".join(nb["cells"][0]["source"])
    badge = (
        f"[![Open In Colab]({sync.BADGE_IMAGE})]"
        f"(https://colab.research.google.com/github/{CONFIG.repository}"
        "/blob/abc1234/book/a.ipynb)"
    )
    assert first.split("\n")[:2] == ["# Cédalion title", badge]
    setup = nb["cells"][2]
    assert setup["cell_type"] == "code" and setup["id"] == "colab-setup"
    assert setup["metadata"] == {"tags": ["colab-setup", "hide-input"]}
    assert setup["outputs"] == [] and setup["execution_count"] is None
    assert "".join(setup["source"]) == _render("abc1234")
    assert nb["metadata"]["colab"] == {
        "provenance": [],
        "runtime_attributes": {"runtime_version": CONFIG.runtime_version},
    }
    assert "kernelspec" in nb["metadata"]


def test_other_cells_keep_their_outputs_and_counts(project: Path) -> None:
    before = _read(project, "book/a.ipynb")
    sync.main(["--root", str(project)])
    after = _read(project, "book/a.ipynb")
    assert after["cells"][3] == before["cells"][2]
    assert after["cells"][4] == before["cells"][3]
    assert after["cells"][1] == before["cells"][1]
    assert after["cells"][3]["execution_count"] == 3


def test_a_notebook_with_no_ref_gets_the_default_ref(project: Path) -> None:
    sync.main(["--root", str(project)])
    source = "".join(_read(project, "book/a.ipynb")["cells"][2]["source"])
    assert f'_REF = "{CONFIG.default_ref}"' in source


def test_without_ref_a_notebook_keeps_the_ref_it_carries(project: Path) -> None:
    sync.main(["--root", str(project), "--ref", "v1.2.3"])
    sync.main(["--root", str(project)])
    source = "".join(_read(project, "book/a.ipynb")["cells"][2]["source"])
    assert '_REF = "v1.2.3"' in source


def test_a_second_run_changes_nothing(project: Path) -> None:
    sync.main(["--root", str(project), "--ref", "abc1234"])
    once = _snapshot(project)
    sync.main(["--root", str(project), "--ref", "abc1234"])
    sync.main(["--root", str(project)])
    assert _snapshot(project) == once


def test_ref_a_then_ref_b_equals_ref_b_alone(tmp_path: Path) -> None:
    first = _make_project(tmp_path / "first")
    second = _make_project(tmp_path / "second")
    sync.main(["--root", str(first), "--ref", "aaaa111"])
    sync.main(["--root", str(first), "--ref", "bbbb222"])
    sync.main(["--root", str(second), "--ref", "bbbb222"])
    assert _snapshot(first) == _snapshot(second)


def test_one_badge_and_one_cell_even_after_hand_additions(project: Path) -> None:
    sync.main(["--root", str(project)])
    nb = _read(project, "book/a.ipynb")
    nb["cells"][0]["source"].insert(2, nb["cells"][0]["source"][1])
    _write(project, "book/a.ipynb", nb)
    sync.main(["--root", str(project)])
    nb = _read(project, "book/a.ipynb")
    badges = [
        line
        for line in nb["cells"][0]["source"]
        if line.startswith("[![Open In Colab]")
    ]
    tagged = [c for c in nb["cells"] if "colab-setup" in c["metadata"].get("tags", [])]
    assert len(badges) == 1 and len(tagged) == 1


def test_the_layout_is_kept_outside_the_lines_the_script_owns(project: Path) -> None:
    original = (project / "book/a.ipynb").read_text(encoding="utf-8")
    sync.main(["--root", str(project), "--ref", "aaaa111"])
    written = (project / "book/a.ipynb").read_text(encoding="utf-8")
    assert "é" in written and "\\u00e9" not in written and written.endswith("}\n")
    opcodes = difflib.SequenceMatcher(
        None, original.splitlines(), written.splitlines(), autojunk=False
    ).get_opcodes()
    assert {tag for tag, *_ in opcodes} <= {"equal", "insert"}

    # Moving to another ref changes only the lines that name the ref.
    sync.main(["--root", str(project), "--ref", "bbbb222"])
    moved = (project / "book/a.ipynb").read_text(encoding="utf-8")
    changed = [
        line
        for line in difflib.unified_diff(
            written.splitlines(), moved.splitlines(), lineterm="", n=0
        )
        if line[:1] in "+-" and line[:3] not in ("+++", "---")
    ]
    assert changed and all(
        ("aaaa111" in line) or ("bbbb222" in line) for line in changed
    )


@pytest.mark.parametrize(
    "ref", ["-x", "a..b", "a b", "a;b", "$(id)", "", "a'b", 'a"b', "a\nb", "é", "--ref"]
)
def test_a_refused_ref_writes_nothing(
    project: Path, ref: str, caplog: pytest.LogCaptureFixture
) -> None:
    before = _snapshot(project)
    with caplog.at_level(logging.ERROR):
        assert sync.main(["--root", str(project), f"--ref={ref}"]) == 2
    assert "refused" in caplog.text
    assert _snapshot(project) == before


@pytest.mark.parametrize("ref", ["main", "v1.2.3", "feature/colab_1", "a" * 40])
def test_valid_refs_are_accepted(ref: str) -> None:
    assert sync.validate_ref(ref) == ref


def test_a_layout_other_than_nbformat_is_refused(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = project / "book/a.ipynb"
    path.write_text(json.dumps(_notebook(), indent=4), encoding="utf-8")
    before = _snapshot(project)
    with caplog.at_level(logging.ERROR):
        assert sync.main(["--root", str(project)]) == 2
        assert sync.main(["--root", str(project), "--check"]) == 1
    assert "layout" in caplog.text
    assert _snapshot(project) == before


def test_check_fails_on_notebooks_that_were_never_synced(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    before = _snapshot(project)
    with caplog.at_level(logging.ERROR):
        assert sync.main(["--root", str(project), "--check"]) == 1
    assert "book/a.ipynb" in caplog.text and "book/sub/b.ipynb" in caplog.text
    assert _snapshot(project) == before


def test_check_passes_after_a_sync_and_writes_nothing(project: Path) -> None:
    sync.main(["--root", str(project), "--ref", "abc1234"])
    before = _snapshot(project)
    assert sync.main(["--root", str(project), "--check"]) == 0
    assert _snapshot(project) == before


def _drop_badge(nb: dict[str, Any]) -> None:
    nb["cells"][0]["source"] = [
        line for line in nb["cells"][0]["source"] if not line.startswith("[![")
    ]


def _badge_other_ref(nb: dict[str, Any]) -> None:
    lines = nb["cells"][0]["source"]
    lines[1] = lines[1].replace("/blob/abc1234/", "/blob/zzz9999/")


def _edit_cell(nb: dict[str, Any]) -> None:
    nb["cells"][2]["source"].append("\nprint('hand edit')")


def _cell_not_first(nb: dict[str, Any]) -> None:
    nb["cells"][2], nb["cells"][3] = nb["cells"][3], nb["cells"][2]


def _two_cells(nb: dict[str, Any]) -> None:
    clone = copy.deepcopy(nb["cells"][2])
    clone["id"] = "colab-setup-2"
    nb["cells"].append(clone)


def _runtime_missing(nb: dict[str, Any]) -> None:
    del nb["metadata"]["colab"]["runtime_attributes"]


def _runtime_other(nb: dict[str, Any]) -> None:
    nb["metadata"]["colab"]["runtime_attributes"]["runtime_version"] = "2025.10"


def _cell_with_output(nb: dict[str, Any]) -> None:
    nb["cells"][2]["execution_count"] = 1


def _cell_shown(nb: dict[str, Any]) -> None:
    nb["cells"][2]["metadata"]["tags"] = ["colab-setup"]


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (_drop_badge, "badge is missing"),
        (_badge_other_ref, "badge differs"),
        (_edit_cell, "setup cell differs"),
        (_cell_not_first, "not the first code cell"),
        (_two_cells, "2 cells carry the tag"),
        (_runtime_missing, "runtime request"),
        (_runtime_other, "2025.10"),
        (_cell_with_output, "execution count"),
        (_cell_shown, "exactly the tags"),
    ],
)
def test_check_names_each_difference(
    project: Path,
    caplog: pytest.LogCaptureFixture,
    mutate: Any,
    expected: str,
) -> None:
    sync.main(["--root", str(project), "--ref", "abc1234"])
    nb = _read(project, "book/a.ipynb")
    mutate(nb)
    _write(project, "book/a.ipynb", nb)
    with caplog.at_level(logging.ERROR):
        assert sync.main(["--root", str(project), "--check"]) == 1
    assert "book/a.ipynb" in caplog.text and expected in caplog.text
    assert "book/sub/b.ipynb" not in caplog.text


def test_check_fails_when_the_overrides_of_pyproject_changed(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sync.main(["--root", str(project), "--ref", "abc1234"])
    pyproject = project / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert "scipy==1.16.3" in text
    pyproject.write_text(text.replace("scipy==1.16.3", "scipy==1.16.4"), "utf-8")
    with caplog.at_level(logging.ERROR):
        assert sync.main(["--root", str(project), "--check"]) == 1
    assert "overrides" in caplog.text
    # Writing again brings the notebooks back in line.
    assert sync.main(["--root", str(project)]) == 0
    assert sync.main(["--root", str(project), "--check"]) == 0
