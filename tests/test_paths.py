"""Tests of fnirs_tutorials.paths on a temporary tree, independent of the real repository."""

from pathlib import Path

import pytest

from fnirs_tutorials import paths


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temporary tree holding the marker, with the module anchored inside it."""
    root = tmp_path / "repo"
    module_dir = root / "src" / "pkg"
    module_dir.mkdir(parents=True)
    (root / paths.ROOT_MARKER).write_text("version: 1\n")
    monkeypatch.setattr(paths, "_ANCHOR", module_dir / "paths.py")
    monkeypatch.delenv(paths.DATA_DIR_ENV, raising=False)
    monkeypatch.delenv(paths.CACHE_DIR_ENV, raising=False)
    return root.resolve()


@pytest.fixture
def no_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A module anchor with no marker file above it."""
    module_dir = tmp_path / "isolated" / "pkg"
    module_dir.mkdir(parents=True)
    monkeypatch.setattr(paths, "_ANCHOR", module_dir / "paths.py")
    monkeypatch.delenv(paths.DATA_DIR_ENV, raising=False)
    monkeypatch.delenv(paths.CACHE_DIR_ENV, raising=False)


def test_data_dir_uses_environment_variable(
    fake_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "elsewhere"
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(target))
    assert paths.data_dir() == target


def test_data_dir_defaults_to_root_data_from_another_cwd(
    fake_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "other_cwd"
    other.mkdir()
    monkeypatch.chdir(other)
    assert paths.data_dir() == fake_repo / "data"


def test_repo_root_holds_marker(fake_repo: Path) -> None:
    root = paths.repo_root()
    assert root == fake_repo
    assert (root / "myst.yml").is_file()


def test_data_dir_without_marker_names_variable(no_marker: None) -> None:
    with pytest.raises(paths.RepoRootNotFoundError, match="FNIRS_DATA_DIR"):
        paths.data_dir()


def test_cache_dir_uses_environment_variable(
    fake_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "cache_elsewhere"
    monkeypatch.setenv(paths.CACHE_DIR_ENV, str(target))
    assert paths.cache_dir() == target


def test_cache_dir_default(fake_repo: Path) -> None:
    assert paths.cache_dir() == fake_repo / ".cache" / "fnirs_tutorials"


def test_cache_dir_without_marker_names_variable(no_marker: None) -> None:
    with pytest.raises(paths.RepoRootNotFoundError, match="FNIRS_CACHE_DIR"):
        paths.cache_dir()


def test_empty_variable_is_treated_as_unset(
    fake_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(paths.DATA_DIR_ENV, "")
    assert paths.data_dir() == fake_repo / "data"
