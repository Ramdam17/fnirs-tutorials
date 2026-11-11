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
    """A module anchor in a valid src layout whose root has no marker file."""
    module_dir = tmp_path / "isolated" / "src" / "pkg"
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
    with pytest.raises(
        paths.RepoRootNotFoundError, match=r"^FNIRS_DATA_DIR is not set"
    ):
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
    with pytest.raises(
        paths.RepoRootNotFoundError, match=r"^FNIRS_CACHE_DIR is not set"
    ):
        paths.cache_dir()


def test_empty_variable_is_treated_as_unset(
    fake_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(paths.DATA_DIR_ENV, "")
    assert paths.data_dir() == fake_repo / "data"


def test_tilde_is_expanded(
    fake_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv(paths.DATA_DIR_ENV, "~/x")
    assert paths.data_dir() == tmp_path / "home" / "x"


def test_whitespace_only_variable_is_treated_as_unset(
    fake_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(paths.DATA_DIR_ENV, "   ")
    assert paths.data_dir() == fake_repo / "data"


def test_surrounding_whitespace_is_stripped(
    fake_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "padded"
    monkeypatch.setenv(paths.DATA_DIR_ENV, f"  {target}\n")
    assert paths.data_dir() == target


@pytest.mark.parametrize(
    ("variable", "function"),
    [
        (paths.DATA_DIR_ENV, paths.data_dir),
        (paths.CACHE_DIR_ENV, paths.cache_dir),
    ],
)
def test_relative_variable_is_rejected(
    fake_repo: Path, monkeypatch: pytest.MonkeyPatch, variable: str, function
) -> None:
    monkeypatch.setenv(variable, "relative/dir")
    with pytest.raises(ValueError, match=variable):
        function()


def test_marker_of_an_enclosing_project_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    site = (
        project / ".venv" / "lib" / "python3.12" / "site-packages" / "fnirs_tutorials"
    )
    site.mkdir(parents=True)
    (project / paths.ROOT_MARKER).write_text("version: 1\n")
    monkeypatch.setattr(paths, "_ANCHOR", site / "paths.py")
    with pytest.raises(paths.RepoRootNotFoundError):
        paths.repo_root()


def test_marker_two_levels_up_without_src_folder_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "project" / "lib" / "fnirs_tutorials"
    package.mkdir(parents=True)
    (tmp_path / "project" / paths.ROOT_MARKER).write_text("version: 1\n")
    monkeypatch.setattr(paths, "_ANCHOR", package / "paths.py")
    with pytest.raises(paths.RepoRootNotFoundError):
        paths.repo_root()


def test_marker_that_is_a_directory_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_dir = tmp_path / "repo" / "src" / "pkg"
    module_dir.mkdir(parents=True)
    (tmp_path / "repo" / paths.ROOT_MARKER).mkdir()
    monkeypatch.setattr(paths, "_ANCHOR", module_dir / "paths.py")
    with pytest.raises(paths.RepoRootNotFoundError):
        paths.repo_root()


def test_symlinked_package_resolves_to_the_real_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_pkg = tmp_path / "checkout" / "src" / "pkg"
    real_pkg.mkdir(parents=True)
    (tmp_path / "checkout" / paths.ROOT_MARKER).write_text("version: 1\n")
    (real_pkg / "paths.py").write_text("")
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "pkg").symlink_to(real_pkg, target_is_directory=True)
    monkeypatch.setattr(paths, "_ANCHOR", tmp_path / "elsewhere" / "pkg" / "paths.py")
    assert paths.repo_root() == (tmp_path / "checkout").resolve()
