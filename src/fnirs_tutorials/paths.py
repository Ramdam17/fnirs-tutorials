"""Locate the repository root, the data directory and the cache directory.

No path is hard-coded: the root is found from the module's place in the source layout, and the data
and cache directories can be overridden by environment variables (useful on Colab,
where the package is not installed in editable mode).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT_MARKER = "myst.yml"
SRC_DIR_NAME = "src"
DATA_DIR_ENV = "FNIRS_DATA_DIR"
CACHE_DIR_ENV = "FNIRS_CACHE_DIR"
DEFAULT_DATA_SUBPATH = Path("data")
DEFAULT_CACHE_SUBPATH = Path(".cache") / "fnirs_tutorials"

# Module-level anchor of the root search; a test can monkeypatch it.
_ANCHOR = Path(__file__)

__all__ = [
    "CACHE_DIR_ENV",
    "DATA_DIR_ENV",
    "ROOT_MARKER",
    "RepoRootNotFoundError",
    "cache_dir",
    "data_dir",
    "repo_root",
]


class RepoRootNotFoundError(RuntimeError):
    """Raised when no directory holding the root marker file can be found."""


def _find_root(anchor: Path) -> Path:
    """Return the repository root from the module's place in the source layout.

    With the resolved ``anchor`` at ``<root>/src/<package>/paths.py``, the root is the
    directory two levels above the package folder. It is accepted only when the
    folder between them is named ``SRC_DIR_NAME`` and the root holds ``ROOT_MARKER``.
    No other ancestor is considered, so a ``myst.yml`` of an unrelated project that
    encloses a non-editable install is never mistaken for this repository.
    """
    anchor = anchor.resolve()
    package_dir = anchor.parent
    src_dir = package_dir.parent
    root = src_dir.parent
    if src_dir.name == SRC_DIR_NAME and (root / ROOT_MARKER).is_file():
        return root
    raise RepoRootNotFoundError(
        f"{anchor} is not at <root>/{SRC_DIR_NAME}/<package>/ of a repository whose "
        f"root holds the marker file '{ROOT_MARKER}'. This happens with a "
        f"non-editable install (for example on Colab). "
        f"Set the environment variables {DATA_DIR_ENV} and {CACHE_DIR_ENV} instead."
    )


def _from_env(variable: str) -> Path | None:
    value = os.environ.get(variable, "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"{variable} must be an absolute path (or start with '~'), got {value!r}."
        )
    return path


def repo_root() -> Path:
    """Return the repository root, the directory that holds ``myst.yml``.

    The module must sit at ``<root>/src/<package>/paths.py`` (resolved) and ``<root>``
    must hold the marker. No other directory is searched and nothing is guessed:
    otherwise an error is raised.

    Returns
    -------
    pathlib.Path
        Absolute path of the repository root.

    Raises
    ------
    RepoRootNotFoundError
        If the module is not at that place of a source layout whose root holds the
        marker. The message names the marker and the environment variables ``FNIRS_DATA_DIR`` and ``FNIRS_CACHE_DIR``.

    Notes
    -----
    When to use: to build paths to files that live in the repository (book pages,
    configuration). Nothing is created on disk.

    Examples
    --------
    >>> from fnirs_tutorials.paths import repo_root
    >>> root = repo_root()  # doctest: +SKIP
    """
    root = _find_root(_ANCHOR)
    logger.debug("repo_root resolved by the %s marker: %s", ROOT_MARKER, root)
    return root


def _resolve(variable: str, subpath: Path) -> Path:
    from_env = _from_env(variable)
    if from_env is not None:
        logger.debug("%s resolved by the environment variable: %s", variable, from_env)
        return from_env
    try:
        root = repo_root()
    except RepoRootNotFoundError as error:
        raise RepoRootNotFoundError(
            f"{variable} is not set and the repository root cannot be found. {error}"
        ) from error
    resolved = root / subpath
    logger.debug("%s unset, default under the repository root: %s", variable, resolved)
    return resolved


def data_dir() -> Path:
    """Return the data directory.

    Uses ``FNIRS_DATA_DIR`` when it is set and not blank (stripped, ``~`` expanded), otherwise ``<repo_root>/data``. The directory is not created.

    Returns
    -------
    pathlib.Path
        Absolute path of the data directory.

    Raises
    ------
    ValueError
        If ``FNIRS_DATA_DIR`` is set to a relative path.
    RepoRootNotFoundError
        If the variable is unset and the repository root cannot be found; the message
        names ``FNIRS_DATA_DIR``.

    Notes
    -----
    When to use: every time a notebook reads or downloads an open dataset.

    Examples
    --------
    >>> import os
    >>> os.environ["FNIRS_DATA_DIR"] = "/tmp/fnirs-data"
    >>> from fnirs_tutorials.paths import data_dir
    >>> data_dir()
    PosixPath('/tmp/fnirs-data')
    """
    return _resolve(DATA_DIR_ENV, DEFAULT_DATA_SUBPATH)


def cache_dir() -> Path:
    """Return the cache directory for precomputed results.

    Uses ``FNIRS_CACHE_DIR`` when it is set and not blank (stripped, ``~`` expanded), otherwise ``<repo_root>/.cache/fnirs_tutorials``. The directory is not
    created.

    Returns
    -------
    pathlib.Path
        Absolute path of the cache directory.

    Raises
    ------
    ValueError
        If ``FNIRS_CACHE_DIR`` is set to a relative path.
    RepoRootNotFoundError
        If the variable is unset and the repository root cannot be found; the message
        names ``FNIRS_CACHE_DIR``.

    Notes
    -----
    When to use: to store or reload heavy intermediate results so that notebooks
    execute in a few minutes.

    Examples
    --------
    >>> import os
    >>> os.environ["FNIRS_CACHE_DIR"] = "/tmp/fnirs-cache"
    >>> from fnirs_tutorials.paths import cache_dir
    >>> cache_dir()
    PosixPath('/tmp/fnirs-cache')
    """
    return _resolve(CACHE_DIR_ENV, DEFAULT_CACHE_SUBPATH)
