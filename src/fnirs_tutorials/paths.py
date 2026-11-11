"""Locate the repository root, the data directory and the cache directory.

No path is hard-coded: the root is found by walking up to a marker file, and the data
and cache directories can be overridden by environment variables (useful on Colab,
where the package is not installed in editable mode).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT_MARKER = "myst.yml"
DATA_DIR_ENV = "FNIRS_DATA_DIR"
CACHE_DIR_ENV = "FNIRS_CACHE_DIR"
DEFAULT_DATA_SUBPATH = Path("data")
DEFAULT_CACHE_SUBPATH = Path(".cache") / "fnirs_tutorials"

# Module-level anchor of the upward walk; a test can monkeypatch it.
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


def _find_root(start: Path) -> Path:
    """Walk up from ``start`` to the first directory holding ``ROOT_MARKER``."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if candidate.is_dir() and (candidate / ROOT_MARKER).is_file():
            return candidate
    raise RepoRootNotFoundError(
        f"No directory holding the marker file '{ROOT_MARKER}' was found above "
        f"{start}. This happens with a non-editable install (for example on Colab). "
        f"Set the environment variables {DATA_DIR_ENV} and {CACHE_DIR_ENV} instead."
    )


def _from_env(variable: str) -> Path | None:
    value = os.environ.get(variable, "")
    if not value.strip():
        return None
    return Path(value).expanduser().absolute()


def repo_root() -> Path:
    """Return the repository root, the directory that holds ``myst.yml``.

    The search starts at the resolved location of this module and walks upward. It
    never guesses: when no marker is found, an error is raised.

    Returns
    -------
    pathlib.Path
        Absolute path of the repository root.

    Raises
    ------
    RepoRootNotFoundError
        If no ancestor directory holds the marker file. The message names the marker
        and the environment variables ``FNIRS_DATA_DIR`` and ``FNIRS_CACHE_DIR``.

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

    Uses ``FNIRS_DATA_DIR`` when it is set and non-empty (``~`` expanded, made
    absolute), otherwise ``<repo_root>/data``. The directory is not created.

    Returns
    -------
    pathlib.Path
        Absolute path of the data directory.

    Raises
    ------
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

    Uses ``FNIRS_CACHE_DIR`` when it is set and non-empty (``~`` expanded, made
    absolute), otherwise ``<repo_root>/.cache/fnirs_tutorials``. The directory is not
    created.

    Returns
    -------
    pathlib.Path
        Absolute path of the cache directory.

    Raises
    ------
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
