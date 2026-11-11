"""Execute the notebooks of the book headless, under a per-notebook timeout.

This module is shared by ``tests/test_notebooks.py``, which runs every notebook listed
in the table of contents of ``myst.yml``, and by ``tests/test_notebook_failure.py``,
which runs two fixtures that must fail. It is test support code, not part of the
``fnirs_tutorials`` package.

The timeout of ``nbclient.NotebookClient`` bounds each cell, not the notebook, and a
value of zero or less means no limit at all (``_get_timeout`` in ``nbclient/client.py``,
version 0.11.0). The runner therefore gives each cell the time that remains of the
notebook's budget through the documented ``timeout_func`` hook, never less than
``MIN_CELL_TIMEOUT_S``, and also reports a notebook as a timeout when its measured wall
time exceeds the budget although no single cell timed out.

When to use
-----------
Call :func:`list_notebooks` and :func:`load_timeouts` once, then :func:`execute_notebook`
and :func:`record_result` for each notebook.

Examples
--------
>>> from pathlib import Path
>>> root = Path(".")
>>> notebooks = list_notebooks(root / "myst.yml")  # doctest: +SKIP
>>> config = load_timeouts(root / "config" / "notebooks.yaml", notebooks)  # doctest: +SKIP
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nbformat
import yaml
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

__all__ = [
    "MIN_CELL_TIMEOUT_S",
    "STATUS_ERROR",
    "STATUS_PASSED",
    "STATUS_TIMEOUT",
    "NotebookResult",
    "TimeoutConfig",
    "execute_notebook",
    "list_notebooks",
    "load_timeouts",
    "record_result",
    "remaining_cell_timeout",
    "reset_timings",
]

logger = logging.getLogger(__name__)

STATUS_PASSED = "passed"
STATUS_ERROR = "error"
STATUS_TIMEOUT = "timeout"

# Smallest time given to a cell when the notebook's budget is nearly or fully spent.
# It must stay above zero because nbclient reads zero as "no limit". The overrun is
# caught afterwards by the wall-time check, so this floor never lets a notebook pass
# beyond its budget.
MIN_CELL_TIMEOUT_S = 1.0

# Kinds of entry of a MyST table of contents that name one page (``file``), link out
# (``url``), or only group children under a title. Source: the table-of-contents page
# of https://mystmd.org/guide/table-of-contents. The remaining keys only decorate an
# entry. Any other key is refused so that an entry able to pull notebooks in without
# naming them (``pattern`` is the known one) can never be skipped silently.
_TOC_ENTRY_KEYS = frozenset(
    {
        "file",
        "url",
        "title",
        "short_title",
        "children",
        "hidden",
        "open_in_same_tab",
    }
)
_NOTEBOOK_SUFFIX = ".ipynb"


@dataclass(frozen=True)
class NotebookResult:
    """Outcome of the execution of one notebook.

    Attributes
    ----------
    status : str
        One of ``"passed"``, ``"error"`` or ``"timeout"``.
    wall_time_s : float
        Wall time of the whole execution, kernel start-up included, in seconds.
    timeout_s : float
        Budget of the whole notebook, in seconds.
    error : str or None
        Short description of the failure, None when the notebook passed.
    """

    status: str
    wall_time_s: float
    timeout_s: float
    error: str | None = None


@dataclass(frozen=True)
class TimeoutConfig:
    """Timeouts read from ``config/notebooks.yaml``.

    Attributes
    ----------
    default_timeout_s : float
        Budget of a notebook that has no entry of its own, in seconds.
    timeouts : dict of str to float
        Budget of single notebooks, keyed by their path relative to the repository
        root in POSIX form, in seconds.
    """

    default_timeout_s: float
    timeouts: dict[str, float]

    def timeout_for(self, notebook: str) -> float:
        """Return the budget of ``notebook`` in seconds.

        Parameters
        ----------
        notebook : str
            Path of the notebook relative to the repository root, POSIX form.

        Returns
        -------
        float
            The notebook's own timeout when it has one, the default otherwise.
        """
        return self.timeouts.get(notebook, self.default_timeout_s)


def _collect_notebooks(entries: Any, where: str, found: list[str]) -> None:
    """Append the notebooks named in a list of table-of-contents entries.

    Parameters
    ----------
    entries : Any
        The value of a ``toc`` or ``children`` key.
    where : str
        Human-readable location of ``entries``, used in error messages.
    found : list of str
        Receives the notebook paths, in table-of-contents order.

    Raises
    ------
    TypeError
        If ``entries`` is not a list or an entry is not a mapping.
    ValueError
        If an entry holds a key of an unsupported kind, or a ``file`` entry has no
        extension.
    """
    if not isinstance(entries, list):
        raise TypeError(f"{where}: expected a list of entries, got {entries!r}")
    for index, entry in enumerate(entries):
        here = f"{where}[{index}]"
        if not isinstance(entry, dict):
            raise TypeError(f"{here}: expected a mapping, got {entry!r}")
        unsupported = sorted(set(entry) - _TOC_ENTRY_KEYS)
        if unsupported:
            raise ValueError(
                f"{here}: unsupported table-of-contents key(s) {unsupported}; an entry "
                "of that kind could bring in notebooks without naming them, and a "
                "notebook must never be skipped silently. List the files explicitly."
            )
        file_name = entry.get("file")
        if file_name is not None:
            suffix = Path(str(file_name)).suffix
            if not suffix:
                raise ValueError(
                    f"{here}: file entry {file_name!r} has no extension, so it cannot "
                    "be told from a notebook"
                )
            if suffix == _NOTEBOOK_SUFFIX:
                found.append(Path(str(file_name)).as_posix())
        if "children" in entry:
            _collect_notebooks(entry["children"], f"{here}.children", found)


def list_notebooks(myst_path: Path) -> list[str]:
    """List the notebooks of the table of contents of a ``myst.yml``.

    Parameters
    ----------
    myst_path : Path
        Path of the ``myst.yml`` file. Its folder is the repository root, against which
        the entries of the table of contents are relative.

    Returns
    -------
    list of str
        Path of every ``file:`` entry ending in ``.ipynb``, at any depth of
        ``children``, relative to the repository root in POSIX form, in the order of
        the table of contents.

    Raises
    ------
    ValueError
        If the file has no ``project.toc``, an entry is of an unsupported kind (a
        ``pattern``, for instance), or a listed notebook does not exist.

    Examples
    --------
    >>> list_notebooks(Path("myst.yml"))  # doctest: +SKIP
    ['book/smoke/smoke_cedalion.ipynb']
    """
    document = yaml.safe_load(myst_path.read_text(encoding="utf-8")) or {}
    project = document.get("project") or {}
    if "toc" not in project:
        raise ValueError(f"{myst_path}: no project.toc found")
    found: list[str] = []
    _collect_notebooks(project["toc"], "project.toc", found)
    for notebook in found:
        if not (myst_path.parent / notebook).is_file():
            raise ValueError(f"{myst_path}: notebook {notebook!r} does not exist")
    return found


def load_timeouts(config_path: Path, notebooks: list[str]) -> TimeoutConfig:
    """Read the timeouts of ``config/notebooks.yaml`` and check them.

    Parameters
    ----------
    config_path : Path
        Path of the YAML file, with ``default_timeout_s`` and ``timeouts``.
    notebooks : list of str
        Notebooks of the table of contents, as returned by :func:`list_notebooks`.

    Returns
    -------
    TimeoutConfig
        The default and the per-notebook timeouts.

    Raises
    ------
    ValueError
        If ``default_timeout_s`` is missing or not positive, a timeout is not
        positive, or a path under ``timeouts`` is not a notebook of the table of
        contents (a typo would otherwise leave a notebook on the default unnoticed).
    """
    document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    default = document.get("default_timeout_s")
    if (
        not isinstance(default, (int, float))
        or isinstance(default, bool)
        or default <= 0
    ):
        raise ValueError(f"{config_path}: default_timeout_s must be a positive number")
    raw = document.get("timeouts") or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path}: timeouts must be a mapping")
    unknown = sorted(set(raw) - set(notebooks))
    if unknown:
        raise ValueError(
            f"{config_path}: timeouts names {unknown}, which are not notebooks of the "
            f"table of contents ({notebooks})"
        )
    for path, value in raw.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{config_path}: timeout of {path!r} must be positive")
    return TimeoutConfig(
        default_timeout_s=float(default),
        timeouts={path: float(value) for path, value in raw.items()},
    )


def remaining_cell_timeout(
    budget_s: float, elapsed_s: float, minimum_s: float = MIN_CELL_TIMEOUT_S
) -> float:
    """Return the time to give the next cell so that the notebook stays in budget.

    Parameters
    ----------
    budget_s : float
        Budget of the whole notebook, in seconds.
    elapsed_s : float
        Time already spent on this notebook, in seconds.
    minimum_s : float, optional
        Floor of the result. It must be positive because nbclient reads zero or a
        negative value as "no limit".

    Returns
    -------
    float
        ``budget_s - elapsed_s``, but never less than ``minimum_s``, in seconds.

    Examples
    --------
    >>> remaining_cell_timeout(300.0, 100.0)
    200.0
    >>> remaining_cell_timeout(2.0, 5.0)
    1.0
    """
    return max(budget_s - elapsed_s, minimum_s)


def execute_notebook(source: Path, executed: Path, timeout_s: float) -> NotebookResult:
    """Execute one notebook and write the executed copy.

    The kernel starts in the notebook's own folder. The source file is only read.
    The kernel is shut down after a pass, an error or a timeout.

    Parameters
    ----------
    source : Path
        Notebook to execute.
    executed : Path
        Where the executed copy is written; parent folders are created. The copy is
        written after a failure too, with the outputs produced up to the failure.
    timeout_s : float
        Budget of the whole notebook, in seconds, kernel start-up included.

    Returns
    -------
    NotebookResult
        Status, wall time and, on failure, the error message.
    """
    notebook = nbformat.read(source, as_version=4)
    started = time.monotonic()

    def cell_timeout(_cell: Any) -> float:
        return remaining_cell_timeout(timeout_s, time.monotonic() - started)

    client = NotebookClient(
        notebook,
        timeout_func=cell_timeout,
        resources={"metadata": {"path": str(source.parent)}},
    )
    status = STATUS_PASSED
    error: str | None = None
    try:
        client.execute()
    except CellTimeoutError as exc:
        status, error = STATUS_TIMEOUT, f"cell timed out: {exc}"
    except CellExecutionError as exc:
        status, error = STATUS_ERROR, f"{exc.ename}: {exc.evalue}"
    except Exception as exc:  # noqa: BLE001 - any other failure is a failed notebook
        status, error = STATUS_ERROR, f"{type(exc).__name__}: {exc}"
    wall_time_s = time.monotonic() - started
    if status == STATUS_PASSED and wall_time_s > timeout_s:
        status = STATUS_TIMEOUT
        error = f"wall time {wall_time_s:.1f} s exceeds the budget of {timeout_s:g} s"
    executed.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, executed)
    logger.info(
        "%s: %s in %.2f s (budget %g s)", source, status, wall_time_s, timeout_s
    )
    return NotebookResult(
        status=status, wall_time_s=wall_time_s, timeout_s=timeout_s, error=error
    )


def reset_timings(timings_path: Path) -> None:
    """Empty the timings file so that it never holds the result of an earlier run.

    Parameters
    ----------
    timings_path : Path
        JSON file to write; parent folders are created.
    """
    timings_path.parent.mkdir(parents=True, exist_ok=True)
    timings_path.write_text("{}\n", encoding="utf-8")


def record_result(timings_path: Path, notebook: str, result: NotebookResult) -> None:
    """Add the result of one notebook to the timings file.

    Parameters
    ----------
    timings_path : Path
        JSON object keyed by the notebook's path; created when absent.
    notebook : str
        Path of the notebook relative to the repository root, POSIX form.
    result : NotebookResult
        Outcome to record. The file keeps ``status``, ``wall_time_s`` and
        ``timeout_s``, plus ``error`` for a failure.
    """
    existing: dict[str, Any] = {}
    if timings_path.is_file():
        existing = json.loads(timings_path.read_text(encoding="utf-8") or "{}")
    entry = asdict(result)
    if entry["error"] is None:
        del entry["error"]
    existing[notebook] = entry
    timings_path.parent.mkdir(parents=True, exist_ok=True)
    timings_path.write_text(
        json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
