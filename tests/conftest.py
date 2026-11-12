"""Fixtures shared by the test files.

``case_blind_disk`` imitates a disk that ignores case, so that a test of the exact-name
rules means the same on the Mac of the lab (APFS ignores case by default) and on the Linux
runner (which tells case apart).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest


def _fold_to_disk(path: Path | str) -> Path | None:
    """Return the real path that ``path`` names on a disk that ignores case, or None."""
    wanted = Path(path)
    current = Path(wanted.anchor)
    for part in wanted.parts[1:]:
        try:
            names = os.listdir(current)
        except OSError:
            return None
        if part in names:
            current = current / part
            continue
        folded = [name for name in names if name.casefold() == part.casefold()]
        if not folded:
            return None
        current = current / folded[0]
    return current


@pytest.fixture
def case_blind_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the file system behave as APFS does by default: it ignores case.

    The Mac of the lab has such a disk and the Linux runner has not, so a test that
    depends on it has to mean the same on both. ``Path.resolve`` keeps the spelling it
    was given on such a disk (measured on APFS), so it is left alone; the probes that
    answer "does this exist" and "is this the same file" ignore case.
    """
    real_is_file = Path.is_file
    real_is_dir = Path.is_dir
    real_exists = Path.exists
    real_samefile = os.path.samefile

    def probe(real: Any) -> Any:
        def wrapped(self: Path) -> bool:
            folded = _fold_to_disk(self)
            return folded is not None and real(folded)

        return wrapped

    def samefile(first: Any, second: Any) -> bool:
        one, two = _fold_to_disk(first), _fold_to_disk(second)
        return one is not None and two is not None and real_samefile(one, two)

    monkeypatch.setattr(Path, "is_file", probe(real_is_file))
    monkeypatch.setattr(Path, "is_dir", probe(real_is_dir))
    monkeypatch.setattr(Path, "exists", probe(real_exists))
    monkeypatch.setattr(os.path, "samefile", samefile)
