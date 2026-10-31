"""Download the open datasets used by the fNIRS tutorial series.

The FRESH datasets (Yücel et al. 2025, Commun Biol 8:1149,
doi:10.1038/s42003-025-08412-1) are fetched from the OSF project
https://osf.io/b4wck as two zip archives and unpacked under ``data/fresh/``.
The MNE, MNE-NIRS and Cedalion example datasets download themselves on first
use through their own fetchers and are not handled here.

When to use
-----------
Run once after creating the environment, before executing any notebook that
reads FRESH data. Re-running is safe: archives already present with the right
checksum are not downloaded again.

Example
-------
    uv run python scripts/download_data.py --data-dir data
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("download_data")

# Download chunk size; large enough to keep the loop cheap, small enough to log progress.
CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class Archive:
    """One downloadable archive with its expected size and checksum."""

    name: str
    url: str
    size_bytes: int
    sha256: str | None


# Sizes come from the OSF API listing of project b4wck (2026-10-31).
# Checksums were computed on the first download (2026-10-31); None means not yet recorded.
FRESH_ARCHIVES = (
    Archive(
        "FRESH-Audio.zip",
        "https://osf.io/download/xd2qm/",
        57_527_703,
        "cc9c91c9b55f6c8645949f3b062542bb24c9a769d637294873d5a2c767d330d2",
    ),
    Archive(
        "FRESH-Motor.zip",
        "https://osf.io/download/juxy8/",
        208_284_182,
        "660d01e03e7c9b7bd378e058af40c94ac37a8a980ac164f50c0b3e7c878da692",
    ),
)


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path: Path, archive: Archive) -> bool:
    """Check an existing file against the expected size and, if known, checksum."""
    if not path.exists() or path.stat().st_size != archive.size_bytes:
        return False
    if archive.sha256 is None:
        logger.warning(
            "No recorded checksum for %s; size matches, accepting it.", archive.name
        )
        return True
    return sha256_of(path) == archive.sha256


def download(archive: Archive, target: Path) -> None:
    """Stream one archive to disk, logging progress every 10 %."""
    logger.info("Downloading %s from %s", archive.name, archive.url)
    tmp = target.with_suffix(target.suffix + ".part")
    next_report = 0.1
    with urllib.request.urlopen(archive.url) as response, tmp.open("wb") as out:
        written = 0
        for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
            out.write(chunk)
            written += len(chunk)
            if written / archive.size_bytes >= next_report:
                logger.info(
                    "  %s: %.0f %%", archive.name, 100 * written / archive.size_bytes
                )
                next_report += 0.1
    tmp.rename(target)


def fetch_fresh(data_dir: Path) -> None:
    """Download and unpack both FRESH archives into ``data_dir / "fresh"``."""
    fresh_dir = data_dir / "fresh"
    fresh_dir.mkdir(parents=True, exist_ok=True)
    for archive in FRESH_ARCHIVES:
        target = fresh_dir / archive.name
        if is_valid(target, archive):
            logger.info("%s already present and valid.", archive.name)
        else:
            download(archive, target)
            if not is_valid(target, archive):
                raise RuntimeError(
                    f"{archive.name}: size or checksum mismatch after download"
                )
        logger.info("%s sha256 = %s", archive.name, sha256_of(target))
        with zipfile.ZipFile(target) as zf:
            # The OSF archives were zipped on macOS and carry __MACOSX/ resource-fork copies,
            # including fake *.snirf files; they are skipped so that globbing stays safe.
            members = [m for m in zf.namelist() if not m.startswith("__MACOSX/")]
            zf.extractall(fresh_dir, members=members)
        logger.info("Unpacked %s into %s", archive.name, fresh_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="root data directory"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    fetch_fresh(args.data_dir)


if __name__ == "__main__":
    main()
