"""Fetch pinned CityLearn source data and prime the local misc-data cache."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from citylearn.data import DataSet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CITYLEARN_REPOSITORY = "https://github.com/citylearn-project/CityLearn.git"
CITYLEARN_TAG = "v2.5.0"
CITYLEARN_COMMIT = "29062af6d077409e1c37a3e53a6cac30fd4d02bc"
REQUIRED_MISC_FILES = ("battery_choices.yaml", "lbl-tracking_the_sun-res-pv.csv")


def clone_source_if_needed(source_directory: Path) -> None:
    """Clone the exact CityLearn tag when source data is not already present."""

    if source_directory.exists():
        if not (source_directory / ".git").is_dir():
            raise RuntimeError(f"Source directory exists but is not a Git checkout: {source_directory}")
        return

    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", CITYLEARN_TAG, CITYLEARN_REPOSITORY, str(source_directory)],
        check=True,
    )


def verify_source_version(source_directory: Path) -> None:
    """Fail if the local CityLearn source is not the version used by this project."""

    commit = subprocess.check_output(
        ["git", "-C", str(source_directory), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != CITYLEARN_COMMIT:
        raise RuntimeError(
            f"Expected CityLearn commit {CITYLEARN_COMMIT}, found {commit}. "
            "Use the v2.5.0 tag required by configs/smoke.yaml."
        )


def prime_misc_cache(source_directory: Path) -> Path:
    """Copy CityLearn's required PV and battery reference files into its user cache."""

    source_misc = source_directory / "data/misc"
    cache_misc = Path(DataSet().cache_directory) / "misc"
    cache_misc.mkdir(parents=True, exist_ok=True)

    for filename in REQUIRED_MISC_FILES:
        source = source_misc / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing CityLearn misc file: {source}")
        shutil.copy2(source, cache_misc / filename)

    return cache_misc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-directory",
        type=Path,
        default=PROJECT_ROOT / "data/raw/citylearn-2.5.0",
    )
    parser.add_argument("--skip-clone", action="store_true")
    arguments = parser.parse_args()
    source_directory = arguments.source_directory.resolve()

    if arguments.skip_clone and not source_directory.exists():
        raise FileNotFoundError(f"Cannot skip clone; source directory is absent: {source_directory}")
    if not arguments.skip_clone:
        clone_source_if_needed(source_directory)

    verify_source_version(source_directory)
    cache_misc = prime_misc_cache(source_directory)
    print(f"CityLearn source verified at {source_directory}")
    print(f"CityLearn misc cache primed at {cache_misc}")


if __name__ == "__main__":
    main()
