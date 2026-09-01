"""Verify — and if absent, fetch — the pinned CityLearn 2023 phase 1 dataset.

Only the dataset directory this project consumes is stored under ``data/raw/``:
on fetch, the full CityLearn repository is cloned to a throwaway temporary
directory, verified against the pinned commit, and only the
``citylearn_challenge_2023_phase_1`` payload is copied in. The CityLearn Python
package itself comes from pip (``CityLearn==2.5.0`` in requirements.txt), never
from these files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

from citylearn.data import DataSet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
CITYLEARN_REPOSITORY = "https://github.com/citylearn-project/CityLearn.git"
CITYLEARN_TAG = "v2.5.0"

CITYLEARN_COMMIT = "29062af6d077409e1c37a3e53a6cac30fd4d02bc"
DATASET_NAME = "citylearn_challenge_2023_phase_1"
PROVENANCE_FILENAME = "PROVENANCE.txt"
PROVENANCE_TEMPLATE = """{dataset} — pinned data payload

Source repository : {repository}
Source tag        : {tag}
Source commit     : {commit}
Extracted         : {date} (scripts/foundation/01_fetch_pinned_dataset.py fetch mode)

Only this dataset directory is kept from the tagged clone; the full repository
is never stored in this project. The CityLearn Python package itself comes
from pip (CityLearn==2.5.0 in requirements.txt), not from these files.
"""

# The exact file set of the pinned dataset payload (verified against the tag).
REQUIRED_FILES = (
    "Building_1.csv",
    "Building_1.pth",
    "Building_2.csv",
    "Building_2.pth",
    "Building_3.csv",
    "Building_3.pth",
    "carbon_intensity.csv",
    "pricing.csv",
    "schema.json",
    "weather.csv",
)

# Every CSV carries a header plus 720 hourly simulation steps.
REQUIRED_CSV_FILES = (
    "Building_1.csv",
    "Building_2.csv",
    "Building_3.csv",
    "carbon_intensity.csv",
    "pricing.csv",
    "weather.csv",
)
EXPECTED_CSV_LINES = 721

# CityLearn reference files its autosizing features may download on demand;
# primed into the user cache during fetch so runs never touch the network.
REQUIRED_MISC_FILES = ("battery_choices.yaml", "lbl-tracking_the_sun-res-pv.csv")


def verify_dataset(dataset_directory: Path) -> None:
    """Fail with a clear message if the local payload is not the pinned set."""

    if not dataset_directory.is_dir():
        raise FileNotFoundError(f"Dataset directory is absent: {dataset_directory}")

    missing = [name for name in REQUIRED_FILES if not (dataset_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset is missing pinned file(s): {', '.join(missing)}")

    for name in REQUIRED_CSV_FILES:
        line_count = sum(1 for _ in (dataset_directory / name).open())
        if line_count != EXPECTED_CSV_LINES:
            raise RuntimeError(
                f"{name} has {line_count} lines; expected {EXPECTED_CSV_LINES} "
                "(header + 720 hourly steps) for the pinned v2.5.0 payload."
            )

    schema = json.loads((dataset_directory / "schema.json").read_text())
    buildings = list(schema.get("buildings", {}))
    if sorted(buildings) != ["Building_1", "Building_2", "Building_3"]:
        raise RuntimeError(f"Parent schema buildings differ from the pinned set: {buildings}")

    if not (dataset_directory / PROVENANCE_FILENAME).is_file():
        raise FileNotFoundError(
            f"Missing {PROVENANCE_FILENAME}; run with --fetch to restore a verified payload."
        )


def fetch_dataset(dataset_directory: Path) -> Path:
    """Clone the pinned tag to a temp dir, extract the dataset, prime the cache."""

    dataset_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="citylearn-v2.5.0-") as temporary:
        clone = Path(temporary) / "repo"
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                CITYLEARN_TAG,
                CITYLEARN_REPOSITORY,
                str(clone),
            ],
            check=True,
        )
        commit = subprocess.check_output(
            ["git", "-C", str(clone), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != CITYLEARN_COMMIT:
            raise RuntimeError(
                f"Expected CityLearn commit {CITYLEARN_COMMIT}, found {commit}. "
                f"The {CITYLEARN_TAG} tag moved; pin the new commit explicitly."
            )

        source_dataset = clone / "data/datasets" / DATASET_NAME
        if not source_dataset.is_dir():
            raise FileNotFoundError(f"Dataset absent from the tagged clone: {source_dataset}")

        if dataset_directory.exists():
            shutil.rmtree(dataset_directory)
        shutil.copytree(source_dataset, dataset_directory)

        cache_misc = Path(DataSet().cache_directory) / "misc"
        cache_misc.mkdir(parents=True, exist_ok=True)
        for filename in REQUIRED_MISC_FILES:
            source = clone / "data/misc" / filename
            if not source.is_file():
                raise FileNotFoundError(f"Missing CityLearn misc file: {source}")
            shutil.copy2(source, cache_misc / filename)

    provenance = dataset_directory / PROVENANCE_FILENAME
    provenance.write_text(
        PROVENANCE_TEMPLATE.format(
            dataset=DATASET_NAME,
            repository=CITYLEARN_REPOSITORY,
            tag=CITYLEARN_TAG,
            commit=CITYLEARN_COMMIT,
            date=__import__("datetime").date.today().isoformat(),
        )
    )
    return dataset_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-directory",
        type=Path,
        default=PROJECT_ROOT / "data/raw" / DATASET_NAME,
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Re-fetch the pinned payload from the tagged CityLearn repository "
        "(overwrites the local dataset directory).",
    )
    arguments = parser.parse_args()
    dataset_directory = arguments.dataset_directory.resolve()

    if arguments.fetch or not dataset_directory.exists():
        fetch_dataset(dataset_directory)
    verify_dataset(dataset_directory)
    print(f"Pinned dataset verified at {dataset_directory}")


if __name__ == "__main__":
    main()
