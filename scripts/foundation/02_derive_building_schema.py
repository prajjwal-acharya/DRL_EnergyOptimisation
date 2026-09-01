"""Create the Day 3 Building 1-only CityLearn schema from the read-only parent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from energy_optimisation.environment import create_single_building_schema

DEFAULT_PARENT = PROJECT_ROOT / "data/raw/citylearn_challenge_2023_phase_1/schema.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "configs/schema-building1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-schema", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--building", default="Building_1")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    output = create_single_building_schema(
        arguments.parent_schema,
        arguments.building,
        arguments.output,
        project_root=PROJECT_ROOT,
    )
    print(f"Created single-building schema: {output}")


if __name__ == "__main__":
    main()
