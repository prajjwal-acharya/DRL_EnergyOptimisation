"""Write Day 2 CityLearn interface evidence to a JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energy_optimisation.environment import inspect_environment


DEFAULT_SCHEMA = PROJECT_ROOT / "data/raw/citylearn_challenge_2023_phase_1/schema.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/inspection/citylearn_2023_phase_1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decentralized", action="store_true")
    arguments = parser.parse_args()

    result = inspect_environment(arguments.schema, central_agent=not arguments.decentralized)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote inspection evidence to {arguments.output}")


if __name__ == "__main__":
    main()
