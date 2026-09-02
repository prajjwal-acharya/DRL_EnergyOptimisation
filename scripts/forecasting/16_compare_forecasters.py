"""Apply the frozen Week-4 selection rule and build reporting artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energy_optimisation.forecasting.pipeline import compare_and_select


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/week4-forecasting.yaml"
    )
    args = parser.parse_args()
    selection = compare_and_select(args.config, PROJECT_ROOT)
    for target, record in selection["targets"].items():
        print(f"{target}: {record['selected_variant']} — {record['rule_trace']['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
