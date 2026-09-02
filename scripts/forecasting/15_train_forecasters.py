"""Run the frozen Week-4 rolling-origin model-ladder backtest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energy_optimisation.forecasting.pipeline import run_backtest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/week4-forecasting.yaml"
    )
    args = parser.parse_args()
    metadata = run_backtest(args.config, PROJECT_ROOT)
    print(
        "Week-4 backtest complete: "
        f"{metadata['evaluated_pairs_per_model_target']} valid OOS pairs per model/target, "
        f"{metadata['wall_clock_seconds']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
