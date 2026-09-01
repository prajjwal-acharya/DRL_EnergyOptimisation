"""Run the Week 2 deterministic baselines from the frozen config.

Plan reference: docs/plans/week2-implementation-plan.md §D1–§D2. Loads
``configs/week2-baselines.yaml`` (single source of truth — no constants here),
constructs the B0/B1/B2 controllers, and executes each requested
controller × window combination to terminal, persisting the complete artifact
set (run metadata, trace, district KPIs, derived §B4 metrics, run note) under
``results/runs/baselines/<controller>/<window>/``.

Examples:
    ./.venv/bin/python scripts/06_run_baselines.py --window dev --controllers b0_neutral
    ./.venv/bin/python scripts/06_run_baselines.py --window dev final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energy_optimisation.baselines.controllers import (
    FixedScheduleController,
    NeutralController,
    TariffAwareController,
)
from energy_optimisation.evaluation.runner import run_and_record

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/week2-baselines.yaml"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the frozen Week 2 baseline config",
    )
    parser.add_argument(
        "--window",
        action="append",
        dest="windows",
        metavar="NAME",
        help="Window name(s) from the config to run (repeatable; default: all)",
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        dest="controllers",
        metavar="NAME",
        default=None,
        help="Controller name(s) to run (default: all three)",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open() as file:
        return yaml.safe_load(file)


def build_controllers(config: Mapping[str, Any]) -> Dict[str, object]:
    """Instantiate the three baselines strictly from the frozen config values."""

    b1 = config["b1_fixed_schedule"]
    b2 = config["b2_tariff_aware"]
    controllers = [
        NeutralController(),
        FixedScheduleController(
            electrical_storage_schedule=[
                tuple(band) for band in b1["electrical_storage"]["bands"]
            ],
            dhw_storage_schedule=[tuple(band) for band in b1["dhw_storage"]["bands"]],
            cooling_device_schedule=[tuple(band) for band in b1["cooling_device"]["bands"]],
            electrical_storage_default_level=float(
                b1["electrical_storage"]["default_level"]
            ),
            dhw_storage_default_level=float(b1["dhw_storage"]["default_level"]),
            cooling_device_default_level=float(b1["cooling_device"]["default_level"]),
        ),
        TariffAwareController(
            tariff_threshold_usd_per_kwh=float(config["tariff_threshold_usd_per_kwh"]),
            reserve_low_soc=float(config["reserve_low_soc"]),
            reserve_high_soc=float(config["reserve_high_soc"]),
            peak_discharge_level=float(b2["peak_discharge_level"]),
            off_peak_charge_level=float(b2["off_peak_charge_level"]),
            peak_cooling_device_level=float(b2["peak_cooling_device_level"]),
            # Off-peak cooling follows the same frozen hour bands as B1 (§C2).
            cooling_device_schedule=[
                tuple(band) for band in b1["cooling_device"]["bands"]
            ],
            cooling_device_default_level=float(
                b1["cooling_device"]["default_level"]
            ),
        ),
    ]
    return {controller.name: controller for controller in controllers}


def derived_metric_thresholds(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "comfort_band_c": float(config["comfort_band_c"]),
        "reserve_low_soc": float(config["reserve_low_soc"]),
        "reserve_high_soc": float(config["reserve_high_soc"]),
        "grid_limit_kw": (
            None if config.get("grid_limit_p_max") is None else float(config["grid_limit_p_max"])
        ),
    }


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)

    windows = arguments.windows or list(config["windows"])
    unknown_windows = [name for name in windows if name not in config["windows"]]
    if unknown_windows:
        raise SystemExit(f"Unknown window name(s) in config: {', '.join(unknown_windows)}")

    controllers = build_controllers(config)
    selected = arguments.controllers or list(controllers)
    unknown_controllers = [name for name in selected if name not in controllers]
    if unknown_controllers:
        raise SystemExit(f"Unknown controller(s): {', '.join(unknown_controllers)}")

    schema_path = PROJECT_ROOT / config["schema_path"]
    thresholds = derived_metric_thresholds(config)
    baselines_root = PROJECT_ROOT / config["outputs"]["baselines_root"]

    for window_name in windows:
        bounds = config["windows"][window_name]
        for controller_name in selected:
            output_directory = baselines_root / controller_name / window_name
            kpis, _ = run_and_record(
                controllers[controller_name],
                schema_path,
                output_directory=output_directory,
                simulation_start_time_step=int(bounds["simulation_start_time_step"]),
                simulation_end_time_step=int(bounds["simulation_end_time_step"]),
                seed=int(config["random_seed"]),
                purpose="research result",
                config=config,
                derived_metric_thresholds=thresholds,
            )
            print(
                f"Ran {controller_name} on {window_name} "
                f"({bounds['simulation_start_time_step']}-"
                f"{bounds['simulation_end_time_step']}): cost_total={kpis['cost_total']}"
            )


if __name__ == "__main__":
    main()
