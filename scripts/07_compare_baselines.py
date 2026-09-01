"""Compare recorded baseline runs for one or more windows.

Plan reference: docs/plans/week2-implementation-plan.md §B5. Reads every run
directory under ``results/runs/baselines/<controller>/<window>/`` and writes:

- ``results/tables/baseline_comparison.csv`` (rows = controllers per window,
  columns = fixed primary KPI set), and
- four figures per window under ``results/figures/``: cost-by-controller bar,
  48-hour net-demand overlay, electrical SoC trace, indoor temperature vs
  cooling setpoint trace.

Example:
    ./.venv/bin/python scripts/07_compare_baselines.py --window 0-167 0-719
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energy_optimisation.evaluation.artifacts import DISTRICT_KPIS_FILE, TRACE_FILE


PRIMARY_KPIS = (
    "cost_total",
    "all_time_peak_average",
    "electricity_consumption_total",
    "discomfort_hot_proportion",
    "discomfort_proportion",
    "ramping_average",
    "zero_net_energy",
)

DEFAULT_BASELINES_ROOT = PROJECT_ROOT / "results/runs/baselines"
DEFAULT_OUTPUT_TABLE = PROJECT_ROOT / "results/tables/baseline_comparison.csv"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results/figures"
OVERLAY_HOURS = 48


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baselines-root",
        type=Path,
        default=DEFAULT_BASELINES_ROOT,
        help="Directory containing <controller>/<window> run directories",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=DEFAULT_OUTPUT_TABLE,
        help="Destination CSV for the comparison table",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Destination directory for comparison figures",
    )
    parser.add_argument(
        "--window",
        action="append",
        default=None,
        dest="windows",
        metavar="WINDOW",
        help="Window label(s) to compare, e.g. 0-167 (repeatable)",
    )
    arguments = parser.parse_args()
    if not arguments.windows:
        arguments.windows = ["0-167"]
    return arguments


def load_run(baselines_root: Path, controller: str, window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_directory = baselines_root / controller / window
    kpis_path = run_directory / DISTRICT_KPIS_FILE
    trace_path = run_directory / TRACE_FILE
    if not kpis_path.is_file():
        raise FileNotFoundError(f"Missing district KPIs for comparison: {kpis_path}")
    if not trace_path.is_file():
        raise FileNotFoundError(f"Missing trace for comparison: {trace_path}")
    return pd.read_csv(kpis_path, float_precision="round_trip"), pd.read_csv(trace_path)


def build_comparison_rows(
    baselines_root: Path,
    controllers: List[str],
    windows: List[str],
) -> List[Dict[str, object]]:
    rows = []
    for window in windows:
        for controller in controllers:
            kpis_frame, _ = load_run(baselines_root, controller, window)
            values = kpis_frame.set_index("cost_function")["value"]
            row: Dict[str, object] = {"controller": controller, "window": window}
            for kpi in PRIMARY_KPIS:
                value = values.get(kpi, float("nan"))
                row[kpi] = None if pd.isna(value) else float(value)
            rows.append(row)
    return rows


def write_figures(
    baselines_root: Path,
    controllers: List[str],
    windows: List[str],
    figures_dir: Path,
) -> List[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for window in windows:
        runs = {controller: load_run(baselines_root, controller, window) for controller in controllers}
        costs = {
            controller: float(frame.set_index("cost_function")["value"]["cost_total"])
            for controller, (frame, _) in runs.items()
        }

        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(list(costs), list(costs.values()), color="#2b6cb0")
        axis.set_title(f"cost_total by controller ({window})")
        axis.set_ylabel("Normalised total cost")
        figure.tight_layout()
        path = figures_dir / f"{window}_cost_by_controller.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        written.append(path)

        figure, axis = plt.subplots(figsize=(9, 4.5))
        for controller, (_, trace) in runs.items():
            overlay = trace.head(OVERLAY_HOURS)
            axis.plot(overlay["timestep"], overlay["net_electricity_consumption"], label=controller)
        axis.set_title(f"Net electricity consumption, first {OVERLAY_HOURS} hours ({window})")
        axis.set_xlabel("Timestep")
        axis.set_ylabel("Net electricity consumption")
        axis.legend()
        figure.tight_layout()
        path = figures_dir / f"{window}_net_demand_overlay_{OVERLAY_HOURS}h.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        written.append(path)

        figure, axis = plt.subplots(figsize=(9, 4.5))
        for controller, (_, trace) in runs.items():
            axis.plot(trace["timestep"], trace["electrical_storage_soc"], label=controller)
        axis.set_title(f"Electrical storage SoC ({window})")
        axis.set_xlabel("Timestep")
        axis.set_ylabel("SoC (fraction)")
        axis.legend()
        figure.tight_layout()
        path = figures_dir / f"{window}_electrical_soc_trace.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        written.append(path)

        figure, axis = plt.subplots(figsize=(9, 4.5))
        for index, (controller, (_, trace)) in enumerate(runs.items()):
            axis.plot(
                trace["timestep"],
                trace["indoor_dry_bulb_temperature"],
                label=f"{controller} indoor",
                color=f"C{index}",
            )
            if index == 0:
                axis.plot(
                    trace["timestep"],
                    trace["indoor_dry_bulb_temperature_cooling_set_point"],
                    label="cooling setpoint",
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                )
        axis.set_title(f"Indoor temperature vs cooling setpoint ({window})")
        axis.set_xlabel("Timestep")
        axis.set_ylabel("Dry-bulb temperature (°C)")
        axis.legend(fontsize=8)
        figure.tight_layout()
        path = figures_dir / f"{window}_indoor_temperature_vs_setpoint.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        written.append(path)

    return written


def main() -> None:
    arguments = parse_arguments()

    if not arguments.baselines_root.is_dir():
        raise SystemExit(f"Baselines root does not exist: {arguments.baselines_root}")

    controllers = sorted(
        entry.name
        for entry in arguments.baselines_root.iterdir()
        if entry.is_dir()
        and all((entry / window / DISTRICT_KPIS_FILE).is_file() for window in arguments.windows)
    )
    if not controllers:
        raise SystemExit(
            f"No controller runs with artifacts for window(s): {', '.join(arguments.windows)}"
        )

    rows = build_comparison_rows(arguments.baselines_root, controllers, arguments.windows)
    table = pd.DataFrame(rows, columns=["controller", "window", *PRIMARY_KPIS])
    arguments.output_table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(arguments.output_table, index=False, float_format="%.17g")

    figures = write_figures(arguments.baselines_root, controllers, arguments.windows, arguments.figures_dir)

    print(f"Controllers compared: {', '.join(controllers)}")
    print(f"Wrote comparison table: {arguments.output_table}")
    for path in figures:
        print(f"Wrote figure: {path}")


if __name__ == "__main__":
    main()
