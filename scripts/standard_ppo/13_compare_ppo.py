"""Phase D locked comparison tables: multi-seed summary and PPO vs baselines.

Plan reference: docs/plans/week3-implementation-plan.md §D1–§D2. Reads the recorded
week-2 baseline table (read-only) and the Phase C/D PPO artifacts and writes:

- ``results/tables/ppo_multiseed_summary.csv`` — per KPI mean/min/max across
  the three frozen-rule-selected checkpoints (dev window);
- ``results/tables/ppo_vs_baselines.csv`` — one row per controller per window
  (rows B0, B1, B2, PPO-seed42, PPO-seed43, PPO-seed44 × windows dev/final)
  over the fixed week-2 primary-KPI column set;
- ``results/figures/ppo_vs_baselines_cost.png`` — cost_total by controller,
  both windows.

``results/tables/baseline_comparison.csv`` is consumed read-only and must stay
byte-identical to its week-2 state (14_gate_week3.py enforces this against git).

Usage:
    ./.venv/bin/python scripts/standard_ppo/13_compare_ppo.py --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from energy_optimisation.evaluation.artifacts import DISTRICT_KPIS_FILE
from energy_optimisation.rl import EVALUATION_COLUMNS

BASELINE_TABLE = PROJECT_ROOT / "results/tables/baseline_comparison.csv"
MULTISEED_TABLE = PROJECT_ROOT / "results/tables/ppo_multiseed_summary.csv"
PPO_VS_BASELINES_TABLE = PROJECT_ROOT / "results/tables/ppo_vs_baselines.csv"
COST_FIGURE = PROJECT_ROOT / "results/figures/ppo_vs_baselines_cost.png"

WINDOWS = ("dev", "final")
# Fixed week-2 primary-KPI column set (scripts/cmdp_baselines/07_compare_baselines.py order).
PRIMARY_KPIS = (
    "cost_total",
    "all_time_peak_average",
    "electricity_consumption_total",
    "discomfort_hot_proportion",
    "discomfort_proportion",
    "ramping_average",
    "zero_net_energy",
)
BASELINE_LABELS = {
    "b0_neutral": "B0",
    "b1_fixed_schedule": "B1",
    "b2_tariff_aware": "B2",
}
SUMMARY_KPIS = tuple(
    column for column in EVALUATION_COLUMNS if column not in ("checkpoint", "timestep")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(42, 43, 44),
        help="Seed directories results/runs/ppo/seed<seed> to aggregate.",
    )
    return parser.parse_args()


def load_selected_row(output_dir: Path, seed: int) -> pd.Series:
    """The selected checkpoint's full KPI row from ``evaluations.csv``."""

    selection_path = output_dir / "selected_checkpoint.json"
    if not selection_path.is_file():
        raise SystemExit(f"missing selection record: {selection_path}")
    with selection_path.open() as handle:
        record = json.load(handle)
    selected_name = str(record["selected_checkpoint"])
    evaluations_path = output_dir / "evaluations.csv"
    if not evaluations_path.is_file():
        raise SystemExit(f"missing evaluations table: {evaluations_path}")
    frame = pd.read_csv(evaluations_path)
    matching = frame.loc[frame["checkpoint"] == selected_name]
    if len(matching) != 1:
        raise SystemExit(
            f"{evaluations_path}: expected exactly one row for {selected_name}, found {len(matching)}"
        )
    return matching.iloc[0]


def build_multiseed_summary(seeds: List[int]) -> pd.DataFrame:
    """Per-KPI mean/min/max across the three selected checkpoints (dev)."""

    per_seed: Dict[str, List[float]] = {kpi: [] for kpi in SUMMARY_KPIS}
    for seed in seeds:
        output_dir = PROJECT_ROOT / "results" / "runs" / "ppo" / f"seed{seed}"
        row = load_selected_row(output_dir, seed)
        for kpi in SUMMARY_KPIS:
            per_seed[kpi].append(float(row[kpi]))

    rows: List[Dict[str, Any]] = []
    for kpi in SUMMARY_KPIS:
        values = per_seed[kpi]
        series = pd.Series(values, dtype=float)
        rows.append(
            {
                "kpi": kpi,
                **{f"ppo_seed{seed}": value for seed, value in zip(seeds, values)},
                "mean": float(series.mean()),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(rows, columns=["kpi", *[f"ppo_seed{seed}" for seed in seeds], "mean", "min", "max"])


def load_baseline_rows() -> List[Dict[str, Any]]:
    """Week-2 comparison rows relabelled B0/B1/B2 (source stays untouched)."""

    if not BASELINE_TABLE.is_file():
        raise SystemExit(f"missing week-2 baseline table: {BASELINE_TABLE}")
    frame = pd.read_csv(BASELINE_TABLE)
    rows: List[Dict[str, Any]] = []
    for _, record in frame.iterrows():
        label = BASELINE_LABELS.get(str(record["controller"]))
        if label is None:
            raise SystemExit(f"unexpected controller in {BASELINE_TABLE}: {record['controller']}")
        row: Dict[str, Any] = {
            "controller": label,
            "window": str(record["window"]),
        }
        for kpi in PRIMARY_KPIS:
            value = record.get(kpi, float("nan"))
            row[kpi] = None if pd.isna(value) else float(value)
        rows.append(row)
    return rows


def load_ppo_rows(seeds: List[int]) -> List[Dict[str, Any]]:
    """Dev-window selected-checkpoint KPIs plus final-window run KPIs."""

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        output_dir = PROJECT_ROOT / "results" / "runs" / "ppo" / f"seed{seed}"
        selected = load_selected_row(output_dir, seed)
        dev_row: Dict[str, Any] = {"controller": f"PPO-seed{seed}", "window": "dev"}
        for kpi in PRIMARY_KPIS:
            dev_row[kpi] = float(selected[kpi])
        rows.append(dev_row)

        final_kpis_path = output_dir / "final" / DISTRICT_KPIS_FILE
        if not final_kpis_path.is_file():
            raise SystemExit(f"missing final-window district KPIs: {final_kpis_path}")
        kpis_frame = pd.read_csv(final_kpis_path)
        values = kpis_frame.set_index("cost_function")["value"]
        final_row: Dict[str, Any] = {"controller": f"PPO-seed{seed}", "window": "final"}
        for kpi in PRIMARY_KPIS:
            value = values.get(kpi, float("nan"))
            final_row[kpi] = None if pd.isna(value) else float(value)
        rows.append(final_row)
    return rows


def plot_cost_figure(rows: List[Dict[str, Any]]) -> Path:
    figure, axis = plt.subplots(figsize=(9, 4.8))
    controllers = ["B0", "B1", "B2", *[f"PPO-seed{seed}" for seed in (42, 43, 44)]]
    width = 0.35
    x = list(range(len(controllers)))
    for offset, window in zip((-width / 2, width / 2), WINDOWS):
        costs = []
        for controller in controllers:
            matching = [row for row in rows if row["controller"] == controller and row["window"] == window]
            if not matching:
                raise SystemExit(f"no {window} row for {controller}")
            costs.append(float(matching[0]["cost_total"]))
        bars = axis.bar([position + offset for position in x], costs, width=width, label=window)
        axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    axis.set_xticks(x)
    axis.set_xticklabels(controllers)
    axis.set_ylabel("normalised cost_total")
    axis.set_title("Week-3 standard PPO vs week-2 baselines: cost_total by controller")
    axis.legend()
    axis.grid(alpha=0.25, axis="y")
    figure.tight_layout()
    COST_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(COST_FIGURE, dpi=150)
    plt.close(figure)
    return COST_FIGURE


def main() -> int:
    args = parse_args()
    started = time.time()
    seeds = [int(seed) for seed in args.seeds]

    # Guardrail: this script only ever reads the week-2 table (BASELINE_TABLE
    # is never a write target); 14_gate_week3.py re-derives its content from the
    # week-2 run artifacts to prove it stayed byte-identical.

    summary = build_multiseed_summary(seeds)
    MULTISEED_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(MULTISEED_TABLE, index=False, float_format="%.17g")

    rows = [*load_baseline_rows(), *load_ppo_rows(seeds)]
    table = pd.DataFrame(rows, columns=["controller", "window", *PRIMARY_KPIS])
    table.to_csv(PPO_VS_BASELINES_TABLE, index=False, float_format="%.17g")

    plot_cost_figure(rows)

    print(f"[gate] multiseed summary ({len(summary)} KPI rows): {MULTISEED_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"[gate] comparison table ({len(table)} rows): {PPO_VS_BASELINES_TABLE.relative_to(PROJECT_ROOT)}")
    print(f"[gate] cost figure: {COST_FIGURE.relative_to(PROJECT_ROOT)}")
    print(f"[gate] wall clock: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
