"""Phase C checkpoint evaluation through the locked week-2 harness.

Plan reference: docs/plans/week3-implementation-plan.md §C2–§C3. Evaluates every PPO
checkpoint of one seed on the requested window (dev 0–167 for learning curves;
final 0–719 in Phase D) through ``energy_optimisation.evaluation.runner`` —
the locked week-2 harness, unchanged — producing the same district-KPI set and
§B4 derived metrics as the baselines.

Artifacts written under ``results/runs/ppo/seed<seed>/``:

- ``evaluations.csv`` — one row per checkpoint, exact plan §C2 column order,
  validated NaN-free before writing;
- ``selected_checkpoint.json`` — the frozen selection rule (lowest
  ``cost_total``, tie-break lower ``discomfort_proportion``) executed on the
  table, recording the chosen checkpoint's full KPI row.

Figure written under ``results/figures/``:
``ppo_seed<seed>_kpi_curves.png`` — cost / discomfort / peak vs training
progress with the B0 dev-window reference lines and the selected checkpoint
marked.

Derived-metric thresholds (comfort band, SoC reserve band, grid limit) are
read from the frozen configs only: comfort from the week-3 CMDP reward block,
reserve band and grid limit from ``configs/week2-baselines.yaml`` (read-only
reference) — no constants in code.

Usage:
    ./.venv/bin/python scripts/11_evaluate_checkpoints.py --seed 42 --window dev
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energy_optimisation.evaluation.metrics import compute_derived_metrics
from energy_optimisation.evaluation.runner import (
    district_kpis_as_dict,
    relative_to_project_root,
    resolve_git_commit,
    run_episode,
)
from energy_optimisation.rl import (
    EVALUATION_COLUMNS,
    SELECTION_RULE_TEXT,
    PPOController,
    episode_return_from_trace,
    reward_constants_from_config,
    select_best_checkpoint,
)

NUMBERED_CHECKPOINT_PATTERN = re.compile(r"^ppo_(\d{8})_steps\.zip$")
KPI_COLUMNS = (
    "cost_total",
    "all_time_peak_average",
    "electricity_consumption_total",
    "discomfort_proportion",
    "discomfort_hot_proportion",
    "ramping_average",
    "zero_net_energy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/week3-ppo.yaml",
        help="Frozen week-3 PPO config.",
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=PROJECT_ROOT / "configs/week2-baselines.yaml",
        help="Read-only week-2 config supplying reserve-band/grid-limit thresholds.",
    )
    parser.add_argument("--seed", type=int, required=True, help="Training seed directory seed<>")
    parser.add_argument(
        "--window",
        choices=("dev", "final"),
        default="dev",
        help="Evaluation window name from the config (plan §C2 uses dev).",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=None,
        help="Checkpoint directory; defaults to results/runs/ppo/seed<seed>/checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact directory; defaults to results/runs/ppo/seed<seed>.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures",
    )
    return parser.parse_args()


def discover_checkpoints(checkpoints_dir: Path) -> List[Dict[str, Any]]:
    """Return numbered checkpoints ascending plus ``final`` last."""

    if not checkpoints_dir.is_dir():
        raise SystemExit(f"checkpoint directory not found: {checkpoints_dir}")
    entries: List[Dict[str, Any]] = []
    for path in sorted(checkpoints_dir.glob("*.zip")):
        match = NUMBERED_CHECKPOINT_PATTERN.match(path.name)
        if match:
            entries.append({"name": path.name, "path": path, "timestep": int(match.group(1))})
    entries.sort(key=lambda entry: entry["timestep"])
    final_path = checkpoints_dir / "final.zip"
    if final_path.is_file():
        entries.append({"name": final_path.name, "path": final_path, "timestep": None})
    if not entries:
        raise SystemExit(f"no checkpoints found under {checkpoints_dir}")
    return entries


def resolve_final_timestep(output_dir: Path) -> int:
    """Total training steps for ``final.zip`` from the phase-B run metadata."""

    metadata_path = output_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise SystemExit(
            f"cannot resolve the timestep of final.zip: {metadata_path} is missing"
        )
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    completed = metadata.get("total_timesteps_completed")
    if completed is None:
        raise SystemExit(f"{metadata_path} lacks 'total_timesteps_completed'")
    return int(completed)


def b0_reference_kpis(baselines_root: Path, window: str) -> Optional[Dict[str, float]]:
    """B0 harness KPIs for the same window (read-only reference; None if absent)."""

    kpis_path = baselines_root / "b0_neutral" / window / "district_kpis.csv"
    if not kpis_path.is_file():
        return None
    frame = pd.read_csv(kpis_path)
    district = frame.loc[frame["level"] == "district"]
    return {
        str(cost_function): float(value)
        for cost_function, value in zip(district["cost_function"], district["value"])
        if pd.notna(value)
    }


def evaluate_one_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    schema_path: Path,
    config: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    reward_constants: Mapping[str, float],
    start_step: int,
    end_step: int,
    seed: int,
    controller_name: str,
) -> Dict[str, Any]:
    controller = PPOController(checkpoint["path"], config, device="cpu", name=controller_name)
    kpis, trace = run_episode(
        controller,
        schema_path,
        simulation_start_time_step=start_step,
        simulation_end_time_step=end_step,
        seed=seed,
    )
    missing_kpis = [name for name in KPI_COLUMNS if name not in kpis]
    if missing_kpis:
        raise SystemExit(
            f"{checkpoint['name']}: harness KPIs lack expected columns: {', '.join(missing_kpis)}"
        )
    derived = compute_derived_metrics(trace, **dict(thresholds))
    row: Dict[str, Any] = {
        "checkpoint": str(checkpoint["name"]),
        "timestep": (
            int(checkpoint["timestep"])
            if checkpoint["timestep"] is not None
            else int(checkpoint["resolved_timestep"])
        ),
        "episode_return": episode_return_from_trace(trace, **dict(reward_constants)),
    }
    row.update({name: float(kpis[name]) for name in KPI_COLUMNS})
    row.update(
        {
            "comfort_violation_hours": int(derived["comfort_violation_hours"]),
            "grid_limit_exceedances": (
                0 if derived["grid_limit_exceedance_count"] is None
                else int(derived["grid_limit_exceedance_count"])
            ),
            "clipping_events": int(derived["clipping_event_count"]),
            "reserve_events": int(derived["reserve_event_count"]),
        }
    )
    return row


def validate_nan_free(frame: pd.DataFrame, label: str) -> None:
    if bool(frame.isna().to_numpy().any()):
        offending = frame.columns[frame.isna().any()].tolist()
        raise SystemExit(f"{label} contains NaN values in columns: {', '.join(offending)}")
    numeric = frame.drop(columns=["checkpoint"])
    if not bool(np.isfinite(numeric.to_numpy(dtype=float)).all()):
        raise SystemExit(f"{label} contains non-finite values")


def plot_kpi_curves(
    frame: pd.DataFrame,
    selected_name: str,
    reference: Optional[Mapping[str, float]],
    figures_dir: Path,
    seed: int,
    window: str,
) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / f"ppo_seed{seed}_kpi_curves.png"
    panels = [
        ("cost_total", "normalised cost_total"),
        ("discomfort_proportion", "discomfort_proportion"),
        ("all_time_peak_average", "all_time_peak_average"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 9), sharex=True)
    x = frame["timestep"].to_numpy(dtype=float)
    selected_row = frame.loc[frame["checkpoint"] == selected_name].iloc[0]
    for axis, (column, label) in zip(axes, panels):
        axis.plot(x, frame[column].to_numpy(dtype=float), marker="o", markersize=3, linewidth=0.9)
        if reference is not None and column in reference:
            axis.axhline(reference[column], color="tab:red", linestyle="--", linewidth=1.0,
                        label=f"B0 ({window})")
            axis.legend(loc="best", fontsize=8)
        axis.scatter([selected_row["timestep"]], [float(selected_row[column])],
                     marker="*", s=160, color="tab:green", zorder=5, label="selected")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
    axes[-1].set_xlabel("training environment steps")
    fig.suptitle(f"PPO seed {seed} KPI learning curves ({window} window, locked harness)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def main() -> int:
    args = parse_args()

    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    with args.baseline_config.open() as handle:
        baseline_config = yaml.safe_load(handle)

    seed = int(args.seed)
    schema_path = PROJECT_ROOT / config["schema_path"]
    bounds = config["windows"][args.window]
    start_step = int(bounds["simulation_start_time_step"])
    end_step = int(bounds["simulation_end_time_step"])

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (PROJECT_ROOT / "outputs" / "ppo" / f"seed{seed}").resolve()
    )
    checkpoints_dir = (
        args.checkpoints_dir.resolve()
        if args.checkpoints_dir is not None
        else output_dir / "checkpoints"
    )

    reward_constants = reward_constants_from_config(config)
    thresholds = {
        "comfort_band_c": float(reward_constants["comfort_band_c"]),
        "reserve_low_soc": float(baseline_config["reserve_low_soc"]),
        "reserve_high_soc": float(baseline_config["reserve_high_soc"]),
        "grid_limit_kw": (
            None
            if baseline_config.get("grid_limit_p_max") is None
            else float(baseline_config["grid_limit_p_max"])
        ),
    }

    checkpoints = discover_checkpoints(checkpoints_dir)
    final_timestep = resolve_final_timestep(output_dir)
    for entry in checkpoints:
        entry["resolved_timestep"] = (
            entry["timestep"] if entry["timestep"] is not None else final_timestep
        )

    controller_name = f"ppo_seed{seed}"
    started = time.time()
    rows = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        row = evaluate_one_checkpoint(
            checkpoint,
            schema_path=schema_path,
            config=config,
            thresholds=thresholds,
            reward_constants=reward_constants,
            start_step=start_step,
            end_step=end_step,
            seed=seed,
            controller_name=controller_name,
        )
        rows.append(row)
        print(
            f"[{index:>2}/{len(checkpoints)}] {row['checkpoint']} (step {row['timestep']}): "
            f"cost_total={row['cost_total']:.6f} discomfort={row['discomfort_proportion']:.6f} "
            f"return={row['episode_return']:.4f}",
            flush=True,
        )

    frame = pd.DataFrame(rows, columns=list(EVALUATION_COLUMNS))
    validate_nan_free(frame, "evaluations table")

    evaluations_path = output_dir / "evaluations.csv"
    frame.to_csv(evaluations_path, index=False)

    # Selection rule executed exactly as frozen (plan §C3).
    best = select_best_checkpoint(frame)
    best_record = {column: best[column] for column in EVALUATION_COLUMNS}
    kpi_record: Dict[str, Any] = {}
    for column in EVALUATION_COLUMNS:
        if column == "checkpoint":
            continue
        value = best_record[column]
        kpi_record[column] = float(value) if column not in (
            "timestep", "comfort_violation_hours", "grid_limit_exceedances",
            "clipping_events", "reserve_events",
        ) else int(value)
    reference = b0_reference_kpis(PROJECT_ROOT / "results/runs/baselines", args.window)
    beats_b0 = (
        None if reference is None or "cost_total" not in reference
        else bool(float(best_record["cost_total"]) < float(reference["cost_total"]))
    )

    selection_record: Dict[str, Any] = {
        "purpose": "week-3 phase C frozen checkpoint selection",
        "plan_reference": "docs/plans/week3-implementation-plan.md §C3",
        "selection_rule": SELECTION_RULE_TEXT,
        "seed": seed,
        "window": {
            "name": args.window,
            "simulation_start_time_step": start_step,
            "simulation_end_time_step": end_step,
        },
        "evaluation_harness": "energy_optimisation.evaluation.runner.run_episode (locked)",
        "num_checkpoints_evaluated": int(len(frame)),
        "selected_checkpoint": str(best_record["checkpoint"]),
        "selected_checkpoint_path": relative_to_project_root(
            checkpoints_dir / str(best_record["checkpoint"])
        ),
        "timestep": int(best_record["timestep"]),
        "kpis": kpi_record,
        "beats_b0_cost": beats_b0,
        "b0_reference": reference,
        "git_commit": resolve_git_commit(PROJECT_ROOT),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    selection_path = output_dir / "selected_checkpoint.json"
    with selection_path.open("w") as handle:
        json.dump(selection_record, handle, indent=2, sort_keys=False)
        handle.write("\n")

    figure_path = plot_kpi_curves(
        frame,
        str(best_record["checkpoint"]),
        reference,
        args.figures_dir.resolve(),
        seed,
        args.window,
    )

    print(f"[gate] evaluations: {len(frame)} checkpoints, NaN-free -> {relative_to_project_root(evaluations_path)}")
    print(f"[gate] selected: {best_record['checkpoint']} "
          f"(cost_total={float(best_record['cost_total']):.6f}, "
          f"discomfort_proportion={float(best_record['discomfort_proportion']):.6f})")
    if beats_b0 is not None:
        print(f"[gate] beats B0 cost on {args.window} window: {beats_b0}"
              + ("" if beats_b0 else " — recorded finding, not a failure"))
    print(f"[gate] kpi-curve figure: {relative_to_project_root(figure_path)}")
    print(f"[gate] wall clock: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
