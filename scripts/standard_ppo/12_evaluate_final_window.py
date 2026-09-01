"""Phase D final-window evaluation of each seed's selected PPO checkpoint.

Plan reference: docs/plans/week3-implementation-plan.md §D1. Evaluates every seed's
frozen-rule-selected checkpoint (``results/runs/ppo/seed<seed>/
selected_checkpoint.json``) on the held-out final window 0–719 through the
locked week-2 harness (:func:`energy_optimisation.evaluation.runner.run_episode`
via ``run_and_record``), persisting artifacts under ``results/runs/ppo/seed<seed>/final/``
in the same shape as ``results/runs/baselines/<controller>/<window>/``
(run_metadata.json, trace.csv, district_kpis.csv, derived_metrics.json,
README.md). Nothing under ``results/runs/baselines/`` or ``results/runs/smoke/``
is touched.

Derived-metric thresholds come from the frozen configs only: comfort band from
the week-3 reward block, SoC reserve band and grid limit from
``configs/week2-baselines.yaml`` (read-only reference).

Usage:
    ./.venv/bin/python scripts/standard_ppo/12_evaluate_final_window.py --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from energy_optimisation.evaluation.artifacts import DISTRICT_KPIS_FILE, TRACE_FILE
from energy_optimisation.evaluation.runner import (
    relative_to_project_root,
    resolve_git_commit,
    run_and_record,
)
from energy_optimisation.rl import PPOController

DEFAULT_CONFIG = PROJECT_ROOT / "configs/week3-ppo.yaml"
DEFAULT_BASELINE_CONFIG = PROJECT_ROOT / "configs/week2-baselines.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Frozen week-3 PPO config.",
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=DEFAULT_BASELINE_CONFIG,
        help="Read-only week-2 config supplying reserve-band/grid-limit thresholds.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(42, 43, 44),
        help="Seed directories results/runs/ppo/seed<seed> to evaluate.",
    )
    parser.add_argument(
        "--window",
        choices=("final",),
        default="final",
        help="Window name from the config (Phase D uses the held-out final window).",
    )
    return parser.parse_args()


def load_selected_checkpoint(output_dir: Path, seed: int) -> Dict[str, Any]:
    path = output_dir / "selected_checkpoint.json"
    if not path.is_file():
        raise SystemExit(f"missing frozen-rule selection record: {path}")
    with path.open() as handle:
        record = json.load(handle)
    checkpoint_rel = str(record.get("selected_checkpoint_path", ""))
    if not checkpoint_rel:
        raise SystemExit(f"{path} lacks 'selected_checkpoint_path'")
    checkpoint_path = PROJECT_ROOT / checkpoint_rel
    if not checkpoint_path.is_file():
        raise SystemExit(f"selected checkpoint not found: {checkpoint_path}")
    print(
        f"[seed {seed}] selected by frozen rule: {record['selected_checkpoint']} "
        f"(step {record['timestep']}, dev cost_total={record['kpis']['cost_total']:.6f})",
        flush=True,
    )
    return {
        "path": checkpoint_path,
        "name": str(record["selected_checkpoint"]),
        "timestep": int(record["timestep"]),
        "selection": record,
    }


def main() -> int:
    args = parse_args()

    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    with args.baseline_config.open() as handle:
        baseline_config = yaml.safe_load(handle)

    schema_path = PROJECT_ROOT / config["schema_path"]
    bounds = config["windows"][args.window]
    start_step = int(bounds["simulation_start_time_step"])
    end_step = int(bounds["simulation_end_time_step"])

    thresholds = {
        "comfort_band_c": float(config["reward"]["comfort_band_c"]),
        "reserve_low_soc": float(baseline_config["reserve_low_soc"]),
        "reserve_high_soc": float(baseline_config["reserve_high_soc"]),
        "grid_limit_kw": (
            None
            if baseline_config.get("grid_limit_p_max") is None
            else float(baseline_config["grid_limit_p_max"])
        ),
    }

    started = time.time()
    summaries: List[Dict[str, Any]] = []
    for seed in args.seeds:
        output_dir = (PROJECT_ROOT / "outputs" / "ppo" / f"seed{seed}").resolve()
        selected = load_selected_checkpoint(output_dir, int(seed))

        controller_name = f"ppo_seed{seed}"
        controller = PPOController(
            selected["path"],
            config,
            device="cpu",
            name=controller_name,
        )
        final_dir = output_dir / args.window
        kpis, trace = run_and_record(
            controller,
            schema_path,
            output_directory=final_dir,
            simulation_start_time_step=start_step,
            simulation_end_time_step=end_step,
            seed=int(seed),
            purpose=(
                "week-3 phase D final-window evaluation of the frozen-rule "
                "selected PPO checkpoint (held-out window, no training)"
            ),
            config=config,
            derived_metric_thresholds=thresholds,
        )
        # Augment the harness-written metadata with the frozen-rule selection
        # provenance (week-3 artifact directory; harness itself stays untouched).
        metadata_path = Path(final_dir) / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["selected_checkpoint"] = selected["name"]
        metadata["selected_checkpoint_path"] = relative_to_project_root(selected["path"])
        metadata["selected_checkpoint_timestep"] = selected["timestep"]
        metadata["selection_rule"] = str(selected["selection"].get("selection_rule", ""))
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        summary_entry: Dict[str, Any] = {
            "seed": int(seed),
            "controller": controller_name,
            "selected_checkpoint": selected["name"],
            "selected_timestep": selected["timestep"],
            "window": args.window,
            "simulation_start_time_step": start_step,
            "simulation_end_time_step": end_step,
            "trace_steps": int(len(trace)),
            "cost_total": float(kpis["cost_total"]),
            "all_time_peak_average": float(kpis["all_time_peak_average"]),
            "electricity_consumption_total": float(kpis["electricity_consumption_total"]),
            "discomfort_proportion": float(kpis["discomfort_proportion"]),
            "discomfort_hot_proportion": float(kpis["discomfort_hot_proportion"]),
            "ramping_average": float(kpis["ramping_average"]),
            "zero_net_energy": float(kpis["zero_net_energy"]),
        }
        summaries.append(summary_entry)
        print(
            f"[gate] seed {seed}: {args.window} window {start_step}-{end_step}, "
            f"{len(trace)} steps -> {relative_to_project_root(final_dir)}\n"
            f"[gate]   cost_total={summary_entry['cost_total']:.6f} "
            f"discomfort={summary_entry['discomfort_proportion']:.6f}",
            flush=True,
        )

    # Machine-readable roll-up next to the per-seed artifacts (never inside the
    # week-2 tree).
    rollup_path = PROJECT_ROOT / "outputs" / "ppo" / "final_window_summaries.json"
    with rollup_path.open("w") as handle:
        json.dump(
            {
                "purpose": "week-3 phase D final-window evaluation roll-up",
                "plan_reference": "docs/plans/week3-implementation-plan.md §D1",
                "evaluation_harness": (
                    "energy_optimisation.evaluation.runner.run_and_record (locked)"
                ),
                "selection_rule_source": "results/runs/ppo/seed<seed>/selected_checkpoint.json",
                "git_commit": resolve_git_commit(PROJECT_ROOT),
                "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "runs": summaries,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    print(f"[gate] wall clock: {time.time() - started:.1f}s")
    print(f"[gate] roll-up: {relative_to_project_root(rollup_path)}")
    missing_artifacts = [
        f"seed{s}/{name}"
        for s in args.seeds
        for name in (DISTRICT_KPIS_FILE, TRACE_FILE)
        if not (PROJECT_ROOT / "outputs" / "ppo" / f"seed{s}" / args.window / name).is_file()
    ]
    if missing_artifacts:
        print(f"[gate] FAIL: missing artifacts: {', '.join(missing_artifacts)}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
