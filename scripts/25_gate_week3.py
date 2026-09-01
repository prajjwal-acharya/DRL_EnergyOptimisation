"""Verify the completed Week 3 phase: PPO adapter, training, learning curves.

Plan reference: docs/plans/week3-implementation-plan.md §D3 (mirrors the structure of
``scripts/12_gate_week2.py``). Every check below is a hard pass/fail with a
clear message; the script exits non-zero if any check fails. Nothing under
``results/`` is modified — this reads recorded phase evidence only.

Checks:
1. the full pytest suite passes (week-1 + week-2 + week-3 tests),
2. all three seeds have complete artifact sets: ≥15 numbered checkpoints plus
   ``final.zip``, ``evaluations.csv``, ``selected_checkpoint.json``, monitor
   logs, and dev + final run directories,
3. the frozen checkpoint-selection rule (lowest cost_total, tie-break lower
   discomfort_proportion) reproduces each recorded selection,
4. ``results/tables/ppo_multiseed_summary.csv`` and
   ``results/tables/ppo_vs_baselines.csv`` exist with the expected row
   structure,
5. learning-curve figures exist (return curve + KPI curves per seed) plus the
   comparison cost figure,
6. ``results/tables/baseline_comparison.csv`` is byte-identical to its week-2
   state, re-derived from the untouched baseline run artifacts,
7. no NaN in any PPO trace; clipping/reserve event counts present in derived
   metrics and evaluations tables,
8. ``configs/week3-ppo.yaml`` exists with the frozen hyperparameters and
   normalisation stats, and every training run recorded its exact hash,
9. docs/status/phase-reviews/week3-review.md exists.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import yaml

from energy_optimisation.rl import EVALUATION_COLUMNS as PLAN_EVALUATION_COLUMNS

SEEDS = (42, 43, 44)
PPO_ROOT = PROJECT_ROOT / "results/runs/ppo"
BASELINES_ROOT = PROJECT_ROOT / "results/runs/baselines"
TABLES_DIR = PROJECT_ROOT / "results/tables"
FIGURES_DIR = PROJECT_ROOT / "results/figures"
CONFIG_PATH = PROJECT_ROOT / "configs/week3-ppo.yaml"
REVIEW_PATH = PROJECT_ROOT / "docs/status/phase-reviews/week3-review.md"

NUMBERED_CHECKPOINT_PATTERN = re.compile(r"^ppo_(\d{8})_steps\.zip$")
MIN_NUMBERED_CHECKPOINTS = 15
EVALUATION_ROWS_PER_SEED = 21  # 20 numbered checkpoints + final.zip at 200k steps

RUN_METADATA_FILE = "run_metadata.json"
TRACE_FILE = "trace.csv"
DISTRICT_KPIS_FILE = "district_kpis.csv"
DERIVED_METRICS_FILE = "derived_metrics.json"
README_FILE = "README.md"
REQUIRED_FINAL_ARTIFACTS = (
    RUN_METADATA_FILE,
    TRACE_FILE,
    DISTRICT_KPIS_FILE,
    DERIVED_METRICS_FILE,
    README_FILE,
)

PRIMARY_KPIS = (
    "cost_total",
    "all_time_peak_average",
    "electricity_consumption_total",
    "discomfort_hot_proportion",
    "discomfort_proportion",
    "ramping_average",
    "zero_net_energy",
)
# Canonical plan §C2 column order, single-sourced from the rl package.
EVALUATION_COLUMNS = tuple(PLAN_EVALUATION_COLUMNS)
SUMMARY_KPIS = tuple(
    column for column in EVALUATION_COLUMNS if column not in ("checkpoint", "timestep")
)
BASELINE_CONTROLLERS = ("b0_neutral", "b1_fixed_schedule", "b2_tariff_aware")
WINDOWS = ("dev", "final")
BASELINE_LABELS = {
    "b0_neutral": "B0",
    "b1_fixed_schedule": "B1",
    "b2_tariff_aware": "B2",
}
EXPECTED_VS_BASELINES_CONTROLLERS = (
    "B0",
    "B1",
    "B2",
    "PPO-seed42",
    "PPO-seed43",
    "PPO-seed44",
)
FINAL_WINDOW_BOUNDS = (0, 719)

# Frozen hyperparameters (plan §B1) that configs/week3-ppo.yaml must still pin.
FROZEN_HYPERPARAMETERS = {
    "policy": "MlpPolicy",
    "policy_hidden_layers": [64, 64],
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "total_timesteps": 200000,
    "checkpoint_every": 10000,
    "device": "cpu",
}

# SHA-256 of the original configs/ppo/week3.yaml in the previous workspace
# (/Volumes/code/Research Project/code) with which the week-3 training runs were
# actually produced. The September-2026 fresh-start migration renamed two path
# strings inside the file (schema_path, normalisation stats_source) and moved it
# to configs/week3-ppo.yaml, changing its hash without changing any frozen value.
# Runs whose recorded config hash equals this legacy value are therefore accepted;
# everything else must match the current file exactly.
LEGACY_WEEK3_CONFIG_SHA256 = "1661674d3cd2ea84505a195cadc6598a4723703d350554ebfca6fd54a9ddc0c2"


def require(condition: bool, message: str) -> None:
    """Fail loudly when a Week 3 invariant is broken."""

    if not condition:
        raise AssertionError(message)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pytest_suite() -> str:
    command = [str(PROJECT_ROOT / ".venv/bin/python"), "-m", "pytest", "-q"]
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    require(
        completed.returncode == 0,
        f"pytest suite failed (exit {completed.returncode}):\n{completed.stdout[-2000:]}",
    )
    summary = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    return f"pytest suite green ({summary})"


def read_run_kpis(run_directory: Path) -> Dict[str, float]:
    kpis_path = run_directory / DISTRICT_KPIS_FILE
    require(kpis_path.is_file(), f"Missing district KPIs: {kpis_path}")
    with kpis_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        row["cost_function"]: float(row["value"])
        for row in rows
        if str(row["value"]).strip() != ""
    }


def verify_seed_artifacts() -> str:
    checked_dirs = 0
    for seed in SEEDS:
        seed_dir = PPO_ROOT / f"seed{seed}"
        require(seed_dir.is_dir(), f"Missing seed directory: {seed_dir}")

        checkpoints_dir = seed_dir / "checkpoints"
        require(checkpoints_dir.is_dir(), f"Missing checkpoint directory: {checkpoints_dir}")
        numbered = sorted(
            path.name for path in checkpoints_dir.glob("*.zip")
            if NUMBERED_CHECKPOINT_PATTERN.match(path.name)
        )
        require(
            len(numbered) >= MIN_NUMBERED_CHECKPOINTS,
            f"seed{seed}: expected >= {MIN_NUMBERED_CHECKPOINTS} numbered checkpoints, found {len(numbered)}",
        )
        require(
            (checkpoints_dir / "final.zip").is_file(),
            f"seed{seed}: missing final.zip under {checkpoints_dir}",
        )

        metadata_path = seed_dir / RUN_METADATA_FILE
        require(metadata_path.is_file(), f"seed{seed}: missing {RUN_METADATA_FILE}")
        metadata = json.loads(metadata_path.read_text())
        require(int(metadata["seed"]) == seed, f"seed{seed}: run metadata seed mismatch")
        require(str(metadata["device"]) == "cpu", f"seed{seed}: training device must be cpu")
        require(
            int(metadata["total_timesteps_completed"]) >= int(FROZEN_HYPERPARAMETERS["total_timesteps"]),
            f"seed{seed}: fewer than the configured total timesteps were trained",
        )
        config_hash = sha256_of_file(CONFIG_PATH)
        require(
            str(metadata.get("config_sha256", "")) in (config_hash, LEGACY_WEEK3_CONFIG_SHA256),
            f"seed{seed}: training used a different configs/week3-ppo.yaml than the current frozen file",
        )
        require(len(metadata.get("checkpoints", [])) >= MIN_NUMBERED_CHECKPOINTS + 1,
                f"seed{seed}: metadata checkpoint list incomplete")

        monitor_path = seed_dir / "monitor.csv"
        require(monitor_path.is_file(), f"seed{seed}: missing SB3 monitor log {monitor_path}")
        monitor = pd.read_csv(monitor_path, skiprows=1)
        require(
            bool(pd.to_numeric(monitor["r"], errors="coerce").notna().all())
            and not monitor[["r", "l", "t"]].isna().any().any(),
            f"seed{seed}: monitor log contains NaN episode returns (training diverged?)",
        )
        checked_dirs += 1
    return (
        f"{checked_dirs}/3 seeds have checkpoints (>= {MIN_NUMBERED_CHECKPOINTS} numbered + final), "
        "metadata, and NaN-free monitor logs"
    )


def verify_evaluations_and_selection() -> str:
    checked = 0
    for seed in SEEDS:
        seed_dir = PPO_ROOT / f"seed{seed}"
        evaluations_path = seed_dir / "evaluations.csv"
        require(evaluations_path.is_file(), f"seed{seed}: missing evaluations table {evaluations_path}")
        frame = pd.read_csv(evaluations_path)
        require(
            list(frame.columns) == list(EVALUATION_COLUMNS),
            f"seed{seed}: evaluations columns deviate from the plan §C2 order",
        )
        require(
            len(frame) == EVALUATION_ROWS_PER_SEED,
            f"seed{seed}: evaluations covers {len(frame)} checkpoints, expected {EVALUATION_ROWS_PER_SEED}",
        )
        numeric = frame.drop(columns=["checkpoint"])
        require(not frame.isna().any().any(), f"seed{seed}: evaluations table contains NaN values")
        require(
            bool(np.isfinite(numeric.to_numpy(dtype=float)).all()),
            f"seed{seed}: evaluations table contains non-finite values",
        )

        selection_path = seed_dir / "selected_checkpoint.json"
        require(selection_path.is_file(), f"seed{seed}: missing selection record {selection_path}")
        record = json.loads(selection_path.read_text())
        # Re-execute the frozen rule independently and demand the same winner.
        ordered = frame.sort_values(
            ["cost_total", "discomfort_proportion"], kind="mergesort", na_position="last"
        )
        expected_best = str(ordered.iloc[0]["checkpoint"])
        recorded_best = str(record["selected_checkpoint"])
        require(
            recorded_best == expected_best,
            f"seed{seed}: recorded selection {recorded_best!r} violates the frozen rule "
            f"(lowest cost_total, tie-break lower discomfort_proportion -> {expected_best!r})",
        )
        selected_row = frame.loc[frame["checkpoint"] == recorded_best]
        require(len(selected_row) == 1, f"seed{seed}: selected checkpoint absent from evaluations table")
        require(
            abs(float(selected_row.iloc[0]["cost_total"]) - float(record["kpis"]["cost_total"])) <= 0.0,
            f"seed{seed}: selection-record KPIs disagree with the evaluations table",
        )
        require(
            (checkpoints_dir_of(seed) / recorded_best).is_file(),
            f"seed{seed}: selected checkpoint file missing on disk",
        )
        checked += 1
    return f"{checked}/3 seeds evaluated NaN-free on all checkpoints with the frozen selection rule reproduced"


def checkpoints_dir_of(seed: int) -> Path:
    return PPO_ROOT / f"seed{seed}" / "checkpoints"


def verify_final_window_runs() -> str:
    checked = 0
    for seed in SEEDS:
        directory = PPO_ROOT / f"seed{seed}" / "final"
        missing = [name for name in REQUIRED_FINAL_ARTIFACTS if not (directory / name).is_file()]
        require(not missing, f"ppo_seed{seed}/final: missing artifacts {', '.join(missing)}")
        metadata = json.loads((directory / RUN_METADATA_FILE).read_text())
        require(
            str(metadata["controller"]) == f"ppo_seed{seed}",
            f"ppo_seed{seed}/final: metadata controller mismatch",
        )
        require(int(metadata["seed"]) == seed, f"ppo_seed{seed}/final: seed mismatch")
        bounds = (int(metadata["simulation_start_time_step"]), int(metadata["simulation_end_time_step"]))
        require(
            bounds == FINAL_WINDOW_BOUNDS,
            f"ppo_seed{seed}/final window {bounds} != frozen {FINAL_WINDOW_BOUNDS}",
        )
        trace = pd.read_csv(directory / TRACE_FILE)
        require(not trace.isna().any().any(), f"ppo_seed{seed}/final: NaN values in trace")
        require(len(trace) == 719, f"ppo_seed{seed}/final: trace has {len(trace)} rows, expected 719")
        metrics = json.loads((directory / DERIVED_METRICS_FILE).read_text())
        for key in ("clipping_event_count", "reserve_event_count"):
            require(key in metrics, f"ppo_seed{seed}/final: derived metrics lack '{key}'")
            require(isinstance(metrics[key], int), f"ppo_seed{seed}/final: '{key}' must be an integer count")

        # The final-window run must come from the checkpoint the frozen rule chose.
        record = json.loads((PPO_ROOT / f"seed{seed}" / "selected_checkpoint.json").read_text())
        require(
            str(metadata.get("selected_checkpoint", "")) == str(record["selected_checkpoint"]),
            f"ppo_seed{seed}/final: metadata selected_checkpoint does not match the frozen-rule selection",
        )
        checked += 1
    return f"{checked}/3 final-window runs complete (0-719, locked-harness artifact shape)"


def verify_summary_tables() -> str:
    multiseed_path = TABLES_DIR / "ppo_multiseed_summary.csv"
    require(multiseed_path.is_file(), f"Missing multi-seed summary: {multiseed_path}")
    frame = pd.read_csv(multiseed_path)
    expected_columns = ["kpi", *[f"ppo_seed{seed}" for seed in SEEDS], "mean", "min", "max"]
    require(
        list(frame.columns) == expected_columns,
        f"{multiseed_path.name} must have columns {expected_columns}, found {list(frame.columns)}",
    )
    require(
        set(frame["kpi"]) == set(SUMMARY_KPIS),
        f"{multiseed_path.name} KPI rows deviate from the evaluation KPI set",
    )
    require(not frame.isna().any().any(), f"{multiseed_path.name} contains NaN values")

    comparison_path = TABLES_DIR / "ppo_vs_baselines.csv"
    require(comparison_path.is_file(), f"Missing comparison table: {comparison_path}")
    rows = list(csv.DictReader(comparison_path.open(newline="")))
    pairs = {(row["controller"], row["window"]) for row in rows}
    expected_pairs = {
        (controller, window)
        for controller in EXPECTED_VS_BASELINES_CONTROLLERS
        for window in WINDOWS
    }
    require(pairs == expected_pairs, f"{comparison_path.name} rows deviate from 6 controllers x 2 windows")
    for row in rows:
        require(row["cost_total"] != "", f"{comparison_path.name}: {row['controller']}/{row['window']} lacks cost_total")
    return (
        f"{multiseed_path.name} ({len(frame)} KPI rows) and {comparison_path.name} "
        f"({len(rows)} rows = 6 controllers x 2 windows) present with expected structure"
    )


def verify_figures() -> str:
    require(FIGURES_DIR.is_dir(), f"Missing figures directory: {FIGURES_DIR}")
    for seed in SEEDS:
        for kind in ("return_curve", "kpi_curves"):
            path = FIGURES_DIR / f"ppo_seed{seed}_{kind}.png"
            require(path.is_file() and path.stat().st_size > 0, f"Missing figure: {path}")
    cost_figure = FIGURES_DIR / "ppo_vs_baselines_cost.png"
    require(cost_figure.is_file() and cost_figure.stat().st_size > 0, f"Missing figure: {cost_figure}")
    return f"per-seed return + KPI-curve figures and {cost_figure.name} all present"


def verify_baseline_table_untouched() -> str:
    """Re-derive baseline_comparison.csv from week-2 artifacts; demand equality."""

    comparison_path = TABLES_DIR / "baseline_comparison.csv"
    require(comparison_path.is_file(), f"Missing week-2 evidence: {comparison_path}")

    rows: List[Dict[str, object]] = []
    for window in WINDOWS:
        for controller in BASELINE_CONTROLLERS:
            kpis = read_run_kpis(BASELINES_ROOT / controller / window)
            row: Dict[str, object] = {"controller": controller, "window": window}
            for kpi in PRIMARY_KPIS:
                row[kpi] = float(kpis[kpi]) if kpi in kpis else None
            rows.append(row)
    regenerated = pd.DataFrame(rows, columns=["controller", "window", *PRIMARY_KPIS])
    buffer = io.StringIO()
    regenerated.to_csv(buffer, index=False, float_format="%.17g")
    require(
        buffer.getvalue() == comparison_path.read_text(),
        "baseline_comparison.csv is not byte-identical to its week-2 state "
        "(re-derived from results/runs/baselines artifacts)",
    )
    return "baseline_comparison.csv byte-identical to its week-2 state (re-derived from run artifacts)"


def verify_config_frozen() -> str:
    require(CONFIG_PATH.is_file(), f"Missing frozen config: {CONFIG_PATH}")
    config: Dict = yaml.safe_load(CONFIG_PATH.read_text())
    ppo_block = config.get("ppo")
    require(isinstance(ppo_block, dict), "configs/week3-ppo.yaml lacks the 'ppo' hyperparameter block")
    for key, expected in FROZEN_HYPERPARAMETERS.items():
        actual = ppo_block.get(key)
        matches = list(actual) == list(expected) if isinstance(expected, list) else actual == expected
        require(matches, f"frozen hyperparameter '{key}' changed: {actual!r} != {expected!r}")

    windows = config.get("windows")
    require(
        windows.get("dev", {}).get("simulation_start_time_step") == 0
        and windows["dev"]["simulation_end_time_step"] == 167
        and windows.get("final", {}).get("simulation_start_time_step") == 0
        and windows["final"]["simulation_end_time_step"] == 719,
        "configs/week3-ppo.yaml windows deviate from the frozen dev/final bounds",
    )
    reward = config.get("reward")
    require(
        isinstance(reward, dict)
        and float(reward["w_E"]) == 1.0
        and float(reward["w_P"]) == 1.0
        and float(reward["w_C"]) == 10.0
        and float(reward["E_bar_b0"]) == 0.477229108554339
        and float(reward["P_ref"]) == 7.694016456604004
        and float(reward["comfort_band_c"]) == 2.0,
        "configs/week3-ppo.yaml reward constants deviate from the frozen CMDP spec",
    )
    features = (config.get("normalisation") or {}).get("features")
    require(
        isinstance(features, dict) and len(features) >= 26,
        "configs/week3-ppo.yaml lacks the frozen normalisation stats block",
    )
    for name, entry in features.items():
        require(
            isinstance(entry, dict) and "offset" in entry and "scale" in entry,
            f"normalisation feature '{name}' lacks offset/scale",
        )
    return "week3.yaml pins all frozen hyperparameters, CMDP constants, and normalisation stats"


def verify_review_document() -> str:
    require(REVIEW_PATH.is_file(), f"Missing review document: {REVIEW_PATH}")
    text = REVIEW_PATH.read_text()
    lowered = text.lower()
    require(
        "supervisor update" in lowered,
        "week3-review.md must include the supervisor update verbatim",
    )
    require(
        "ppo did not beat b0" in lowered or "did not beat b0" in lowered,
        "week3-review.md must state honestly whether PPO beat B0",
    )
    return "week3-review.md present with supervisor update and honest B0 comparison"


CHECKS = (
    ("Pytest", verify_pytest_suite),
    ("Seed artifacts", verify_seed_artifacts),
    ("Evaluations & selection", verify_evaluations_and_selection),
    ("Final-window runs", verify_final_window_runs),
    ("Summary tables", verify_summary_tables),
    ("Figures", verify_figures),
    ("Week-2 evidence intact", verify_baseline_table_untouched),
    ("Frozen config", verify_config_frozen),
    ("Review doc", verify_review_document),
)


def main() -> None:
    passed: List[str] = []
    failed: List[str] = []
    for name, check in CHECKS:
        try:
            detail = check()
        except AssertionError as error:
            failed.append(name)
            print(f"FAIL  {name}: {error}")
            continue
        passed.append(name)
        print(f"PASS  {name}: {detail}")

    require(not failed, f"Week 3 verification failed check(s): {', '.join(failed)}")
    print("Week 3 verification passed: standard PPO adapter, training, learning curves, and locked comparison.")


if __name__ == "__main__":
    main()
