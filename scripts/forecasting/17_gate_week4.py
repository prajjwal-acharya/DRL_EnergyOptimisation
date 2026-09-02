"""Hard verification gate for the completed Week-4 forecasting phase."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energy_optimisation.forecasting.pipeline import (
    CONFORMAL_VARIANTS,
    PREDICTION_COLUMNS,
    sha256_file,
)

CONFIG_PATH = PROJECT_ROOT / "configs/week4-forecasting.yaml"
RUN_ROOT = PROJECT_ROOT / "results/runs/forecasting"
TABLES_DIR = PROJECT_ROOT / "results/tables"
FIGURES_DIR = PROJECT_ROOT / "results/figures"
REVIEW_PATH = PROJECT_ROOT / "docs/status/phase-reviews/week4-review.md"
PROTECTED_TABLES = (
    "results/tables/baseline_comparison.csv",
    "results/tables/ppo_multiseed_summary.csv",
    "results/tables/ppo_vs_baselines.csv",
)
PROTECTED_TABLE_HASHES = {
    "results/tables/baseline_comparison.csv": "f11d9f312d273f00adfddf690413f39d5d118593c96fb71a3cb951aa18d08c0c",
    "results/tables/ppo_multiseed_summary.csv": "41cd6fa46641ba96707525d293d86363e2d6e1d07d1e8ad7065d45ce94723aca",
    "results/tables/ppo_vs_baselines.csv": "745407e5c0fb13be3077289c98bb6137c2099b8e754700c0d221d38052ac2363",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _run(command: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False
    )


def verify_pytest() -> str:
    completed = _run([str(PROJECT_ROOT / ".venv/bin/python"), "-m", "pytest", "-q"])
    require(completed.returncode == 0, f"pytest failed:\n{completed.stdout}\n{completed.stderr}")
    return completed.stdout.strip().splitlines()[-1]


def verify_config_and_selection(config: dict) -> str:
    metadata_path = RUN_ROOT / "backtest_run_metadata.json"
    require(metadata_path.is_file(), f"missing {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    config_hash = sha256_file(CONFIG_PATH)
    require(metadata.get("config_sha256") == config_hash, "backtest config hash mismatch")
    selection_path = RUN_ROOT / "selected_models.json"
    require(selection_path.is_file(), f"missing {selection_path}")
    selection = json.loads(selection_path.read_text())
    require(selection.get("config_sha256") == config_hash, "selection config hash mismatch")
    targets = selection.get("targets", {})
    require(set(targets) == set(config["targets"]), "selection target set is incomplete")
    for target, record in targets.items():
        require(record.get("selected_variant"), f"{target}: selected variant missing")
        require(record.get("metrics"), f"{target}: selected metrics missing")
        require(record.get("rule_trace"), f"{target}: rule trace missing")
        artifact = PROJECT_ROOT / record["artifact_path"]
        require(artifact.is_file(), f"{target}: refit artifact missing: {artifact}")
    return f"config hash and {len(targets)} target selections verified"


def verify_predictions(config: dict) -> str:
    variants = tuple(config["models"]["order"]) + CONFORMAL_VARIANTS
    checked = 0
    for variant in variants:
        for target in config["targets"]:
            path = RUN_ROOT / variant / target / "predictions.csv"
            require(path.is_file(), f"missing {path}")
            frame = pd.read_csv(path)
            require(tuple(frame.columns) == PREDICTION_COLUMNS, f"{path}: schema mismatch")
            require(len(frame) == 1434, f"{path}: expected 1434 honest evaluable pairs")
            require(not frame.isna().any().any(), f"{path}: contains NaN")
            numeric = frame.to_numpy(dtype=float)
            require(np.isfinite(numeric).all(), f"{path}: contains non-finite values")
            quantiles = frame[["q05", "q25", "q50", "q75", "q95"]].to_numpy(float)
            require(
                bool(np.all(np.diff(quantiles, axis=1) >= -1e-12)),
                f"{path}: crossing quantiles",
            )
            counts = frame.groupby("horizon").size().to_dict()
            require(counts == {1: 479, 2: 478, 3: 477}, f"{path}: boundary counts wrong")
            require((path.parent / "metrics.json").is_file(), f"{path}: metrics missing")
            checked += 1
    return f"{checked} prediction/metric artifact pairs are complete and monotone"


def verify_tables_figures_docs(config: dict) -> str:
    comparison_path = TABLES_DIR / "forecast_model_comparison.csv"
    calibration_path = TABLES_DIR / "forecast_calibration_by_hour.csv"
    require(comparison_path.is_file(), "forecast comparison table missing")
    require(calibration_path.is_file(), "calibration-by-hour table missing")
    comparison = pd.read_csv(comparison_path)
    calibration = pd.read_csv(calibration_path)
    require(len(comparison) == 7 * len(config["targets"]), "comparison row count mismatch")
    for name, frame in (("comparison", comparison), ("calibration", calibration)):
        require(not frame.isna().any().any(), f"{name} table contains NaN")
    figures = [
        FIGURES_DIR / f"forecast_{target}_fanchart.png" for target in config["targets"]
    ] + [
        FIGURES_DIR / "forecast_coverage_by_target.png",
        FIGURES_DIR / "forecast_pinball_by_horizon.png",
    ]
    for path in figures:
        require(path.is_file() and path.stat().st_size > 0, f"missing/empty figure {path}")
    require(REVIEW_PATH.is_file(), f"missing review doc {REVIEW_PATH}")
    review = REVIEW_PATH.read_text()
    required_text = (
        "no dev-window forecast-accuracy claim is made",
        "solar_generation is exactly 0 at night",
        "full-series refit",
        "No controller, RL, or safety-shield work was touched.",
    )
    for text in required_text:
        require(text in review, f"review lacks required disclosure: {text!r}")
    return f"2 tables, {len(figures)} figures, and review disclosures verified"


def verify_prior_evidence() -> str:
    for relative in PROTECTED_TABLES:
        current = PROJECT_ROOT / relative
        require(current.is_file(), f"protected table missing: {relative}")
        require(
            sha256_file(current) == PROTECTED_TABLE_HASHES[relative],
            f"protected table changed from the pre-Week-4 byte snapshot: {relative}",
        )
    status = _run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "results/runs/baselines",
            "results/runs/ppo",
        ]
    )
    require(not status.stdout.strip(), "protected Week 2/3 run paths have tracked changes")
    return "Week 1-3 comparison tables and run paths are untouched"


def main() -> int:
    with CONFIG_PATH.open() as handle:
        config = yaml.safe_load(handle)
    checks = (
        verify_pytest(),
        verify_config_and_selection(config),
        verify_predictions(config),
        verify_tables_figures_docs(config),
        verify_prior_evidence(),
    )
    print("Week 4 gate: PASS")
    for check in checks:
        print(f"  - {check}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
