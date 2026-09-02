"""Backtest, scoring, selection, refit, and reporting orchestration for Week 4."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from .data import QUANTILES, Dataset, daylight_mask, load_dataset, make_folds
from .metrics import (
    empirical_coverage,
    mae,
    mean_interval_width,
    pinball_loss,
    rmse,
    winkler_score,
)
from .models import Conformalized, ForecastModel, create_model, save_model

PREDICTION_COLUMNS = (
    "fold",
    "t",
    "horizon",
    "y_true",
    "q05",
    "q25",
    "q50",
    "q75",
    "q95",
)
RAW_MODELS = (
    "persistence_last",
    "persistence_24h",
    "climatology_hourly",
    "linear_quantile",
    "gru_quantile",
)
CONFORMAL_VARIANTS = ("linear_quantile+conformal", "gru_quantile+conformal")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_config(config_path: Path, project_root: Path) -> Tuple[Dict[str, Any], Path]:
    resolved = Path(config_path)
    if not resolved.is_absolute():
        resolved = Path(project_root) / resolved
    with resolved.open() as handle:
        config = yaml.safe_load(handle)
    if config.get("device") != "cpu" or int(config.get("random_seed", -1)) != 42:
        raise ValueError("Week-4 config must freeze CPU execution and seed 42")
    return config, resolved.resolve()


def _project_path(project_root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else Path(project_root) / path


def _git_commit(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _valid_pairs(dataset: Dataset, test_start: int, test_end: int, horizons: Sequence[int]):
    for origin in range(int(test_start), int(test_end)):
        for horizon in horizons:
            if origin + int(horizon) < dataset.row_count:
                yield origin, int(horizon)


def _prediction_rows(
    model: ForecastModel,
    dataset: Dataset,
    fold_number: int,
    test_start: int,
    test_end: int,
) -> List[Dict[str, float]]:
    rows = []
    cache: Dict[int, Dict[int, Dict[float, float]]] = {}
    for origin, horizon in _valid_pairs(dataset, test_start, test_end, model.horizons):
        if origin not in cache:
            cache[origin] = model.predict_all(dataset, origin)
        prediction = cache[origin][horizon]
        rows.append(
            {
                "fold": int(fold_number),
                "t": int(origin),
                "horizon": int(horizon),
                "y_true": float(dataset.building.at[origin + horizon, model.target]),
                "q05": prediction[0.05],
                "q25": prediction[0.25],
                "q50": prediction[0.50],
                "q75": prediction[0.75],
                "q95": prediction[0.95],
            }
        )
    return rows


def _calibration_origins(train_end: int, fraction: float, max_horizon: int) -> List[int]:
    start = int(np.floor(int(train_end) * (1.0 - float(fraction))))
    return list(range(max(24, start), int(train_end) - int(max_horizon)))


def _fit_conformal_variant(
    base: ForecastModel,
    dataset: Dataset,
    train_end: int,
    model_name: str,
    model_config: Mapping[str, Any],
    validation_fraction: float,
) -> Conformalized:
    origins = _calibration_origins(train_end, validation_fraction, max(base.horizons))
    calibration_model = base
    # Use a separate pre-calibration fit so residuals are genuinely held out from the
    # refit-on-all-data model for both learned variants.
    if model_name in ("linear_quantile", "gru_quantile"):
        calibration_start = origins[0]
        calibration_model = create_model(
            model_name,
            base.target,
            model_config,
            base.horizons,
            base.quantiles,
            base.seed,
        ).fit(dataset, calibration_start)
    return Conformalized(base).calibrate(
        dataset, origins, calibration_model=calibration_model
    )


def _metric_block(frame: pd.DataFrame) -> Dict[str, Any]:
    truth = frame["y_true"].to_numpy(dtype=float)
    result: Dict[str, Any] = {
        "row_count": int(len(frame)),
        "mae": mae(truth, frame["q50"]),
        "rmse": rmse(truth, frame["q50"]),
        "coverage_90": empirical_coverage(truth, frame["q05"], frame["q95"]),
        "coverage_50": empirical_coverage(truth, frame["q25"], frame["q75"]),
        "mean_width_90": mean_interval_width(frame["q05"], frame["q95"]),
        "mean_width_50": mean_interval_width(frame["q25"], frame["q75"]),
        "winkler_90": winkler_score(truth, frame["q05"], frame["q95"], 0.10),
        "winkler_50": winkler_score(truth, frame["q25"], frame["q75"], 0.50),
    }
    pinballs = {
        f"q{int(round(tau * 100)):02d}": pinball_loss(
            truth, frame[f"q{int(round(tau * 100)):02d}"], tau
        )
        for tau in QUANTILES
    }
    result["pinball"] = pinballs
    result["mean_pinball"] = float(np.mean(list(pinballs.values())))
    return result


def score_predictions(frame: pd.DataFrame, dataset: Dataset, target: str) -> Dict[str, Any]:
    if frame.empty or frame[list(PREDICTION_COLUMNS)].isna().any().any():
        raise ValueError("prediction table must be non-empty and NaN-free")
    payload: Dict[str, Any] = {
        "pooled": _metric_block(frame),
        "per_horizon": {
            str(int(horizon)): _metric_block(group)
            for horizon, group in frame.groupby("horizon", sort=True)
        },
        "per_fold": {
            str(int(fold)): _metric_block(group)
            for fold, group in frame.groupby("fold", sort=True)
        },
    }
    if target == "solar_generation":
        mask = daylight_mask(dataset, frame["t"].to_numpy(dtype=int))
        daylight = frame.loc[mask].copy()
        if daylight.empty:
            raise ValueError("solar daylight metric scope contains no rows")
        payload["daylight"] = _metric_block(daylight)
        payload["daylight_per_horizon"] = {
            str(int(horizon)): _metric_block(group)
            for horizon, group in daylight.groupby("horizon", sort=True)
        }
    return payload


def run_backtest(config_path: Path, project_root: Path) -> Dict[str, Any]:
    """Execute the frozen fold/model/target ladder and write raw evidence."""

    started = time.monotonic()
    config, resolved_config = load_config(config_path, project_root)
    dataset = load_dataset(_project_path(project_root, config["dataset_path"]))
    folds = make_folds(dataset, config["folds"])
    run_root = _project_path(project_root, config["outputs"]["run_root"])
    targets = tuple(config["targets"])
    horizons = tuple(int(value) for value in config["horizons"])
    quantiles = tuple(float(value) for value in config["quantiles"])
    model_names = tuple(config["models"]["order"])
    seed = int(config["random_seed"])
    validation_fraction = float(config["folds"]["internal_validation_fraction"])
    collected: Dict[Tuple[str, str], List[Dict[str, float]]] = {
        (variant, target): []
        for target in targets
        for variant in model_names + CONFORMAL_VARIANTS
    }

    for target in targets:
        for fold in folds:
            for model_name in model_names:
                hyperparameters = config["models"].get(model_name, {})
                model = create_model(
                    model_name,
                    target,
                    hyperparameters,
                    horizons,
                    quantiles,
                    seed,
                ).fit(dataset, fold.train_end)
                collected[(model_name, target)].extend(
                    _prediction_rows(
                        model, dataset, fold.number, fold.test_start, fold.test_end
                    )
                )
                conformal_name = f"{model_name}+conformal"
                if conformal_name in CONFORMAL_VARIANTS:
                    conformal = _fit_conformal_variant(
                        model,
                        dataset,
                        fold.train_end,
                        model_name,
                        hyperparameters,
                        validation_fraction,
                    )
                    collected[(conformal_name, target)].extend(
                        _prediction_rows(
                            conformal, dataset, fold.number, fold.test_start, fold.test_end
                        )
                    )

    artifacts = []
    for (variant, target), rows in collected.items():
        frame = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
        destination = run_root / variant / target
        destination.mkdir(parents=True, exist_ok=True)
        predictions_path = destination / "predictions.csv"
        frame.to_csv(predictions_path, index=False, float_format="%.10g")
        metrics = score_predictions(frame, dataset, target)
        _json_dump(destination / "metrics.json", metrics)
        artifacts.append(str(predictions_path.relative_to(project_root)))

    metadata = {
        "experiment_name": config["experiment_name"],
        "git_commit": _git_commit(project_root),
        "config_path": str(resolved_config.relative_to(project_root)),
        "config_sha256": sha256_file(resolved_config),
        "dataset_path": config["dataset_path"],
        "dataset_rows": dataset.row_count,
        "folds": len(folds),
        "origins": sum(fold.test_end - fold.test_start for fold in folds),
        "evaluated_pairs_per_model_target": len(next(iter(collected.values()))),
        "boundary_note": (
            "Origins 717-719 retain only horizons with observed labels; 1434 honest "
            "pairs are scored rather than fabricating truth beyond row 719."
        ),
        "targets": list(targets),
        "models": list(model_names),
        "conformal_variants": list(CONFORMAL_VARIANTS),
        "seed": seed,
        "device": "cpu",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "wall_clock_seconds": time.monotonic() - started,
        "artifacts": artifacts,
    }
    _json_dump(run_root / "backtest_run_metadata.json", metadata)
    return metadata


def _selection_scope(metrics: Mapping[str, Any], target: str) -> Mapping[str, Any]:
    return metrics["daylight"] if target == "solar_generation" else metrics["pooled"]


def _selection_horizons(metrics: Mapping[str, Any], target: str) -> Mapping[str, Any]:
    key = "daylight_per_horizon" if target == "solar_generation" else "per_horizon"
    return metrics[key]


def _is_calibrated(metrics: Mapping[str, Any], target: str, config: Mapping[str, Any]) -> bool:
    rule = config["selection"]
    return all(
        float(rule["coverage_90_min"]) <= block["coverage_90"] <= float(rule["coverage_90_max"])
        and float(rule["coverage_50_min"])
        <= block["coverage_50"]
        <= float(rule["coverage_50_max"])
        for block in _selection_horizons(metrics, target).values()
    )


def _average_horizon_mae(metrics: Mapping[str, Any], target: str) -> float:
    return float(
        np.mean([block["mae"] for block in _selection_horizons(metrics, target).values()])
    )


def _read_variant_metrics(run_root: Path, variant: str, target: str) -> Dict[str, Any]:
    return json.loads((run_root / variant / target / "metrics.json").read_text())


def _refit_selected(
    variant: str,
    target: str,
    config: Mapping[str, Any],
    dataset: Dataset,
    destination: Path,
) -> None:
    base_name = variant.replace("+conformal", "")
    model = create_model(
        base_name,
        target,
        config["models"].get(base_name, {}),
        config["horizons"],
        config["quantiles"],
        config["random_seed"],
    ).fit(dataset, dataset.row_count)
    if variant.endswith("+conformal"):
        model = _fit_conformal_variant(
            model,
            dataset,
            dataset.row_count,
            base_name,
            config["models"].get(base_name, {}),
            config["folds"]["internal_validation_fraction"],
        )
    save_model(model, destination)


def _plot_forecast_figures(
    selected: Mapping[str, Any],
    run_root: Path,
    figures_dir: Path,
    targets: Sequence[str],
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    coverage_rows = []
    pinball_rows = []
    for target in targets:
        variant = selected[target]["selected_variant"]
        frame = pd.read_csv(run_root / variant / target / "predictions.csv")
        representative = frame[(frame["horizon"] == 1) & (frame["t"].between(240, 287))]
        fig, axis = plt.subplots(figsize=(10, 4.8))
        axis.fill_between(
            representative["t"], representative["q05"], representative["q95"],
            alpha=0.18, label="90% interval"
        )
        axis.fill_between(
            representative["t"], representative["q25"], representative["q75"],
            alpha=0.30, label="50% interval"
        )
        axis.plot(representative["t"], representative["y_true"], label="truth", linewidth=1.2)
        axis.plot(representative["t"], representative["q50"], label="median", linewidth=1.1)
        axis.set_title(f"{target}: {variant}, representative OOS 48 h")
        axis.set_xlabel("forecast origin t")
        axis.set_ylabel(target)
        axis.legend(ncol=4, fontsize=8)
        axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures_dir / f"forecast_{target}_fanchart.png", dpi=150)
        plt.close(fig)

        metrics = json.loads((run_root / variant / target / "metrics.json").read_text())
        scope = _selection_scope(metrics, target)
        coverage_rows.extend(
            ((target, "90%", scope["coverage_90"]), (target, "50%", scope["coverage_50"]))
        )
        for horizon, block in _selection_horizons(metrics, target).items():
            pinball_rows.append((target, int(horizon), block["mean_pinball"]))

    coverage = pd.DataFrame(coverage_rows, columns=["target", "interval", "coverage"])
    pivot = coverage.pivot(index="target", columns="interval", values="coverage")
    axis = pivot.plot(kind="bar", figsize=(8, 4.5), ylim=(0, 1))
    axis.axhline(0.9, color="C0", linestyle="--", alpha=0.6)
    axis.axhline(0.5, color="C1", linestyle="--", alpha=0.6)
    axis.set_ylabel("empirical coverage")
    axis.set_title("Selected-variant interval calibration")
    axis.grid(axis="y", alpha=0.2)
    axis.figure.tight_layout()
    axis.figure.savefig(figures_dir / "forecast_coverage_by_target.png", dpi=150)
    plt.close(axis.figure)

    pinball = pd.DataFrame(pinball_rows, columns=["target", "horizon", "mean_pinball"])
    pivot = pinball.pivot(index="horizon", columns="target", values="mean_pinball")
    axis = pivot.plot(marker="o", figsize=(8, 4.5))
    axis.set_ylabel("mean pinball loss")
    axis.set_title("Selected-variant pinball loss by horizon")
    axis.grid(alpha=0.2)
    axis.figure.tight_layout()
    axis.figure.savefig(figures_dir / "forecast_pinball_by_horizon.png", dpi=150)
    plt.close(axis.figure)


def compare_and_select(config_path: Path, project_root: Path) -> Dict[str, Any]:
    """Execute the frozen rule, refit winners, and create tables/figures."""

    config, resolved_config = load_config(config_path, project_root)
    config_hash = sha256_file(resolved_config)
    dataset = load_dataset(_project_path(project_root, config["dataset_path"]))
    run_root = _project_path(project_root, config["outputs"]["run_root"])
    tables_dir = _project_path(project_root, config["outputs"]["tables_dir"])
    figures_dir = _project_path(project_root, config["outputs"]["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    targets = tuple(config["targets"])
    variants = tuple(config["models"]["order"]) + CONFORMAL_VARIANTS
    comparison_rows = []
    selected: Dict[str, Any] = {}

    for target in targets:
        metrics_by_variant = {
            variant: _read_variant_metrics(run_root, variant, target) for variant in variants
        }
        naive_mae = min(
            _average_horizon_mae(metrics_by_variant[name], target)
            for name in ("persistence_last", "persistence_24h")
        )
        candidates = []
        for variant in variants:
            metrics = metrics_by_variant[variant]
            scope = _selection_scope(metrics, target)
            average_mae = _average_horizon_mae(metrics, target)
            calibrated = _is_calibrated(metrics, target, config)
            competitive = average_mae <= naive_mae + 1e-12
            row = {
                "model": variant,
                "target": target,
                "evaluation_scope": "daylight" if target == "solar_generation" else "pooled",
                "median_mae_avg_horizons": average_mae,
                "mean_pinball": scope["mean_pinball"],
                "coverage_90": scope["coverage_90"],
                "coverage_50": scope["coverage_50"],
                "mean_width_90": scope["mean_width_90"],
                "winkler_90": scope["winkler_90"],
                "calibrated": calibrated,
                "competitive": competitive,
            }
            comparison_rows.append(row)
            if calibrated and competitive and variant not in RAW_MODELS[:3]:
                candidates.append(row)
        if candidates:
            winner = sorted(
                candidates, key=lambda row: (row["mean_pinball"], row["mean_width_90"])
            )[0]
            reason = "lowest mean pinball among calibrated and competitive learned variants"
        else:
            naive_rows = [
                row
                for row in comparison_rows
                if row["target"] == target
                and row["model"] in ("persistence_last", "persistence_24h")
            ]
            winner = sorted(naive_rows, key=lambda row: row["median_mae_avg_horizons"])[0]
            reason = "no learned variant passed both frozen bars; best persistence ships"
        artifact = run_root / "refit" / f"{target}.pt"
        _refit_selected(winner["model"], target, config, dataset, artifact)
        selected[target] = {
            "selected_variant": winner["model"],
            "artifact_path": str(artifact.relative_to(project_root)),
            "metrics": winner,
            "rule_trace": {
                "naive_floor_mae": naive_mae,
                "eligible_learned_variants": [row["model"] for row in candidates],
                "decision": reason,
            },
            "config_sha256": config_hash,
        }

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        tables_dir / "forecast_model_comparison.csv", index=False, float_format="%.10g"
    )
    selection_payload = {
        "experiment_name": config["experiment_name"],
        "dataset_dir": config["dataset_path"],
        "config_sha256": config_hash,
        "selection_rule": config["selection"],
        "targets": selected,
    }
    selection_path = _project_path(project_root, config["outputs"]["selection_path"])
    _json_dump(selection_path, selection_payload)

    calibration_rows = []
    for target, record in selected.items():
        variant = record["selected_variant"]
        frame = pd.read_csv(run_root / variant / target / "predictions.csv")
        frame["hour"] = (
            dataset.building.loc[frame["t"].astype(int), "hour"].to_numpy(dtype=int)
        )
        if target == "solar_generation":
            frame = frame.loc[daylight_mask(dataset, frame["t"].to_numpy(dtype=int))]
        for hour, group in frame.groupby("hour", sort=True):
            calibration_rows.append(
                {
                    "target": target,
                    "hour": int(hour),
                    "selected_variant": variant,
                    "row_count": int(len(group)),
                    "coverage_90": empirical_coverage(
                        group["y_true"], group["q05"], group["q95"]
                    ),
                }
            )
    pd.DataFrame(calibration_rows).to_csv(
        tables_dir / "forecast_calibration_by_hour.csv", index=False, float_format="%.10g"
    )
    _plot_forecast_figures(selected, run_root, figures_dir, targets)
    return selection_payload
