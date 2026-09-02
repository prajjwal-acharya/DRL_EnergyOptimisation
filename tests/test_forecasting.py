"""Contracts for the Week-4 causal probabilistic forecasting package."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energy_optimisation.forecasting.api import ForecastProvider
from energy_optimisation.forecasting.data import (
    STATIC_FEATURE_NAMES,
    build_forecast_frame,
    load_dataset,
    make_folds,
)
from energy_optimisation.forecasting.metrics import (
    empirical_coverage,
    enforce_quantile_monotonicity,
    mae,
    pinball_loss,
    winkler_score,
)
from energy_optimisation.forecasting.models import (
    ClimatologyHourly,
    Conformalized,
    GruQuantile,
    LinearQuantile,
    Persistence24h,
    PersistenceLast,
    save_model,
)

DATASET_DIR = PROJECT_ROOT / "data/raw/citylearn_challenge_2023_phase_1"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATASET_DIR)


def test_dataset_row_counts_and_columns(dataset) -> None:
    assert dataset.row_count == 720
    assert len(STATIC_FEATURE_NAMES) == 22
    assert {"solar_generation", "non_shiftable_load", "cooling_demand"}.issubset(
        dataset.building.columns
    )


def test_forecast_frame_causality(dataset) -> None:
    t = 100
    original, _ = build_forecast_frame(dataset, t, "cooling_demand", 3)
    future_changed = dataset.copy()
    future_changed.building.loc[t + 1 :, :] = 999999.0
    future_changed.weather.loc[t + 1 :, :] = 999999.0
    future_changed.pricing.loc[t + 1 :, :] = 999999.0
    future_changed.carbon.loc[t + 1 :, :] = 999999.0
    future_features, _ = build_forecast_frame(future_changed, t, "cooling_demand", 3)
    np.testing.assert_array_equal(future_features, original)

    present_changed = dataset.copy()
    present_changed.weather.at[t, "outdoor_dry_bulb_temperature_predicted_1"] += 1.0
    present_features, _ = build_forecast_frame(present_changed, t, "cooling_demand", 3)
    assert not np.array_equal(present_features, original)


def test_folds_match_frozen_scheme(dataset) -> None:
    folds = make_folds(
        dataset, {"initial_train_size": 240, "fold_size": 40, "fold_count": 12}
    )
    assert len(folds) == 12
    assert (folds[0].train_end, folds[0].test_start, folds[0].test_end) == (240, 240, 280)
    assert (folds[-1].train_end, folds[-1].test_start, folds[-1].test_end) == (
        680,
        680,
        720,
    )
    assert all(fold.train_end == fold.test_start for fold in folds)


def test_metrics_on_hand_computed_cases() -> None:
    truth = np.asarray([0.0, 2.0, 4.0])
    point = np.asarray([1.0, 2.0, 2.0])
    assert mae(truth, point) == pytest.approx(1.0)
    assert pinball_loss(truth, point, 0.5) == pytest.approx(0.5)
    assert empirical_coverage(truth, [0.0, 1.0, 3.0], [1.0, 2.0, 5.0]) == 1.0
    # Widths are 1; one miss is two below the lower bound with alpha=.1.
    assert winkler_score([0.0], [2.0], [3.0], 0.1) == pytest.approx(41.0)
    with pytest.raises(ValueError, match="NaN"):
        mae([1.0, np.nan], [1.0, 2.0])


def test_quantile_monotonicity_enforcement() -> None:
    repaired = enforce_quantile_monotonicity([1.0, 0.0, 2.0, 1.5, 3.0])
    np.testing.assert_array_equal(repaired, [1.0, 1.0, 2.0, 2.0, 3.0])


def test_persistence_definitions(dataset) -> None:
    synthetic = dataset.copy()
    synthetic.building["non_shiftable_load"] = np.arange(720, dtype=float)
    last = PersistenceLast("non_shiftable_load").fit(synthetic, 48)
    day = Persistence24h("non_shiftable_load").fit(synthetic, 48)
    assert last.predict_quantiles(synthetic, 40, 3)[0.50] == 40.0
    assert day.predict_quantiles(synthetic, 40, 3)[0.50] == 19.0


def test_climatology_uses_only_training_hours(dataset) -> None:
    synthetic = dataset.copy()
    synthetic.building.loc[:239, "cooling_demand"] = 1.0
    synthetic.building.loc[240:, "cooling_demand"] = 999.0
    model = ClimatologyHourly("cooling_demand").fit(synthetic, 240)
    assert model.predict_quantiles(synthetic, 300, 1)[0.50] == pytest.approx(1.0)


def test_linear_and_gru_deterministic_per_seed(dataset) -> None:
    linear_kwargs = {"steps": 8, "learning_rate": 0.01, "seed": 42}
    linear_a = LinearQuantile("solar_generation", **linear_kwargs).fit(dataset, 80)
    linear_b = LinearQuantile("solar_generation", **linear_kwargs).fit(dataset, 80)
    assert linear_a.predict_all(dataset, 70) == linear_b.predict_all(dataset, 70)

    gru_kwargs = {
        "hidden_size": 4,
        "max_epochs": 2,
        "patience": 2,
        "batch_size": 16,
        "seed": 42,
    }
    gru_a = GruQuantile("solar_generation", **gru_kwargs).fit(dataset, 80)
    gru_b = GruQuantile("solar_generation", **gru_kwargs).fit(dataset, 80)
    assert gru_a.predict_all(dataset, 70) == gru_b.predict_all(dataset, 70)


def test_gru_quantiles_monotone(dataset) -> None:
    model = GruQuantile(
        "non_shiftable_load", hidden_size=4, max_epochs=2, patience=2, batch_size=16
    ).fit(dataset, 80)
    predictions = model.predict_all(dataset, 70)
    for values in predictions.values():
        ordered = [values[tau] for tau in sorted(values)]
        assert ordered == sorted(ordered)


def test_conformal_widens_intervals(dataset) -> None:
    base = PersistenceLast("cooling_demand").fit(dataset, 100)
    model = Conformalized(base).calibrate(dataset, list(range(70, 90)))
    raw = base.predict_quantiles(dataset, 95, 1)
    adjusted = model.predict_quantiles(dataset, 95, 1)
    assert adjusted[0.05] <= raw[0.05]
    assert adjusted[0.95] >= raw[0.95]
    assert adjusted[0.25] <= raw[0.25]
    assert adjusted[0.75] >= raw[0.75]


def test_provider_causality(dataset, tmp_path: Path) -> None:
    targets = {}
    for target in ("solar_generation", "non_shiftable_load", "cooling_demand"):
        artifact = tmp_path / f"{target}.pt"
        save_model(PersistenceLast(target).fit(dataset, 720), artifact)
        targets[target] = {"artifact_path": str(artifact), "selected_variant": "persistence_last"}
    selection = tmp_path / "selected_models.json"
    selection.write_text(json.dumps({"targets": targets}))

    baseline = ForecastProvider(selection, dataset=dataset.copy())
    expected = baseline.predict_quantiles(100)
    future = dataset.copy()
    future.building.loc[101:, :] = 99999.0
    assert ForecastProvider(selection, dataset=future).predict_quantiles(100) == expected
    present = dataset.copy()
    present.building.at[100, "solar_generation"] += 1.0
    assert ForecastProvider(selection, dataset=present).predict_quantiles(100) != expected
    assert baseline.feature_vector(100, "point").shape == (9,)
    assert baseline.feature_vector(100, "interval").shape == (36,)
    assert baseline.feature_vector(0, "point").shape == (9,)
    assert baseline.feature_vector(0, "interval").shape == (36,)
    assert baseline.predict_quantiles(0)["solar_generation"][1][0.50] == pytest.approx(
        dataset.building.at[0, "solar_generation"]
    )
