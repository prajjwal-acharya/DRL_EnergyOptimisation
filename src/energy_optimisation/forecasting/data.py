"""Causal data access for the frozen Week-4 forecasting experiment.

This module is the only place that turns the four CityLearn CSV files into model
features.  Keeping that operation here makes the no-lookahead rule inspectable and
unit-testable instead of relying on individual models to remember it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

TARGETS = ("solar_generation", "non_shiftable_load", "cooling_demand")
HORIZONS = (1, 2, 3)
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)

BUILDING_COLUMNS = (
    "month",
    "hour",
    "day_type",
    "daylight_savings_status",
    "indoor_dry_bulb_temperature",
    "average_unmet_cooling_setpoint_difference",
    "indoor_relative_humidity",
    "non_shiftable_load",
    "dhw_demand",
    "cooling_demand",
    "heating_demand",
    "solar_generation",
    "occupant_count",
    "indoor_dry_bulb_temperature_cooling_set_point",
    "indoor_dry_bulb_temperature_heating_set_point",
    "hvac_mode",
)
WEATHER_ACTUAL_COLUMNS = (
    "outdoor_dry_bulb_temperature",
    "outdoor_relative_humidity",
    "diffuse_solar_irradiance",
    "direct_solar_irradiance",
)
WEATHER_PREDICTED_COLUMNS = tuple(
    f"{name}_predicted_{horizon}"
    for name in (
        "outdoor_dry_bulb_temperature",
        "outdoor_relative_humidity",
        "diffuse_solar_irradiance",
        "direct_solar_irradiance",
    )
    for horizon in HORIZONS
)
WEATHER_FORECAST_FEATURE_COLUMNS = tuple(
    f"{name}_predicted_{horizon}"
    for name in (
        "outdoor_dry_bulb_temperature",
        "diffuse_solar_irradiance",
        "direct_solar_irradiance",
    )
    for horizon in HORIZONS
)
PRICING_COLUMNS = ("electricity_pricing",) + tuple(
    f"electricity_pricing_predicted_{horizon}" for horizon in HORIZONS
)
CARBON_COLUMNS = ("carbon_intensity",)

STATIC_FEATURE_NAMES = (
    "sin_hour",
    "cos_hour",
    "day_type",
    "outdoor_dry_bulb_temperature",
    "diffuse_solar_irradiance",
    "direct_solar_irradiance",
) + WEATHER_FORECAST_FEATURE_COLUMNS + PRICING_COLUMNS + CARBON_COLUMNS + (
    "target_current",
    "target_lag_24",
)

SEQUENCE_FEATURE_NAMES = (
    "target",
    "outdoor_dry_bulb_temperature",
    "diffuse_solar_irradiance",
    "direct_solar_irradiance",
    "electricity_pricing",
)


@dataclass
class Dataset:
    """Aligned, numeric views of the four immutable source CSVs."""

    building: pd.DataFrame
    weather: pd.DataFrame
    pricing: pd.DataFrame
    carbon: pd.DataFrame
    source_directory: Path

    @property
    def row_count(self) -> int:
        return len(self.building)

    def copy(self) -> "Dataset":
        """Return an independent copy, useful for causality mutation tests."""

        return Dataset(
            building=self.building.copy(deep=True),
            weather=self.weather.copy(deep=True),
            pricing=self.pricing.copy(deep=True),
            carbon=self.carbon.copy(deep=True),
            source_directory=self.source_directory,
        )

    def through(self, origin_t: int) -> "Dataset":
        """Return a defensive view containing rows 0..t and no future actuals."""

        stop = int(origin_t) + 1
        if stop <= 0 or stop > self.row_count:
            raise IndexError(f"causal dataset boundary is outside 0..{self.row_count - 1}")
        return Dataset(
            building=self.building.iloc[:stop].copy().reset_index(drop=True),
            weather=self.weather.iloc[:stop].copy().reset_index(drop=True),
            pricing=self.pricing.iloc[:stop].copy().reset_index(drop=True),
            carbon=self.carbon.iloc[:stop].copy().reset_index(drop=True),
            source_directory=self.source_directory,
        )


@dataclass(frozen=True)
class Fold:
    """One expanding-window rolling-origin fold (end bounds are exclusive)."""

    number: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_indices(self) -> range:
        return range(self.train_start, self.train_end)

    @property
    def test_indices(self) -> range:
        return range(self.test_start, self.test_end)


def _load_numeric_csv(path: Path, required: Sequence[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"required dataset file does not exist: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")
    numeric = frame.copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="raise").astype(np.float64)
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path.name} contains NaN or infinite values")
    return numeric


def load_dataset(dataset_dir: Path) -> Dataset:
    """Load and validate the pinned 720-row CityLearn forecasting inputs."""

    directory = Path(dataset_dir).expanduser().resolve()
    building = _load_numeric_csv(directory / "Building_1.csv", BUILDING_COLUMNS)
    weather = _load_numeric_csv(
        directory / "weather.csv", WEATHER_ACTUAL_COLUMNS + WEATHER_PREDICTED_COLUMNS
    )
    pricing = _load_numeric_csv(directory / "pricing.csv", PRICING_COLUMNS)
    carbon = _load_numeric_csv(directory / "carbon_intensity.csv", CARBON_COLUMNS)
    lengths = {len(building), len(weather), len(pricing), len(carbon)}
    if lengths != {720}:
        raise ValueError(
            "forecast source files must be aligned at exactly 720 rows; "
            f"found building={len(building)}, weather={len(weather)}, "
            f"pricing={len(pricing)}, carbon={len(carbon)}"
        )
    return Dataset(building, weather, pricing, carbon, directory)


def _validate_request(dataset: Dataset, origin_t: int, target: str) -> None:
    if target not in TARGETS:
        raise KeyError(f"unsupported forecast target {target!r}; expected one of {TARGETS}")
    if origin_t < 24 or origin_t >= dataset.row_count:
        raise IndexError(
            f"forecast origin must be in [24, {dataset.row_count - 1}], got {origin_t}"
        )


def build_static_features(dataset: Dataset, origin_t: int, target: str) -> np.ndarray:
    """Build the frozen 22-value feature vector using information available at ``t``."""

    _validate_request(dataset, origin_t, target)
    t = int(origin_t)
    hour = float(dataset.building.at[t, "hour"])
    angle = 2.0 * np.pi * ((hour - 1.0) % 24.0) / 24.0
    values = [
        np.sin(angle),
        np.cos(angle),
        float(dataset.building.at[t, "day_type"]),
        float(dataset.weather.at[t, "outdoor_dry_bulb_temperature"]),
        float(dataset.weather.at[t, "diffuse_solar_irradiance"]),
        float(dataset.weather.at[t, "direct_solar_irradiance"]),
    ]
    values.extend(
        float(dataset.weather.at[t, column]) for column in WEATHER_FORECAST_FEATURE_COLUMNS
    )
    values.extend(float(dataset.pricing.at[t, column]) for column in PRICING_COLUMNS)
    values.append(float(dataset.carbon.at[t, "carbon_intensity"]))
    values.extend(
        (
            float(dataset.building.at[t, target]),
            float(dataset.building.at[t - 24, target]),
        )
    )
    features = np.asarray(values, dtype=np.float64)
    if features.shape != (22,) or not np.isfinite(features).all():
        raise ValueError("forecast feature construction did not produce 22 finite values")
    return features


def build_sequence_features(dataset: Dataset, origin_t: int, target: str) -> np.ndarray:
    """Return the causal 24×5 history consumed by the GRU."""

    _validate_request(dataset, origin_t, target)
    start = int(origin_t) - 23
    stop = int(origin_t) + 1
    sequence = np.column_stack(
        (
            dataset.building.loc[start : stop - 1, target].to_numpy(dtype=np.float64),
            dataset.weather.loc[
                start : stop - 1, "outdoor_dry_bulb_temperature"
            ].to_numpy(dtype=np.float64),
            dataset.weather.loc[
                start : stop - 1, "diffuse_solar_irradiance"
            ].to_numpy(dtype=np.float64),
            dataset.weather.loc[
                start : stop - 1, "direct_solar_irradiance"
            ].to_numpy(dtype=np.float64),
            dataset.pricing.loc[start : stop - 1, "electricity_pricing"].to_numpy(
                dtype=np.float64
            ),
        )
    )
    if sequence.shape != (24, 5) or not np.isfinite(sequence).all():
        raise ValueError("GRU history construction did not produce 24x5 finite values")
    return sequence


def build_forecast_frame(
    dataset: Dataset, origin_t: int, target: str, horizon: int
) -> Tuple[np.ndarray, float]:
    """Return causal features at ``t`` and the evaluation label at ``t+h``.

    The future target is returned as a label only.  It is never part of the feature
    vector and is deliberately read after feature construction.
    """

    if int(horizon) not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}, got {horizon}")
    x = build_static_features(dataset, int(origin_t), target)
    target_index = int(origin_t) + int(horizon)
    if target_index >= dataset.row_count:
        raise IndexError(
            f"target row t+h={target_index} is outside the {dataset.row_count}-row dataset"
        )
    return x, float(dataset.building.at[target_index, target])


def make_folds(dataset: Dataset, scheme: Mapping[str, int]) -> List[Fold]:
    """Materialise the frozen 12×40 expanding-window fold scheme."""

    initial = int(scheme.get("initial_train_size", 240))
    fold_size = int(scheme.get("fold_size", 40))
    count = int(scheme.get("fold_count", 12))
    if initial <= 24 or fold_size <= 0 or count <= 0:
        raise ValueError("fold scheme values must be positive and leave at least 24 history rows")
    if initial + fold_size * count != dataset.row_count:
        raise ValueError(
            "fold scheme must cover the 720-row dataset exactly after the initial segment"
        )
    return [
        Fold(
            number=index + 1,
            train_start=0,
            train_end=initial + fold_size * index,
            test_start=initial + fold_size * index,
            test_end=initial + fold_size * (index + 1),
        )
        for index in range(count)
    ]


def training_origins(train_end: int, max_horizon: int = 3) -> np.ndarray:
    """Origins whose entire target horizon is inside an exclusive train end."""

    return np.arange(24, int(train_end) - int(max_horizon), dtype=np.int64)


def training_arrays(
    dataset: Dataset,
    target: str,
    train_end: int,
    horizons: Sequence[int] = HORIZONS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return origins, static inputs, sequences, and multi-horizon labels."""

    horizon_values = tuple(int(value) for value in horizons)
    origins = training_origins(train_end, max(horizon_values))
    static = np.stack([build_static_features(dataset, int(t), target) for t in origins])
    sequences = np.stack([build_sequence_features(dataset, int(t), target) for t in origins])
    labels = np.asarray(
        [
            [float(dataset.building.at[int(t) + horizon, target]) for horizon in horizon_values]
            for t in origins
        ],
        dtype=np.float64,
    )
    return origins, static, sequences, labels


def daylight_mask(dataset: Dataset, origins: Sequence[int]) -> np.ndarray:
    """Whether each origin has positive measured direct+diffuse irradiance."""

    indices = np.asarray(origins, dtype=np.int64)
    direct = dataset.weather.loc[indices, "direct_solar_irradiance"].to_numpy(dtype=float)
    diffuse = dataset.weather.loc[indices, "diffuse_solar_irradiance"].to_numpy(dtype=float)
    return (direct + diffuse) > 0.0
