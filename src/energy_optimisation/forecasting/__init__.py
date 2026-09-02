"""Probabilistic forecasting contracts used by Weeks 4 and 5."""

from .api import ForecastProvider
from .data import Dataset, Fold, build_forecast_frame, load_dataset, make_folds
from .metrics import (
    empirical_coverage,
    enforce_quantile_monotonicity,
    mae,
    mean_interval_width,
    pinball_loss,
    rmse,
    winkler_score,
)

__all__ = [
    "Dataset",
    "Fold",
    "ForecastProvider",
    "build_forecast_frame",
    "empirical_coverage",
    "enforce_quantile_monotonicity",
    "load_dataset",
    "mae",
    "make_folds",
    "mean_interval_width",
    "pinball_loss",
    "rmse",
    "winkler_score",
]
