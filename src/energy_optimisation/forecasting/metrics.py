"""NaN-strict numpy metrics for point and interval forecasts."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _paired(y_true: Iterable[float], y_pred: Iterable[float]) -> tuple:
    truth = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != predicted.shape:
        raise ValueError(f"metric array shapes differ: {truth.shape} != {predicted.shape}")
    if truth.size == 0:
        raise ValueError("metrics require at least one value")
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("metrics reject NaN and infinite values")
    return truth, predicted


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    truth, predicted = _paired(y_true, y_pred)
    return float(np.mean(np.abs(truth - predicted)))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    truth, predicted = _paired(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(truth - predicted))))


def pinball_loss(y_true: Iterable[float], y_q: Iterable[float], tau: float) -> float:
    truth, predicted = _paired(y_true, y_q)
    quantile = float(tau)
    if not 0.0 < quantile < 1.0:
        raise ValueError("tau must lie strictly between 0 and 1")
    residual = truth - predicted
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def empirical_coverage(
    y_true: Iterable[float], q_lo: Iterable[float], q_hi: Iterable[float]
) -> float:
    truth, lower = _paired(y_true, q_lo)
    _, upper = _paired(truth, q_hi)
    if np.any(lower > upper):
        raise ValueError("interval lower bounds exceed upper bounds")
    return float(np.mean((truth >= lower) & (truth <= upper)))


def mean_interval_width(q_lo: Iterable[float], q_hi: Iterable[float]) -> float:
    lower, upper = _paired(q_lo, q_hi)
    if np.any(lower > upper):
        raise ValueError("interval lower bounds exceed upper bounds")
    return float(np.mean(upper - lower))


def winkler_score(
    y_true: Iterable[float],
    q_lo: Iterable[float],
    q_hi: Iterable[float],
    alpha: float,
) -> float:
    truth, lower = _paired(y_true, q_lo)
    _, upper = _paired(truth, q_hi)
    miss_probability = float(alpha)
    if not 0.0 < miss_probability < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if np.any(lower > upper):
        raise ValueError("interval lower bounds exceed upper bounds")
    score = upper - lower
    below = truth < lower
    above = truth > upper
    score = score + (2.0 / miss_probability) * (lower - truth) * below
    score = score + (2.0 / miss_probability) * (truth - upper) * above
    return float(np.mean(score))


def enforce_quantile_monotonicity(values: Iterable[float]) -> np.ndarray:
    """Repair crossing quantiles with the frozen cumulative-maximum rule."""

    quantiles = np.asarray(values, dtype=np.float64)
    if quantiles.ndim == 0 or quantiles.shape[-1] == 0:
        raise ValueError("quantile array must have a non-empty final dimension")
    if not np.isfinite(quantiles).all():
        raise ValueError("quantile monotonicity helper rejects NaN and infinite values")
    return np.maximum.accumulate(quantiles, axis=-1)
