"""Unit tests for derived metrics computed from harness traces."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energy_optimisation.evaluation.metrics import REQUIRED_TRACE_COLUMNS, compute_derived_metrics


def make_trace(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "net_electricity_consumption": 1.0,
        "dhw_storage_soc": 0.5,
        "electrical_storage_soc": 0.5,
        "indoor_dry_bulb_temperature": 24.0,
        "indoor_dry_bulb_temperature_cooling_set_point": 25.0,
        "action_requested_dhw_storage": 0.0,
        "action_requested_electrical_storage": 0.0,
        "action_requested_cooling_device": 0.0,
        "action_applied_dhw_storage": 0.0,
        "action_applied_electrical_storage": 0.0,
        "action_applied_cooling_device": 0.0,
        "solar_generation": 1.0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows], columns=REQUIRED_TRACE_COLUMNS)


BAND = 2.0
RESERVE_LOW = 0.2
RESERVE_HIGH = 0.9


def test_comfort_violation_hours_count_hot_side_only() -> None:
    trace = make_trace(
        [
            {"indoor_dry_bulb_temperature": 27.5},  # > setpoint + band -> violation
            {"indoor_dry_bulb_temperature": 27.0},  # == boundary -> no violation
            {"indoor_dry_bulb_temperature": 20.0},  # cold side is not counted here
        ]
    )

    metrics = compute_derived_metrics(trace, comfort_band_c=BAND, reserve_low_soc=RESERVE_LOW, reserve_high_soc=RESERVE_HIGH)

    assert metrics["comfort_violation_hours"] == 1
    assert metrics["comfort_violation_proportion"] == pytest.approx(1 / 3)


def test_soc_min_max_and_peak() -> None:
    trace = make_trace(
        [
            {"electrical_storage_soc": 0.4, "dhw_storage_soc": 0.6, "net_electricity_consumption": 2.5},
            {"electrical_storage_soc": 0.8, "dhw_storage_soc": 0.3, "net_electricity_consumption": 1.1},
        ]
    )

    metrics = compute_derived_metrics(trace, comfort_band_c=BAND, reserve_low_soc=RESERVE_LOW, reserve_high_soc=RESERVE_HIGH)

    assert metrics["electrical_storage_soc_min"] == pytest.approx(0.4)
    assert metrics["electrical_storage_soc_max"] == pytest.approx(0.8)
    assert metrics["dhw_storage_soc_min"] == pytest.approx(0.3)
    assert metrics["dhw_storage_soc_max"] == pytest.approx(0.6)
    assert metrics["peak_net_demand_kw"] == pytest.approx(2.5)


def test_clipping_event_counts_requested_outside_bounds() -> None:
    trace = make_trace(
        [
            {"action_requested_electrical_storage": 1.5, "action_applied_electrical_storage": 1.0},
            {"action_requested_dhw_storage": -2.0, "action_applied_dhw_storage": -1.0},
            {"action_requested_cooling_device": -0.5, "action_applied_cooling_device": 0.0},
            {},
        ]
    )

    metrics = compute_derived_metrics(trace, comfort_band_c=BAND, reserve_low_soc=RESERVE_LOW, reserve_high_soc=RESERVE_HIGH)

    assert metrics["clipping_event_count"] == 3


def test_reserve_events_count_requested_band_violations() -> None:
    trace = make_trace(
        [
            # discharge request with SoC at/below the low reserve bound
            {"action_requested_electrical_storage": 0.5, "electrical_storage_soc": 0.2},
            # charge request with SoC at/above the high reserve bound
            {"action_requested_dhw_storage": -0.5, "dhw_storage_soc": 0.9},
            # within band: not a reserve event
            {"action_requested_electrical_storage": 0.5, "electrical_storage_soc": 0.5},
            {"action_requested_dhw_storage": -0.5, "dhw_storage_soc": 0.5},
        ]
    )

    metrics = compute_derived_metrics(trace, comfort_band_c=BAND, reserve_low_soc=RESERVE_LOW, reserve_high_soc=RESERVE_HIGH)

    assert metrics["reserve_event_count"] == 2


def test_solar_self_consumption_and_grid_limit() -> None:
    trace = make_trace(
        [
            {"solar_generation": 6.0, "net_electricity_consumption": -2.0},
            {"solar_generation": 4.0, "net_electricity_consumption": 3.0},
        ]
    )

    without_limit = compute_derived_metrics(
        trace, comfort_band_c=BAND, reserve_low_soc=RESERVE_LOW, reserve_high_soc=RESERVE_HIGH
    )
    assert without_limit["solar_self_consumption_proportion"] == pytest.approx(0.8)
    assert without_limit["grid_limit_exceedance_count"] is None

    with_limit = compute_derived_metrics(
        trace, comfort_band_c=BAND, reserve_low_soc=RESERVE_LOW, reserve_high_soc=RESERVE_HIGH, grid_limit_kw=2.5
    )
    assert with_limit["grid_limit_exceedance_count"] == 1

    no_solar = make_trace([{"solar_generation": 0.0}])
    zero_generation = compute_derived_metrics(
        no_solar, comfort_band_c=BAND, reserve_low_soc=RESERVE_LOW, reserve_high_soc=RESERVE_HIGH
    )
    assert zero_generation["solar_self_consumption_proportion"] == 0.0


def test_missing_required_column_raises() -> None:
    trace = make_trace([{}]).drop(columns=["solar_generation"])

    with pytest.raises(KeyError):
        compute_derived_metrics(trace, comfort_band_c=BAND, reserve_low_soc=RESERVE_LOW, reserve_high_soc=RESERVE_HIGH)
