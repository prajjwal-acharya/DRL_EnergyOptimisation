"""Derived metrics computed from evaluation-harness traces.

Plan reference: docs/plans/week2-implementation-plan.md §B4. All metrics are pure
functions of the runner's trace plus explicitly passed thresholds (comfort
band, SoC reserve band, grid limit) so that frozen constants stay in config,
never in code.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


REQUIRED_TRACE_COLUMNS = (
    "net_electricity_consumption",
    "dhw_storage_soc",
    "electrical_storage_soc",
    "indoor_dry_bulb_temperature",
    "indoor_dry_bulb_temperature_cooling_set_point",
    "action_requested_dhw_storage",
    "action_requested_electrical_storage",
    "action_requested_cooling_device",
    "action_applied_dhw_storage",
    "action_applied_electrical_storage",
    "action_applied_cooling_device",
    "solar_generation",
)


def _require_columns(trace: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_TRACE_COLUMNS if column not in trace.columns]
    if missing:
        raise KeyError(f"Trace lacks required columns: {', '.join(missing)}")


def compute_derived_metrics(
    trace: pd.DataFrame,
    *,
    comfort_band_c: float,
    reserve_low_soc: float,
    reserve_high_soc: float,
    grid_limit_kw: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute the §B4 metric set from one controller trace.

    Parameters
    ----------
    trace:
        Runner trace DataFrame (see ``evaluation.runner.TRACE_COLUMNS``).
    comfort_band_c:
        Comfort half-band ΔT in °C; a step violates comfort when
        ``indoor_dry_bulb_temperature > cooling_set_point + ΔT``.
    reserve_low_soc / reserve_high_soc:
        SoC reserve band [low, high]. A *requested* storage action is a
        reserve event when it pushes further past the band than the current
        SoC already is: discharge request (> 0) with SoC <= low, or charge
        request (< 0) with SoC >= high. CityLearn enforces physical bounds
        internally; requested violations are counted here.
    grid_limit_kw:
        Grid import limit ``P_grid,max``; steps with net consumption above it
        are counted. Pass None while the limit is pending first B0 run.

    Returns
    -------
    Dict[str, Any]
        JSON-safe metrics: comfort_violation_hours, comfort_violation_proportion,
        per-storage SoC min/max, clipping_event_count, reserve_event_count,
        peak_net_demand_kw, solar_self_consumption_proportion, and
        grid_limit_exceedance_count (None when no grid limit is configured).
    """

    _require_columns(trace)

    hot_violation = (
        trace["indoor_dry_bulb_temperature"]
        - (trace["indoor_dry_bulb_temperature_cooling_set_point"] + comfort_band_c)
    ) > 0.0

    clipping = pd.DataFrame(
        {
            name: trace[f"action_requested_{name}"] != trace[f"action_applied_{name}"]
            for name in ("dhw_storage", "electrical_storage", "cooling_device")
        }
    )

    reserve_events = pd.Series(0, index=trace.index, dtype=int)
    for storage in ("dhw_storage", "electrical_storage"):
        soc = trace[f"{storage}_soc"]
        requested = trace[f"action_requested_{storage}"]
        reserve_events += (
            (requested > 0.0) & (soc <= reserve_low_soc)
        ) | (
            (requested < 0.0) & (soc >= reserve_high_soc)
        )

    generated_total = float(trace["solar_generation"].clip(lower=0.0).sum())
    exported_total = float((-trace["net_electricity_consumption"]).clip(lower=0.0).sum())
    if generated_total > 0.0:
        solar_self_consumption = max(0.0, 1.0 - exported_total / generated_total)
    else:
        solar_self_consumption = 0.0

    return {
        "comfort_violation_hours": int(hot_violation.sum()),
        "comfort_violation_proportion": float(hot_violation.mean()),
        "electrical_storage_soc_min": float(trace["electrical_storage_soc"].min()),
        "electrical_storage_soc_max": float(trace["electrical_storage_soc"].max()),
        "dhw_storage_soc_min": float(trace["dhw_storage_soc"].min()),
        "dhw_storage_soc_max": float(trace["dhw_storage_soc"].max()),
        "clipping_event_count": int(clipping.to_numpy().sum()),
        "reserve_event_count": int(reserve_events.sum()),
        "peak_net_demand_kw": float(trace["net_electricity_consumption"].max()),
        "solar_self_consumption_proportion": float(solar_self_consumption),
        "grid_limit_exceedance_count": (
            None
            if grid_limit_kw is None
            else int((trace["net_electricity_consumption"] > grid_limit_kw).sum())
        ),
    }
