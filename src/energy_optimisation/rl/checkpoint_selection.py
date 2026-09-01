"""Frozen checkpoint selection rule (plan §C3).

Plan reference: docs/plans/week3-implementation-plan.md §0 (decision table) and
§C3: lowest dev-window ``cost_total``, tie-break lower
``discomfort_proportion``. The rule was frozen before training; this module
only executes it — it never judges or reweights.
"""

from __future__ import annotations

import pandas as pd

# Exact column order of results/runs/ppo/<seed>/evaluations.csv (plan §C2).
EVALUATION_COLUMNS = (
    "checkpoint",
    "timestep",
    "episode_return",
    "cost_total",
    "all_time_peak_average",
    "electricity_consumption_total",
    "discomfort_proportion",
    "discomfort_hot_proportion",
    "ramping_average",
    "zero_net_energy",
    "comfort_violation_hours",
    "grid_limit_exceedances",
    "clipping_events",
    "reserve_events",
)

SELECTION_RULE_KEYS = ("cost_total", "discomfort_proportion")
SELECTION_RULE_TEXT = (
    "lowest cost_total; tie-break lower discomfort_proportion "
    "(frozen in docs/plans/week3-implementation-plan.md section 0 before training)"
)


def select_best_checkpoint(evaluations: pd.DataFrame) -> pd.Series:
    """Return the row chosen by the frozen selection rule.

    Stable sort so equal ``(cost_total, discomfort_proportion)`` ties resolve
    to the earlier training checkpoint (fewer environment steps), keeping the
    outcome deterministic.
    """

    missing = [key for key in SELECTION_RULE_KEYS if key not in evaluations.columns]
    if missing:
        raise KeyError(f"evaluations lack selection-rule columns: {', '.join(missing)}")
    ordered = evaluations.sort_values(
        list(SELECTION_RULE_KEYS), kind="mergesort", na_position="last"
    )
    return ordered.iloc[0]
