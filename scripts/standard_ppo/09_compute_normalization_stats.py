"""Compute frozen per-feature observation-normalisation stats for Week 3 PPO.

Plan reference: docs/plans/week3-implementation-plan.md §A1. The RL adapter consumes
per-feature ``(offset, scale)`` pairs so the plain CityLearn observation enters
the policy normalised. Stats are computed ONCE by this script and written into
``configs/week3-ppo.yaml``; hand-editing the stats afterwards is forbidden.

Rule (frozen, deterministic):

1. Features whose schema-static observation range is exactly ``[0, 1]``
   (SoCs and other ratio-type signals) use identity: ``offset = 0``, ``scale = 1``.
2. Other features with a column in the B0 dev-window trace
   (``results/runs/baselines/b0_neutral/dev/trace.csv``) use min-max statistics from
   that trace: ``offset = min``, ``scale = max - min``.
3. Remaining features use the schema-static observation ranges (data-derived
   limits exposed by the loaded CityLearn observation space) the same way.

Every scale carries an epsilon guard: spans <= epsilon fall back to
``scale = 1`` so the transform stays finite and invertible.

The script is idempotent: rerunning it with unchanged inputs reproduces the
same stats byte-for-byte. It never modifies anything under ``results/``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from energy_optimisation.environment import load_environment
from energy_optimisation.observation_names import observation_layout

SCHEMA_PATH = PROJECT_ROOT / "configs/schema-building1.json"
TRACE_PATH = PROJECT_ROOT / "results/runs/baselines/b0_neutral/dev/trace.csv"
CONFIG_PATH = PROJECT_ROOT / "configs/week3-ppo.yaml"
DEFAULT_EPSILON = 1e-8

# Non-action trace columns eligible as empirical statistic sources.
TRACE_STATISTIC_COLUMNS = (
    "hour",
    "electricity_pricing",
    "net_electricity_consumption",
    "indoor_dry_bulb_temperature",
    "indoor_dry_bulb_temperature_cooling_set_point",
    "solar_generation",
)

# Frozen CMDP reward constants (docs/reference/cmdp-spec.md §4; plan §0). Written into
# the config skeleton so the adapter reads them from config, never from code.
REWARD_CONSTANTS = {
    "w_E": 1.0,
    "w_P": 1.0,
    "w_C": 10.0,
    "E_bar_b0": 0.477229108554339,
    "P_ref": 7.694016456604004,
    "comfort_band_c": 2.0,
}


def _static_ranges() -> Dict[str, tuple]:
    """Return per-feature static ``(low, high)`` ranges from the loaded schema."""

    environment = load_environment(str(SCHEMA_PATH), central_agent=True)
    space = environment.observation_space[0]
    low = np.asarray(space.low, dtype=float)
    high = np.asarray(space.high, dtype=float)
    return {
        name: (float(low[i]), float(high[i]))
        for i, name in enumerate(observation_layout(SCHEMA_PATH))
    }


def _trace_statistics() -> Dict[str, tuple]:
    """Return per-column ``(min, max)`` statistics from the B0 dev trace."""

    frame = pd.read_csv(TRACE_PATH, keep_default_na=False, float_precision="round_trip")
    statistics: Dict[str, tuple] = {}
    for column in TRACE_STATISTIC_COLUMNS:
        if column not in frame.columns:
            raise KeyError(f"B0 dev trace lacks expected column: {column}")
        values = frame[column].astype(float)
        statistics[column] = (float(values.min()), float(values.max()))
    return statistics


def compute_feature_stats(epsilon: float) -> Dict[str, Dict[str, float]]:
    """Compute ordered ``{name: {offset, scale}}`` stats for every feature."""

    static = _static_ranges()
    trace = _trace_statistics()

    features: Dict[str, Dict[str, float]] = {}
    for name, (low, high) in static.items():
        if low == 0.0 and high == 1.0:
            features[name] = {"offset": 0.0, "scale": 1.0}
            continue
        if name in trace:
            offset, maximum = trace[name]
        else:
            offset, maximum = low, high
        span = maximum - offset
        if span <= epsilon:
            scale = 1.0
        else:
            scale = span
        features[name] = {"offset": offset, "scale": scale}
    return features


def build_config_skeleton() -> Dict[str, Any]:
    """Return the minimal Week 3 config skeleton (Phase A parts only)."""

    return {
        "experiment_name": "week3-standard-ppo",
        "schema_path": "configs/schema-building1.json",
        "central_agent": True,
        "random_seed": 42,
        "windows": {
            "dev": {
                "simulation_start_time_step": 0,
                "simulation_end_time_step": 167,
            },
            "final": {
                "simulation_start_time_step": 0,
                "simulation_end_time_step": 719,
            },
        },
        "reward": dict(REWARD_CONSTANTS),
    }


def write_config(features: Dict[str, Dict[str, float]], epsilon: float) -> None:
    """Write the normalisation block into ``configs/week3-ppo.yaml``."""

    if CONFIG_PATH.is_file():
        config = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    else:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        config = build_config_skeleton()

    config["normalisation"] = {
        "epsilon": epsilon,
        "stats_source": str(TRACE_PATH.relative_to(PROJECT_ROOT)),
        "rule": (
            "identity when static range is [0,1]; otherwise min-max from the "
            "B0 dev-window trace when a trace column exists; otherwise "
            "min-max from schema-static observation ranges"
        ),
        "features": features,
    }
    CONFIG_PATH.write_text(yaml.safe_dump(config, sort_keys=False))


def main() -> int:
    features = compute_feature_stats(DEFAULT_EPSILON)
    write_config(features, DEFAULT_EPSILON)

    print(f"Wrote {CONFIG_PATH.relative_to(PROJECT_ROOT)}")
    print(f"features: {len(features)} (epsilon={DEFAULT_EPSILON:g})")
    identity = [n for n, s in features.items() if s == {"offset": 0.0, "scale": 1.0}]
    print(f"identity features ({len(identity)}): {', '.join(identity)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
