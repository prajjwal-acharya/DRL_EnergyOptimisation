"""Single-loop runner shared by every controller.

Plan reference: docs/plans/week2-implementation-plan.md §B2. The runner loads the
CityLearn environment from a schema, supports dev/final window overrides via
``simulation_start_time_step``/``simulation_end_time_step``, steps to terminal
with one seeded reset, records a per-step trace, and returns district KPIs.

Measurement convention (docs/reference/cmdp-spec.md §1): under CityLearn 2.5.0 the
computed entries of a returned observation vector (``net_electricity_consumption``
and the storage SoCs) are written only when their step executes, so the vector
returned after stepping to ``t`` still holds uncomputed zeros for ``t``.
The runner therefore applies the documented workaround:

- each trace row records the *executed* step ``t`` values read from the
  authoritative building-level time series after ``env.step`` returns
  (``Building_1.net_electricity_consumption[t]`` etc.), which are the same
  series consumed by ``env.evaluate()``;
- the controller receives the observation vector with those computed slots
  repaired to the latest known executed values (step ``t - 1``), a strictly
  causal state estimate with no lookahead.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from energy_optimisation.environment import load_environment
from energy_optimisation.observation_names import build_observation_index


PROJECT_ROOT = Path(__file__).resolve().parents[3]

REQUIRED_OBSERVATIONS = (
    "hour",
    "electricity_pricing",
    "net_electricity_consumption",
    "dhw_storage_soc",
    "electrical_storage_soc",
    "indoor_dry_bulb_temperature",
    "indoor_dry_bulb_temperature_cooling_set_point",
    "solar_generation",
)

# Observation slots whose returned values are uncomputed zeros in CityLearn
# 2.5.0 post-step vectors (see module docstring); repaired from the building
# time series for both the controller input and the trace.
SERIES_REPAIRED_OBSERVATIONS = (
    "net_electricity_consumption",
    "dhw_storage_soc",
    "electrical_storage_soc",
    "indoor_dry_bulb_temperature",
)

TRACE_COLUMNS = (
    "timestep",
    "hour",
    "electricity_pricing",
    "action_requested_dhw_storage",
    "action_requested_electrical_storage",
    "action_requested_cooling_device",
    "action_applied_dhw_storage",
    "action_applied_electrical_storage",
    "action_applied_cooling_device",
    "net_electricity_consumption",
    "dhw_storage_soc",
    "electrical_storage_soc",
    "indoor_dry_bulb_temperature",
    "indoor_dry_bulb_temperature_cooling_set_point",
    "solar_generation",
)


def resolve_git_commit(project_root: Path = PROJECT_ROOT) -> Optional[str]:
    """Return ``git rev-parse HEAD`` for the repository, or None outside git."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def relative_to_project_root(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    """Express a path relative to the project root, never an absolute path."""

    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(project_root).resolve()))
    except ValueError:
        raise ValueError(
            f"schema path {resolved} is outside project root {Path(project_root).resolve()}"
        )


def district_kpis_as_dict(kpis_frame: pd.DataFrame) -> Dict[str, float]:
    """Reduce ``env.evaluate()`` output to district-level KPI name/value pairs.

    The two outage KPIs are empty in this dataset (plan §0 guardrail), so any
    district KPI whose value is missing or NaN is excluded from the mapping.
    """

    district = kpis_frame.loc[kpis_frame["level"] == "district"]
    kpis: Dict[str, float] = {}
    for cost_function, value in zip(district["cost_function"], district["value"]):
        if pd.isna(value):
            continue
        kpis[str(cost_function)] = float(value)
    return kpis


def executed_step_values(building, time_step: int) -> Dict[str, float]:
    """Read the authoritative executed-step values for ``time_step`` from a building.

    These are the building-level series entries written when the step at
    ``time_step`` executed — the same series ``env.evaluate()`` consumes
    (docs/reference/cmdp-spec.md §1). ``solar_generation`` follows the observation
    convention of reporting generation as a non-negative quantity.
    """

    return {
        "net_electricity_consumption": float(building.net_electricity_consumption[time_step]),
        "dhw_storage_soc": float(building.dhw_storage.soc[time_step]),
        "electrical_storage_soc": float(building.electrical_storage.soc[time_step]),
        "indoor_dry_bulb_temperature": float(
            building.energy_simulation.indoor_dry_bulb_temperature[time_step]
        ),
        "solar_generation": float(abs(building.solar_generation[time_step])),
    }


def repair_observation(
    observation_vector: np.ndarray,
    index: Mapping[str, int],
    latest_executed: Mapping[str, float],
) -> np.ndarray:
    """Return a copy of ``observation_vector`` with computed slots made valid.

    Computed slots (``SERIES_REPAIRED_OBSERVATIONS``) are replaced with the
    latest known executed values; every other slot is left untouched. The
    input vector is not mutated.
    """

    repaired = observation_vector.copy()
    for name in SERIES_REPAIRED_OBSERVATIONS:
        if name in latest_executed:
            repaired[index[name]] = latest_executed[name]
    return repaired


def run_episode(
    controller,
    schema_path: str | Path,
    *,
    simulation_start_time_step: int,
    simulation_end_time_step: int,
    seed: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Run one controller to terminal and return ``(kpis, trace)``.

    Parameters
    ----------
    controller:
        Object implementing :class:`energy_optimisation.baselines.controllers.Controller`.
    schema_path:
        CityLearn schema; window overrides are passed to the environment.
    simulation_start_time_step / simulation_end_time_step:
        Inclusive window bounds (dev 0–167, final 0–719 in the frozen config).
    seed:
        Seed forwarded to ``env.reset(seed=...)`` and ``controller.reset``.

    Returns
    -------
    Tuple[Dict[str, float], pd.DataFrame]
        District KPIs (empty KPIs excluded) and a per-step trace with columns
        ``TRACE_COLUMNS``, including requested (pre-clip) and applied
        (post-clip to the environment action space) actions.
    """

    if simulation_end_time_step <= simulation_start_time_step:
        raise ValueError("simulation_end_time_step must be greater than simulation_start_time_step")

    environment = load_environment(
        schema_path,
        central_agent=True,
        simulation_start_time_step=simulation_start_time_step,
        simulation_end_time_step=simulation_end_time_step,
    )

    index = build_observation_index(schema_path)
    missing = [name for name in REQUIRED_OBSERVATIONS if name not in index]
    if missing:
        raise KeyError(f"Schema lacks required observations: {', '.join(missing)}")

    action_space = environment.action_space[0]
    low = np.asarray(action_space.low, dtype=float)
    high = np.asarray(action_space.high, dtype=float)
    building = environment.buildings[0]

    observations, _ = environment.reset(seed=seed)
    controller.reset(seed=seed)

    rows = []
    latest_executed: Dict[str, float] = {}
    while not environment.terminated:
        time_step = int(environment.time_step)
        observation_vector = np.asarray(observations[0], dtype=float)
        controller_observation = repair_observation(
            observation_vector, index, latest_executed
        )
        requested = np.asarray(controller.act(controller_observation), dtype=float)
        applied = np.clip(requested, low, high)

        observations, _, _, _, _ = environment.step([applied.tolist()])

        # CityLearn 2.5.0 writes the computed series entries for `time_step`
        # only once this step call has executed; read the executed values now.
        executed = executed_step_values(building, time_step)
        latest_executed = executed

        rows.append(
            {
                "timestep": time_step,
                "hour": float(observation_vector[index["hour"]]),
                "electricity_pricing": float(observation_vector[index["electricity_pricing"]]),
                "action_requested_dhw_storage": float(requested[0]),
                "action_requested_electrical_storage": float(requested[1]),
                "action_requested_cooling_device": float(requested[2]),
                "action_applied_dhw_storage": float(applied[0]),
                "action_applied_electrical_storage": float(applied[1]),
                "action_applied_cooling_device": float(applied[2]),
                "net_electricity_consumption": executed["net_electricity_consumption"],
                "dhw_storage_soc": executed["dhw_storage_soc"],
                "electrical_storage_soc": executed["electrical_storage_soc"],
                "indoor_dry_bulb_temperature": executed["indoor_dry_bulb_temperature"],
                "indoor_dry_bulb_temperature_cooling_set_point": float(
                    observation_vector[index["indoor_dry_bulb_temperature_cooling_set_point"]]
                ),
                "solar_generation": executed["solar_generation"],
            }
        )

    if not environment.terminated:
        raise RuntimeError(f"{controller.name} run finished without the expected terminal state")

    trace = pd.DataFrame(rows, columns=TRACE_COLUMNS)
    kpis = district_kpis_as_dict(environment.evaluate())
    return kpis, trace


def build_run_metadata(
    controller_name: str,
    schema_path: str | Path,
    *,
    simulation_start_time_step: int,
    simulation_end_time_step: int,
    seed: int,
    completed_steps: int,
    purpose: str = "research result",
    config: Optional[Mapping[str, Any]] = None,
    project_root: Path = PROJECT_ROOT,
) -> Dict[str, Any]:
    """Assemble JSON-safe provenance metadata for one run.

    Paths are recorded relative to the project root (never absolute machine
    paths) together with the resolved git commit of the workspace.
    """

    metadata: Dict[str, Any] = {
        "project_root": ".",
        "schema": relative_to_project_root(Path(schema_path), project_root),
        "controller": controller_name,
        "seed": int(seed),
        "simulation_start_time_step": int(simulation_start_time_step),
        "simulation_end_time_step": int(simulation_end_time_step),
        "completed_steps": int(completed_steps),
        "purpose": purpose,
        "git_commit": resolve_git_commit(project_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata["config"] = dict(config) if config is not None else None
    return metadata


def run_and_record(
    controller,
    schema_path: str | Path,
    *,
    output_directory: str | Path,
    simulation_start_time_step: int,
    simulation_end_time_step: int,
    seed: int,
    purpose: str = "research result",
    config: Optional[Mapping[str, Any]] = None,
    derived_metric_thresholds: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Run an episode and persist the full artifact set under ``output_directory``.

    When ``derived_metric_thresholds`` is provided (comfort band, SoC reserve
    band, grid limit from the frozen config), the §B4 derived metrics are
    computed from the trace and persisted alongside as ``derived_metrics.json``.
    """

    from energy_optimisation.evaluation.artifacts import write_run_artifacts

    kpis, trace = run_episode(
        controller,
        schema_path,
        simulation_start_time_step=simulation_start_time_step,
        simulation_end_time_step=simulation_end_time_step,
        seed=seed,
    )
    metadata = build_run_metadata(
        controller.name,
        schema_path,
        simulation_start_time_step=simulation_start_time_step,
        simulation_end_time_step=simulation_end_time_step,
        seed=seed,
        completed_steps=len(trace),
        purpose=purpose,
        config=config,
    )
    derived_metrics = None
    if derived_metric_thresholds is not None:
        from energy_optimisation.evaluation.metrics import compute_derived_metrics

        derived_metrics = compute_derived_metrics(trace, **dict(derived_metric_thresholds))
    write_run_artifacts(
        output_directory,
        metadata=metadata,
        kpis=kpis,
        trace=trace,
        derived_metrics=derived_metrics,
    )
    return kpis, trace
