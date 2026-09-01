"""Inspection utilities for a locally stored CityLearn scenario."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from citylearn.citylearn import CityLearnEnv


def describe_space(space: Any) -> dict[str, Any]:
    """Return a JSON-safe summary of a Gymnasium-style space."""

    description: dict[str, Any] = {
        "type": type(space).__name__,
        "shape": list(space.shape) if getattr(space, "shape", None) is not None else None,
    }

    if hasattr(space, "low") and hasattr(space, "high"):
        low = np.asarray(space.low, dtype=float)
        high = np.asarray(space.high, dtype=float)
        description["low_min"] = float(low.min())
        description["low_max"] = float(low.max())
        description["high_min"] = float(high.min())
        description["high_max"] = float(high.max())

    return description


def load_environment(
    schema_path: str | Path,
    *,
    central_agent: bool = True,
    **environment_overrides: Any,
) -> CityLearnEnv:
    """Load a CityLearn environment from a local schema file."""

    path = Path(schema_path)
    if not path.is_file():
        raise FileNotFoundError(f"CityLearn schema not found: {path}")

    return CityLearnEnv(str(path), central_agent=central_agent, **environment_overrides)


def create_single_building_schema(
    parent_schema_path: str | Path,
    building_name: str,
    output_path: str | Path,
    *,
    project_root: str | Path,
) -> Path:
    """Derive a CityLearn schema for exactly one building without changing the parent."""

    parent_path = Path(parent_schema_path).resolve()
    destination = Path(output_path).resolve()
    root = Path(project_root).resolve()
    schema = json.loads(parent_path.read_text())

    if building_name not in schema["buildings"]:
        raise KeyError(f"Building {building_name!r} is not present in {parent_path}")

    schema["buildings"] = {building_name: schema["buildings"][building_name]}
    schema["central_agent"] = True
    schema["root_directory"] = str(parent_path.parent.relative_to(root))

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(schema, indent=2) + "\n")
    return destination


def neutral_actions(environment: CityLearnEnv) -> list[list[float]]:
    """Return deterministic zero actions clipped to each controller action space."""

    actions: list[list[float]] = []
    for space in environment.action_space:
        values = np.zeros(space.shape, dtype=float)
        actions.append(np.clip(values, space.low, space.high).tolist())

    return actions


def inspect_environment(schema_path: str | Path, *, central_agent: bool = True) -> dict[str, Any]:
    """Collect only environment-interface evidence; do not train or step an agent."""

    environment = load_environment(schema_path, central_agent=central_agent)
    observations, info = environment.reset()
    first_building = environment.buildings[0]
    kpis = environment.evaluate()

    return {
        "schema_path": str(Path(schema_path)),
        "central_agent": environment.central_agent,
        "building_count": len(environment.buildings),
        "building_names": [building.name for building in environment.buildings],
        "episode_time_steps": environment.episode_tracker.episode_time_steps,
        "observation_agent_count": len(observations),
        "observation_space": describe_space(environment.observation_space[0]),
        "action_space": describe_space(environment.action_space[0]),
        "first_building": {
            "name": first_building.name,
            "observations": first_building.active_observations,
            "actions": first_building.active_actions,
            "pv_nominal_power_kw": float(first_building.pv.nominal_power),
            "battery_capacity_kwh": float(first_building.electrical_storage.capacity),
        },
        "district_kpis": kpis.loc[kpis["level"] == "district", "cost_function"].tolist(),
        "reset_info_keys": sorted(info),
    }
