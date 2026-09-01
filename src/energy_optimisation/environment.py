"""Inspection utilities for a locally stored CityLearn scenario."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from citylearn.citylearn import CityLearnEnv

# <project root>/src/energy_optimisation/environment.py -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def _resolve_root_directory(schema: Mapping[str, Any], schema_path: Path) -> Path:
    """Absolutize the schema's dataset root so env loading is CWD-independent.

    CityLearn resolves a relative ``root_directory`` against the process working
    directory, so any run launched from outside the repository root fails with a
    ``FileNotFoundError`` for the dataset CSVs. Resolve the relative path against
    the CWD first (historic behaviour), then the project root, then the schema's
    own directory, and hand CityLearn an absolute path.
    """

    raw_value = schema.get("root_directory")
    if raw_value is None:
        # CityLearn's convention for datasets bundled beside their schema (the
        # pinned parent schema uses root_directory: null): the dataset root is
        # the schema's own directory.
        return schema_path.parent.resolve()
    raw = Path(raw_value)
    if raw.is_absolute():
        return raw

    candidates = [
        (Path.cwd() / raw).resolve(),
        (PROJECT_ROOT / raw).resolve(),
        (schema_path.parent / raw).resolve(),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Schema root_directory {str(raw)!r} does not exist; tried: {tried}"
    )


def load_environment(
    schema_path: str | Path,
    *,
    central_agent: bool = True,
    **environment_overrides: Any,
) -> CityLearnEnv:
    """Load a CityLearn environment from a local schema file.

    Works from any working directory: the schema's relative ``root_directory``
    is absolutized (see ``_resolve_root_directory``) before the environment is
    constructed.
    """

    path = Path(schema_path)
    if not path.is_file():
        raise FileNotFoundError(f"CityLearn schema not found: {path}")

    schema = json.loads(path.read_text())
    root_directory = _resolve_root_directory(schema, path)
    return CityLearnEnv(
        str(path),
        root_directory=root_directory,
        central_agent=central_agent,
        **environment_overrides,
    )


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
