"""Canonical observation-name to central-agent index mapping for CityLearn.

CityLearn addresses observations positionally, so every controller in this
project resolves observations by name through :data:`BUILDING_1_OBSERVATION_INDEX`
instead of using magic indices.  The ordering replicated here follows CityLearn
2.5.0 semantics (see ``CityLearnEnv.observation_space``): for a central agent,
building 0 contributes all of its active observations and each subsequent
building contributes only observations that are not shared or have not already
been contributed by an earlier building.  Because building-specific observation
names recur once per building, :func:`build_observation_index` maps every name
to its first (Building_1) slot; :func:`observation_layout` exposes the full
per-slot name list for schemas where that distinction matters.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILDING_1_SCHEMA_PATH = PROJECT_ROOT / "configs/schema-building1.json"


def _active_observation_names(schema: dict[str, object], building_name: str) -> list[str]:
    """Return a building's active observation names in schema declaration order."""

    building_schema = schema["buildings"][building_name]
    inactive = set(building_schema.get("inactive_observations") or [])
    return [
        name
        for name, metadata in schema["observations"].items()
        if metadata.get("active", False) and name not in inactive
    ]


def _shared_observation_names(schema: dict[str, object]) -> list[str]:
    """Return shared-in-central-agent observation names, mirroring CityLearn."""

    return [
        name
        for name, metadata in schema["observations"].items()
        if not name.startswith("electric_vehicle_")
        and "washing_machine" not in name
        and metadata.get("shared_in_central_agent", False)
    ]


def observation_layout(schema_path: str | Path) -> list[str]:
    """Return every central-agent observation slot name in CityLearn order.

    The resulting list contains one entry per observation slot (length equals
    the loaded environment's single observation space dimension); names of
    unshared observations recur once per building after the first.
    """

    path = Path(schema_path)
    if not path.is_file():
        raise FileNotFoundError(f"CityLearn schema not found: {path}")

    schema = json.loads(path.read_text())
    if not schema.get("central_agent", False):
        raise ValueError(f"Schema does not enable a central agent: {path}")

    shared = set(_shared_observation_names(schema))
    seen_shared: set[str] = set()
    layout: list[str] = []

    for building_index, building_name in enumerate(schema["buildings"]):
        for name in _active_observation_names(schema, building_name):
            if building_index == 0 or name not in shared or name not in seen_shared:
                layout.append(name)
            if name in shared and name not in seen_shared:
                seen_shared.add(name)

    return layout


def build_observation_index(schema_path: str | Path) -> OrderedDict[str, int]:
    """Map each observation name to its first index in the central-agent vector."""

    index: OrderedDict[str, int] = OrderedDict()
    for slot, name in enumerate(observation_layout(schema_path)):
        index.setdefault(name, slot)
    return index


def frozen_observation_index(schema_path: str | Path = BUILDING_1_SCHEMA_PATH) -> Mapping[str, int]:
    """Build an immutable observation index for the given schema."""

    return MappingProxyType(build_observation_index(schema_path))


BUILDING_1_OBSERVATION_INDEX: Mapping[str, int] = frozen_observation_index(BUILDING_1_SCHEMA_PATH)
