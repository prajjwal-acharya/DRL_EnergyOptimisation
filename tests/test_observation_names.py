from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energy_optimisation.environment import load_environment
from energy_optimisation.observation_names import (
    BUILDING_1_OBSERVATION_INDEX,
    BUILDING_1_SCHEMA_PATH,
    build_observation_index,
    observation_layout,
)


PARENT_SCHEMA_PATH = (
    PROJECT_ROOT
    / "data/raw/citylearn-2.5.0/data/datasets/citylearn_challenge_2023_phase_1/schema.json"
)
INSPECTION_PATH = PROJECT_ROOT / "results/inspection/citylearn_2023_phase_1.json"
EXPECTED_CENTRAL_DIMENSION = 49


def _inspection_building_observations() -> list[str]:
    inspection = json.loads(INSPECTION_PATH.read_text())
    return list(inspection["first_building"]["observations"])


def test_observation_layout_reproduces_inspection_positions() -> None:
    expected_names = _inspection_building_observations()
    layout = observation_layout(PARENT_SCHEMA_PATH)
    index = build_observation_index(PARENT_SCHEMA_PATH)

    assert len(layout) == EXPECTED_CENTRAL_DIMENSION
    assert layout[: len(expected_names)] == expected_names
    for position, name in enumerate(expected_names):
        assert index[name] == position, f"expected {name!r} at index {position}, found {index[name]}"


def test_parent_schema_layout_matches_loaded_observation_space() -> None:
    layout = observation_layout(PARENT_SCHEMA_PATH)
    environment = load_environment(PARENT_SCHEMA_PATH, central_agent=True)

    assert len(environment.observation_space) == 1
    assert environment.observation_space[0].shape[0] == len(layout) == EXPECTED_CENTRAL_DIMENSION


def test_frozen_building_1_index_matches_single_building_environment() -> None:
    expected_names = _inspection_building_observations()
    environment = load_environment(BUILDING_1_SCHEMA_PATH, central_agent=True)

    assert len(environment.observation_space) == 1
    assert environment.observation_space[0].shape[0] == len(BUILDING_1_OBSERVATION_INDEX)
    assert list(BUILDING_1_OBSERVATION_INDEX.values()) == list(range(len(expected_names)))
    for position, name in enumerate(expected_names):
        assert BUILDING_1_OBSERVATION_INDEX[name] == position


def test_frozen_index_is_immutable() -> None:
    try:
        BUILDING_1_OBSERVATION_INDEX["hour"] = -1  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("frozen observation index must reject mutation")


def test_key_control_observations_are_distinct_and_present() -> None:
    required = (
        "hour",
        "electricity_pricing",
        "electrical_storage_soc",
        "dhw_storage_soc",
        "indoor_dry_bulb_temperature",
        "indoor_dry_bulb_temperature_cooling_set_point",
        "net_electricity_consumption",
    )
    indices = [BUILDING_1_OBSERVATION_INDEX[name] for name in required]

    assert len(set(indices)) == len(required)
