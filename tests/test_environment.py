from __future__ import annotations

import sys
from pathlib import Path

from gymnasium.spaces import Box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energy_optimisation.environment import (
    describe_space,
    load_environment,
    neutral_actions,
)

SCHEMA_PATH = PROJECT_ROOT / "configs/schema-building1.json"


def test_describe_space_returns_json_safe_bounds() -> None:
    summary = describe_space(Box(low=-1.0, high=2.0, shape=(3,)))

    assert summary == {
        "type": "Box",
        "shape": [3],
        "low_min": -1.0,
        "low_max": -1.0,
        "high_min": 2.0,
        "high_max": 2.0,
    }


def test_neutral_actions_respect_space_bounds() -> None:
    class Environment:
        action_space = [Box(low=-1.0, high=2.0, shape=(3,))]

    assert neutral_actions(Environment()) == [[0.0, 0.0, 0.0]]


def test_load_environment_is_cwd_independent(tmp_path, monkeypatch) -> None:
    """Regression: CityLearn resolves a relative root_directory against the CWD,
    which broke every run launched outside the repository root (e.g. from inside
    scripts/<phase>/). load_environment must absolutize it instead."""

    monkeypatch.chdir(tmp_path)
    environment = load_environment(SCHEMA_PATH)

    assert len(environment.buildings) == 1
    assert environment.root_directory.is_absolute()
    assert Path(environment.root_directory).is_dir()
