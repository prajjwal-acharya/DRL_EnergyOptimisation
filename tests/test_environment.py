from __future__ import annotations

import sys
from pathlib import Path

from gymnasium.spaces import Box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energy_optimisation.environment import describe_space, neutral_actions


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
