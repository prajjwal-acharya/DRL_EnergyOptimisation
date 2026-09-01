"""Phase C unit tests: PPOController wrapper, episode return, selection rule.

Plan reference: docs/plans/week3-implementation-plan.md §C1–§C3. These tests use a
tiny randomly-initialised PPO policy (49-dim observation stub) so they do not
depend on training artifacts; end-to-end checkpoint evaluation runs through
``scripts/22_evaluate_checkpoints.py`` on the locked harness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import yaml
from gymnasium import spaces


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stable_baselines3 import PPO

from energy_optimisation.rl import (
    EVALUATION_COLUMNS,
    PPOController,
    episode_return_from_trace,
    reward_constants_from_config,
    select_best_checkpoint,
)
from energy_optimisation.rl.env_adapter import (
    compute_cmdp_reward,
    map_rl_action_to_citylearn,
)
from energy_optimisation.observation_names import BUILDING_1_OBSERVATION_INDEX


CONFIG_PATH = PROJECT_ROOT / "configs/week3-ppo.yaml"
OBSERVATION_DIMENSION = len(BUILDING_1_OBSERVATION_INDEX)


class _StubEnv(gym.Env):
    """Minimal 49-dim-observation env so PPO can build an MlpPolicy."""

    metadata = {"render_modes": []}

    def __init__(self, dimension: int = OBSERVATION_DIMENSION) -> None:
        self.observation_space = spaces.Box(
            low=np.float32(0.0), high=np.float32(1.0), shape=(dimension,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.float32(-1.0), high=np.float32(1.0), shape=(3,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, True, {}


@pytest.fixture(scope="module")
def rl_config() -> dict:
    with CONFIG_PATH.open() as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def tiny_model_path(tmp_path_factory) -> Path:
    environment = _StubEnv()
    model = PPO(
        "MlpPolicy",
        environment,
        seed=0,
        device="cpu",
        n_steps=64,
        batch_size=64,
        policy_kwargs={"net_arch": [16]},
    )
    path = tmp_path_factory.mktemp("model") / "tiny_ppo.zip"
    model.save(path)
    return path


@pytest.fixture(scope="module")
def controller(tiny_model_path, rl_config) -> PPOController:
    return PPOController(tiny_model_path, rl_config, device="cpu", name="ppo_seed42")


def test_evaluation_columns_match_plan() -> None:
    assert EVALUATION_COLUMNS == (
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


def test_act_is_deterministic_and_mapped(controller) -> None:
    raw_observation = np.linspace(0.0, 1.0, OBSERVATION_DIMENSION)

    first = controller.act(raw_observation)
    second = controller.act(raw_observation)

    assert first.shape == (3,)
    assert np.array_equal(first, second), "deterministic policy must not vary between calls"
    assert np.all(np.isfinite(first))

    normalised = controller.normalise_observation(raw_observation)
    policy_action, _ = controller.model.predict(normalised, deterministic=True)
    expected = map_rl_action_to_citylearn(np.asarray(policy_action, dtype=float))
    assert np.allclose(first, expected), "act must return the mapped policy action verbatim"


def test_act_applies_frozen_action_mapping(controller) -> None:
    raw_observation = np.zeros(OBSERVATION_DIMENSION, dtype=float)
    action = controller.act(raw_observation)
    normalised = controller.normalise_observation(raw_observation)
    policy_action = np.asarray(
        controller.model.predict(normalised, deterministic=True)[0], dtype=float
    )
    # Storages pass through; cooling is the affine map onto [0, 1].
    assert action[0] == pytest.approx(policy_action[0])
    assert action[1] == pytest.approx(policy_action[1])
    assert action[2] == pytest.approx((policy_action[2] + 1.0) / 2.0)


def test_normalisation_matches_frozen_transform(controller, rl_config) -> None:
    features = rl_config["normalisation"]["features"]

    assert features["electrical_storage_soc"] == {"offset": 0.0, "scale": 1.0}

    slot_hour = BUILDING_1_OBSERVATION_INDEX["hour"]
    slot_soc = BUILDING_1_OBSERVATION_INDEX["electrical_storage_soc"]
    slot_net = BUILDING_1_OBSERVATION_INDEX["net_electricity_consumption"]
    hour_stats = features["hour"]
    net_stats = features["net_electricity_consumption"]

    raw_observation = np.zeros(OBSERVATION_DIMENSION, dtype=float)
    raw_observation[slot_hour] = 13.0  # CityLearn serves hours 1–24
    raw_observation[slot_soc] = 0.75
    raw_observation[slot_net] = float(net_stats["offset"]) + 0.5 * float(net_stats["scale"])

    normalised = controller.normalise_observation(raw_observation)

    assert normalised.dtype == np.float32
    assert np.all(normalised >= 0.0) and np.all(normalised <= 1.0)
    assert float(normalised[slot_hour]) == pytest.approx((13.0 - hour_stats["offset"]) / hour_stats["scale"])
    assert float(normalised[slot_soc]) == pytest.approx(0.75), "identity transform on SoCs"
    assert float(normalised[slot_net]) == pytest.approx(0.5)


def test_normalisation_saturates_outside_frozen_ranges(controller, rl_config) -> None:
    net_slot = BUILDING_1_OBSERVATION_INDEX["net_electricity_consumption"]
    stats = rl_config["normalisation"]["features"]["net_electricity_consumption"]
    raw = np.zeros(OBSERVATION_DIMENSION, dtype=float)
    raw[net_slot] = float(stats["offset"]) + 7.0 * float(stats["scale"])
    normalised = controller.normalise_observation(raw)
    assert float(normalised[net_slot]) == 1.0


def test_observation_dimension_mismatch_raises(controller) -> None:
    with pytest.raises(ValueError):
        controller.act(np.zeros(OBSERVATION_DIMENSION - 1, dtype=float))


def test_reset_is_stateless(controller) -> None:
    assert controller.reset(seed=42) is None
    assert controller.name == "ppo_seed42"


def test_reward_constants_from_config(rl_config) -> None:
    constants = reward_constants_from_config(rl_config)
    assert constants == {
        "w_E": 1.0,
        "w_P": 1.0,
        "w_C": 10.0,
        "E_bar_b0": 0.477229108554339,
        "P_ref": 7.694016456604004,
        "comfort_band_c": 2.0,
    }


def test_episode_return_from_trace_hand_computed(rl_config) -> None:
    constants = reward_constants_from_config(rl_config)
    e_bar = constants["E_bar_b0"]
    p_ref = constants["P_ref"]

    trace = pd.DataFrame(
        {
            "net_electricity_consumption": [e_bar, 2.0 * p_ref, 0.0],
            "indoor_dry_bulb_temperature": [24.0, 30.0, 21.5],
            "indoor_dry_bulb_temperature_cooling_set_point": [26.0, 23.0, 23.0],
        }
    )

    expected = (
        compute_cmdp_reward(e_bar, 24.0, 26.0, **constants)
        + compute_cmdp_reward(2.0 * p_ref, 30.0, 23.0, **constants)
        + compute_cmdp_reward(0.0, 21.5, 23.0, **constants)
    )
    hand_computed = (
        -1.0  # exactly the reference consumption, comfort satisfied
        + (-(2.0 * p_ref / e_bar) - 1.0 - 10.0 * (30.0 - 25.0))  # peak excess + hot discomfort
        + 0.0  # zero consumption below both thresholds
    )
    assert hand_computed == pytest.approx(expected)

    result = episode_return_from_trace(trace, **constants)
    assert result == pytest.approx(expected)
    assert result < 0.0


def test_select_best_checkpoint_lowest_cost_then_tie_break() -> None:
    frame = pd.DataFrame(
        {
            "checkpoint": ["a.zip", "b.zip", "c.zip"],
            "cost_total": [0.5, 0.4, 0.4],
            "discomfort_proportion": [0.1, 0.9, 0.05],
        }
    )
    best = select_best_checkpoint(frame)
    assert best["checkpoint"] == "c.zip", "tie on cost resolves to lower discomfort"

    no_tie = pd.DataFrame(
        {
            "checkpoint": ["a.zip", "b.zip"],
            "cost_total": [0.5, 0.4],
            "discomfort_proportion": [0.01, 0.99],
        }
    )
    assert select_best_checkpoint(no_tie)["checkpoint"] == "b.zip"


def test_select_best_checkpoint_requires_rule_columns() -> None:
    with pytest.raises(KeyError):
        select_best_checkpoint(pd.DataFrame({"checkpoint": ["a.zip"], "cost_total": [0.1]}))


def test_select_best_checkpoint_deterministic_on_full_tie() -> None:
    frame = pd.DataFrame(
        {
            "checkpoint": ["first.zip", "second.zip"],
            "timestep": [10000, 20000],
            "cost_total": [0.3, 0.3],
            "discomfort_proportion": [0.2, 0.2],
        }
    )
    assert select_best_checkpoint(frame)["checkpoint"] == "first.zip"
