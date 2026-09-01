"""Phase A adapter tests, including the neutral-action B0 anchor regression.

Plan reference: docs/plans/week3-implementation-plan.md §A2. The regression drives
the constant RL action ``[-1, -1, -1]`` through ``CityLearnRLEnv`` on the dev
window (0–167, seed 42) and requires exact reproduction of the week-2 B0
smoke anchors within 1e-9 — the adapter-validation equivalent of week 2's
harness check (plan §0 anchors).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energy_optimisation.evaluation.runner import district_kpis_as_dict
from energy_optimisation.rl import (
    NEUTRAL_RL_ACTION,
    CityLearnRLEnv,
    compute_cmdp_reward,
    count_pre_clip_violations,
    map_rl_action_to_citylearn,
)


SCHEMA_PATH = str(PROJECT_ROOT / "configs/schema-building1.json")
CONFIG_PATH = PROJECT_ROOT / "configs/week3-ppo.yaml"
SMOKE_ANCHORS_PATH = PROJECT_ROOT / "results/runs/smoke/district_kpis.csv"
SEED = 42
SIMULATION_START_TIME_STEP = 0
SIMULATION_END_TIME_STEP = 167
EXPECTED_STEPS = SIMULATION_END_TIME_STEP - SIMULATION_START_TIME_STEP
TOLERANCE = 1e-9

EXCLUDED_EMPTY_KPIS = (
    "one_minus_thermal_resilience_proportion",
    "power_outage_normalized_unserved_energy_total",
)


@pytest.fixture(scope="module")
def rl_config() -> dict:
    with CONFIG_PATH.open() as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def smoke_anchors() -> dict:
    frame = pd.read_csv(SMOKE_ANCHORS_PATH, keep_default_na=False, float_precision="round_trip")
    district = frame.loc[frame["level"] == "district"]
    return {
        str(cost_function): float(value)
        for cost_function, value in zip(district["cost_function"], district["value"])
        if str(value).strip() != ""
    }


def make_env(rl_config: dict, **overrides) -> CityLearnRLEnv:
    return CityLearnRLEnv(
        SCHEMA_PATH,
        config=rl_config,
        simulation_start_time_step=overrides.pop(
            "simulation_start_time_step", SIMULATION_START_TIME_STEP
        ),
        simulation_end_time_step=overrides.pop(
            "simulation_end_time_step", SIMULATION_END_TIME_STEP
        ),
        **overrides,
    )


def test_observation_space_shape_and_finiteness(rl_config) -> None:
    env = make_env(rl_config)
    env.action_space.seed(SEED)
    observation, _ = env.reset(seed=SEED)

    assert env.observation_space.shape == (env.observation_dim,)
    assert env.action_space.shape == (3,)
    assert np.all(env.action_space.low == -1.0) and np.all(env.action_space.high == 1.0)

    assert observation.shape == (env.observation_dim,)
    assert np.all(np.isfinite(observation)), "reset observation must be finite"

    for step in range(EXPECTED_STEPS):
        observation, reward, terminated, truncated, _ = env.step(
            env.action_space.sample()
        )
        assert np.all(np.isfinite(observation)), f"step {step} observation must be finite"
        assert np.isfinite(reward)
        assert not terminated
        if step < EXPECTED_STEPS - 1:
            assert not truncated, "truncated may be True only at the terminal step"
        else:
            assert truncated

    # Normalised observations live inside the declared [0, 1] Box.
    assert np.all(observation >= 0.0) and np.all(observation <= 1.0)


def test_reset_deterministic_per_seed(rl_config) -> None:
    first_env = make_env(rl_config)
    second_env = make_env(rl_config)

    first, _ = first_env.reset(seed=SEED)
    again_on_same_instance, _ = first_env.reset(seed=SEED)
    second, _ = second_env.reset(seed=SEED)

    assert np.array_equal(first, again_on_same_instance)
    assert np.array_equal(first, second)


def test_episode_length_is_window_length(rl_config) -> None:
    env = make_env(rl_config)
    env.reset(seed=SEED)

    steps = 0
    truncated_flags = []
    while True:
        _, _, terminated, truncated, _ = env.step([-1.0, -1.0, -1.0])
        steps += 1
        truncated_flags.append(truncated)
        if truncated:
            break
        assert not terminated

    assert steps == EXPECTED_STEPS
    assert truncated_flags[-1] is True
    assert not any(truncated_flags[:-1]), "truncated may be True only at the terminal step"


def test_neutral_action_reproduces_b0_anchors(rl_config, smoke_anchors) -> None:
    env = make_env(rl_config)
    env.reset(seed=SEED)

    for index in range(EXPECTED_STEPS):
        _, _, terminated, truncated, info = env.step(NEUTRAL_RL_ACTION.tolist())
        assert not terminated
        assert info["requested_action"] == [-1.0, -1.0, -1.0]
        if truncated:
            assert index == EXPECTED_STEPS - 1
            break

    assert env.pre_clip_violation_count == 0, "the neutral action lies inside the box"

    kpis = district_kpis_as_dict(env.citylearn_environment.evaluate())

    assert set(smoke_anchors) <= set(kpis), "B0 must produce every non-empty smoke KPI"
    worst_gap = 0.0
    worst_name = ""
    for cost_function, expected in smoke_anchors.items():
        gap = abs(kpis[cost_function] - expected)
        if gap > worst_gap:
            worst_gap, worst_name = gap, cost_function
        assert gap <= TOLERANCE, (
            f"{cost_function}: expected {expected!r}, got {kpis[cost_function]!r} "
            f"(|delta|={gap} > {TOLERANCE})"
        )

    for excluded in EXCLUDED_EMPTY_KPIS:
        assert excluded not in kpis, f"{excluded} is empty in this dataset and must be excluded"


def test_action_mapping_bounds(rl_config) -> None:
    cooling_low = map_rl_action_to_citylearn(np.array([0.0, 0.0, -1.0]))
    cooling_high = map_rl_action_to_citylearn(np.array([0.0, 0.0, 1.0]))
    centre = map_rl_action_to_citylearn(np.array([0.25, -0.5, 0.0]))

    assert cooling_low[2] == 0.0, "a2 = -1 must map to cooling_device 0.0"
    assert cooling_high[2] == 1.0, "a2 = +1 must map to cooling_device 1.0"
    assert centre[0] == 0.25 and centre[1] == -0.5 and centre[2] == 0.5

    # End-to-end: the applied action recorded in info follows the mapping.
    env = make_env(rl_config)
    env.reset(seed=SEED)
    _, _, _, _, info = env.step([-0.4, 0.6, -1.0])
    assert info["applied_action"] == [-0.4, 0.6, 0.0]
    _, _, _, _, info = env.step([-0.4, 0.6, 1.0])
    assert info["applied_action"] == [-0.4, 0.6, 1.0]


def test_reward_matches_frozen_formula(rl_config) -> None:
    w_E = float(rl_config["reward"]["w_E"])
    w_P = float(rl_config["reward"]["w_P"])
    w_C = float(rl_config["reward"]["w_C"])
    E_bar = float(rl_config["reward"]["E_bar_b0"])
    P_ref = float(rl_config["reward"]["P_ref"])
    delta_t = float(rl_config["reward"]["comfort_band_c"])

    # Triple 1: exactly the reference consumption, comfort satisfied.
    # r = -(E/E_bar) = -1.0 exactly.
    reward = compute_cmdp_reward(
        E_bar, 24.0, 26.0, w_E=w_E, w_P=w_P, w_C=w_C, E_bar_b0=E_bar,
        P_ref=P_ref, comfort_band_c=delta_t,
    )
    assert reward == pytest.approx(-1.0, abs=1e-15)

    # Triple 2: peak excursion above P_ref plus hot discomfort.
    # r = -(E/E_bar) - (max(0, E - P_ref)/P_ref) - w_C * max(0, T - (T_set + dT))
    #   = -(2*P_ref/E_bar) - 1.0 - 10 * max(0, 30 - (23 + 2))
    expected = -(2.0 * P_ref / E_bar) - 1.0 - w_C * max(0.0, 30.0 - (23.0 + delta_t))
    reward = compute_cmdp_reward(
        2.0 * P_ref, 30.0, 23.0, w_E=w_E, w_P=w_P, w_C=w_C, E_bar_b0=E_bar,
        P_ref=P_ref, comfort_band_c=delta_t,
    )
    assert reward == pytest.approx(expected, abs=1e-12)

    # Triple 3: zero consumption below both thresholds.
    # r = -0 - 0 - 10 * max(0, T_in - (T_set + dT)) with T_in below band.
    expected = -0.0 - 0.0 - w_C * max(0.0, 21.5 - (23.0 + delta_t))
    reward = compute_cmdp_reward(
        0.0, 21.5, 23.0, w_E=w_E, w_P=w_P, w_C=w_C, E_bar_b0=E_bar,
        P_ref=P_ref, comfort_band_c=delta_t,
    )
    assert reward == pytest.approx(expected, abs=1e-15)
    assert reward == 0.0

    # Triple 4 (extra): negative consumption (PV export) stays unclipped.
    expected = -(-1.0 / E_bar)  # energy term only; no peak excess, no discomfort
    reward = compute_cmdp_reward(
        -1.0, 22.0, 23.0, w_E=w_E, w_P=w_P, w_C=w_C, E_bar_b0=E_bar,
        P_ref=P_ref, comfort_band_c=delta_t,
    )
    assert reward == pytest.approx(expected, abs=1e-12)

    with pytest.raises(ZeroDivisionError):
        compute_cmdp_reward(
            1.0, 22.0, 23.0, w_E=w_E, w_P=w_P, w_C=w_C, E_bar_b0=0.0,
            P_ref=P_ref, comfort_band_c=delta_t,
        )


def test_pre_clip_violations_are_counted(rl_config) -> None:
    assert count_pre_clip_violations(np.array([-1.0, 0.0, 1.0])) == 0
    assert count_pre_clip_violations(np.array([1.5, -2.0, 0.0])) == 2
    assert count_pre_clip_violations(np.array([3.0, 3.0, 3.0])) == 3

    env = make_env(rl_config)
    env.reset(seed=SEED)
    _, _, _, _, info = env.step([2.0, 0.0, 0.5])
    assert env.pre_clip_violation_count == 1
    assert info["pre_clip_violation_count"] == 1
    assert info["requested_action"] == [2.0, 0.0, 0.5]
    assert info["applied_action"][0] == 1.0, "out-of-box storage request is clipped"


def test_window_override_and_invalid_window(rl_config) -> None:
    env = make_env(
        rl_config,
        simulation_start_time_step=0,
        simulation_end_time_step=24,
    )
    env.reset(seed=SEED)
    steps = 0
    while True:
        _, _, terminated, truncated, _ = env.step([-1.0, -1.0, -1.0])
        steps += 1
        if truncated:
            break
    assert steps == 24
    assert env.window_bounds == (0, 24)

    with pytest.raises(ValueError):
        make_env(
            rl_config,
            simulation_start_time_step=10,
            simulation_end_time_step=10,
        )


def test_normalised_observation_matches_frozen_transform(rl_config) -> None:
    """Spot-check the frozen transform against hand-computed normalisations."""

    env = make_env(rl_config)
    observation, _ = env.reset(seed=SEED)
    layout = env.layout
    features = rl_config["normalisation"]["features"]

    # Dataset values at t=0 as recorded in the week-2 B0 dev trace (row 0);
    # the reset observation carries the same slots untransformed.
    known_raw_values = {
        "hour": 1.0,
        "electricity_pricing": 0.028929999098181725,
    }
    for name, raw_value in known_raw_values.items():
        stats = features[name]
        expected = np.clip(
            (raw_value - stats["offset"]) / stats["scale"], 0.0, 1.0
        )
        slot = layout.index(name)
        assert float(observation[slot]) == pytest.approx(float(expected), abs=1e-6)

    # SoC slots use the frozen identity transform: any [0, 1] SoC maps to itself.
    soc_stats = features["electrical_storage_soc"]
    assert soc_stats == {"offset": 0.0, "scale": 1.0}
    soc_slot = layout.index("electrical_storage_soc")
    assert 0.0 <= float(observation[soc_slot]) <= 1.0
