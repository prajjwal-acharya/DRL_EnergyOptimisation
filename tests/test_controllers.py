"""Phase C baseline-controller tests.

Plan reference: docs/plans/week2-implementation-plan.md §C and §C tests. The core
unit checks pin the contrast properties that later research questions rely on:
B1 must be price-blind (calendar-only) while B2 must respond to the current
price around the frozen threshold τ and respect the SoC reserve band.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energy_optimisation.baselines.controllers import (
    ACTION_DIMENSION,
    PEAK_COOLING_DEVICE_LEVEL,
    RESERVE_HIGH_SOC,
    RESERVE_LOW_SOC,
    TARIFF_THRESHOLD_USD_PER_KWH,
    FixedScheduleController,
    NeutralController,
    TariffAwareController,
)
from energy_optimisation.observation_names import (
    BUILDING_1_OBSERVATION_INDEX,
    build_observation_index,
)


SCHEMA_PATH = PROJECT_ROOT / "configs/schema-building1.json"
# The derived single-building schema exposes 29 active observations (one slot
# each); synthetic observations are sized from the Phase A index, never
# hard-coded.
OBSERVATION_DIMENSION = len(build_observation_index(SCHEMA_PATH))
OFF_PEAK_PRICE = 0.02893
MID_PRICE = 0.02915
PEAK_PRICE = 0.05867
SEED = 42
DEV_WINDOW = {"simulation_start_time_step": 0, "simulation_end_time_step": 167}
EXPECTED_DEV_STEPS = 167
FORBIDDEN_SOURCE_TOKENS = ("forecasting", "stable_baselines3", "torch")


def make_observation(**values: float) -> np.ndarray:
    """Build a central-agent-sized observation with named entries set."""

    vector = np.zeros(OBSERVATION_DIMENSION, dtype=float)
    for name, value in values.items():
        vector[BUILDING_1_OBSERVATION_INDEX[name]] = value
    return vector


ALL_CONTROLLERS = (
    NeutralController(),
    FixedScheduleController(),
    TariffAwareController(),
)


@pytest.mark.parametrize("controller", ALL_CONTROLLERS, ids=lambda c: c.name)
def test_action_shape_dtype_and_finiteness_under_random_observations(controller) -> None:
    for seed in range(10):
        generator = np.random.default_rng(seed)
        for _ in range(25):
            random_observation = generator.uniform(-5.0, 5.0, OBSERVATION_DIMENSION)

            action = controller.act(random_observation)

            assert action.shape == (ACTION_DIMENSION,)
            assert np.issubdtype(action.dtype, np.floating)
            assert np.all(np.isfinite(action))


def test_frozen_observation_index_covers_central_vector() -> None:
    index = build_observation_index(SCHEMA_PATH)

    assert len(index) == OBSERVATION_DIMENSION
    for name in ("hour", "electricity_pricing", "electrical_storage_soc", "dhw_storage_soc"):
        assert name in index


def test_b0_is_always_zero() -> None:
    controller = NeutralController()
    generator = np.random.default_rng(SEED)

    for _ in range(50):
        action = controller.act(generator.uniform(-5.0, 5.0, OBSERVATION_DIMENSION))
        assert np.array_equal(action, np.zeros(ACTION_DIMENSION))


def test_b1_follows_frozen_calendar_schedule() -> None:
    controller = FixedScheduleController()

    expected_by_hour = {
        3: [-0.5, -0.5, 0.2],  # charge window, night cooling floor
        6: [0.0, 0.0, 0.2],  # after charge window, before cooling ramp
        10: [0.0, 0.0, 0.5],  # morning cooling band
        14: [0.0, 0.0, 0.8],  # midday high cooling band
        18: [0.0, 0.5, 0.8],  # evening discharge inside high cooling band
        21: [0.0, 0.0, 0.5],  # late-evening cooling band
        23: [0.0, 0.0, 0.2],  # night cooling floor
    }
    for hour, expected in expected_by_hour.items():
        action = controller.act(make_observation(hour=float(hour), electricity_pricing=PEAK_PRICE))
        assert np.allclose(action, expected), f"hour {hour}: {action} != {expected}"


def test_b1_ignores_price_signal() -> None:
    controller = FixedScheduleController()
    prices = (OFF_PEAK_PRICE, MID_PRICE, PEAK_PRICE, TARIFF_THRESHOLD_USD_PER_KWH * 0.5, 1.0)

    for hour in range(1, 25):
        reference = controller.act(
            make_observation(hour=float(hour), electricity_pricing=prices[0])
        )
        for price in prices[1:]:
            action = controller.act(make_observation(hour=float(hour), electricity_pricing=price))
            assert np.array_equal(action, reference), (
                f"B1 changed action at hour {hour} when price moved to {price}"
            )


def test_b2_discharges_above_threshold() -> None:
    controller = TariffAwareController()

    action = controller.act(
        make_observation(
            hour=17.0,
            electricity_pricing=PEAK_PRICE,
            electrical_storage_soc=0.5,
            dhw_storage_soc=0.5,
        )
    )

    assert np.allclose(action, [0.5, 0.5, PEAK_COOLING_DEVICE_LEVEL])


def test_b2_charges_below_threshold() -> None:
    controller = TariffAwareController()

    action = controller.act(
        make_observation(
            hour=14.0,
            electricity_pricing=MID_PRICE,
            electrical_storage_soc=0.5,
            dhw_storage_soc=0.5,
        )
    )

    assert np.allclose(action, [-0.5, -0.5, 0.8])  # cooling follows the B1 hour bands


def test_b2_peak_branch_is_inclusive_at_threshold() -> None:
    controller = TariffAwareController()
    mid_soc_observation = dict(electrical_storage_soc=0.5, dhw_storage_soc=0.5)

    at_threshold = controller.act(
        make_observation(hour=12.0, electricity_pricing=TARIFF_THRESHOLD_USD_PER_KWH, **mid_soc_observation)
    )
    just_below = controller.act(
        make_observation(
            hour=12.0, electricity_pricing=TARIFF_THRESHOLD_USD_PER_KWH * 0.999, **mid_soc_observation
        )
    )

    assert np.allclose(at_threshold, [0.5, 0.5, PEAK_COOLING_DEVICE_LEVEL])
    assert np.allclose(just_below, [-0.5, -0.5, 0.8])


def test_b2_respects_soc_reserve_band() -> None:
    controller = TariffAwareController()

    depleted_peak = controller.act(
        make_observation(
            hour=17.0,
            electricity_pricing=PEAK_PRICE,
            electrical_storage_soc=RESERVE_LOW_SOC - 0.05,
            dhw_storage_soc=RESERVE_LOW_SOC - 0.05,
        )
    )
    full_off_peak = controller.act(
        make_observation(
            hour=14.0,
            electricity_pricing=OFF_PEAK_PRICE,
            electrical_storage_soc=RESERVE_HIGH_SOC + 0.05,
            dhw_storage_soc=RESERVE_HIGH_SOC + 0.05,
        )
    )

    assert np.allclose(depleted_peak[:2], [0.0, 0.0]), "no discharge when SoC < reserve low"
    assert depleted_peak[2] == PEAK_COOLING_DEVICE_LEVEL, "comfort protection unaffected"
    assert np.allclose(full_off_peak[:2], [0.0, 0.0]), "no charge when SoC > reserve high"


def test_b2_reserve_band_edges_are_inclusive() -> None:
    controller = TariffAwareController()

    edge_low_peak = controller.act(
        make_observation(
            hour=17.0,
            electricity_pricing=PEAK_PRICE,
            electrical_storage_soc=RESERVE_LOW_SOC,
            dhw_storage_soc=RESERVE_LOW_SOC,
        )
    )
    edge_high_off_peak = controller.act(
        make_observation(
            hour=14.0,
            electricity_pricing=OFF_PEAK_PRICE,
            electrical_storage_soc=RESERVE_HIGH_SOC,
            dhw_storage_soc=RESERVE_HIGH_SOC,
        )
    )

    assert np.allclose(edge_low_peak[:2], [0.5, 0.5])
    assert np.allclose(edge_high_off_peak[:2], [-0.5, -0.5])


def test_b2_off_peak_cooling_matches_b1_hour_bands() -> None:
    b1 = FixedScheduleController()
    b2 = TariffAwareController()

    for hour in range(1, 25):
        observation = make_observation(hour=float(hour), electricity_pricing=OFF_PEAK_PRICE)
        assert b2.act(observation)[2] == b1.act(observation)[2], f"hour {hour}"


@pytest.mark.parametrize("controller", ALL_CONTROLLERS, ids=lambda c: c.name)
def test_controllers_deterministic_under_seed(controller) -> None:
    def rollout(seed: int):
        controller.reset(seed=seed)
        generator = np.random.default_rng(seed)
        return [
            controller.act(obs).copy()
            for obs in (generator.uniform(0.0, 40.0, OBSERVATION_DIMENSION) for _ in range(20))
        ]

    for seed in (42, 7):
        assert all(np.array_equal(a, b) for a, b in zip(rollout(seed), rollout(seed)))


def test_baselines_sources_contain_no_forbidden_imports() -> None:
    baselines_directory = PROJECT_ROOT / "src" / "energy_optimisation" / "baselines"
    sources = sorted(baselines_directory.glob("*.py"))
    assert sources, "baselines package sources not found"

    for source in sources:
        text = source.read_text().lower()
        for token in FORBIDDEN_SOURCE_TOKENS:
            assert token not in text, f"{source.name} mentions forbidden dependency '{token}'"


@pytest.fixture(scope="module")
def dev_window_runs():
    from energy_optimisation.evaluation.runner import run_episode

    return {
        controller.name: run_episode(controller, SCHEMA_PATH, seed=SEED, **DEV_WINDOW)
        for controller in ALL_CONTROLLERS
    }


@pytest.mark.parametrize(
    "controller_name", [controller.name for controller in ALL_CONTROLLERS]
)
def test_dev_window_run_completes_without_clipping(dev_window_runs, controller_name) -> None:
    kpis, trace = dev_window_runs[controller_name]

    assert len(trace) == EXPECTED_DEV_STEPS
    assert not trace.isna().any().any()
    assert kpis, "district KPIs must be non-empty"

    for storage in ("dhw_storage", "electrical_storage"):
        requested = trace[f"action_requested_{storage}"]
        applied = trace[f"action_applied_{storage}"]
        assert ((requested >= -1.0) & (requested <= 1.0)).all(), (
            f"{controller_name} requested out-of-bounds {storage} action"
        )
        assert np.allclose(requested, applied), f"{controller_name} actions were clipped"

    requested_cooling = trace["action_requested_cooling_device"]
    applied_cooling = trace["action_applied_cooling_device"]
    assert ((requested_cooling >= 0.0) & (requested_cooling <= 1.0)).all()
    assert np.allclose(requested_cooling, applied_cooling)
