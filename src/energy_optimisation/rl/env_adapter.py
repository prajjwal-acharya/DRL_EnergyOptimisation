"""Gymnasium adapter around the pinned CityLearn 2.5.0 single-building env.

Plan reference: docs/plans/week3-implementation-plan.md §A1. The adapter exposes the
plain central-agent CityLearn observation (repaired causally exactly as the
week-2 harness does), a symmetric ``Box([-1, -1, -1], [1, 1, 1])`` action
space, and the frozen CMDP reward of ``docs/reference/cmdp-spec.md`` §4. No forecast-
derived features, no uncertainty representation, no safety shield: the policy
sees only normalised plain-state dimensions.

Observation repair (binding convention, ``docs/reference/cmdp-spec.md`` §1): under
CityLearn 2.5.0 the post-step observation vector carries uncomputed zeros for
computed slots (``net_electricity_consumption``, storage SoCs, indoor
temperature). This module reuses the week-2 runner's shared helpers
(:func:`energy_optimisation.evaluation.runner.repair_observation` and
:func:`~executed_step_values`) verbatim — no reimplementation — so controller
inputs are strictly causal with no lookahead.

Action mapping (plan §A1, fixed):

- ``dhw_storage = a0``, ``electrical_storage = a1`` (identity pass-through),
- ``cooling_device = (a2 + 1) / 2`` (affine from ``[-1, 1]`` to ``[0, 1]``).

RL action ``[0, 0, 0]`` is NOT neutral B0 — it maps to CityLearn
``[0, 0, 0.5]`` (half-rate cooling). The neutral action is
:data:`NEUTRAL_RL_ACTION` ``= [-1, -1, -1]``, which maps to CityLearn
``[-1, -1, 0]``. Under the pinned CityLearn 2.5.0 semantics (verified in
source: ``Battery.charge`` charges on (+) and discharges on (-) energy) and
this dataset (battery initialises at its depth-of-discharge floor ≈ 0.2 and
the DHW tank at 0.0), full-discharge requests are clamped to zero energy by
the device models, so ``[-1, -1, 0]`` is behaviourally identical to B0's
``[0, 0, 0]``; the anchor regression in ``tests/test_rl_env.py`` pins this
equivalence against the week-2 smoke KPIs within 1e-9.

Requested actions are clipped to the RL box before mapping; components outside
the box are counted as pre-clip violations before clipping.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from energy_optimisation.environment import load_environment
from energy_optimisation.evaluation.runner import (
    district_kpis_as_dict,
    executed_step_values,
    repair_observation,
)
from energy_optimisation.observation_names import build_observation_index

ACTION_DIMENSION = 3

# The RL action that is behaviourally equivalent to the B0 neutral baseline
# (maps to CityLearn [-1, -1, 0]; see module docstring).
NEUTRAL_RL_ACTION = np.array([-1.0, -1.0, -1.0])

DEFAULT_EPSILON = 1e-8


def map_rl_action_to_citylearn(action: np.ndarray) -> np.ndarray:
    """Map an RL action to the CityLearn action space (plan §A1 mapping).

    Parameters
    ----------
    action:
        RL action with shape ``(3,)``; storages in ``[-1, 1]``, cooling in
        ``[-1, 1]`` before the affine map.

    Returns
    -------
    np.ndarray
        CityLearn action ``[dhw_storage, electrical_storage, cooling_device]``
        where ``dhw_storage = a0``, ``electrical_storage = a1`` and
        ``cooling_device = (a2 + 1) / 2``.
    """

    values = np.asarray(action, dtype=float).reshape(ACTION_DIMENSION)
    return np.array([values[0], values[1], (values[2] + 1.0) / 2.0], dtype=float)


def count_pre_clip_violations(action: np.ndarray) -> int:
    """Count action components outside the RL box ``[-1, 1]`` before clipping."""

    values = np.asarray(action, dtype=float).reshape(ACTION_DIMENSION)
    return int(np.sum((values < -1.0) | (values > 1.0)))


def compute_cmdp_reward(
    net_electricity_consumption: float,
    indoor_dry_bulb_temperature: float,
    cooling_set_point: float,
    *,
    w_E: float,
    w_P: float,
    w_C: float,
    E_bar_b0: float,
    P_ref: float,
    comfort_band_c: float,
) -> float:
    """Evaluate the frozen CMDP reward (``docs/reference/cmdp-spec.md`` §4) verbatim.

    ``r_t = − w_E · (E_t / Ē_B0) − w_P · max(0, E_t − P_ref) / P_ref − w_C · D_t``
    with ``D_t = max(0, T_in − (T_set + ΔT))`` in °C. No rescaling, no extra
    terms, no clipping.
    """

    energy_term = -w_E * (float(net_electricity_consumption) / float(E_bar_b0))
    peak_term = (
        -w_P
        * max(0.0, float(net_electricity_consumption) - float(P_ref))
        / float(P_ref)
    )
    discomfort_c = max(
        0.0,
        float(indoor_dry_bulb_temperature)
        - (float(cooling_set_point) + float(comfort_band_c)),
    )
    comfort_term = -w_C * discomfort_c
    return float(energy_term + peak_term + comfort_term)


def resolve_normalisation_arrays(
    features: Mapping[str, Mapping[str, float]],
    layout: Sequence[str],
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(offsets, scales)`` arrays aligned to ``layout`` slots.

    Shared by :class:`CityLearnRLEnv` (training observations) and
    :class:`energy_optimisation.rl.controller.PPOController` (harness-side
    policy inputs) so both apply one frozen transform — never a second
    implementation.
    """

    offsets = np.zeros(len(layout), dtype=float)
    scales = np.ones(len(layout), dtype=float)
    missing = []
    for slot, name in enumerate(layout):
        stats = features.get(name)
        if stats is None:
            missing.append(name)
            continue
        offset = float(stats["offset"])
        scale = float(stats["scale"])
        if scale <= epsilon:
            scale = 1.0
        offsets[slot] = offset
        scales[slot] = scale
    if missing:
        raise KeyError(
            "normalisation stats lack features: " + ", ".join(missing)
        )
    return offsets, scales


def normalise_observation_vector(
    observation_vector: np.ndarray,
    offsets: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Min-max normalise with frozen ``(offset, scale)`` pairs into ``[0, 1]``.

    Excursions beyond the frozen statistic ranges saturate at the declared Box
    edges; the result is ``float32`` to match the policy's observation dtype.
    """

    normalised = (
        np.asarray(observation_vector, dtype=float) - offsets
    ) / scales
    return np.clip(normalised, 0.0, 1.0).astype(np.float32)


class CityLearnRLEnv(gym.Env):
    """Gymnasium wrapper producing normalised observations and CMDP rewards.

    Parameters
    ----------
    schema_path:
        CityLearn schema; loaded through
        :func:`energy_optimisation.environment.load_environment`.
    config:
        Mapping loaded from ``configs/week3-ppo.yaml``; provides the window
        defaults, frozen reward constants, and the frozen per-feature
        normalisation ``(offset, scale)`` pairs.
    simulation_start_time_step / simulation_end_time_step:
        Inclusive window bounds overriding the config's dev window when given.
    central_agent:
        Forwarded to ``load_environment`` (single-building schema uses True).
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        schema_path: str,
        *,
        config: Mapping[str, Any],
        simulation_start_time_step: Optional[int] = None,
        simulation_end_time_step: Optional[int] = None,
        central_agent: bool = True,
    ) -> None:
        windows = config.get("windows", {}).get("dev", {})
        start = (
            int(simulation_start_time_step)
            if simulation_start_time_step is not None
            else int(windows["simulation_start_time_step"])
        )
        end = (
            int(simulation_end_time_step)
            if simulation_end_time_step is not None
            else int(windows["simulation_end_time_step"])
        )
        if end <= start:
            raise ValueError("simulation_end_time_step must be greater than simulation_start_time_step")

        self._schema_path = str(schema_path)
        self._config = dict(config)
        self._start_time_step = start
        self._end_time_step = end

        self._citylearn_env = load_environment(
            schema_path,
            central_agent=central_agent,
            simulation_start_time_step=start,
            simulation_end_time_step=end,
        )
        self._building = self._citylearn_env.buildings[0]
        self._index = dict(build_observation_index(schema_path))

        observation_dimension = int(self._citylearn_env.observation_space[0].shape[0])
        self._offsets, self._scales = self._resolve_normalisation(config)

        # Spaces: observations are min-max normalised into [0, 1]; actions are
        # the symmetric RL box mapped to CityLearn coordinates inside `step`.
        self.observation_space = gym.spaces.Box(
            low=np.float32(0.0),
            high=np.float32(1.0),
            shape=(observation_dimension,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(ACTION_DIMENSION,),
            dtype=np.float32,
        )

        reward = self._reward_constants(config)
        self._reward_kwargs = reward

        self._observation_vector: Optional[np.ndarray] = None
        self._latest_executed: Dict[str, float] = {}
        self._pre_clip_violation_count = 0

    # --- construction helpers -------------------------------------------------

    @staticmethod
    def _reward_constants(config: Mapping[str, Any]) -> Dict[str, float]:
        try:
            reward = config["reward"]
        except KeyError:
            raise KeyError("config must provide a 'reward' block (configs/week3-ppo.yaml)")
        names = ("w_E", "w_P", "w_C", "E_bar_b0", "P_ref", "comfort_band_c")
        missing = [name for name in names if name not in reward]
        if missing:
            raise KeyError(f"config 'reward' block lacks constants: {', '.join(missing)}")
        return {name: float(reward[name]) for name in names}

    def _resolve_normalisation(
        self, config: Mapping[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(offsets, scales)`` arrays aligned to the observation layout."""

        normalisation = config.get("normalisation")
        if not normalisation or "features" not in normalisation:
            raise KeyError(
                "config must provide a 'normalisation.features' block "
                "(run scripts/09_compute_normalization_stats.py once)"
            )
        epsilon = float(normalisation.get("epsilon", DEFAULT_EPSILON))
        return resolve_normalisation_arrays(
            normalisation["features"], self.layout, epsilon=epsilon
        )

    # --- public introspection ---------------------------------------------------

    @property
    def citylearn_environment(self):
        """The underlying pinned CityLearn environment (read-only access)."""

        return self._citylearn_env

    @property
    def layout(self):
        """Ordered observation-slot names of the wrapped environment."""

        return list(build_observation_index(self._schema_path))

    @property
    def observation_dim(self) -> int:
        return int(self._citylearn_env.observation_space[0].shape[0])

    @property
    def pre_clip_violation_count(self) -> int:
        """Cumulative count of requested action components outside the RL box."""

        return self._pre_clip_violation_count

    @property
    def window_bounds(self) -> tuple:
        return (self._start_time_step, self._end_time_step)

    # --- normalisation ----------------------------------------------------------

    def _normalise(self, repaired_vector: np.ndarray) -> np.ndarray:
        """Min-max normalise with the frozen per-feature (offset, scale) pairs.

        Values are clipped into the declared ``[0, 1]`` observation space;
        excursions beyond the frozen statistic ranges saturate rather than
        leave the declared Box.
        """

        return normalise_observation_vector(repaired_vector, self._offsets, self._scales)

    # --- Gymnasium API ------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset to the window's first step, deterministic for a fixed seed."""

        super().reset(seed=seed, options=options)
        observations, info = self._citylearn_env.reset(seed=seed)
        self._observation_vector = np.asarray(observations[0], dtype=float)
        self._latest_executed = {}
        self._pre_clip_violation_count = 0
        controller_observation = repair_observation(
            self._observation_vector, self._index, self._latest_executed
        )
        return self._normalise(controller_observation), dict(info)

    def step(self, action):
        """Apply one mapped action; return ``(obs, reward, terminated, truncated, info)``.

        Contract (plan §A1): ``terminated`` is always False; ``truncated`` is
        True exactly at the window's terminal step. The reward is evaluated
        from executed step values (no lookahead): ``E_t`` and ``T_in`` come
        from the building time series consumed by ``env.evaluate()``; ``T_set``
        comes from the observation the action was chosen on.
        """

        if self._observation_vector is None:
            raise RuntimeError("step() called before reset()")
        chosen_on_vector = self._observation_vector

        requested = np.asarray(action, dtype=float).reshape(ACTION_DIMENSION)
        if not np.all(np.isfinite(requested)):
            raise ValueError(f"action must be finite, got {requested!r}")
        self._pre_clip_violation_count += count_pre_clip_violations(requested)
        clipped = np.clip(requested, -1.0, 1.0)
        applied = map_rl_action_to_citylearn(clipped)

        time_step = int(self._citylearn_env.time_step)
        result = self._citylearn_env.step([applied.tolist()])
        # CityLearn returns per-agent vectors: a list of 1 list for the
        # central agent (same structure as reset).
        step_observations = result[0]
        if (
            len(step_observations) == 1
            and hasattr(step_observations[0], "__len__")
        ):
            step_observations = step_observations[0]
        self._observation_vector = np.asarray(step_observations, dtype=float)

        # Executed-step values exist only after the step call has run
        # (CityLearn 2.5.0 measurement convention); they feed both the causal
        # repair of the returned observation and the frozen reward.
        executed = executed_step_values(self._building, time_step)
        self._latest_executed = executed

        cooling_set_point = float(
            chosen_on_vector[self._index["indoor_dry_bulb_temperature_cooling_set_point"]]
        )
        discomfort_c = max(
            0.0,
            executed["indoor_dry_bulb_temperature"]
            - (cooling_set_point + self._reward_kwargs["comfort_band_c"]),
        )
        reward = compute_cmdp_reward(
            executed["net_electricity_consumption"],
            executed["indoor_dry_bulb_temperature"],
            cooling_set_point,
            **self._reward_kwargs,
        )

        controller_observation = repair_observation(
            self._observation_vector, self._index, self._latest_executed
        )
        observation = self._normalise(controller_observation)

        terminated = False
        truncated = bool(self._citylearn_env.terminated)
        info: Dict[str, Any] = {
            "time_step": time_step,
            "requested_action": [float(value) for value in requested],
            "applied_action": [float(value) for value in applied],
            "pre_clip_violation_count": self._pre_clip_violation_count,
            "net_electricity_consumption": executed["net_electricity_consumption"],
            "indoor_dry_bulb_temperature": executed["indoor_dry_bulb_temperature"],
            "cooling_set_point": cooling_set_point,
            "discomfort_c": float(discomfort_c),
        }
        return observation, reward, terminated, truncated, info

    def evaluate(self) -> Dict[str, float]:
        """District KPIs via ``env.evaluate()`` (empty outage KPIs excluded)."""

        return district_kpis_as_dict(self._citylearn_environment.evaluate())

    def render(self):  # pragma: no cover - rendering unused in this project
        return None
