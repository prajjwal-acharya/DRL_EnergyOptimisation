"""PPO controller wrapper implementing the week-2 Controller interface.

Plan reference: docs/plans/week3-implementation-plan.md §C1. :class:`PPOController`
loads a stable-baselines3 PPO checkpoint and plugs into the locked evaluation
harness (:mod:`energy_optimisation.evaluation.runner`) unchanged: the harness
repairs the observation causally and calls :meth:`~Controller.act`, which
normalises the raw central-agent vector with the frozen per-feature
``(offset, scale)`` pairs from ``configs/week3-ppo.yaml`` — the identical
transform the Phase A adapter applied during training, shared through
:func:`energy_optimisation.rl.env_adapter.resolve_normalisation_arrays` — and
returns the mapped 3-dim CityLearn action, deterministic (no sampling).

Action mapping (plan §A1): ``dhw_storage = a0``, ``electrical_storage = a1``,
``cooling_device = (a2 + 1) / 2``. The controller performs no clipping of its
own; requested actions are recorded by the harness trace and clipped to the
environment bounds exactly like every week-2 baseline, so any out-of-box
policy request surfaces as a counted clipping event.

The episode return reported in checkpoint evaluations is reconstructed from
the harness trace with the frozen CMDP reward formula via
:func:`episode_return_from_trace`: the trace's executed ``E_t``, ``T_in`` and
pre-step ``T_set`` columns are the same values the adapter consumed during
training, so the sum equals the training episode return.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd

from energy_optimisation.baselines.controllers import Controller
from energy_optimisation.observation_names import BUILDING_1_OBSERVATION_INDEX
from energy_optimisation.rl.env_adapter import (
    DEFAULT_EPSILON,
    compute_cmdp_reward,
    map_rl_action_to_citylearn,
    normalise_observation_vector,
    resolve_normalisation_arrays,
)


class PPOController(Controller):
    """Deterministic SB3-PPO policy over the repaired CityLearn observation.

    Parameters
    ----------
    model_path:
        Path to an SB3 ``.zip`` checkpoint saved by ``scripts/10_train_ppo.py``.
    config:
        Mapping loaded from ``configs/week3-ppo.yaml``; provides the frozen
        per-feature normalisation ``(offset, scale)`` pairs.
    observation_index:
        Observation-name → slot mapping; defaults to the canonical
        Building_1 central-agent index.
    device:
        Torch device for policy inference; pinned to ``cpu`` per plan §0.
    name:
        Optional controller identifier override (e.g. ``"ppo_seed42"``);
        defaults to ``"ppo"``.
    """

    name = "ppo"

    def __init__(
        self,
        model_path: Union[str, Path],
        config: Mapping[str, Any],
        *,
        observation_index: Optional[Mapping[str, int]] = None,
        device: str = "cpu",
        name: Optional[str] = None,
    ) -> None:
        from stable_baselines3 import PPO  # local import keeps harness imports light

        self._observation_index = (
            dict(observation_index)
            if observation_index is not None
            else dict(BUILDING_1_OBSERVATION_INDEX)
        )
        layout = self._layout_from_index(self._observation_index)

        normalisation = config.get("normalisation")
        if not normalisation or "features" not in normalisation:
            raise KeyError(
                "config must provide a 'normalisation.features' block "
                "(configs/week3-ppo.yaml)"
            )
        epsilon = float(normalisation.get("epsilon", DEFAULT_EPSILON))
        self._offsets, self._scales = resolve_normalisation_arrays(
            normalisation["features"], layout, epsilon=epsilon
        )

        self.model = PPO.load(str(model_path), device=str(device))
        policy_dimension = int(self.model.observation_space.shape[0])
        if policy_dimension != len(layout):
            raise ValueError(
                f"checkpoint expects {policy_dimension} observation dims but "
                f"{len(layout)} slots were resolved from the schema"
            )

        if name is not None:
            self.name = str(name)

    @staticmethod
    def _layout_from_index(observation_index: Mapping[str, int]) -> list:
        """Invert a name→slot mapping into an ordered per-slot name list."""

        dimension = max(int(slot) for slot in observation_index.values()) + 1
        layout = [""] * dimension
        for observation_name, slot in observation_index.items():
            if layout[int(slot)]:
                raise ValueError(f"slot {slot} claimed twice in observation index")
            layout[int(slot)] = observation_name
        missing = [str(slot) for slot, entry in enumerate(layout) if not entry]
        if missing:
            raise ValueError(f"observation index leaves slots unnamed: {', '.join(missing)}")
        return layout

    # --- Controller interface ---------------------------------------------------

    def reset(self, seed: int) -> None:
        """Stateless-per-episode (MlpPolicy): nothing to reset."""

        return None

    def normalise_observation(self, observation: np.ndarray) -> np.ndarray:
        """Apply the frozen transform: ``(x - offset) / scale``, clip to [0, 1]."""

        raw = np.asarray(observation, dtype=float).reshape(-1)
        expected = self._offsets.shape[0]
        if raw.shape[0] != expected:
            raise ValueError(
                f"observation has {raw.shape[0]} dims; expected {expected}"
            )
        return normalise_observation_vector(raw, self._offsets, self._scales)

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return the mapped 3-dim CityLearn action, deterministic."""

        normalised = self.normalise_observation(observation)
        action, _ = self.model.predict(normalised, deterministic=True)
        return map_rl_action_to_citylearn(np.asarray(action, dtype=float))


def episode_return_from_trace(
    trace: pd.DataFrame,
    *,
    w_E: float,
    w_P: float,
    w_C: float,
    E_bar_b0: float,
    P_ref: float,
    comfort_band_c: float,
) -> float:
    """Sum the frozen CMDP reward over one evaluation-harness trace.

    Each row contributes ``r_t`` evaluated on the executed step values the
    runner recorded (``E_t``, ``T_in``) and the pre-step cooling set-point the
    action was chosen on — the same quantities the Phase A adapter used, so
    this reproduces the training episode return exactly.
    """

    required_columns = (
        "net_electricity_consumption",
        "indoor_dry_bulb_temperature",
        "indoor_dry_bulb_temperature_cooling_set_point",
    )
    missing = [column for column in required_columns if column not in trace.columns]
    if missing:
        raise KeyError(f"Trace lacks required columns: {', '.join(missing)}")

    total = 0.0
    energies = trace["net_electricity_consumption"].to_numpy(dtype=float)
    indoor = trace["indoor_dry_bulb_temperature"].to_numpy(dtype=float)
    set_points = trace["indoor_dry_bulb_temperature_cooling_set_point"].to_numpy(dtype=float)
    for net_consumption, temperature, set_point in zip(energies, indoor, set_points):
        total += compute_cmdp_reward(
            net_consumption,
            temperature,
            set_point,
            w_E=w_E,
            w_P=w_P,
            w_C=w_C,
            E_bar_b0=E_bar_b0,
            P_ref=P_ref,
            comfort_band_c=comfort_band_c,
        )
    return float(total)


def reward_constants_from_config(config: Mapping[str, Any]) -> Dict[str, float]:
    """Extract the frozen reward constants from ``configs/week3-ppo.yaml``."""

    reward = config["reward"]
    return {
        "w_E": float(reward["w_E"]),
        "w_P": float(reward["w_P"]),
        "w_C": float(reward["w_C"]),
        "E_bar_b0": float(reward["E_bar_b0"]),
        "P_ref": float(reward["P_ref"]),
        "comfort_band_c": float(reward["comfort_band_c"]),
    }
