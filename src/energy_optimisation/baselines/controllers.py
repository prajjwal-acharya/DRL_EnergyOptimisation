"""Shared controller interface and the three Week 2 deterministic baselines.

Plan reference: docs/plans/week2-implementation-plan.md §B1 (interface) and §C0–§C2
(B0 neutral, B1 fixed-schedule, B2 tariff-aware). Every Week 2 baseline (and
September's PPO wrapper) implements exactly this interface; the evaluation
harness drives controllers only through :meth:`Controller.reset` and
:meth:`Controller.act`.

All frozen constants below come verbatim from the plan (§0 and §C). They are
declared once here so no logic branch hides a magic number; Phase D's config
lock (``configs/week2-baselines.yaml``) injects them through constructor
arguments, which is why every constant is also a keyword parameter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from energy_optimisation.observation_names import BUILDING_1_OBSERVATION_INDEX

ACTION_DIMENSION = 3

# --- B1 frozen schedule constants (plan §C1) ---------------------------------
# Inclusive [hour_start, hour_end, action_level] bands over the raw `hour`
# observation (CityLearn serves hours 1–24); levels are requested actions.
FIXED_ELECTRICAL_STORAGE_SCHEDULE: Tuple[Tuple[float, float, float], ...] = (
    (0.0, 5.0, -0.5),  # hours 0-5 -> charge
    (17.0, 20.0, 0.5),  # hours 17-20 -> discharge
)
FIXED_DHW_STORAGE_SCHEDULE: Tuple[Tuple[float, float, float], ...] = (
    (0.0, 5.0, -0.5),  # hours 0-5 -> charge
)
# First matching band wins; hours 8-11 and 21-22 -> 0.5, else 0.2.
FIXED_COOLING_DEVICE_SCHEDULE: Tuple[Tuple[float, float, float], ...] = (
    (12.0, 20.0, 0.8),  # hours 12-20
    (8.0, 11.0, 0.5),  # hours 8-11
    (21.0, 22.0, 0.5),  # hours 21-22
)
FIXED_ELECTRICAL_STORAGE_DEFAULT_LEVEL = 0.0
FIXED_DHW_STORAGE_DEFAULT_LEVEL = 0.0
FIXED_COOLING_DEVICE_DEFAULT_LEVEL = 0.2

# --- B2 frozen constants (plan §0 and §C2) -----------------------------------
TARIFF_THRESHOLD_USD_PER_KWH = 0.0439  # τ between mid and peak bands (frozen)
RESERVE_LOW_SOC = 0.2  # SoC reserve band lower edge (research assumption)
RESERVE_HIGH_SOC = 0.9  # SoC reserve band upper edge (research assumption)
PEAK_DISCHARGE_LEVEL = 0.5
OFF_PEAK_CHARGE_LEVEL = -0.5
PEAK_COOLING_DEVICE_LEVEL = 0.6


class Controller(ABC):
    """Stateless-per-episode controller over the central-agent observation.

    Attributes
    ----------
    name:
        Stable identifier used for artifact directories and comparison tables.
        Subclasses must set this class attribute.
    """

    name: str

    def reset(self, seed: int) -> None:
        """Prepare the controller for a new episode.

        The default implementation is a no-op for stateless deterministic
        controllers; stateful controllers must override it.

        Parameters
        ----------
        seed:
            Episode seed resolved by the runner (frozen to 42 in the Week 2
            config); controllers must not draw additional randomness beyond
            what they derive from it.
        """

        return None

    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return the requested action for one observation.

        Parameters
        ----------
        observation:
            Central-agent observation vector for the current time step.

        Returns
        -------
        np.ndarray
            Requested action with shape ``(3,)`` and float dtype, ordered
            ``[dhw_storage, electrical_storage, cooling_device]``. Values may
            lie outside the environment bounds; the harness clips and logs
            both requested and applied actions.
        """


def _scheduled_level(
    schedule: Sequence[Tuple[float, float, float]],
    hour: float,
    default_level: float,
) -> float:
    """Return the level of the first inclusive ``[start, end]`` band containing ``hour``."""

    for start, end, level in schedule:
        if start <= hour <= end:
            return level
    return default_level


class IndexedObservationController(Controller):
    """Base class for controllers that resolve observations by name.

    Controllers never use magic indices: every observation access goes through
    an observation-name → index mapping built by Phase A
    (:mod:`energy_optimisation.observation_names`).
    """

    REQUIRED_OBSERVATIONS: Tuple[str, ...] = ()

    def __init__(self, observation_index: Optional[Mapping[str, int]] = None) -> None:
        index = (
            dict(observation_index)
            if observation_index is not None
            else dict(BUILDING_1_OBSERVATION_INDEX)
        )
        missing = [name for name in self.REQUIRED_OBSERVATIONS if name not in index]
        if missing:
            raise KeyError(f"{type(self).__name__} requires observations: {', '.join(missing)}")
        self._observation_index = index

    def _value(self, observation: np.ndarray, name: str) -> float:
        return float(np.asarray(observation, dtype=float)[self._observation_index[name]])


class NeutralController(Controller):
    """B0 neutral baseline (plan §C0): request zero action at every step.

    Produces the same zeros as ``environment.neutral_actions`` for this action
    space; all three components lie inside the bounds, so harness clipping is
    a no-op.
    """

    name = "b0_neutral"

    def act(self, observation: np.ndarray) -> np.ndarray:
        return np.zeros(ACTION_DIMENSION, dtype=float)


class FixedScheduleController(IndexedObservationController):
    """B1 calendar-only baseline (plan §C1).

    Reads ONLY the ``hour`` observation. The price-blind fixed bands are the
    contrast condition for RQ3: B1 cannot adapt when tariffs change.
    """

    name = "b1_fixed_schedule"

    REQUIRED_OBSERVATIONS = ("hour",)

    def __init__(
        self,
        *,
        observation_index: Optional[Mapping[str, int]] = None,
        electrical_storage_schedule: Sequence[Tuple[float, float, float]] = (
            FIXED_ELECTRICAL_STORAGE_SCHEDULE
        ),
        dhw_storage_schedule: Sequence[Tuple[float, float, float]] = (
            FIXED_DHW_STORAGE_SCHEDULE
        ),
        cooling_device_schedule: Sequence[Tuple[float, float, float]] = (
            FIXED_COOLING_DEVICE_SCHEDULE
        ),
        electrical_storage_default_level: float = FIXED_ELECTRICAL_STORAGE_DEFAULT_LEVEL,
        dhw_storage_default_level: float = FIXED_DHW_STORAGE_DEFAULT_LEVEL,
        cooling_device_default_level: float = FIXED_COOLING_DEVICE_DEFAULT_LEVEL,
    ) -> None:
        super().__init__(observation_index=observation_index)
        self._electrical_storage_schedule = tuple(electrical_storage_schedule)
        self._dhw_storage_schedule = tuple(dhw_storage_schedule)
        self._cooling_device_schedule = tuple(cooling_device_schedule)
        self._electrical_storage_default_level = electrical_storage_default_level
        self._dhw_storage_default_level = dhw_storage_default_level
        self._cooling_device_default_level = cooling_device_default_level

    def act(self, observation: np.ndarray) -> np.ndarray:
        hour = self._value(observation, "hour")
        return np.array(
            [
                _scheduled_level(
                    self._dhw_storage_schedule, hour, self._dhw_storage_default_level
                ),
                _scheduled_level(
                    self._electrical_storage_schedule,
                    hour,
                    self._electrical_storage_default_level,
                ),
                _scheduled_level(
                    self._cooling_device_schedule, hour, self._cooling_device_default_level
                ),
            ],
            dtype=float,
        )


class TariffAwareController(IndexedObservationController):
    """B2 current-price heuristic (plan §C2).

    Reads ONLY ``electricity_pricing``, ``hour``, ``electrical_storage_soc``
    and ``dhw_storage_soc`` — no lookahead of any kind, an interpretable
    heuristic that discharges storage and protects comfort above the frozen
    tariff threshold τ and charges below it.
    """

    name = "b2_tariff_aware"

    REQUIRED_OBSERVATIONS = (
        "electricity_pricing",
        "hour",
        "electrical_storage_soc",
        "dhw_storage_soc",
    )

    def __init__(
        self,
        *,
        observation_index: Optional[Mapping[str, int]] = None,
        tariff_threshold_usd_per_kwh: float = TARIFF_THRESHOLD_USD_PER_KWH,
        reserve_low_soc: float = RESERVE_LOW_SOC,
        reserve_high_soc: float = RESERVE_HIGH_SOC,
        peak_discharge_level: float = PEAK_DISCHARGE_LEVEL,
        off_peak_charge_level: float = OFF_PEAK_CHARGE_LEVEL,
        peak_cooling_device_level: float = PEAK_COOLING_DEVICE_LEVEL,
        cooling_device_schedule: Sequence[Tuple[float, float, float]] = (
            FIXED_COOLING_DEVICE_SCHEDULE
        ),
        cooling_device_default_level: float = FIXED_COOLING_DEVICE_DEFAULT_LEVEL,
    ) -> None:
        super().__init__(observation_index=observation_index)
        self._tariff_threshold_usd_per_kwh = tariff_threshold_usd_per_kwh
        self._reserve_low_soc = reserve_low_soc
        self._reserve_high_soc = reserve_high_soc
        self._peak_discharge_level = peak_discharge_level
        self._off_peak_charge_level = off_peak_charge_level
        self._peak_cooling_device_level = peak_cooling_device_level
        self._cooling_device_schedule = tuple(cooling_device_schedule)
        self._cooling_device_default_level = cooling_device_default_level

    def act(self, observation: np.ndarray) -> np.ndarray:
        hour = self._value(observation, "hour")
        price = self._value(observation, "electricity_pricing")
        electrical_soc = self._value(observation, "electrical_storage_soc")
        dhw_soc = self._value(observation, "dhw_storage_soc")

        if price >= self._tariff_threshold_usd_per_kwh:
            dhw_action = (
                self._peak_discharge_level if dhw_soc >= self._reserve_low_soc else 0.0
            )
            electrical_action = (
                self._peak_discharge_level
                if electrical_soc >= self._reserve_low_soc
                else 0.0
            )
            cooling_action = self._peak_cooling_device_level
        else:
            dhw_action = (
                self._off_peak_charge_level if dhw_soc <= self._reserve_high_soc else 0.0
            )
            electrical_action = (
                self._off_peak_charge_level
                if electrical_soc <= self._reserve_high_soc
                else 0.0
            )
            cooling_action = _scheduled_level(
                self._cooling_device_schedule, hour, self._cooling_device_default_level
            )

        return np.array([dhw_action, electrical_action, cooling_action], dtype=float)
