"""Week 3 reinforcement-learning package (standard PPO controller).

Plan reference: docs/plans/week3-implementation-plan.md §A–§C. Exposes the Gymnasium
adapter around the pinned CityLearn environment, the pure mapping/reward
helpers consumed by the adapter and its tests, the PPO controller wrapper that
plugs into the locked evaluation harness, and the frozen checkpoint-selection
rule.
"""

from energy_optimisation.rl.checkpoint_selection import (
    EVALUATION_COLUMNS,
    SELECTION_RULE_TEXT,
    select_best_checkpoint,
)
from energy_optimisation.rl.controller import (
    PPOController,
    episode_return_from_trace,
    reward_constants_from_config,
)
from energy_optimisation.rl.env_adapter import (
    NEUTRAL_RL_ACTION,
    CityLearnRLEnv,
    compute_cmdp_reward,
    count_pre_clip_violations,
    map_rl_action_to_citylearn,
)

__all__ = [
    "EVALUATION_COLUMNS",
    "NEUTRAL_RL_ACTION",
    "PPOController",
    "SELECTION_RULE_TEXT",
    "CityLearnRLEnv",
    "compute_cmdp_reward",
    "count_pre_clip_violations",
    "episode_return_from_trace",
    "map_rl_action_to_citylearn",
    "reward_constants_from_config",
    "select_best_checkpoint",
]
