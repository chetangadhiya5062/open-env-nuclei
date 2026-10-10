"""Gymnasium wrapper around the data-cleaning environment (in-process, no HTTP). Needs ``pip install -e ".[rl]"``.

* observation: ``featurize`` vector (see ``rl_common``), shape ``(N_FEATURES,)``
* action: ``Discrete(N_ACTIONS)`` over the curated action list
* reward: the environment's own shaped reward, unchanged
* every episode picks a task and a seed at random; **training seeds are >= TRAIN_SEED_MIN**, while validation uses
  seeds 50-59 and the benchmark/test uses seeds 0-9, so evaluation seeds are never seen during training.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from data_cleaning_env.server.data_cleaning_env_environment import DataCleaningEnvironment
from data_cleaning_env.server.datagen import TASKS

from .rl_common import N_ACTIONS, N_FEATURES, build_action, featurize

TRAIN_SEED_MIN = 100
TRAIN_SEED_MAX = 100_000
VALIDATION_SEEDS = tuple(range(50, 60))
TEST_SEEDS = tuple(range(0, 10))


class DataCleaningGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self, tasks: Sequence[str] = tuple(TASKS), seed_range: Tuple[int, int] = (TRAIN_SEED_MIN, TRAIN_SEED_MAX)
    ):
        super().__init__()
        self.tasks = tuple(tasks)
        self.seed_range = seed_range
        self.observation_space = spaces.Box(low=-0.5, high=1.5, shape=(N_FEATURES,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._env = DataCleaningEnvironment()
        self._obs = None
        self._initial_rows = 1

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        options = options or {}
        task_id = options.get("task_id") or self.tasks[int(self.np_random.integers(len(self.tasks)))]
        ep_seed = options.get("seed")
        if ep_seed is None:
            ep_seed = int(self.np_random.integers(self.seed_range[0], self.seed_range[1]))
        self._obs = self._env.reset(task_id=task_id, seed=ep_seed)
        self._initial_rows = self._obs.row_count
        return featurize(self._obs, self._initial_rows), {"task_id": task_id, "seed": ep_seed}

    def step(self, action: int):
        action_obj = build_action(int(action), self._obs)
        self._obs = self._env.step(action_obj)
        info: Dict[str, Any] = {"last_action_ok": self._obs.last_action_ok}
        if self._obs.done:
            info["final_score"] = self._obs.final_score
            info["improvement"] = (self._obs.score_breakdown or {}).get("improvement")
            info["task_id"] = self._obs.task_id
        return (
            featurize(self._obs, self._initial_rows),
            float(self._obs.reward or 0.0),
            bool(self._obs.done),
            False,
            info,
        )
