# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Data Cleaning Env Environment Implementation.
"""

import pandas as pd
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from openenv.core.client_types import StepResult  # ✅ IMPORTANT

try:
    from ..models import DataCleaningAction, DataCleaningObservation
except ImportError:
    from models import DataCleaningAction, DataCleaningObservation


class DataCleaningEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.df: pd.DataFrame = None
        self.original_df: pd.DataFrame = None

    def reset(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)

        data = {
            "name": ["Alice", "Bob", "Alice", "Charlie", "David"],
            "age": [25, None, 25, 30, None],
            "city": ["Delhi", "Mumbai", "Delhi", None, "Pune"]
        }

        self.df = pd.DataFrame(data)
        self.original_df = self.df.copy()

        return self._get_obs()  

    def step(self, action: DataCleaningAction):
        self._state.step_count += 1

        prev_missing = self.df.isnull().sum().sum()
        prev_rows = len(self.df)
        prev_duplicates = self.df.duplicated().sum()

        reward_score = 0.0

        if action.action_type == "fill_missing":
            if action.column_name in self.df.columns:
                fill_value = action.value if action.value is not None else "missing"
                self.df[action.column_name] = self.df[action.column_name].fillna(fill_value)
            else:
                reward_score -= 1.0

        elif action.action_type == "drop_rows_with_missing":
            self.df = self.df.dropna()

        elif action.action_type == "remove_duplicates":
            self.df = self.df.drop_duplicates()

        new_missing = self.df.isnull().sum().sum()
        new_rows = len(self.df)
        new_duplicates = self.df.duplicated().sum()

        # =========================
        # ACTION-AWARE REWARD (FINAL)
        # =========================
        if action.action_type == "fill_missing":
            if new_missing < prev_missing:
                reward_score += 2.0   # strong reward
            else:
                reward_score -= 1.0   # useless action

        elif action.action_type == "remove_duplicates":
            if new_duplicates < prev_duplicates:
                reward_score += 1.5   # correct behavior
            else:
                reward_score -= 0.5   # no effect

        elif action.action_type == "drop_rows_with_missing":
            if new_rows < prev_rows:
                reward_score -= 2.0   # heavy penalty
            else:
                reward_score -= 1.0

        done = (new_missing == 0 and new_duplicates == 0) or self._state.step_count >= 10

        obs = self._get_obs()
        obs.reward = reward_score
        obs.done = done

        return obs   # ✅ ONLY observation

    def _get_obs(self) -> DataCleaningObservation:
        return DataCleaningObservation(
            missing_values_count_per_column=self.df.isnull().sum().to_dict(),
            duplicate_row_count=int(self.df.duplicated().sum()),
            total_row_count=len(self.df),
            column_names=list(self.df.columns),
        )

    @property
    def state(self) -> State:
        return self._state