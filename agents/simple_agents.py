"""Trivial baselines: they bound what a "real" agent has to beat."""

import random
from typing import Optional

from data_cleaning_env.models import DataCleaningAction, DataCleaningObservation

from .base import Agent

_STRATEGIES = ["mean", "median", "mode", "constant"]
_TARGETS = ["int", "float", "str"]


class DoNothingAgent(Agent):
    """Finishes immediately: the score of the untouched dirty table (a floor)."""

    name = "do-nothing"

    def act(self, obs: DataCleaningObservation) -> DataCleaningAction:
        return DataCleaningAction(action_type="finish")


class RandomAgent(Agent):
    """Picks a random action type and random arguments. Many of its actions are invalid, on purpose."""

    name = "random"

    def __init__(self, seed: int = 0, finish_prob: float = 0.05) -> None:
        self._base_seed = seed
        self._finish_prob = finish_prob
        self._rng = random.Random(seed)

    def reset(self, seed: int = 0) -> None:
        super().reset(seed)
        self._rng = random.Random(f"{self._base_seed}-{seed}")

    def act(self, obs: DataCleaningObservation) -> DataCleaningAction:
        r = self._rng
        if r.random() < self._finish_prob:
            return DataCleaningAction(action_type="finish")
        columns = [c.name for c in obs.columns]
        kind = r.choice(
            [
                "fill_missing",
                "drop_duplicates",
                "standardize_categories",
                "cast_type",
                "strip_whitespace",
                "fix_dates",
                "clip_outliers",
                "drop_rows_with_missing",
            ]
        )
        col: Optional[str] = r.choice(columns)
        if kind == "fill_missing":
            strategy = r.choice(_STRATEGIES)
            return DataCleaningAction(
                action_type=kind, column_name=col, strategy=strategy, value="0" if strategy == "constant" else None
            )
        if kind == "cast_type":
            return DataCleaningAction(action_type=kind, column_name=col, target_type=r.choice(_TARGETS))
        if kind in ("drop_duplicates",):
            return DataCleaningAction(action_type=kind)
        if kind in ("strip_whitespace", "drop_rows_with_missing") and r.random() < 0.5:
            return DataCleaningAction(action_type=kind)
        return DataCleaningAction(action_type=kind, column_name=col)
