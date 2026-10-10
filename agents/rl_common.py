"""Shared pieces of the RL agent: the observation -> feature-vector map and the curated action list.

No heavy dependencies here (only numpy), so the deployed server can run a trained policy without torch.

Action space (``ACTIONS``): a fixed, curated list of 19 concrete actions (action + column + arguments), including
``finish`` and the "trap" ``drop_rows_with_missing``. Two design notes worth knowing:

* ``standardize_aliases:<col>`` is a *macro*: it builds the alias mapping (``NY -> New York``) from the
  observation's ``bad_values`` with the same prefix/initials matcher the rule-based agent uses. So the policy does not
  have to invent mappings; it only has to learn *when* to use it. This is action-space engineering and is stated in
  the README.
* fill strategies are fixed per column (median/mean for numbers, mode for categories).
"""

from typing import Dict, List, Optional

import numpy as np

from data_cleaning_env.models import DataCleaningAction, DataCleaningObservation
from data_cleaning_env.schema import COLUMNS

from .rule_based import suggest_mapping

ISSUE_KEYS = ("missing", "wrong_type", "out_of_range", "whitespace", "not_allowed_value", "bad_date_format")
MAX_STEPS = 45  # largest step budget over all tasks
FEATURES_PER_COLUMN = len(ISSUE_KEYS) + 1  # issue fractions + "dtype is object"
N_FEATURES = len(COLUMNS) * FEATURES_PER_COLUMN + 5


def _fill(col: str, strategy: str) -> dict:
    return {"action_type": "fill_missing", "column_name": col, "strategy": strategy}


# (label, payload). A payload of None marks a macro handled in ``build_action``.
ACTIONS: List[tuple] = [
    ("drop_duplicates", {"action_type": "drop_duplicates"}),
    ("strip_whitespace", {"action_type": "strip_whitespace"}),
    ("standardize:name", {"action_type": "standardize_categories", "column_name": "name"}),
    ("standardize:city", {"action_type": "standardize_categories", "column_name": "city"}),
    ("standardize:plan", {"action_type": "standardize_categories", "column_name": "plan"}),
    ("standardize_aliases:city", None),
    ("cast:age:int", {"action_type": "cast_type", "column_name": "age", "target_type": "int"}),
    ("cast:monthly_spend:float", {"action_type": "cast_type", "column_name": "monthly_spend", "target_type": "float"}),
    ("fix_dates:signup_date", {"action_type": "fix_dates", "column_name": "signup_date"}),
    ("clip:age", {"action_type": "clip_outliers", "column_name": "age"}),
    ("clip:monthly_spend", {"action_type": "clip_outliers", "column_name": "monthly_spend"}),
    ("fill:age:median", _fill("age", "median")),
    ("fill:age:mean", _fill("age", "mean")),
    ("fill:monthly_spend:median", _fill("monthly_spend", "median")),
    ("fill:monthly_spend:mean", _fill("monthly_spend", "mean")),
    ("fill:city:mode", _fill("city", "mode")),
    ("fill:plan:mode", _fill("plan", "mode")),
    ("drop_rows_with_missing", {"action_type": "drop_rows_with_missing"}),
    ("finish", {"action_type": "finish"}),
]
ACTION_LABELS = [label for label, _ in ACTIONS]
N_ACTIONS = len(ACTIONS)
FINISH_INDEX = ACTION_LABELS.index("finish")


def build_action(index: int, obs: DataCleaningObservation) -> DataCleaningAction:
    label, payload = ACTIONS[index]
    if payload is not None:
        return DataCleaningAction(**payload)
    # macro: alias mapping for a category column
    col = label.split(":")[1]
    profile = next(c for c in obs.columns if c.name == col)
    mapping: Optional[Dict[str, str]] = suggest_mapping(profile.bad_values, profile.allowed_values or [])
    if not mapping:  # nothing to map: a harmless no-op the reward will penalise
        return DataCleaningAction(action_type="standardize_categories", column_name=col)
    return DataCleaningAction(action_type="standardize_categories", column_name=col, mapping=mapping)


def featurize(obs: DataCleaningObservation, initial_rows: int) -> np.ndarray:
    """Fixed-size float32 vector with everything scaled to roughly [0, 1]."""
    rows = max(1, initial_rows)
    by_name = {c.name: c for c in obs.columns}
    feats: List[float] = []
    for name in COLUMNS:
        c = by_name[name]
        issues = dict(c.issues)
        issues["missing"] = c.missing
        feats.extend(min(1.0, issues.get(k, 0) / rows) for k in ISSUE_KEYS)
        feats.append(1.0 if c.dtype == "object" else 0.0)
    feats.extend(
        [
            min(1.0, obs.duplicate_count / rows),
            obs.quality_score,
            obs.steps_remaining / MAX_STEPS,
            1.0 if obs.last_action_ok else 0.0,
            min(1.5, obs.row_count / rows),
        ]
    )
    return np.asarray(feats, dtype=np.float32)
