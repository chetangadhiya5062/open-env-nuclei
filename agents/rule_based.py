"""A strong hand-written baseline.

It uses only what the observation shows (column profiles, issue counts, ``allowed_values`` and
``bad_values``) and works through a fixed checklist:

    drop_duplicates -> strip_whitespace -> standardize categories (auto, then alias mapping) -> cast numeric
    -> fix_dates -> clip_outliers -> fill_missing (median / mode) -> finish

It does not know anything about specific cities: aliases like ``NY`` are matched to ``allowed_values``
by normalised prefix/initials. Each (action, column) pair is tried at most once, so it always terminates.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

from data_cleaning_env.models import ColumnProfile, DataCleaningAction, DataCleaningObservation

from .base import Agent


def _norm(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def suggest_mapping(bad_values: List[str], allowed: List[str]) -> Dict[str, str]:
    """Map leftover spellings to an allowed value when the match is unambiguous."""
    mapping: Dict[str, str] = {}
    for bad in bad_values:
        nb = _norm(bad)
        if not nb:
            continue
        hits = set()
        for a in allowed:
            na = _norm(a)
            initials = "".join(w[0] for w in re.findall(r"[A-Za-z]+", a.casefold()))
            by_initials = len(initials) >= 2 and nb.startswith(initials) and len(nb) <= len(initials) + 2
            if nb == na or by_initials or (len(nb) >= 3 and na.startswith(nb)):
                hits.add(a)
        if len(hits) == 1:
            mapping[bad] = hits.pop()
    return mapping


class RuleBasedAgent(Agent):
    name = "rule-based"

    def reset(self, seed: int = 0) -> None:
        super().reset(seed)
        self._tried: Set[Tuple[str, Optional[str]]] = set()

    def _once(self, kind: str, column: Optional[str] = None) -> bool:
        key = (kind, column)
        if key in self._tried:
            return False
        self._tried.add(key)
        return True

    def act(self, obs: DataCleaningObservation) -> DataCleaningAction:
        if not hasattr(self, "_tried"):
            self.reset()
        cols: List[ColumnProfile] = obs.columns

        if obs.duplicate_count and self._once("drop_duplicates"):
            return DataCleaningAction(action_type="drop_duplicates")

        if any(c.issues.get("whitespace") for c in cols) and self._once("strip_whitespace"):
            return DataCleaningAction(action_type="strip_whitespace")

        for c in cols:
            if c.allowed_values and c.issues.get("not_allowed_value"):
                if self._once("standardize_categories", c.name):
                    return DataCleaningAction(action_type="standardize_categories", column_name=c.name)
                mapping = suggest_mapping(c.bad_values, c.allowed_values)
                if mapping and self._once("standardize_mapping", c.name):
                    return DataCleaningAction(action_type="standardize_categories", column_name=c.name, mapping=mapping)

        for c in cols:
            if c.expected_type in ("int", "float") and (c.issues.get("wrong_type") or c.dtype == "object"):
                if self._once("cast_type", c.name):
                    return DataCleaningAction(action_type="cast_type", column_name=c.name, target_type=c.expected_type)

        for c in cols:
            if c.issues.get("bad_date_format") and self._once("fix_dates", c.name):
                return DataCleaningAction(action_type="fix_dates", column_name=c.name)

        for c in cols:
            if c.issues.get("out_of_range") and self._once("clip_outliers", c.name):
                return DataCleaningAction(action_type="clip_outliers", column_name=c.name)

        for c in cols:
            if c.missing and self._once("fill_missing", c.name):
                strategy = "median" if c.expected_type in ("int", "float") else "mode"
                return DataCleaningAction(action_type="fill_missing", column_name=c.name, strategy=strategy)

        return DataCleaningAction(action_type="finish")
