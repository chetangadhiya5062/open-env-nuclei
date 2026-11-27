"""Leakage-safe train/test splitting (Kaufman et al. 2012; Kapoor & Narayanan 2023).

Rules:

* rows with identical feature values (exact duplicates, possibly with different labels) always land on the same side,
  so the test set never contains a copy of a training row;
* ``group`` keeps every row of one entity (customer, patient, ...) on one side;
* ``time`` puts the most recent rows in the test set (no training on the future);
* ``stratified`` keeps the class balance (combined with the duplicate groups via ``StratifiedGroupKFold``).

The report says what was done and checks that no feature row is shared between the two sides.
"""

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .profile import infer_task
from .types import parse_datetime

Strategy = Literal["auto", "random", "stratified", "group", "time"]


def _row_hash(df: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(df.astype(str), index=False)


def leakage_safe_split(
    df: pd.DataFrame,
    target: Optional[str] = None,
    test_size: float = 0.2,
    strategy: Strategy = "auto",
    group_column: Optional[str] = None,
    time_column: Optional[str] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    for col, what in ((target, "target"), (group_column, "group_column"), (time_column, "time_column")):
        if col is not None and col not in df.columns:
            raise KeyError(f"{what} {col!r} not found")
    if len(df) < 4:
        raise ValueError("need at least 4 rows to split")
    if strategy == "auto":
        if time_column:
            strategy = "time"
        elif group_column:
            strategy = "group"
        elif target and infer_task(df[target]) == "classification":
            strategy = "stratified"
        else:
            strategy = "random"
    data = df.reset_index(drop=True)
    features = data.drop(columns=[target]) if target else data
    dup_groups = _row_hash(features)
    groups = dup_groups if group_column is None else data[group_column].astype(str)

    if strategy == "time":
        if not time_column:
            raise ValueError("strategy 'time' needs time_column")
        t = parse_datetime(data[time_column])
        if t.isna().all():
            raise ValueError(f"time_column {time_column!r} has no parsable dates")
        cutoff = t.quantile(1 - test_size)
        is_test = (t >= cutoff).to_numpy()
        # duplicates must not straddle the cut: a group goes to test if any member is in test
        test_groups = set(groups[is_test])
        is_test = groups.isin(test_groups).to_numpy()
    else:
        is_test = _group_split(data, groups, target, strategy, test_size, random_state)

    train, test = data.loc[~is_test].reset_index(drop=True), data.loc[is_test].reset_index(drop=True)
    tr_hash, te_hash = set(dup_groups[~is_test]), set(dup_groups[is_test])
    report: Dict[str, Any] = {
        "strategy": strategy,
        "test_size": test_size,
        "train_rows": len(train),
        "test_rows": len(test),
        "duplicate_feature_rows": int(dup_groups.duplicated().sum()),
        "shared_feature_rows": len(tr_hash & te_hash),
        "group_column": group_column,
        "time_column": time_column,
    }
    if group_column:
        report["shared_groups"] = len(set(train[group_column].astype(str)) & set(test[group_column].astype(str)))
    if target and infer_task(df[target]) == "classification":
        report["class_balance"] = {
            "train": train[target].astype(str).value_counts(normalize=True).round(4).to_dict(),
            "test": test[target].astype(str).value_counts(normalize=True).round(4).to_dict(),
        }
    if time_column:
        tt, te = parse_datetime(train[time_column]), parse_datetime(test[time_column])
        report["train_time_range"] = [str(tt.min()), str(tt.max())]
        report["test_time_range"] = [str(te.min()), str(te.max())]
    return train, test, report


def _group_split(
    data: pd.DataFrame, groups: pd.Series, target: Optional[str], strategy: str, test_size: float, seed: int
) -> np.ndarray:
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

    codes = groups.astype("category").cat.codes.to_numpy()
    n = len(data)
    if strategy == "stratified" and target:
        y = data[target].astype(str).fillna("__nan__").to_numpy()
        n_splits = max(2, int(round(1 / test_size)))
        if pd.Series(y).value_counts().min() >= n_splits:
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            _, test_idx = next(sgkf.split(np.zeros(n), y, codes))
            mask = np.zeros(n, dtype=bool)
            mask[test_idx] = True
            return mask
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    _, test_idx = next(gss.split(np.zeros(n), groups=codes))
    mask = np.zeros(n, dtype=bool)
    mask[test_idx] = True
    return mask
