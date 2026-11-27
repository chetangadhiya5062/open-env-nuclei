"""One-click mode: build a sensible recipe from the data's profile.

Levels
------
Columns the profiler flags as target leaks (``target_copy``, ``possible_target_leak``) are dropped by the first step;
remove them from ``drop_columns`` to keep them.

* ``clean``    cleaning only (drop unusable columns, normalise strings, fix types, deduplicate, cap outliers, impute)
* ``standard`` clean + datetime / text features, skew transforms, categorical encoding (default)
* ``full``     standard + interactions, group aggregations, OpenFE-style generation and feature selection

Every choice is a plain recipe step, so the result can be inspected, edited and saved before it is run.
"""

from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from .profile import infer_task, profile
from .recipe import Recipe, StepSpec
from .types import ColumnType, infer_types

Level = Literal["clean", "standard", "full"]
ONE_HOT_MAX = 15


def _step(name: str, **params: Any) -> StepSpec:
    return StepSpec(step=name, params=params)


def auto_recipe(
    df: pd.DataFrame,
    target: Optional[str] = None,
    level: Level = "standard",
    id_columns: Optional[List[str]] = None,
    report: Optional[Dict[str, Any]] = None,
    name: str = "auto",
) -> Recipe:
    """Recipe tailored to ``df``. Pass a precomputed ``report`` (from :func:`profile`) to avoid profiling twice."""
    if target is not None and target not in df.columns:
        raise KeyError(f"target column {target!r} not found")
    report = report or profile(df, target=target)
    types = {c["name"]: ColumnType(c["type"]) for c in report["columns"]}
    cols = {c["name"]: c for c in report["columns"]}
    task = infer_task(df[target]) if target else "none"
    supervised = task in ("classification", "regression")
    feats = {
        c: t
        for c, t in types.items()
        if c != target
        and c not in (id_columns or [])
        and t not in (ColumnType.ID, ColumnType.CONSTANT)
        and (cols[c]["missing_pct"] or 0) <= 0.95
    }

    def any_type(*kinds: ColumnType) -> bool:
        return any(t in kinds for t in feats.values())

    leaks = sorted(
        {
            w["column"]
            for w in report["warnings"]
            if w["code"] in ("target_copy", "possible_target_leak") and w["column"] in feats
        }
    )
    for c in leaks:
        feats.pop(c)
    steps: List[StepSpec] = [_step("drop_columns", columns=leaks, drop_constant=True, max_missing=0.95)]
    steps.append(_step("normalize_strings"))
    if any_type(ColumnType.NUMERIC, ColumnType.DATETIME, ColumnType.BOOLEAN):
        steps.append(_step("fix_types"))
    if report["dataset"]["duplicate_rows"] or (target and df[target].isna().any()):
        steps.append(_step("deduplicate"))
    outlier_cols = [
        c for c, p in cols.items() if c in feats and (p.get("stats") or {}).get("outliers_iqr_pct", 0) > 0.01
    ]
    if outlier_cols:
        steps.append(_step("outliers", method="quantile", lower_quantile=0.005, upper_quantile=0.995, action="clip"))
    n_rows = report["dataset"]["rows"]
    numeric_missing = any(cols[c]["missing"] for c, t in feats.items() if t == ColumnType.NUMERIC)
    strategy = "median"
    if level == "full" and numeric_missing:
        strategy = "iterative" if n_rows <= 200_000 else "median"
    steps.append(_step("impute", numeric_strategy=strategy, add_indicator=True))

    if level in ("standard", "full"):
        if any_type(ColumnType.DATETIME):
            steps.append(_step("datetime_features"))
        if any_type(ColumnType.TEXT):
            steps.append(_step("text_features", tfidf=level == "full", svd_components=20 if level == "full" else 0))
        skewed = [c for c, p in cols.items() if c in feats and abs((p.get("stats") or {}).get("skew") or 0) > 1.0]
        if skewed:
            steps.append(_step("numeric_transform", method="auto"))
        if level == "full":
            if sum(t == ColumnType.NUMERIC for t in feats.values()) >= 2:
                steps.append(_step("interactions", max_columns=5))
            if any_type(ColumnType.CATEGORICAL) and any_type(ColumnType.NUMERIC):
                steps.append(_step("group_aggregations"))
            if supervised:
                steps.append(_step("feature_generation", max_new=10))
        cats = [c for c, t in feats.items() if t == ColumnType.CATEGORICAL]
        low = [c for c in cats if cols[c]["unique"] <= ONE_HOT_MAX]
        high = [c for c in cats if cols[c]["unique"] > ONE_HOT_MAX]
        if high:
            steps.append(_step("frequency_encode", columns=high))
            if supervised:
                steps.append(_step("target_encode", columns=high, smoothing=10.0, n_folds=5))
            else:
                steps.append(_step("ordinal_encode", columns=high))
        if low:
            steps.append(_step("one_hot", columns=low, max_categories=ONE_HOT_MAX, min_frequency=0.01))
        steps.append(_step("variance_filter"))
        if level == "full":
            steps.append(_step("correlation_filter", threshold=0.97))
    return Recipe(
        name=name,
        description=f"auto-generated ({level}) for {report['dataset']['rows']} rows x {report['dataset']['columns']} columns",
        target=target,
        task=task if target else "none",
        id_columns=[c for c in (id_columns or []) if c in df.columns]
        + [c for c, t in types.items() if t == ColumnType.ID and c != target and c not in (id_columns or [])],
        steps=steps,
    )


def quick_types(df: pd.DataFrame) -> Dict[str, str]:
    return {c: t.value for c, t in infer_types(df).items()}
