"""Pure pandas implementations of the cleaning actions.

Every function takes a DataFrame and a typed action and returns ``(new_df, message)``.
They never mutate their input and raise ``ActionError`` (with an agent-readable message) when the
action is not applicable. The environment turns that error into a small penalty, not a crash.
"""

from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .. import actions as A
from ..schema import ISO_DATE_FORMAT, PLACEHOLDERS, SCHEMA_BY_NAME, ColumnSpec
from .datagen import DATE_FORMATS
from .quality import is_number, normalize_ws

OUTLIER_IQR_K = 3.0  # only used for columns without a declared valid range


class ActionError(ValueError):
    """The action could not be applied; the message is shown to the agent."""


def _spec(df: pd.DataFrame, column: str) -> ColumnSpec:
    if column not in df.columns:
        raise ActionError(f"unknown column {column!r}; columns: {list(df.columns)}")
    return SCHEMA_BY_NAME[column]


def _require_clean_numeric(df: pd.DataFrame, column: str, spec: ColumnSpec, what: str) -> None:
    if not spec.is_numeric:
        raise ActionError(f"{what} needs a numeric column, but {column!r} is {spec.kind}")
    if any(isinstance(v, str) for v in df[column].dropna().tolist()):
        raise ActionError(f"column {column!r} still contains text values; run cast_type to {spec.kind} first")


def _as_expected_number(x: float, spec: ColumnSpec):
    return int(round(x)) if spec.kind == "int" else round(float(x), 2)


def fill_missing(df: pd.DataFrame, a: A.FillMissing) -> Tuple[pd.DataFrame, str]:
    spec = _spec(df, a.column_name)
    col = df[a.column_name]
    n_missing = int(col.isna().sum())
    if n_missing == 0:
        return df, f"column {a.column_name!r} has no missing values; nothing changed"

    if spec.is_numeric:
        _require_clean_numeric(df, a.column_name, spec, f"fill_missing/{a.strategy}")
    if a.strategy in ("mean", "median"):
        if not spec.is_numeric:
            raise ActionError(
                f"strategy {a.strategy!r} needs a numeric column; use 'mode' or 'constant' for {a.column_name!r}"
            )
        stat = col.astype(float).mean() if a.strategy == "mean" else col.astype(float).median()
        if pd.isna(stat):
            raise ActionError(f"column {a.column_name!r} has no values to compute the {a.strategy} from")
        fill = _as_expected_number(stat, spec)
    elif a.strategy == "mode":
        modes = col.dropna().mode()
        if modes.empty:
            raise ActionError(f"column {a.column_name!r} has no values to compute the mode from")
        fill = modes.iloc[0]
        if spec.is_numeric:
            fill = _as_expected_number(fill, spec)
    else:  # constant
        if spec.is_numeric:
            try:
                fill = _as_expected_number(float(a.value), spec)
            except (TypeError, ValueError):
                raise ActionError(f"constant {a.value!r} is not a number but {a.column_name!r} is numeric") from None
        elif spec.kind == "date":
            if not _parses_iso(a.value):
                raise ActionError(f"constant {a.value!r} is not an ISO date (YYYY-MM-DD)")
            fill = a.value
        else:
            fill = a.value
            if spec.allowed is not None and fill not in spec.allowed:
                raise ActionError(
                    f"constant {a.value!r} is not an allowed value for {a.column_name!r}; allowed: {list(spec.allowed)}"
                )
    out = df.copy()
    out[a.column_name] = col.fillna(fill) if not pd.api.types.is_object_dtype(col) else col.where(col.notna(), fill)
    return out, f"filled {n_missing} missing cell(s) in {a.column_name!r} using {a.strategy}" + (
        f" = {fill!r}" if a.strategy != "constant" else ""
    )


def _parses_iso(value: Optional[str]) -> bool:
    try:
        datetime.strptime(str(value), ISO_DATE_FORMAT)
        return True
    except ValueError:
        return False


def drop_duplicates(df: pd.DataFrame, a: A.DropDuplicates) -> Tuple[pd.DataFrame, str]:
    out = df.drop_duplicates().reset_index(drop=True)
    removed = len(df) - len(out)
    return out, (f"removed {removed} duplicate row(s)" if removed else "no exact duplicate rows; nothing changed")


def _canonical_for(raw: str, spec: ColumnSpec, counts: dict) -> str:
    key = normalize_ws(raw).casefold()
    if spec.allowed is not None:
        for allowed in spec.allowed:
            if allowed.casefold() == key:
                return allowed
    variants = {v: c for v, c in counts.items() if normalize_ws(v).casefold() == key}
    best = max(variants.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return normalize_ws(best)


def standardize_categories(df: pd.DataFrame, a: A.StandardizeCategories) -> Tuple[pd.DataFrame, str]:
    spec = _spec(df, a.column_name)
    if spec.is_numeric or spec.kind in ("date", "id"):
        raise ActionError(f"standardize_categories needs a text/category column; {a.column_name!r} is {spec.kind}")
    col = df[a.column_name]
    out = df.copy()
    if a.mapping is None:
        counts = col.dropna().astype(str).value_counts().to_dict()
        new = col.map(lambda v: v if pd.isna(v) else _canonical_for(str(v), spec, counts))
    else:
        if spec.allowed is not None:
            bad = sorted({v for v in a.mapping.values() if v not in spec.allowed})
            if bad:
                raise ActionError(
                    f"mapping targets {bad} are not allowed values for {a.column_name!r}; allowed: {list(spec.allowed)}"
                )
        lookup = {normalize_ws(k).casefold(): v for k, v in a.mapping.items()}
        new = col.map(lambda v: v if pd.isna(v) else lookup.get(normalize_ws(str(v)).casefold(), v))
    changed = int((col.fillna("\0") != new.fillna("\0")).sum())
    out[a.column_name] = new
    return (
        out,
        f"standardized {changed} cell(s) in {a.column_name!r}" if changed else f"no cells in {a.column_name!r} changed",
    )


def strip_whitespace(df: pd.DataFrame, a: A.StripWhitespace) -> Tuple[pd.DataFrame, str]:
    if a.column_name is not None:
        _spec(df, a.column_name)
        columns = [a.column_name]
    else:
        columns = [c for c in df.columns if not SCHEMA_BY_NAME[c].is_numeric]
    out = df.copy()
    changed = 0
    for c in columns:
        new = out[c].map(lambda v: normalize_ws(v) if isinstance(v, str) else v)
        changed += int((out[c].fillna("\0") != new.fillna("\0")).sum())
        out[c] = new
    return (
        out,
        f"stripped/collapsed whitespace in {changed} cell(s)"
        if changed
        else "no whitespace problems found; nothing changed",
    )


def _to_number(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if is_number(v):
        return float(v)
    s = str(v).strip()
    if s.lower() in PLACEHOLDERS:
        return np.nan
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return np.nan


def cast_type(df: pd.DataFrame, a: A.CastType) -> Tuple[pd.DataFrame, str]:
    spec = _spec(df, a.column_name)
    col = df[a.column_name]
    out = df.copy()
    if a.target_type == "str":
        out[a.column_name] = col.map(lambda v: v if pd.isna(v) else str(v))
        return out, f"converted {a.column_name!r} to text"
    if not spec.is_numeric and spec.kind != "id":
        raise ActionError(f"cannot cast {a.column_name!r} ({spec.kind}) to {a.target_type}")
    nums = col.map(_to_number)
    newly_missing = int((nums.isna() & col.notna()).sum())
    if a.target_type == "int":
        nums = nums.round()
    out[a.column_name] = nums.astype(float)
    msg = f"converted {a.column_name!r} to {a.target_type}"
    if newly_missing:
        msg += f"; {newly_missing} unparsable/placeholder value(s) became missing"
    return out, msg


def _parse_date(v) -> Optional[str]:
    if not isinstance(v, str):
        return None
    s = v.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime(ISO_DATE_FORMAT)
        except ValueError:
            continue
    return None


def fix_dates(df: pd.DataFrame, a: A.FixDates) -> Tuple[pd.DataFrame, str]:
    spec = _spec(df, a.column_name)
    if spec.kind != "date":
        raise ActionError(f"fix_dates needs a date column; {a.column_name!r} is {spec.kind}")
    col = df[a.column_name]
    parsed = col.map(lambda v: v if pd.isna(v) else _parse_date(v))
    unparsable = int((parsed.isna() & col.notna()).sum())
    changed = int((col.fillna("\0") != parsed.fillna("\0")).sum())
    out = df.copy()
    out[a.column_name] = parsed
    msg = (
        f"normalized {changed} date cell(s) to YYYY-MM-DD"
        if changed
        else "all dates already in YYYY-MM-DD; nothing changed"
    )
    if unparsable:
        msg += f"; {unparsable} unparsable date(s) became missing"
    return out, msg


def clip_outliers(df: pd.DataFrame, a: A.ClipOutliers) -> Tuple[pd.DataFrame, str]:
    spec = _spec(df, a.column_name)
    _require_clean_numeric(df, a.column_name, spec, "clip_outliers")
    col = df[a.column_name].astype(float)
    if spec.valid_range:
        lo, hi = spec.valid_range  # clip to the schema's valid range
    else:  # no declared range: robust IQR fences
        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        lo, hi = q1 - OUTLIER_IQR_K * (q3 - q1), q3 + OUTLIER_IQR_K * (q3 - q1)
    clipped = col.clip(lower=lo, upper=hi)
    changed = int(((clipped != col) & col.notna()).sum())
    out = df.copy()
    if spec.kind == "int":
        clipped = clipped.round()
    out[a.column_name] = clipped
    return out, (
        f"clipped {changed} value(s) in {a.column_name!r} to [{lo:.4g}, {hi:.4g}]"
        if changed
        else f"no outliers in {a.column_name!r}; nothing changed"
    )


def drop_rows_with_missing(df: pd.DataFrame, a: A.DropRowsWithMissing) -> Tuple[pd.DataFrame, str]:
    if a.column_name is not None:
        _spec(df, a.column_name)
    out = df.dropna(subset=[a.column_name] if a.column_name else None).reset_index(drop=True)
    removed = len(df) - len(out)
    return out, (
        f"dropped {removed} row(s) with missing values" if removed else "no rows with missing values; nothing changed"
    )


def normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Turn float columns that should be integers into int64 once they contain no nulls."""
    out = df
    for name, spec in SCHEMA_BY_NAME.items():
        if name in out.columns and spec.kind in ("int", "id") and pd.api.types.is_float_dtype(out[name]):
            if out[name].notna().all() and (out[name] == out[name].round()).all():
                out = out.assign(**{name: out[name].astype("int64")})
    return out


HANDLERS = {
    A.FillMissing: fill_missing,
    A.DropDuplicates: drop_duplicates,
    A.StandardizeCategories: standardize_categories,
    A.StripWhitespace: strip_whitespace,
    A.CastType: cast_type,
    A.FixDates: fix_dates,
    A.ClipOutliers: clip_outliers,
    A.DropRowsWithMissing: drop_rows_with_missing,
}


def apply_action(df: pd.DataFrame, action) -> Tuple[pd.DataFrame, str]:
    new_df, message = HANDLERS[type(action)](df, action)
    return normalize_dtypes(new_df), message
