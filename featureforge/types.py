"""Semantic column-type inference and robust value parsers.

``infer_types(df)`` assigns every column one of :class:`ColumnType`. It looks at values, not only the pandas dtype,
so ``"1,250"`` / ``"$12.50"`` / ``"45%"`` columns are *numeric* (stored as text), ``"2024-01-05"`` / ``"5 Jan 2024"``
columns are *datetime*, ``yes/no`` columns are *boolean*, and long free-form strings are *text*.

The parsers (``parse_numeric``, ``parse_datetime``, ``parse_bool``) are what the ``fix_types`` cleaning step uses.
"""

import re
import warnings
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd

# String tokens that mean "no value" although they are not real nulls (a superset of the environment's list).
PLACEHOLDERS = frozenset(
    {
        "",
        "n/a",
        "na",
        "nan",
        "null",
        "none",
        "nil",
        "-",
        "--",
        "?",
        "missing",
        "unknown",
        "undefined",
        "#n/a",
        "(blank)",
    }
)
TRUE_TOKENS = frozenset({"true", "t", "yes", "y", "1", "on"})
FALSE_TOKENS = frozenset({"false", "f", "no", "n", "0", "off"})

_ID_NAME = re.compile(r"(^id$|_id$|^id_|uuid|guid|(^|_)key$|^index$|^unnamed: ?0$)", re.I)
_WS = re.compile(r"\s+")
_NUM_CLEAN = re.compile(r"[\s,$€£₹%]")
_DATE_HINT = re.compile(
    r"(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4})|(?:\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{2,4})|(?:[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{2,4})"
)
SAMPLE = 10_000


class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    ID = "id"
    BOOLEAN = "boolean"
    CONSTANT = "constant"


def normalize_ws(value: str) -> str:
    return _WS.sub(" ", value).strip()


def is_placeholder(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() in PLACEHOLDERS


def _null_placeholders(s: pd.Series) -> pd.Series:
    if s.dtype != object and not pd.api.types.is_string_dtype(s):
        return s
    return s.map(lambda v: np.nan if is_placeholder(v) else v)


def parse_numeric(s: pd.Series) -> pd.Series:
    """Parse numbers stored as text (``"1,250"``, ``"$12.5"``, ``"45%"`` -> 45.0, ``"(3)"`` -> -3). Others -> NaN."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)

    def conv(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
            return float(v)
        t = str(v).strip()
        if t.lower() in PLACEHOLDERS:
            return np.nan
        neg = t.startswith("(") and t.endswith(")")
        t = _NUM_CLEAN.sub("", t.strip("()"))
        try:
            x = float(t)
        except ValueError:
            return np.nan
        return -x if neg else x

    return s.map(conv).astype(float)


def parse_datetime(s: pd.Series) -> pd.Series:
    """Parse mixed date formats to ``datetime64[ns]``; unparsable values become NaT."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.tz_localize(None) if getattr(s.dt, "tz", None) is not None else s
    cleaned = _null_placeholders(s.astype(object))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pd.to_datetime(
            cleaned.map(lambda v: v if pd.isna(v) else str(v).strip()), errors="coerce", format="mixed"
        )
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert(None)
    return out


def parse_bool(s: pd.Series) -> pd.Series:
    """Map true/false tokens to 1.0 / 0.0 (float so NaN survives); anything else -> NaN."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)

    def conv(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        t = str(v).strip().lower()
        if t in ("1.0",):
            t = "1"
        if t in ("0.0",):
            t = "0"
        if t in TRUE_TOKENS:
            return 1.0
        if t in FALSE_TOKENS:
            return 0.0
        return np.nan

    return s.map(conv).astype(float)


def _sample(s: pd.Series, n: int = SAMPLE) -> pd.Series:
    nn = s.dropna()
    return nn.sample(n, random_state=0) if len(nn) > n else nn


def infer_column(s: pd.Series, name: str = "") -> ColumnType:
    """Infer the semantic type of one column."""
    name = name or str(s.name or "")
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        s = _null_placeholders(s.astype(object))
    vals = _sample(s)
    n = len(vals)
    if n == 0:
        return ColumnType.CONSTANT
    nunique = vals.nunique(dropna=True)
    if nunique <= 1:
        return ColumnType.CONSTANT
    total_unique_ratio = s.nunique(dropna=True) / max(1, s.notna().sum())

    if pd.api.types.is_bool_dtype(s):
        return ColumnType.BOOLEAN
    if pd.api.types.is_datetime64_any_dtype(s):
        return ColumnType.DATETIME
    if pd.api.types.is_numeric_dtype(s):
        if nunique == 2 and set(np.unique(vals.astype(float))) <= {0.0, 1.0}:
            return ColumnType.BOOLEAN
        is_int = bool(np.all(np.mod(vals.astype(float), 1) == 0))
        if is_int and _ID_NAME.search(name) and total_unique_ratio > 0.9:
            return ColumnType.ID
        if is_int and total_unique_ratio > 0.999 and len(s) > 50 and _looks_sequential(vals):
            return ColumnType.ID
        return ColumnType.NUMERIC

    strs = vals.astype(str).str.strip()
    lower = strs.str.lower()
    if lower.nunique() <= 6 and lower.isin(TRUE_TOKENS | FALSE_TOKENS).mean() >= 0.98:
        if parse_bool(strs).dropna().nunique() == 2:
            return ColumnType.BOOLEAN
    num_ok = parse_numeric(strs).notna().mean()
    if num_ok >= 0.9:
        return ColumnType.ID if _ID_NAME.search(name) and total_unique_ratio > 0.9 else ColumnType.NUMERIC
    if strs.str.contains(_DATE_HINT).mean() >= 0.8 and parse_datetime(strs).notna().mean() >= 0.9:
        return ColumnType.DATETIME
    if _ID_NAME.search(name) and total_unique_ratio > 0.9:
        return ColumnType.ID
    avg_words = strs.str.split().str.len().mean()
    avg_len = strs.str.len().mean()
    if (
        total_unique_ratio > 0.95
        and len(s) > 50
        and avg_words < 2
        and avg_len < 40
        and not strs.str.contains(" ").any()
    ):
        return ColumnType.ID
    if avg_words >= 4 or (avg_len > 40 and total_unique_ratio > 0.5):
        return ColumnType.TEXT
    return ColumnType.CATEGORICAL


def _looks_sequential(vals: pd.Series) -> bool:
    arr = np.sort(vals.astype(float).to_numpy())
    if len(arr) < 3:
        return False
    diffs = np.diff(arr)
    return bool(np.mean(diffs == 1) > 0.9)


def infer_types(df: pd.DataFrame, overrides: Optional[Dict[str, str]] = None) -> Dict[str, ColumnType]:
    """Semantic type per column. ``overrides`` (``{"col": "categorical"}``) win over inference."""
    overrides = overrides or {}
    out: Dict[str, ColumnType] = {}
    for c in df.columns:
        if c in overrides:
            out[c] = ColumnType(overrides[c])
        else:
            out[c] = infer_column(df[c], c)
    return out
