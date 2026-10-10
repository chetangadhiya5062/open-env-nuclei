"""Observable data-quality measurement (no ground truth needed).

``detect_issues`` counts, per column, how many cells violate the schema. ``quality_score`` turns the
total into a number in [0, 1] that is used for reward shaping:

    Q = 1 - issue_cells / (initial_rows * n_columns)

``issue_cells`` = schema violations + (duplicate rows * n_columns). The denominator is fixed for an
episode (it uses the *initial* shape), so Q is a pure function of the current table: that is what
makes the shaped reward potential-based (see the module docstring of the environment).
"""

import numbers
import re
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..schema import COLUMNS, PLACEHOLDERS, SCHEMA, ColumnSpec

_WS_RE = re.compile(r"\s+")
_ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def normalize_ws(value: str) -> str:
    """Strip and collapse internal whitespace runs to one space."""
    return _WS_RE.sub(" ", value).strip()


def is_iso_date(value: object) -> bool:
    if not isinstance(value, str) or not _ISO_RE.fullmatch(value):
        return False
    try:
        date.fromisoformat(value)  # rejects impossible dates such as 2023-02-30
        return True
    except ValueError:
        return False


def is_number(value: object) -> bool:
    # numbers.Real also covers numpy scalars (np.int64, np.float64)
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def column_issues(series: pd.Series, spec: ColumnSpec) -> Dict[str, int]:
    """Count schema violations in one column. Only non-zero counts are returned."""
    issues: Dict[str, int] = {}
    missing = int(series.isna().sum())
    if missing:
        issues["missing"] = missing

    if (
        spec.kind in ("int", "float", "id")
        and pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
    ):
        # fast vectorised path for genuinely numeric columns (same rules as the loop below)
        arr = series.dropna().to_numpy(dtype=float)
        if spec.kind in ("int", "id"):
            integral = arr == np.floor(arr)
            wrong = int((~integral).sum())
            arr = arr[integral]
        else:
            wrong = 0
        out_of_range = int(((arr < spec.valid_range[0]) | (arr > spec.valid_range[1])).sum()) if spec.valid_range else 0
        if wrong:
            issues["wrong_type"] = wrong
        if out_of_range:
            issues["out_of_range"] = out_of_range
        return issues

    values = series.dropna().tolist()

    if spec.kind in ("int", "float", "id"):
        wrong = 0
        out_of_range = 0
        for v in values:
            if not is_number(v):
                wrong += 1
            elif spec.kind in ("int", "id") and float(v) != int(v):
                wrong += 1
            elif spec.valid_range and not (spec.valid_range[0] <= v <= spec.valid_range[1]):
                out_of_range += 1
        if wrong:
            issues["wrong_type"] = wrong
        if out_of_range:
            issues["out_of_range"] = out_of_range
    elif spec.kind == "date":
        bad = sum(1 for v in values if not is_iso_date(v))
        if bad:
            issues["bad_date_format"] = bad
    else:  # text / cat
        whitespace = sum(1 for v in values if isinstance(v, str) and v != normalize_ws(v))
        if whitespace:
            issues["whitespace"] = whitespace
        if spec.allowed is not None:
            allowed = set(spec.allowed)
            bad = sum(1 for v in values if v not in allowed)
            if bad:
                issues["not_allowed_value"] = bad
        wrong = sum(1 for v in values if not isinstance(v, str))
        if wrong:
            issues["wrong_type"] = wrong
    return issues


def detect_issues(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    return {c.name: column_issues(df[c.name], c) for c in SCHEMA if c.name in df.columns}


def duplicate_count(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def issue_cells(
    df: pd.DataFrame, issues: Optional[Dict[str, Dict[str, int]]] = None, duplicates: Optional[int] = None
) -> int:
    """Total violating cells (+ duplicate rows * n_columns). Pass precomputed ``issues``/``duplicates`` to avoid rework."""
    per_column = detect_issues(df) if issues is None else issues
    dups = duplicate_count(df) if duplicates is None else duplicates
    return sum(sum(v.values()) for v in per_column.values()) + dups * len(COLUMNS)


def quality_score(
    df: pd.DataFrame,
    initial_cells: int,
    issues: Optional[Dict[str, Dict[str, int]]] = None,
    duplicates: Optional[int] = None,
) -> float:
    """Observable quality in [0, 1]; 1.0 means no detectable schema violation or duplicate."""
    return max(0.0, min(1.0, 1.0 - issue_cells(df, issues, duplicates) / max(1, initial_cells)))


def is_placeholder(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() in PLACEHOLDERS
