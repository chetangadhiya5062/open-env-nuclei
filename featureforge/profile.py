"""Profiling, data-quality warnings, leakage checks and drift.

``profile(df, target=...)`` returns a JSON-friendly dict:

* ``dataset``  rows, columns, duplicates, missing cells, memory, whether a sample was used
* ``columns``  one entry per column: semantic type, missingness, cardinality, top values, numeric statistics
  (mean / std / quantiles / skew / IQR outliers), datetime range, text length
* ``correlations`` strongest numeric pairs (Pearson) and a small matrix for charts
* ``target``   task type, class balance or distribution, per-feature association (normalised mutual information)
* ``warnings`` ``{code, severity, column, message}``; severity is ``info``, ``warning`` or ``critical``
* ``quality_score`` 0-100 summary (see ``_quality_score``)

Large inputs are profiled on a random sample of ``FF_PROFILE_SAMPLE_ROWS`` rows; duplicate counting and missingness
always use the full table because they are cheap.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import get_settings
from .types import ColumnType, infer_types, is_placeholder, normalize_ws, parse_datetime, parse_numeric

logger = logging.getLogger(__name__)

HIGH_MISSING = 0.30
CRITICAL_MISSING = 0.80
HIGH_CORRELATION = 0.95
LEAK_NMI = 0.95
SKEW_LIMIT = 2.0
IMBALANCE = 0.05


def _py(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else round(f, 6)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def infer_task(y: pd.Series) -> str:
    """``classification`` or ``regression`` from the target values."""
    yy = y.dropna()
    if yy.empty:
        return "none"
    if not pd.api.types.is_numeric_dtype(yy):
        num = parse_numeric(yy.astype(str))
        if num.notna().mean() < 0.95:
            return "classification"
        yy = num.dropna()
    nunique = yy.nunique()
    integral = bool(np.all(np.mod(yy.astype(float), 1) == 0))
    if nunique <= 2 or (integral and nunique <= 20):
        return "classification"
    return "regression"


def _discretize(s: pd.Series, bins: int = 20) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > bins:
        return pd.qcut(s.rank(method="first"), q=bins, labels=False, duplicates="drop").astype("float")
    return s.astype(str).where(s.notna(), "__nan__").astype("category").cat.codes.astype(float)


def _entropy(codes: pd.Series) -> float:
    p = codes.value_counts(normalize=True).to_numpy()
    return float(-(p * np.log(p + 1e-12)).sum())


def normalized_mi(x: pd.Series, y: pd.Series) -> float:
    """Mutual information of discretised ``x`` and ``y`` divided by H(y): 1.0 means x determines y."""
    xd, yd = _discretize(x), _discretize(y)
    hy = _entropy(yd)
    if hy <= 1e-12:
        return 0.0
    joint = pd.DataFrame({"x": xd.to_numpy(), "y": yd.to_numpy()})
    hxy = _entropy(joint["x"].astype(str) + "|" + joint["y"].astype(str))
    hx = _entropy(joint["x"])
    return float(max(0.0, min(1.0, (hx + hy - hxy) / hy)))


def _numeric_stats(x: pd.Series) -> Dict[str, Any]:
    x = x.dropna().astype(float)
    if x.empty:
        return {}
    q1, med, q3 = x.quantile([0.25, 0.5, 0.75]).tolist()
    iqr = q3 - q1
    out_mask = (x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)
    hist, edges = np.histogram(x, bins=min(30, max(5, int(np.sqrt(len(x))))))
    return {
        "mean": _py(x.mean()),
        "std": _py(x.std()),
        "min": _py(x.min()),
        "q1": _py(q1),
        "median": _py(med),
        "q3": _py(q3),
        "max": _py(x.max()),
        "skew": _py(x.skew()) if len(x) > 2 else 0.0,
        "kurtosis": _py(x.kurt()) if len(x) > 3 else 0.0,
        "zeros_pct": _py((x == 0).mean()),
        "negatives_pct": _py((x < 0).mean()),
        "outliers_iqr": int(out_mask.sum()),
        "outliers_iqr_pct": _py(out_mask.mean()),
        "histogram": {"counts": hist.tolist(), "edges": [_py(e) for e in edges]},
    }


def _case_variants(s: pd.Series) -> int:
    """Distinct spellings that collapse to the same value after trimming and case-folding."""
    vals = s.dropna().astype(str)
    if vals.empty:
        return 0
    uniq = pd.Series(vals.unique())
    return int(len(uniq) - uniq.map(lambda v: normalize_ws(v).casefold()).nunique())


def _column_profile(s: pd.Series, ctype: ColumnType, n_rows_full: int, missing_full: int) -> Dict[str, Any]:
    nn = s.dropna()
    placeholders = int(s.map(is_placeholder).sum()) if s.dtype == object else 0
    prof: Dict[str, Any] = {
        "name": str(s.name),
        "type": ctype.value,
        "dtype": str(s.dtype),
        "missing": int(missing_full),
        "missing_pct": _py(missing_full / max(1, n_rows_full)),
        "placeholders": placeholders,
        "unique": int(nn.nunique()),
        "unique_pct": _py(nn.nunique() / max(1, len(nn))),
    }
    top = nn.astype(str).value_counts().head(10)
    prof["top_values"] = [{"value": k, "count": int(v)} for k, v in top.items()]
    if ctype == ColumnType.NUMERIC:
        num = parse_numeric(s)
        prof["stored_as_text"] = bool(s.dtype == object)
        prof["unparsable"] = int((num.isna() & s.notna()).sum() - placeholders) if s.dtype == object else 0
        prof["stats"] = _numeric_stats(num)
    elif ctype == ColumnType.DATETIME:
        dt = parse_datetime(s)
        prof["stored_as_text"] = not pd.api.types.is_datetime64_any_dtype(s)
        prof["unparsable"] = int((dt.isna() & s.notna()).sum() - placeholders)
        if dt.notna().any():
            prof["min"] = dt.min().isoformat()
            prof["max"] = dt.max().isoformat()
    elif ctype in (ColumnType.CATEGORICAL, ColumnType.TEXT, ColumnType.ID):
        lengths = nn.astype(str).str.len()
        prof["avg_length"] = _py(lengths.mean()) if len(lengths) else 0
        if s.dtype == object:
            prof["whitespace_issues"] = int(nn.astype(str).map(lambda v: v != normalize_ws(v)).sum())
            prof["case_variants"] = _case_variants(nn)
    return prof


def _warn(out: List[dict], code: str, severity: str, message: str, column: Optional[str] = None) -> None:
    out.append({"code": code, "severity": severity, "column": column, "message": message})


def profile(
    df: pd.DataFrame,
    target: Optional[str] = None,
    sample_rows: Optional[int] = None,
    types: Optional[Dict[str, ColumnType]] = None,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Profile ``df``. See the module docstring for the result layout."""
    if target is not None and target not in df.columns:
        raise KeyError(f"target column {target!r} not found; columns: {list(df.columns)[:30]}")
    settings = get_settings()
    sample_rows = sample_rows or settings.profile_sample_rows
    n_rows, n_cols = df.shape
    sampled = n_rows > sample_rows
    sdf = df.sample(sample_rows, random_state=random_state) if sampled else df
    types = types or infer_types(sdf)

    missing_full = df.isna().sum()
    dup_rows = int(pd.util.hash_pandas_object(df.astype(str), index=False).duplicated().sum()) if n_rows else 0
    warnings_: List[dict] = []
    columns = [_column_profile(sdf[c], types[c], n_rows, int(missing_full[c])) for c in df.columns]

    for col in columns:
        name, t = col["name"], col["type"]
        mp = col["missing_pct"] or 0
        if mp >= CRITICAL_MISSING:
            _warn(warnings_, "high_missing", "critical", f"{mp:.0%} of values are missing", name)
        elif mp >= HIGH_MISSING:
            _warn(warnings_, "high_missing", "warning", f"{mp:.0%} of values are missing", name)
        if col["placeholders"]:
            _warn(
                warnings_,
                "placeholders",
                "warning",
                f"{col['placeholders']} placeholder value(s) such as 'N/A' or '-' mean 'missing'",
                name,
            )
        if t == "constant":
            _warn(warnings_, "constant", "warning", "column has at most one distinct value", name)
        if t == "id" and name != target:
            _warn(warnings_, "id_column", "info", "looks like an identifier; excluded from features", name)
        if t == "numeric" and col.get("stored_as_text"):
            _warn(warnings_, "numeric_as_text", "warning", "numbers are stored as text", name)
        if t == "datetime" and col.get("stored_as_text"):
            _warn(warnings_, "dates_as_text", "info", "dates are stored as text (formats may be mixed)", name)
        if col.get("unparsable", 0) > 0:
            _warn(
                warnings_, "unparsable_values", "warning", f"{col['unparsable']} value(s) cannot be parsed as {t}", name
            )
        if col.get("whitespace_issues"):
            _warn(warnings_, "whitespace", "info", f"{col['whitespace_issues']} value(s) with extra spaces", name)
        if col.get("case_variants"):
            _warn(
                warnings_,
                "inconsistent_categories",
                "warning",
                f"{col['case_variants']} spelling variant(s) differ only by case/spacing",
                name,
            )
        if t == "categorical" and col["unique"] > 50 and (col["unique_pct"] or 0) > 0.5:
            _warn(warnings_, "high_cardinality", "info", f"{col['unique']} distinct values", name)
        st = col.get("stats") or {}
        if st and abs(st.get("skew") or 0) > SKEW_LIMIT:
            _warn(warnings_, "skewed", "info", f"skewness {st['skew']:.2f}; a log / power transform may help", name)
        if st and (st.get("outliers_iqr_pct") or 0) > 0.05:
            _warn(warnings_, "outliers", "warning", f"{st['outliers_iqr_pct']:.1%} values outside 1.5*IQR fences", name)
    if dup_rows:
        _warn(warnings_, "duplicates", "warning", f"{dup_rows} exact duplicate row(s)")
    if n_rows < 50:
        _warn(warnings_, "few_rows", "info", f"only {n_rows} rows; statistics are unreliable")

    correlations = _correlations(sdf, types, target)
    for pair in correlations["top_pairs"]:
        if abs(pair["r"]) >= HIGH_CORRELATION:
            _warn(
                warnings_,
                "high_correlation",
                "info",
                f"|r| = {abs(pair['r']):.3f} with {pair['b']!r}; one of them is redundant",
                pair["a"],
            )

    target_info = _target_section(sdf, types, target, warnings_) if target else None
    result = {
        "dataset": {
            "rows": int(n_rows),
            "columns": int(n_cols),
            "duplicate_rows": dup_rows,
            "duplicate_pct": _py(dup_rows / max(1, n_rows)),
            "missing_cells": int(missing_full.sum()),
            "missing_pct": _py(missing_full.sum() / max(1, n_rows * n_cols)),
            "memory_mb": _py(df.memory_usage(deep=True).sum() / 1e6),
            "sampled": sampled,
            "sample_rows": int(len(sdf)),
            "type_counts": pd.Series([c["type"] for c in columns]).value_counts().to_dict(),
        },
        "columns": columns,
        "correlations": correlations,
        "target": target_info,
        "warnings": warnings_,
    }
    result["quality_score"] = _quality_score(result)
    return result


def _correlations(sdf: pd.DataFrame, types: Dict[str, ColumnType], target: Optional[str]) -> Dict[str, Any]:
    num_cols = [c for c, t in types.items() if t == ColumnType.NUMERIC]
    if len(num_cols) < 2:
        return {"columns": num_cols, "matrix": [], "top_pairs": []}
    num = pd.DataFrame({c: parse_numeric(sdf[c]) for c in num_cols})
    num = num.loc[:, num.std() > 0]
    corr = num.corr()
    cols = list(corr.columns)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr.loc[a, b]
            if not np.isnan(r):
                pairs.append({"a": a, "b": b, "r": round(float(r), 4)})
    pairs.sort(key=lambda p: -abs(p["r"]))
    show = cols[:25]
    return {
        "columns": show,
        "matrix": [[_py(corr.loc[a, b]) for b in show] for a in show],
        "top_pairs": pairs[:20],
    }


def _target_section(
    sdf: pd.DataFrame, types: Dict[str, ColumnType], target: str, warnings_: List[dict]
) -> Dict[str, Any]:
    y = sdf[target]
    task = infer_task(y)
    info: Dict[str, Any] = {"name": target, "task": task, "missing": int(y.isna().sum())}
    if info["missing"]:
        _warn(warnings_, "target_missing", "warning", f"{info['missing']} row(s) without a target value", target)
    if task == "classification":
        dist = y.dropna().astype(str).value_counts(normalize=True)
        info["classes"] = {k: round(float(v), 4) for k, v in dist.head(50).items()}
        if len(dist) >= 2 and dist.min() < IMBALANCE:
            _warn(warnings_, "class_imbalance", "warning", f"smallest class is {dist.min():.1%} of rows", target)
    else:
        info["stats"] = _numeric_stats(parse_numeric(y))

    yv = y if task == "classification" else parse_numeric(y)
    mask = yv.notna()
    assoc = []
    tname = target.casefold()
    for c, t in types.items():
        if c == target or t in (ColumnType.CONSTANT,):
            continue
        x = sdf[c][mask]
        if t == ColumnType.NUMERIC:
            x = parse_numeric(x)
        elif t == ColumnType.DATETIME:
            x = parse_datetime(x).astype("int64").where(parse_datetime(x).notna())
        score = normalized_mi(x, yv[mask])
        assoc.append({"column": c, "nmi": round(score, 4)})
        same = x.astype(str).reset_index(drop=True).equals(yv[mask].astype(str).reset_index(drop=True))
        if same:
            _warn(warnings_, "target_copy", "critical", "column is identical to the target (leak)", c)
        elif score >= LEAK_NMI and t != ColumnType.ID and sdf[c].nunique() < 0.5 * max(1, mask.sum()):
            _warn(
                warnings_,
                "possible_target_leak",
                "critical",
                f"this column alone determines the target (normalised MI {score:.2f}); check it is known at "
                "prediction time",
                c,
            )
        if tname and len(tname) > 2 and tname in c.casefold():
            _warn(warnings_, "target_name_in_column", "warning", f"name contains the target name {target!r}", c)
    assoc.sort(key=lambda a: -a["nmi"])
    info["associations"] = assoc[:50]
    return info


def _quality_score(p: Dict[str, Any]) -> float:
    """0-100: 100 * (1 - 0.5*missing_fraction - 0.3*duplicate_fraction - 0.2*problem_column_fraction).

    A *problem column* has at least one ``warning`` or ``critical`` finding. The weights are a product choice to give
    a single headline number; the individual warnings are the real output.
    """
    ds = p["dataset"]
    problem = {w["column"] for w in p["warnings"] if w["column"] and w["severity"] in ("warning", "critical")}
    frac_problem = len(problem) / max(1, ds["columns"])
    score = 100 * (1 - 0.5 * (ds["missing_pct"] or 0) - 0.3 * (ds["duplicate_pct"] or 0) - 0.2 * frac_problem)
    return round(max(0.0, min(100.0, score)), 1)


# ----------------------------------------------------------------------------------------------------------- drift
def _psi(ref_p: np.ndarray, cur_p: np.ndarray) -> float:
    eps = 1e-4
    ref_p, cur_p = np.clip(ref_p, eps, None), np.clip(cur_p, eps, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, types: Optional[Dict[str, ColumnType]] = None, bins: int = 10
) -> Dict[str, Any]:
    """Per-column drift between a reference (training) table and a new table.

    Numeric columns: PSI over reference deciles + two-sample Kolmogorov-Smirnov test. Categorical / boolean: PSI over
    the reference's top categories (+ "other") and the share of unseen categories. Status by PSI: < 0.1 ``stable``,
    < 0.25 ``moderate``, otherwise ``major`` (the usual credit-scoring thresholds).
    """
    from scipy.stats import ks_2samp

    types = types or infer_types(reference)
    common = [c for c in reference.columns if c in current.columns]
    report: Dict[str, Any] = {
        "missing_columns": [c for c in reference.columns if c not in current.columns],
        "new_columns": [c for c in current.columns if c not in reference.columns],
        "columns": [],
    }
    for c in common:
        t = types.get(c, ColumnType.CATEGORICAL)
        entry: Dict[str, Any] = {
            "column": c,
            "type": t.value,
            "missing_ref": _py(reference[c].isna().mean()),
            "missing_cur": _py(current[c].isna().mean()),
        }
        if t == ColumnType.NUMERIC:
            r, k = parse_numeric(reference[c]).dropna(), parse_numeric(current[c]).dropna()
            if len(r) < 2 or len(k) < 2:
                continue
            edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
            if len(edges) < 3:
                edges = np.array([r.min() - 1e-9, r.max() + 1e-9])
            edges[0], edges[-1] = -np.inf, np.inf
            rp = np.histogram(r, edges)[0] / len(r)
            cp = np.histogram(k, edges)[0] / len(k)
            ks = ks_2samp(r, k)
            entry.update(psi=round(_psi(rp, cp), 4), ks_stat=round(float(ks.statistic), 4), ks_p=_py(ks.pvalue))
        elif t in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN):
            r, k = reference[c].dropna().astype(str), current[c].dropna().astype(str)
            if r.empty or k.empty:
                continue
            top = r.value_counts().head(20).index
            rp = np.append(r.value_counts(normalize=True).reindex(top).to_numpy(), (~r.isin(top)).mean())
            cp = np.append(k.value_counts(normalize=True).reindex(top).fillna(0).to_numpy(), (~k.isin(top)).mean())
            entry.update(psi=round(_psi(rp, cp), 4), unseen_pct=_py((~k.isin(set(r))).mean()))
        else:
            continue
        psi = entry["psi"]
        entry["status"] = "stable" if psi < 0.1 else ("moderate" if psi < 0.25 else "major")
        report["columns"].append(entry)
    report["columns"].sort(key=lambda e: -e["psi"])
    report["drifted"] = [e["column"] for e in report["columns"] if e["status"] == "major"]
    return report
