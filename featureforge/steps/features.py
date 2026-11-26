"""Feature-construction steps: datetime parts, numeric transforms, scaling, interactions, group aggregations, text."""

from itertools import combinations
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ..types import ColumnType
from .base import FitContext, Step, register, subsample

EPS = 1e-9
DATE_PARTS = (
    "year",
    "month",
    "day",
    "dayofweek",
    "dayofyear",
    "quarter",
    "hour",
    "minute",
    "is_weekend",
    "is_month_start",
    "is_month_end",
)
CYCLES = {"month": 12, "dayofweek": 7, "hour": 24, "dayofyear": 366, "minute": 60, "day": 31}


class _P(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _numeric_features(df: pd.DataFrame, ctx: FitContext, requested: Optional[List[str]] = None) -> List[str]:
    cols = ctx.columns_of(df, (ColumnType.NUMERIC,), requested)
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def _rank_by_target(df: pd.DataFrame, cols: List[str], ctx: FitContext) -> List[str]:
    """Order columns by mutual information with the target (or by variance-normalised spread without one)."""
    y = ctx.y(df)
    if not cols:
        return cols
    if y is None or ctx.task not in ("classification", "regression"):
        cv = {c: float(df[c].std() / (abs(df[c].mean()) + EPS)) for c in cols}
        return sorted(cols, key=lambda c: -cv[c])
    from .selection import mi_scores

    scores = mi_scores(df[cols], y, ctx)
    return sorted(cols, key=lambda c: -scores.get(c, 0.0))


# ------------------------------------------------------------------------------------------ datetime parts
class DatetimeParams(_P):
    columns: Optional[List[str]] = None
    parts: List[Literal[DATE_PARTS]] = Field(  # type: ignore[valid-type]
        default_factory=lambda: list(DATE_PARTS)
    )
    cyclical: bool = True
    elapsed: bool = Field(True, description="days since the earliest training date")
    drop_original: bool = True


@register
class DatetimeFeatures(Step):
    name = "datetime_features"
    kind = "feature"
    description = "Calendar parts, weekend/month-boundary flags, sin/cos cyclical encodings and elapsed days."
    Params = DatetimeParams

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "DatetimeFeatures":
        cols = [
            c
            for c in ctx.columns_of(df, (ColumnType.DATETIME,), self.params.columns)
            if pd.api.types.is_datetime64_any_dtype(df[c])
        ]
        self.columns_ = cols
        self.parts_: Dict[str, List[str]] = {}
        self.ref_: Dict[str, pd.Timestamp] = {}
        for c in cols:
            s = df[c].dropna()
            parts = []
            for part in self.params.parts:
                vals = self._part(s, part)
                if vals.nunique() > 1:  # a part that never varies (e.g. hour of a date-only column) is useless
                    parts.append(part)
            self.parts_[c] = parts
            self.ref_[c] = s.min() if len(s) else pd.Timestamp("1970-01-01")
            for name in self._names(c):
                ctx.types[name] = ColumnType.NUMERIC
            if self.params.drop_original:
                ctx.types.pop(c, None)
            self.note(f"{c}: {len(self._names(c))} feature(s)")
        return self

    @staticmethod
    def _part(s: pd.Series, part: str) -> pd.Series:
        if part in ("is_weekend",):
            return (s.dt.dayofweek >= 5).astype(float)
        if part == "is_month_start":
            return s.dt.is_month_start.astype(float)
        if part == "is_month_end":
            return s.dt.is_month_end.astype(float)
        return getattr(s.dt, part).astype(float)

    def _names(self, c: str) -> List[str]:
        names = [f"{c}__{p}" for p in self.parts_[c]]
        if self.params.cyclical:
            for p in self.parts_[c]:
                if p in CYCLES:
                    names += [f"{c}__{p}_sin", f"{c}__{p}_cos"]
        if self.params.elapsed:
            names.append(f"{c}__days_since_start")
        return names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.columns_:
            if c not in out.columns:
                continue
            s = pd.to_datetime(out[c], errors="coerce")
            new: Dict[str, pd.Series] = {}
            for p in self.parts_[c]:
                v = self._part(s, p).where(s.notna())
                new[f"{c}__{p}"] = v
                if self.params.cyclical and p in CYCLES:
                    ang = 2 * np.pi * v / CYCLES[p]
                    new[f"{c}__{p}_sin"] = np.sin(ang)
                    new[f"{c}__{p}_cos"] = np.cos(ang)
            if self.params.elapsed:
                new[f"{c}__days_since_start"] = (s - self.ref_[c]).dt.total_seconds() / 86400.0
            out = pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)
            if self.params.drop_original:
                out = out.drop(columns=[c])
        return out


# -------------------------------------------------------------------------------------- numeric transforms
class NumericTransformParams(_P):
    columns: Optional[List[str]] = None
    method: Literal["auto", "log1p", "yeo_johnson", "quantile_bin"] = "auto"
    skew_threshold: float = Field(1.0, ge=0)
    n_bins: int = Field(10, ge=2, le=100)
    keep_original: bool = False


@register
class NumericTransform(Step):
    name = "numeric_transform"
    kind = "feature"
    description = "Reduce skew with log1p or Yeo-Johnson, or discretise into quantile bins."
    Params = NumericTransformParams

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "NumericTransform":
        from sklearn.preprocessing import KBinsDiscretizer, PowerTransformer

        p = self.params
        cols = [c for c in _numeric_features(df, ctx, p.columns) if df[c].nunique() > 2 and "__" not in c]
        if p.columns is not None:
            cols = [c for c in _numeric_features(df, ctx, p.columns) if df[c].nunique() > 2]
        self.plan_: Dict[str, str] = {}
        self.models_: Dict[str, object] = {}
        for c in cols:
            x = df[c].astype(float).dropna()
            if x.empty:
                continue
            method = p.method
            if method == "auto":
                if abs(x.skew()) < p.skew_threshold:
                    continue
                method = "log1p" if x.min() >= 0 else "yeo_johnson"
            if method == "log1p" and x.min() < 0:
                self.note(f"{c}: has negative values, using yeo_johnson instead of log1p")
                method = "yeo_johnson"
            if method == "yeo_johnson":
                self.models_[c] = PowerTransformer(method="yeo-johnson", standardize=False).fit(x.to_frame())
            elif method == "quantile_bin":
                n_bins = int(min(p.n_bins, x.nunique()))
                self.models_[c] = KBinsDiscretizer(
                    n_bins=n_bins, encode="ordinal", strategy="quantile", quantile_method="averaged_inverted_cdf"
                ).fit(x.to_frame())
            self.plan_[c] = method
            if p.keep_original:
                ctx.types[self._out(c)] = ColumnType.NUMERIC
        if self.plan_:
            self.note(", ".join(f"{c}: {m}" for c, m in self.plan_.items()))
        return self

    def _out(self, c: str) -> str:
        return f"{c}__{self.plan_[c]}" if self.params.keep_original else c

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c, method in self.plan_.items():
            if c not in out.columns:
                continue
            x = out[c].astype(float)
            if method == "log1p":
                y = np.log1p(x.clip(lower=0))
            else:
                y = pd.Series(np.nan, index=x.index)
                ok = x.notna()
                if ok.any():
                    y[ok] = self.models_[c].transform(x[ok].to_frame()).ravel()
            out[self._out(c)] = y
        return out


# ----------------------------------------------------------------------------------------------- scaling
class ScaleParams(_P):
    columns: Optional[List[str]] = None
    method: Literal["standard", "robust", "minmax"] = "standard"


@register
class Scale(Step):
    name = "scale"
    kind = "feature"
    description = "Standardise, robust-scale or min-max scale numeric features (flags and indicators are left alone)."
    Params = ScaleParams

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "Scale":
        cols = [c for c in _numeric_features(df, ctx, self.params.columns) if df[c].nunique() > 2]
        self.stats_: Dict[str, tuple] = {}
        for c in cols:
            x = df[c].astype(float)
            if self.params.method == "standard":
                center, spread = x.mean(), x.std()
            elif self.params.method == "robust":
                center, spread = x.median(), x.quantile(0.75) - x.quantile(0.25)
            else:
                center, spread = x.min(), x.max() - x.min()
            if pd.isna(spread) or spread < EPS:
                continue
            self.stats_[c] = (float(center), float(spread))
        self.note(f"{self.params.method} scaling of {len(self.stats_)} column(s)")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c, (center, spread) in self.stats_.items():
            if c in out.columns:
                out[c] = (out[c].astype(float) - center) / spread
        return out


# ------------------------------------------------------------------------------------------- interactions
class InteractionsParams(_P):
    columns: Optional[List[str]] = None
    max_columns: int = Field(6, ge=2, le=30, description="use the top-N numeric columns (by target MI)")
    ops: List[Literal["ratio", "product", "difference", "sum"]] = Field(
        default_factory=lambda: ["ratio", "product", "difference"]
    )


@register
class Interactions(Step):
    name = "interactions"
    kind = "feature"
    description = "Pairwise ratios, products, differences and sums of the most informative numeric columns."
    Params = InteractionsParams

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "Interactions":
        cols = [c for c in _numeric_features(df, ctx, self.params.columns) if df[c].nunique() > 2]
        cols = _rank_by_target(subsample(df, ctx.max_fit_rows, ctx.random_state), cols, ctx)[: self.params.max_columns]
        self.pairs_ = list(combinations(cols, 2))
        for name in self._names():
            ctx.types[name] = ColumnType.NUMERIC
        self.note(f"{len(self._names())} interaction feature(s) from {len(cols)} column(s)")
        return self

    def _names(self) -> List[str]:
        return [_op_name(op, a, b) for a, b in self.pairs_ for op in self.params.ops]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new = {}
        for a, b in self.pairs_:
            if a not in df.columns or b not in df.columns:
                continue
            for op in self.params.ops:
                new[_op_name(op, a, b)] = apply_op(op, df[a].astype(float), df[b].astype(float))
        return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1) if new else df.copy()


def _op_name(op: str, a: str, b: str) -> str:
    sym = {"ratio": "div", "product": "mul", "difference": "sub", "sum": "add"}[op]
    return f"{a}__{sym}__{b}"


def apply_op(op: str, a: pd.Series, b: pd.Series) -> pd.Series:
    if op == "ratio":
        r = a / b.where(b.abs() > EPS)
        return r.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if op == "product":
        return a * b
    if op == "difference":
        return a - b
    return a + b


# ------------------------------------------------------------------------------------- group aggregations
class GroupAggParams(_P):
    group_columns: Optional[List[str]] = None
    value_columns: Optional[List[str]] = None
    max_value_columns: int = Field(5, ge=1, le=30)
    max_group_cardinality: int = Field(50, ge=2)
    aggs: List[Literal["mean", "std", "median", "min", "max", "count"]] = Field(
        default_factory=lambda: ["mean", "std", "count"]
    )
    add_deviation: bool = True
    min_group_size: int = Field(5, ge=1, description="smaller groups fall back to the global statistic")


@register
class GroupAggregations(Step):
    name = "group_aggregations"
    kind = "feature"
    description = "Deep-Feature-Synthesis-style group statistics: mean/std/count of numeric columns per category, and deviation from the group mean."
    Params = GroupAggParams

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "GroupAggregations":
        p = self.params
        groups = p.group_columns or [
            c for c in ctx.columns_of(df, (ColumnType.CATEGORICAL,)) if 2 <= df[c].nunique() <= p.max_group_cardinality
        ]
        values = p.value_columns or _rank_by_target(
            subsample(df, ctx.max_fit_rows, ctx.random_state),
            [c for c in _numeric_features(df, ctx) if "__" not in c and df[c].nunique() > 2],
            ctx,
        )
        values = values[: p.max_value_columns]
        self.tables_: Dict[tuple, pd.DataFrame] = {}
        self.global_: Dict[tuple, Dict[str, float]] = {}
        for g in groups:
            key = df[g].astype(str).where(df[g].notna(), "__nan__")
            for v in values:
                x = df[v].astype(float)
                stats = x.groupby(key).agg(list(dict.fromkeys(p.aggs + ["count"])))
                small = stats["count"] < p.min_group_size
                glob = {a: float(getattr(x, a)()) if a != "count" else float(x.count()) for a in p.aggs}
                for a in p.aggs:
                    if a != "count":
                        stats.loc[small, a] = glob[a]
                self.tables_[(g, v)] = stats[p.aggs]
                self.global_[(g, v)] = glob
        for name in self._names():
            ctx.types[name] = ColumnType.NUMERIC
        self.note(f"{len(self._names())} group feature(s) over {len(groups)} group column(s)")
        return self

    def _names(self) -> List[str]:
        names = []
        for g, v in self.tables_:
            names += [f"{v}__{a}_by_{g}" for a in self.params.aggs]
            if self.params.add_deviation and "mean" in self.params.aggs:
                names.append(f"{v}__dev_from_mean_by_{g}")
        return names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new = {}
        for (g, v), table in self.tables_.items():
            if g not in df.columns or v not in df.columns:
                continue
            key = df[g].astype(str).where(df[g].notna(), "__nan__")
            for a in self.params.aggs:
                new[f"{v}__{a}_by_{g}"] = key.map(table[a]).astype(float).fillna(self.global_[(g, v)][a])
            if self.params.add_deviation and "mean" in self.params.aggs:
                new[f"{v}__dev_from_mean_by_{g}"] = df[v].astype(float) - new[f"{v}__mean_by_{g}"]
        return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1) if new else df.copy()


# -------------------------------------------------------------------------------------------------- text
class TextParams(_P):
    columns: Optional[List[str]] = None
    stats: bool = True
    tfidf: bool = False
    max_features: int = Field(200, ge=1, le=20_000)
    svd_components: int = Field(0, ge=0, le=500, description="0 = keep raw TF-IDF columns")
    drop_original: bool = True


@register
class TextFeatures(Step):
    name = "text_features"
    kind = "feature"
    description = "Text statistics (length, words, digits, uppercase, punctuation) and optional TF-IDF (+ SVD)."
    Params = TextParams

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "TextFeatures":
        p = self.params
        self.columns_ = ctx.columns_of(df, (ColumnType.TEXT,), p.columns)
        self.vec_: Dict[str, object] = {}
        self.svd_: Dict[str, object] = {}
        for c in self.columns_:
            if p.tfidf:
                from sklearn.feature_extraction.text import TfidfVectorizer

                text = subsample(df[[c]], ctx.max_fit_rows, ctx.random_state)[c].fillna("").astype(str)
                vec = TfidfVectorizer(max_features=p.max_features, min_df=2, sublinear_tf=True)
                try:
                    m = vec.fit_transform(text)
                except ValueError:  # empty vocabulary
                    continue
                self.vec_[c] = vec
                if p.svd_components and m.shape[1] > p.svd_components:
                    from sklearn.decomposition import TruncatedSVD

                    self.svd_[c] = TruncatedSVD(p.svd_components, random_state=ctx.random_state).fit(m)
            for name in self._names(c):
                ctx.types[name] = ColumnType.NUMERIC
            if p.drop_original:
                ctx.types.pop(c, None)
            self.note(f"{c}: {len(self._names(c))} feature(s)")
        return self

    def _names(self, c: str) -> List[str]:
        names = []
        if self.params.stats:
            names += [f"{c}__{s}" for s in ("n_chars", "n_words", "n_digits", "upper_ratio", "n_punct")]
        if c in self.svd_:
            names += [f"{c}__svd{i}" for i in range(self.params.svd_components)]
        elif c in self.vec_:
            names += [f"{c}__tfidf_{t}" for t in self.vec_[c].get_feature_names_out()]
        return names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.columns_:
            if c not in out.columns:
                continue
            t = out[c].fillna("").astype(str)
            new = {}
            if self.params.stats:
                n = t.str.len()
                new[f"{c}__n_chars"] = n.astype(float)
                new[f"{c}__n_words"] = t.str.split().str.len().fillna(0).astype(float)
                new[f"{c}__n_digits"] = t.str.count(r"\d").astype(float)
                new[f"{c}__upper_ratio"] = (t.str.count(r"[A-Z]") / n.where(n > 0)).fillna(0.0)
                new[f"{c}__n_punct"] = t.str.count(r"[^\w\s]").astype(float)
            block = pd.DataFrame(new, index=out.index)
            if c in self.vec_:
                m = self.vec_[c].transform(t)
                if c in self.svd_:
                    arr = self.svd_[c].transform(m)
                    names = [f"{c}__svd{i}" for i in range(arr.shape[1])]
                else:
                    arr = m.toarray()
                    names = [f"{c}__tfidf_{w}" for w in self.vec_[c].get_feature_names_out()]
                block = pd.concat([block, pd.DataFrame(arr, columns=names, index=out.index)], axis=1)
            out = pd.concat([out, block], axis=1)
            if self.params.drop_original:
                out = out.drop(columns=[c])
        return out
