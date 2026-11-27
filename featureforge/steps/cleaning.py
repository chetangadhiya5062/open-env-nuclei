"""Cleaning steps: drop unusable columns, normalise strings, fix types, deduplicate, impute, handle outliers."""

from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ..types import (
    ColumnType,
    is_placeholder,
    normalize_ws,
    parse_bool,
    parse_datetime,
    parse_numeric,
)
from .base import FitContext, Step, StepError, register, subsample

ALL = tuple(ColumnType)
NUMERIC_LIKE = (ColumnType.NUMERIC, ColumnType.BOOLEAN)
STRINGY = (ColumnType.CATEGORICAL, ColumnType.TEXT, ColumnType.BOOLEAN, ColumnType.NUMERIC, ColumnType.DATETIME)


class _P(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------------------- drop_columns
class DropColumnsParams(_P):
    columns: List[str] = Field(default_factory=list, description="columns to drop explicitly")
    drop_constant: bool = True
    max_missing: Optional[float] = Field(0.95, ge=0, le=1, description="drop columns with more missing; null = off")
    drop_ids: bool = False


@register
class DropColumns(Step):
    name = "drop_columns"
    kind = "clean"
    description = "Drop listed columns, constant columns and columns that are (almost) entirely missing."
    Params = DropColumnsParams

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "DropColumns":
        p = self.params
        drop = [c for c in p.columns if c in df.columns]
        missing = [c for c in p.columns if c not in df.columns]
        if missing:
            self.note(f"listed columns not present: {missing}")
        for c in df.columns:
            if c in ctx.protected or c in drop:
                continue
            if p.drop_constant and ctx.types.get(c) == ColumnType.CONSTANT:
                drop.append(c)
            elif p.max_missing is not None and df[c].isna().mean() > p.max_missing:
                drop.append(c)
        if p.drop_ids:
            drop += [c for c in ctx.id_columns if c in df.columns and c not in drop]
        self.drop_ = drop
        for c in drop:
            ctx.types.pop(c, None)
        if drop:
            self.note(f"dropped {len(drop)} column(s): {drop}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[c for c in self.drop_ if c in df.columns])


# --------------------------------------------------------------------------------------- normalize_strings
class NormalizeStringsParams(_P):
    columns: Optional[List[str]] = None
    strip: bool = True
    collapse_spaces: bool = True
    placeholders_to_null: bool = True
    unify_case: bool = Field(True, description="merge spellings that differ only by case/spacing into the most common")
    lowercase: bool = False
    include_target: bool = True


@register
class NormalizeStrings(Step):
    name = "normalize_strings"
    kind = "clean"
    description = "Trim/collapse whitespace, turn placeholders like 'N/A' into nulls, unify case variants."
    Params = NormalizeStringsParams

    def _clean_value(self, v):
        if not isinstance(v, str):
            return v
        p = self.params
        if p.placeholders_to_null and is_placeholder(v):
            return np.nan
        if p.collapse_spaces:
            v = normalize_ws(v)
        elif p.strip:
            v = v.strip()
        if p.lowercase:
            v = v.lower()
        return v

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "NormalizeStrings":
        exclude = set(ctx.id_columns) if self.params.include_target else ctx.protected
        cols = ctx.columns_of(df, STRINGY, self.params.columns, exclude=exclude)
        self.columns_ = [c for c in cols if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
        self.maps_: Dict[str, Dict[str, str]] = {}
        if self.params.unify_case:
            for c in self.columns_:
                if ctx.types.get(c) not in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN):
                    continue
                vals = df[c].map(self._clean_value).dropna().astype(str)
                counts = vals.value_counts()
                canon: Dict[str, str] = {}
                for v, _n in counts.items():  # most frequent first -> it becomes the canonical spelling
                    canon.setdefault(v.casefold(), v)
                mapping = {v: canon[v.casefold()] for v in counts.index if canon[v.casefold()] != v}
                if mapping:
                    self.maps_[c] = mapping
                    self.note(f"{c}: merged {len(mapping)} spelling variant(s)")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.columns_:
            if c not in out.columns:
                continue
            col = out[c].astype(object).map(self._clean_value)
            m = self.maps_.get(c)
            if m:
                col = col.map(lambda v, m=m: m.get(v, v) if isinstance(v, str) else v)
            out[c] = col
        return out


# ---------------------------------------------------------------------------------------------- fix_types
class FixTypesParams(_P):
    columns: Optional[List[str]] = None
    numeric: bool = True
    datetime: bool = True
    boolean: bool = True


@register
class FixTypes(Step):
    name = "fix_types"
    kind = "clean"
    description = "Convert numbers stored as text ('1,250', '$12', '45%'), dates in mixed formats, and yes/no flags."
    Params = FixTypesParams

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "FixTypes":
        p = self.params
        cols = ctx.columns_of(df, ALL, p.columns)
        self.plan_: Dict[str, str] = {}
        for c in cols:
            t = ctx.types.get(c)
            is_text = df[c].dtype == object or pd.api.types.is_string_dtype(df[c])
            if p.numeric and t == ColumnType.NUMERIC and is_text:
                self.plan_[c] = "numeric"
            elif p.datetime and t == ColumnType.DATETIME and not pd.api.types.is_datetime64_any_dtype(df[c]):
                self.plan_[c] = "datetime"
            elif p.datetime and t == ColumnType.DATETIME and getattr(df[c].dt, "tz", None) is not None:
                self.plan_[c] = "datetime"
            elif p.boolean and t == ColumnType.BOOLEAN and not pd.api.types.is_numeric_dtype(df[c]):
                self.plan_[c] = "boolean"
            elif p.boolean and t == ColumnType.BOOLEAN and pd.api.types.is_bool_dtype(df[c]):
                self.plan_[c] = "boolean"
        for c, kind in self.plan_.items():
            before = int(df[c].notna().sum())
            after = int(self._convert(df[c], kind).notna().sum())
            lost = before - after
            self.note(f"{c} -> {kind}" + (f" ({lost} unparsable value(s) became null)" if lost else ""))
        return self

    @staticmethod
    def _convert(s: pd.Series, kind: str) -> pd.Series:
        if kind == "numeric":
            return parse_numeric(s)
        if kind == "datetime":
            return parse_datetime(s)
        return parse_bool(s)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c, kind in self.plan_.items():
            if c in out.columns:
                out[c] = self._convert(out[c], kind)
        return out


# ------------------------------------------------------------------------------------------- deduplicate
class DeduplicateParams(_P):
    subset: Optional[List[str]] = None
    keep: Literal["first", "last"] = "first"
    drop_missing_target: bool = True


@register
class Deduplicate(Step):
    name = "deduplicate"
    kind = "clean"
    description = "Remove exact duplicate rows and rows without a target (training data only)."
    Params = DeduplicateParams
    train_only = True

    def fit_transform(self, df: pd.DataFrame, ctx: FitContext) -> pd.DataFrame:
        p = self.params
        subset = p.subset
        if subset:
            bad = [c for c in subset if c not in df.columns]
            if bad:
                raise StepError(f"deduplicate subset columns not found: {bad}")
        n0 = len(df)
        key = df[subset] if subset else df
        dup = pd.util.hash_pandas_object(key.astype(str), index=False).duplicated(keep=p.keep).to_numpy()
        out = df.loc[~dup]
        n_dup = n0 - len(out)
        n_target = 0
        if p.drop_missing_target and ctx.target and ctx.target in out.columns:
            mask = out[ctx.target].notna()
            n_target = int((~mask).sum())
            out = out.loc[mask]
        self.note(f"removed {n_dup} duplicate row(s)" + (f" and {n_target} row(s) without target" if n_target else ""))
        self.fitted_ = True
        return out.reset_index(drop=True)

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "Deduplicate":
        self.fit_transform(df, ctx)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:  # not applied to new data
        return df


# ------------------------------------------------------------------------------------------------ impute
class ImputeParams(_P):
    columns: Optional[List[str]] = None
    numeric_strategy: Literal["median", "mean", "knn", "iterative", "constant"] = "median"
    numeric_fill_value: float = 0.0
    categorical_strategy: Literal["most_frequent", "constant"] = "constant"
    categorical_fill_value: str = "__missing__"
    datetime_strategy: Literal["median", "none"] = "median"
    add_indicator: bool = True
    knn_neighbors: int = Field(5, ge=1, le=50)
    max_iter: int = Field(10, ge=1, le=50)


@register
class Impute(Step):
    name = "impute"
    kind = "clean"
    description = "Fill missing values (median/mean/KNN/iterative for numbers, constant or mode for categories) and add missing-indicator flags."
    Params = ImputeParams

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "Impute":
        p = self.params
        self.num_cols_ = [
            c for c in ctx.columns_of(df, NUMERIC_LIKE, p.columns) if pd.api.types.is_numeric_dtype(df[c])
        ]
        self.cat_cols_ = ctx.columns_of(df, (ColumnType.CATEGORICAL, ColumnType.TEXT), p.columns)
        self.dt_cols_ = [
            c
            for c in ctx.columns_of(df, (ColumnType.DATETIME,), p.columns)
            if pd.api.types.is_datetime64_any_dtype(df[c])
        ]
        skipped = [c for c in ctx.columns_of(df, NUMERIC_LIKE, p.columns) if not pd.api.types.is_numeric_dtype(df[c])]
        if skipped:
            self.note(f"skipped non-numeric numeric-like columns (run fix_types first): {skipped}")
        all_cols = self.num_cols_ + self.cat_cols_ + self.dt_cols_
        self.indicator_cols_ = [c for c in all_cols if p.add_indicator and df[c].isna().any()]
        for c in self.indicator_cols_:
            ctx.types[f"{c}__was_missing"] = ColumnType.BOOLEAN

        self.num_fill_: Dict[str, float] = {}
        self.model_ = None
        num = df[self.num_cols_].astype(float)
        if p.numeric_strategy in ("knn", "iterative") and self.num_cols_ and num.isna().any().any():
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer, KNNImputer

            train = subsample(num, ctx.max_fit_rows, ctx.random_state)
            if p.numeric_strategy == "knn":
                self.model_ = KNNImputer(n_neighbors=p.knn_neighbors, keep_empty_features=True)
            else:
                self.model_ = IterativeImputer(
                    max_iter=p.max_iter, random_state=ctx.random_state, keep_empty_features=True
                )
            self.model_.fit(train)
        for c in self.num_cols_:
            col = num[c]
            if p.numeric_strategy == "mean":
                v = col.mean()
            elif p.numeric_strategy == "constant":
                v = p.numeric_fill_value
            else:
                v = col.median()
            self.num_fill_[c] = float(0.0 if pd.isna(v) else v)
        self.cat_fill_: Dict[str, str] = {}
        for c in self.cat_cols_:
            if p.categorical_strategy == "most_frequent":
                m = df[c].dropna().astype(str).mode()
                self.cat_fill_[c] = str(m.iloc[0]) if len(m) else p.categorical_fill_value
            else:
                self.cat_fill_[c] = p.categorical_fill_value
        self.dt_fill_ = {}
        if p.datetime_strategy == "median":
            for c in self.dt_cols_:
                med = df[c].dropna().median() if df[c].notna().any() else pd.NaT
                self.dt_fill_[c] = med
        n_missing = int(df[all_cols].isna().sum().sum()) if all_cols else 0
        self.note(f"{n_missing} missing cell(s) in {len(all_cols)} column(s); numeric strategy {p.numeric_strategy}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.indicator_cols_:
            if c in out.columns:
                out[f"{c}__was_missing"] = out[c].isna().astype("int8")
        present = [c for c in self.num_cols_ if c in out.columns]
        if self.model_ is not None and len(present) == len(self.num_cols_):
            vals = out[self.num_cols_].astype(float)
            if vals.isna().any().any():
                filled = self.model_.transform(vals)
                out[self.num_cols_] = pd.DataFrame(filled, columns=self.num_cols_, index=out.index)
        for c in present:
            out[c] = out[c].astype(float).fillna(self.num_fill_[c])
        for c, v in self.cat_fill_.items():
            if c in out.columns:
                out[c] = out[c].astype(object).where(out[c].notna(), v)
        for c, v in self.dt_fill_.items():
            if c in out.columns and not pd.isna(v):
                out[c] = out[c].fillna(v)
        return out


# ---------------------------------------------------------------------------------------------- outliers
class OutliersParams(_P):
    columns: Optional[List[str]] = None
    method: Literal["iqr", "mad", "zscore", "quantile", "isolation_forest"] = "iqr"
    k: float = Field(1.5, gt=0, description="IQR fence multiplier")
    lower_quantile: float = Field(0.005, ge=0, lt=0.5, description="method quantile: lower cut")
    upper_quantile: float = Field(0.995, gt=0.5, le=1, description="method quantile: upper cut")
    threshold: float = Field(3.5, gt=0, description="cut-off for mad (modified z) and zscore")
    action: Literal["clip", "flag", "both", "null"] = "clip"
    contamination: Union[float, Literal["auto"]] = "auto"
    bounds: Dict[str, List[float]] = Field(
        default_factory=dict, description="explicit valid ranges {column: [low, high]}; override the method"
    )


@register
class Outliers(Step):
    name = "outliers"
    kind = "clean"
    description = "Cap, flag or null outliers (IQR fences, MAD modified z-score, z-score, quantile winsorising) or flag rows with Isolation Forest."
    Params = OutliersParams

    def fit(self, df: pd.DataFrame, ctx: FitContext) -> "Outliers":
        p = self.params
        cols = [
            c
            for c in ctx.columns_of(df, (ColumnType.NUMERIC,), p.columns)
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2
        ]
        self.columns_ = cols
        self.bounds_: Dict[str, tuple] = {}
        self.forest_ = None
        if p.method == "isolation_forest":
            if not cols:
                return self
            from sklearn.ensemble import IsolationForest

            x = df[cols].astype(float)
            self.medians_ = x.median().fillna(0).to_dict()
            train = subsample(x.fillna(self.medians_), ctx.max_fit_rows, ctx.random_state)
            self.forest_ = IsolationForest(
                contamination=p.contamination, random_state=ctx.random_state, n_estimators=200
            ).fit(train)
            ctx.types["is_outlier"] = ColumnType.BOOLEAN
            ctx.types["outlier_score"] = ColumnType.NUMERIC
            self.note(f"isolation forest over {len(cols)} numeric column(s)")
            return self
        for c in p.bounds:
            if c not in df.columns:
                raise StepError(f"bounds given for unknown column {c!r}")
            if c not in cols:
                cols.append(c)
        for c in cols:
            x = df[c].astype(float).dropna()
            if x.empty:
                continue
            if c in p.bounds:
                lo, hi = p.bounds[c]
            elif p.method == "iqr":
                q1, q3 = x.quantile([0.25, 0.75])
                lo, hi = q1 - p.k * (q3 - q1), q3 + p.k * (q3 - q1)
            elif p.method == "quantile":
                lo, hi = x.quantile([p.lower_quantile, p.upper_quantile])
            elif p.method == "mad":
                med = x.median()
                mad = (x - med).abs().median()
                if mad == 0:
                    continue
                spread = p.threshold * mad / 0.6745
                lo, hi = med - spread, med + spread
            else:
                mu, sd = x.mean(), x.std()
                if not sd:
                    continue
                lo, hi = mu - p.threshold * sd, mu + p.threshold * sd
            n_out = int(((x < lo) | (x > hi)).sum())
            if n_out:
                self.bounds_[c] = (float(lo), float(hi))
                self.note(f"{c}: {n_out} outlier(s) outside [{lo:.4g}, {hi:.4g}]")
                if p.action in ("flag", "both"):
                    ctx.types[f"{c}__outlier"] = ColumnType.BOOLEAN
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        p = self.params
        if self.forest_ is not None:
            if all(c in out.columns for c in self.columns_):
                x = out[self.columns_].astype(float).fillna(self.medians_)
                out["is_outlier"] = (self.forest_.predict(x) == -1).astype("int8")
                out["outlier_score"] = -self.forest_.score_samples(x)
            return out
        for c, (lo, hi) in self.bounds_.items():
            if c not in out.columns:
                continue
            x = out[c].astype(float)
            mask = (x < lo) | (x > hi)
            if p.action in ("flag", "both"):
                out[f"{c}__outlier"] = mask.astype("int8")
            if p.action in ("clip", "both"):
                out[c] = x.clip(lo, hi)
            elif p.action == "null":
                out[c] = x.mask(mask)
        return out
