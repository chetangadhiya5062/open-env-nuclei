"""FeatureForge: individual steps, recipes, pipelines, auto mode, persistence and the environment bridge."""

import numpy as np
import pandas as pd
import pytest

from featureforge import Pipeline, PipelineError, Recipe, StepSpec, auto_recipe
from featureforge.samples import TARGETS, make_dataset
from featureforge.steps import REGISTRY, catalog


@pytest.fixture(autouse=True)
def _workdir(tmp_path, monkeypatch):
    monkeypatch.setenv("FF_WORK_DIR", str(tmp_path / "ws"))


def run(df, *steps, target=None, task="auto", **kw):
    r = Recipe(target=target, task=task, steps=[StepSpec(step=s, params=p) for s, p in steps], **kw)
    pipe = Pipeline(r)
    return pipe, pipe.fit_transform(df)


def test_catalog_lists_every_step_with_schema():
    names = {c["step"] for c in catalog()}
    assert names == set(REGISTRY) and len(names) >= 20
    assert all("properties" in c["params"] or c["params"].get("type") == "object" for c in catalog())


# ------------------------------------------------------------------------------------------- cleaning
def test_normalize_and_fix_types():
    df = pd.DataFrame(
        {
            "city": [" New York", "new york", "New  York", "Boston", "N/A"] * 4,
            "price": ["$1,000", "20", "-", "3.5", None] * 4,
            "flag": ["yes", "no", "Y", "N", "yes"] * 4,
            "when": ["2024-01-02", "02/01/2024", "Jan 3, 2024", "2024/01/04", "?"] * 4,
        }
    )
    pipe, out = run(df, ("normalize_strings", {}), ("fix_types", {}))
    assert set(out["city"].dropna()) == {"New York", "Boston"} and out["city"].isna().sum() == 4
    assert out["price"].dtype == float and out["price"].iloc[0] == 1000.0
    assert set(out["flag"]) == {0.0, 1.0}
    assert pd.api.types.is_datetime64_any_dtype(out["when"])


def test_deduplicate_is_train_only_and_drops_missing_target():
    df = pd.DataFrame({"a": [1, 1, 2, 3], "y": [0, 0, 1, None]})
    pipe, out = run(df, ("deduplicate", {}), target="y")
    assert len(out) == 2
    assert len(pipe.transform(df)) == 4  # never removes rows from new data


@pytest.mark.parametrize("strategy", ["median", "mean", "knn", "iterative", "constant"])
def test_impute_strategies(strategy):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100), "c": rng.choice(["x", "y"], 100)})
    df.loc[::7, "a"] = np.nan
    df.loc[::9, "c"] = None
    pipe, out = run(df, ("impute", {"numeric_strategy": strategy}))
    assert out[["a", "b", "c"]].isna().sum().sum() == 0
    assert out["a__was_missing"].sum() == df["a"].isna().sum()
    new = pd.DataFrame({"a": [np.nan], "b": [0.1], "c": [None]})
    assert pipe.transform(new).isna().sum().sum() == 0


@pytest.mark.parametrize("method", ["iqr", "mad", "zscore", "quantile"])
def test_outlier_capping_learns_bounds_on_train(method):
    df = pd.DataFrame({"x": list(np.random.default_rng(1).normal(50, 5, 300)) + [1000.0, -900.0]})
    pipe, out = run(df, ("outliers", {"method": method, "action": "both", "threshold": 3.0}))
    assert out["x"].max() < 1000 and out["x"].min() > -900
    assert out["x__outlier"].sum() >= 2
    assert pipe.transform(pd.DataFrame({"x": [5000.0]}))["x"].iloc[0] < 1000


def test_outlier_explicit_bounds_and_isolation_forest():
    df = pd.DataFrame({"age": [20, 30, 999, 40, -5] * 10, "inc": np.arange(50.0)})
    _, out = run(df, ("outliers", {"bounds": {"age": [0, 120]}}))
    assert out["age"].between(0, 120).all()
    _, out = run(df, ("outliers", {"method": "isolation_forest"}))
    assert {"is_outlier", "outlier_score"} <= set(out.columns)


# -------------------------------------------------------------------------------------------- features
def test_datetime_features_cyclical_and_elapsed():
    df = pd.DataFrame({"t": pd.date_range("2024-01-01 00:00", periods=2000, freq="7h")})
    _, out = run(df, ("datetime_features", {}))
    assert "t" not in out.columns
    for c in ("t__hour", "t__hour_sin", "t__month_cos", "t__is_weekend", "t__days_since_start"):
        assert c in out.columns
    assert np.allclose(out["t__hour_sin"] ** 2 + out["t__hour_cos"] ** 2, 1)
    assert "t__minute" not in out.columns  # constant parts are skipped


def test_numeric_transform_interactions_groups_text_scale():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.lognormal(0, 1.5, 300),
            "b": rng.normal(size=300),
            "g": rng.choice(["p", "q", "r"], 300),
            "txt": [f"word {i} and 12 MORE!" for i in range(300)],
        }
    )
    pipe, out = run(
        df,
        ("numeric_transform", {"method": "auto"}),
        ("interactions", {"max_columns": 2}),
        ("group_aggregations", {}),
        ("text_features", {"tfidf": True, "max_features": 5}),
        ("scale", {"method": "standard"}),
    )
    assert abs(out["a"].skew()) < abs(df["a"].skew())
    assert {"a__mul__b", "b__mul__a"} & set(out.columns)
    assert "a__mean_by_g" in out.columns and "a__dev_from_mean_by_g" in out.columns
    assert "txt__n_words" in out.columns and any(c.startswith("txt__tfidf_") for c in out.columns)
    assert abs(out["b"].mean()) < 1e-9
    new = df.head(5).assign(g=["zzz"] * 5)  # unseen group -> global statistics, no NaN
    assert pipe.transform(new).filter(like="_by_g").isna().sum().sum() == 0


# -------------------------------------------------------------------------------------------- encoding
def test_one_hot_frequency_ordinal_handle_unseen():
    df = pd.DataFrame({"c": ["a"] * 50 + ["b"] * 45 + ["rare"] * 1, "d": ["x", "y"] * 48})
    pipe, out = run(df, ("frequency_encode", {"columns": ["c"]}), ("one_hot", {"min_frequency": 0.05}))
    assert {"c=a", "c=b", "c=__other__", "c__freq"} <= set(out.columns)
    new = pipe.transform(pd.DataFrame({"c": ["new"], "d": ["x"]}))
    assert new["c=__other__"].iloc[0] == 1 and new["c__freq"].iloc[0] == 0
    _, out = run(df, ("ordinal_encode", {}))
    assert out["c"].dtype == "int64"


def test_target_encoding_out_of_fold_does_not_leak():
    """A pure-noise ID-like category must not look predictive on the training rows (the classic leak)."""
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({"noise": [f"k{i}" for i in rng.integers(0, 700, n)], "y": rng.integers(0, 2, n)})
    for scheme in ("oof", "ordered"):
        _, out = run(df, ("target_encode", {"columns": ["noise"], "smoothing": 1.0, "scheme": scheme}), target="y")
        corr = np.corrcoef(out["noise__te"], out["y"])[0, 1]
        assert abs(corr) < 0.08, (scheme, corr)
    # the naive in-sample encoding *would* leak; this guards the test itself
    means = df.groupby("noise")["y"].transform("mean")
    assert np.corrcoef(means, df["y"])[0, 1] > 0.4


def test_target_encoding_learns_real_signal_and_multiclass():
    rng = np.random.default_rng(1)
    g = rng.choice(list("abcdefghij"), 3000)
    y = np.where(np.isin(g, ["a", "b", "c"]), "hi", np.where(g == "d", "mid", "lo"))
    flip = rng.random(3000) < 0.2
    y[flip] = rng.choice(["hi", "mid", "lo"], flip.sum())
    df = pd.DataFrame({"g": g, "y": y})
    pipe, out = run(df, ("target_encode", {}), target="y")
    assert {"g__te_hi", "g__te_lo", "g__te_mid"} <= set(out.columns)
    test = pipe.transform(pd.DataFrame({"g": ["a", "zzz"]}))
    assert test["g__te_hi"].iloc[0] > 0.7 and 0.2 < test["g__te_hi"].iloc[1] < 0.5  # unseen -> prior


def test_target_encode_requires_target():
    with pytest.raises(PipelineError, match="needs the target"):
        run(pd.DataFrame({"c": ["a", "b"]}), ("target_encode", {}))


# ------------------------------------------------------------------------------------------- selection
def test_variance_correlation_and_mi_selection():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    df = pd.DataFrame(
        {"x": x, "x_copy": x * 2 + 1e-6, "noise": rng.normal(size=500), "const": 3.0, "y": (x > 0).astype(int)}
    )
    _, out = run(
        df,
        ("variance_filter", {}),
        ("correlation_filter", {"threshold": 0.99}),
        ("mutual_info_select", {"k": 1}),
        target="y",
    )
    assert list(out.columns) in (["x", "y"], ["x_copy", "y"])


def test_feature_generation_finds_the_hidden_product():
    rng = np.random.default_rng(0)
    a, b, c = rng.uniform(-3, 3, 1500), rng.uniform(-3, 3, 1500), rng.normal(size=1500)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "y": a * b + 0.1 * rng.normal(size=1500)})
    pipe, out = run(df, ("feature_generation", {"max_new": 3, "base_model": "linear"}), target="y", task="regression")
    gen = [col for col in out.columns if col.startswith("gen__")]
    assert gen[0] in ("gen__a__mul__b", "gen__b__mul__a"), gen
    assert all(col in pipe.transform(df.drop(columns="y")).columns for col in gen)


def test_feature_generation_adds_nothing_for_pure_noise():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(800, 4)), columns=list("abcd")).assign(y=rng.normal(size=800))
    _, out = run(df, ("feature_generation", {}), target="y", task="regression")
    assert not [c for c in out.columns if c.startswith("gen__")]


# ------------------------------------------------------------------------------------- pipeline & auto
@pytest.mark.parametrize("kind", ["churn", "housing", "transactions"])
@pytest.mark.parametrize("level", ["clean", "standard", "full"])
def test_auto_recipes_run_and_replay(kind, level, tmp_path):
    df = make_dataset(kind, 600, 3)
    target = TARGETS[kind]
    r = auto_recipe(df, target=target, level=level)
    pipe = Pipeline(r)
    out = pipe.fit_transform(df)
    feats = [c for c in out.columns if c not in pipe.ctx_.id_columns and c != target]
    assert out[feats].isna().sum().sum() == 0
    if level != "clean":
        assert all(pd.api.types.is_numeric_dtype(out[c]) for c in feats), out[feats].dtypes.to_dict()
    pipe.save(tmp_path / "p")
    again = Pipeline.load(tmp_path / "p").transform(df.drop(columns=[target]))
    assert list(again.columns) == [c for c in out.columns if c != target]
    assert (tmp_path / "p" / "recipe.yaml").exists() and (tmp_path / "p" / "metadata.json").exists()


def test_saved_pipeline_reproduces_identical_output(tmp_path):
    df = make_dataset("churn", 500, 4)
    test = make_dataset("churn", 200, 9).drop(columns=["churned"])
    pipe = Pipeline(auto_recipe(df, "churned"))
    pipe.fit(df)
    a = pipe.transform(test)
    pipe.save(tmp_path / "art")
    b = Pipeline.load(tmp_path / "art").transform(test)
    pd.testing.assert_frame_equal(a, b)
    # the recipe alone refits to the same result (deterministic)
    c = Pipeline(Recipe.from_file(tmp_path / "art" / "recipe.yaml")).fit(df).transform(test)
    pd.testing.assert_frame_equal(a, c)


def test_auto_drops_leaky_column():
    df = make_dataset("churn", 800, 0)
    r = auto_recipe(df, "churned")
    assert "cancellation_reason" in r.steps[0].params["columns"]


def test_input_validation_on_transform():
    df = make_dataset("housing", 300, 0)
    pipe = Pipeline(auto_recipe(df, "price"))
    pipe.fit(df)
    with pytest.raises(PipelineError, match="missing"):
        pipe.transform(df.drop(columns=["sqft"]))
    pipe.transform(df.assign(extra=1))
    assert any("not seen during fitting" in w for w in pipe.warnings_)


def test_invalid_recipes_give_clear_errors():
    with pytest.raises(PipelineError, match="unknown step"):
        Pipeline({"steps": [{"step": "magic"}]})
    with pytest.raises(PipelineError, match="impute"):
        Pipeline({"steps": [{"step": "impute", "params": {"numeric_strategy": "guess"}}]})
    with pytest.raises(PipelineError, match="target column"):
        Pipeline({"target": "nope", "steps": []}).fit(pd.DataFrame({"a": [1]}))
    with pytest.raises(PipelineError, match="not fitted"):
        Pipeline({"steps": []}).transform(pd.DataFrame({"a": [1]}))


def test_recipe_yaml_and_json_roundtrip(tmp_path):
    r = auto_recipe(make_dataset("transactions", 300, 0), "is_fraud", level="full")
    assert Recipe.from_text(r.to_yaml()) == r == Recipe.from_text(r.to_json())
    r.save(tmp_path / "r.json")
    assert Recipe.from_file(tmp_path / "r.json") == r


def test_step_report_tracks_shapes():
    df = make_dataset("churn", 400, 1)
    pipe = Pipeline(auto_recipe(df, "churned"))
    out = pipe.fit_transform(df)
    rep = pipe.summary()["steps"]
    assert rep[0]["rows_before"] == len(df) and rep[-1]["cols_after"] == out.shape[1]
    assert any(s["step"] == "deduplicate" and s["rows_after"] < s["rows_before"] for s in rep)


# ---------------------------------------------------------------------------------- environment bridge
def test_library_cleaning_beats_do_nothing_on_env_tasks():
    from featureforge.envbridge import clean_and_grade

    for task in ("easy-clean", "hard-clean"):
        res = clean_and_grade(task, 0)
        assert res["score"] > res["do_nothing_score"] and res["improvement"] > 0.5
        assert res["nan_cells"] == 0
