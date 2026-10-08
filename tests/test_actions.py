"""Each action does what it says; invalid ones are rejected with an explanation (no crash)."""

import numpy as np
import pandas as pd
import pytest

from data_cleaning_env import actions as A
from data_cleaning_env.server import operations as ops
from tests.conftest import act


def table(**cols):
    base = {
        "customer_id": [1, 2, 3, 4],
        "name": ["a b", "c d", "e f", "g h"],
        "age": [20.0, 30.0, np.nan, 40.0],
        "city": ["New York", "Chicago", None, "Chicago"],
        "signup_date": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
        "plan": ["basic", "pro", "pro", "pro"],
        "monthly_spend": [10.0, 20.0, 30.0, 40.0],
    }
    base.update(cols)
    return pd.DataFrame(base)


# ---------------------------------------------------------------- parse_action
def test_parse_rejects_unknown_type_and_bad_args():
    assert A.parse_action({"action_type": "explode"})[1].startswith("unknown action_type")
    assert (
        "requires 'value'"
        in A.parse_action({"action_type": "fill_missing", "column_name": "age", "strategy": "constant"})[1]
    )
    assert "Usage" in A.parse_action({"action_type": "cast_type", "column_name": "age"})[1]
    action, err = A.parse_action({"action_type": "fill_missing", "column_name": "age", "strategy": "mean"})
    assert err is None and isinstance(action, A.FillMissing)


# ---------------------------------------------------------------- fill_missing
@pytest.mark.parametrize("strategy,expected", [("mean", 30), ("median", 30), ("constant", 7)])
def test_fill_numeric(strategy, expected):
    value = "7" if strategy == "constant" else None
    out, _ = ops.apply_action(
        table(), A.FillMissing(action_type="fill_missing", column_name="age", strategy=strategy, value=value)
    )
    assert out["age"].tolist()[2] == expected
    assert out["age"].dtype == np.int64  # whole numbers in an int column are normalised to int64


def test_fill_mode_categorical():
    out, _ = ops.apply_action(table(), A.FillMissing(action_type="fill_missing", column_name="city", strategy="mode"))
    assert out["city"].tolist()[2] == "Chicago"


def test_fill_rejects_string_in_numeric_column():
    """The old exploit: filling numeric `age` with the text 'missing'."""
    with pytest.raises(ops.ActionError, match="not a number"):
        ops.apply_action(
            table(), A.FillMissing(action_type="fill_missing", column_name="age", strategy="constant", value="missing")
        )


def test_fill_rejects_disallowed_category_and_mean_on_text():
    with pytest.raises(ops.ActionError, match="not an allowed value"):
        ops.apply_action(
            table(),
            A.FillMissing(action_type="fill_missing", column_name="city", strategy="constant", value="Atlantis"),
        )
    with pytest.raises(ops.ActionError, match="numeric column"):
        ops.apply_action(table(), A.FillMissing(action_type="fill_missing", column_name="city", strategy="mean"))


def test_fill_requires_cast_when_numeric_column_has_text():
    df = table(age=pd.Series(["20", "30", None, "N/A"], dtype=object))
    with pytest.raises(ops.ActionError, match="cast_type"):
        ops.apply_action(df, A.FillMissing(action_type="fill_missing", column_name="age", strategy="median"))


def test_fill_unknown_column():
    with pytest.raises(ops.ActionError, match="unknown column"):
        ops.apply_action(table(), A.FillMissing(action_type="fill_missing", column_name="zzz", strategy="mode"))


# ---------------------------------------------------------------- other actions
def test_drop_duplicates():
    df = pd.concat([table(), table().iloc[[0]]], ignore_index=True)
    out, msg = ops.apply_action(df, A.DropDuplicates(action_type="drop_duplicates"))
    assert len(out) == 4 and "1 duplicate" in msg


def test_strip_whitespace():
    df = table(name=[" a  b ", "c d", "e f", "g h"])
    out, _ = ops.apply_action(df, A.StripWhitespace(action_type="strip_whitespace"))
    assert out["name"].tolist()[0] == "a b"


def test_standardize_auto_and_mapping():
    df = table(city=["new york", "NY", "CHICAGO ", None])
    out, _ = ops.apply_action(df, A.StandardizeCategories(action_type="standardize_categories", column_name="city"))
    assert out["city"].tolist() == ["New York", "NY", "Chicago", None]
    out, _ = ops.apply_action(
        out,
        A.StandardizeCategories(action_type="standardize_categories", column_name="city", mapping={"ny": "New York"}),
    )
    assert out["city"].tolist()[1] == "New York"
    with pytest.raises(ops.ActionError, match="not allowed"):
        ops.apply_action(
            df,
            A.StandardizeCategories(action_type="standardize_categories", column_name="city", mapping={"NY": "Gotham"}),
        )


def test_cast_type_converts_strings_and_placeholders():
    df = table(age=pd.Series(["20", " 30 ", "N/A", 40], dtype=object))
    out, msg = ops.apply_action(df, A.CastType(action_type="cast_type", column_name="age", target_type="int"))
    assert out["age"].tolist()[:2] == [20.0, 30.0] and np.isnan(out["age"].tolist()[2])
    assert "became missing" in msg


def test_fix_dates():
    df = table(signup_date=["2022-01-01", "05 Mar 2023", "Mar 05, 2023", "2023/03/05"])
    out, _ = ops.apply_action(df, A.FixDates(action_type="fix_dates", column_name="signup_date"))
    assert out["signup_date"].tolist() == ["2022-01-01", "2023-03-05", "2023-03-05", "2023-03-05"]


def test_clip_outliers_uses_valid_range():
    df = table(age=[20.0, 999.0, -5.0, 40.0])
    out, _ = ops.apply_action(df, A.ClipOutliers(action_type="clip_outliers", column_name="age"))
    assert out["age"].tolist() == [20, 120, 0, 40]


def test_drop_rows_with_missing():
    df = table(city=["New York", None, "Chicago", "Chicago"])  # age is missing in row 2, city in row 1
    out, _ = ops.apply_action(df, A.DropRowsWithMissing(action_type="drop_rows_with_missing"))
    assert len(out) == 2
    out, _ = ops.apply_action(df, A.DropRowsWithMissing(action_type="drop_rows_with_missing", column_name="city"))
    assert len(out) == 3


def test_operations_do_not_mutate_input():
    df = table()
    before = df.copy()
    ops.apply_action(df, A.FillMissing(action_type="fill_missing", column_name="age", strategy="mean"))
    pd.testing.assert_frame_equal(df, before)


# ---------------------------------------------------------------- through the environment
def test_invalid_action_returns_error_not_exception(env):
    env.reset(task_id="easy-clean", seed=0)
    obs = env.step(act(action_type="fill_missing", column_name="age", strategy="constant", value="missing"))
    assert obs.last_action_ok is False and "not a number" in obs.last_error
    assert obs.invalid_action_count == 1 and obs.reward < 0
    obs = env.step(act(action_type="nonsense"))
    assert obs.last_action_ok is False and obs.invalid_action_count == 2
    obs = env.step(act(action_type="fill_missing", column_name="nope", strategy="mode"))
    assert "unknown column" in obs.last_error
