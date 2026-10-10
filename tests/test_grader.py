import pandas as pd
import pytest

from data_cleaning_env.server.datagen import generate
from data_cleaning_env.server.grader import grade, improvement


@pytest.mark.parametrize("task_id", ["easy-clean", "medium-clean", "hard-clean"])
def test_truth_scores_exactly_one(task_id):
    _, truth = generate(task_id, 4)
    result = grade(truth.copy(), truth)
    assert result["score"] == 1.0
    assert result["cell_accuracy"] == result["row_fidelity"] == result["schema_score"] == 1.0


@pytest.mark.parametrize("task_id", ["easy-clean", "medium-clean", "hard-clean"])
def test_dirty_scores_strictly_between_zero_and_one(task_id):
    dirty, truth = generate(task_id, 4)
    score = grade(dirty, truth)["score"]
    assert 0.0 < score < 1.0


def test_scores_stay_in_bounds_for_degenerate_tables():
    dirty, truth = generate("hard-clean", 0)
    for df in (dirty.iloc[:0], dirty.iloc[:1], dirty.drop(columns=["age"]), dirty.drop(columns=["customer_id"])):
        result = grade(df, truth)
        assert all(0.0 <= v <= 1.0 for v in result.values())


def test_lost_rows_and_leftover_duplicates_lower_the_score():
    _, truth = generate("easy-clean", 0)
    half = grade(truth.iloc[:20], truth)["score"]
    dup = grade(pd.concat([truth, truth.iloc[:10]], ignore_index=True), truth)["score"]
    assert half < 0.7 and dup < 1.0


def test_numeric_credit_is_partial_and_numpy_numbers_count():
    """Regression: numpy ints/floats used to score 0 because isinstance(x, (int, float)) is False for them."""
    _, truth = generate("easy-clean", 0)
    near = truth.copy()
    near["age"] = near["age"] + 1
    far = truth.copy()
    far["age"] = 999
    s_near, s_far = grade(near, truth)["cell_accuracy"], grade(far, truth)["cell_accuracy"]
    assert 0.9 < s_near < 1.0 and s_far < s_near


def test_text_in_numeric_column_scores_zero_for_those_cells():
    _, truth = generate("easy-clean", 0)
    bad = truth.copy()
    bad["age"] = bad["age"].astype(str)
    result = grade(bad, truth)
    assert result["schema_score"] < 1.0
    assert result["cell_accuracy"] < 1.0


def test_improvement_formula_and_edge_cases():
    assert improvement(1.0, 0.8) == pytest.approx(1.0)
    assert improvement(0.8, 0.8) == pytest.approx(0.0)
    assert improvement(0.9, 0.8) == pytest.approx(0.5)
    assert improvement(0.6, 0.8) == pytest.approx(-1.0)  # worse than leaving the data alone
    assert improvement(0.5, 1.0) == 0.0  # nothing to improve: no division by zero


@pytest.mark.parametrize("task_id", ["easy-clean", "medium-clean", "hard-clean"])
def test_final_observation_reports_improvement(task_id):
    from agents import DoNothingAgent, LocalEnv, RuleBasedAgent, run_episode

    env = LocalEnv()
    nothing = run_episode(DoNothingAgent(), env, task_id, 0)
    good = run_episode(RuleBasedAgent(), env, task_id, 0)
    assert nothing.improvement == pytest.approx(0.0, abs=1e-6)  # finishing at once is the baseline by definition
    assert 0.0 < good.improvement <= 1.0
    _, truth = generate(task_id, 0)
    dirty, _ = generate(task_id, 0)
    assert good.improvement == pytest.approx(improvement(good.final_score, grade(dirty, truth)["score"]), abs=1e-5)


def test_breakdown_exposes_baseline_and_improvement():
    from agents.runner import LocalEnv
    from data_cleaning_env.models import DataCleaningAction

    env = LocalEnv()
    env.reset("hard-clean", 1)
    obs = env.step(DataCleaningAction(action_type="finish"))
    assert {"score", "do_nothing_score", "improvement"} <= set(obs.score_breakdown)
    assert obs.score_breakdown["improvement"] == pytest.approx(0.0, abs=1e-6)
