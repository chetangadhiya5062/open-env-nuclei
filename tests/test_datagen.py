import pandas as pd
import pytest

from data_cleaning_env.schema import COLUMNS
from data_cleaning_env.server.datagen import TASKS, generate
from data_cleaning_env.server.quality import detect_issues, issue_cells

TASK_IDS = sorted(TASKS)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_generation_is_deterministic(task_id):
    a, ta = generate(task_id, 7)
    b, tb = generate(task_id, 7)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ta, tb)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_different_seeds_differ(task_id):
    assert not generate(task_id, 1)[0].equals(generate(task_id, 2)[0])


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_truth_is_clean_and_dirty_is_not(task_id):
    dirty, truth = generate(task_id, 0)
    assert list(dirty.columns) == COLUMNS == list(truth.columns)
    assert issue_cells(truth) == 0
    assert truth["customer_id"].is_unique
    assert issue_cells(dirty) > 0
    assert len(truth) == TASKS[task_id].n_rows


def test_easy_only_has_missing_ages():
    dirty, _ = generate("easy-clean", 3)
    issues = {c: i for c, i in detect_issues(dirty).items() if i}
    assert set(issues) == {"age"}
    assert set(issues["age"]) == {"missing"}
    assert dirty.duplicated().sum() == 0


def test_medium_has_missing_and_duplicates_only():
    dirty, truth = generate("medium-clean", 3)
    assert dirty.duplicated().sum() > 0
    assert len(dirty) > len(truth)
    kinds = {k for i in detect_issues(dirty).values() for k in i}
    assert kinds == {"missing"}


def test_hard_has_every_noise_type():
    dirty, _ = generate("hard-clean", 3)
    kinds = {k for i in detect_issues(dirty).values() for k in i}
    assert {"missing", "wrong_type", "out_of_range", "whitespace", "not_allowed_value", "bad_date_format"} <= kinds
    assert dirty.duplicated().sum() > 0


def test_unknown_task_raises():
    with pytest.raises(ValueError, match="unknown task_id"):
        generate("nope", 0)


def test_openenv_yaml_lists_exactly_the_implemented_tasks():
    from pathlib import Path

    import yaml

    manifest = yaml.safe_load((Path(__file__).parents[1] / "openenv.yaml").read_text(encoding="utf-8"))
    assert sorted(t["id"] for t in manifest["tasks"]) == TASK_IDS
    for t in manifest["tasks"]:
        assert t["difficulty"] == TASKS[t["id"]].difficulty
