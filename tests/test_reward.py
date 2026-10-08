"""The reward cannot be farmed (regression tests for the exploits in the original environment)."""

import pytest

from agents import RuleBasedAgent, run_episode
from agents.runner import LocalEnv
from data_cleaning_env.server import data_cleaning_env_environment as E
from tests.conftest import act

CLEAN_EASY = [act(action_type="fill_missing", column_name="age", strategy="median")]


def clean_easy(env):
    env.reset(task_id="easy-clean", seed=0)
    return env.step(CLEAN_EASY[0])


def test_reward_is_per_step_not_cumulative(env):
    env.reset(task_id="easy-clean", seed=0)
    o1 = env.step(CLEAN_EASY[0])
    o2 = env.step(act(action_type="drop_duplicates"))
    assert o2.reward == pytest.approx(-(E.STEP_COST + E.NOOP_PENALTY))  # not o1.reward + something
    assert o2.total_reward == pytest.approx(o1.reward + o2.reward)


def test_no_bonus_for_staying_in_a_clean_state(env):
    """Old bug: +5 on every step once clean, so spamming until step 10 beat finishing."""
    obs = clean_easy(env)
    assert obs.quality_score == 1.0
    rewards = []
    for _ in range(5):
        obs = env.step(act(action_type="strip_whitespace"))
        rewards.append(obs.reward)
    assert all(r < 0 for r in rewards)


def test_spamming_is_worse_than_finishing(env):
    clean_easy(env)
    finish_total = env.step(act(action_type="finish")).total_reward

    clean_easy(env)
    for _ in range(10):
        obs = env.step(act(action_type="fix_dates", column_name="signup_date"))
    assert obs.total_reward < finish_total


def test_terminal_reward_paid_once(env):
    clean_easy(env)
    first = env.step(act(action_type="finish"))
    assert first.done and first.reward > 0
    again = env.step(act(action_type="finish"))
    assert again.reward == 0.0 and again.total_reward == first.total_reward and again.last_action_ok is False


def test_shaping_telescopes(env):
    """Sum of rewards == 10*(Q_T - Q_0) - step costs when no penalties fire."""
    obs = env.reset(task_id="medium-clean", seed=1)
    q0 = obs.quality_score
    obs = env.step(act(action_type="drop_duplicates"))
    obs = env.step(act(action_type="fill_missing", column_name="age", strategy="median"))
    expected = E.SHAPING_SCALE * (obs.quality_score - q0) - 2 * E.STEP_COST
    assert obs.total_reward == pytest.approx(expected, abs=1e-4)


def test_repeating_an_action_earns_nothing(env):
    env.reset(task_id="medium-clean", seed=1)
    first = env.step(act(action_type="drop_duplicates"))
    assert first.reward > 0
    for _ in range(3):
        assert env.step(act(action_type="drop_duplicates")).reward < 0


def test_destroying_rows_is_penalised(env):
    env.reset(task_id="medium-clean", seed=1)
    obs = env.step(act(action_type="drop_rows_with_missing"))
    assert obs.reward < 0  # quality gain is smaller than the data-loss penalty
    assert obs.row_count < 60


def test_dropping_duplicates_does_not_trigger_loss_penalty(env):
    env.reset(task_id="medium-clean", seed=1)
    obs = env.step(act(action_type="drop_duplicates"))
    assert obs.reward > 0 and obs.row_count == 60


def test_premature_finish_is_penalised(env):
    env.reset(task_id="hard-clean", seed=0)
    obs = env.step(act(action_type="finish"))
    assert obs.reward < -0.5 and obs.done and obs.final_score is not None


def test_step_budget_ends_episode(env):
    env.reset(task_id="easy-clean", seed=0)
    for _ in range(E.get_task("easy-clean").max_steps):
        obs = env.step(act(action_type="strip_whitespace"))
    assert obs.done and obs.steps_remaining == 0 and obs.final_score is not None


def test_final_score_hidden_until_done(env):
    obs = env.reset(task_id="easy-clean", seed=0)
    assert obs.final_score is None and obs.score_breakdown is None
    obs = env.step(CLEAN_EASY[0])
    assert obs.final_score is None
    assert env.step(act(action_type="finish")).final_score is not None


def test_rule_based_beats_do_nothing_on_reward_and_score():
    from agents import DoNothingAgent

    env = LocalEnv()
    for task in ("medium-clean", "hard-clean"):
        good = run_episode(RuleBasedAgent(), env, task, 0)
        nothing = run_episode(DoNothingAgent(), env, task, 0)
        assert good.final_score > nothing.final_score
        assert good.total_reward > nothing.total_reward
