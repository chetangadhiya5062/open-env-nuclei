"""RL agent: feature/action mapping, numpy policy, and (when the optional [rl] extra is installed) the gym wrapper."""

import numpy as np
import pytest

from agents import LocalEnv, RLAgent, run_episode
from agents.rl_agent import NumpyPolicy
from agents.rl_common import ACTION_LABELS, ACTIONS, FINISH_INDEX, N_ACTIONS, N_FEATURES, build_action, featurize
from data_cleaning_env.actions import parse_action
from data_cleaning_env.models import DataCleaningAction
from data_cleaning_env.server.datagen import TASKS


def first_obs(task="hard-clean", seed=0):
    return LocalEnv().reset(task, seed)


def test_feature_vector_shape_range_and_determinism():
    obs = first_obs()
    x = featurize(obs, obs.row_count)
    assert x.shape == (N_FEATURES,) and x.dtype == np.float32
    assert np.all(x >= 0) and np.all(x <= 1.5)
    assert np.array_equal(x, featurize(first_obs(), obs.row_count))


def test_features_change_when_the_data_changes():
    env = LocalEnv()
    obs = env.reset("medium-clean", 0)
    before = featurize(obs, obs.row_count)
    after = featurize(env.step(DataCleaningAction(action_type="drop_duplicates")), obs.row_count)
    assert not np.array_equal(before, after)


def test_every_curated_action_is_a_valid_typed_action():
    obs = first_obs("hard-clean", 1)
    assert N_ACTIONS == len(ACTIONS) == len(ACTION_LABELS) and ACTION_LABELS[FINISH_INDEX] == "finish"
    for i in range(N_ACTIONS):
        action = build_action(i, obs)
        _, error = parse_action(action.to_payload())
        assert error is None, (ACTION_LABELS[i], error)


def test_alias_macro_maps_leftover_spellings_only_when_there_are_some():
    env = LocalEnv()
    obs = env.reset("hard-clean", 0)
    obs = env.step(DataCleaningAction(action_type="strip_whitespace"))
    obs = env.step(DataCleaningAction(action_type="standardize_categories", column_name="city"))
    macro = build_action(ACTION_LABELS.index("standardize_aliases:city"), obs)
    assert macro.mapping and set(macro.mapping.values()) <= {"New York", "Los Angeles", "Chicago"}
    after = env.step(macro)
    assert after.last_action_ok and after.reward > 0
    again = build_action(ACTION_LABELS.index("standardize_aliases:city"), after)
    assert again.mapping is None  # nothing left to map


def test_numpy_policy_forward_and_agent_runs_an_episode(tmp_path):
    rng = np.random.default_rng(0)
    weights = {
        "W0": rng.normal(size=(8, N_FEATURES)).astype(np.float32),
        "b0": np.zeros(8, np.float32),
        "Wa": rng.normal(size=(N_ACTIONS, 8)).astype(np.float32),
        "ba": np.zeros(N_ACTIONS, np.float32),
    }
    path = tmp_path / "p.npz"
    np.savez(path, **weights)
    policy = NumpyPolicy.load(path)
    x = rng.random(N_FEATURES).astype(np.float32)
    expected = np.tanh(x @ weights["W0"].T) @ weights["Wa"].T
    assert np.allclose(policy.logits(x), expected, atol=1e-5)
    result = run_episode(RLAgent(model_path=path), LocalEnv(), "easy-clean", 0)
    assert 0.0 <= result.final_score <= 1.0 and result.steps <= TASKS["easy-clean"].max_steps


def test_rl_agent_unavailable_without_a_model(tmp_path):
    assert RLAgent.available(tmp_path / "missing.npz") is False


# ---------------------------------------------------------------- needs the optional [rl] extra
gym = pytest.importorskip("gymnasium")


def test_gym_wrapper_api_and_held_out_seed_ranges():
    from agents.rl_env import TEST_SEEDS, TRAIN_SEED_MIN, VALIDATION_SEEDS, DataCleaningGymEnv

    assert max(TEST_SEEDS) < min(VALIDATION_SEEDS) < max(VALIDATION_SEEDS) < TRAIN_SEED_MIN  # disjoint splits
    env = DataCleaningGymEnv()
    obs, info = env.reset(seed=0, options={"task_id": "easy-clean", "seed": 3})
    assert env.observation_space.contains(obs) and info["task_id"] == "easy-clean"
    obs, reward, terminated, truncated, info = env.step(ACTION_LABELS.index("fill:age:median"))
    assert env.observation_space.contains(obs) and isinstance(reward, float) and not truncated
    done = False
    while not done:
        _, _, done, _, info = env.step(FINISH_INDEX)
    assert 0.0 <= info["final_score"] <= 1.0 and "improvement" in info


def test_gym_training_episodes_use_training_seeds_only():
    from agents.rl_env import TRAIN_SEED_MIN, DataCleaningGymEnv

    env = DataCleaningGymEnv()
    seeds = {env.reset(seed=i)[1]["seed"] for i in range(20)}
    assert min(seeds) >= TRAIN_SEED_MIN


def test_tiny_smoke_train_and_numpy_export_matches_sb3(tmp_path):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from agents.rl_env import DataCleaningGymEnv
    from train_rl import export_policy

    env = DataCleaningGymEnv()
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, seed=0, device="cpu", verbose=0)
    model.learn(total_timesteps=256)
    export_policy(model, tmp_path / "p.npz")
    policy = NumpyPolicy.load(tmp_path / "p.npz")
    obs, _ = env.reset(seed=0, options={"task_id": "medium-clean", "seed": 1})
    for _ in range(5):
        sb3_action, _ = model.predict(obs, deterministic=True)
        assert policy.act(obs) == int(sb3_action)
        obs, _, done, _, _ = env.step(int(sb3_action))
        if done:
            break
