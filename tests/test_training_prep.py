"""GPU training prep: configs, committed datasets, reward function and prompt fidelity (fast; no model training)."""

import json
from pathlib import Path

import numpy as np
import pytest

from training.common import DATASETS, apply_smoke, load_config, read_jsonl

ROOT = Path(__file__).parents[1]
CONFIGS = sorted((ROOT / "training" / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_configs_load_and_have_smoke_section(path):
    cfg = load_config(str(path))
    assert "smoke" in cfg, f"{path.name} needs a smoke section for the CPU smoke test"
    assert isinstance(apply_smoke(dict(cfg)), dict)


def test_config_overrides_are_typed():
    cfg = load_config(
        str(ROOT / "training/configs/sft.yaml"), ["train.learning_rate=1e-5", "lora.r=8", "model.load_in_4bit=true"]
    )
    assert cfg["train"]["learning_rate"] == 1e-5 and cfg["lora"]["r"] == 8 and cfg["model"]["load_in_4bit"] is True


def test_committed_datasets_match_manifest():
    from training.verify_datasets import verify

    assert verify(DATASETS) == []
    manifest = json.loads((DATASETS / "manifest.json").read_text(encoding="utf-8"))
    kinds = {f["kind"] for f in manifest["files"]}
    assert kinds == {"seeds", "sft", "grpo", "tabular"}


def test_seed_ranges_never_touch_test_seeds():
    seeds = json.loads((DATASETS / "env_seeds.json").read_text(encoding="utf-8"))
    test = set(range(*seeds["test"]))
    assert test == set(range(10))
    ppo_validation = set(range(50, 60))
    for name in ("val", "train", "grpo_val", "grpo_train"):
        r = set(range(*seeds[name]))
        assert not r & test and not r & ppo_validation, name
    rows = read_jsonl(DATASETS / "sft" / "val.jsonl.gz")[:200]
    assert all(r["seed"] not in test for r in rows)


def test_sft_prompts_are_exactly_what_the_llm_agent_sends():
    from agents.llm_agent import SYSTEM_PROMPT, format_observation
    from data_cleaning_env.server.data_cleaning_env_environment import DataCleaningEnvironment
    from training.build_datasets import trajectory

    rec = next(trajectory("hard-clean", 123, 0, np.random.default_rng(0)))
    obs = DataCleaningEnvironment().reset(task_id="hard-clean", seed=123)
    assert rec["prompt"][0]["content"] == SYSTEM_PROMPT
    assert rec["prompt"][1]["content"] == format_observation(obs, [])
    assert json.loads(rec["completion"][0]["content"])["action_type"] == "drop_duplicates"


def test_grpo_reward_uses_the_environment():
    from training.rewards import INVALID_REWARD, env_reward, format_reward

    rows = read_jsonl(DATASETS / "grpo" / "val.jsonl.gz")[:5]
    sft = {(r["task_id"], r["seed"], r["prefix"]): r for r in read_jsonl(DATASETS / "sft" / "val.jsonl.gz")}
    good = '{"action_type":"drop_duplicates"}'
    rewards = env_reward(
        completions=[[{"role": "assistant", "content": "not json"}], good, '{"action_type":"fly"}'],
        task_id=["hard-clean"] * 3,
        seed=[123] * 3,
        prefix=["[]"] * 3,
    )
    assert rewards[0] == INVALID_REWARD and rewards[2] == INVALID_REWARD and rewards[1] > 0
    assert format_reward([good, "Sure! " + good]) == [0.1, 0.0]
    assert rows and all({"prompt", "task_id", "seed", "prefix"} <= set(r) for r in rows)
    assert sft  # val SFT records exist for reward sanity checks on the GPU machine


def test_runbook_lists_every_script():
    text = (ROOT / "training" / "GPU_RUNBOOK.md").read_text(encoding="utf-8")
    for script in ROOT.joinpath("training").glob("*.py"):
        if script.stem not in ("__init__", "common", "rewards", "build_datasets", "verify_datasets"):
            assert f"training.{script.stem}" in text, script.stem
