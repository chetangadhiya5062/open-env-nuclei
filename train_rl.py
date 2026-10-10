"""Train a PPO agent on the data-cleaning environment (CPU, a few minutes to ~30 min).

    pip install -e ".[rl]"
    python train_rl.py                       # defaults: 400k steps, seed 0
    python train_rl.py --timesteps 50000     # quick run

Seeds: training episodes use environment seeds >= 100 (random task and seed every episode); checkpoints are
selected on validation seeds 50-59; the final benchmark uses held-out test seeds 0-9 (never seen in training).

Outputs: ``models/ppo_policy.npz`` (numpy weights used by ``agents.RLAgent``), ``models/ppo_policy.json``
(hyper-parameters, seeds, validation history) and ``docs/benchmarks/learning_curve.png``.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from agents import DoNothingAgent, LocalEnv, RandomAgent, RuleBasedAgent, run_episode
from agents.rl_env import VALIDATION_SEEDS, DataCleaningGymEnv
from data_cleaning_env.server.datagen import TASKS

logger = logging.getLogger("train_rl")

HYPERPARAMS = dict(
    n_envs=4,
    n_steps=512,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.02,
    learning_rate=3e-4,
    net_arch=[64, 64],
)


def evaluate(predict, seeds=VALIDATION_SEEDS) -> Dict[str, Dict[str, float]]:
    """Run ``predict(feature_vector) -> action_index`` greedily on fixed seeds. Returns per-task mean metrics."""
    env = DataCleaningGymEnv()
    out: Dict[str, Dict[str, float]] = {}
    for task_id in TASKS:
        scores, imps, rewards = [], [], []
        for seed in seeds:
            obs, _ = env.reset(options={"task_id": task_id, "seed": seed})
            done, total = False, 0.0
            while not done:
                obs, reward, done, _, info = env.step(predict(obs))
                total += reward
            scores.append(info["final_score"])
            imps.append(info["improvement"])
            rewards.append(total)
        out[task_id] = {
            "score": float(np.mean(scores)),
            "improvement": float(np.mean(imps)),
            "reward": float(np.mean(rewards)),
        }
    return out


def baseline_validation() -> Dict[str, Dict[str, Dict[str, float]]]:
    """do-nothing / random / rule-based on the same validation seeds (reference lines for the curve)."""
    res: Dict[str, Dict[str, Dict[str, float]]] = {}
    env = LocalEnv()
    for name, agent in (("do-nothing", DoNothingAgent()), ("random", RandomAgent()), ("rule-based", RuleBasedAgent())):
        res[name] = {}
        for task_id in TASKS:
            runs = [run_episode(agent, env, task_id, s) for s in VALIDATION_SEEDS]
            res[name][task_id] = {
                "score": float(np.mean([r.final_score for r in runs])),
                "improvement": float(np.mean([r.improvement for r in runs])),
            }
    return res


def export_policy(model, path: Path) -> None:
    net = model.policy.mlp_extractor.policy_net
    linears = [m for m in net if m.__class__.__name__ == "Linear"]
    weights = {}
    for i, layer in enumerate(linears):
        weights[f"W{i}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        weights[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    weights["Wa"] = model.policy.action_net.weight.detach().cpu().numpy().astype(np.float32)
    weights["ba"] = model.policy.action_net.bias.detach().cpu().numpy().astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **weights)


def plot_curve(history: List[dict], baselines: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(TASKS), figsize=(13, 3.8), sharey=False)
    steps = [h["timesteps"] for h in history]
    colors = {"do-nothing": "tab:gray", "random": "tab:red", "rule-based": "tab:green"}
    for ax, task_id in zip(axes, TASKS, strict=True):
        ax.plot(
            steps,
            [h["metrics"][task_id]["score"] for h in history],
            color="tab:blue",
            marker="o",
            ms=3,
            label="PPO (validation)",
        )
        for name, per_task in baselines.items():
            ax.axhline(per_task[task_id]["score"], color=colors[name], ls="--", lw=1.2, label=name)
        ax.set_title(task_id)
        ax.set_xlabel("training steps")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("grader score (validation seeds 50-59)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("PPO learning curve (held-out validation seeds, greedy policy)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=20_480, help="timesteps between validation runs")
    parser.add_argument("--out", default="models/ppo_policy.npz")
    parser.add_argument("--curve", default="docs/benchmarks/learning_curve.png")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    set_random_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    hp = HYPERPARAMS
    venv = DummyVecEnv([lambda: Monitor(DataCleaningGymEnv()) for _ in range(hp["n_envs"])])
    venv.seed(args.seed)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=hp["gamma"])
    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=hp["n_steps"],
        batch_size=hp["batch_size"],
        n_epochs=hp["n_epochs"],
        gamma=hp["gamma"],
        gae_lambda=hp["gae_lambda"],
        ent_coef=hp["ent_coef"],
        learning_rate=hp["learning_rate"],
        policy_kwargs=dict(net_arch=dict(pi=hp["net_arch"], vf=hp["net_arch"])),
        seed=args.seed,
        device="cpu",
        verbose=0,
    )

    history: List[dict] = []
    best = {"improvement": -1e9, "weights": None, "timesteps": 0}
    started = time.time()

    def validate(timesteps: int) -> None:
        def predict(x):
            obs_t = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
            action, _ = model.policy.predict(obs_t.numpy(), deterministic=True)
            return int(action[0])

        metrics = evaluate(predict)
        mean_impr = float(np.mean([m["improvement"] for m in metrics.values()]))
        history.append({"timesteps": timesteps, "metrics": metrics, "mean_improvement": mean_impr})
        logger.info(
            "t=%7d  val score easy/med/hard = %s  mean improvement %.3f  (%.0fs)",
            timesteps,
            "/".join(f"{metrics[t]['score']:.3f}" for t in TASKS),
            mean_impr,
            time.time() - started,
        )
        if mean_impr > best["improvement"]:
            best.update(improvement=mean_impr, timesteps=timesteps)
            export_policy(model, Path(args.out))

    class ValidationCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self._next = 0

        def _on_step(self) -> bool:
            if self.num_timesteps >= self._next:
                validate(self.num_timesteps)
                self._next += args.eval_every
            return True

    model.learn(total_timesteps=args.timesteps, callback=ValidationCallback())
    validate(model.num_timesteps)
    train_seconds = time.time() - started
    logger.info("best checkpoint: t=%d mean validation improvement %.3f", best["timesteps"], best["improvement"])

    baselines = baseline_validation()
    plot_curve(history, baselines, Path(args.curve))
    meta = {
        "algorithm": "PPO (stable-baselines3)",
        "timesteps": int(model.num_timesteps),
        "seed": args.seed,
        "hyperparameters": hp,
        "train_seed_range": [100, 100_000],
        "validation_seeds": list(VALIDATION_SEEDS),
        "best_checkpoint_timesteps": best["timesteps"],
        "best_validation_mean_improvement": best["improvement"],
        "train_seconds": round(train_seconds, 1),
        "torch": torch.__version__,
        "history": history,
        "baselines_validation": baselines,
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("wrote %s, %s, %s", args.out, Path(args.out).with_suffix(".json"), args.curve)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
