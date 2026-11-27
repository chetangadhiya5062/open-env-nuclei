"""PPO on the data-cleaning environment, driven by a YAML config (GPU-machine version of ``train_rl.py``).

    python -m training.train_ppo --config training/configs/ppo.yaml
    python -m training.train_ppo --set ppo.learning_rate=1e-4 total_timesteps=2000000 --run-name lr1e-4
    python -m training.train_ppo --smoke              # ~2 rollouts on CPU, proves the script runs

Seeds follow the repo's protocol: training episodes use env seeds >= 100, checkpoints are chosen on validation seeds
50-59, and test seeds 0-9 are never touched here (evaluate with ``training/evaluate_ppo.py``).

Outputs in ``training/runs/ppo/<run>/``: ``ppo_policy.npz`` (numpy weights loadable by ``agents.RLAgent``),
``ppo_policy.json`` (config, history, environment info), ``learning_curve.png``, ``sb3_model.zip``.
Note: a 64x64 MLP policy is CPU-bound; a GPU helps little here. More env workers (``n_envs``) help most.
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

from training.common import (
    apply_smoke,
    base_parser,
    environment_info,
    load_config,
    pick_device,
    run_dir,
    save_json,
    seed_everything,
    setup,
)

logger = logging.getLogger("training.ppo")


def make_env_fn(seed_range):
    def _make():
        import logging as _logging

        from stable_baselines3.common.monitor import Monitor

        from agents.rl_env import DataCleaningGymEnv

        _logging.getLogger("data_cleaning_env").setLevel(_logging.WARNING)
        return Monitor(DataCleaningGymEnv(seed_range=tuple(seed_range)))

    return _make


def train(cfg: Dict[str, Any], out: Path, report: Callable[[int, float], bool] = lambda t, s: False) -> Dict[str, Any]:
    """Train one PPO model. ``report(timesteps, mean_val_improvement)`` returning True stops early (Optuna pruning)."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    from agents.rl_env import VALIDATION_SEEDS
    from train_rl import evaluate, export_policy, plot_curve

    seed_everything(cfg["seed"])
    hp = cfg["ppo"]
    device = pick_device(cfg.get("device", "auto"))
    vec_cls = SubprocVecEnv if cfg.get("subprocess", True) and cfg["n_envs"] > 1 else DummyVecEnv
    venv = vec_cls([make_env_fn(cfg["train_seed_range"]) for _ in range(cfg["n_envs"])])
    venv.seed(cfg["seed"])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=hp["gamma"])
    net = hp.get("net_arch", [64, 64])
    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=hp["n_steps"],
        batch_size=hp["batch_size"],
        n_epochs=hp["n_epochs"],
        gamma=hp["gamma"],
        gae_lambda=hp["gae_lambda"],
        clip_range=hp.get("clip_range", 0.2),
        ent_coef=hp["ent_coef"],
        vf_coef=hp.get("vf_coef", 0.5),
        learning_rate=hp["learning_rate"],
        max_grad_norm=hp.get("max_grad_norm", 0.5),
        policy_kwargs=dict(
            net_arch=dict(pi=net, vf=net), activation_fn=getattr(torch.nn, hp.get("activation", "Tanh"))
        ),
        seed=cfg["seed"],
        device=device,
        verbose=0,
    )
    val_seeds = tuple(cfg.get("validation_seeds", VALIDATION_SEEDS))
    history: List[Dict[str, Any]] = []
    best = {"improvement": -1e9, "timesteps": 0}
    started = time.time()
    stop = {"flag": False}

    def validate(t: int) -> None:
        def predict(x):
            action, _ = model.policy.predict(np.asarray(x)[None, :], deterministic=True)
            return int(action[0])

        metrics = evaluate(predict, seeds=val_seeds)
        mean_impr = float(np.mean([m["improvement"] for m in metrics.values()]))
        history.append({"timesteps": t, "metrics": metrics, "mean_improvement": mean_impr})
        logger.info("t=%8d  mean val improvement %.4f  (%.0fs)", t, mean_impr, time.time() - started)
        if mean_impr > best["improvement"]:
            best.update(improvement=mean_impr, timesteps=t)
            export_policy(model, out / "ppo_policy.npz")
            model.save(out / "sb3_model.zip")
        stop["flag"] = report(t, mean_impr)

    class Cb(BaseCallback):
        def __init__(self):
            super().__init__()
            self._next = 0

        def _on_step(self) -> bool:
            if self.num_timesteps >= self._next:
                validate(self.num_timesteps)
                self._next += cfg["eval_every"]
            return not stop["flag"]

    model.learn(total_timesteps=cfg["total_timesteps"], callback=Cb())
    if not stop["flag"]:
        validate(model.num_timesteps)
    venv.close()
    if cfg.get("plot", True) and history:
        from train_rl import baseline_validation

        plot_curve(
            history, baseline_validation() if cfg.get("plot_baselines", True) else {}, out / "learning_curve.png"
        )
    meta = {
        "config": cfg,
        "device": device,
        "timesteps": int(model.num_timesteps),
        "best_checkpoint_timesteps": best["timesteps"],
        "best_validation_mean_improvement": best["improvement"],
        "train_seconds": round(time.time() - started, 1),
        "validation_seeds": list(val_seeds),
        "environment": environment_info(),
        "history": history,
    }
    save_json(out / "ppo_policy.json", meta)
    return meta


def main() -> int:
    args = base_parser(__doc__, "ppo.yaml").parse_args()
    setup()
    cfg = load_config(args.config, args.set)
    if args.smoke:
        cfg = apply_smoke(cfg)
    else:
        cfg.pop("smoke", None)
    out = run_dir("ppo_smoke" if args.smoke else "ppo", args.run_name)
    meta = train(cfg, out)
    logger.info(
        "best mean validation improvement %.4f at t=%d -> %s",
        meta["best_validation_mean_improvement"],
        meta["best_checkpoint_timesteps"],
        out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
