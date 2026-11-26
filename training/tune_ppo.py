"""Optuna hyper-parameter search for PPO (TPE sampler + median pruning on validation improvement).

    python -m training.tune_ppo --config training/configs/ppo_optuna.yaml
    python -m training.tune_ppo --set n_trials=60 --run-name big       # resume: same --run-name reuses the study
    python -m training.tune_ppo --smoke

Each trial trains PPO with ``trial_timesteps`` steps (from ``base_config``) and reports the mean validation
improvement (validation seeds only; test seeds are never used for selection). The study is stored in SQLite under
``training/runs/ppo_optuna/<run>/study.db`` so it can be resumed or run from several processes at once.
Outputs: ``best_params.yaml`` (ready for ``train_ppo --set ...``), ``trials.csv``.
"""

import copy
import logging
from typing import Any, Dict

import yaml

from training.common import apply_smoke, base_parser, load_config, run_dir, save_json, setup

logger = logging.getLogger("training.tune_ppo")


def suggest(trial, space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    params = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
        else:
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params


def main() -> int:
    args = base_parser(__doc__, "ppo_optuna.yaml").parse_args()
    setup()
    import optuna

    from training.train_ppo import train

    cfg = load_config(args.config, args.set)
    if args.smoke:
        cfg = apply_smoke(cfg)
    else:
        cfg.pop("smoke", None)
    out = run_dir("ppo_optuna_smoke" if args.smoke else "ppo_optuna", args.run_name or "study")
    base = load_config(cfg["base_config"])
    base.pop("smoke", None)
    base.update(cfg.get("base_overrides", {}))
    storage = f"sqlite:///{(out / 'study.db').as_posix()}"
    study = optuna.create_study(
        study_name=cfg.get("study_name", "ppo"),
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.get("seed", 0), n_startup_trials=cfg.get("startup_trials", 8)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=cfg.get("startup_trials", 8), n_warmup_steps=1),
    )

    def objective(trial) -> float:
        params = suggest(trial, cfg["search_space"])
        tcfg = copy.deepcopy(base)
        net = params.pop("net_width", None)
        if net:
            tcfg["ppo"]["net_arch"] = [net, net]
        tcfg["ppo"].update(params)
        tcfg["ppo"]["batch_size"] = min(tcfg["ppo"]["batch_size"], tcfg["ppo"]["n_steps"] * tcfg["n_envs"])
        tcfg["seed"] = cfg.get("seed", 0) + trial.number
        tdir = out / f"trial_{trial.number:03d}"
        tdir.mkdir(exist_ok=True)
        step = {"i": 0}

        def report(t: int, score: float) -> bool:
            trial.report(score, step["i"])
            step["i"] += 1
            return trial.should_prune()

        meta = train(tcfg, tdir, report)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return float(meta["best_validation_mean_improvement"])

    study.optimize(objective, n_trials=cfg["n_trials"], timeout=cfg.get("timeout_seconds"))
    complete = [t for t in study.trials if t.state.name == "COMPLETE"]
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    if complete:
        best = study.best_trial
        (out / "best_params.yaml").write_text(
            yaml.safe_dump({"value": best.value, "params": best.params}), encoding="utf-8"
        )
        bp = dict(best.params)
        width = bp.pop("net_width", None)
        set_args = [f"ppo.{k}={v}" for k, v in bp.items()] + ([f"ppo.net_arch=[{width},{width}]"] if width else [])
        save_json(
            out / "best.json", {"value": best.value, "params": best.params, "trial": best.number, "set_args": set_args}
        )
        logger.info("train the final model with: python -m training.train_ppo --set %s", " ".join(set_args))
        logger.info("best trial %d: %.4f %s", best.number, best.value, best.params)
    logger.info("%d trials (%d complete) -> %s", len(study.trials), len(complete), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
