"""Optuna search over SFT hyper-parameters (learning rate, LoRA rank/alpha/dropout), objective = validation loss.

    python -m training.tune_sft --config training/configs/sft_optuna.yaml
    python -m training.tune_sft --smoke

Every trial is a short SFT run (``trial_max_steps``) on the training split, scored on ``sft/val`` (validation env
seeds 60-89). Test seeds are never used. Then train the final model with the best values:
``python -m training.train_sft --set train.learning_rate=<lr> lora.r=<r> ...`` (printed at the end as ``set_args``).
"""

import copy
import logging
from pathlib import Path

from training.common import DATASETS, apply_smoke, base_parser, load_config, run_dir, save_json, setup
from training.tune_ppo import suggest

logger = logging.getLogger("training.tune_sft")


def main() -> int:
    args = base_parser(__doc__, "sft_optuna.yaml").parse_args()
    setup()
    import optuna

    from training.train_sft import build_trainer

    cfg = load_config(args.config, args.set)
    if args.smoke:
        cfg = apply_smoke(cfg)
    else:
        cfg.pop("smoke", None)
    base = load_config(cfg["base_config"])
    if args.smoke:
        base = apply_smoke(base)
    else:
        base.pop("smoke", None)
    for section, values in cfg.get("base_overrides", {}).items():
        base.setdefault(section, {}).update(values)
    out = run_dir("sft_optuna_smoke" if args.smoke else "sft_optuna", args.run_name or "study")
    data_dir = Path(base["data"]["dir"]) if base["data"].get("dir") else DATASETS
    study = optuna.create_study(
        study_name=cfg.get("study_name", "sft"),
        storage=f"sqlite:///{(out / 'study.db').as_posix()}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.get("seed", 0)),
    )

    def objective(trial) -> float:
        params = suggest(trial, cfg["search_space"])
        tcfg = copy.deepcopy(base)
        trainer, _, _ = build_trainer(tcfg, out / f"trial_{trial.number:03d}", data_dir, trial_params=params)
        trainer.train()
        loss = float(trainer.evaluate()["eval_loss"])
        del trainer
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        return loss

    study.optimize(objective, n_trials=cfg["n_trials"], timeout=cfg.get("timeout_seconds"))
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    best = study.best_trial
    set_args = [
        f"lora.{k.removeprefix('lora_')}={v}" if k.startswith("lora_") else f"train.{k}={v}"
        for k, v in best.params.items()
    ]
    save_json(out / "best.json", {"value": best.value, "params": best.params, "set_args": set_args})
    logger.info("best eval_loss %.4f; train the final model with: --set %s", best.value, " ".join(set_args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
