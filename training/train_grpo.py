"""GRPO (group-relative policy optimisation) of the LLM agent, rewarded by the environment itself.

    python -m training.train_grpo --config training/configs/grpo.yaml --set model.adapter=training/runs/sft/<run>/adapter
    python -m training.train_grpo --smoke          # tiny offline model, 1-2 steps on CPU

Recommended order: SFT first (``train_sft``), then GRPO starting from the SFT adapter (``model.adapter``), so that
sampled actions are mostly valid and the group advantages carry signal.

For each prompt (a real environment state from ``training/datasets/grpo``) the trainer samples ``num_generations``
replies; each reply is scored by ``training.rewards.env_reward`` (rebuild the state, apply the action, return the
environment's step reward; invalid reply -> -1) plus ``format_reward``. Output: ``training/runs/grpo/<run>/adapter``.
"""

import logging
from pathlib import Path
from typing import Any, Dict

from training.common import (
    DATASETS,
    apply_smoke,
    base_parser,
    environment_info,
    load_config,
    run_dir,
    save_json,
    seed_everything,
    setup,
)

logger = logging.getLogger("training.grpo")


def main() -> int:
    args = base_parser(__doc__, "grpo.yaml").parse_args()
    setup()
    cfg = load_config(args.config, args.set)
    if args.smoke:
        cfg = apply_smoke(cfg)
    else:
        cfg.pop("smoke", None)
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from training.rewards import env_reward, format_reward
    from training.train_sft import resolve_model

    seed_everything(cfg["seed"])
    data_dir = Path(cfg["data"]["dir"]) if cfg["data"].get("dir") else DATASETS
    out = run_dir("grpo_smoke" if args.smoke else "grpo", args.run_name)
    sft_data_dir = data_dir if (data_dir / "sft").exists() else DATASETS
    model_name = resolve_model(cfg, sft_data_dir)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tcfg: Dict[str, Any] = dict(cfg["train"])
    use_cpu = bool(tcfg.pop("use_cpu", False)) or not torch.cuda.is_available()
    bf16 = (not use_cpu) and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16 if bf16 else torch.float32)
    peft_config = None
    if cfg["model"].get("adapter"):
        model = PeftModel.from_pretrained(model, cfg["model"]["adapter"], is_trainable=True)
        logger.info("continuing from adapter %s", cfg["model"]["adapter"])
    elif cfg["lora"].pop("enabled", True):
        peft_config = LoraConfig(task_type="CAUSAL_LM", **cfg["lora"])

    files = {"train": str(data_dir / "grpo" / "train.jsonl.gz")}
    ds = load_dataset("json", data_files=files)["train"].select_columns(["prompt", "task_id", "seed", "prefix"])
    if cfg["data"].get("limit_train"):
        ds = ds.shuffle(seed=0).select(range(min(cfg["data"]["limit_train"], len(ds))))

    gargs = GRPOConfig(
        output_dir=str(out / "checkpoints"),
        use_cpu=use_cpu,
        bf16=bf16,
        report_to=cfg.get("report_to", "none"),
        seed=cfg["seed"],
        save_total_limit=2,
        **tcfg,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[env_reward, format_reward],
        args=gargs,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    result = trainer.train()
    trainer.save_model(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))
    save_json(
        out / "run.json",
        {"config": cfg, "base_model": model_name, "train_metrics": result.metrics, "environment": environment_info()},
    )
    logger.info("done in %.0fs -> %s", result.metrics.get("train_runtime", 0), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
