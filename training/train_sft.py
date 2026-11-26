"""Supervised fine-tuning (LoRA) of a small instruct LLM to act as the data-cleaning agent.

    python -m training.train_sft --config training/configs/sft.yaml
    python -m training.train_sft --set model.name=Qwen/Qwen2.5-7B-Instruct model.load_in_4bit=true --run-name q7b
    python -m training.train_sft --smoke          # tiny offline model, 2 steps on CPU

Data: ``training/datasets/sft/{train,val}.jsonl.gz`` (from ``training.build_datasets``): the prompt is exactly what
``agents.LLMAgent`` sends, the completion is the expert action. Loss is computed on the completion only.

Output: ``training/runs/sft/<run>/adapter`` (LoRA adapter + tokenizer) and ``run.json``. Evaluate in the real
environment with ``python -m training.evaluate_llm --adapter training/runs/sft/<run>/adapter``.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from training.common import (
    DATASETS,
    RUNS,
    apply_smoke,
    base_parser,
    environment_info,
    load_config,
    read_jsonl,
    run_dir,
    save_json,
    seed_everything,
    setup,
    tiny_causal_lm,
)

logger = logging.getLogger("training.sft")


def load_splits(data_dir: Path, limit_train: Optional[int] = None, limit_val: Optional[int] = None):
    from datasets import load_dataset

    files = {"train": str(data_dir / "sft" / "train.jsonl.gz"), "validation": str(data_dir / "sft" / "val.jsonl.gz")}
    ds = load_dataset("json", data_files=files)
    ds = ds.select_columns(["prompt", "completion"])
    if limit_train:
        ds["train"] = ds["train"].shuffle(seed=0).select(range(min(limit_train, len(ds["train"]))))
    if limit_val:
        ds["validation"] = ds["validation"].select(range(min(limit_val, len(ds["validation"]))))
    return ds


def resolve_model(cfg: Dict[str, Any], data_dir: Path) -> str:
    name = cfg["model"]["name"]
    if name != "tiny-offline":
        return name
    rows = read_jsonl(data_dir / "sft" / "train.jsonl.gz")
    corpus = [m["content"] for r in rows for m in r["prompt"] + r["completion"]]
    return str(tiny_causal_lm(RUNS / "smoke_model", corpus))


def build_trainer(cfg: Dict[str, Any], out: Path, data_dir: Path, trial_params: Optional[Dict[str, Any]] = None):
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    seed_everything(cfg["seed"])
    tcfg = dict(cfg["train"])
    lcfg = dict(cfg["lora"])
    if trial_params:
        for k, v in trial_params.items():
            (lcfg if k.startswith("lora_") else tcfg)[k.removeprefix("lora_") if k.startswith("lora_") else k] = v
    model_name = resolve_model(cfg, data_dir)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_cpu = bool(tcfg.pop("use_cpu", False)) or not torch.cuda.is_available()
    bf16 = (not use_cpu) and torch.cuda.is_bf16_supported()
    kwargs: Dict[str, Any] = {"dtype": torch.bfloat16 if bf16 else torch.float32}
    if cfg["model"].get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    if cfg["model"].get("attn_implementation"):
        kwargs["attn_implementation"] = cfg["model"]["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    ds = load_splits(data_dir, cfg["data"].get("limit_train"), cfg["data"].get("limit_val"))
    args = SFTConfig(
        output_dir=str(out / "checkpoints"),
        use_cpu=use_cpu,
        bf16=bf16,
        report_to=cfg.get("report_to", "none"),
        seed=cfg["seed"],
        completion_only_loss=True,
        save_total_limit=2,
        **tcfg,
    )
    peft_config = LoraConfig(task_type="CAUSAL_LM", **lcfg) if lcfg.pop("enabled", True) else None
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tok,
        peft_config=peft_config,
    )
    return trainer, tok, model_name


def main() -> int:
    args = base_parser(__doc__, "sft.yaml").parse_args()
    setup()
    cfg = load_config(args.config, args.set)
    if args.smoke:
        cfg = apply_smoke(cfg)
    else:
        cfg.pop("smoke", None)
    data_dir = Path(cfg["data"]["dir"]) if cfg["data"].get("dir") else DATASETS
    out = run_dir("sft_smoke" if args.smoke else "sft", args.run_name)
    trainer, tok, model_name = build_trainer(cfg, out, data_dir)
    result = trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))
    save_json(
        out / "run.json",
        {
            "config": cfg,
            "base_model": model_name,
            "train_metrics": result.metrics,
            "eval_metrics": metrics,
            "environment": environment_info(),
        },
    )
    logger.info(
        "train loss %.4f, eval loss %.4f -> %s", result.training_loss, metrics.get("eval_loss", float("nan")), out
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
