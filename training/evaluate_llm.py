"""Evaluate a local (fine-tuned) LLM as the data-cleaning agent on held-out test seeds, like ``benchmark.py``.

    python -m training.evaluate_llm --config training/configs/eval_llm.yaml --set adapter=training/runs/sft/<run>/adapter
    python -m training.evaluate_llm --set adapter=null                       # zero-shot base model
    python -m training.evaluate_llm --smoke                                  # tiny offline model, 1 episode

The agent uses the *same* prompt, validation and retry logic as ``agents.LLMAgent`` (only the text generation is
local instead of an API call), so numbers are comparable with the README table. Test seeds (default 0-9) are never
used for training or model selection. Output: ``training/runs/eval_llm/<run>/{results.md,episodes.csv,run.json}``.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

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

logger = logging.getLogger("training.eval_llm")


class LocalChatClient:
    """Minimal OpenAI-compatible ``chat.completions.create`` backed by a local transformers model."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 128) -> None:
        self.model, self.tok, self.max_new_tokens = model, tokenizer, max_new_tokens
        self.chat = self
        self.completions = self

    def create(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 400):
        import torch

        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = {k: v.to(self.model.device) for k, v in ids.items()}
        with torch.no_grad():
            out = self.model.generate(
                **ids,
                max_new_tokens=min(max_tokens, self.max_new_tokens),
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tok.pad_token_id,
            )
        text = self.tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)

        class _Msg:
            content = text

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


def load_model(cfg: Dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from training.train_sft import model_kwargs, resolve_model

    name = resolve_model(cfg, Path(cfg["data_dir"]) if cfg.get("data_dir") else DATASETS)
    tok = AutoTokenizer.from_pretrained(cfg.get("adapter") or name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    quantized = bool(cfg["model"].get("load_in_4bit"))
    model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs(cfg["model"], bf16))
    if cfg.get("adapter"):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, cfg["adapter"])
        if not quantized:  # merging into 4-bit weights is lossy; keep the adapter separate then
            model = model.merge_and_unload()
    if quantized:  # already placed on the GPU by device_map
        return model.eval(), tok, name
    device = "cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu"
    return model.to(device).eval(), tok, name


def main() -> int:
    args = base_parser(__doc__, "eval_llm.yaml").parse_args()
    setup()
    cfg = load_config(args.config, args.set)
    if args.smoke:
        cfg = apply_smoke(cfg)
    else:
        cfg.pop("smoke", None)
    seed_everything(cfg.get("seed", 0))
    from agents import LLMAgent, LocalEnv, run_episode

    model, tok, base = load_model(cfg)
    client = LocalChatClient(model, tok, cfg.get("max_new_tokens", 128))
    label = cfg.get("label") or (Path(cfg["adapter"]).parent.name if cfg.get("adapter") else base.split("/")[-1])
    out = run_dir("eval_llm_smoke" if args.smoke else "eval_llm", args.run_name)
    env = LocalEnv()
    rows = []
    for task in cfg["tasks"]:
        for seed in cfg["seeds"]:
            agent = LLMAgent(client=client, model=label, max_retries=cfg.get("max_retries", 2), api_retries=0)
            res = run_episode(agent, env, task, seed).as_dict()
            rows.append(res)
            logger.info(
                "%s seed %d: score %.3f improvement %+.3f invalid %d",
                task,
                seed,
                res["final_score"],
                res["improvement"],
                res["invalid_actions"],
            )
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(out / "episodes.csv", index=False)
    lines = [
        f"# LLM agent evaluation: {label}",
        "",
        f"Base `{base}`"
        + (f", adapter `{cfg['adapter']}`" if cfg.get("adapter") else " (no adapter)")
        + f"; seeds {cfg['seeds']}; greedy decoding.",
        "",
        "| task | score (mean ± std) | improvement | invalid-action rate | fallbacks |",
        "|---|---|---|---|---|",
    ]
    for task, g in df.groupby("task_id", sort=False):
        rate = g["invalid_actions"].sum() / max(1, g["steps"].sum())
        lines.append(
            f"| {task} | {g['final_score'].mean():.3f} ± {g['final_score'].std(ddof=0):.3f} | "
            f"{g['improvement'].mean():+.3f} | {100 * rate:.1f}% | {int(g['fallbacks'].sum())} |"
        )
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(
        out / "run.json",
        {
            "config": cfg,
            "base_model": base,
            "environment": environment_info(),
            "mean_score": float(np.mean(df["final_score"])),
        },
    )
    logger.info("\n%s", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
