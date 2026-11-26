"""Shared helpers for the training scripts: config loading, seeding, devices, run folders, manifests, smoke models."""

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import platform
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
DATASETS = TRAINING / "datasets"
RUNS = TRAINING / "runs"

logger = logging.getLogger("training")


def setup(level: str = "INFO") -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("data_cleaning_env").setLevel(logging.WARNING)
    logging.getLogger("featureforge").setLevel(logging.WARNING)


def base_parser(description: str, default_config: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(TRAINING / "configs" / default_config))
    p.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override config values, e.g. --set train.learning_rate=1e-4 model.name=Qwen/Qwen2.5-1.5B-Instruct",
    )
    p.add_argument("--smoke", action="store_true", help="1-2 steps on CPU with a tiny model/budget to prove it runs")
    p.add_argument("--run-name", default=None)
    return p


def load_config(path: str, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        value = yaml.safe_load(raw)
        if isinstance(value, str):  # YAML 1.1 reads "1e-5" as text; numbers given on the command line are numbers
            try:
                value = float(value)
            except ValueError:
                pass
        node[parts[-1]] = value
    return cfg


def apply_smoke(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge the config's ``smoke:`` section over the rest (nested keys)."""
    smoke = cfg.pop("smoke", {}) or {}

    def merge(a: Dict[str, Any], b: Dict[str, Any]) -> None:
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v

    merge(cfg, smoke)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def run_dir(kind: str, name: Optional[str] = None) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = RUNS / kind / (name or stamp)
    d.mkdir(parents=True, exist_ok=True)
    return d


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    for mod in ("transformers", "trl", "peft", "stable_baselines3", "optuna", "sklearn"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            pass
    return info


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(data, indent=2, default=str) + "\n").encode("utf-8"))  # LF on every OS (checksums)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    import gzip

    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    # mtime=0 keeps the gzip bytes (and so the checksum) identical across rebuilds
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for r in rows:
            gz.write((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
            n += 1
    return n


def tiny_causal_lm(out_dir: Path, corpus: List[str]) -> Path:
    """Build a tiny randomly initialised Llama-style model + tokenizer *offline* (for CPU smoke tests only).

    The tokenizer is a small BPE trained on ``corpus`` and has a chat template, so the real scripts run unchanged.
    It learns nothing useful; it only proves the training code path works.
    """
    if (out_dir / "config.json").exists():
        return out_dir
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=2000,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(corpus, trainer)
    from tokenizers import decoders

    tok.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>"
    )
    fast.chat_template = (
        "{% for m in messages %}<s>{{ m['role'] }}\n"
        "{% if m['role'] == 'assistant' %}{% generation %}{{ m['content'] }}</s>{% endgeneration %}"
        "{% else %}{{ m['content'] }}</s>{% endif %}\n{% endfor %}"
        "{% if add_generation_prompt %}<s>assistant\n{% endif %}"
    )
    cfg = LlamaConfig(
        vocab_size=len(fast),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        bos_token_id=fast.bos_token_id,
        eos_token_id=fast.eos_token_id,
        pad_token_id=fast.pad_token_id,
    )
    model = LlamaForCausalLM(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    fast.save_pretrained(out_dir)
    return out_dir
