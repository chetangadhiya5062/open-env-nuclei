"""Score FeatureForge's one-shot generic cleaning with the environment's grader on held-out test seeds.

python scripts/eval_library_on_env.py --seeds 10 --out docs/benchmarks/library_on_env.md
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from data_cleaning_env.server.datagen import TASKS
from featureforge.envbridge import clean_and_grade


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10, help="test seeds 0..N-1 (same as benchmark.py)")
    ap.add_argument("--out", default="docs/benchmarks/library_on_env.md")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    lines = [
        "# FeatureForge generic cleaning on the OpenEnv tasks",
        "",
        f"`python scripts/eval_library_on_env.py --seeds {args.seeds}` - one-shot recipe (no agent loop), graded by",
        "`data_cleaning_env/server/grader.py` on test seeds 0..N-1. The recipe knows expected types and valid ranges",
        "(both visible to agents) but not the allowed category vocabulary.",
        "",
        "| task | score (mean ± std) | improvement | do-nothing |",
        "|---|---|---|---|",
    ]
    for task in TASKS:
        rs = [clean_and_grade(task, s) for s in range(args.seeds)]
        sc = np.array([r["score"] for r in rs])
        im = np.array([r["improvement"] for r in rs])
        dn = np.array([r["do_nothing_score"] for r in rs])
        lines.append(f"| {task} | {sc.mean():.3f} ± {sc.std():.3f} | {im.mean():+.3f} | {dn.mean():.3f} |")
        print(lines[-1])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
