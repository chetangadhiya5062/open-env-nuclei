"""Check every dataset file against ``manifest.json`` (presence, size, sha256).

python -m training.verify_datasets            # training/datasets
python -m training.verify_datasets --dir training/runs/smoke_datasets
"""

import argparse
import json
import sys
from pathlib import Path

from training.common import DATASETS, sha256


def verify(folder: Path) -> list:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    problems = []
    for f in manifest["files"]:
        p = folder / f["path"]
        if not p.exists():
            problems.append(f"missing: {f['path']}")
        elif p.stat().st_size != f["bytes"]:
            problems.append(f"size mismatch: {f['path']}")
        elif sha256(p) != f["sha256"]:
            problems.append(f"checksum mismatch: {f['path']}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(DATASETS))
    folder = Path(ap.parse_args().dir)
    problems = verify(folder)
    for p in problems:
        print(p, file=sys.stderr)
    n = len(json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["files"])
    print(f"{n - len(problems)}/{n} files OK in {folder}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
