"""Build a clean deploy folder and upload it to the Hugging Face Space.

    python scripts/deploy_space.py --dry-run     # only build the folder and list what would be uploaded
    python scripts/deploy_space.py               # upload (needs `hf auth login` first)

Login is NOT done here and tokens are never read from .env: run ``hf auth login`` yourself (paste a WRITE token
into that prompt). The Space README.md is ``data_cleaning_env/README.md`` (it holds the YAML Space card).
Excluded: .venv, caches, tests, docs, .git, .github, .env, training/benchmark outputs.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "chetangadhiya017/data-cleaning-env"

# What the Space needs to build and run (everything else stays on GitHub).
INCLUDE_DIRS = ["data_cleaning_env", "agents", "models"]
INCLUDE_FILES = ["Dockerfile", "requirements.txt", "openenv.yaml", ".dockerignore"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", ".pytest_cache", ".ruff_cache", "README.md")


def build_deploy_folder(dest: Path) -> list:
    """Copy the deployable files into ``dest`` and return the sorted list of relative paths."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in INCLUDE_DIRS:
        src = ROOT / name
        if src.exists():
            shutil.copytree(src, dest / name, ignore=IGNORE)
    for name in INCLUDE_FILES:
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, dest / name)
    # the Space card (YAML front-matter) becomes the Space's own README.md
    shutil.copy2(ROOT / "data_cleaning_env" / "README.md", dest / "README.md")
    return sorted(str(p.relative_to(dest)).replace("\\", "/") for p in dest.rglob("*") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Space id, e.g. user/space-name")
    parser.add_argument("--dry-run", action="store_true", help="build the folder and list files; upload nothing")
    parser.add_argument("--message", default="Deploy v3: seeded tasks, grader, agents (incl. RL), Gradio demo")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="space_deploy_") as tmp:
        folder = Path(tmp) / "space"
        files = build_deploy_folder(folder)
        total_kb = sum((folder / f).stat().st_size for f in files) / 1024
        print(f"Deploy folder: {len(files)} files, {total_kb:.0f} kB")
        for f in files:
            print("  ", f)
        if args.dry_run:
            print("Dry run: nothing uploaded.")
            return 0

        from huggingface_hub import HfApi

        api = HfApi()
        try:
            who = api.whoami()["name"]
        except Exception:
            print(
                "Not logged in to Hugging Face. Run this yourself and paste a WRITE token at the prompt:\n  hf auth login"
            )
            return 1
        print(f"Logged in as {who}; uploading to Space {args.repo} ...")
        api.upload_folder(
            repo_id=args.repo,
            repo_type="space",
            folder_path=str(folder),
            commit_message=args.message,
            # remove files from the previous (pre-v2) deployment that no longer exist locally
            delete_patterns=["app_ui.py", "data_cleaning_env/*", "agents/*"],
        )
        print(f"Done. https://huggingface.co/spaces/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
