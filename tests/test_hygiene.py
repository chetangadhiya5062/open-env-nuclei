"""Guards for the repo hygiene issues fixed in Phase 1 (so they cannot silently come back)."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _norm_req(line: str) -> str:
    return re.sub(r"\s+", "", line.split("#")[0])


def test_requirements_is_utf8_and_matches_pyproject():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    raw = (ROOT / "requirements.txt").read_bytes()
    assert not raw.startswith((b"\xff\xfe", b"\xfe\xff")), "requirements.txt must be UTF-8, not UTF-16"
    assert b"\r\n" not in raw
    reqs = {_norm_req(line) for line in raw.decode("utf-8").splitlines() if _norm_req(line)}
    deps = {_norm_req(d) for d in tomllib.loads(_read("pyproject.toml"))["project"]["dependencies"]}
    assert reqs == deps


def test_no_prints_or_debug_emoji_in_library_code():
    for path in list((ROOT / "data_cleaning_env").rglob("*.py")) + list((ROOT / "agents").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*print\(", text, re.M), f"print() in {path}"
        assert "🔥" not in text, path


def test_single_inference_script_and_no_stray_files():
    assert (ROOT / "inference.py").exists()
    assert not (ROOT / "data_cleaning_env" / "inference.py").exists()
    assert not (ROOT / "project_details_for_portfolio.txt").exists()


def test_one_port_everywhere():
    assert "7860" in _read("Dockerfile")
    assert "port: 7860" in _read("openenv.yaml")
    assert "app_port: 7860" in _read("data_cleaning_env/README.md")
    assert "7860" in _read(".env.example")
    for text in (_read("Dockerfile"), _read("openenv.yaml"), _read("inference.py")):
        assert "8001" not in text and ":8000" not in text


def test_secrets_are_not_hardcoded():
    assert not re.search(
        r"hf_[A-Za-z0-9]{20,}",
        "".join(p.read_text(encoding="utf-8", errors="ignore") for p in ROOT.rglob("*.py") if ".venv" not in p.parts),
    )
    assert ".env" in _read(".gitignore").splitlines()
    assert (ROOT / ".env.example").exists()


def test_space_deploy_folder_is_clean_and_uses_the_space_card(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("deploy_space", ROOT / "scripts" / "deploy_space.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = module.build_deploy_folder(tmp_path / "space")
    assert "Dockerfile" in files and "requirements.txt" in files and "data_cleaning_env/server/app.py" in files
    assert not [f for f in files if f.startswith(("tests/", ".venv/", ".git/", ".github/", "docs/")) or f == ".env"]
    assert not [f for f in files if "__pycache__" in f or f.endswith(".pyc")]
    readme = (tmp_path / "space" / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\n") and "app_port: 7860" in readme and "sdk: docker" in readme
