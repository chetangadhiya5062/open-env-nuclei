"""Download local copies of the sources cited in docs/RESEARCH.md into docs/references/.

    python scripts/fetch_references.py            # skip files that already exist
    python scripts/fetch_references.py --force    # download again

* open-access papers -> ``<year>_<firstauthor>_<short-title>.pdf``
* web pages / docs   -> ``<year>_<site>_<short-title>.html`` with a comment header (URL + retrieval date)
* paywalled / books  -> ``<year>_<firstauthor>_<short-title>.md`` (citation, DOI, notes) - written by hand, not here

Writes ``docs/references/fetch_log.json`` (url, file, bytes, sha256, status, date). The files are stored as data only.
"""

import argparse
import datetime as dt
import hashlib
import json
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "references"
UA = "Mozilla/5.0 (compatible; reference-archiver/1.0)"

# (local file name, url)
PDFS = [
    (
        "2015_kanter_deep-feature-synthesis.pdf",
        "https://groups.csail.mit.edu/EVO-DesignOpt/groupWebSite/uploads/Site/DSAA_DSM_2015.pdf",
    ),
    ("2023_zhang_openfe.pdf", "https://arxiv.org/pdf/2211.12507"),
    ("2019_horn_autofeat.pdf", "https://arxiv.org/pdf/1901.07329"),
    ("2023_hollmann_caafe.pdf", "https://arxiv.org/pdf/2305.03403"),
    ("2018_prokhorenkova_catboost.pdf", "https://arxiv.org/pdf/1706.09516"),
    ("2022_cerda_encoding-high-cardinality-strings.pdf", "https://arxiv.org/pdf/1907.01860"),
    ("2011_vanbuuren_mice.pdf", "https://www.jstatsoft.org/index.php/jss/article/download/v045i03/550"),
    ("2021_lemorvan_good-imputation-to-predict.pdf", "https://arxiv.org/pdf/2106.00311"),
    ("2008_liu_isolation-forest.pdf", "https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf"),
    ("2004_kraskov_estimating-mutual-information.pdf", "https://arxiv.org/pdf/cond-mat/0305641"),
    (
        "2014_ross_mi-discrete-continuous.pdf",
        "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0087357&type=printable",
    ),
    ("2019_breck_data-validation-for-ml.pdf", "https://mlsys.org/Conferences/2019/doc/2019/167.pdf"),
    ("2023_kapoor_leakage-reproducibility-crisis.pdf", "https://arxiv.org/pdf/2207.07048"),
    ("2021_gebru_datasheets-for-datasets.pdf", "https://arxiv.org/pdf/1803.09010"),
    ("2024_li_autodcworkflow.pdf", "https://arxiv.org/pdf/2412.06724"),
    ("2024_qi_cleanagent.pdf", "https://arxiv.org/pdf/2403.08291"),
    ("2025_bendinelli_llm-agents-cleaning-tabular.pdf", "https://arxiv.org/pdf/2503.06664"),
    ("2026_survey_llm-data-preparation.pdf", "https://arxiv.org/pdf/2601.17058"),
    ("2017_rekatsinas_holoclean.pdf", "https://arxiv.org/pdf/1702.00820"),
    ("2016_krishnan_activeclean.pdf", "https://arxiv.org/pdf/1601.03797"),
    ("2017_schulman_ppo.pdf", "https://arxiv.org/pdf/1707.06347"),
    ("2021_hu_lora.pdf", "https://arxiv.org/pdf/2106.09685"),
    ("2024_shao_deepseekmath-grpo.pdf", "https://arxiv.org/pdf/2402.03300"),
    ("2011_ross_dagger.pdf", "https://arxiv.org/pdf/1011.0686"),
    ("2019_akiba_optuna.pdf", "https://arxiv.org/pdf/1907.10902"),
    (
        "2011_bergstra_hyperparameter-optimization-tpe.pdf",
        "https://papers.nips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf",
    ),
    ("2022_grinsztajn_tree-models-tabular.pdf", "https://arxiv.org/pdf/2207.08815"),
]

PAGES = [
    ("2026_tensorflow_tfdv-guide.html", "https://www.tensorflow.org/tfx/guide/tfdv"),
    ("2026_greatexpectations_docs-home.html", "https://docs.greatexpectations.io/docs/home/"),
    ("2026_featuretools_dfs-docs.html", "https://featuretools.alteryx.com/en/stable/getting_started/afe.html"),
    ("2026_sklearn_impute.html", "https://scikit-learn.org/stable/modules/impute.html"),
    ("2026_sklearn_preprocessing-target-encoder.html", "https://scikit-learn.org/stable/modules/preprocessing.html"),
    ("2026_sklearn_isolation-forest.html", "https://scikit-learn.org/stable/modules/outlier_detection.html"),
    ("2026_sklearn_feature-selection.html", "https://scikit-learn.org/stable/modules/feature_selection.html"),
    ("2026_openfe_github-readme.html", "https://github.com/IIIS-Li-Group/OpenFE"),
    ("2026_trl_sft-trainer.html", "https://huggingface.co/docs/trl/sft_trainer"),
    ("2026_trl_grpo-trainer.html", "https://huggingface.co/docs/trl/grpo_trainer"),
    ("2026_sb3_ppo.html", "https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html"),
    ("2026_optuna_docs.html", "https://optuna.readthedocs.io/en/stable/"),
    ("2026_openenv_github.html", "https://github.com/meta-pytorch/OpenEnv"),
]


def fetch(url: str, timeout: int = 60) -> bytes:
    """urllib with the certifi bundle; falls back to the system curl (uses the OS certificate store)."""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:  # noqa: S310 - fixed https list
            return resp.read()
    except (urllib.error.URLError, ssl.SSLError, ImportError):
        curl = shutil.which("curl")
        if not curl:
            raise
        res = subprocess.run(
            [curl, "-sSfL", "--max-time", str(timeout), "-A", UA, url], capture_output=True, check=True
        )
        return res.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "fetch_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    today = dt.date.today().isoformat()
    for kind, items in (("pdf", PDFS), ("page", PAGES)):
        for name, url in items:
            path = OUT / name
            if path.exists() and not args.force:
                continue
            try:
                data = fetch(url)
                if kind == "pdf" and not data.startswith(b"%PDF"):
                    raise ValueError("response is not a PDF")
                if kind == "page":
                    header = f"<!-- Snapshot of {url}\n     retrieved {today}; saved for reference only. -->\n"
                    data = header.encode("utf-8") + data
                path.write_bytes(data)
                log[name] = {
                    "url": url,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "status": "ok",
                    "retrieved": today,
                }
                print(f"ok      {name} ({len(data) // 1024} KB)")
            except Exception as exc:  # keep going; failures are listed in the log and in INDEX.md
                log[name] = {"url": url, "status": f"failed: {type(exc).__name__}: {exc}"[:200], "retrieved": today}
                print(f"FAILED  {name}: {exc}", file=sys.stderr)
            time.sleep(1.0)  # be polite to arxiv.org
    log_path.write_text(json.dumps(log, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
