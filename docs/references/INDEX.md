# Reference library

Local copies of every source cited in [docs/RESEARCH.md](../RESEARCH.md), so the research basis can be reviewed
offline. Retrieved 2026-11-27 with `python scripts/fetch_references.py` (URLs, byte sizes and sha256 checksums are in
[`fetch_log.json`](fetch_log.json); re-run the script to rebuild the folder).

**Access notes.** "Open PDF" = an openly downloadable copy (arXiv preprint, open proceedings, open-access journal,
or the authors' own copy). Their copyright stays with the authors and publishers.

> **Local copies only.** The PDFs and HTML snapshots are kept in this folder on the project machine but are not in
> the public GitHub repository (git-ignored; most arXiv licences do not allow redistribution). They are planned for a
> separate private repository. To get them on any machine, run `python scripts/fetch_references.py` from the
> repository root: it downloads every file listed below into this folder; compare with the sha256 values in
> `fetch_log.json`.

Paywalled papers and books have a `.md` note with the full citation, DOI and notes instead of a PDF (they cannot be
legally downloaded); those notes, this index and `fetch_log.json` are in the repository.

## Papers

| # | Citation | Link | Local file | Access | Used in |
|---|---|---|---|---|---|
| 1 | Kanter & Veeramachaneni (2015). Deep Feature Synthesis: Towards Automating Data Science Endeavors. IEEE DSAA. | [author copy](https://groups.csail.mit.edu/EVO-DesignOpt/groupWebSite/uploads/Site/DSAA_DSM_2015.pdf) | `2015_kanter_deep-feature-synthesis.pdf` | open PDF (authors' group page) | `steps/features.py` (`group_aggregations`, `datetime_features`, `interactions`) |
| 2 | Zhang et al. (2023). OpenFE: Automated Feature Generation with Expert-level Performance. ICML, PMLR 202. | [arXiv:2211.12507](https://arxiv.org/abs/2211.12507) | `2023_zhang_openfe.pdf` | open PDF (arXiv) | `steps/selection.py` (`feature_generation`) |
| 3 | Horn, Pack & Rieger (2019). The autofeat Python Library for Automated Feature Engineering and Selection. ECML PKDD workshops. | [arXiv:1901.07329](https://arxiv.org/abs/1901.07329) | `2019_horn_autofeat.pdf` | open PDF (arXiv) | `steps/features.py` (`numeric_transform`) |
| 4 | Hollmann, Müller & Hutter (2023). Large Language Models for Automated Data Science: Introducing CAAFE. NeurIPS. | [arXiv:2305.03403](https://arxiv.org/abs/2305.03403) | `2023_hollmann_caafe.pdf` | open PDF (arXiv) | design of `feature_generation` (keep only validated features); `training/` plan |
| 5 | Micci-Barreca (2001). A Preprocessing Scheme for High-Cardinality Categorical Attributes. SIGKDD Explorations 3(1). | [doi:10.1145/507533.507538](https://doi.org/10.1145/507533.507538) | `2001_micci-barreca_target-encoding-high-cardinality.md` | paywalled: note only | `steps/encoding.py` (`target_encode`) |
| 6 | Prokhorenkova et al. (2018). CatBoost: unbiased boosting with categorical features. NeurIPS. | [arXiv:1706.09516](https://arxiv.org/abs/1706.09516) | `2018_prokhorenkova_catboost.pdf` | open PDF (arXiv) | `steps/encoding.py` (out-of-fold / ordered target statistics) |
| 7 | Cerda & Varoquaux (2022). Encoding high-cardinality string categorical variables. IEEE TKDE 34(3). | [arXiv:1907.01860](https://arxiv.org/abs/1907.01860) | `2022_cerda_encoding-high-cardinality-strings.pdf` | open PDF (arXiv) | not implemented (roadmap); informs `frequency_encode` / top-k `one_hot` |
| 8 | van Buuren & Groothuis-Oudshoorn (2011). mice: Multivariate Imputation by Chained Equations in R. JSS 45(3). | [jstatsoft.org](https://www.jstatsoft.org/article/view/v045i03) | `2011_vanbuuren_mice.pdf` | open access (CC BY 3.0, JSS) | `steps/cleaning.py` (`impute`, `iterative`) |
| 9 | Troyanskaya et al. (2001). Missing value estimation methods for DNA microarrays. Bioinformatics 17(6). | [doi:10.1093/bioinformatics/17.6.520](https://doi.org/10.1093/bioinformatics/17.6.520) | `2001_troyanskaya_missing-value-estimation-knn.md` | free to read, no redistribution: note only | `steps/cleaning.py` (`impute`, `knn`) |
| 10 | Le Morvan, Josse, Scornet & Varoquaux (2021). What's a good imputation to predict with missing values? NeurIPS. | [arXiv:2106.00311](https://arxiv.org/abs/2106.00311) | `2021_lemorvan_good-imputation-to-predict.pdf` | open PDF (arXiv) | `steps/cleaning.py` (`__was_missing` indicators) |
| 11 | Liu, Ting & Zhou (2008). Isolation Forest. IEEE ICDM. | [author copy](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) | `2008_liu_isolation-forest.pdf` | open PDF (author page) | `steps/cleaning.py` (`outliers`, `isolation_forest`) |
| 12 | Tukey (1977). Exploratory Data Analysis. Addison-Wesley. | ISBN 978-0201076165 | `1977_tukey_exploratory-data-analysis.md` | book: note only | `profile.py`, `outliers` (`iqr`) |
| 13 | Iglewicz & Hoaglin (1993). How to Detect and Handle Outliers. ASQC. | - | `1993_iglewicz_how-to-detect-outliers.md` | book: note only | `outliers` (`mad`) |
| 14 | Yeo & Johnson (2000). A new family of power transformations. Biometrika 87(4). | [doi:10.1093/biomet/87.4.954](https://doi.org/10.1093/biomet/87.4.954) | `2000_yeo_power-transformations.md` | paywalled: note only | `numeric_transform` (`yeo_johnson`) |
| 15 | Kraskov, Stögbauer & Grassberger (2004). Estimating mutual information. Phys. Rev. E 69. | [arXiv:cond-mat/0305641](https://arxiv.org/abs/cond-mat/0305641) | `2004_kraskov_estimating-mutual-information.pdf` | open PDF (arXiv) | `steps/selection.py` (`mi_scores`, `mutual_info_select`) |
| 16 | Ross (2014). Mutual information between discrete and continuous data sets. PLoS ONE 9(2). | [doi:10.1371/journal.pone.0087357](https://doi.org/10.1371/journal.pone.0087357) | `2014_ross_mi-discrete-continuous.pdf` | open access (CC BY 4.0, PLoS) | `mi_scores` (classification targets) |
| 17 | Breck, Polyzotis, Roy, Whang & Zinkevich (2019). Data Validation for Machine Learning. MLSys. | [mlsys.org](https://mlsys.org/Conferences/2019/doc/2019/167.pdf) | `2019_breck_data-validation-for-ml.pdf` | open PDF (proceedings) | `pipeline.py` (input validation), `profile.py` (drift) |
| 18 | Siddiqi (2006). Credit Risk Scorecards. Wiley. | ISBN 978-0471754510 | `2006_siddiqi_credit-risk-scorecards-psi.md` | book: note only | `profile.py` (`drift_report`, PSI thresholds) |
| 19 | Kaufman, Rosset, Perlich & Stitelman (2012). Leakage in data mining. ACM TKDD 6(4). | [doi:10.1145/2382577.2382579](https://doi.org/10.1145/2382577.2382579) | `2012_kaufman_leakage-in-data-mining.md` | paywalled: note only | `profile.py`, `auto.py`, `split.py`, `pipeline.py`, `training/fe_benchmark.py` |
| 20 | Kapoor & Narayanan (2023). Leakage and the reproducibility crisis in ML-based science. Patterns 4(9). | [arXiv:2207.07048](https://arxiv.org/abs/2207.07048) | `2023_kapoor_leakage-reproducibility-crisis.pdf` | open PDF (arXiv) | `split.py`, `fe_benchmark.py` |
| 21 | Gebru et al. (2021). Datasheets for Datasets. CACM 64(12). | [arXiv:1803.09010](https://arxiv.org/abs/1803.09010) | `2021_gebru_datasheets-for-datasets.pdf` | open PDF (arXiv) | `datacard.py` |
| 22 | Li, Fang & Torvik (2024). AutoDCWorkflow: LLM-based Data Cleaning Workflow Auto-Generation and Benchmark. | [arXiv:2412.06724](https://arxiv.org/abs/2412.06724) | `2024_li_autodcworkflow.pdf` | open PDF (arXiv) | typed recipes; `training/` plan |
| 23 | Qi & Wang (2024). CleanAgent: Automating Data Standardization with LLM-based Agents. | [arXiv:2403.08291](https://arxiv.org/abs/2403.08291) | `2024_qi_cleanagent.pdf` | open PDF (arXiv) | declarative step API (`recipe.py`) |
| 24 | Bendinelli, Dox & Holz (2025). Exploring LLM Agents for Cleaning Tabular ML Datasets. ICLR 2025 workshop. | [arXiv:2503.06664](https://arxiv.org/abs/2503.06664) | `2025_bendinelli_llm-agents-cleaning-tabular.pdf` | open PDF (arXiv) | `training/` plan (LLM agent fine-tuning) |
| 25 | Can LLMs Clean Up Your Mess? A Survey of Application-Ready Data Preparation with LLMs (2026). | [arXiv:2601.17058](https://arxiv.org/abs/2601.17058) | `2026_survey_llm-data-preparation.pdf` | open PDF (arXiv) | background (RESEARCH.md §8) |
| 26 | Rekatsinas, Chu, Ilyas & Ré (2017). HoloClean: Holistic Data Repairs with Probabilistic Inference. PVLDB 10(11). | [arXiv:1702.00820](https://arxiv.org/abs/1702.00820) | `2017_rekatsinas_holoclean.pdf` | open PDF (arXiv) | not implemented (RESEARCH.md §10) |
| 27 | Krishnan et al. (2016). ActiveClean: Interactive Data Cleaning for Statistical Modeling. PVLDB 9(12). (arXiv title: "...While Learning Convex Loss Models") | [arXiv:1601.03797](https://arxiv.org/abs/1601.03797) | `2016_krishnan_activeclean.pdf` | open PDF (arXiv) | background |
| 28 | Schulman et al. (2017). Proximal Policy Optimization Algorithms. | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) | `2017_schulman_ppo.pdf` | open PDF (arXiv) | `train_rl.py`, `training/train_ppo.py` |
| 29 | Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) | `2021_hu_lora.pdf` | open PDF (arXiv) | `training/train_sft.py`, `train_grpo.py` |
| 30 | Shao et al. (2024). DeepSeekMath (introduces GRPO). | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) | `2024_shao_deepseekmath-grpo.pdf` | open PDF (arXiv) | `training/train_grpo.py`, `training/rewards.py` |
| 31 | Ross, Gordon & Bagnell (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger). AISTATS. | [arXiv:1011.0686](https://arxiv.org/abs/1011.0686) | `2011_ross_dagger.pdf` | open PDF (arXiv) | `training/build_datasets.py` (perturbed starts) |
| 32 | Akiba et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. KDD. | [arXiv:1907.10902](https://arxiv.org/abs/1907.10902) | `2019_akiba_optuna.pdf` | open PDF (arXiv) | `training/tune_ppo.py`, `tune_sft.py`, `fe_benchmark.py` |
| 33 | Bergstra, Bardenet, Bengio & Kégl (2011). Algorithms for Hyper-Parameter Optimization (TPE). NeurIPS. | [proceedings](https://papers.nips.cc/paper_files/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html) | `2011_bergstra_hyperparameter-optimization-tpe.pdf` | open PDF (proceedings) | TPE sampler in the tuning scripts |
| 34 | Grinsztajn, Oyallon & Varoquaux (2022). Why do tree-based models still outperform deep learning on tabular data? NeurIPS D&B. | [arXiv:2207.08815](https://arxiv.org/abs/2207.08815) | `2022_grinsztajn_tree-models-tabular.pdf` | open PDF (arXiv) | model choice in `training/fe_benchmark.py`, `feature_generation` base model |

## Documentation and web pages (HTML snapshots, retrieved 2026-11-27)

Each snapshot starts with an HTML comment holding the URL and the retrieval date.

| Source | URL | Local file | Used in |
|---|---|---|---|
| TensorFlow Data Validation guide | https://www.tensorflow.org/tfx/guide/tfdv | `2026_tensorflow_tfdv-guide.html` | schema validation / drift design (`pipeline.py`, `profile.py`) |
| Great Expectations docs | https://docs.greatexpectations.io/docs/home/ | `2026_greatexpectations_docs-home.html` | data-quality checks design (`profile.py`) |
| Featuretools: automated feature engineering | https://featuretools.alteryx.com/en/stable/getting_started/afe.html | `2026_featuretools_dfs-docs.html` | `group_aggregations` |
| scikit-learn: imputation | https://scikit-learn.org/stable/modules/impute.html | `2026_sklearn_impute.html` | `impute` (KNN / iterative) |
| scikit-learn: preprocessing (TargetEncoder, PowerTransformer, KBinsDiscretizer) | https://scikit-learn.org/stable/modules/preprocessing.html | `2026_sklearn_preprocessing-target-encoder.html` | `target_encode`, `numeric_transform` |
| scikit-learn: outlier detection | https://scikit-learn.org/stable/modules/outlier_detection.html | `2026_sklearn_isolation-forest.html` | `outliers` |
| scikit-learn: feature selection | https://scikit-learn.org/stable/modules/feature_selection.html | `2026_sklearn_feature-selection.html` | selection steps |
| OpenFE GitHub README | https://github.com/IIIS-Li-Group/OpenFE | `2026_openfe_github-readme.html` | `feature_generation` |
| TRL SFTTrainer docs | https://huggingface.co/docs/trl/sft_trainer | `2026_trl_sft-trainer.html` | `training/train_sft.py` |
| TRL GRPOTrainer docs | https://huggingface.co/docs/trl/grpo_trainer | `2026_trl_grpo-trainer.html` | `training/train_grpo.py` |
| Stable-Baselines3 PPO docs | https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html | `2026_sb3_ppo.html` | `training/train_ppo.py` |
| Optuna docs | https://optuna.readthedocs.io/en/stable/ | `2026_optuna_docs.html` | tuning scripts |
| OpenEnv (meta-pytorch) GitHub | https://github.com/meta-pytorch/OpenEnv | `2026_openenv_github.html` | the environment server |
