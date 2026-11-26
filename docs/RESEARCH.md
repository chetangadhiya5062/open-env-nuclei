# Research basis

This document lists the papers and tools that shaped the FeatureForge library (`featureforge/`) and the GPU training
plan (`training/`). For each one: what it says, **what we implement**, and **why** (or why not).

Status keys: **Implemented** = in the code and tested. **Simplified** = the idea is implemented in a cheaper form
(stated how). **Not implemented** = read and deliberately left out, with the reason.

---

## 1. Automated feature engineering

### Deep Feature Synthesis (Featuretools)
Kanter, J. M., & Veeramachaneni, K. (2015). *Deep Feature Synthesis: Towards Automating Data Science Endeavors.*
IEEE International Conference on Data Science and Advanced Analytics (DSAA).

- **Idea:** stack simple "primitives" over relational entities: aggregation primitives (mean, count, std of a child
  table grouped by a parent key) and transform primitives (month of a date, ratio of two columns), to depth *d*.
- **We implement (Simplified):** the single-table version. `group_aggregations` computes `mean / std / count` of numeric
  columns grouped by each low/medium-cardinality categorical column, plus the row's deviation from its group mean.
  `datetime_features` and `interactions` are transform primitives. Depth is 1 (no stacking).
- **Why:** most uploaded files are one table. Depth-1 primitives give most of the benefit and stay interpretable; deep
  stacking explodes the column count, which our selection steps would then have to undo.

### OpenFE
Zhang, T., Zhang, Z., Fan, Z., Luo, H., Liu, F., Liu, Q., Cao, W., & Li, J. (2023). *OpenFE: Automated Feature
Generation with Expert-level Performance.* ICML 2023, PMLR 202:41880–41901. arXiv:2211.12507.

- **Idea:** generate a large set of candidate features (unary/binary operators, group-by), then score each by
  *feature boosting*: keep a base model's predictions fixed and measure how much a small model on the residual improves
  when the candidate is added. Prune in two stages (cheap coarse screen, then a finer one).
- **We implement (Simplified):** step `feature_generation`. Candidates: `a*b`, `a/b`, `a-b`, `a+b` for the top numeric
  columns and `groupby(cat)[num].mean()`. Stage 1 (coarse): mutual information with the target on a subsample, keep
  the top `stage1_keep`. Stage 2 (fine): fit a base gradient-boosting model, take its out-of-fold residuals, fit a
  depth-limited booster on each candidate alone against the residual and keep candidates that reduce the held-out
  residual error. Only the formulas are stored in the pipeline, so `transform` recomputes them on new data.
- **Why:** this is the part of OpenFE that matters for a product: a principled, cheap screen that does not need a
  full model refit per candidate. We do not port OpenFE's full operator set or its successive-halving schedule (cost).

### AutoFeat
Horn, F., Pack, R., & Rieger, M. (2019). *The autofeat Python Library for Automated Feature Engineering and
Selection.* ECML PKDD 2019 Workshops. arXiv:1901.07329.

- **Idea:** generate non-linear transforms (log, sqrt, 1/x, x², products) and select with an L1-regularised linear
  model, which suits linear models and keeps results interpretable.
- **We implement (Simplified):** `numeric_transform` (log1p for skewed non-negative columns, Yeo-Johnson, quantile
  binning) and the correlation/MI selection steps. We do not run the L1 selection loop.
- **Why:** unary transforms help linear models a lot and cost little; the L1 loop is model-specific and is covered by
  our model-agnostic MI and correlation filters.

### CAAFE (LLM feature engineering)
Hollmann, N., Müller, S., & Hutter, F. (2023). *Large Language Models for Automated Data Science: Introducing CAAFE
for Context-Aware Automated Feature Engineering.* NeurIPS 2023. arXiv:2305.03403.

- **Idea:** an LLM reads the dataset description and column names, writes pandas code for new features with an
  explanation, the code is executed, and a feature is kept only if cross-validated performance improves (reported: mean
  ROC AUC 0.798 -> 0.822, better on 11 of 14 datasets).
- **We implement (Not yet, prepared):** the *accept only if validated* rule is how `feature_generation` works (a
  candidate must lower held-out error). The LLM part is the planned next step: the SFT/GRPO training plan in
  `training/` fine-tunes a small model to propose cleaning actions first, because that action space is small, typed and
  safely executable. Free-form generated code is not executed by the product (security: no arbitrary code execution on
  uploaded data).
- **Why:** running model-written code on a customer's server is a security and reproducibility risk. Recipes made of
  declared, typed steps are safe to replay and to audit.

## 2. Categorical encoding and target leakage

### Target (mean) encoding with smoothing
Micci-Barreca, D. (2001). *A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and
Prediction Problems.* ACM SIGKDD Explorations 3(1), 27–32.

- **Idea:** replace a category by a blend of its target mean and the global mean, weighted by the category count, so
  rare categories shrink toward the prior.
- **We implement (Implemented):** `target_encode` with `encoded = (n * mean_cat + m * prior) / (n + m)`, `m` =
  `smoothing`. Regression: target mean. Binary: positive rate. Multiclass: one column per class.

### CatBoost ordered target statistics
Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). *CatBoost: unbiased boosting with
categorical features.* NeurIPS 2018.

- **Idea:** a target statistic computed on the same rows it is used for leaks the target ("target leakage /
  prediction shift"). CatBoost computes it only from rows *before* the current one in a random permutation.
- **We implement (Implemented, out-of-fold variant):** during `fit_transform` the training rows are encoded with
  K-fold out-of-fold statistics (each row's value comes from the other folds). `transform` on new data uses statistics
  from the full training set. Both an out-of-fold mode (default) and an ordered mode (`scheme: ordered`) are available.
  A test checks that a pure-noise high-cardinality column does not get a spuriously high train-set correlation.
- **Why:** naive target encoding is the most common leakage bug in tabular pipelines; it makes validation scores lie.

### Encoding high-cardinality strings
Cerda, P., & Varoquaux, G. (2022). *Encoding high-cardinality string categorical variables.* IEEE TKDE 34(3).

- **Not implemented:** similarity / min-hash encoders. We use frequency encoding, top-k one-hot with an `__other__`
  bucket, and target encoding, which cover most cases. Listed on the roadmap.

## 3. Missing values

### MICE / iterative imputation
van Buuren, S., & Groothuis-Oudshoorn, K. (2011). *mice: Multivariate Imputation by Chained Equations in R.* Journal
of Statistical Software 45(3).

- **We implement (Implemented):** `impute` with `numeric_strategy: iterative` (scikit-learn `IterativeImputer`, a
  chained-equations imputer; single imputation, not multiple).
- **Why single imputation:** the product outputs one dataset. Multiple imputation is for inference on uncertainty;
  documented as a limitation.

### KNN imputation
Troyanskaya, O., et al. (2001). *Missing value estimation methods for DNA microarrays.* Bioinformatics 17(6), 520–525.

- **We implement (Implemented):** `numeric_strategy: knn` (scikit-learn `KNNImputer`, fit on a capped subsample so
  memory stays bounded on large files).

### Missingness indicators
Missing values are often informative (missing not at random). Le Morvan, M., Josse, J., Scornet, E., & Varoquaux, G.
(2021). *What's a good imputation to predict with missing values?* NeurIPS 2021 — shows that imputation plus a mask is a
strong, consistent approach for prediction.

- **We implement (Implemented):** `add_indicator: true` adds `<col>__was_missing` flags for columns that had missing
  values at fit time.

## 4. Outliers

### Isolation Forest
Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest.* IEEE ICDM 2008.

- **We implement (Implemented):** `outliers` with `method: isolation_forest` adds a row-level `is_outlier` flag and an
  `outlier_score`. It is a flag, not a deletion, so the downstream model decides.

### Robust univariate rules
Tukey's IQR fences (Tukey, 1977, *Exploratory Data Analysis*) and the MAD-based modified z-score (Iglewicz, B., &
Hoaglin, D. C. (1993). *How to Detect and Handle Outliers.* ASQC).

- **We implement (Implemented):** `method: iqr` (`k` configurable, default 1.5 for flagging and 3.0 for capping in the
  auto recipe) and `method: mad` (modified z-score > 3.5). `action: clip | flag | both`. Bounds are learned on the
  training data and reused on new data.
- **Why both:** univariate rules are explainable per column; Isolation Forest catches multivariate oddities.

## 5. Distribution transforms

Yeo, I.-K., & Johnson, R. A. (2000). *A new family of power transformations to improve normality or symmetry.*
Biometrika 87(4), 954–959.

- **We implement (Implemented):** `numeric_transform` with `method: yeo_johnson` (works with zero and negative values,
  unlike Box-Cox), `log1p` (auto-selected for skewed non-negative columns) and `quantile_bin`.

Cyclical encoding of periodic features (hour, weekday, month as `sin`/`cos` of the angle) is standard practice so that
23:00 and 00:00 are close; `datetime_features` adds it.

## 6. Feature selection

- **Mutual information:** Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). *Estimating mutual information.*
  Physical Review E 69; Ross, B. C. (2014). *Mutual information between discrete and continuous data sets.* PLoS ONE.
  scikit-learn's `mutual_info_classif / regression` use these estimators. **Implemented:** `mutual_info_select`.
- **Correlation pruning:** drop one of each pair with |Pearson r| above a threshold, keeping the one more informative
  about the target. **Implemented:** `correlation_filter`.
- **Variance filter:** drop (near-)constant columns. **Implemented:** `variance_filter`.

## 7. Data validation, profiling and drift

### Data validation for ML
Breck, E., Polyzotis, N., Roy, S., Whang, S. E., & Zinkevich, M. (2019). *Data Validation for Machine Learning.*
MLSys 2019. Tooling: TensorFlow Data Validation (TFDV) and Great Expectations.

- **Idea:** infer a schema from training data, then validate every new batch against it (types, domains, presence)
  and detect training/serving skew and drift.
- **We implement (Implemented):** the fitted pipeline stores the input schema it was fitted on (column names, inferred
  types, category domains). `transform` validates new data: missing required columns are a clear error, unexpected
  columns a warning, unseen categories go to `__other__`. `featureforge.profile.drift_report` compares a reference and a
  new dataset per column with the Population Stability Index (PSI; Siddiqi, N. (2006). *Credit Risk Scorecards.*
  Wiley) and the two-sample Kolmogorov–Smirnov test for numeric columns.
- **Why not depend on Great Expectations / TFDV:** heavy dependencies; we need a small, embeddable subset.

### Leakage
Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). *Leakage in data mining: Formulation, detection, and
avoidance.* ACM TKDD 6(4). Kapoor, S., & Narayanan, A. (2023). *Leakage and the reproducibility crisis in
machine-learning-based science.* Patterns 4(9).

- **We implement (Implemented):**
  - profiling warnings: columns that predict the target almost perfectly (`possible_target_leak`), columns equal to
    the target, ID-like columns (`id_column`), columns whose name contains the target name;
  - all statistics are learned on training data only (`fit`) and replayed on test data (`transform`);
  - `featureforge.split.leakage_safe_split`: exact duplicate rows are kept on the same side of the split (grouped by a
    row hash), optional group-aware and time-ordered splits, and a report of any overlap that remains;
  - out-of-fold target encoding (section 2).
  - `training/fe_benchmark.py` fits the whole pipeline *inside* each CV fold.

### Datasheets for Datasets
Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021).
*Datasheets for Datasets.* Communications of the ACM 64(12). arXiv:1803.09010.

- **We implement (Implemented):** `featureforge.datacard` writes a Markdown dataset card for each output: source
  file and checksum, composition (rows, columns, types), the preprocessing applied (every step and parameter of the
  recipe), quality warnings, and sections the data owner must fill in (motivation, collection process, consent,
  intended uses). We do not invent answers to the questions only the data owner can answer.

## 8. Data-cleaning agents and LLMs

- Li, L., Fang, L., & Torvik, V. I. (2024). *AutoDCWorkflow: LLM-based Data Cleaning Workflow Auto-Generation and
  Benchmark.* arXiv:2412.06724. LLMs can plan cleaning workflows of typed operations without fine-tuning, but errors
  compound across steps.
- Qi, D., & Wang, J. (2024). *CleanAgent: Automating Data Standardization with LLM-based Agents.* arXiv:2403.08291.
  Declarative, typed cleaning APIs make LLM-generated plans shorter and more reliable.
- Bendinelli, T., Dox, A., & Holz, C. (2025). *Exploring LLM Agents for Cleaning Tabular Machine Learning Datasets.*
  ICLR 2025 Workshop on Foundation Models in the Wild. arXiv:2503.06664.
- Survey: *Can LLMs Clean Up Your Mess? A Survey of Application-Ready Data Preparation with LLMs.* arXiv:2601.17058
  (2026).
- Earlier systems: Rekatsinas, T., Chu, X., Ilyas, I. F., & Ré, C. (2017). *HoloClean: Holistic Data Repairs with
  Probabilistic Inference.* PVLDB 10(11); Krishnan, S., et al. (2016). *ActiveClean: Interactive Data Cleaning for
  Statistical Modeling.* PVLDB 9(12).

**What we take from them:** (1) a *typed, declarative operation set* (our recipe steps and the environment's actions)
instead of free-form code; (2) per-step validation and the ability to stop errors from compounding (each step's
before/after report in the dashboard; invalid environment actions are rejected, not applied); (3) a benchmark with
hidden ground truth (the OpenEnv environment and its grader). Our own measured result on that environment is in the
README: the zero-shot Llama 3.1 8B agent trails the rule-based and PPO agents on `hard-clean` (0.838 vs 0.968), mostly
by choosing the wrong operation for dates and by not stopping. That is the motivation for the supervised fine-tuning
and GRPO plan in `training/`.

## 9. Training methods used in `training/`

- PPO: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization
  Algorithms.* arXiv:1707.06347. Used for the environment policy (already trained on CPU in v3; GPU configs added).
- LoRA: Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685. Used for
  parameter-efficient SFT of a small instruct model on expert trajectories.
- GRPO: Shao, Z., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
  Models.* arXiv:2402.03300. Group-relative policy optimisation needs no value model; our reward is the environment's
  own step reward plus a JSON-validity term.
- DAgger-style data: Ross, S., Gordon, G., & Bagnell, D. (2011). *A Reduction of Imitation Learning and Structured
  Prediction to No-Regret Online Learning.* AISTATS. We do not run full DAgger; the SFT set includes states reached
  after a few random actions, labelled by the expert, so the model sees recovery situations and not only perfect
  trajectories.
- Optuna: Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter
  Optimization Framework.* KDD 2019 (TPE sampler: Bergstra, J., et al. (2011). NeurIPS).
- Tabular model choice for the feature-engineering benchmark: Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022).
  *Why do tree-based models still outperform deep learning on typical tabular data?* NeurIPS 2022 Datasets and
  Benchmarks. We evaluate with gradient-boosted trees and a linear model.

## 10. What we deliberately did not build

| Idea | Reason |
|---|---|
| Executing LLM-written feature code (CAAFE style) on uploaded data | arbitrary code execution risk; recipes must be safe to replay |
| Multiple imputation | product outputs one table; documented limitation |
| Deep (stacked) feature synthesis, multi-table | single-table uploads; column explosion |
| Probabilistic repair (HoloClean) | needs denial constraints from the user; out of scope for v4 |
| Similarity / min-hash encoders | roadmap |
