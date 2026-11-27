# FeatureForge generic cleaning on the OpenEnv tasks

`python scripts/eval_library_on_env.py --seeds 10` - one-shot recipe (no agent loop), graded by
`data_cleaning_env/server/grader.py` on test seeds 0..N-1. The recipe knows expected types and valid ranges
(both visible to agents) but not the allowed category vocabulary.

| task | score (mean ± std) | improvement | do-nothing |
|---|---|---|---|
| easy-clean | 0.967 ± 0.005 | +0.774 | 0.854 |
| medium-clean | 0.960 ± 0.003 | +0.806 | 0.795 |
| hard-clean | 0.932 ± 0.004 | +0.826 | 0.608 |
