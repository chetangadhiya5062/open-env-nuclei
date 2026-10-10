Mean over 10 seeds per task (score = grader score in [0, 1]; improvement = share of the possible gain over doing nothing, 1 = perfect, <0 = worse than leaving the data alone).

| Agent | Task | Score (mean ± std) | Improvement (mean ± std) | Mean reward | Mean steps | Invalid-action rate | LLM fallbacks | API errors |
|---|---|---|---|---|---|---|---|---|
| do-nothing | easy-clean | 0.854 ± 0.000 | +0.000 ± 0.000 | -0.22 | 1.0 | 0.0% | 0 | 0 |
| do-nothing | medium-clean | 0.795 ± 0.000 | +0.000 ± 0.000 | -0.50 | 1.0 | 0.0% | 0 | 0 |
| do-nothing | hard-clean | 0.608 ± 0.000 | +0.000 ± 0.000 | -0.97 | 1.0 | 0.0% | 0 | 0 |
| random | easy-clean | 0.627 ± 0.140 | -1.558 ± 0.964 | -4.86 | 10.7 | 27.1% | 0 | 0 |
| random | medium-clean | 0.605 ± 0.120 | -0.930 ± 0.587 | -6.78 | 14.1 | 29.1% | 0 | 0 |
| random | hard-clean | 0.534 ± 0.058 | -0.189 ± 0.148 | -7.03 | 17.7 | 31.1% | 0 | 0 |
| rule-based | easy-clean | 0.967 ± 0.006 | +0.774 ± 0.038 | 1.92 | 4.0 | 0.0% | 0 | 0 |
| rule-based | medium-clean | 0.960 ± 0.004 | +0.806 ± 0.017 | 3.27 | 6.0 | 0.0% | 0 | 0 |
| rule-based | hard-clean | 0.968 ± 0.003 | +0.918 ± 0.008 | 5.44 | 15.0 | 0.0% | 0 | 0 |
| rl-ppo | easy-clean | 0.964 ± 0.006 | +0.755 ± 0.044 | 1.92 | 4.0 | 0.0% | 0 | 0 |
| rl-ppo | medium-clean | 0.960 ± 0.004 | +0.806 ± 0.018 | 3.27 | 6.0 | 0.0% | 0 | 0 |
| rl-ppo | hard-clean | 0.967 ± 0.004 | +0.917 ± 0.009 | 5.44 | 14.9 | 0.0% | 0 | 0 |
