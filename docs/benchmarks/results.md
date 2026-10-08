Mean over 10 seeds per task (score = grader score in [0, 1]; higher is better).

| Agent | Task | Score (mean ± std) | Mean reward | Mean steps | Invalid-action rate | LLM fallbacks |
|---|---|---|---|---|---|---|
| do-nothing | easy-clean | 0.954 ± 0.000 | -0.08 | 1.0 | 0.0% | 0 |
| do-nothing | medium-clean | 0.862 ± 0.000 | -0.33 | 1.0 | 0.0% | 0 |
| do-nothing | hard-clean | 0.608 ± 0.000 | -0.97 | 1.0 | 0.0% | 0 |
| random | easy-clean | 0.871 ± 0.079 | -1.96 | 10.7 | 27.1% | 0 |
| random | medium-clean | 0.766 ± 0.073 | -4.16 | 14.1 | 29.1% | 0 |
| random | hard-clean | 0.534 ± 0.058 | -7.03 | 17.7 | 31.1% | 0 |
| rule-based | easy-clean | 0.987 ± 0.001 | 1.25 | 2.0 | 0.0% | 0 |
| rule-based | medium-clean | 0.980 ± 0.003 | 2.47 | 5.0 | 0.0% | 0 |
| rule-based | hard-clean | 0.968 ± 0.003 | 5.44 | 15.0 | 0.0% | 0 |
