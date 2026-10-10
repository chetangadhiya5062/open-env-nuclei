Mean over 5 seeds per task (score = grader score in [0, 1]; improvement = share of the possible gain over doing nothing, 1 = perfect, <0 = worse than leaving the data alone).

| Agent | Task | Score (mean ± std) | Improvement (mean ± std) | Mean reward | Mean steps | Invalid-action rate | LLM fallbacks | API errors |
|---|---|---|---|---|---|---|---|---|
| llm:llama3.1:8b | easy-clean | 0.956 ± 0.012 | +0.698 ± 0.080 | 0.11 | 15.0 | 0.0% | 0 | 0 |
| llm:llama3.1:8b | medium-clean | 0.959 ± 0.005 | +0.799 ± 0.024 | 0.88 | 25.0 | 0.0% | 0 | 0 |
| llm:llama3.1:8b | hard-clean | 0.838 ± 0.037 | +0.586 ± 0.094 | 1.43 | 45.0 | 4.0% | 0 | 0 |
