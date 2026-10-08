---
title: OpenEnv Data Cleaning
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
base_path: /demo
tags:
  - openenv
  - reinforcement-learning
  - data-cleaning
---

# OpenEnv Data Cleaning Environment

An OpenEnv environment where an agent cleans a dirty pandas table one action at a time. Three seeded tasks
(`easy-clean`, `medium-clean`, `hard-clean`), nine typed actions, a shaped reward, and a 0-1 grader against a hidden
ground truth.

- **Demo UI:** `/demo` (choose task, seed and agent, watch each step).
- **API docs:** `/docs`. Health: `/health`. Schemas: `/schema`. Persistent sessions: WebSocket `/ws`.
- `/reset` and `/step` over plain HTTP are stateless; use the WebSocket client for real episodes.

```python
from data_cleaning_env import DataCleaningAction, DataCleaningEnv

with DataCleaningEnv(base_url="https://<your-space>.hf.space").sync() as env:
    result = env.reset(task_id="hard-clean", seed=0)
    result = env.step(DataCleaningAction(action_type="strip_whitespace"))
```

Source, reward formula, grader and benchmark results: https://github.com/chetangadhiya5062/open-env-nuclei

Optional secret: `HF_TOKEN` enables the zero-shot LLM agent in the demo (the rule-based, random and do-nothing agents
need no token).
