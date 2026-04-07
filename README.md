# OpenEnv Data Cleaning RL Agent

<p align="center">
  <b>Reinforcement Learning + LLMs for Intelligent Data Cleaning</b><br>
  A practical system exploring how agents make decisions under imperfect rewards
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LLM-OpenRouter-blue" />
  <img src="https://img.shields.io/badge/RL-Environment-green" />
  <img src="https://img.shields.io/badge/Python-3.10+-yellow" />
  <img src="https://img.shields.io/badge/Status-Active-black" />
</p>

---

## Overview

This project combines **LLMs with a custom reinforcement learning environment** to solve a real problem: data cleaning.

Instead of hardcoded rules, the system uses an agent that:

* observes dataset issues
* decides cleaning actions
* improves based on rewards

It’s a small system — but it exposes a deeper idea:

> Agents don’t “understand” tasks. They optimize what you reward.

---

## Workflow

```text
Dataset → Environment → Agent (LLM) → Action → Reward → Next State → Repeat
```

---

## Features

### Intelligent Actions

* `fill_missing`
* `drop_rows_with_missing`
* `remove_duplicates`

### Reward Design

* Encourages data quality improvement
* Penalizes unnecessary data loss

### System Capabilities

* LLM-based decision making (OpenRouter)
* Multi-step agent loop
* Model fallback handling
* Modular and extensible architecture

---

## Project Structure

```text
open-env-nuclei/
│
├── data_cleaning_env/
│   ├── models.py
│   ├── client.py
│   ├── inference.py
│   ├── server/
│   │   ├── app.py
│   │   ├── data_cleaning_env_environment.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup

### Clone the repository

```bash
git clone https://github.com/chetangadhiya5062/open-env-nuclei.git
cd open-env-nuclei
```

### Create virtual environment

```bash
python -m venv .venv_scalar
```

### Activate (Windows)

```bash
.\.venv_scalar\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the System

### Start the environment server

```bash
python -m data_cleaning_env.server.app
```

### Run the agent

```bash
python -m data_cleaning_env.inference
```

---

## Core Idea

Traditional data cleaning pipelines are static.

This system explores a different approach:

* decisions are dynamic
* behavior emerges from rewards
* outcomes depend on system design

It’s less about cleaning data —
and more about **how intelligent systems make trade-offs**.

---

## Use Cases

* Automated data preprocessing
* AI-assisted ETL pipelines
* Dataset optimization workflows
* RL experimentation platform

---

## Best Practices

* Keep API keys in `.env` (never commit them)
* Exclude `.venv_scalar` from version control
* Use `requirements.txt` for reproducibility

---

## Roadmap

* RL training loop (PPO / DQN)
* Visualization dashboard
* Docker support
* CI/CD integration
* Benchmark datasets

---

## Author

**Chetan Gadhiya**

---

## Support

If this project was useful or interesting:

* Star the repository
* Share it with others
* Open issues or contribute

---

## Recruiter Note

This project demonstrates:

* Reinforcement learning fundamentals
* LLM integration in real systems
* Reward design and agent behavior
* Clean architecture and modular thinking

---

### Final touch (subtle but powerful)

You can optionally add this at the very bottom:

> Built as an exploration of how LLM agents behave in structured environments.

---
