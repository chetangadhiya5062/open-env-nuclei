# OpenEnv Data Cleaning RL Agent

A reinforcement learning-based system for intelligent data cleaning using the OpenEnv framework and LLM-driven decision making.

---

## Overview

This project builds a **custom RL environment** where an agent learns to clean datasets while preserving maximum useful information.

It combines:

* 🧪 Reinforcement Learning (environment + rewards)
* 🤖 LLM-based decision making (OpenRouter integration)
* 🔄 Multi-step inference loop
* 📊 Data cleaning strategies

---

## ✨ Features

* Custom OpenEnv environment for data cleaning
* Actions:

  * `fill_missing`
  * `drop_rows_with_missing`
  * `remove_duplicates`
* Reward shaping to prioritize **data preservation**
* LLM agent integration (fallback supported)
* Multi-step reasoning and execution
* Modular and extensible design

---

## 📁 Project Structure

```
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

## ⚙️ Setup

### 01. Clone the repository

```bash
git clone https://github.com/chetangadhiya5062/open-env-nuclei.git
cd open-env-nuclei
```

---

### 02. Create virtual environment

```bash
python -m venv .venv_scalar
```

---

### 03. Activate environment

**Windows:**

```bash
.\.venv_scalar\Scripts\activate
```

---

### 04. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Start the OpenEnv server

```bash
python -m data_cleaning_env.server.app
```

---

### Run the agent

```bash
python -m data_cleaning_env.inference
```

---

## How it Works

1. Environment provides dataset state
2. Agent (LLM or logic) selects an action
3. Action is applied (cleaning step)
4. Reward is calculated based on:

   * Data preserved
   * Quality improvement
5. Loop continues for multiple steps

---

## Goal

Train an intelligent agent that:

* Cleans messy datasets
* Avoids unnecessary data loss
* Makes **context-aware decisions**

---

## Important Notes

* ❌ Do NOT commit `.env` or API keys
* ❌ Do NOT push virtual environment (`.venv_scalar`)
* ✅ Use `requirements.txt` for dependencies

---

## Future Improvements

* Add training pipeline (RL algorithms)
* Improve reward shaping
* Add evaluation metrics
* Dockerize for deployment
* CI/CD integration

---

## Author

**Chetan Gadhiya**

---

## ⭐ If you found this useful

Give this repo a ⭐ and feel free to contribute!
