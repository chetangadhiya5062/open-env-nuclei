# 🧠 OpenEnv Data Cleaning RL Agent

<p align="center">
  🚀 Reinforcement Learning + LLMs for Intelligent Data Cleaning  
</p>

---

## 📌 Project Highlights

* 🤖 LLM-powered decision making (OpenRouter)
* 🧪 Custom RL Environment (OpenEnv)
* 📊 Smart data cleaning with reward shaping
* 🔁 Multi-step reasoning agent
* ⚡ Clean, modular architecture

---

## 🖼️ Workflow

```text
Dataset → Environment → Agent (LLM) → Action → Reward → Next State → Repeat
```

---

## ✨ Features

* Custom environment for real-world data cleaning
* Intelligent actions:

  * `fill_missing`
  * `drop_rows_with_missing`
  * `remove_duplicates`
* Reward system focused on:

  * Maximizing data retention
  * Improving data quality
* LLM fallback mechanism
* Extensible architecture for future RL training

---

## 📁 Project Structure

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

## ⚙️ Setup Guide

### 1️⃣ Clone the repository

```bash
git clone https://github.com/chetangadhiya5062/open-env-nuclei.git
cd open-env-nuclei
```

---

### 2️⃣ Create virtual environment

```bash
python -m venv .venv_scalar
```

---

### 3️⃣ Activate environment

**Windows:**

```bash
.\.venv_scalar\Scripts\activate
```

---

### 4️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Start server

```bash
python -m data_cleaning_env.server.app
```

---

### Run agent

```bash
python -m data_cleaning_env.inference
```

---

## 🧠 Core Idea

Traditional data cleaning is rule-based and rigid.

This project introduces:

👉 An **intelligent agent** that:

* Understands context
* Chooses optimal cleaning actions
* Learns from rewards

---

## 🎯 Use Cases

* Data preprocessing automation
* AI-assisted ETL pipelines
* Smart dataset optimization
* RL experimentation platform

---

## 🔐 Best Practices

* ❌ Never commit `.env` or API keys
* ❌ Do not push `.venv_scalar`
* ✅ Use `requirements.txt` for reproducibility

---

## 🚀 Future Roadmap

* [ ] Add RL training loop (PPO/DQN)
* [ ] Visualization dashboard
* [ ] Docker support
* [ ] CI/CD pipeline
* [ ] Benchmark datasets

---

## 👨‍💻 Author

**Chetan Gadhiya**

---

## ⭐ Support

If you like this project:

👉 Star ⭐ the repo
👉 Share it
👉 Contribute

---

## 💡 Recruiter Note

This project demonstrates:

* Reinforcement Learning fundamentals
* LLM integration
* System design thinking
* Clean Git workflow & collaboration

---
