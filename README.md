<div align="center">

# 🧹 OpenEnv Data Cleaning RL Agent

### Reinforcement Learning meets Large Language Models for Autonomous Data Cleaning

*An agent that doesn't just clean data — it **learns** how to clean it.*

<br>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Llama 3](https://img.shields.io/badge/LLM-Llama%203-FF6B35?style=for-the-badge&logo=meta&logoColor=white)](https://llama.meta.com)
[![HuggingFace](https://img.shields.io/badge/Deployed-HuggingFace%20Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://chetangadhiya017-data-cleaning-env.hf.space)
[![OpenRouter](https://img.shields.io/badge/Inference-OpenRouter-6C63FF?style=for-the-badge)](https://openrouter.ai)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br>

**[🚀 Live Demo](https://chetangadhiya017-data-cleaning-env.hf.space)** · **[📖 Documentation](#-architecture)** · **[⚡ Quick Start](#️-setup--installation)**

<br>

> *"Traditional pipelines tell the agent what to do. This system lets the agent figure it out."*

</div>

---

## 🧠 The Core Idea

Most data cleaning pipelines are **static rule engines**: hardcoded `if null → fill with mean`, `if duplicate → drop`. They're brittle, opinionated, and blind to context.

**This project takes a completely different approach.**

Instead of rules, it uses a **reward signal**. The agent observes a dataset, chooses a cleaning action, receives feedback from the environment, and adapts. The behavior isn't programmed — it *emerges* from what gets rewarded.

This is the intersection of two powerful ideas:

```
Reinforcement Learning  +  Large Language Models
   (reward-driven          (context-aware
    decision making)        action selection)
         └────────────────────┘
                    ↓
        Autonomous Data Cleaning Agent
```

The LLM provides **reasoning and language understanding**. The RL environment provides **structured feedback and ground truth**. Together, they form an agent that can handle messy, real-world tabular data without being told exactly what to do.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Agent Decision Loop                       │
│                                                                  │
│   ┌─────────────┐    Observation     ┌──────────────────────┐   │
│   │             │ ◄─────────────────  │                      │   │
│   │  LLM Agent  │                    │  OpenEnv Environment │   │
│   │  (Llama 3)  │  Action Selection  │                      │   │
│   │             │ ──────────────────► │  • Apply transform   │   │
│   │ via          │                    │  • Compute reward    │   │
│   │ OpenRouter  │ ◄─────────────────  │  • Return new state  │   │
│   └─────────────┘    Reward + State   └──────────────────────┘   │
│                                                                  │
│            Repeat until termination condition                    │
└──────────────────────────────────────────────────────────────────┘
```

### How a single step works

```
Step 1 — Observation
  The environment sends the agent a snapshot:
  { "missing_values": 12, "duplicate_rows": 3, "data_sample": [...] }

Step 2 — LLM Reasoning
  The LLM reads the observation and selects the best action:
  → "fill_missing" | "drop_rows_with_missing" | "remove_duplicates"

Step 3 — Environment Execution
  The action is applied to the actual DataFrame.

Step 4 — Reward Assignment
  +reward  → dataset improved (fewer issues, no data loss)
  -penalty → loop detected, or destructive operation

Step 5 — Next Observation
  Updated state is returned. Agent iterates.
```

---

## 📦 Project Structure

```
open-env-nuclei/
│
├── data_cleaning_env/
│   ├── models.py                      # Pydantic schemas for state, actions, rewards
│   ├── client.py                      # Agent-side: sends observations, receives actions
│   ├── inference.py                   # LLM inference loop via OpenRouter
│   │
│   └── server/
│       ├── app.py                     # FastAPI server — exposes environment as REST API
│       └── data_cleaning_env_environment.py  # Core RL environment logic
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ The RL Environment (OpenEnv)

The custom environment is the heart of this project. It's designed around three principles:

**1. Observations are informative, not overwhelming**

Rather than dumping the full dataset at every step, the environment sends a *structured summary* — missing value counts, duplicate counts, and a small `data_sample` for context. This mirrors how a real analyst would assess a dataset at a glance.

**2. Rewards are shaped carefully**

| Situation | Reward |
|---|---|
| Missing values reduced | `+1.0` |
| Duplicates removed | `+0.8` |
| Dataset already clean | `+0.0` (done) |
| Unnecessary operation | `-0.3` |
| Loop detected (same action repeated) | `-1.0` (hard penalty) |

**3. The anti-loop system is non-negotiable**

One of the hardest problems in RL agent design is the agent getting stuck in a reward-neutral cycle. This system tracks action history and applies an exponentially increasing penalty if the same action is selected consecutively — forcing the agent to explore or terminate.

---

## 🤖 The LLM Agent

The agent uses **Llama 3 via OpenRouter** as its reasoning engine. At each step, the agent receives a formatted prompt containing:

- The current observation (state)
- The full action space
- A brief history of recent actions and rewards
- An instruction to reason before selecting

This is a **zero-shot, context-window-driven** decision loop — no fine-tuning required. The LLM's world knowledge about data quality, combined with the structured reward signal from the environment, produces surprisingly coherent cleaning behavior.

---

## 🔄 Available Actions

| Action | Description | Best Used When |
|---|---|---|
| `fill_missing` | Fills null values (mean/mode/ffill) | Missing values present, data is numeric or sequential |
| `drop_rows_with_missing` | Removes rows with any null | Missing data is sparse, completeness > volume |
| `remove_duplicates` | Drops exact duplicate rows | Identical records detected in sample |

> Actions are intentionally minimal for this version — the reward system ensures they're used judiciously, not blindly.

---

## 🌐 Live Demo

The agent is deployed on **Hugging Face Spaces** and accessible via API:

**🔗 [https://chetangadhiya017-data-cleaning-env.hf.space](https://chetangadhiya017-data-cleaning-env.hf.space)**

You can interact with it directly or call the REST endpoints:

```bash
# Check environment status
curl https://chetangadhiya017-data-cleaning-env.hf.space/status

# Send a dataset observation and get an action
curl -X POST https://chetangadhiya017-data-cleaning-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"missing_values": 5, "duplicate_rows": 2, "data_sample": [...]}'
```

---

## ⚡ Setup & Installation

### Prerequisites

- Python 3.10+
- An [OpenRouter](https://openrouter.ai) API key (for Llama 3 inference)

### 1. Clone the repository

```bash
git clone https://github.com/chetangadhiya5062/open-env-nuclei.git
cd open-env-nuclei
```

### 2. Create a virtual environment

```bash
python -m venv .venv_scalar

# Activate — Linux/macOS
source .venv_scalar/bin/activate

# Activate — Windows
.\.venv_scalar\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root directory:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 5. Start the environment server

```bash
python -m data_cleaning_env.server.app
```

### 6. Run the agent

```bash
python -m data_cleaning_env.inference
```

You'll see the agent's decision loop print live to the console — observations, chosen actions, rewards received, and state transitions.

---

## 🧪 Example Session

```
[STEP 1] Observation: missing_values=12, duplicates=3
         → Agent selects: fill_missing
         → Reward: +1.0  |  Remaining missing: 0

[STEP 2] Observation: missing_values=0, duplicates=3
         → Agent selects: remove_duplicates
         → Reward: +0.8  |  Duplicates removed

[STEP 3] Observation: missing_values=0, duplicates=0
         → Agent selects: TERMINATE
         → Dataset is clean. Episode complete.

Total reward: 1.8  |  Steps: 3  |  Data loss: 0 rows
```

---

## 🐳 Docker Deployment

```bash
docker build -t openenv-agent .
docker run -p 8000:8000 -e OPENROUTER_API_KEY=your_key openenv-agent
```

---

## 🗺️ Roadmap

- [x] Core RL environment with reward shaping
- [x] Anti-loop penalty system
- [x] LLM action selection via OpenRouter
- [x] FastAPI server deployment
- [x] Hugging Face Spaces live demo
- [ ] **Phase 13**: Ground-truth evaluation system & benchmark scoring
- [ ] Gradio / Streamlit UI dashboard with step-by-step visualisation
- [ ] Multi-dataset support (CSV, JSON, Parquet)
- [ ] Smart normalization actions (`NY → New York`, date standardisation)
- [ ] Expanded action space (outlier detection, type coercion, schema validation)
- [ ] Benchmarking suite with real-world noisy datasets
- [ ] Fine-tuned reward model replacing handcrafted reward functions

---

## 🧩 Why This Matters

The standard approach to data cleaning is deterministic: write rules, apply rules. This works until your data changes — new sources, new schemas, new noise patterns.

An RL agent changes the contract. Instead of rules, you specify **what good looks like** (the reward). The agent figures out the how. This opens the door to:

- **Adaptive pipelines** that improve as they process more data
- **Transfer learning** across datasets with similar structure
- **Exploratory cleaning** in domains where best practices are unknown

This project is a minimal but complete proof of concept for that paradigm.

---

## 🤝 Contributing

Contributions, issues, and forks are very welcome — especially around reward design, action expansion, and the evaluation system.

```bash
git checkout -b feature/your-feature
git commit -m "feat: describe your change"
git push origin feature/your-feature
# Open a Pull Request
```

---

## 👨‍💻 Author

**Chetan Gadhiya**
- GitHub: [@chetangadhiya5062](https://github.com/chetangadhiya5062)
- Live demo: [HuggingFace Spaces](https://chetangadhiya017-data-cleaning-env.hf.space)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Meta AI](https://llama.meta.com) — Llama 3 open-weight model
- [OpenRouter](https://openrouter.ai) — Unified LLM inference API
- [Hugging Face](https://huggingface.co) — Spaces deployment platform
- [FastAPI](https://fastapi.tiangolo.com) — High-performance Python API framework
- [OpenAI Gym](https://gymnasium.farama.org) — Inspiration for the environment interface design

## 🤝 Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/chetangadhiya5062">
        <img src="https://github.com/chetangadhiya5062.png" width="100px;" alt=""/>
        <br />
        <sub><b>Chetan Gadhiya</b></sub>
      </a>
    </td>
   <td align="center">
      <a href="https://github.com/HilKalathiya">
        <img src="https://github.com/HilKalathiya.png" width="100px;" alt=""/>
        <br />
        <sub><b>Hil Kalathiya</b></sub>
      </a>
    </td>
  </tr>
</table>

---

<div align="center">

*If this project was useful or interesting — star it, fork it, break it, improve it.*

⭐ **Star the repo** · 🐛 **Open an issue** · 🔀 **Submit a PR**

</div>
