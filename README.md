# OpenEnv Data Cleaning Environment

A small [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment for **tabular data cleaning**, plus a
**zero-shot LLM agent** (`inference.py`) that talks to it. The server is deployed on Hugging Face Spaces:
<https://chetangadhiya017-data-cleaning-env.hf.space>

> **Status: v2 rewrite in progress.** This README only states what the code currently does. The environment is
> being rebuilt (seeded datasets, easy/medium/hard tasks, a grader, tests, benchmarks) on branch `v2`.

## What it does today

- The environment holds a pandas `DataFrame`. An agent sends one cleaning action per step and receives an
  observation (missing-value counts, duplicate count, row count, first 5 rows) and a reward.
- Actions: `fill_missing`, `drop_rows_with_missing`, `remove_duplicates`, `finish_cleaning`.
- `inference.py` asks an LLM (Llama 3 8B Instruct through the **Hugging Face router**, using the OpenAI client)
  for the next action as JSON. It is **zero-shot prompting: the agent does not learn or get trained.**
- The client still contains hand-written overrides (it forces a column switch and stops early on its own).
  These are being removed in the agent rewrite.

## Known limitations of the current (pre-v2) environment

- The dataset is a hard-coded 5-row table.
- `openenv.yaml` declares `easy-clean` / `medium-clean` / `hard-clean`, but they are not implemented yet.
- The reward function can be exploited and there is no grader / 0-1 score yet. Both are fixed in v2.

## Quick start

Requires Python 3.10+. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# terminal 1: start the server (port from $env:PORT, default 7860)
python -m data_cleaning_env.server.app

# terminal 2: run the LLM agent (needs HF_TOKEN, see .env.example)
$env:HF_TOKEN = "hf_..."
python inference.py
```

Configuration is via environment variables (see `.env.example`): `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME`,
`PORT` (server, default 7860) and `ENV_URL` (client, default `http://localhost:7860`).

### API notes

The server exposes `/health`, `/schema`, `/reset`, `/step`, `/state` and a WebSocket at `/ws`. `/reset` and
`/step` over plain HTTP are **stateless** (each call creates a fresh environment). For a real multi-step
episode use the WebSocket client (`DataCleaningEnv(...).sync()`), as `inference.py` does.

### Docker

```bash
docker build -t openenv-data-cleaning .
docker run -p 7860:7860 openenv-data-cleaning
```

## License

MIT, see [LICENSE](LICENSE).

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
   <td align="center">
      <a href="https://github.com/SahajIVVIX-1">
        <img src="https://github.com/SahajIVVIX-1.png" width="100px;" alt=""/>
        <br />
        <sub><b>Sahaj Saliya</b></sub>
      </a>
    </td>
  </tr>
</table>

