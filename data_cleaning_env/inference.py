import json
import requests
import os
import re

from data_cleaning_env.client import DataCleaningEnv
from data_cleaning_env.models import DataCleaningAction

# =========================
# 🔥 HUGGINGFACE SETUP
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise Exception("HF_TOKEN not set")

# =========================
# 💤 OPENROUTER (BACKUP)
# =========================
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

ENV_URL = "http://localhost:8001"


def call_hf_model(prompt):
    API_URL = "https://router.huggingface.co/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",  # 🔥 CHANGED
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    return response.json()


def run_episode():
    with DataCleaningEnv(base_url=ENV_URL).sync() as env:
        result = env.reset()
        done = False

        while not done:
            obs = result.observation

            prompt = f"""
You are a data cleaning agent.

Your PRIMARY goal is to clean ALL missing values.

STRICT RULES:
- NEVER repeat the same useless action
- ALWAYS move to a different column if one is already clean
- DO NOT fill a column if it has no missing values

DATA SAMPLE:
{obs.data_sample}

CURRENT STATE:
Columns: {obs.column_names}
Missing values: {obs.missing_values_count_per_column}
Duplicate rows: {obs.duplicate_row_count}

TASK:
Choose the BEST next action to reduce missing values.

Available actions:
- fill_missing (requires column_name and value)
- remove_duplicates

Return ONLY valid JSON:
{{
  "action_type": "...",
  "column_name": "...",
  "value": "..."
}}
"""

            # =========================
            # 🔥 HUGGINGFACE CALL
            # =========================
            response_json = call_hf_model(prompt)

            print("\nRAW RESPONSE:", response_json)

            try:
                action_text = response_json["choices"][0]["message"]["content"]
            except Exception:
                raise Exception(f"HF failed: {response_json}")

            # =========================
            # CLEAN OUTPUT
            # =========================
            if "```" in action_text:
                action_text = action_text.split("```")[1]

            action_text = action_text.replace("json", "").strip()

            # 🔥 CLEAN INVALID JSON (REMOVE COMMENTS)
            action_text = re.sub(r"#.*", "", action_text)

            # Remove trailing commas (optional safety)
            action_text = action_text.replace(",}", "}").replace(",]", "]")

            try:
                action_json = json.loads(action_text)
            except Exception as e:
                print("❌ Failed to parse JSON:", action_text)

                # ✅ FALLBACK ACTION (ADD HERE)
                action_json = {
                    "action_type": "fill_missing",
                    "column_name": obs.column_names[0],
                    "value": "missing"
                }

            if "value" in action_json and action_json["value"] is not None:
                action_json["value"] = str(action_json["value"])

            action = DataCleaningAction(**action_json)

            result = env.step(action)
            obs = result.observation

            # 🔥 FORCE STOP FROM CLIENT SIDE
            if sum(obs.missing_values_count_per_column.values()) == 0:
                print("🔥 CLIENT: Data cleaned → stopping early")
                break

            done = obs.done

        print("\n✅ Final Reward:", result.observation.reward)


if __name__ == "__main__":
    run_episode()