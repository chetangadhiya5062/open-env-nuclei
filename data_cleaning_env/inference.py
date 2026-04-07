import json
import requests
import os

from data_cleaning_env.client import DataCleaningEnv
from data_cleaning_env.models import DataCleaningAction

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ✅ SAFETY FIX (add this)
if not OPENROUTER_API_KEY:
    raise Exception("OPENROUTER_API_KEY not set")

ENV_URL = "http://localhost:8001"


def run_episode():
    with DataCleaningEnv(base_url=ENV_URL).sync() as env:
        result = env.reset()
        done = False

        while not done:
            obs = result.observation

            prompt = f"""
You are a data cleaning agent.

Your PRIMARY goal is to preserve as much data as possible.

STRICT RULES:
- NEVER drop rows unless they are exact duplicates
- ALWAYS prefer fill_missing over dropping rows
- Dropping rows will be heavily penalized

Available columns: {obs.column_names}
Missing values: {obs.missing_values_count_per_column}
Duplicate rows: {obs.duplicate_row_count}

Choose ONE action:
- fill_missing (requires column_name and value)
- drop_rows_with_missing
- remove_duplicates

Return ONLY valid JSON in this format:
{{
  "action_type": "...",
  "column_name": null or "column",
  "value": null or "value"
}}

Only return valid JSON. No explanation.
"""

            models_to_try = [
                "qwen/qwen3.6-plus:free",
                "nvidia/nemotron-3-super-120b-a12b:free"
            ]

            response_json = None

            for model_name in models_to_try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": "You are a data cleaning agent."},
                            {"role": "user", "content": prompt}
                        ]
                    }
                )

                response_json = response.json()

                print(f"\nTrying model: {model_name}")
                print("RAW RESPONSE:", response_json)

                if "choices" in response_json:
                    break

            if "choices" not in response_json:
                raise Exception(f"All models failed: {response_json}")

            action_text = response_json["choices"][0]["message"]["content"].strip()

            if action_text.startswith("```"):
                action_text = action_text.split("```")[1]

            action_text = action_text.replace("json", "").strip()

            try:
                action_json = json.loads(action_text)
            except Exception as e:
                print("Failed to parse JSON:", action_text)
                raise e

            # Fix type mismatch (LLM may return int)
            if "value" in action_json and action_json["value"] is not None:
                action_json["value"] = str(action_json["value"])

            action = DataCleaningAction(**action_json)

            result = env.step(action)
            done = result.observation.done

        print("\n✅ Final Reward:", result.observation.reward)


if __name__ == "__main__":
    run_episode()