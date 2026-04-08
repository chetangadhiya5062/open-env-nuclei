import json
import os
import re

from data_cleaning_env.client import DataCleaningEnv
from data_cleaning_env.models import DataCleaningAction

from openai import OpenAI

# =========================
# 🔥 ENV VARIABLES (REQUIRED)
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise Exception("HF_TOKEN not set")

ENV_URL = "http://localhost:8001"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)


def call_model(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an intelligent data cleaning agent. Always return valid JSON."},
            {"role": "user", "content": prompt}
        ]
    )
    return response


def run_episode():
    step_num = 0
    rewards = []

    with DataCleaningEnv(base_url=ENV_URL).sync() as env:
        result = env.reset()
        done = False

        # ✅ START FORMAT
        print(f"[START] task=data-cleaning env=openenv model={MODEL_NAME}", flush=True)

        try:
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

Return ONLY valid JSON:
{{
  "action_type": "...",
  "column_name": "...",
  "value": "..."
}}
"""

                response = call_model(prompt)

                try:
                    action_text = response.choices[0].message.content
                except Exception:
                    raise Exception(f"HF failed: {response}")

                # CLEAN OUTPUT
                if "```" in action_text:
                    action_text = action_text.split("```")[1]

                action_text = action_text.replace("json", "").strip()
                action_text = re.sub(r"#.*", "", action_text)
                action_text = action_text.replace(",}", "}").replace(",]", "]")

                try:
                    action_json = json.loads(action_text)

                    # 🔥 FIX: handle list response from LLM
                    if isinstance(action_json, list):
                        action_json = action_json[0]
                        
                except Exception:
                    action_json = {
                        "action_type": "fill_missing",
                        "column_name": obs.column_names[0],
                        "value": "missing"
                    }

                if action_json["action_type"] == "fill_missing":
                    col = action_json["column_name"]
                    if obs.missing_values_count_per_column.get(col, 0) == 0:
                        for c, v in obs.missing_values_count_per_column.items():
                            if v > 0:
                                action_json["column_name"] = c
                                break

                if "value" in action_json and action_json["value"] is not None:
                    action_json["value"] = str(action_json["value"])

                action = DataCleaningAction(**action_json)

                result = env.step(action)
                obs = result.observation

                step_num += 1
                reward = obs.reward if obs.reward is not None else 0.0
                rewards.append(reward)

                # ✅ STEP FORMAT
                print(
                    f"[STEP] step={step_num} action={action_json} "
                    f"reward={reward:.2f} done={str(obs.done).lower()} error=null",
                    flush=True
                )

                if sum(obs.missing_values_count_per_column.values()) == 0:
                    done = True

                done = done or obs.done

        finally:
            total_steps = step_num
            total_reward = sum(rewards)

            # normalize score to [0,1]
            score = min(max(total_reward / 10.0, 0.0), 1.0)
            success = str(score > 0).lower()

            rewards_str = ",".join([f"{r:.2f}" for r in rewards])

            # ✅ END FORMAT
            print(
                f"[END] success={success} steps={total_steps} score={score:.2f} rewards={rewards_str}",
                flush=True
            )


if __name__ == "__main__":
    run_episode()