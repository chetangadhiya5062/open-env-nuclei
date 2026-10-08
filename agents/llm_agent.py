"""LLM agent: an OpenAI-compatible chat model (default: Llama 3 8B via the Hugging Face router).

Zero-shot prompting only: nothing is trained. Behaviour you can rely on:

* The model must answer with one JSON object. It is validated with Pydantic (``DataCleaningAction`` and
  the strict models in ``data_cleaning_env.actions``).
* If the answer is invalid, the agent re-asks (up to ``max_retries``) and shows the model the exact error.
* There are NO hidden overrides of what the model chose. If every retry fails, the agent sends a clearly
  invalid action (``invalid_llm_output``) so the environment penalises it, logs a warning and increments
  ``fallbacks``. The benchmark reports that counter.
"""

import json
import logging
import os
import re
from typing import Any, Deque, Dict, List, Optional, Tuple
from collections import deque

from pydantic import ValidationError

from data_cleaning_env.actions import ACTION_TYPES, USAGE, parse_action
from data_cleaning_env.models import DataCleaningAction, DataCleaningObservation

from .base import Agent

logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
HISTORY_LENGTH = 5

SYSTEM_PROMPT = f"""You are a careful data-cleaning agent working on a pandas table, one action per turn.
Reply with exactly ONE JSON object and nothing else (no markdown, no comments).

Valid actions:
{chr(10).join('- ' + USAGE[t] for t in ACTION_TYPES)}

Guidance:
- Each column shows `issues` (what is wrong and how many cells), `allowed_values`, `bad_values` and `valid_range`.
- Fix problems one column at a time; do not repeat an action that already succeeded.
- Numeric columns that contain text (issue wrong_type) need cast_type before fill_missing / clip_outliers.
- Map leftover category spellings to allowed values with standardize_categories + mapping.
- Use drop_duplicates for duplicate rows, fix_dates for bad_date_format, strip_whitespace for whitespace.
- Prefer fixing over dropping rows: dropping rows loses data and is penalised.
- When quality_score is 1.0 or no issues remain, send {{"action_type":"finish"}}."""


def format_observation(obs: DataCleaningObservation, history: List[str]) -> str:
    lines = [
        f"Task {obs.task_id}: {obs.row_count} rows, {obs.duplicate_count} duplicate rows, "
        f"{obs.total_missing} missing cells. quality_score={obs.quality_score:.3f}. "
        f"Steps used {obs.step_count}, remaining {obs.steps_remaining}.",
        "Columns:",
    ]
    for c in obs.columns:
        parts = [f"- {c.name} ({c.expected_type}, dtype={c.dtype}) missing={c.missing} unique={c.unique}"]
        if c.issues:
            parts.append(f"issues={json.dumps(c.issues)}")
        if c.allowed_values:
            parts.append(f"allowed={c.allowed_values}")
        if c.bad_values:
            parts.append(f"bad_values={c.bad_values}")
        if c.valid_range:
            parts.append(f"valid_range={c.valid_range}")
        parts.append(f"samples={c.sample_values}")
        lines.append(" ".join(parts))
    lines.append(f"First rows: {json.dumps(obs.data_sample[:3], default=str)}")
    if history:
        lines.append("Recent actions (oldest first):")
        lines.extend(history)
    if obs.last_error:
        lines.append(f"LAST ACTION WAS REJECTED: {obs.last_error}")
    lines.append("Reply with the next action as one JSON object.")
    return "\n".join(lines)


def extract_json(text: str) -> Dict[str, Any]:
    """Pull the first JSON object out of a model reply (tolerates code fences and chatter)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in the reply")
    depth, in_str, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                data = json.loads(text[start : i + 1])
                if not isinstance(data, dict):
                    raise ValueError("reply is not a JSON object")
                return data
    raise ValueError("unterminated JSON object in the reply")


def validate_reply(text: str) -> DataCleaningAction:
    """Raise ValueError with an agent-readable message if the reply is not a valid action."""
    try:
        raw = extract_json(text)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse JSON: {exc}") from None
    raw = {k: v for k, v in raw.items() if v is not None}
    if isinstance(raw.get("value"), (int, float)) and not isinstance(raw.get("value"), bool):
        raw["value"] = str(raw["value"])
    try:
        action = DataCleaningAction(**raw)
    except ValidationError as exc:
        raise ValueError(f"unexpected fields: {exc.errors()[0]['msg']} ({exc.errors()[0]['loc']})") from None
    _, error = parse_action(action.to_payload())
    if error:
        raise ValueError(error)
    return action


class LLMAgent(Agent):
    name = "llm"

    def __init__(
        self,
        client: Any = None,
        model: Optional[str] = None,
        max_retries: int = 2,
        temperature: float = 0.0,
    ) -> None:
        self.model = model or os.getenv("MODEL_NAME", DEFAULT_MODEL)
        self.max_retries = max_retries
        self.temperature = temperature
        self._client = client
        self._history: Deque[str] = deque(maxlen=HISTORY_LENGTH)
        self._last: Optional[Tuple[str, str]] = None
        self.llm_calls = 0
        self.name = f"llm:{self.model.split('/')[-1]}"

    @staticmethod
    def available() -> bool:
        return bool(os.getenv("HF_TOKEN"))

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            token = os.getenv("HF_TOKEN")
            if not token:
                raise RuntimeError("HF_TOKEN is not set (see .env.example)")
            self._client = OpenAI(base_url=os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL), api_key=token)
        return self._client

    def reset(self, seed: int = 0) -> None:
        super().reset(seed)
        self._history.clear()
        self._last = None
        self.llm_calls = 0

    def _complete(self, messages: List[Dict[str, str]]) -> str:
        self.llm_calls += 1
        resp = self._get_client().chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, max_tokens=400
        )
        return resp.choices[0].message.content or ""

    def _record_previous(self, obs: DataCleaningObservation) -> None:
        if self._last is not None:
            status = "ok" if obs.last_action_ok else f"REJECTED ({obs.last_error})"
            self._history.append(f"- {self._last[0]} -> {status}; reward={obs.reward}")

    def act(self, obs: DataCleaningObservation) -> DataCleaningAction:
        self._record_previous(obs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_observation(obs, list(self._history))},
        ]
        last_error = ""
        for attempt in range(self.max_retries + 1):
            reply = self._complete(messages)
            try:
                action = validate_reply(reply)
                self._last = (json.dumps(action.to_payload()), "")
                return action
            except ValueError as exc:
                last_error = str(exc)
                logger.warning("invalid LLM reply (attempt %d/%d): %s", attempt + 1, self.max_retries + 1, last_error)
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"That reply was invalid: {last_error}\nReply again with one valid JSON action."})
        self.fallbacks += 1
        logger.warning("LLM fallback #%d: sending an invalid action after %d failed attempts", self.fallbacks, self.max_retries + 1)
        self._last = ("invalid_llm_output", last_error)
        return DataCleaningAction(action_type="invalid_llm_output")
