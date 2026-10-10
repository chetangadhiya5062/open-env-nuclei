import json
from types import SimpleNamespace

import pytest

from agents import DoNothingAgent, LLMAgent, LocalEnv, RandomAgent, RuleBasedAgent, run_episode
from agents.llm_agent import extract_json, validate_reply
from agents.rule_based import suggest_mapping


def test_random_agent_is_reproducible():
    env = LocalEnv()
    a = run_episode(RandomAgent(seed=1), env, "medium-clean", 0)
    b = run_episode(RandomAgent(seed=1), env, "medium-clean", 0)
    assert a == b


@pytest.mark.parametrize("task_id", ["easy-clean", "medium-clean", "hard-clean"])
def test_rule_based_terminates_without_invalid_actions_and_scores_high(task_id):
    result = run_episode(RuleBasedAgent(), LocalEnv(), task_id, 0)
    assert result.invalid_actions == 0
    assert result.final_score > 0.9
    assert result.steps < 20


def test_do_nothing_finishes_in_one_step():
    assert run_episode(DoNothingAgent(), LocalEnv(), "easy-clean", 0).steps == 1


def test_suggest_mapping_matches_aliases_without_ambiguity():
    allowed = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
    m = suggest_mapping(["NY", "N.Y.", "NYC", "LA", "L.A.", "Chi", "???", "H"], allowed)
    assert m == {
        "NY": "New York",
        "N.Y.": "New York",
        "NYC": "New York",
        "LA": "Los Angeles",
        "L.A.": "Los Angeles",
        "Chi": "Chicago",
    }


# ---------------------------------------------------------------- LLM agent with a fake client
class FakeClient:
    """Returns scripted replies, one per call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs["messages"])
        text = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def first_obs():
    return LocalEnv().reset("easy-clean", 0)


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": {"b": "}"}}\n```\nbye') == {"a": {"b": "}"}}
    with pytest.raises(ValueError):
        extract_json("no json here")
    with pytest.raises(ValueError):
        extract_json('{"a": 1')


def test_validate_reply_accepts_numbers_as_values_and_rejects_bad_actions():
    a = validate_reply('{"action_type":"fill_missing","column_name":"age","strategy":"constant","value":30}')
    assert a.value == "30"
    with pytest.raises(ValueError, match="unknown action_type"):
        validate_reply('{"action_type":"delete_everything"}')
    with pytest.raises(ValueError, match="unexpected fields"):
        validate_reply('{"action_type":"finish","bogus":1}')


def test_llm_agent_parses_valid_reply_without_retry():
    client = FakeClient(['{"action_type":"fill_missing","column_name":"age","strategy":"median"}'])
    agent = LLMAgent(client=client, model="x/test")
    agent.reset()
    action = agent.act(first_obs())
    assert action.action_type == "fill_missing" and len(client.calls) == 1 and agent.fallbacks == 0


def test_llm_agent_retries_with_error_message_then_succeeds():
    client = FakeClient(["I think we should fill ages.", '{"action_type":"finish"}'])
    agent = LLMAgent(client=client, model="x/test")
    agent.reset()
    action = agent.act(first_obs())
    assert action.action_type == "finish" and agent.fallbacks == 0
    retry_prompt = client.calls[1][-1]["content"]
    assert "invalid" in retry_prompt and "could not parse JSON" in retry_prompt


def test_llm_agent_fallback_is_logged_counted_and_is_an_invalid_action(caplog):
    client = FakeClient(["nope"] * 3)
    agent = LLMAgent(client=client, model="x/test", max_retries=2)
    agent.reset()
    action = agent.act(first_obs())
    assert action.action_type == "invalid_llm_output"  # not silently replaced by a good action
    assert agent.fallbacks == 1
    assert "LLM fallback" in caplog.text


def test_llm_agent_full_episode_counts_fallback_via_environment():
    replies = ["garbage", "garbage", "garbage", '{"action_type":"finish"}']
    agent = LLMAgent(client=FakeClient(replies), model="x/test")
    result = run_episode(agent, LocalEnv(), "easy-clean", 0)
    assert result.fallbacks == 1 and result.invalid_actions == 1 and result.steps == 2


def test_llm_prompt_contains_history_and_last_error():
    client = FakeClient(
        [
            '{"action_type":"fill_missing","column_name":"age","strategy":"constant","value":"missing"}',
            '{"action_type":"finish"}',
        ]
    )
    agent = LLMAgent(client=client, model="x/test")
    run_episode(agent, LocalEnv(), "easy-clean", 0)
    second_user_prompt = client.calls[1][1]["content"]
    assert "REJECTED" in second_user_prompt and "Recent actions" in second_user_prompt
    json.loads(first_obs().model_dump_json())  # observation is JSON-serialisable


class FailingClient:
    """Every call raises, like an unknown model or a rate limit."""

    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        raise RuntimeError("429 rate limited")


def test_llm_api_errors_are_retried_counted_and_do_not_crash_the_episode():
    client = FailingClient()
    agent = LLMAgent(client=client, model="x/test", api_retries=1, retry_sleep=0)
    action = agent.act(first_obs())
    assert action.action_type == "invalid_llm_output"
    assert client.calls == 2 and agent.api_errors == 1 and agent.fallbacks == 1
    assert agent.probe() is not None  # probe reports the error instead of raising


def test_llm_recovers_after_a_transient_api_error():
    replies = [RuntimeError("503"), '{"action_type":"finish"}']

    class Flaky(FakeClient):
        def _create(self, **kwargs):
            r = self.replies.pop(0)
            if isinstance(r, Exception):
                raise r
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=r))])

    agent = LLMAgent(client=Flaky(replies), model="x/test", api_retries=2, retry_sleep=0)
    assert agent.act(first_obs()).action_type == "finish" and agent.api_errors == 0
