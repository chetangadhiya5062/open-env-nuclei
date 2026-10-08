"""API smoke tests with FastAPI's TestClient (no real network)."""

import pytest
from fastapi.testclient import TestClient

from data_cleaning_env.server.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_and_schema(client):
    assert client.get("/health").json()["status"] == "healthy"
    schema = client.get("/schema").json()
    assert "action" in schema and "observation" in schema


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/docs"


def test_http_reset_selects_task_and_seed(client):
    body = client.post("/reset", json={"task_id": "medium-clean", "seed": 5}).json()
    obs = body["observation"]
    assert obs["task_id"] == "medium-clean" and obs["seed"] == 5
    assert obs["duplicate_count"] > 0 and body["done"] is False
    again = client.post("/reset", json={"task_id": "medium-clean", "seed": 5}).json()["observation"]
    assert again["data_sample"] == obs["data_sample"]  # seeded -> reproducible


def test_http_step_is_stateless_but_does_not_crash(client):
    r = client.post("/step", json={"action": {"action_type": "drop_duplicates"}})
    assert r.status_code == 200
    r = client.post("/step", json={"action": {"action_type": "not_an_action"}})
    assert r.status_code == 200 and r.json()["observation"]["last_action_ok"] is False


def test_http_rejects_malformed_payload(client):
    assert client.post("/step", json={"action": {"column_name": "age"}}).status_code == 422


def test_websocket_session_keeps_state_across_steps(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "reset", "data": {"task_id": "medium-clean", "seed": 2}})
        first = ws.receive_json()["data"]
        assert first["observation"]["duplicate_count"] > 0

        ws.send_json({"type": "step", "data": {"action_type": "drop_duplicates"}})
        second = ws.receive_json()["data"]
        assert second["observation"]["duplicate_count"] == 0
        assert second["reward"] > 0 and second["done"] is False

        ws.send_json({"type": "step", "data": {"action_type": "finish"}})
        last = ws.receive_json()["data"]
        assert last["done"] is True
        assert 0.0 <= last["observation"]["final_score"] <= 1.0
        assert last["observation"]["step_count"] == 2
