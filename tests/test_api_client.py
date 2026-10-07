import pytest
import requests

from app import api_client
from app.api_client import ApiError, ask


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, json_error=False):
        self.status_code = status_code
        self._payload = payload
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise ValueError("not json")
        return self._payload


def _patch_post(monkeypatch, outcome):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(api_client.requests, "post", fake_post)
    return calls


def test_success_sends_key_and_question(monkeypatch):
    calls = _patch_post(monkeypatch, _FakeResponse(200, {"answer": "hi"}))
    result = ask("https://api/ask", "KEY", "How?", session_id="s1")
    assert result == {"answer": "hi"}
    assert calls[0]["headers"]["x-api-key"] == "KEY"
    assert calls[0]["json"] == {"question": "How?", "session_id": "s1"}


def test_session_id_omitted_when_none(monkeypatch):
    calls = _patch_post(monkeypatch, _FakeResponse(200, {}))
    ask("https://api/ask", "KEY", "How?")
    assert calls[0]["json"] == {"question": "How?"}


@pytest.mark.parametrize(
    "status, fragment",
    [(400, "rephras"), (403, "API key"), (429, "Too many"), (500, "couldn't answer")],
)
def test_error_statuses_have_friendly_messages(monkeypatch, status, fragment):
    _patch_post(monkeypatch, _FakeResponse(status))
    with pytest.raises(ApiError, match=fragment):
        ask("https://api/ask", "KEY", "q")


def test_timeout_message(monkeypatch):
    _patch_post(monkeypatch, requests.Timeout())
    with pytest.raises(ApiError, match="timed out"):
        ask("https://api/ask", "KEY", "q")


def test_connection_error_message(monkeypatch):
    _patch_post(monkeypatch, requests.ConnectionError())
    with pytest.raises(ApiError, match="reach the API"):
        ask("https://api/ask", "KEY", "q")


def test_unreadable_success_body(monkeypatch):
    _patch_post(monkeypatch, _FakeResponse(200, json_error=True))
    with pytest.raises(ApiError, match="unreadable"):
        ask("https://api/ask", "KEY", "q")
