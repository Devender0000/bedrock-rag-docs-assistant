"""UI smoke tests using Streamlit's built-in AppTest runner (no browser, no AWS)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")

COVERED_RESULT = {
    "answer": "Use FastAPI() to create the app.",
    "covered": True,
    "blocked": False,
    "grounded": True,
    "sources": [
        {
            "title": "First Steps",
            "url": "https://fastapi.tiangolo.com/tutorial/first-steps/",
            "snippet": "Create a FastAPI app.",
        }
    ],
    "session_id": "s1",
}

NOT_COVERED_RESULT = {
    "answer": "I couldn't find that in the FastAPI documentation.",
    "covered": False,
    "blocked": False,
    "grounded": False,
    "sources": [],
    "session_id": "s1",
}


def _configured_app(monkeypatch, fake_ask):
    monkeypatch.setenv("DOCS_ASSISTANT_API_URL", "https://example.test/prod/ask")
    monkeypatch.setenv("DOCS_ASSISTANT_API_KEY", "test-key")
    monkeypatch.setattr("app.api_client.ask", fake_ask)
    return AppTest.from_file(APP_PATH, default_timeout=10).run()


def test_missing_configuration_shows_setup_warning(monkeypatch):
    monkeypatch.delenv("DOCS_ASSISTANT_API_URL", raising=False)
    monkeypatch.delenv("DOCS_ASSISTANT_API_KEY", raising=False)
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()
    assert not app.exception
    assert any("DOCS_ASSISTANT_API_URL" in w.value for w in app.warning)
    assert len(app.chat_input) == 0


def test_question_shows_answer_and_sources(monkeypatch):
    calls = []

    def fake_ask(url, key, question, session_id=None, timeout=40):
        calls.append((url, key, question, session_id))
        return COVERED_RESULT

    app = _configured_app(monkeypatch, fake_ask)
    app.chat_input[0].set_value("How do I start?").run()

    assert not app.exception
    assert calls == [("https://example.test/prod/ask", "test-key", "How do I start?", None)]
    assert any("Use FastAPI() to create the app." in m.value for m in app.markdown)
    assert any("Sources (1)" in e.label for e in app.expander)


def test_not_covered_answer_is_shown_as_info(monkeypatch):
    app = _configured_app(monkeypatch, lambda *a, **k: NOT_COVERED_RESULT)
    app.chat_input[0].set_value("How do I train a neural network?").run()
    assert not app.exception
    assert any("couldn't find that" in i.value for i in app.info)


def test_api_error_is_shown_to_user(monkeypatch):
    from app.api_client import ApiError

    def failing_ask(*args, **kwargs):
        raise ApiError("Too many requests right now.")

    app = _configured_app(monkeypatch, failing_ask)
    app.chat_input[0].set_value("anything").run()
    assert not app.exception
    assert any("Too many requests" in e.value for e in app.error)
