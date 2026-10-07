import json

import pytest
from botocore.exceptions import ClientError

from lambda_api import handler


class _FakeClient:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    def retrieve_and_generate(self, **kwargs):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_ID", "KB123")
    monkeypatch.setenv("MODEL_ARN", "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-lite-v1:0")
    monkeypatch.setenv("GUARDRAIL_ID", "gr-abc")
    monkeypatch.setenv("GUARDRAIL_VERSION", "1")
    monkeypatch.setenv("NUM_RESULTS", "5")


def _event(body):
    return {"body": body if isinstance(body, str) else json.dumps(body)}


def _use_client(monkeypatch, outcome):
    client = _FakeClient(outcome)
    monkeypatch.setattr(handler, "get_client", lambda: client)
    return client


def _client_error(code):
    return ClientError({"Error": {"Code": code, "Message": "boom"}}, "RetrieveAndGenerate")


GOOD_RESPONSE = {
    "output": {"text": "Use FastAPI() to create the app."},
    "citations": [
        {
            "retrievedReferences": [
                {
                    "content": {"text": "Create a FastAPI app."},
                    "location": {"s3Location": {"uri": "s3://b/docs/tutorial/first-steps.md"}},
                    "metadata": {
                        "source_url": "https://fastapi.tiangolo.com/tutorial/first-steps/",
                        "title": "First Steps",
                    },
                }
            ]
        }
    ],
    "sessionId": "s1",
}


def test_success_returns_answer_and_sources(monkeypatch):
    _use_client(monkeypatch, GOOD_RESPONSE)
    result = handler.lambda_handler(_event({"question": "How do I start?"}), None)
    body = json.loads(result["body"])
    assert result["statusCode"] == 200
    assert body["covered"] is True and body["grounded"] is True
    assert body["sources"][0]["title"] == "First Steps"
    assert body["session_id"] == "s1"


def test_not_covered_is_a_normal_200(monkeypatch):
    _use_client(monkeypatch, {"output": {"text": "NOT_COVERED"}})
    result = handler.lambda_handler(_event({"question": "How do I train a neural network?"}), None)
    body = json.loads(result["body"])
    assert result["statusCode"] == 200
    assert body["covered"] is False and body["sources"] == []


def test_guardrail_block_is_reported(monkeypatch):
    _use_client(monkeypatch, {"output": {"text": "Blocked."}, "guardrailAction": "INTERVENED"})
    body = json.loads(handler.lambda_handler(_event({"question": "bad"}), None)["body"])
    assert body["blocked"] is True


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": 5},
        {"question": "x" * 2000},
        {"question": "ok", "session_id": "bad id!"},
    ],
)
def test_invalid_input_returns_400_without_calling_bedrock(monkeypatch, body):
    client = _use_client(monkeypatch, GOOD_RESPONSE)
    result = handler.lambda_handler(_event(body), None)
    assert result["statusCode"] == 400
    assert client.calls == 0


def test_invalid_json_returns_400(monkeypatch):
    _use_client(monkeypatch, GOOD_RESPONSE)
    assert handler.lambda_handler(_event("{not json"), None)["statusCode"] == 400


def test_non_object_json_returns_400(monkeypatch):
    _use_client(monkeypatch, GOOD_RESPONSE)
    assert handler.lambda_handler(_event("[1, 2]"), None)["statusCode"] == 400


def test_throttling_maps_to_429(monkeypatch):
    _use_client(monkeypatch, _client_error("ThrottlingException"))
    assert handler.lambda_handler(_event({"question": "q"}), None)["statusCode"] == 429


def test_other_aws_errors_map_to_502_without_leaking_details(monkeypatch):
    _use_client(monkeypatch, _client_error("AccessDeniedException"))
    result = handler.lambda_handler(_event({"question": "q"}), None)
    assert result["statusCode"] == 502
    assert "AccessDenied" not in result["body"]


def test_unexpected_error_maps_to_500(monkeypatch):
    _use_client(monkeypatch, RuntimeError("kaboom"))
    result = handler.lambda_handler(_event({"question": "q"}), None)
    assert result["statusCode"] == 500
    assert "kaboom" not in result["body"]
