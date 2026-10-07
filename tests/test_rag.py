import pytest

from lambda_api import rag

CONFIG = rag.RagConfig(
    knowledge_base_id="KB123",
    model_arn="arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.nova-lite-v1:0",
    guardrail_id="gr-abc",
    guardrail_version="1",
    num_results=5,
)


def _citation(
    url="https://fastapi.tiangolo.com/tutorial/first-steps/",
    title="First Steps",
    text="Create a FastAPI app.",
):
    return {
        "retrievedReferences": [
            {
                "content": {"text": text},
                "location": {"type": "S3", "s3Location": {"uri": "s3://bucket/docs/tutorial/first-steps.md"}},
                "metadata": {"source_url": url, "title": title},
            }
        ]
    }


def test_from_env_reads_all_settings():
    env = {
        "KNOWLEDGE_BASE_ID": "KB1",
        "MODEL_ARN": "arn:model",
        "GUARDRAIL_ID": "g",
        "GUARDRAIL_VERSION": "2",
        "NUM_RESULTS": "7",
    }
    config = rag.RagConfig.from_env(env)
    assert (config.knowledge_base_id, config.guardrail_version, config.num_results) == ("KB1", "2", 7)


def test_build_request_includes_guardrail_prompt_and_retrieval_settings():
    request = rag.build_request("How do I add a path parameter?", CONFIG)
    kb = request["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"]
    assert request["input"]["text"] == "How do I add a path parameter?"
    assert kb["knowledgeBaseId"] == "KB123"
    assert kb["modelArn"] == CONFIG.model_arn
    assert kb["retrievalConfiguration"]["vectorSearchConfiguration"]["numberOfResults"] == 5
    assert kb["generationConfiguration"]["guardrailConfiguration"] == {
        "guardrailId": "gr-abc",
        "guardrailVersion": "1",
    }
    template = kb["generationConfiguration"]["promptTemplate"]["textPromptTemplate"]
    assert "$search_results$" in template and "$query$" in template
    assert "sessionId" not in request


def test_build_request_matches_the_real_aws_api_schema():
    """Validate against botocore's bundled service model (offline, no credentials)."""
    boto3 = pytest.importorskip("boto3")
    from botocore.validate import validate_parameters

    client = boto3.client(
        "bedrock-agent-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    shape = client.meta.service_model.operation_model("RetrieveAndGenerate").input_shape
    validate_parameters(rag.build_request("q", CONFIG, "sess-1"), shape)
    validate_parameters(rag.build_request("q", CONFIG), shape)


def test_build_request_adds_session_id_when_given():
    assert rag.build_request("q", CONFIG, "sess-1")["sessionId"] == "sess-1"


@pytest.mark.parametrize("value", [None, "", "   ", 42, "x" * 1001])
def test_validate_question_rejects_bad_input(value):
    with pytest.raises(ValueError):
        rag.validate_question(value)


def test_validate_question_strips_whitespace():
    assert rag.validate_question("  hello  ") == "hello"


@pytest.mark.parametrize("value, expected", [(None, None), ("", None), ("abc-123_x", "abc-123_x")])
def test_validate_session_id_accepts(value, expected):
    assert rag.validate_session_id(value) == expected


@pytest.mark.parametrize("value", ["has space", "x" * 101, 5, "semi;colon"])
def test_validate_session_id_rejects(value):
    with pytest.raises(ValueError):
        rag.validate_session_id(value)


def test_parse_response_grounded_answer_with_sources():
    raw = {
        "output": {"text": "Use FastAPI() to create the app."},
        "citations": [_citation()],
        "sessionId": "s1",
    }
    result = rag.parse_response(raw)
    assert result["covered"] and result["grounded"] and not result["blocked"]
    assert result["session_id"] == "s1"
    assert result["sources"] == [
        {
            "title": "First Steps",
            "url": "https://fastapi.tiangolo.com/tutorial/first-steps/",
            "snippet": "Create a FastAPI app.",
        }
    ]


def test_parse_response_deduplicates_sources_by_url():
    raw = {"output": {"text": "answer"}, "citations": [_citation(), _citation()]}
    assert len(rag.parse_response(raw)["sources"]) == 1


def test_parse_response_not_covered():
    result = rag.parse_response({"output": {"text": "NOT_COVERED"}, "citations": [_citation()]})
    assert result["covered"] is False
    assert result["answer"] == rag.NOT_COVERED_MESSAGE
    assert result["sources"] == []


def test_parse_response_empty_answer_counts_as_not_covered():
    assert rag.parse_response({"output": {"text": ""}})["covered"] is False


def test_parse_response_guardrail_intervened():
    raw = {"output": {"text": "Blocked by guardrail."}, "guardrailAction": "INTERVENED"}
    result = rag.parse_response(raw)
    assert result["blocked"] is True and result["covered"] is False
    assert result["answer"] == "Blocked by guardrail."


def test_parse_response_answer_without_citations_is_not_grounded():
    result = rag.parse_response({"output": {"text": "Some answer"}, "citations": []})
    assert result["covered"] is True and result["grounded"] is False


def test_snippet_is_truncated():
    long_text = "word " * 200
    sources = rag.extract_sources([_citation(text=long_text)])
    assert len(sources[0]["snippet"]) <= rag.SNIPPET_CHARS + 3
    assert sources[0]["snippet"].endswith("...")


class _FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def retrieve_and_generate(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _AwsError(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


def test_answer_question_retries_without_stale_session():
    ok = {"output": {"text": "answer"}, "citations": [_citation()]}
    client = _FakeClient([_AwsError("ValidationException"), ok])
    result = rag.answer_question(client, CONFIG, "q", session_id="old-session")
    assert result["covered"] is True
    assert "sessionId" in client.calls[0] and "sessionId" not in client.calls[1]


def test_answer_question_does_not_retry_other_errors():
    client = _FakeClient([_AwsError("ThrottlingException")])
    with pytest.raises(_AwsError):
        rag.answer_question(client, CONFIG, "q", session_id="s")
    assert len(client.calls) == 1
