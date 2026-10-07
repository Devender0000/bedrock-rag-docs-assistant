"""Core RAG logic: build the Bedrock request and shape the response.

Kept free of AWS-client setup so the Lambda handler, the evaluation harness, and
the unit tests all share exactly the same prompt and parsing code.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

NOT_COVERED_TOKEN = "NOT_COVERED"
NOT_COVERED_MESSAGE = "I couldn't find that in the FastAPI documentation."
BLOCKED_MESSAGE = "Sorry, I can't help with that request."
MAX_QUESTION_CHARS = 1000
SNIPPET_CHARS = 300

PROMPT_TEMPLATE = """You are a documentation assistant for the FastAPI web framework.
Answer the question using only the search results below. Do not use outside knowledge.
If the search results do not contain the answer, reply with exactly NOT_COVERED and nothing else.
Be concise and accurate. Include short code examples from the search results when they help.

Search results:
$search_results$

Question: $query$

$output_format_instructions$"""

_SESSION_ID = re.compile(r"^[\w\-.:]{1,100}$")
_STALE_SESSION_CODES = {"ValidationException", "ResourceNotFoundException"}


@dataclass(frozen=True)
class RagConfig:
    knowledge_base_id: str
    model_arn: str
    guardrail_id: str
    guardrail_version: str
    num_results: int = 5

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RagConfig":
        env = os.environ if env is None else env
        return cls(
            knowledge_base_id=env["KNOWLEDGE_BASE_ID"],
            model_arn=env["MODEL_ARN"],
            guardrail_id=env["GUARDRAIL_ID"],
            guardrail_version=env["GUARDRAIL_VERSION"],
            num_results=int(env.get("NUM_RESULTS", "5")),
        )


def validate_question(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("'question' must be a non-empty string.")
    question = value.strip()
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"'question' must be at most {MAX_QUESTION_CHARS} characters.")
    return question


def validate_session_id(value) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not _SESSION_ID.match(value):
        raise ValueError("'session_id' is not valid.")
    return value


def build_request(question: str, config: RagConfig, session_id: str | None = None) -> dict:
    """Keyword arguments for bedrock-agent-runtime retrieve_and_generate."""
    request = {
        "input": {"text": question},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": config.knowledge_base_id,
                "modelArn": config.model_arn,
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {"numberOfResults": config.num_results}
                },
                "generationConfiguration": {
                    "promptTemplate": {"textPromptTemplate": PROMPT_TEMPLATE},
                    "guardrailConfiguration": {
                        "guardrailId": config.guardrail_id,
                        "guardrailVersion": config.guardrail_version,
                    },
                    "inferenceConfig": {
                        "textInferenceConfig": {"temperature": 0.0, "maxTokens": 800}
                    },
                },
            },
        },
    }
    if session_id:
        request["sessionId"] = session_id
    return request


def _snippet(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= SNIPPET_CHARS:
        return collapsed
    return collapsed[:SNIPPET_CHARS].rstrip() + "..."


def extract_sources(citations: list[dict]) -> list[dict]:
    """Flatten Bedrock citations into a de-duplicated list of source pages."""
    sources: list[dict] = []
    seen: set[str] = set()
    for citation in citations:
        for ref in citation.get("retrievedReferences", []):
            metadata = ref.get("metadata") or {}
            s3_uri = (ref.get("location") or {}).get("s3Location", {}).get("uri", "")
            url = metadata.get("source_url") or ""
            key = url or s3_uri
            if not key or key in seen:
                continue
            seen.add(key)
            title = metadata.get("title") or s3_uri.rsplit("/", 1)[-1] or "Documentation page"
            sources.append(
                {
                    "title": str(title),
                    "url": str(url),
                    "snippet": _snippet((ref.get("content") or {}).get("text", "")),
                }
            )
    return sources


def parse_response(raw: dict) -> dict:
    """Turn a retrieve_and_generate response into the API's response body."""
    answer = ((raw.get("output") or {}).get("text") or "").strip()
    session_id = raw.get("sessionId")

    if raw.get("guardrailAction") == "INTERVENED":
        return {
            "answer": answer or BLOCKED_MESSAGE,
            "covered": False,
            "blocked": True,
            "grounded": False,
            "sources": [],
            "session_id": session_id,
        }

    if not answer or answer.upper().startswith(NOT_COVERED_TOKEN):
        return {
            "answer": NOT_COVERED_MESSAGE,
            "covered": False,
            "blocked": False,
            "grounded": False,
            "sources": [],
            "session_id": session_id,
        }

    sources = extract_sources(raw.get("citations", []))
    return {
        "answer": answer,
        "covered": True,
        "blocked": False,
        "grounded": bool(sources),
        "sources": sources,
        "session_id": session_id,
    }


def _is_stale_session(exc: Exception) -> bool:
    code = getattr(exc, "response", {}).get("Error", {}).get("Code")
    return code in _STALE_SESSION_CODES


def answer_question(
    client, config: RagConfig, question: str, session_id: str | None = None
) -> dict:
    """Ask the Knowledge Base a question. Retries once without the session if it expired."""
    try:
        raw = client.retrieve_and_generate(**build_request(question, config, session_id))
    except Exception as exc:
        if session_id and _is_stale_session(exc):
            raw = client.retrieve_and_generate(**build_request(question, config, None))
        else:
            raise
    return parse_response(raw)
