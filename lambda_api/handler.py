"""Lambda handler for POST /ask.

Request body:  {"question": "How do I add a path parameter?", "session_id": "optional"}
Response body: {"answer", "covered", "blocked", "grounded", "sources", "session_id"}
"""

import json
import logging

import boto3
from botocore.exceptions import ClientError

try:  # imported as a package (tests, evaluation harness)
    from . import rag
except ImportError:  # Lambda runtime uses a flat layout
    import rag

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_THROTTLE_CODES = {"ThrottlingException", "ServiceQuotaExceededException", "TooManyRequestsException"}
_client = None


def get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-agent-runtime")
    return _client


def _response(status: int, payload: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _response(400, {"error": "Request body must be valid JSON."})
    if not isinstance(body, dict):
        return _response(400, {"error": "Request body must be a JSON object."})

    try:
        question = rag.validate_question(body.get("question"))
        session_id = rag.validate_session_id(body.get("session_id"))
    except ValueError as exc:
        return _response(400, {"error": str(exc)})

    config = rag.RagConfig.from_env()
    # Log metadata only, not the question text.
    logger.info("question received: %d chars, session=%s", len(question), bool(session_id))

    try:
        result = rag.answer_question(get_client(), config, question, session_id)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        logger.error("Bedrock call failed: %s", code)
        if code in _THROTTLE_CODES:
            return _response(429, {"error": "The service is busy. Please try again in a moment."})
        return _response(502, {"error": "The assistant could not answer right now."})
    except Exception:
        logger.exception("Unexpected error while answering")
        return _response(500, {"error": "Something went wrong."})

    logger.info(
        "answered: covered=%s blocked=%s sources=%d",
        result["covered"],
        result["blocked"],
        len(result["sources"]),
    )
    return _response(200, result)
