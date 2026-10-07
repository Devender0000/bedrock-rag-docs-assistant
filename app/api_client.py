"""Small client for the docs-assistant API, used by the Streamlit app."""

from __future__ import annotations

import requests


class ApiError(Exception):
    """Raised with a user-friendly message when an API call fails."""


_STATUS_MESSAGES = {
    400: "That question couldn't be processed. Try rephrasing it.",
    403: "The API key was rejected. Check DOCS_ASSISTANT_API_KEY.",
    429: "Too many requests right now. Please wait a moment and try again.",
}


def ask(
    api_url: str,
    api_key: str,
    question: str,
    session_id: str | None = None,
    timeout: int = 40,
) -> dict:
    """POST a question to the API and return the parsed JSON response."""
    payload: dict = {"question": question}
    if session_id:
        payload["session_id"] = session_id

    try:
        response = requests.post(
            api_url,
            json=payload,
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            timeout=timeout,
        )
    except requests.Timeout:
        raise ApiError("The request timed out. Please try again.") from None
    except requests.RequestException:
        raise ApiError("Couldn't reach the API. Check the API URL and your connection.") from None

    if response.status_code == 200:
        try:
            return response.json()
        except ValueError:
            raise ApiError("The API returned an unreadable response.") from None

    message = _STATUS_MESSAGES.get(response.status_code)
    if message is None:
        message = "The assistant couldn't answer right now. Please try again."
    raise ApiError(message)
