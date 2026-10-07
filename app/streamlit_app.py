"""Streamlit chat page for the FastAPI docs assistant.

Run from the project root:
    export DOCS_ASSISTANT_API_URL="https://<id>.execute-api.<region>.amazonaws.com/prod/ask"
    export DOCS_ASSISTANT_API_KEY="<your key>"
    streamlit run app/streamlit_app.py

Or put both values in .streamlit/secrets.toml (see secrets.toml.example).
"""

import os

import streamlit as st

try:  # run from the project root as a package
    from app.api_client import ApiError, ask
except ModuleNotFoundError:  # streamlit puts the script's own folder on the path
    from api_client import ApiError, ask

EXAMPLE_QUESTIONS = [
    "How do I add a path parameter?",
    "How does dependency injection work?",
    "How do I enable CORS?",
    "How do I deploy FastAPI with Docker?",
    "How do I train a neural network?",
]


def get_setting(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            title = source.get("title") or "Documentation page"
            url = source.get("url")
            st.markdown(f"[{title}]({url})" if url else title)
            if source.get("snippet"):
                st.caption(source["snippet"])


def render_assistant_message(message: dict) -> None:
    result = message.get("result") or {}
    if result.get("blocked"):
        st.warning(message["content"])
    elif not result.get("covered", True):
        st.info(message["content"])
    else:
        st.markdown(message["content"])
        if result.get("sources"):
            render_sources(result["sources"])
        elif result:
            st.caption("No source was returned for this answer. Please verify it in the docs.")


def main() -> None:
    st.set_page_config(page_title="FastAPI Docs Assistant", layout="centered")
    st.title("FastAPI Docs Assistant")
    st.caption(
        "Ask a question about FastAPI. Answers come only from the official documentation, "
        "with links to the pages used."
    )

    api_url = get_setting("DOCS_ASSISTANT_API_URL")
    api_key = get_setting("DOCS_ASSISTANT_API_KEY")
    if not api_url or not api_key:
        st.warning(
            "Set DOCS_ASSISTANT_API_URL and DOCS_ASSISTANT_API_KEY as environment variables "
            "or in .streamlit/secrets.toml, then reload."
        )
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.session_id = None

    with st.sidebar:
        st.subheader("Try asking")
        for example in EXAMPLE_QUESTIONS:
            if st.button(example, use_container_width=True):
                st.session_state.pending_question = example
        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(message["content"])

    question = st.chat_input("Ask about FastAPI") or st.session_state.pop("pending_question", None)
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the docs..."):
            try:
                result = ask(api_url, api_key, question, st.session_state.session_id)
            except ApiError as error:
                st.error(str(error))
                return
        st.session_state.session_id = result.get("session_id") or st.session_state.session_id
        message = {"role": "assistant", "content": result["answer"], "result": result}
        render_assistant_message(message)
    st.session_state.messages.append(message)


main()
