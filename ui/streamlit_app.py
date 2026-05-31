import requests
from requests import Response

import streamlit as st

import logging
import os
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def post_file(file_name: str, file_bytes: bytes, content_type: str | None) -> dict[str, Any]:
    files = {
        "file": (
            file_name,
            file_bytes,
            content_type or "application/octet-stream",
        )
    }

    response = requests.post(
        f"{API_BASE_URL}/upload",
        files=files,
        timeout=60,
    )

    return parse_response(response)


def post_question(question: str) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/ask",
        json={"question": question},
        timeout=120,
    )

    return parse_response(response)


def parse_response(response: Response) -> dict[str, Any]:
    try:
        response_json = response.json()
    except ValueError:
        response.raise_for_status()
        return {}

    if response.status_code >= 400:
        detail = response_json.get("detail", "Request failed.")
        raise RuntimeError(detail)

    return response_json


def render_upload_section() -> None:
    st.header("Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
    )

    if uploaded_file is None:
        return

    if st.button("Upload and Index"):
        try:
            result = post_file(
                file_name=uploaded_file.name,
                file_bytes=uploaded_file.getvalue(),
                content_type=uploaded_file.type,
            )

            st.success(result["message"])
            st.info("Index update started. Please wait a few seconds before asking questions.")

        except requests.exceptions.RequestException as error:
            logger.exception("Failed to connect to backend API")
            st.error(f"Could not connect to backend API: {error}")

        except RuntimeError as error:
            logger.exception("Upload request failed")
            st.error(str(error))


def render_question_section() -> None:
    st.header("Ask a Question")

    question = st.text_input("Enter your question")

    if not st.button("Ask"):
        return

    if not question.strip():
        st.warning("Please enter a question.")
        return

    try:
        with st.spinner("Thinking... retrieving relevant context and generating an answer"):
            result = post_question(question.strip())

        st.subheader("Answer")
        st.write(result["answer"])

        render_sources(result.get("sources", []))

    except requests.exceptions.RequestException as error:
        logger.exception("Failed to connect to backend API")
        st.error(f"Could not connect to backend API: {error}")

    except RuntimeError as error:
        logger.exception("Question request failed")
        st.error(str(error))


def render_sources(sources: list[dict[str, Any]]) -> None:
    st.subheader("Sources")

    if not sources:
        st.write("No sources returned.")
        return

    for source in sources:
        file_name = source.get("file_name", "Unknown file")
        page = source.get("page")
        source_path = source.get("source", "")

        if page is not None:
            st.write(f"- **{file_name}**, page {page}")
        else:
            st.write(f"- **{file_name}**")

        if source_path:
            st.caption(source_path)


def main() -> None:
    st.set_page_config(
        page_title="AI RAG Assistant",
        layout="centered",
    )

    st.title("AI RAG Assistant")
    st.write("Upload PDF/TXT documents and ask questions based on them.")

    render_upload_section()
    render_question_section()


if __name__ == "__main__":
    main()