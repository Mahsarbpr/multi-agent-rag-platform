from typing import Any

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from rag_assistant.config import TOP_K


def search_documents(
    vectorstore: FAISS,
    question: str,
    k: int = TOP_K,
) -> list[Document]:
    """Search indexed documents using semantic similarity."""
    return vectorstore.similarity_search(question, k=k)


def build_context(documents: list[Document]) -> str:
    context_parts = []

    for doc in documents:
        file_name = doc.metadata.get("file_name", "unknown source")
        page = doc.metadata.get("page")

        source = file_name
        if page is not None:
            source += f", page {page}"

        context_parts.append(
            f"Source: {source}\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)


def build_sources(documents: list[Document]) -> list[dict[str, Any]]:
    return [
        {
            "file_name": doc.metadata.get("file_name", "unknown"),
            "page": doc.metadata.get("page"),
            "source": doc.metadata.get("source"),
        }
        for doc in documents
    ]


def list_sources(documents: list[Document]) -> str:
    """Return retrieved source metadata as readable text."""
    sources = build_sources(documents)

    return "\n".join(
        f"- {source['file_name']}, page {source['page']}"
        for source in sources
    )