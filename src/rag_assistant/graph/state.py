from typing import Any, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    context: str
    documents: list[Document]
    answer: str
    evaluation: str
    sources: list[dict[str, Any]]