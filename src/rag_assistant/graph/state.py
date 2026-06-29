from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


class GraphState(TypedDict):
    question: str
    vectorstore: FAISS
    documents: list[Document]
    context: str
    sources: list[dict[str, Any]]
    tool_used: str
    answer: str
    evaluation: str