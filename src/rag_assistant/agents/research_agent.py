from langchain_community.vectorstores import FAISS

from rag_assistant.config import TOP_K
from rag_assistant.llm.base_provider import LLMProvider
from rag_assistant.tools import (
    search_documents,
    build_context,
    build_sources,
)


class ResearchAgent:
    """Uses document tools to gather evidence for downstream reasoning."""

    def __init__(
        self,
        llm: LLMProvider,
        vectorstore: FAISS,
        k: int = TOP_K,
    ):
        self.llm = llm
        self.vectorstore = vectorstore
        self.k = k

    def run(self, question: str) -> dict:
        tool_prompt = f"""
You are a research agent.

You have access to this tool:
- search_documents: searches indexed documents for relevant context.

Given the user question, decide which tool to use.

Return only the tool name.

Question:
{question}

Tool:
"""
        selected_tool = self.llm.invoke(tool_prompt).strip().lower()

        if "search_documents" not in selected_tool:
            selected_tool = "search_documents"

        documents = search_documents(
            vectorstore=self.vectorstore,
            question=question,
            k=self.k,
        )

        context = build_context(documents)
        sources = build_sources(documents)

        return {
            "question": question,
            "tool_used": selected_tool,
            "documents": documents,
            "context": context,
            "sources": sources,
        }