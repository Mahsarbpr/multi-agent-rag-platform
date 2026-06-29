from typing import Any

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from rag_assistant.agents import AnalysisAgent
from rag_assistant.config import TOP_K
from rag_assistant.graph import create_rag_workflow
from rag_assistant.llm.base_provider import LLMProvider
from rag_assistant.tools import (
    search_documents,
    build_context,
    build_sources,
)


def retrieve_similar_documents(
    vectorstore: FAISS,
    question: str,
    k: int = TOP_K,
) -> list[Document]:
    return search_documents(vectorstore, question, k)


def build_prompt(question: str, context: str) -> str:
    return f"""
You are a helpful assistant.

Answer the question using ONLY the provided context.

If the answer is not in the context, say:
"I do not know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""


def generate_answer(
    llm: LLMProvider,
    question: str,
    context: str,
) -> str:
    analysis_agent = AnalysisAgent(llm)
    return analysis_agent.run(question, context)


def find_answer_to_question(
    vectorstore: FAISS,
    llm: LLMProvider,
    question: str,
    k: int = TOP_K,
) -> dict:
    workflow = create_rag_workflow(llm)

    initial_state = {
        "question": question,
        "vectorstore": vectorstore,
        "documents": [],
        "context": "",
        "sources": [],
        "tool_used": "",
        "answer": "",
        "evaluation": "",
    }

    result = workflow.invoke(initial_state)

    return {
        "question": question,
        "answer": result["answer"],
        "evaluation": result["evaluation"],
        "tool_used": result["tool_used"],
        "sources": result["sources"],
        "retrieved_documents": result["documents"],
    }