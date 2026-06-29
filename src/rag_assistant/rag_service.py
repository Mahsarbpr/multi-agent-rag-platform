from typing import Any

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from rag_assistant.agents import AnalysisAgent
from rag_assistant.config import TOP_K
from rag_assistant.graph import create_rag_workflow
from rag_assistant.llm.base_provider import LLMProvider


def retrieve_similar_documents(
    vectorstore: FAISS,
    question: str,
    k: int = TOP_K,
) -> list[Document]:
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


def build_sources(documents: list[Document]) -> list[dict[str, Any]]:
    return [
        {
            "file_name": doc.metadata.get("file_name", "unknown"),
            "page": doc.metadata.get("page"),
            "source": doc.metadata.get("source"),
        }
        for doc in documents
    ]


def find_answer_to_question(
    vectorstore: FAISS,
    llm: LLMProvider,
    question: str,
    k: int = TOP_K,
) -> dict:
    retrieved_docs = retrieve_similar_documents(vectorstore, question, k)
    context = build_context(retrieved_docs)
    sources = build_sources(retrieved_docs)

    workflow = create_rag_workflow(llm)

    initial_state = {
        "question": question,
        "context": context,
        "documents": retrieved_docs,
        "answer": "",
        "evaluation": "",
        "sources": sources,
    }

    result = workflow.invoke(initial_state)

    return {
        "question": question,
        "answer": result["answer"],
        "evaluation": result["evaluation"],
        "sources": result["sources"],
        "retrieved_documents": retrieved_docs,
    }
