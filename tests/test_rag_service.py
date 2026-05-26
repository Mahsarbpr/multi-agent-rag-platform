from langchain_core.documents import Document

from rag_assistant.rag_service import build_context, build_prompt, generate_answer


class FakeLLM:
    def invoke(self, prompt: str) -> str:
        return "fake answer"


def test_build_context_includes_source_and_content():
    documents = [
        Document(
            page_content="This document explains RAG.",
            metadata={"file_name": "rag.txt", "page": 1},
        )
    ]

    context = build_context(documents)

    assert "rag.txt" in context
    assert "page 1" in context
    assert "This document explains RAG." in context


def test_build_prompt_includes_question_and_context():
    question = "What is RAG?"
    context = "RAG means retrieval augmented generation."

    prompt = build_prompt(question, context)

    assert question in prompt
    assert context in prompt


def test_generate_answer_uses_llm():
    llm = FakeLLM()

    answer = generate_answer(
        llm=llm,
        question="What is RAG?",
        context="RAG means retrieval augmented generation.",
    )

    assert answer == "fake answer"