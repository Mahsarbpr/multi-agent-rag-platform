from pathlib import Path

from rag_assistant.rag_pipeline import RAGPipeline


class FakeLLM:
    def invoke(self, prompt: str) -> str:
        return "This is a fake RAG answer."


def test_rag_pipeline_builds_vectorstore_from_txt_file(tmp_path):
    data_folder = tmp_path / "data"
    data_folder.mkdir()

    test_file = data_folder / "rag.txt"
    test_file.write_text(
        "RAG stands for retrieval augmented generation.",
        encoding="utf-8",
    )

    pipeline = RAGPipeline(data_folder=str(data_folder))
    pipeline.llm = FakeLLM()

    pipeline.load_or_build_vectorstore()

    assert pipeline.vectorstore is not None


def test_rag_pipeline_answers_question_from_documents(tmp_path):
    data_folder = tmp_path / "data"
    data_folder.mkdir()

    test_file = data_folder / "rag.txt"
    test_file.write_text(
        "RAG stands for retrieval augmented generation.",
        encoding="utf-8",
    )

    pipeline = RAGPipeline(data_folder=str(data_folder))
    pipeline.llm = FakeLLM()

    pipeline.load_or_build_vectorstore()

    result = pipeline.ask_question("What does RAG stand for?")

    assert "answer" in result
    assert "sources" in result
    assert result["answer"] == "This is a fake RAG answer."
    assert len(result["sources"]) > 0