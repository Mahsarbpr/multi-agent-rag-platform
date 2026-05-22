from pathlib import Path

from rag_assistant.document_loader import load_documents, load_pdf_file, load_txt_file


def test_load_txt_file_returns_document(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello world", encoding="utf-8")

    document = load_txt_file(file_path)

    assert document.page_content == "hello world"
    assert document.metadata["source"] == str(file_path)
    assert document.metadata["file_name"] == "example.txt"
    assert document.metadata["type"] == "txt"


def test_load_pdf_file_returns_documents_from_sample_pdf():
    sample_pdf = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "sample_docs"
        / "What is RAG_ - Retrieval-Augmented Generation AI Explained - AWS.pdf"
    )

    assert sample_pdf.exists(), f"Sample PDF not found at {sample_pdf}"

    documents = load_pdf_file(sample_pdf)

    assert isinstance(documents, list)
    assert len(documents) > 0
    assert all(doc.metadata["type"] == "pdf" for doc in documents)
    assert all(doc.metadata["file_name"] == sample_pdf.name for doc in documents)
    assert all(isinstance(doc.page_content, str) for doc in documents)
    assert any(doc.page_content.strip() for doc in documents)


def test_load_documents_loads_txt_and_pdf_files(tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("hello txt", encoding="utf-8")

    sample_pdf = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "sample_docs"
        / "What is RAG_ - Retrieval-Augmented Generation AI Explained - AWS.pdf"
    )

    assert sample_pdf.exists(), f"Sample PDF not found at {sample_pdf}"

    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(sample_pdf.read_bytes())

    documents = load_documents(str(tmp_path))

    assert isinstance(documents, list)
    assert len(documents) >= 2

    metadata_types = {doc.metadata["type"] for doc in documents}

    assert {"txt", "pdf"}.issubset(metadata_types)
    assert any(doc.metadata["file_name"] == "test.txt" for doc in documents)
    assert any(doc.metadata["file_name"] == "test.pdf" for doc in documents)


def test_load_documents_ignores_unsupported_files(tmp_path):
    txt_file = tmp_path / "test.txt"
    png_file = tmp_path / "image.png"

    txt_file.write_text("hello txt", encoding="utf-8")
    png_file.write_text("not supported", encoding="utf-8")

    documents = load_documents(str(tmp_path))

    assert len(documents) == 1
    assert documents[0].metadata["file_name"] == "test.txt"