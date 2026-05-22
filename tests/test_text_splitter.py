from langchain_core.documents import Document

from rag_assistant.text_splitter import split_documents


def test_split_documents_creates_chunks():
    documents = [
        Document(
            page_content="This is a long test document. " * 100,
            metadata={"file_name": "test.txt"},
        )
    ]

    chunks = split_documents(documents)

    assert len(chunks) > 1
    assert all(chunk.page_content for chunk in chunks)


def test_split_documents_preserves_metadata():
    documents = [
        Document(
            page_content="This is a test document. " * 100,
            metadata={"file_name": "test.txt", "type": "txt"},
        )
    ]

    chunks = split_documents(documents)

    assert chunks[0].metadata["file_name"] == "test.txt"
    assert chunks[0].metadata["type"] == "txt"