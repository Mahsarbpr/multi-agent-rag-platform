from langchain_core.documents import Document

from rag_assistant.vector_store import create_vectorstore, save_vectorstore, load_vectorstore


def test_create_vectorstore_and_retrieve_relevant_document():
    documents = [
        Document(page_content="Python is a programming language.", metadata={"file_name": "python.txt"}),
        Document(page_content="Cats are small animals.", metadata={"file_name": "cats.txt"}),
    ]

    vectorstore = create_vectorstore(documents)

    results = vectorstore.similarity_search("What is Python?", k=1)

    assert len(results) == 1
    assert "Python" in results[0].page_content


def test_save_and_load_vectorstore(tmp_path):
    documents = [
        Document(page_content="FAISS stores vectors for similarity search.", metadata={"file_name": "faiss.txt"})
    ]

    index_path = tmp_path / "faiss_index"

    vectorstore = create_vectorstore(documents)
    save_vectorstore(vectorstore, str(index_path))

    loaded_vectorstore = load_vectorstore(str(index_path))
    results = loaded_vectorstore.similarity_search("What stores vectors?", k=1)

    assert len(results) == 1
    assert "FAISS" in results[0].page_content