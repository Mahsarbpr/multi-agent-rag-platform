from fastapi.testclient import TestClient

from rag_assistant.api import app as api_app


class FakeRAGPipeline:
    def ask_question(self, question: str) -> dict:
        return {
            "answer": "fake answer",
            "sources": [
                {
                    "file_name": "sample.txt",
                    "page": None,
                    "source": "tests/fixtures/sample.txt",
                }
            ],
        }


def test_root_endpoint_returns_success():
    client = TestClient(api_app.app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "RAG API is running"}


def test_ask_endpoint_returns_valid_response_structure(monkeypatch):
    monkeypatch.setattr(api_app, "rag", FakeRAGPipeline())

    client = TestClient(api_app.app)

    response = client.post(
        "/ask",
        json={"question": "What is the main topic?"},
    )

    assert response.status_code == 200

    response_json = response.json()

    assert response_json["answer"] == "fake answer"
    assert isinstance(response_json["sources"], list)
    assert response_json["sources"][0]["file_name"] == "sample.txt"


def test_ask_endpoint_rejects_missing_question():
    client = TestClient(api_app.app)

    response = client.post("/ask", json={})

    assert response.status_code == 422


def test_ask_endpoint_rejects_wrong_field():
    client = TestClient(api_app.app)

    response = client.post("/ask", json={"wrong_field": "hello"})

    assert response.status_code == 422