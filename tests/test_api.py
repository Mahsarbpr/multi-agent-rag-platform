from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


def test_root_endpoint_returns_success():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "RAG API is running"
    }


def test_ask_endpoint_returns_valid_response_structure():
    """Test that ask endpoint returns properly structured response with valid types"""
    response = client.post(
        "/ask",
        json={
            "question": "What is the main topic?"
        },
    )

    assert response.status_code == 200

    response_json = response.json()

    # Validate response structure
    assert isinstance(response_json, dict)
    assert "answer" in response_json
    assert "sources" in response_json

    # Validate field types
    assert isinstance(response_json["answer"], str)
    assert isinstance(response_json["sources"], list)

    # Validate content is not empty
    assert len(response_json["answer"]) > 0
    assert isinstance(response_json["sources"], list)


def test_ask_endpoint_rejects_invalid_request():
    response = client.post(
        "/ask",
        json={
            "wrong_field": "hello"
        },
    )

    assert response.status_code == 422