from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_upload_document():
    with open("data/sample_docs/test.pdf", "rb") as file:
        response = client.post("/upload", files={"file": ("test.pdf", file, "application/pdf")})
    assert response.status_code == 200
    assert response.json() == {"message": "Document uploaded and processed successfully."}

def test_query():
    response = client.post("/query", json={"question": "What is AI?"})
    assert response.status_code == 200
    assert "answer" in response.json()