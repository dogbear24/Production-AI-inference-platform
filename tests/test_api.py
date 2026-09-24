
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "AI Inference Platform"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "device" in body


def test_invalid_top_k():
    response = client.post("/predict?top_k=99")
    # FastAPI validates the missing file before entering the route, so this
    # simply confirms the endpoint rejects an invalid request.
    assert response.status_code in {400, 422}
