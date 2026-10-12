from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root():

    response = client.get("/")

    assert response.status_code == 200

    assert (
        response.json()["version"]
        == "1.0.0"
    )

    assert (
        "X-Request-ID"
        in response.headers
    )

    assert (
        "X-Response-Time-Ms"
        in response.headers
    )


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"

    assert "device" in body


def test_metrics():

    client.get("/health")

    response = client.get("/metrics")

    assert response.status_code == 200

    body = response.json()

    assert "api" in body

    assert "model" in body

    assert (
        body["api"]["total_requests"]
        >= 1
    )

    assert (
        "p95_latency_ms"
        in body["api"]
    )


def test_predict_rejects_bad_file():

    response = client.post(

        "/predict",

        files={
            "file": (
                "test.txt",
                b"hello",
                "text/plain"
            )
        }
    )

    assert response.status_code == 415