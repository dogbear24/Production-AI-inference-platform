from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_queue_health():

    response = client.get(
        "/queue/health"
    )

    assert (
        response.status_code
        == 200
    )

    body = response.json()

    assert (
        "redis_connected"
        in body
    )

    assert (
        "queue_size"
        in body
    )