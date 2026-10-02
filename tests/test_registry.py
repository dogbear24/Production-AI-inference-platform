from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_model_registry():

    unique_name = (
        "test-model-"
        + uuid4().hex[:8]
    )

    # -------------------------
    # REGISTER V1
    # -------------------------

    response = client.post(
        "/models",
        json={
            "name": unique_name,
            "version": "1.0.0",
            "framework": "pytorch",
            "task": "image-classification",
        },
    )

    assert response.status_code == 201

    model_v1 = response.json()

    assert model_v1["version"] == "1.0.0"

    # -------------------------
    # REGISTER V2
    # -------------------------

    response = client.post(
        "/models",
        json={
            "name": unique_name,
            "version": "2.0.0",
            "framework": "pytorch",
            "task": "image-classification",
        },
    )

    assert response.status_code == 201

    model_v2 = response.json()

    # -------------------------
    # ACTIVATE V2
    # -------------------------

    response = client.post(
        f"/models/{model_v2['id']}/activate"
    )

    assert response.status_code == 200

    assert response.json()["is_active"] is True

    # -------------------------
    # CHECK ACTIVE VERSION
    # -------------------------

    response = client.get(
        f"/models/active/{unique_name}"
    )

    assert response.status_code == 200

    active_model = response.json()

    assert active_model["version"] == "2.0.0"