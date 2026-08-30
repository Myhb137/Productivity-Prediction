from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_predict_accepts_valid_json():
    payload = {
        "study_hours_per_day": 4.5,
        "focus_score": 7,
        "sleep_hours": 8,
        "phone_usage_hours": 1.5,
        "stress_level": 3,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200, response.text
    assert "productivity_score" in response.json()


def test_predict_returns_clear_error_for_malformed_json():
    malformed = '{"study_hours_per_day": 4.5, "focus_score": 7, "sleep_hours": 8, "phone_usage_hours": 1.5, "stress_level": 3'

    response = client.post("/predict", content=malformed)

    assert response.status_code == 400, response.text
    assert "Invalid JSON request body" in response.json()["detail"]
