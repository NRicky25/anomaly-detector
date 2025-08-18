import os
import pytest
from fastapi.testclient import TestClient
from src.main import app


VALID_TRANSACTION_PAYLOAD = {
    "Time": 123.45, "V1": -0.966, "V2": -0.847, "V3": 1.196, "V4": 0.25,
    "V5": -0.88, "V6": -0.306, "V7": 0.672, "V8": -0.046, "V9": 0.917,
    "V10": -0.992, "V11": -0.702, "V12": -0.141, "V13": -0.41,
    "V14": -0.066, "V15": 0.991, "V16": -0.692, "V17": -0.279,
    "V18": 0.038, "V19": 0.198, "V20": -0.063, "V21": -0.177,
    "V22": -0.076, "V23": -0.088, "V24": -0.015, "V25": 0.278,
    "V26": 0.147, "V27": -0.013, "V28": 0.008, "Amount": 50.0
}

API_KEY = os.environ.get("API_KEY", "test-api-key-for-unit-tests")
HEADERS = {"x-api-key": API_KEY}

@pytest.fixture
def client():
    # It's a good practice to set the environment variable for testing
    os.environ["API_KEY"] = "test-api-key-for-unit-tests"
    with TestClient(app) as c:
        yield c
    # Clean up the environment variable after the test
    del os.environ["API_KEY"]


def test_healthcheck(client):
    """Check if the API is running."""
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_single_valid_transaction(client):
    """Test prediction for a single valid transaction."""
    response = client.post("/predict", json=[VALID_TRANSACTION_PAYLOAD], headers=HEADERS)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 1
    assert "predicted_class" in response.json()[0]

def test_predict_multiple_transactions(client):
    """Test prediction for multiple transactions."""
    multiple_transactions = [VALID_TRANSACTION_PAYLOAD, VALID_TRANSACTION_PAYLOAD]
    response = client.post("/predict", json=multiple_transactions, headers=HEADERS)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 2

def test_predict_without_api_key(client):
    """Ensure requests without a valid API key are rejected."""
    # Provide a valid payload but with an invalid API key
    invalid_headers = {"x-api-key": "invalid_key"}
    response = client.post("/predict", json=[VALID_TRANSACTION_PAYLOAD], headers=invalid_headers)
    assert response.status_code == 401
    assert "detail" in response.json()