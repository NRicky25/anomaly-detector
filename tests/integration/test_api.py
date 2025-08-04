import pytest
from fastapi.testclient import TestClient
from src.main import app # Import the app directly



@pytest.fixture(scope="module")
def client():
    print("\nDEBUG (Test Fixture): Setting up TestClient with lifespan...")
    with TestClient(app) as c:
        print("DEBUG (Test Fixture): TestClient yielded.")
        yield c # This yields the client to the test functions
    print("DEBUG (Test Fixture): TestClient teardown complete.")


# Modify test functions to accept the 'client' fixture as an argument
def test_read_root(client): # <--- ADD 'client' here
    """Test the root endpoint."""
    print("DEBUG (Test): Running test_read_root")
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Credit Card Fraud Detection API! Visit /docs for API documentation."}

# Modify test functions to accept the 'client' fixture as an argument
def test_predict_single_valid_transaction(client): 
    """Test prediction for a single valid transaction."""
    print("DEBUG (Test): Running test_predict_single_valid_transaction")
    valid_transaction_data = {
        "Time": 123.45, "V1": -0.966, "V2": -0.847, "V3": 1.196, "V4": 0.25,
        "V5": -0.88, "V6": -0.306, "V7": 0.672, "V8": -0.046, "V9": 0.917,
        "V10": -0.992, "V11": -0.702, "V12": -0.141, "V13": -0.41, "V14": -0.066,
        "V15": 0.991, "V16": -0.692, "V17": -0.279, "V18": 0.038, "V19": 0.198,
        "V20": -0.063, "V21": -0.177, "V22": -0.076, "V23": -0.088, "V24": -0.015,
        "V25": 0.278, "V26": 0.147, "V27": -0.013, "V28": 0.008, "Amount": 50.0
    }
    response = client.post("/predict", json=valid_transaction_data)
    print(f"\nResponse Status Code: {response.status_code}")
    print(f"Response JSON: {response.json()}")

    assert response.status_code == 200
    # Change these assertions:
    assert isinstance(response.json(), list)
    assert len(response.json()) == 1        
    assert "predicted_class" in response.json()[0] 
    assert "prediction_probability" in response.json()[0]
    assert "is_fraud" in response.json()[0]

def test_predict_multiple_valid_transactions(client):
    """Test prediction for multiple valid transactions (batch request)."""
    valid_transactions_data = [
        { # Transaction 1 (similar to your existing valid one)
            "Time": 123.45, "V1": -0.966, "V2": -0.847, "V3": 1.196, "V4": 0.25,
            "V5": -0.88, "V6": -0.306, "V7": 0.672, "V8": -0.046, "V9": 0.917,
            "V10": -0.992, "V11": -0.702, "V12": -0.141, "V13": -0.41, "V14": -0.066,
            "V15": 0.991, "V16": -0.692, "V17": -0.279, "V18": 0.038, "V19": 0.198,
            "V20": -0.063, "V21": -0.177, "V22": -0.076, "V23": -0.088, "V24": -0.015,
            "V25": 0.278, "V26": 0.147, "V27": -0.013, "V28": 0.008, "Amount": 50.0
        },
        { # Transaction 2 (slightly different values)
            "Time": 500.0, "V1": 1.0, "V2": 0.5, "V3": -0.2, "V4": 0.8,
            "V5": -0.3, "V6": 0.1, "V7": 0.4, "V8": 0.0, "V9": -0.1,
            "V10": 0.2, "V11": -0.5, "V12": 0.6, "V13": -0.7, "V14": 0.9,
            "V15": -0.2, "V16": 0.3, "V17": -0.4, "V18": 0.5, "V19": -0.6,
            "V20": 0.7, "V21": -0.8, "V22": 0.9, "V23": -0.1, "V24": 0.2,
            "V25": -0.3, "V26": 0.4, "V27": -0.5, "V28": 0.6, "Amount": 150.0
        }
    ]
    response = client.post("/predict", json=valid_transactions_data)
    print(f"\nResponse Status Code: {response.status_code}")
    print(f"Response JSON: {response.json()}")
    assert response.status_code == 200
    assert isinstance(response.json(), list) # Expect a list back for multiple inputs
    assert len(response.json()) == 2
    for item in response.json():
        assert "predicted_class" in item
        assert "prediction_probability" in item
        assert "is_fraud" in item   

def test_predict_invalid_transaction_missing_field(client):
    """Test prediction for invalid transaction data (missing a required field, e.g., 'V5')."""
    invalid_transaction_data = {
        "Time": 123.45, "V1": -0.966, "V2": -0.847, "V3": 1.196, "V4": 0.25,
        # "V5": -0.88, <-- MISSING THIS FIELD
        "V6": -0.306, "V7": 0.672, "V8": -0.046, "V9": 0.917,
        "V10": -0.992, "V11": -0.702, "V12": -0.141, "V13": -0.41, "V14": -0.066,
        "V15": 0.991, "V16": -0.692, "V17": -0.279, "V18": 0.038, "V19": 0.198,
        "V20": -0.063, "V21": -0.177, "V22": -0.076, "V23": -0.088, "V24": -0.015,
        "V25": 0.278, "V26": 0.147, "V27": -0.013, "V28": 0.008, "Amount": 50.0
    }
    response = client.post("/predict", json=invalid_transaction_data)
    print(f"\nResponse Status Code: {response.status_code}")
    print(f"Response JSON: {response.json()}")
    assert response.status_code == 422 # FastAPI's validation error for missing fields
    assert "detail" in response.json()
    assert any("V5" in error["loc"] for error in response.json()["detail"]) # Specific check for V5

def test_predict_invalid_transaction_wrong_data_type(client):
    """Test prediction for invalid transaction data (e.g., 'Time' as a string)."""
    invalid_transaction_data = {
        "Time": "not_a_number",
        "V1": -0.966, "V2": -0.847, "V3": 1.196, "V4": 0.25,
        "V5": -0.88, "V6": -0.306, "V7": 0.672, "V8": -0.046, "V9": 0.917,
        "V10": -0.992, "V11": -0.702, "V12": -0.141, "V13": -0.41, "V14": -0.066,
        "V15": 0.991, "V16": -0.692, "V17": -0.279, "V18": 0.038, "V19": 0.198,
        "V20": -0.063, "V21": -0.177, "V22": -0.076, "V23": -0.088, "V24": -0.015,
        "V25": 0.278, "V26": 0.147, "V27": -0.013, "V28": 0.008, "Amount": 50.0
    }
    response = client.post("/predict", json=invalid_transaction_data)
    print(f"\nResponse Status Code: {response.status_code}")
    print(f"Response JSON: {response.json()}")
    assert response.status_code == 422 # FastAPI's validation error for wrong type
    assert "detail" in response.json()
    assert any("Time" in error["loc"] for error in response.json()["detail"])


def test_predict_empty_request_body(client):
    """Test sending an empty request body to the predict endpoint."""
    response = client.post("/predict", json={})
    print(f"\nResponse Status Code: {response.status_code}")
    print(f"Response JSON: {response.json()}")
    assert response.status_code == 422 # Expect validation error for missing required fields
    assert "detail" in response.json()

def test_predict_malformed_json_body(client):
    """Test sending a malformed JSON request body."""
    # This is not valid JSON
    malformed_json = "this is not json"
    response = client.post("/predict", content=malformed_json, headers={"Content-Type": "application/json"})
    print(f"\nResponse Status Code: {response.status_code}")
    print(f"Response JSON: {response.json()}")
    # ASSERTION CHANGE: Expect 422 instead of 400
    assert response.status_code == 422
    assert "detail" in response.json()
    # For 422 errors, 'detail' is typically a list of validation errors
    assert isinstance(response.json()["detail"], list)
    # You can be more specific, checking for errors related to the request body
    assert any("body" in error["loc"] for error in response.json()["detail"])


def test_unknown_endpoint(client):
    """Test accessing an endpoint that does not exist."""
    response = client.get("/nonexistent-endpoint")
    print(f"\nResponse Status Code: {response.status_code}")
    print(f"Response JSON: {response.json()}")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}

def test_predict_transaction_with_very_large_amount(client):
    """Test prediction for a transaction with a very large 'Amount'."""
    # Define the transaction data with the large amount
    transaction_data_for_test = {
        "Time": 123.45, "V1": -0.966, "V2": -0.847, "V3": 1.196, "V4": 0.25,
        "V5": -0.88, "V6": -0.306, "V7": 0.672, "V8": -0.046, "V9": 0.917,
        "V10": -0.992, "V11": -0.702, "V12": -0.141, "V13": -0.41, "V14": -0.066,
        "V15": 0.991, "V16": -0.692, "V17": -0.279, "V18": 0.038, "V19": 0.198,
        "V20": -0.063, "V21": -0.177, "V22": -0.076, "V23": -0.088, "V24": -0.015,
        "V25": 0.278, "V26": 0.147, "V27": -0.013, "V28": 0.008,
        "Amount": 1000000.0
    }

    response = client.post("/predict", json=transaction_data_for_test)

    print(f"\nResponse Status Code: {response.status_code}")
    print(f"Response JSON: {response.json()}")

    assert response.status_code == 200
    # ASSERTION CHANGES:
    assert isinstance(response.json(), list)  # Ensure it's a list
    assert len(response.json()) == 1          # Ensure it has exactly one item
    result_data = response.json()[0]          # Get the first (and only) item
    assert "predicted_class" in result_data
    assert "prediction_probability" in result_data
    assert "is_fraud" in result_data

    # CRUCIAL: Assert on the EXPECTED PREDICTION for this large amount
    # From your output, it was: {'is_fraud': False, 'predicted_class': 0, 'prediction_probability': 0.01, 'transaction_index': 0}
    assert result_data["is_fraud"] == False
    assert result_data["predicted_class"] == 0
    assert result_data["prediction_probability"] == 0.01