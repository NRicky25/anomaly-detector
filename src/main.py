# src/main.py
from contextlib import asynccontextmanager # NEW IMPORT
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union

# --- Configuration for Model and Scalers ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'development', 'random_forest_model.joblib')
SCALER_AMOUNT_PATH = os.path.join(BASE_DIR, 'models', 'development', 'scaler_amount.joblib')
SCALER_TIME_PATH = os.path.join(BASE_DIR, 'models', 'development', 'scaler_time.joblib')
OPTIMAL_THRESHOLD = 0.26

# Default list of expected feature names (must match training order)
DEFAULT_MODEL_FEATURES = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
    'Amount_Scaled', 'Time_Scaled'
]

# Global variables (will be populated during lifespan startup)
model = None
scaler_amount = None
scaler_time = None
MODEL_FEATURES = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Loads the trained model and scalers during startup.
    """
    global model, scaler_amount, scaler_time, MODEL_FEATURES
    print("DEBUG (Lifespan Startup): Attempting to load ML artifacts...")

    # Explicitly check if files exist - CRITICAL DEBUGGING STEP
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR (Lifespan Startup): Model file not found at: {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    if not os.path.exists(SCALER_AMOUNT_PATH):
        print(f"ERROR (Lifespan Startup): Amount scaler file not found at: {SCALER_AMOUNT_PATH}")
        raise FileNotFoundError(f"Amount scaler file not found at: {SCALER_AMOUNT_PATH}")
    if not os.path.exists(SCALER_TIME_PATH):
        print(f"ERROR (Lifespan Startup): Time scaler file not found at: {SCALER_TIME_PATH}")
        raise FileNotFoundError(f"Time scaler file not found at: {SCALER_TIME_PATH}")

    try:
        model = joblib.load(MODEL_PATH)
        scaler_amount = joblib.load(SCALER_AMOUNT_PATH)
        scaler_time = joblib.load(SCALER_TIME_PATH)

        if hasattr(model, 'feature_names_in_'):
            MODEL_FEATURES = model.feature_names_in_.tolist()
            print(f"DEBUG (Lifespan Startup): Model expects features in this order: {MODEL_FEATURES}")
        else:
            print("Warning: Model does not have 'feature_names_in_'. Using hardcoded feature names as fallback.")
            MODEL_FEATURES = DEFAULT_MODEL_FEATURES
            print(f"DEBUG (Lifespan Startup): Falling back to assumed features: {MODEL_FEATURES}")

        print(f"DEBUG (Lifespan Startup): MODEL_FEATURES is: {MODEL_FEATURES}")
        print("DEBUG (Lifespan Startup): Model and scalers loaded successfully!")
        yield # Application startup is complete, now handle requests

    except Exception as e:
        print(f"ERROR (Lifespan Startup): Error during artifact loading: {e}")
        # Re-raise the exception or capture it to prevent the server from starting incorrectly
        raise HTTPException(status_code=500, detail=f"Failed to load ML artifacts during startup: {e}.")

    finally:
        # Optional: Clean up resources on shutdown (e.g., close database connections)
        print("DEBUG (Lifespan Shutdown): Application shutdown completed.")

# Initialize the FastAPI app with the lifespan context manager
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="A FastAPI endpoint for predicting credit card fraud using a pre-trained Random Forest model.",
    version="1.0.0",
    lifespan=lifespan # THIS IS THE KEY CHANGE for FastAPI app initialization
)

# --- Define Input Data Schema using Pydantic ---
class TransactionInput(BaseModel):
    Time: float = Field(..., example=123.45, description="Transaction time in seconds since first transaction")
    V1: float = Field(..., example=-0.966, description="PCA transformed feature V1")
    V2: float = Field(..., example=-0.847, description="PCA transformed feature V2")
    V3: float = Field(..., example=1.196, description="PCA transformed feature V3")
    V4: float = Field(..., example=0.25, description="PCA transformed feature V4")
    V5: float = Field(..., example=-0.88, description="PCA transformed feature V5")
    V6: float = Field(..., example=-0.306, description="PCA transformed feature V6")
    V7: float = Field(..., example=0.672, description="PCA transformed feature V7")
    V8: float = Field(..., example=-0.046, description="PCA transformed feature V8")
    V9: float = Field(..., example=0.917, description="PCA transformed feature V9")
    V10: float = Field(..., example=-0.992, description="PCA transformed feature V10")
    V11: float = Field(..., example=-0.702, description="PCA transformed feature V11")
    V12: float = Field(..., example=-0.141, description="PCA transformed feature V12")
    V13: float = Field(..., example=-0.41, description="PCA transformed feature V13")
    V14: float = Field(..., example=-0.066, description="PCA transformed feature V14")
    V15: float = Field(..., example=0.991, description="PCA transformed feature V15")
    V16: float = Field(..., example=-0.692, description="PCA transformed feature V16")
    V17: float = Field(..., example=-0.279, description="PCA transformed feature V17")
    V18: float = Field(..., example=0.038, description="PCA transformed feature V18")
    V19: float = Field(..., example=0.198, description="PCA transformed feature V19")
    V20: float = Field(..., example=-0.063, description="PCA transformed feature V20")
    V21: float = Field(..., example=-0.177, description="PCA transformed feature V21")
    V22: float = Field(..., example=-0.076, description="PCA transformed feature V22")
    V23: float = Field(..., example=-0.088, description="PCA transformed feature V23")
    V24: float = Field(..., example=-0.015, description="PCA transformed feature V24")
    V25: float = Field(..., example=0.278, description="PCA transformed feature V25")
    V26: float = Field(..., example=0.147, description="PCA transformed feature V26")
    V27: float = Field(..., example=-0.013, description="PCA transformed feature V27")
    V28: float = Field(..., example=0.008, description="PCA transformed feature V28")
    Amount: float = Field(..., example=123.45, description="Transaction amount")

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "Time": 123.45, "V1": -0.966, "V2": -0.847, "V3": 1.196, "V4": 0.25,
                "V5": -0.88, "V6": -0.306, "V7": 0.672, "V8": -0.046, "V9": 0.917,
                "V10": -0.992, "V11": -0.702, "V12": -0.141, "V13": -0.41,
                "V14": -0.066, "V15": 0.991, "V16": -0.692, "V17": -0.279,
                "V18": 0.038, "V19": 0.198, "V20": -0.063, "V21": -0.177,
                "V22": -0.076, "V23": -0.088, "V24": -0.015, "V25": 0.278,
                "V26": 0.147, "V27": -0.013, "V28": 0.008, "Amount": 50.0
            }
        }


# --- API Endpoint for Prediction ---
@app.post('/predict', summary="Predict if a transaction is fraudulent")
async def predict_fraud(transactions: Union[TransactionInput, List[TransactionInput]]):
    """
    Receives one or more transaction records and predicts if they are fraudulent.
    """
    global model, scaler_amount, scaler_time, MODEL_FEATURES

    # NEW DEBUG PRINT: What is MODEL_FEATURES right at the start of predict_fraud?
    print(f"DEBUG (Predict Entry): MODEL_FEATURES at start of predict: {MODEL_FEATURES}")

    if model is None or scaler_amount is None or scaler_time is None or MODEL_FEATURES is None:
        error_detail = "ML artifacts not fully loaded or feature names missing. Server startup failed."
        print(f"ERROR (Predict Check): {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

    # Convert single input to a list for consistent processing
    if not isinstance(transactions, list):
        transactions = [transactions]

    # Convert Pydantic models to dictionaries, then to Pandas DataFrame
    input_data_list = [t.model_dump() for t in transactions]
    input_df = pd.DataFrame(input_data_list)

    try:
        # --- Preprocessing Steps (Must match training preprocessing) ---
        input_df['Amount_Scaled'] = scaler_amount.transform(input_df['Amount'].values.reshape(-1, 1))
        input_df['Time_Scaled'] = scaler_time.transform(input_df['Time'].values.reshape(-1, 1))

        input_df = input_df.drop(['Time', 'Amount'], axis=1)

        # CRITICAL: Ensure column order matches the model's expected features
        input_df = input_df[MODEL_FEATURES]

        # --- Make Prediction ---
        fraud_probabilities = model.predict_proba(input_df)[:, 1]

        binary_predictions = (fraud_probabilities >= OPTIMAL_THRESHOLD).astype(int)

        results = []
        for i in range(len(input_data_list)):
            results.append({
                'transaction_index': i,
                'prediction_probability': float(fraud_probabilities[i]),
                'predicted_class': int(binary_predictions[i]),
                'is_fraud': bool(binary_predictions[i])
            })

        if len(results) == 1 and not isinstance(transactions, list):
            return results[0]
        else:
            return results
    except KeyError as e:
        print(f"ERROR (Predict): Missing expected feature(s) for prediction: {e}")
        raise HTTPException(status_code=400, detail=f"Missing expected feature(s) for prediction: {e}. Ensure all V1-V28, Amount, Time are provided and correctly named.")
    except Exception as e:
        print(f"ERROR (Predict): Prediction failed due to an internal server error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal server error: {e}")

# --- Basic Root Endpoint ---
@app.get('/')
async def read_root():
    return {"message": "Welcome to the Credit Card Fraud Detection API! Visit /docs for API documentation."}