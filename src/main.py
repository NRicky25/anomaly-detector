# src/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="A FastAPI endpoint for predicting credit card fraud using a pre-trained Random Forest model.",
    version="1.0.0"
)

# --- Configuration for Model and Scalers ---
# Define the base directory (one level up from src to anomaly-detector-api)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct paths to the models and scalers
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'development', 'random_forest_model.joblib')
SCALER_AMOUNT_PATH = os.path.join(BASE_DIR, 'models', 'development', 'scaler_amount.joblib')
SCALER_TIME_PATH = os.path.join(BASE_DIR, 'models', 'development', 'scaler_time.joblib')

# Define the optimal threshold determined during model training
OPTIMAL_THRESHOLD = 0.26 # Make sure this matches the threshold you found!

# --- Load Model and Scalers ---
model = None
scaler_amount = None
scaler_time = None

@app.on_event("startup")
async def load_artifacts():
    """Loads the trained model and scalers when the FastAPI application starts."""
    global model, scaler_amount, scaler_time
    try:
        model = joblib.load(MODEL_PATH)
        scaler_amount = joblib.load(SCALER_AMOUNT_PATH)
        scaler_time = joblib.load(SCALER_TIME_PATH)
        print("Model and scalers loaded successfully!")
    except Exception as e:
        print(f"Error loading model or scalers: {e}")
        print(f"Expected model path: {MODEL_PATH}")
        print(f"Expected amount scaler path: {SCALER_AMOUNT_PATH}")
        print(f"Expected time scaler path: {SCALER_TIME_PATH}")
        # In a real production app, you might want to raise an exception here
        # to prevent the app from starting if models can't be loaded.
        raise HTTPException(status_code=500, detail="Failed to load ML artifacts.")


# --- Define Input Data Schema using Pydantic ---
# This automatically provides validation and generates API documentation
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
        # This config enables parsing data from ORM models (e.g., SQLAlchemy)
        # but is good practice to include for general purpose BaseModels
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
    if model is None or scaler_amount is None or scaler_time is None:
        raise HTTPException(status_code=500, detail="ML artifacts not loaded. Server startup failed.")

    # Convert single input to a list for consistent processing
    if not isinstance(transactions, list):
        transactions = [transactions]

    # Convert Pydantic models to dictionaries, then to Pandas DataFrame
    input_data_list = [t.model_dump() for t in transactions]
    input_df = pd.DataFrame(input_data_list)

    # --- Preprocessing Steps (Must match training preprocessing) ---
    # 1. Scale 'Amount' and 'Time' using the loaded scalers
    #    Ensure reshape(-1, 1) for single feature scaling
    input_df['Amount_Scaled'] = scaler_amount.transform(input_df['Amount'].values.reshape(-1, 1))
    input_df['Time_Scaled'] = scaler_time.transform(input_df['Time'].values.reshape(-1, 1))

    # 2. Drop original 'Time' and 'Amount' columns
    input_df = input_df.drop(['Time', 'Amount'], axis=1)

    # 3. Ensure column order matches X_train
    # This is CRITICAL for model inference. The order of V1-V28, Amount_Scaled, Time_Scaled must match.
    # It's best practice to save the exact column order from X_train during training.
    # For this dataset, after the above steps, the columns should naturally be in the order V1..V28, Amount_Scaled, Time_Scaled
    # If you were to add/remove columns, you'd need to explicitly reorder.
    # Our X_train columns after scaling were roughly ['V1', 'V2', ..., 'V28', 'Amount_Scaled', 'Time_Scaled']
    # Let's assume this order based on the initial PCA features and added scaled features.

    # --- Make Prediction ---
    # Get probability of fraud (class 1)
    fraud_probabilities = model.predict_proba(input_df)[:, 1]

    # Apply the optimal threshold to get binary prediction
    binary_predictions = (fraud_probabilities >= OPTIMAL_THRESHOLD).astype(int)

    results = []
    for i in range(len(input_data_list)):
        results.append({
            'transaction_index': i, # Useful for multi-record requests
            'prediction_probability': float(fraud_probabilities[i]),
            'predicted_class': int(binary_predictions[i]),
            'is_fraud': bool(binary_predictions[i]) # Return as boolean for clarity
        })

    # If the original input was a single transaction, return a single result
    if len(results) == 1 and not isinstance(transactions, list):
        return results[0]
    else:
        return results

# --- Basic Root Endpoint ---
@app.get('/')
async def read_root():
    return {"message": "Welcome to the Credit Card Fraud Detection API! Visit /docs for API documentation."}