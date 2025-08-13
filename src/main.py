# src/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from io import StringIO
import joblib
import os
import pandas as pd
import numpy as np
from typing import List, Union
import random

# --- Configuration for Model and Scalers ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'development', 'random_forest_model.joblib')
SCALER_AMOUNT_PATH = os.path.join(BASE_DIR, 'models', 'development', 'scaler_amount.joblib')
SCALER_TIME_PATH = os.path.join(BASE_DIR, 'models', 'development', 'scaler_time.joblib')
OPTIMAL_THRESHOLD = 0.01
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'creditcard.csv')

DEFAULT_MODEL_FEATURES = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
    'Amount_Scaled', 'Time_Scaled'
]

model = None
scaler_amount = None
scaler_time = None
MODEL_FEATURES = None
sample_data = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler_amount, scaler_time, MODEL_FEATURES, sample_data
    print("DEBUG (Lifespan Startup): Attempting to load ML artifacts...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    if not os.path.exists(SCALER_AMOUNT_PATH):
        raise FileNotFoundError(f"Amount scaler file not found at: {SCALER_AMOUNT_PATH}")
    if not os.path.exists(SCALER_TIME_PATH):
        raise FileNotFoundError(f"Time scaler file not found at: {SCALER_TIME_PATH}")
    if os.path.exists(DATA_PATH):
        try:
            sample_data = pd.read_csv(DATA_PATH)
            print("DEBUG (Lifespan Startup): Sample data loaded successfully!")
        except Exception as e:
            print(f"ERROR (Lifespan Startup): Failed to load sample data from {DATA_PATH}: {e}")
            sample_data = None
    else:
        print(f"Warning: Sample data file not found at: {DATA_PATH}. Dashboard will show static data.")
        sample_data = None

    try:
        model = joblib.load(MODEL_PATH)
        scaler_amount = joblib.load(SCALER_AMOUNT_PATH)
        scaler_time = joblib.load(SCALER_TIME_PATH)

        if hasattr(model, 'feature_names_in_'):
            MODEL_FEATURES = model.feature_names_in_.tolist()
        else:
            MODEL_FEATURES = DEFAULT_MODEL_FEATURES

        print("DEBUG (Lifespan Startup): Model and scalers loaded successfully!")
        yield
    except Exception as e:
        print(f"ERROR (Lifespan Startup): Error during artifact loading: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load ML artifacts during startup: {e}.")
    finally:
        print("DEBUG (Lifespan Shutdown): Application shutdown completed.")

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="A FastAPI endpoint for predicting credit card fraud using a pre-trained Random Forest model.",
    version="1.0.0",
    lifespan=lifespan
)

origins = ["http://localhost", "http://localhost:5173", "http://127.0.0.1:5173"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class TransactionInput(BaseModel):
    Time: float = Field(...)
    V1: float = Field(...)
    V2: float = Field(...)
    V3: float = Field(...)
    V4: float = Field(...)
    V5: float = Field(...)
    V6: float = Field(...)
    V7: float = Field(...)
    V8: float = Field(...)
    V9: float = Field(...)
    V10: float = Field(...)
    V11: float = Field(...)
    V12: float = Field(...)
    V13: float = Field(...)
    V14: float = Field(...)
    V15: float = Field(...)
    V16: float = Field(...)
    V17: float = Field(...)
    V18: float = Field(...)
    V19: float = Field(...)
    V20: float = Field(...)
    V21: float = Field(...)
    V22: float = Field(...)
    V23: float = Field(...)
    V24: float = Field(...)
    V25: float = Field(...)
    V26: float = Field(...)
    V27: float = Field(...)
    V28: float = Field(...)
    Amount: float = Field(...)

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

# File upload endpoint
@app.post('/upload', summary="Upload a CSV file for prediction")
async def upload_file(file: UploadFile = File(...)):
    global model, scaler_amount, scaler_time, MODEL_FEATURES
    
    if not model or not scaler_amount or not scaler_time:
        raise HTTPException(status_code=500, detail="ML artifacts not loaded.")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files are accepted.")

    try:
        content = await file.read()
        csv_data = StringIO(content.decode('utf-8'))
        input_df = pd.read_csv(csv_data)

        input_df = input_df.fillna(0)
        original_amounts = input_df['Amount'].values
        # Preprocessing and scaling
        input_df['Amount_Scaled'] = scaler_amount.transform(input_df['Amount'].values.reshape(-1, 1))
        input_df['Time_Scaled'] = scaler_time.transform(input_df['Time'].values.reshape(-1, 1))
        input_df = input_df.drop(['Time', 'Amount'], axis=1)

        # Check for missing features
        if not all(feature in input_df.columns for feature in MODEL_FEATURES):
            missing_features = [f for f in MODEL_FEATURES if f not in input_df.columns]
            raise HTTPException(
                status_code=400, 
                detail=f"Input file is missing required features: {', '.join(missing_features)}."
            )

        input_df = input_df[MODEL_FEATURES]

        # Model prediction
        fraud_probabilities = model.predict_proba(input_df)[:, 1]
        binary_predictions = (fraud_probabilities >= OPTIMAL_THRESHOLD).astype(int)

        #Calculate
        
        is_fraud_mask = binary_predictions.astype(bool)
        total_transactions = len(input_df)
        fraud_count = int(is_fraud_mask.sum())
        non_fraud_count = total_transactions - fraud_count
        total_fraud_amount = float(original_amounts[is_fraud_mask].sum())

        summary = {
            'totalTransactions': total_transactions,
            'fraudCount': fraud_count,
            'nonFraudCount': non_fraud_count,
            'totalFraudAmount': round(total_fraud_amount, 2)
        }

        # Format and return the results
        results = []
        for i in range(len(input_df)):
            results.append({
                'id': i,
                'probability': float(fraud_probabilities[i]),
                'is_fraud': bool(binary_predictions[i])
            })

        return {"filename": file.filename, 
            "message": "File processed successfully.", 
            "results": results,
            "summary": summary}

    except Exception as e:
        print(f"ERROR (Upload): Failed to process file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")


@app.post('/predict', summary="Predict if a transaction is fraudulent")
async def predict_fraud(transactions: Union[TransactionInput, List[TransactionInput]]):
    global model, scaler_amount, scaler_time, MODEL_FEATURES

    if model is None or scaler_amount is None or scaler_time is None or MODEL_FEATURES is None:
        raise HTTPException(status_code=500, detail="ML artifacts not fully loaded. Server startup failed.")

    if not isinstance(transactions, list):
        transactions = [transactions]

    input_data_list = [t.model_dump() for t in transactions]
    input_df = pd.DataFrame(input_data_list)

    try:
        input_df['Amount_Scaled'] = scaler_amount.transform(input_df['Amount'].values.reshape(-1, 1))
        input_df['Time_Scaled'] = scaler_time.transform(input_df['Time'].values.reshape(-1, 1))
        input_df = input_df.drop(['Time', 'Amount'], axis=1)
        input_df = input_df[MODEL_FEATURES]

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
        return results if len(results) > 1 or isinstance(transactions, list) else results[0]

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing expected feature(s): {e}.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.get("/dashboard/data")
async def get_dashboard_data():
    global model, scaler_amount, scaler_time, sample_data
    
    if model is None or sample_data is None or scaler_amount is None or scaler_time is None:
        return {
            "metrics": {"total_anomalies": 0, "total_transactions": 0, "revenue": 0, "traffic": 0},
            "chart": {"labels": [], "datasets": []},
            "recent_anomalies": []
        }

    try:
        # Get a sample of recent data
        num_transactions_to_process = 500
        recent_transactions = sample_data.tail(num_transactions_to_process).copy()
        
        # Preprocess the data
        recent_transactions['Amount_Scaled'] = scaler_amount.transform(recent_transactions['Amount'].values.reshape(-1, 1))
        recent_transactions['Time_Scaled'] = scaler_time.transform(recent_transactions['Time'].values.reshape(-1, 1))
        
        features_to_predict = recent_transactions[MODEL_FEATURES]
        
        fraud_probabilities = model.predict_proba(features_to_predict)[:, 1]
        is_fraud = (fraud_probabilities >= OPTIMAL_THRESHOLD)

        total_anomalies = int(is_fraud.sum())
        total_transactions = num_transactions_to_process
        total_revenue = round(float(recent_transactions['Amount'].sum()), 2)
        total_traffic = round(num_transactions_to_process / 10, 1)

        daily_anomalies = [int(random.randint(5, 20)) for _ in range(7)]
        
        anomaly_indices = np.where(is_fraud)[0]
        recent_anomalies_list = []
        for i, idx in enumerate(anomaly_indices[-5:]):
            transaction = recent_transactions.iloc[idx]
            recent_anomalies_list.append({
                "id": len(recent_anomalies_list) + 1,
                "type": "Fraudulent Transaction",
                "user": f"txn_{int(idx)}",
                "time": f"{pd.to_datetime(transaction['Time'], unit='s').strftime('%Y-%m-%d %H:%M')}",
                "amount": float(transaction['Amount']),
                "status": "Pending"
            })

        response_data = {
            "metrics": {
                "total_anomalies": total_anomalies,
                "total_transactions": total_transactions,
                "revenue": total_revenue,
                "traffic": total_traffic
            },
            "chart": {
                "labels": ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                "datasets": [
                    {
                        "label": 'Anomalies',
                        "data": daily_anomalies,
                        "borderColor": '#6366F1',
                        "backgroundColor": 'rgba(99, 102, 241, 0.5)',
                        "tension": 0.4,
                    }
                ]
            },
            "recent_anomalies": recent_anomalies_list
        }
        
        print(f"DEBUG (Dashboard Data): Returning data: {response_data}")

        return response_data
    except Exception as e:
        print(f"ERROR (Dashboard Data): Failed to generate dynamic data: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate dynamic dashboard data.")

@app.get('/')
async def read_root():
    return {"message": "Welcome to the Credit Card Fraud Detection API! Visit /docs for API documentation."}
