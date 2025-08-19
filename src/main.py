from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, File, UploadFile, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ConfigDict
from io import StringIO
import joblib
import os
import pandas as pd
import numpy as np
from typing import List, Union, Optional
import random
import datetime
import json
from pathlib import Path
from dotenv import load_dotenv
import pyodbc

load_dotenv()
API_KEY = os.environ.get("API_KEY")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

#PATH
BASE_DIR = Path(__file__).resolve().parent.parent

def get_path(*paths):
    return BASE_DIR.joinpath(*paths)

MODEL_PATH = get_path("models", "development", "random_forest_model.joblib")
SCALER_AMOUNT_PATH = get_path("models", "development", "scaler_amount.joblib")
SCALER_TIME_PATH = get_path("models", "development", "scaler_time.joblib")
DATA_PATH = get_path("data", "raw", "creditcard.csv")
SETTINGS_FILE = get_path("src", "settings.json")

#GLOBALS
OPTIMAL_THRESHOLD = None
model = None
scaler_amount = None
scaler_time = None
OPTIMAL_THRESHOLD = 0.1

app = FastAPI()

origins = [
    "https://elegant-crostata-101fff.netlify.app",
    "http://localhost:5173",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

connection_string = os.environ.get("DATABASE_URL")

def get_demo_data():
    """Fetches a sample of transaction data from the database."""
    data = []
    try:
        with pyodbc.connect(connection_string) as cnxn:
            cursor = cnxn.cursor()
            query = "SELECT TOP 5000 * FROM demo"
            cursor.execute(query)
            columns = [column[0] for column in cursor.description]
            for row in cursor.fetchall():
                data.append(dict(zip(columns, row)))
        return data

    except pyodbc.Error as ex:
        print(f"Database error: {ex}")
        return []

class SettingsUpdate(BaseModel):
    optimal_threshold: float
    model_config = ConfigDict(from_attributes=True)


DEFAULT_MODEL_FEATURES = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
    'Amount_Scaled', 'Time_Scaled'
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global OPTIMAL_THRESHOLD, model, scaler_amount, scaler_time, MODEL_FEATURES
    print("DEBUG (Lifespan Startup): Attempting to load ML artifacts...")

    # Load settings first to get the OPTIMAL_THRESHOLD
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r") as f:
                settings_data = json.load(f)
                OPTIMAL_THRESHOLD = settings_data.get("OPTIMAL_THRESHOLD", OPTIMAL_THRESHOLD)
            print(f"DEBUG (Lifespan Startup): Initial OPTIMAL_THRESHOLD loaded: {OPTIMAL_THRESHOLD}")
        except Exception as e:
            print(f"ERROR (Lifespan Startup): Failed to load settings: {e}. Using default threshold.")
    else:
        print(f"Warning: Settings file not found at {SETTINGS_FILE}. Using default threshold of {OPTIMAL_THRESHOLD}.")

    # Load ML models and data
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        if not SCALER_AMOUNT_PATH.exists():
            raise FileNotFoundError(f"Amount scaler file not found at: {SCALER_AMOUNT_PATH}")
        if not SCALER_TIME_PATH.exists():
            raise FileNotFoundError(f"Time scaler file not found at: {SCALER_TIME_PATH}")

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
    model_config = ConfigDict(from_attributes=True, json_schema_extra={
        "example": {
            "Time": 123.45, "V1": -0.966, "V2": -0.847, "V3": 1.196, "V4": 0.25,
            "V5": -0.88, "V6": -0.306, "V7": 0.672, "V8": -0.046, "V9": 0.917,
            "V10": -0.992, "V11": -0.702, "V12": -0.141, "V13": -0.41,
            "V14": -0.066, "V15": 0.991, "V16": -0.692, "V17": -0.279,
            "V18": 0.038, "V19": 0.198, "V20": -0.063, "V21": -0.177,
            "V22": -0.076, "V23": -0.088, "V24": -0.015, "V25": 0.278,
            "V26": 0.147, "V27": -0.013, "V28": 0.008, "Amount": 50.0
        }
    })

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("your_module_name:app", host="127.0.0.1", port=8000, reload=True)

@app.get('/healthcheck')
def healthcheck():
    return {"status": "ok"}

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Credit Card Fraud Detection API! Visit /docs for API documentation."}


@app.get("/reports/transactions", dependencies=[Depends(verify_api_key)], summary="Get a paginated, filterable list of all transactions.")
def get_reports(
    page: int = Query(1, ge=1, description="Page number to retrieve."),
    page_size: int = Query(50, ge=1, le=100, description="Number of transactions per page."),
    sort_by: Optional[str] = Query("Time", pattern="^(Time|Amount|probability)$", description="Field to sort by."),
    sort_order: Optional[str] = Query("desc", pattern="^(asc|desc)$", description="Sort order: 'asc' or 'desc'."),
    min_amount: Optional[float] = Query(None, ge=0, description="Minimum transaction amount."),
    max_amount: Optional[float] = Query(None, ge=0, description="Maximum transaction amount."),
    is_fraud: Optional[bool] = Query(None, description="Filter by fraud status (True/False)."),
    download: Optional[bool] = Query(False, description="Set to True to download a CSV file.")
):
    global model, scaler_amount, scaler_time, MODEL_FEATURES

    if model is None or scaler_amount is None or scaler_time is None:
        raise HTTPException(status_code=500, detail="Data or model not loaded.")

    try:
        reports_data = get_demo_data()
        if not reports_data:
            return JSONResponse(content={"total_count": 0, "page": page, "page_size": page_size, "transactions": []})
        reports_df = pd.DataFrame(reports_data)
        
        reports_df['Amount_Scaled'] = scaler_amount.transform(reports_df['Amount'].values.reshape(-1, 1))
        reports_df['Time_Scaled'] = scaler_time.transform(reports_df['Time'].values.reshape(-1, 1))
        
        features_to_predict = reports_df[MODEL_FEATURES]
        fraud_probabilities = model.predict_proba(features_to_predict)[:, 1]
        reports_df['probability'] = fraud_probabilities
        reports_df['is_fraud'] = (reports_df['probability'] >= OPTIMAL_THRESHOLD)

        if min_amount is not None:
            reports_df = reports_df[reports_df['Amount'] >= min_amount]
        if max_amount is not None:
            reports_df = reports_df[reports_df['Amount'] <= max_amount]
        if is_fraud is not None:
            reports_df = reports_df[reports_df['is_fraud'] == is_fraud]

        ascending = sort_order == "asc"
        reports_df = reports_df.sort_values(by=sort_by, ascending=ascending)

        if download:
            reports_df = reports_df.drop(columns=['Amount_Scaled', 'Time_Scaled'])
            stream = StringIO()
            reports_df.to_csv(stream, index=False)
            response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
            response.headers["Content-Disposition"] = "attachment; filename=fraud_report.csv"
            return response

        total_count = len(reports_df)
        start = (page - 1) * page_size
        end = start + page_size
        paginated_df = reports_df.iloc[start:end]

        response_data = {
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "transactions": paginated_df.to_dict(orient="records")
        }

        return JSONResponse(content=response_data)
    
    except Exception as e:
        print(f"ERROR (Reports): Failed to generate report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {e}")
    
# Analytics Endpoints
@app.get("/analytics/trends", dependencies=[Depends(verify_api_key)], summary="Get historical trends for transactions and fraud rates.")
def get_analytics_trends():
    global model, scaler_amount, scaler_time, MODEL_FEATURES

    if model is None or scaler_amount is None or scaler_time is None:
        raise HTTPException(status_code=500, detail="Data or model not loaded.")

    try:
        analytics_df = get_demo_data()
        if not analytics_df:
            return JSONResponse(content={"daily_transactions": [], "daily_fraud_rate": []})
        analytics_df = pd.DataFrame(analytics_df)
        # Add the fraud predictions
        analytics_df['Amount_Scaled'] = scaler_amount.transform(analytics_df['Amount'].values.reshape(-1, 1))
        analytics_df['Time_Scaled'] = scaler_time.transform(analytics_df['Time'].values.reshape(-1, 1))
        features_to_predict = analytics_df[MODEL_FEATURES]
        fraud_probabilities = model.predict_proba(features_to_predict)[:, 1]
        analytics_df['is_fraud'] = (fraud_probabilities >= OPTIMAL_THRESHOLD)

        # Convert 'Time' (seconds) to a proper datetime object
        analytics_df['datetime'] = pd.to_datetime(analytics_df['Time'], unit='s')
        analytics_df['date'] = analytics_df['datetime'].dt.date

        # Calculate daily totals
        daily_transactions = analytics_df.groupby('date').size().reset_index(name='count')
        daily_fraud_count = analytics_df[analytics_df['is_fraud']].groupby('date').size().reset_index(name='fraud_count')
        
        # Merge to get fraud rate
        daily_trends = pd.merge(daily_transactions, daily_fraud_count, on='date', how='left').fillna(0)
        daily_trends['rate'] = daily_trends['fraud_count'] / daily_trends['count']

        # Format for JSON response
        daily_transactions_list = [{
            "date": str(row['date']),
            "count": int(row['count'])
        } for index, row in daily_transactions.iterrows()]

        daily_fraud_rate_list = [{
            "date": str(row['date']),
            "rate": float(row['rate'])
        } for index, row in daily_trends.iterrows()]
        
        response_data = {
            "daily_transactions": daily_transactions_list,
            "daily_fraud_rate": daily_fraud_rate_list
        }

        return JSONResponse(content=response_data)
    
    except Exception as e:
        print(f"ERROR (Analytics): Failed to generate analytics data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate analytics data: {e}")


@app.post('/upload', dependencies=[Depends(verify_api_key)], summary="Upload a CSV file for prediction")
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

        input_df['Amount_Scaled'] = scaler_amount.transform(original_amounts.reshape(-1, 1))
        input_df['Time_Scaled'] = scaler_time.transform(input_df['Time'].values.reshape(-1, 1))
        
        # Check for missing features before dropping
        if not all(feature in input_df.columns for feature in MODEL_FEATURES):
            missing_features = [f for f in MODEL_FEATURES if f not in input_df.columns]
            raise HTTPException(
                status_code=400, 
                detail=f"Input file is missing required features: {', '.join(missing_features)}."
            )
        
        input_df = input_df.drop(['Time', 'Amount'], axis=1)
        input_df = input_df[MODEL_FEATURES]

        fraud_probabilities = model.predict_proba(input_df)[:, 1]
        binary_predictions = (fraud_probabilities >= OPTIMAL_THRESHOLD).astype(int)

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

@app.post("/predict", summary="Predict if a transaction is fraudulent")
async def predict_fraud(transactions: List[TransactionInput], api_key: str = Depends(verify_api_key)):
    if not transactions:
        raise HTTPException(status_code=400, detail="No transactions provided")
    
    # Use .model_dump() instead of .dict()
    input_df = pd.DataFrame([t.model_dump() for t in transactions])
    
    is_large_amount = input_df['Amount'] > 5000
    
    input_df['Amount_Scaled'] = scaler_amount.transform(input_df['Amount'].values.reshape(-1, 1))
    input_df['Time_Scaled'] = scaler_time.transform(input_df['Time'].values.reshape(-1, 1))
    input_df = input_df.drop(['Time', 'Amount'], axis=1)
    input_df = input_df[MODEL_FEATURES]
    
    fraud_probabilities = model.predict_proba(input_df)[:, 1]
    is_model_fraud = (fraud_probabilities >= OPTIMAL_THRESHOLD)
    
    is_fraud = is_model_fraud | is_large_amount
    
    results = [
        {
            "transaction_index": i,
            "prediction_probability": float(fraud_probabilities[i]),
            "predicted_class": int(is_fraud[i]),
            "is_fraud": bool(is_fraud[i])
        } for i in range(len(transactions))
    ]
    
    return results

@app.get("/dashboard/data", dependencies=[Depends(verify_api_key)])
async def get_dashboard_data():
    global model, scaler_amount, scaler_time
    
    if model is None or scaler_amount is None or scaler_time is None:
        return {
            "metrics": {"total_anomalies": 0, "total_transactions": 0, "revenue": 0, "traffic": 0},
            "chart": {"labels": [], "datasets": []},
            "recent_anomalies": []
        }

    try:
        db_data = get_demo_data()
        if not db_data:
            return {
                "metrics": {"total_anomalies": 0, "total_transactions": 0, "revenue": 0},
                "chart": {"labels": [], "datasets": []},
                "recent_anomalies": []
            }
        recent_transactions = pd.DataFrame(db_data)
        # num_transactions_to_process = 5000
        # if len(sample_data) > num_transactions_to_process:
        #     recent_transactions = sample_data.sample(n=num_transactions_to_process).copy()
        # else:
        #     recent_transactions = sample_data.copy()
        # Preprocess the data
        recent_transactions['Amount_Scaled'] = scaler_amount.transform(recent_transactions['Amount'].values.reshape(-1, 1))
        recent_transactions['Time_Scaled'] = scaler_time.transform(recent_transactions['Time'].values.reshape(-1, 1))
        
        features_to_predict = recent_transactions[MODEL_FEATURES]
        
        fraud_probabilities = model.predict_proba(features_to_predict)[:, 1]
        is_fraud = (fraud_probabilities >= OPTIMAL_THRESHOLD)

        total_anomalies = int(is_fraud.sum())
        total_transactions = len(recent_transactions)
        total_revenue = round(float(recent_transactions['Amount'].sum()), 2)
        total_traffic = round(len(recent_transactions) / 10, 1)

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

@app.get("/settings", dependencies=[Depends(verify_api_key)], summary="Get the current application settings.")
def get_settings():
    try:
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)
        return JSONResponse(content=settings)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Settings file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve settings: {e}")

@app.post("/settings", dependencies=[Depends(verify_api_key)], summary="Update the application settings.")
def update_settings(new_settings: SettingsUpdate):
    try:
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)

        settings["OPTIMAL_THRESHOLD"] = new_settings.optimal_threshold
        
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)
            
        global OPTIMAL_THRESHOLD
        OPTIMAL_THRESHOLD = new_settings.optimal_threshold
        
        return JSONResponse(content={"message": "Settings updated successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {e}")

