# Credit Card Fraud Detection API

## Project Overview

This project demonstrates an end-to-end Machine Learning solution for detecting fraudulent credit card transactions. It encompasses data preprocessing, model training and optimization, and deployment as a containerized RESTful API. The goal is to identify anomalous transactions that might indicate fraud, leveraging a real-world imbalanced dataset.

## Key Features

- **Data Preprocessing:** Handles raw transaction data, including feature scaling (`Amount`, `Time`) and preparation for model training.
- **Machine Learning Model:** Utilizes a **Random Forest Classifier** trained on a highly imbalanced dataset, optimized for high precision to minimize false alarms.
- **Model Optimization:** Implements threshold optimization to fine-tune the balance between precision and recall, achieving an F1-score of **0.85** (with 0.91 Precision and 0.79 Recall for the fraud class).
- **RESTful API:** Developed using **FastAPI** to serve real-time fraud predictions.
- **Robust Input Validation:** Leverages Pydantic for automatic data validation and clear error handling for API requests.
- **Interactive API Documentation:** Automatically generated Swagger UI (`/docs`) and ReDoc (`/redoc`) for easy API exploration and integration.
- **Containerization:** Packaged using **Docker** for consistent, portable, and scalable deployment.

## Technologies Used

- **Python 3.10**
- **Machine Learning:** `scikit-learn`, `pandas`, `numpy`, `joblib`
- **API Framework:** `FastAPI`, `uvicorn`, `pydantic`
- **Containerization:** `Docker`
- **Version Control:** `Git`, `GitHub`

## Project Structure

```
anomaly-detector-api/
├── notebooks/
│ └── model_training_v1.ipynb # Jupyter notebook for data exploration, model training, and evaluation
├── src/
│ └── main.py # FastAPI application for the fraud prediction API
├── models/
│ └── development/
│ ├── random_forest_model.joblib # Saved trained Random Forest model
│ ├── scaler_amount.joblib # Saved StandardScaler for 'Amount'
│ └── scaler_time.joblib # Saved StandardScaler for 'Time'
├── .gitignore # Specifies intentionally untracked files to ignore
├── Dockerfile # Defines the Docker image for the API
├── requirements.txt # Python dependencies for the project
└── README.md # This documentation file
```

## Setup and Running the Application

### Prerequisites

- **Python 3.10** (for local development/notebooks)
- **Docker Desktop** (or Docker Engine) installed and running on your system.
- **Git**

### Steps to Run

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/NRicky25/anomaly-detector.git
    cd anomaly-detector
    ```

    _(Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username)_

2.  **[Optional] Set up Python Virtual Environment (for local development/notebook):**

    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    source venv/bin/activate # On macOS/Linux
    pip install -r requirements.txt
    ```

    _You can skip this if you only plan to run via Docker._

3.  **Build the Docker Image:**
    Navigate to the root directory of the project where `Dockerfile` is located.

    ```bash
    docker build -t fraud-detection-api .
    ```

4.  **Run the Docker Container:**
    ```bash
    docker run -d -p 8000:8000 --name fraud-api-container fraud-detection-api
    ```
    _The `-d` flag runs the container in detached mode (background)._
    _The `-p 8000:8000` flag maps port 8000 on your host machine to port 8000 inside the container._

## API Usage

Once the Docker container is running, you can access the API:

- **API Root:** `http://localhost:8000/`
- **Interactive API Documentation (Swagger UI):** `http://localhost:8000/docs`
- **Alternative API Documentation (ReDoc):** `http://localhost:8000/redoc`

### Making a Prediction

You can use the interactive documentation (`/docs`) to test the `/predict` endpoint, or send a `POST` request to `http://localhost:8000/predict` with a JSON body representing one or more transactions.

**Example Request Body (single transaction):**

```json
{
  "Time": 123.45,
  "V1": -0.966,
  "V2": -0.847,
  "V3": 1.196,
  "V4": 0.25,
  "V5": -0.88,
  "V6": -0.306,
  "V7": 0.672,
  "V8": -0.046,
  "V9": 0.917,
  "V10": -0.992,
  "V11": -0.702,
  "V12": -0.141,
  "V13": -0.41,
  "V14": -0.066,
  "V15": 0.991,
  "V16": -0.692,
  "V17": -0.279,
  "V18": 0.038,
  "V19": 0.198,
  "V20": -0.063,
  "V21": -0.177,
  "V22": -0.076,
  "V23": -0.088,
  "V24": -0.015,
  "V25": 0.278,
  "V26": 0.147,
  "V27": -0.013,
  "V28": 0.008,
  "Amount": 50.0
}
```
