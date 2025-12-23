# Customer Churn Prediction System

An end-to-end Machine Learning project that predicts customer churn using structured customer data.
This project demonstrates the complete ML lifecycle including data preprocessing, model training,
experiment tracking, API deployment, containerization, and monitoring.

---

## 🚀 Key Features

- End-to-end ML pipeline for churn prediction
- Trained on 50,000+ customer records
- Feature engineering and hyperparameter tuning using GridSearchCV
- Experiment tracking with MLflow
- REST API for real-time predictions using FastAPI
- Dockerized deployment
- Prometheus-compatible monitoring

---

## 🛠 Tech Stack

- Python
- Scikit-learn
- MLflow
- FastAPI
- Docker
- Prometheus

---

## 📁 Project Structure

customer-churn/
│── app/
│   └── main.py          # FastAPI inference service
│── data/
│   └── churn.csv        # Dataset
│── model/
│   └── churn_model.pkl  # Trained model
│── train.py             # Model training pipeline
│── requirements.txt
│── Dockerfile
│── README.md

---

## ⚙️ Setup & Run (Local)

### 1️⃣ Clone repository

git clone https://github.com/Gaganpreet-S1ngh/Customer-Churn.git
cd Customer-Churn

---

### 2️⃣ Create virtual environment

python -m venv venv

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

---

### 3️⃣ Install dependencies

pip install -r requirements.txt

---

## 🧠 Train the Model

Ensure dataset exists at:
data/churn.csv

Run:
python train.py

This will:
- Preprocess data
- Train and tune a RandomForest model
- Log experiments using MLflow
- Save the trained model to model/churn_model.pkl

---

## 📊 MLflow Tracking

Start MLflow UI:
mlflow ui

Open:
http://localhost:5000

---

## 🌐 Run Prediction API

uvicorn app.main:app --reload

API:
http://127.0.0.1:8000

Swagger UI:
http://127.0.0.1:8000/docs

---

## 🔮 Prediction Endpoint

POST /predict

Example request:
{
  "gender": 1,
  "SeniorCitizen": 0,
  "Partner": 1,
  "Dependents": 0,
  "tenure": 12,
  "PhoneService": 1,
  "MonthlyCharges": 70.5,
  "TotalCharges": 840.0
}

Example response:
{
  "churn": 0
}

---

## 📈 Monitoring

Metrics endpoint:
/metrics

Prometheus-compatible metrics for monitoring API usage.

---

## 🐳 Run with Docker

docker build -t churn-api .
docker run -p 8000:8000 churn-api

Access:
http://localhost:8000

---

## ✅ Resume Highlights

- Built an end-to-end ML pipeline for customer churn prediction
- Implemented hyperparameter tuning and experiment tracking with MLflow
- Deployed a containerized ML inference API using FastAPI and Docker
- Integrated Prometheus monitoring

---

## 👤 Author

Gaganpreet Singh  
GitHub: https://github.com/Gaganpreet-S1ngh
