# Customer Churn Prediction System

An end-to-end Machine Learning project that predicts customer churn using structured customer data.  
The project demonstrates the complete ML lifecycle: data preprocessing, model training, experiment tracking, API deployment, containerization, and monitoring.

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
│ └── main.py # FastAPI inference service
│── data/
│ └── churn.csv # Dataset
│── model/
│ └── churn_model.pkl # Trained model
│── train.py # Model training pipeline
│── requirements.txt
│── Dockerfile
│── README.md

yaml
Copy code

---

## ⚙️ Setup & Run (Local)

### 1️⃣ Clone repository
```bash
git clone https://github.com/Gaganpreet-S1ngh/Customer-Churn.git
cd Customer-Churn
2️⃣ Create virtual environment
bash
Copy code
python -m venv venv
Windows

bash
Copy code
venv\Scripts\activate
Mac/Linux

bash
Copy code
source venv/bin/activate
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
🧠 Train the Model
Ensure dataset exists at:

bash
Copy code
data/churn.csv
Run:

bash
Copy code
python train.py
This will:

Preprocess data

Train and tune a RandomForest model

Log experiments using MLflow

Save the trained model to model/churn_model.pkl

📊 MLflow Tracking
Start MLflow UI:

bash
Copy code
mlflow ui
Open:

arduino
Copy code
http://localhost:5000
🌐 Run Prediction API
bash
Copy code
uvicorn app.main:app --reload
API available at:

cpp
Copy code
http://127.0.0.1:8000
Swagger UI:

arduino
Copy code
http://127.0.0.1:8000/docs
🔮 Prediction Endpoint
POST /predict

Example request:

json
Copy code
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

json
Copy code
{
  "churn": 0
}
📈 Monitoring
Metrics endpoint:

bash
Copy code
/metrics
Exposes Prometheus-compatible metrics for request monitoring.

🐳 Run with Docker
bash
Copy code
docker build -t churn-api .
docker run -p 8000:8000 churn-api
Access API:

arduino
Copy code
http://localhost:8000
