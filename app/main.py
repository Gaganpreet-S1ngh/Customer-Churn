from fastapi import FastAPI, HTTPException, Request, Response
import joblib
import numpy as np
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()
REQUEST_COUNT = Counter("request_count", "Total requests")

model = joblib.load("model/churn_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
default_values = joblib.load("model/default_values.pkl")

FEATURE_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

@app.middleware("http")
async def count_requests(request: Request, call_next):
    REQUEST_COUNT.inc()
    return await call_next(request)

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}

@app.post("/predict")
def predict(data: dict):
    input_data = []

    for col in FEATURE_COLUMNS:
        val = data.get(col, default_values[col])

        if col in label_encoders:
            le = label_encoders[col]
            val_str = str(val)
            if val_str not in le.classes_:
                default_str = str(default_values[col])
                if default_str in le.classes_:
                    val_str = default_str
                else:
                    val_str = le.classes_[0]
            val = le.transform([val_str])[0]

        input_data.append(val)

    values = np.array(input_data).reshape(1, -1)

    try:
        prediction = model.predict(values)

        
        if isinstance(prediction[0], str):
            pred_map = {"No": 0, "Yes": 1}
            churn_val = pred_map.get(prediction[0], 0)
        else:
            churn_val = int(prediction[0])

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    return {"churn": churn_val}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
