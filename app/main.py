from fastapi import FastAPI, Request, Response
import joblib
import numpy as np
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()


REQUEST_COUNT = Counter("request_count", "Total requests")

model = joblib.load("model/churn_model.pkl")

@app.middleware("http")
async def count_requests(request: Request, call_next):
    REQUEST_COUNT.inc()
    return await call_next(request)

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}

@app.post("/predict")
def predict(data: dict):

    values = np.array([data[k] for k in sorted(data.keys())]).reshape(1, -1)
    prediction = model.predict(values)
    return {"churn": int(prediction[0])}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
