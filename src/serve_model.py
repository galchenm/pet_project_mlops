import os
from fastapi import FastAPI, Response
from pydantic import BaseModel
import joblib
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.joblib")
PREPROCESSOR_PATH = os.path.join(
    os.path.dirname(__file__), "models", "preprocessor.pkl"
)

# Load model and preprocessor once
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


class PatientData(BaseModel):
    age: float
    avg_glucose_level: float
    bmi: float
    gender: str
    ever_married: str
    work_type: str
    Residence_type: str
    smoking_status: str


# Prometheus metrics
REQUEST_COUNT = Counter("request_count", "Number of requests received")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency in seconds")


@app.get("/metrics")
def metrics():
    # Expose Prometheus metrics
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
@REQUEST_LATENCY.time()
def predict(data: PatientData):
    REQUEST_COUNT.inc()

    input_df = pd.DataFrame([data.model_dump()])
    X_processed = preprocessor.transform(input_df)
    pred_prob = model.predict_proba(X_processed)[:, 1][0]
    prediction = int(pred_prob > 0.5)

    return {"stroke_probability": pred_prob, "stroke_prediction": prediction}
