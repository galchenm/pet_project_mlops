# src/serve_model.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model and preprocessor once
model = joblib.load("models/model.joblib")
preprocessor = joblib.load("models/preprocessor.pkl")

class PatientData(BaseModel):
    age: float
    avg_glucose_level: float
    bmi: float
    gender: str
    ever_married: str
    work_type: str
    Residence_type: str
    smoking_status: str

@app.post("/predict")
def predict(data: PatientData):
    # Convert input to DataFrame with same columns as training data
    import pandas as pd
    input_df = pd.DataFrame([data.dict()])

    # Preprocess
    X_processed = preprocessor.transform(input_df)

    # Predict
    pred_prob = model.predict_proba(X_processed)[:, 1][0]
    prediction = int(pred_prob > 0.5)
    
    return {"stroke_probability": pred_prob, "stroke_prediction": prediction}
