from dotenv import load_dotenv
import os
from fastapi import FastAPI
import pandas as pd
from typing import Literal
from pydantic import BaseModel, Field
import mlflow

load_dotenv()

LABELS = ["No Churn", "Churn"]
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

model_name = "telco-churn"
model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
print(f"Loaded model: {model_name}")

class PredictionData(BaseModel):
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., description="Type of contract")
    tenure: int = Field(..., description="Number of months the customer has stayed with the company")
    MonthlyCharges: float = Field(..., description="Monthly charges")
    TechSupport: Literal["No", "Yes", "No internet service"] = Field(..., description="Tech support status")
    OnlineSecurity: Literal["No", "Yes", "No internet service"] = Field(..., description="Online security status")

app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn with monitoring",
    version="1.0.0"
)

@app.post("/predict")
def predict(data: PredictionData):
    """
    Args:
        Contract: "Month-to-month", "One year", "Two year"
        tenure: int
        MonthlyCharges: float
        TechSupport: "No", "Yes", "No internet service"
        OnlineSecurity: "No", "Yes", "No internet service"

    Returns:
        prediction: 0 or 1
        confidence: float (0.0 to 1.0)
    """
    df = pd.DataFrame([data.model_dump()])
    pred_id = model.predict(df)[0]
    confidence = model.predict_proba(df)[0][1]

    return {
        "prediction": int(pred_id),
        "confidence": round(confidence, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)