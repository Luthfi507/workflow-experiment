import os
import time
import psutil
from dotenv import load_dotenv
from fastapi import FastAPI, Response
import pandas as pd
from typing import Literal
from pydantic import BaseModel, Field
import mlflow
from prometheus_client import Gauge, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

load_dotenv()

LABELS = ["No Churn", "Churn"]
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

model_name = "telco-churn"
model = None

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')  
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency')  
THROUGHPUT = Counter('http_requests_throughput', 'Total number of requests per second')  
MODEL_PREDICTED = Gauge("model_predicted", "Model predicted class")
MODEL_CONFIDENCE = Gauge("model_confidence", "Model output probability")
 
# Metrik untuk sistem
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')  
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')

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

@app.on_event("startup")
async def load_model():
    global model
    model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
    print(f"Loaded model: {model_name}")

@app.get('/metrics')
def metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=1)) 
    RAM_USAGE.set(psutil.virtual_memory().percent)
    
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

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
    start_time = time.time()
    df = pd.DataFrame([data.model_dump()])

    try:
        pred = model.predict(df)[0]
        conf = model.predict_proba(df)[0][1]
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        MODEL_PREDICTED.set(pred)
        MODEL_CONFIDENCE.set(conf)

        return {
            "prediction": LABELS[pred],
            "confidence": round(conf, 2)
        }

    except Exception as e:
        return {"error": str(e)}
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5080)