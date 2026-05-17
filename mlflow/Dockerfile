# Build an image that can serve mlflow models.
FROM python:3.12.7-slim

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    nginx \
    build-essential \
    gcc \
    g++ \
    curl 

WORKDIR /opt/mlflow

# Install MLflow
RUN pip install --no-cache-dir  mlflow==2.19.0 psycopg2-binary boto3

# Expose the MLflow tracking server port
EXPOSE 5000

# Start the MLflow tracking server
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
    --default-artifact-root $MLFLOW_DEFAULT_ARTIFACT_ROOT