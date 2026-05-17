FROM python:3.12.7-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5080
CMD ["uvicorn", "prometheus_exporter:app", "--host", "0.0.0.0", "--port", "5080"]