from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
from prometheus_client import Gauge, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__file__)

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')  
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency')  
THROUGHPUT = Counter('http_requests_throughput', 'Total number of requests per second')  
MODEL_PREDICTED = Gauge("model_predicted", "Model predicted class")
MODEL_CONFIDENCE = Gauge("model_confidence", "Model output probability")
 
# Metrik untuk sistem
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')  
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')

@app.route('/metrics', methods=['GET'])
def metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=1)) 
    RAM_USAGE.set(psutil.virtual_memory().percent)
    
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()  
    THROUGHPUT.inc()  

    api_url = "http://127.0.0.1:5050/predict"
    data = request.get_json()

    try:
        response = requests.post(api_url, json=data)
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        response.raise_for_status()
        result = response.json()
        pred = result['prediction']
        conf = result['confidence']
        MODEL_PREDICTED.set(pred)
        MODEL_CONFIDENCE.set(conf)
        return jsonify(result)

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5080)