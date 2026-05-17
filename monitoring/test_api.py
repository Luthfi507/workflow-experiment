import os
import time
import pandas as pd
import tqdm
import requests

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_dir, "MLProject", "telco_churn.csv")

api_url = "http://localhost:8000/predict"
df = pd.read_csv(data_path)
data = df.loc[:10]

for _, row in tqdm.tqdm(data.iterrows(), total=data.shape[0]):
    payload = {
        "Contract": row["Contract"],
        "tenure": int(row["tenure"]),
        "MonthlyCharges": float(row["MonthlyCharges"]),
        "TechSupport": row["TechSupport"],
        "OnlineSecurity": row["OnlineSecurity"]
    }
    response = requests.post(api_url, json=payload)
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")

    time.sleep(5) # Simulate delay between requests