import json
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from time import time
from loguru import logger
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("telco-customer-churn")

FEATURES = ["Contract", "tenure", "MonthlyCharges", "TechSupport", "OnlineSecurity"]
TARGET = "Churn"
CAT_FEATURES = ["Contract", "TechSupport", "OnlineSecurity"]
NUM_FEATURES = ["tenure", "MonthlyCharges"]
LABELS = ["No Churn", "Churn"]
CM_PATH = "confusion_matrix.png"
RUN_NAME = "churn-classifier"
MODEL = LogisticRegression()

def build_pipeline(base_model) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_FEATURES),
            ("num", StandardScaler(), NUM_FEATURES),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", base_model),
    ])
    return pipeline

def load_and_validate(file_path: str):
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Feature '{missing}' not found in the dataset.")

    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

    le_target = LabelEncoder()
    df[TARGET] = le_target.fit_transform(df[TARGET])

    return df[FEATURES + [TARGET]]

def split_data(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train(df: pd.DataFrame, base_model, param_grid: dict):
    logger.info("Model training started")
    start = time()

    x_train, x_test, y_train, y_test = split_data(df)

    pipeline = build_pipeline(base_model)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,      
        verbose=1,
        refit=True,     
    )
    grid_search.fit(x_train, y_train)

    elapsed = time() - start
    logger.success(f"Model trained successfully in {elapsed:.4f}s | best_score={grid_search.best_score_:.4f}")
    return grid_search, x_test, y_test

def evaluate(pipeline, x_test, y_test):
    start = time()

    y_pred = pipeline.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CM_PATH)
    plt.close(fig)

    elapsed = time() - start
    logger.success(f"Data evaluated successfully in {elapsed:.4f}")
    return metrics

def run(file_path: str, param_grid: dict):
    logger.info("=== Pipeline Started ===")
    total_start = time()

    df = load_and_validate(file_path)

    # with mlflow.start_run(run_name=f"{RUN_NAME}-{version}", nested=True):
    grid_search, x_test, y_test = train(df, MODEL, param_grid)

    best_pipeline = grid_search.best_estimator_  
    best_params = grid_search.best_params_

    metrics = evaluate(best_pipeline, x_test, y_test)

    # Log ke MLflow
    mlflow.log_params(best_params)
    mlflow.log_param("param_grid", json.dumps(param_grid))
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(CM_PATH)

    mlflow.sklearn.log_model(
        sk_model=best_pipeline,
        artifact_path="pipeline",
        input_example=x_test.iloc[:3],
    )

    total_elapsed = time() - total_start
    logger.success(f"=== Pipeline processed successfully in {total_elapsed:.4f}s ===")

def parse_params(s):
    result = []

    for v in s.split(","):
        v = v.strip()

        # Handle None
        if v.lower() == "none":
            result.append(None)
            continue

        # Handle boolean
        if v.lower() == "true":
            result.append(True)
            continue

        if v.lower() == "false":
            result.append(False)
            continue

        # Handle int
        try:
            result.append(int(v))
            continue
        except ValueError:
            pass

        # Handle float
        try:
            result.append(float(v))
            continue
        except ValueError:
            pass

        # Fallback -> string
        result.append(v)

    return result

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--C", type=str, default="0.01,0.1,1,10,100")
    parser.add_argument("--solver", type=str, default="liblinear,lbfgs")
    parser.add_argument("--max_iter", type=str, default="100,200,300")
    parser.add_argument("--penalty", type=str, default="l1,l2")
    args = parser.parse_args()

    PARAM_GRID = {
        "classifier__C": parse_params(args.C),
        "classifier__solver": parse_params(args.solver),
        "classifier__max_iter": parse_params(args.max_iter),
        "classifier__penalty": parse_params(args.penalty)
    }

    run(
        file_path=args.data_path,
        param_grid=PARAM_GRID,
    )

if __name__ == "__main__":
    main()