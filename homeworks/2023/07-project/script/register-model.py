import os
import pickle
import sys

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import mlflow

MODEL_BUCKET = os.getenv("MODEL_BUCKET")

REMOTE_TRACKING_IP = os.getenv("REMOTE_IP", "localhost")
MLFLOW_TRACKING_URI = f"http://{REMOTE_TRACKING_IP}:5000"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "chosen-models-loan"
MODEL_NAME = "loan-prediction"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def read_data():
    df_test = pd.read_csv("../data/test.csv")
    X_test, y_test = df_test.drop('Personal Loan', axis=1), df_test['Personal Loan']
    return X_test, y_test


run_id = sys.argv[1]

X_test, y_test = read_data()
print(X_test.shape)

logged_model = f"runs:/{run_id}/model"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
y_pred = loaded_model.predict(X_test)

test_score = f1_score(y_test, y_pred)
try:
    client.create_registered_model(name=MODEL_NAME)
except:
    pass
description = f"test score: {test_score}"
mv = client.create_model_version(
    name=MODEL_NAME, source=logged_model, run_id=run_id, description=description
)
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=mv.version,
    stage="production",
    archive_existing_versions=True,
)

# Update Model to bucket

mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{run_id}/model/model.xgb", dst_path="../model"
)

