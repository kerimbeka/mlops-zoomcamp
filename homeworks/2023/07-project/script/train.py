
import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from prefect.blocks.system import String
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import mlflow
from prefect import flow, task

#REMOTE_TRACKING_IP = os.getenv("REMOTE_IP", "localhost")
#MLFLOW_TRACKING_URI = f"http://{REMOTE_TRACKING_IP}:5000"
#MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

HPO_EXPERIMENT_NAME = "hpo-xgboost-loan"
EXPERIMENT_NAME = "chosen-models-loan"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


@task
def read_data():
    """
    Reads data"""
    df_train = pd.read_csv("../data/train.csv")
    df_val = pd.read_csv("../data/val.csv")
    X_train, y_train = df_train.drop('Personal Loan', axis=1), df_train['Personal Loan']
    X_val, y_val = df_val.drop('Personal Loan', axis=1), df_val['Personal Loan'] 
    return X_train, y_train, X_val, y_val


def predict_binary(probs):
    return (probs >= 0.5).astype("int")


@task
def hyperoptimizer(X_train, y_train, X_val, y_val, xgb_param_grid):
    """
    Does the Hyperparamater passes over the data"""

    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            model = XGBClassifier(verbosity=0)
            model.set_params(**params)
            model.fit(X_train, y_train)
            y_pred_probs = model.predict(X_val)
            y_pred = predict_binary(y_pred_probs)

            
            mlflow.log_metric("Accuracy", accuracy_score(y_val, y_pred))
            mlflow.log_metric("Precision", precision_score(y_val, y_pred, average='binary'))
            mlflow.log_metric("Recall", recall_score(y_val, y_pred, average='binary'))
            f1 = f1_score(y_val, y_pred, average='binary')
            mlflow.log_metric("Fscore", f1)
            mlflow.log_metric("AUC", roc_auc_score(y_val, model.predict_proba(X_val)[::,1]))

        return {"loss": -f1, "status": STATUS_OK}

    best_result = fmin(
        fn=objective,
        space=xgb_param_grid,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials(),
        return_argmin=False,
    )
    print("best estimate parameters",best_result)
    return 0


@task
def train_and_log_model(params, X_train, y_train, X_val, y_val, tag, xgb_param_grid):
    """
    Once given the parameters of the model, it retrains and saves all output along with a version tag"""

    with mlflow.start_run():
        mlflow.set_tag("version_tag", tag)
        print(params)
        params = params #space_eval(xgb_param_grid, params)
        model = XGBClassifier(verbosity=0)
        model.set_params(**params)
        model.fit(X_train, y_train)

        # evaluate model on the validation set
        start_time = time.time()
        y_pred = model.predict(X_val)
        end_time = time.time()
        inference_time = end_time - start_time
        f1 = f1_score(y_val, y_pred, average='binary')
        
        mlflow.log_metric("Accuracy", accuracy_score(y_val, y_pred))
        mlflow.log_metric("Precision", precision_score(y_val, y_pred, average='binary'))
        mlflow.log_metric("Recall", recall_score(y_val, y_pred, average='binary'))
        mlflow.log_metric("Fscore", f1)
        mlflow.log_metric("AUC", roc_auc_score(y_val, model.predict_proba(X_val)[::,1]))
        mlflow.log_metric("Inference time", inference_time)
        

@flow
def run(log_top, X_train, y_train, X_val, y_val, xgb_param_grid, *args):
    """
    Passes over the best log_top experiments and calls train_and_log_params on each"""
    # Setup version tag
    current_tag_block = String.load("version-counter")
    print(current_tag_block)
    current_tag = int(current_tag_block.value)

    # retrieve the top_n model runs and log the models to MLflow
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=log_top,
        order_by=["metrics.Fscore DESC"],
    )
    for run in runs:
        train_and_log_model(
            params=run.data.params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            #tag=current_tag,
            tag='1.0',
            xgb_param_grid=xgb_param_grid
        )
    
    # Clean the HPO Experiment, we only need the chosen models
    experiment_id = client.get_experiment_by_name(HPO_EXPERIMENT_NAME).experiment_id
    all_runs = client.search_runs(experiment_ids=experiment_id)
    for mlflow_run in all_runs:
        client.delete_run(mlflow_run.info.run_id)
    # Updates Version tag
    new_tag = String(value=f"{current_tag + 1}")
    new_tag.save(name="version-counter", overwrite=True)


@flow
def main():
    """
    Main function. Reads data, preprocesses it and gives out the best models"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(HPO_EXPERIMENT_NAME)

    X_train, y_train, X_val, y_val = read_data()
    
    ratio = sum(y_train==0)/sum(y_train==1) 

    xgb_param_grid = {
        'max_depth': scope.int(hp.choice('max_depth', [5, 6, 7])),
        'learning_rate': scope.float(hp.choice('learning_rate', [0.1, 0.2, 0.3])),
        'n_estimators': scope.int(hp.choice('n_estimators', [50, 100, 200])),
        'min_child_weight': scope.int(hp.choice('min_child_weight', [1, 5, 10])),
        'scale_pos_weight': scope.float(hp.choice('scale_pos_weight', [ratio, ratio*1.3, ratio*1.5])),
        'subsample': scope.float(hp.choice('subsample', [0.6, 0.8, 1.0])),
        'colsample_bytree': scope.float(hp.choice('colsample_bytree', [0.6, 0.8, 1.0])),
        'colsample_bylevel': scope.float(hp.choice('colsample_bylevel', [0.6, 0.8, 1.0])),
        'reg_alpha': scope.float(hp.choice('reg_alpha', [0.0, 0.1, 1.0])),
        'reg_lambda': scope.float(hp.choice('reg_lambda', [0.0, 0.1, 1.0])),
        'max_delta_step': scope.int(hp.choice('max_delta_step', [0, 1, 2])),
        'gamma': scope.float(hp.choice('gamma', [0.0, 0.1, 1.0])),
        'max_leaf_nodes': scope.int(hp.choice('max_leaf_nodes', [2, 4, 6])),
        'booster': hp.choice('booster', ['gbtree']),
        'objective': hp.choice('objective', ['binary:logistic']),
        'eval_metric': hp.choice('eval_metric', ['error']), 
        'random_state': scope.int(hp.choice('random_state', [0])),
    }

    hpo_return_code = hyperoptimizer(X_train, y_train, X_val, y_val, xgb_param_grid)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.xgboost.autolog()
    run(10, X_train, y_train, X_val, y_val, xgb_param_grid, hpo_return_code)
