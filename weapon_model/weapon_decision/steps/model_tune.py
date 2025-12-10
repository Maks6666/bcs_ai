from zenml import step 
import optuna 
import logging
import pandas as pd
from sklearn.base import ClassifierMixin
from typing import Tuple
from sklearn.metrics import accuracy_score

from src.models import SVC_custom, RFC_custom, DTC_custom

import mlflow
from zenml.client import Client
import json 

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def tune(model_name: str, n_trials: int, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:

    if model_name == "SVC":
        def objective(trial):
            try:
                C = trial.suggest_int("C", 1.0, 5.0)
                gamma = trial.suggest_categorical("gamma", ["scale", "auto"])

                model = SVC_custom(C=C, gamma=gamma)
                trained_model = model.train(x_train, y_train)
                y_preds = trained_model.predict(x_test)

                score = accuracy_score(y_test, y_preds)
                return score
            except Exception as e:
                logging.error(f"Error: {e}")
                raise e
    
    elif model_name == "RFC":
        def objective(trial):
            try:
                n_estimators = trial.suggest_int("n_estimators", 100, 500)
                max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
                bootstrap = trial.suggest_categorical("bootstrap", [True, False])
                max_depth = trial.suggest_int("max_depth", 3, 10)

                model = RFC_custom(n_estimators=n_estimators, max_features=max_features, bootstrap=bootstrap, max_depth=max_depth)
                trained_model = model.train(x_train, y_train)
                y_preds = trained_model.predict(x_test)
                score = accuracy_score(y_test, y_preds)
                return score
            except Exception as e:
                logging.error(f"Error: {e}")
                raise e

    elif model_name == "DTC":
        def objective(trial):
            try:
                max_depth = trial.suggest_int("max_depth", 3, 8)
                min_samples_split = trial.suggest_int("min_samples_split", 5, 15)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 10)
                criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])

                model = DTC_custom(max_dept=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion)
                trained_model = model.train(x_train, y_train)
                y_preds = trained_model.predict(x_test)
                score = accuracy_score(y_test, y_preds)
                return score
            except Exception as e:
                logging.error(f"Error: {e}")
                raise e
            

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    for key, value in best_params.items():
        mlflow.log_param(key, value)

    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    mlflow.log_artifact("best_params.json")
    
    mlflow.log_param("model_name", model_name)

    logging.info(f"Best params: {best_params}")
    print(f"Best parameters found: {best_params}")

    return best_params
    