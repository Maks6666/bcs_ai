from zenml import step

import optuna 
from src.models import SVC_custom, RFC_custom, DTC_custom
from sklearn.metrics import accuracy_score
import pandas as pd
import logging
from typing import Tuple

import mlflow
from zenml.client import Client 
import json

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def tune_model(x_train: pd.DataFrame, 
               x_test: pd.DataFrame, 
               y_train: pd.Series, 
               y_test: pd.Series,
               n_trials: int = 15,
               model_name: str = "RFC") -> Tuple[dict, str]:

    if model_name == "RFC":
        def objective(trial):
            try:

                n_estimators = trial.suggest_int("n_estimators", 100, 300)
                max_depth = trial.suggest_int("max_depth", 5, 50)
                min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
                max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

                model = RFC_custom(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features)
                trained_model = model.train(x_train, y_train)

                preds = trained_model.predict(x_test)
                
                score = accuracy_score(y_test, preds)

                print(f"[Trial {trial.number}] Accuracy Score: {score:.4f} | n_estimators: {n_estimators}, max_depth: {max_depth}, min_samples_split: {min_samples_split}, max_features: {max_features}")

                return score

            except Exception as e:
                logging.error(f"Error: {e}")
                raise e 
            
    if model_name == "SVC":
        def objective(trial):
            try:
                C = trial.suggest_int("C", 1.0, 5.0)
                gamma = trial.suggest_categorical("gamma", ["scale", "auto"])

                model = SVC_custom(C=C, gamma=gamma)
                trained_model = model.train(x_train, y_train)
                preds = trained_model.predict(x_test)
                score = accuracy_score(preds, y_test)
                print(f"[Trial {trial.number}] Accuracy Score: {score:.4f} | C: {C}, gamma: {gamma}")
                return score

            except Exception as e:
                logging.error(f"Error: {e}")
                raise e
        
    if model_name == "DTC":
        def objective(trial):
            try:
                max_depth = trial.suggest_int("max_depth", 3, 8)
                min_samples_split = trial.suggest_int("min_samples_split", 5, 15)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 10)
                criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])

                model = DTC_custom(max_dept=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion)
                trained_model = model.train_model(x_train, y_train)
                preds = trained_model.predict(x_test)
                score = accuracy_score(y_test, preds)
                print(f"[Trial {trial.number}] Accuracy Score: {score:.4f} | max_depth: {max_depth}, min_samples_split: {min_samples_split}, min_samples_leaf: {min_samples_leaf}, criterion: {criterion}")
                return score
                
            except Exception as e:
                logging.error(f"Error: {e}")
                raise e
            
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    for key, value in best_params.items():
        mlflow.log_param(key, value)
    mlflow.log_param("model_name", model_name)

    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    mlflow.log_artifact("best_params.json")





    logging.info(f"Best params: {best_params}")
    print(f"Best parameters found: {best_params}")

    return best_params, model_name
