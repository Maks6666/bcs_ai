from zenml import step
from src.models import RFC_custom, DTC_custom, SVC_custom
import pandas as pd
from sklearn.base import ClassifierMixin
import logging
import mlflow

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(x_train: pd.DataFrame, y_train: pd.Series, best_params: dict, name: str) -> ClassifierMixin:
    try:
       
        if name == "SVC":
            model = SVC_custom(**best_params)
            trained_model = model.train(x_train, y_train)
            mlflow.sklearn.log_model(trained_model, name)
            return trained_model
        
        elif name == "RFC":
            model = RFC_custom(**best_params)
            trained_model = model.train(x_train, y_train)
            mlflow.sklearn.log_model(trained_model, name)
            return trained_model
        
        elif name == "DTC":
            model = DTC_custom(**best_params)
            trained_model = model.train(x_train, y_train)
            mlflow.sklearn.log_model(trained_model, name)
            return trained_model

    except Exception as e:
        logging.error(f"Error: {e}")
        raise e
