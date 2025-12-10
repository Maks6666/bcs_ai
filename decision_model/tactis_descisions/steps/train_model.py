from src.models import SVC_custom, RFC_custom, DTC_custom
from zenml import step
import pandas as pd
import logging
from sklearn.base import ClassifierMixin

from src.models import SVC_custom, RFC_custom, DTC_custom
from zenml.client import Client 
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(x_train: pd.DataFrame, y_train: pd.Series, model_name: str, params: dict) -> ClassifierMixin:
    try:
        if model_name == "SVC":
            model = SVC_custom(**params)
            trained_model = model.train(x_train=x_train, y_train=y_train)
            mlflow.sklearn.log_model(trained_model, model_name)
            return trained_model

        elif model_name == "RFC":
            model = RFC_custom(**params)
            trained_model = model.train(x_train=x_train, y_train=y_train)
            mlflow.sklearn.log_model(trained_model, model_name)
            return trained_model

        elif model_name == "DTC":
            model = DTC_custom(**params)
            trained_model = model.train(x_train=x_train, y_train=y_train)
            mlflow.sklearn.log_model(trained_model, model_name)
            return trained_model

    except Exception as e:
        logging(f"Error: {e}")
        raise e
