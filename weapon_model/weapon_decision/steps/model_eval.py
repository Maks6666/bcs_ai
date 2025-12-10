from src.metrics import AS, F1, RMSE
from zenml import step
from sklearn.base import ClassifierMixin
import pandas as pd
from typing_extensions import Annotated
from typing import Tuple
import logging

import mlflow
from zenml.client import Client 

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def eval(model: ClassifierMixin, x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Annotated[float, "accuracy_score"],
                                                                                   Annotated[float, "f1_score"],
                                                                                   Annotated[float, "rmse"],]:
    try:
        y_preds = model.predict(x_test)

        as_metric = AS()
        a_s = as_metric.calculate(y_test, y_preds)
        mlflow.log_metric('accuracy_score', a_s)

        f1_metric = F1()
        f_1 = f1_metric.calculate(y_test, y_preds)
        mlflow.log_metric('F1', f_1)

        rmse_metric = RMSE()
        rm_se = rmse_metric.calculate(y_test, y_preds)
        mlflow.log_metric('RMSE', rm_se)

        return a_s, f_1, rm_se
    
    except Exception as e:
        logging.error(f"Error: {e}")
        raise e
