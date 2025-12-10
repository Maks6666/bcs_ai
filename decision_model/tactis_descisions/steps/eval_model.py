from src.eval import F1, AS, RMSE
from zenml import step
from sklearn.base import ClassifierMixin
import pandas as pd
import logging 
from typing_extensions import Annotated
from typing import Tuple

from zenml.client import Client 
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def eval(model: ClassifierMixin, x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Annotated[float, "f1_score"],
                                                                                   Annotated[float, "accuracy_score"],
                                                                                   Annotated[float, "rmse"]]:
    try:
        preds = model.predict(x_test)

        f1_func = F1()
        f1_score = f1_func.calculate(y_test, preds)
        

        as_func = AS()
        accuracy_score = as_func.calculate(y_test, preds)

        rsme_func = RMSE()
        rmse_score = rsme_func.calculate(y_test, preds)

        return f1_score, accuracy_score, rmse_score
    
    except Exception as e: 
        logging.error(f"Error: {e}")
        raise e