from abc import ABC, abstractmethod
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import root_mean_squared_error as rmse
import numpy as np
import logging
import mlflow

class Metric(ABC):
    @abstractmethod
    def calculate(y_test: np.ndarray, y_pred: np.ndarray):
        ...


class F1(Metric):
    def calculate(self, y_test: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating f1_score...")
            f_1 = f1_score(y_test, y_pred, average='weighted')
            logging.info(f"F1_score: {f_1}")
            mlflow.log_metric("F1_score", f_1)
            return f_1
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
        
class AS(Metric):
    def calculate(self, y_test: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating accuracy_score...")
            a_s = accuracy_score(y_test, y_pred)
            logging.info(f"Accuracy_score: {a_s}")
            mlflow.log_metric("accuracy_score", a_s)
            return a_s
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
        
class RMSE(Metric):
    def calculate(self, y_test: np.ndarray, y_preds: np.ndarray):
        try:
            logging.info("Calculating RMSE...")
            rmse_score = rmse(y_test, y_preds) 
            logging.info(f"RMSE: {rmse_score}")
            mlflow.log_metric("RMSE", rmse_score)
            return rmse_score
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
    
