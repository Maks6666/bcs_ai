from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import root_mean_squared_error as rmse

from abc import ABC, abstractmethod
import numpy as np

import logging

class Metric(ABC):
    @abstractmethod
    def calculate(y_test: np.ndarray, y_preds: np.ndarray):
        ...


class AS(Metric): 
    def calculate(self, y_test: np.ndarray, y_preds: np.ndarray):
        try:
            logging.info("Calculating accuracy_score...")
            a_s = accuracy_score(y_test, y_preds) 
            logging.info(f"Accuracy_score: {a_s}")
            return a_s
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e

class F1(Metric):
    def calculate(self, y_test: np.ndarray, y_preds: np.ndarray):
        try:
            logging.info("Calculating f1_score...")
            f_1 = accuracy_score(y_test, y_preds) 
            logging.info(f"F1_score...: {f_1}")
            return f_1
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e

class RMSE(Metric):
    def calculate(self, y_test: np.ndarray, y_preds: np.ndarray):
        try:
            logging.info("Calculating RMSE...")
            rm_se = rmse(y_test, y_preds) 
            logging.info(f"RMSE: {rm_se}")
            return rm_se
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
