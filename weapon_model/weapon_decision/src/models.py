from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import pandas as pd
from abc import ABC, abstractmethod
import logging

class Model(ABC):
    @abstractmethod
    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
        ...


class RFC_custom(Model):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None
    def train(self, x_train: pd.DataFrame, y_train: pd.Series):
        try:
            self.model = RandomForestClassifier(**self.kwargs)
            self.model.fit(x_train, y_train)
            return self.model
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e

class DTC_custom(Model):
    def __init__(self, **kwargs):
        self.model = None
        self.kwargs = kwargs
    def train(self, x_train: pd.DataFrame, y_train: pd.Series):
        try:
            self.model = DecisionTreeClassifier(**self.kwargs)
            self.model.fit(x_train, y_train)
            return self.model
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
        
class SVC_custom(Model):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None
    def train(self, x_train: pd.DataFrame, y_train: pd.Series):
        try:
            self.model = SVC(**self.kwargs)
            self.model.fit(x_train, y_train)
            return self.model
        except Exception as e: 
            logging.error(f"Error: {e}")
            raise e
