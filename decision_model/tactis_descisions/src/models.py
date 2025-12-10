from sklearn.base import ClassifierMixin
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from abc import ABC, abstractmethod
import logging


class Model(ABC):
    @abstractmethod
    def train(x_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
        ...


class SVC_custom(Model):
    def __init__(self, **kwargs):
        self.model = None
        self.kwargs = kwargs
    
    def train(self, x_train, y_train):
        try:
            self.model = SVC(**self.kwargs)
            self.model.fit(x_train, y_train)
            return self.model
        except Exception as e:
            logging(f"Error: {e}")
            raise e

class DTC_custom(Model):
    def __init__(self, **kwargs):
        self.model = None
        self.kwargs = kwargs
    
    def train(self, x_train, y_train):
        try:
            self.model = DecisionTreeClassifier(**self.kwargs)
            self.model.fit(x_train, y_train)
            return self.model
        except Exception as e: 
            logging(f"Error: {e}")
            raise e
        

class RFC_custom(Model):
    def __init__(self, **kwargs):
        self.model = None
        self.kwargs = kwargs
    
    def train(self, x_train, y_train):
        try:
            self.model = RandomForestClassifier(**self.kwargs)
            self.model.fit(x_train, y_train)
            return self.model
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
        
        