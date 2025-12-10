import pandas as pd
import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Union

class DataCleaner(ABC):
    @abstractmethod
    def preprocess_data(self) -> Union[pd.DataFrame, pd.Series]:
        ...


class DataSpliter(DataCleaner):
    def preprocess_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            x = data.drop("descision", axis="columns")
            y = data["descision"]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
            return x_train, x_test, y_train, y_test
        except Exception as e: 
            logging.error(f"Error: {e}")
            raise e
        

class DataTool:
    def __init__(self, data: pd.DataFrame, tool: DataCleaner):
        self.data = data
        self.tool = tool
    def preprocess_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.tool.preprocess_data(self.data)
        except Exception as e: 
            logging.error(f"Error: {e}")
            raise e
    