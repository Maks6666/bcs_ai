import pandas as pd
from typing import Tuple, Union
from abc import ABC, abstractmethod
import logging
from sklearn.model_selection import train_test_split

class DataCleaner(ABC):
    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        ...


class MultiplyData(DataCleaner):
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = pd.concat([data] * 5, ignore_index=True)
            return data
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e

class SplitData(DataCleaner):
    def preprocess_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            x = data.drop("result", axis="columns")
            y = data["result"]

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
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
        
        
