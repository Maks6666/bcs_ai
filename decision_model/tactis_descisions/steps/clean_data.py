from src.clean_data import MultiplyData, SplitData, DataTool
from zenml import step
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
import logging

@step
def preprocess_data(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    
    try: 
        tool = DataTool(data, MultiplyData())
        data = tool.preprocess_data()

        tool = DataTool(data, SplitData())
        x_train, x_test, y_train, y_test = tool.preprocess_data()

        return x_train, x_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error: {e}")
        raise e

