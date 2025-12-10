from zenml import step 
from src.data_cleaner import DataSpliter, DataTool
import pandas as pd
import logging
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_data(data: pd.DataFrame) -> Tuple[Annotated[pd.DataFrame, "x_train"],
                                            Annotated[pd.DataFrame, "x_test"],
                                            Annotated[pd.Series, "y_train"],
                                            Annotated[pd.Series, "y_test"]]:
    try:
        divider = DataTool(data, DataSpliter())
        x_train, x_test, y_train, y_test = divider.preprocess_data()
        return x_train, x_test, y_train, y_test
    
    except Exception as e: 
            logging.error(f"Error: {e}")
            raise e