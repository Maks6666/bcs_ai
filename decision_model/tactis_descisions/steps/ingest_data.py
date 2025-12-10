import pandas as pd
from zenml import step
import logging

class DataIngester:
    def __init__(self, data_link: str) -> pd.DataFrame:
        self.data_link = data_link
    
    def read_data(self):
        try:
            return pd.read_csv(self.data_link)
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e 
        

@step()
def ingest_data(data_link: str) -> pd.DataFrame:
    tool = DataIngester(data_link)
    try:
        return tool.read_data() 
    except Exception as e:
        logging.error(f"Error: {e}")
        raise e