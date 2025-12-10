import pandas as pd
from zenml import step
import logging



class DataIngester:
    def __init__(self, link: str) -> pd.DataFrame:
        self.link = link 
    def read_data(self):
        try:
            data = pd.read_csv(self.link)
            return data
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e



@step
def ingest_data(link: str) -> pd.DataFrame:
    try:
        tool = DataIngester(link)
        data = tool.read_data()
        return data
    except Exception as e:
        logging.error(f"Error: {e}")
        raise e