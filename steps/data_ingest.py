import logging
from abc import ABC, abstractmethod
from zenml import step
import pandas as pd


class DataIngest(ABC):
    """
    abstract class for data ingest
    """

    def __init__(self, data_path):

        self.data_path = data_path

    def get_data(self):

        logging.info(f"Ingesting data from: {self.data_path}")
        return pd.read_csv(self.data_path)


@step
def Ingest_df(data_path: str) -> pd.DataFrame:

    try:
        ingest_data = DataIngest(data_path)
        df = ingest_data.get_data()
        if "Incident Zip" in df.columns:
            df["Incident Zip"] = df["Incident Zip"].astype(str).str.zfill(5)
        logging.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

        return df
    except Exception as e:
        logging.error("Error in Data Ingest: {}".format(e))
        raise e
