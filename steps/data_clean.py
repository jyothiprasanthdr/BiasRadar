import logging
from zenml import step
import pandas as pd
from src.data_cleaning import (
    DataDivideStrategy,
    DataPreProcessingStrategy,
    DataCleaning,
)
from typing import Tuple
import numpy as np
import polars as pl


@step
def data_clean_step(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:

    try:
        df_polars = pl.from_pandas(df)
        preprocess = DataPreProcessingStrategy()
        data_cleaner = DataCleaning(df_polars, preprocess)
        preprocessed_data = data_cleaner.handle_data()

        # Convert back to pandas for sklearn compatibility
        preprocessed_data_pd = preprocessed_data.to_pandas()

        split_features = DataDivideStrategy()
        data_cleaner = DataCleaning(preprocessed_data_pd, split_features)
        X_train, X_test, y_train, y_test = data_cleaner.handle_data()

        logging.info("Data cleaning complete.")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f" Error in cleaning data: {e}")
        raise e
