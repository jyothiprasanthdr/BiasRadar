from sklearn.base import RegressorMixin
from zenml import step
import logging
from src.model_dev import XGBoostReg, LightGBM
import pandas as pd
from steps.config import ModelConfigName
import mlflow
from zenml.client import Client
import numpy as np


experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def model_train(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelConfigName,
) -> RegressorMixin:
    """
    trains the model on Ingest Data
    """
    try:
        model = None
        mlflow.sklearn.autolog()
        if config.model_name == "LGBMRegressor":

            model = LightGBM()
        elif config.model_name == "XGBoostRegressor":

            model = XGBoostReg()
        else:
            raise ValueError(f"Model {config.model_name} not supported")

        trained_model = model.train(
            X_train,
            y_train,
            learning_rate=config.learning_rate,
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
        )
        return trained_model

    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e
