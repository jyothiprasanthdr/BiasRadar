from zenml import step
from sklearn.base import RegressorMixin
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import logging
from typing import Annotated, Tuple
from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> Tuple[
    Annotated[float, "r2_score"], Annotated[float, "rmse"], Annotated[float, "mae"]
]:
    try:
        y_pred = model.predict(X_test)
        # r2 = r2_score(y_test, y_pred)
        # rmse = root_mean_squared_error(y_test, y_pred)
        # mae = mean_absolute_error(y_test, y_pred)
        y_pred_exp = np.expm1(y_pred)
        y_test_exp = np.expm1(y_test)

        r2 = r2_score(y_test_exp, y_pred_exp)
        rmse = root_mean_squared_error(y_test_exp, y_pred_exp)
        mae = mean_absolute_error(y_test_exp, y_pred_exp)

        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        return r2, rmse, mae
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise e
