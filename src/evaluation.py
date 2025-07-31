from abc import ABC, abstractmethod
import logging
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


class Evaluation(ABC):
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Abstract method to calculate evaluation score.
        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.
        """
        pass


class MAE(Evaluation):
    def calculate_scores(self, y_true, y_pred):
        try:
            logging.info("Calculating MAE")
            mae = mean_absolute_error(y_true, y_pred)
            logging.info(f"MAE: {mae}")
            return mae
        except Exception as e:
            logging.error(f"Error in MAE calculation: {e}")
            raise e


class RMSE(Evaluation):
    def calculate_scores(self, y_true, y_pred):
        try:
            logging.info("Calculating RMSE")
            rmse = root_mean_squared_error(y_true, y_pred)
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in RMSE calculation: {e}")
            raise e


class R2_score(Evaluation):
    def calculate_scores(self, y_true, y_pred):
        try:
            logging.info("Calculating R² score")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R²: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in R² score calculation: {e}")
            raise e
