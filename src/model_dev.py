from abc import ABC, abstractmethod
import logging
import lightgbm
from sklearn.base import RegressorMixin
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        Args:
            X_train: Training Data
            y_train_ Training Labels
        Returns:
             None

        """
        pass


class XGBoostReg(Model):

    def train(self, X_train, y_train, **kwargs) -> RegressorMixin:

        try:
            logging.info("XGBOOST Model training initiated")
            xgboost = XGBRegressor(**kwargs)
            xgboost.fit(X_train, y_train)
            logging.info("XGBOOST Model training Completed")
            return xgboost

        except Exception as e:
            logging.error("Error in XGBoost: {}".format(e))
            raise e


class LightGBM(Model):

    def train(self, X_train, y_train, **kwargs) -> RegressorMixin:

        try:
            logging.info("LGBM Model training initiated")
            lgbm = LGBMRegressor(**kwargs)
            lgbm.fit(X_train, y_train)
            logging.info("LGBM Model training Completed")
            return lgbm

        except Exception as e:
            logging.error("Error in LGBM: {}".format(e))
            raise e
