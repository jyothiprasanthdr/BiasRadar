# pipelines/training_pipeline.py

from zenml import pipeline
from steps.data_ingest import Ingest_df
from steps.data_clean import data_clean_step
from steps.model_evaluate import evaluate_model
from steps.model_train import model_train
from steps.config import ModelConfigName


@pipeline(enable_cache=False)
def training_pipeline(data_path: str):
    df = Ingest_df(data_path)  # returns artifact
    X_train, X_test, y_train, y_test = data_clean_step(df)  # returns 4 artifacts

    lgbm_config = ModelConfigName(model_name="LGBMRegressor")
    lgbm_model = model_train(X_train, X_test, y_train, y_test, lgbm_config)
    _ = evaluate_model(lgbm_model, X_test, y_test)

    xgb_config = ModelConfigName(model_name="XGBoostRegressor")
    xgb_model = model_train(X_train, X_test, y_train, y_test, xgb_config)
    _ = evaluate_model(xgb_model, X_test, y_test)
