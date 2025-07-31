from pydantic import BaseModel, Field
from typing import Literal, Optional


class ModelConfigName(BaseModel):
    model_name: Literal["LGBMRegressor", "XGBoostRegressor"] = "LGBMRegressor"

    # Common hyperparameters (can be extended)
    learning_rate: Optional[float] = Field(
        default=0.1, description="Learning rate for the model"
    )
    n_estimators: Optional[int] = Field(
        default=100, description="Number of boosting rounds"
    )
    max_depth: Optional[int] = Field(
        default=6, description="Maximum depth of the trees"
    )
