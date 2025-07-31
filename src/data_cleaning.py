from abc import ABC, abstractmethod
import logging
import numpy as np
import pandas as pd
import polars as pl
from typing import Union, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pl.DataFrame) -> pl.DataFrame:
        pass


class DataPreProcessingStrategy(DataStrategy):
    def handle_data(self, data: pl.DataFrame) -> pl.DataFrame:
        try:
            logging.info(f"Initial shape: {data.shape}")

            # Normalize column names
            data = data.rename(
                {col: col.strip().lower().replace(" ", "_") for col in data.columns}
            )

            required_cols = ["closed_date", "complaint_type", "borough", "latitude"]
            for col in required_cols:
                if col not in data.columns:
                    raise KeyError(f"Missing required column: {col}")

            logging.info("All required columns present.")

            # Clean ZIP
            if "incident_zip" in data.columns:
                data = data.with_columns(
                    [
                        pl.col("incident_zip")
                        .cast(pl.Utf8)
                        .str.strip_chars()
                        .str.replace_all(r"\\.0+$", "")
                        .str.replace_all(r"[^\d]", "")
                        .alias("incident_zip")
                    ]
                )
                data = data.with_columns(
                    [
                        pl.when(pl.col("incident_zip").str.len_chars() == 0)
                        .then(None)
                        .otherwise(pl.col("incident_zip"))
                        .alias("incident_zip")
                    ]
                )

            # Drop irrelevant columns
            columns_to_drop = [
                "unique_key",
                "incident_address",
                "intersection_street_1",
                "intersection_street_2",
                "street_name",
                "cross_street_1",
                "cross_street_2",
                "landmark",
                "address_type",
                "x_coordinate_state_plane",
                "y_coordinate_state_plane",
                "bbl",
                "park_facility_name",
                "park_borough",
                "vehicle_type",
                "taxi_company_borough",
                "taxi_pick_up_location",
                "bridge_highway_name",
                "bridge_highway_direction",
                "road_ramp",
                "bridge_highway_segment",
                "open_data_channel_type",
                "due_date",
                "resolution_action_updated_date",
                "resolution_description",
                "community_board",
                "location",
                "city",
            ]
            data = data.drop([col for col in columns_to_drop if col in data.columns])

            # Drop nulls in critical fields
            before_drop = data.shape[0]
            data = data.drop_nulls(required_cols)
            logging.info(
                f"Dropped {before_drop - data.shape[0]} rows with nulls in required cols."
            )

            # Fill null ZIPs with mode
            if "incident_zip" in data.columns:
                zip_mode_df = (
                    data.filter(pl.col("incident_zip").is_not_null())
                    .group_by("incident_zip")
                    .agg(pl.count().alias("count"))
                    .sort("count", descending=True)
                    .select("incident_zip")
                )
                zip_mode = (
                    zip_mode_df.to_series()[0] if zip_mode_df.shape[0] > 0 else "00000"
                )
                if zip_mode_df.shape[0] == 0:
                    logging.warning(
                        "[DataPreProcessingStrategy] No valid incident_zip values to impute with."
                    )
                data = data.with_columns([pl.col("incident_zip").fill_null(zip_mode)])

            # Fill null descriptor
            if "descriptor" in data.columns:
                desc_mode_expr = (
                    data.filter(pl.col("descriptor").is_not_null())
                    .group_by(["complaint_type", "descriptor"])
                    .agg(pl.count().alias("count"))
                    .sort("count", descending=True)
                    .group_by("complaint_type")
                    .agg(pl.col("descriptor").first().alias("desc_mode"))
                )
                data = data.join(desc_mode_expr, on="complaint_type", how="left")
                data = data.with_columns(
                    [
                        pl.when(pl.col("descriptor").is_null())
                        .then(pl.col("desc_mode"))
                        .otherwise(pl.col("descriptor"))
                        .alias("descriptor")
                    ]
                ).drop("desc_mode")

            if "location_type" in data.columns:
                data = data.with_columns([pl.col("location_type").fill_null("Missing")])

            # Parse datetime safely
            # Clean and parse datetime fields
            data = data.with_columns(
                [
                    pl.col("created_date")
                    .str.strip_chars()
                    .str.strptime(
                        pl.Datetime("us"), format="%m/%d/%Y %I:%M:%S %p", strict=False
                    )
                    .alias("created_date"),
                    pl.col("closed_date")
                    .str.strip_chars()
                    .str.strptime(
                        pl.Datetime("us"), format="%m/%d/%Y %I:%M:%S %p", strict=False
                    )
                    .alias("closed_date"),
                ]
            )

            # Calculate resolution time in microseconds and convert to hours
            # First: calculate the duration in microseconds
            data = data.with_columns(
                [
                    (pl.col("closed_date") - pl.col("created_date"))
                    .cast(pl.Int64)
                    .alias("resolution_time_us")
                ]
            )

            # Then: convert microseconds to hours using the newly created column
            data = data.with_columns(
                [
                    (pl.col("resolution_time_us") / 3_600_000_000).alias(
                        "resolution_time_hrs"
                    )
                ]
            )

            # Filter invalid durations
            data = data.filter(pl.col("resolution_time_hrs").is_not_null())
            invalid_rows = data.filter(pl.col("resolution_time_hrs") < 0)
            logging.warning(
                f"Found {invalid_rows.shape[0]} rows with negative resolution time."
            )
            data = data.filter(pl.col("resolution_time_hrs") >= 0)

            if data.is_empty():
                raise ValueError(
                    "[DataPreProcessingStrategy] No valid resolution_time_hrs remaining after filter."
                )

            # Cap and log transform
            upper = data.select(pl.col("resolution_time_hrs")).quantile(0.995).item()
            data = data.with_columns(
                [
                    pl.when(pl.col("resolution_time_hrs") > upper)
                    .then(upper)
                    .otherwise(pl.col("resolution_time_hrs"))
                    .alias("resolution_time_hrs"),
                    pl.col("resolution_time_hrs")
                    .log1p()
                    .alias("log_resolution_time_hrs"),
                ]
            )

            logging.info(
                f"log_resolution_time_hrs stats: {data.select('log_resolution_time_hrs').describe()}"
            )

            # Complaint grouping
            complaint_mapping = {
                "Noise - Residential": "Noise",
                "Noise - Commercial": "Noise",
                "Noise - Street/Sidewalk": "Noise",
                "Noise - Vehicle": "Noise",
                "Noise": "Noise",
                "Illegal Parking": "Parking",
                "Blocked Driveway": "Parking",
                "Water Leak": "Plumbing",
                "Water System": "Plumbing",
                "PAINT/PLASTER": "Plumbing",
                "PLUMBING": "Plumbing",
                "HEAT/HOT WATER": "Heat/Water",
                "UNSANITARY CONDITION": "Sanitation",
                "Dirty Condition": "Sanitation",
                "Street Condition": "Street",
                "Traffic Signal Condition": "Street",
                "Street Light Condition": "Street",
                "Derelict Vehicles": "Street",
                "Abandoned Vehicle": "Street",
                "Encampment": "Public Safety",
                "DOOR/WINDOW": "Maintenance",
                "Snow or Ice": "Weather",
                "General": "Other",
            }
            complaint_keys = list(complaint_mapping.keys())
            data = data.with_columns(
                [
                    pl.when(pl.col("complaint_type").is_in(complaint_keys))
                    .then(
                        pl.col("complaint_type").map_elements(
                            lambda x: complaint_mapping.get(x, "Other"),
                            return_dtype=pl.String,
                        )
                    )
                    .otherwise(pl.lit("Other"))
                    .alias("complaint_grouped")
                ]
            )

            logging.info(f"Final shape: {data.shape}")
            return data

        except Exception as e:
            logging.error(f"[DataPreProcessingStrategy] Error: {e}")
            raise e


class DataDivideStrategy(DataStrategy):
    def handle_data(
        self, data: Union[pl.DataFrame, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        try:
            if isinstance(data, pl.DataFrame):
                data = data.to_pandas()

            data = data.copy()

            if "created_date" not in data.columns:
                raise KeyError("Missing 'created_date' in input data.")

            data["created_hour"] = data["created_date"].dt.hour
            data["created_dayofweek"] = data["created_date"].dt.dayofweek
            data["created_month"] = data["created_date"].dt.month
            data["is_weekend"] = data["created_dayofweek"].isin([5, 6]).astype(int)

            target = "log_resolution_time_hrs"
            if target not in data.columns:
                raise KeyError(
                    f"[DataDivideStrategy] Target column '{target}' missing from data."
                )

            null_target = data[target].isna().sum()
            logging.info(f"[DataDivideStrategy] Null '{target}' values: {null_target}")

            data = data[data[target].notna()]
            logging.info(f"[DataDivideStrategy] Rows after filtering: {len(data)}")

            if data.empty:
                raise ValueError(
                    "[DataDivideStrategy] No valid rows remaining after filtering."
                )

            features = [
                "complaint_type",
                "descriptor",
                "borough",
                "agency",
                "incident_zip",
                "location_type",
                "facility_type",
                "latitude",
                "longitude",
                "created_hour",
                "created_dayofweek",
                "created_month",
                "is_weekend",
            ]

            missing_features = [col for col in features if col not in data.columns]
            if missing_features:
                raise KeyError(
                    f"[DataDivideStrategy] Missing features: {missing_features}"
                )

            X = data[features]
            y = data[target]

            categorical_cols = [
                "complaint_type",
                "descriptor",
                "borough",
                "agency",
                "incident_zip",
                "location_type",
                "facility_type",
            ]
            numeric_cols = [
                "latitude",
                "longitude",
                "created_hour",
                "created_dayofweek",
                "created_month",
            ]

            preprocessor = ColumnTransformer(
                [
                    (
                        "cat",
                        Pipeline(
                            [
                                ("impute", SimpleImputer(strategy="most_frequent")),
                                (
                                    "encode",
                                    OneHotEncoder(
                                        handle_unknown="ignore", drop="first"
                                    ),
                                ),
                            ]
                        ),
                        categorical_cols,
                    ),
                    (
                        "num",
                        Pipeline(
                            [
                                ("impute", SimpleImputer(strategy="mean")),
                                ("scale", StandardScaler()),
                            ]
                        ),
                        numeric_cols,
                    ),
                ],
                remainder="passthrough",
            )

            X_processed = preprocessor.fit_transform(X)
            if hasattr(X_processed, "toarray"):
                X_processed = X_processed.toarray()

            return train_test_split(X_processed, y, test_size=0.2, random_state=42)

        except Exception as e:
            logging.error(f"[DataDivideStrategy] Error: {e}")
            raise e


class DataCleaning:
    def __init__(self, data: Union[pd.DataFrame, pl.DataFrame], strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pl.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"[DataCleaning] Error in handling data: {e}")
            raise e
