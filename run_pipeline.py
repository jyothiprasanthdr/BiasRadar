from pipelines.training_pipeline import training_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(data_path="data/NYC_311_2k10_to_2k25.csv")
    # training_pipeline(data_path="data/nyc_311_2022_to_2025_sample_150k.csv")
    # training_pipeline(
    #     data_path="data/311_Service_Requests_from_2010_to_Present_20250715.csv"
    # )
