from pipelines.training_pipeline import train_pipeline
import os
from zenml.client import Client

from src.data_ingester import ImageIngester

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline("data")
    # data = ImageIngester("data")
    # print(data.get_data())

# mlflow ui --backend-store-uri file:C:\Users\Supreet\AppData\Roaming\zenml\local_stores\eaa91478-6f29-466d-ad57-a82844d64630\mlruns
