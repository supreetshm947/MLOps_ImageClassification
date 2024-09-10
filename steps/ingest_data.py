from mylogger import get_logger
from torch.utils.data import DataLoader
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from src.data_ingester import ImageIngester

logger = get_logger()


@step
def ingest_data(data_str) -> Tuple[
    Annotated[DataLoader, "train_loader"],
    Annotated[DataLoader, "test_loader"],
    Annotated[int, "num_classes"]
]:
    image_ingestor = ImageIngester(data_str)
    train_loader, test_loader, num_classes = image_ingestor.get_data()
    return train_loader, test_loader, num_classes
