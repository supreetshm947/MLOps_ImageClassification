from mylogger import get_logger
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from src.data_ingester import ImageFolderIngestor

logger = get_logger()


@step(enable_cache=False)
def ingest_data(data_str) -> Tuple[
    Annotated[DatasetFolder, "dataset"],
    Annotated[int, "num_classes"]
]:
    image_ingestor = ImageFolderIngestor(data_str)
    dataset, num_classes = image_ingestor.get_data()
    return dataset, num_classes
