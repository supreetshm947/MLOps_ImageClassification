from zenml import step
from torch.utils.data import random_split
from typing import Tuple
from typing_extensions import Annotated
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import DatasetFolder


@step(enable_cache=False)
def split_data(dataset: DatasetFolder, train_val_ratio: float = .9) -> Tuple[
    Annotated[Dataset, "train_dataset"], Annotated[Dataset, "test_dataset"]]:
    train_size = int(train_val_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset
