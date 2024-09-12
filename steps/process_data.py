from zenml import step
from torch.utils.data import random_split
from typing import Tuple
from typing_extensions import Annotated
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets.folder import DatasetFolder

@step(enable_cache=False)
def process_data(dataset:Dataset, batch_size:int=64, shuffle:bool=False, num_workers:int=1) -> DataLoader:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
