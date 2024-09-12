from torchvision import transforms, datasets
import os
from mylogger import get_logger
from torch.utils.data import DataLoader

logger = get_logger()


class ImageFolderIngestor:
    def __init__(self, root: str):
        self.path = root
        # self.num_workers = num_workers
        # self.batch_size = batch_size
        # self.shuffle = shuffle
        self.transform = transforms.Compose([
            # transforms.Resize((32, 32)),  # Resize to 32x32
            transforms.ToTensor(),  # Convert image to a PyTorch tensor
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])

    def get_data(self):
        try:
            logger.info(f"Ingesting data from '{self.path}'")

            if not os.path.exists(self.path):
                raise Exception(f"Check the data path {self.path}")

            dataset = datasets.ImageFolder(root=self.path, transform=self.transform)
            # loader = DataLoader(data, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

            num_classes = len(dataset.classes)

            return dataset, num_classes
        except Exception as e:
            logger.error(e)
            raise e
