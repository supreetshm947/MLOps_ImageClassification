from torchvision import transforms, datasets
import os
from constants import BATCH_SIZE
from mylogger import get_logger
from torch.utils.data import DataLoader

logger = get_logger()


class ImageIngester:
    def __init__(self, root: str, num_workers=1):
        self.path = root
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def get_data(self):
        try:
            logger.info(f"Ingesting data from '{self.path}'")

            train_path = os.path.join(self.path, "train")
            test_path = os.path.join(self.path, "test")

            if not os.path.exists(train_path) or not os.path.exists(test_path):
                raise Exception(f"Check the data path, train or test directory doesn't exist at {self.path}")

            train_data = datasets.ImageFolder(root=train_path, transform=self.transform)
            test_data = datasets.ImageFolder(root=test_path, transform=self.transform)

            num_classes = len(train_data.classes)

            train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=self.num_workers)
            test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=self.num_workers)

            return train_loader, test_loader, num_classes
            # return num_classes, num_classes
        except Exception as e:
            logger.error(e)
            raise e
