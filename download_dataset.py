from torchvision import datasets
import os
from PIL import Image
from mylogger import get_logger

from constants import DATASET_DIR

logger = get_logger()


def save_torchvision_as_image(data, dataset_dir, classes):
    for idx, (image, label) in enumerate(data):
        class_name = classes[label]
        class_dir = os.path.join(dataset_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        image_path = os.path.join(class_dir, f"{idx}.png")
        image.save(image_path)


def download_cifar100():
    logger.info("Downloading cifar100.")
    train = datasets.CIFAR100(DATASET_DIR, train=True, download=True)  # Download the training dataset
    test = datasets.CIFAR100(DATASET_DIR, train=False, download=True)  # Download the test dataset

    train_dir = os.path.join(DATASET_DIR, "train")
    test_dir = os.path.join(DATASET_DIR, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    classes = train.classes

    classes_file = "classes.txt"
    with open(classes_file, 'w') as file:
        file.write(" ".join(classes))

    save_torchvision_as_image(train, train_dir, classes)
    save_torchvision_as_image(test, test_dir, classes)

    logger.info("Converted cifar100 to generic format.")


if __name__ == "__main__":
    download_cifar100()
