from zenml import step
import torch
from tqdm import tqdm
import torch.nn as nn
from mylogger import get_logger
from src.utils import get_device_type
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from zenml.client import Client
import mlflow
from typing import Tuple
from typing_extensions import Annotated

logger = get_logger()

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def evaluate_model(model: nn.Module, criterion: nn.Module, test_loader: DataLoader) -> Tuple[
    Annotated[float, "test_loss"], Annotated[float, "test_accuracy"]]:
    device = get_device_type()

    total_loss = 0
    all_labels = []
    all_preds = []

    model.eval()

    with torch.no_grad():

        # from PIL import Image
        # from torchvision import transforms
        # image_path = 'charlie.jpg'
        # image = Image.open(image_path)
        # transform = transforms.ToTensor()
        #
        # # Apply the transform to the image
        # image_tensor = transform(image)
        # # images = image_tensor.to(device)
        # images = torch.unsqueeze(image_tensor, 0)
        # images = images.numpy()
        # outputs = model(images)

        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Compute the loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get predictions by choosing the index with the maximum output
            _, preds = torch.max(outputs, 1)

            # Collect labels and predictions for accuracy calculation
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(test_loader)

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # Print classification report for more detailed performance metrics (precision, recall, F1-score)
    # class_report = classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes)
    #
    # Log the metrics
    logger.info(f'Average Loss: {avg_loss:.4f}')
    logger.info(f'Accuracy: {accuracy:.4f}')
    # logger.info("Classification Report:")
    # logger.info(class_report)

    mlflow.log_metric("Average Loss", avg_loss)
    mlflow.log_metric("Accuracy", accuracy)

    return avg_loss, accuracy
