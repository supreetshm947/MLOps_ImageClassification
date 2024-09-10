from zenml import step
from .config import ModelConfig
import torch.nn as nn
from torch import optim
import torch
from torch.utils.data import DataLoader
from mylogger import get_logger
from src.model import ImageClassifier
from src.utils import get_device_type
from tqdm import tqdm
from typing import Tuple
from typing_extensions import Annotated
from zenml.client import Client
import mlflow

logger = get_logger()

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def train_model(train_loader: DataLoader, num_classes: int, config: ModelConfig) -> Tuple[
    Annotated[nn.Module, "model"],
    Annotated[nn.Module, "criterion"],
]:
    device = get_device_type()
    model = ImageClassifier(num_classes, hidden_channels=config.hidden_size)
    model = model.to(device)
    mlflow.pytorch.autolog()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    train_losses = []
    for epoch in range(config.epochs):
        loss_sum = 0
        correct = 0
        sample_count = 0
        model.train()
        for image, label in tqdm(train_loader):
            image = image.to(device)
            label = label.to(device)
            out = model(image)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * image.shape[0]
            _, pred = torch.max(out, 1)
            correct += (pred == label).sum().item()
            sample_count += image.shape[0]
            break

        train_loss = loss_sum / sample_count
        train_accuracy = correct / sample_count
        train_losses.append(train_loss)
        logger.info(f"Epoch {epoch + 1}, Loss: {train_loss}, Accuracy: {train_accuracy}")
        mlflow.log_metric(f"Training Loss {epoch + 1}", train_loss)
        mlflow.log_metric(f"Training Accuracy {epoch + 1}", train_accuracy)

    logger.info(f"Total Average Loss: {sum(train_losses) / len(train_losses)}")
    # only uncomment if cant get the model logged automatically
    #mlflow.pytorch.log_model(model, artifact_path="model")

    return model, criterion
