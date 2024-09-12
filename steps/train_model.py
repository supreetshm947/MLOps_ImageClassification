import numpy as np
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
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score

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
    # mlflow.pytorch.autolog()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    example_input = None
    example_output = None

    train_losses = []
    for epoch in range(config.epochs):
        train_loss_sum = 0
        train_labels = []
        train_pred = []
        model.train()
        for image, label in tqdm(train_loader):
            image = image.to(device)
            label = label.to(device)
            out = model(image)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * image.size(0)
            _, pred = torch.max(out, 1)

            train_labels.append(label.cpu().numpy())
            train_pred.append(pred.cpu().numpy())

            if example_input is None or example_output is None:
                example_input = image
                example_output = out

        train_labels = np.concatenate(train_labels)
        train_pred = np.concatenate(train_pred)

        train_loss_sum = train_loss_sum / len(train_loader.dataset)
        train_accuracy = accuracy_score(train_labels, train_pred)
        train_losses.append(train_loss_sum)
        logger.info(f"Epoch {epoch + 1}, Loss: {train_loss_sum}, Accuracy: {train_accuracy}")
        mlflow.log_metric(f"Training Loss {epoch + 1}", train_loss_sum)
        mlflow.log_metric(f"Training Accuracy {epoch + 1}", train_accuracy)

    logger.info(f"Total Average Loss: {sum(train_losses) / len(train_losses)}")
    # only uncomment if cant get the model logged automatically

    example_input_np = example_input.detach().cpu().numpy()
    example_output_np = example_output.detach().cpu().numpy()

    signature = infer_signature(example_input_np, example_output_np)
    mlflow.pytorch.log_model(model, artifact_path="model", pip_requirements="requirements.txt" \
                             , signature=signature, input_example=example_input_np)

    return model, criterion
