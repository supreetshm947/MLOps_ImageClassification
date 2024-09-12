from this import s
import numpy as np
from src.data_ingester import ImageFolderIngestor
from src.model import ImageClassifier
from src.utils import get_device_type
from torch import nn
from torch import optim
from tqdm import tqdm
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from sklearn.metrics import accuracy_score

def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    ])
    # train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    image_ingestor = ImageFolderIngestor("data/train")
    full_dataset, num_classes = image_ingestor.get_data()

    targets = np.array(full_dataset.targets)

    # Set up stratified splitting
    train_size = 0.95  # 95% for training
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.95, test_size=0.05, random_state=42)

    # Perform the stratified split
    train_indices, val_indices = next(splitter.split(np.zeros(len(targets)), targets))

    # Create the train and validation datasets using Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    device = get_device_type()
    model = ImageClassifier(100, 24)
    model = model.to(device)
    # mlflow.pytorch.autolog()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    example_input = None
    example_output = None

    train_losses = []
    for epoch in range(10):
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
        print(f"Epoch {epoch + 1}, Loss: {train_loss_sum}, Accuracy: {train_accuracy}")

        model.eval()
        with torch.no_grad():
            val_loss_sum = 0
            val_labels = []
            val_pred = []
            for images, labels in tqdm(val_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * images.size(0)

                _, pred = torch.max(outputs, 1)

                val_labels.append(labels.cpu().numpy())
                val_pred.append(pred.cpu().numpy())

            val_labels = np.concatenate(val_labels)
            val_pred = np.concatenate(val_pred)

            val_loss = val_loss_sum / len(val_loader.dataset)
            val_accuracy = accuracy_score(val_labels, val_pred)
            print(f"Validation Loss: {val_loss}, Accuracy: {val_accuracy}")

        scheduler.step()

    print(f"Total Average Loss: {sum(train_losses) / len(train_losses)}")

        # scheduler.step()
        #
        # train_loss = loss_sum / sample_count
        # train_accuracy = correct / sample_count
        # train_losses.append(train_loss)
        # print(f"Epoch {epoch + 1}, Loss: {train_loss}, Accuracy: {train_accuracy}")

if __name__ == "__main__":
    train()