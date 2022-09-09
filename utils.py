from unittest import TestLoader
from torchvision import datasets
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from tensorflow.python.eager import context
from model import Network
from torchvision import transforms


def load_data(batch_size, num_workers, pin_memory=False):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train = datasets.CIFAR100("data", train=True, download=True, transform=transform)
    test = datasets.CIFAR100("data", train=False, transform=transform)

   

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, sampler=SubsetRandomSampler(
        range(len(train))), num_workers=num_workers, pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, sampler=SubsetRandomSampler(
        range(len(train))), num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader


def get_device_type():
    return torch.device("gpu" if torch.cuda.is_available() else "cpu")


def get_num_gpus():
    return context.num_gpus()


def trainandplot(train_loader, test_loader, device,model: Network, optimizer, criterion, epochs=1, print_every=10):
    train_losses = []
    test_losses = []
    running_loss = 0
    steps = 0
    for i in range(epochs):
        print("nigga im here")
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        #todo
                model.train()
    model.save_model("model.pth")

