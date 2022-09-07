from torchvision import datasets
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler


def load_data(batch_size, num_workers, pin_memory):
    train = datasets.CIFAR100("data", train=True, download=True)
    test = datasets.CIFAR100("data", train=False)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, sampler=SubsetRandomSampler(
        range(len(train))), num_workers=num_workers, pin_memory=pin_memory)
    
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, sampler=SubsetRandomSampler(
        range(len(train))), num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader

def get_device_type():
    return torch.device("gpu" if torch.cuda.is_available() else "cpu")


    
