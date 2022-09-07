import torch.nn.functional as F
import torch.nn as nn
import os
import torch

class Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.fc = nn.Linear(24*10*10,10)
    
    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn3(self.conv3(output)))
        output = F.relu(self.bn4(self.conv4(output)))
        output = output.view(-1, 10*10*24)
        output = self.fc(output)
    
    def save_model(self, filename="model.pth"):
        dir = "./model"
        if os.path.exists(dir):
            os.makedirs(dir)
        filename = os.path.join(dir, filename)
        torch.save(self.state_dict, filename)
