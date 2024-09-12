import torch.nn.functional as F
import torch.nn as nn
import os
import torch
import numpy as np


class ImageClassifier(nn.Module):
    # def __init__(self, number_of_classes, in_channels=3, hidden_channels=64, kernel_size=3, stride=1,
    #              padding=1, dropout_prob=0.5) -> None:
    #     super(ImageClassifier, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size,
        #                        stride=stride, padding=padding)
        # self.bn1 = nn.BatchNorm2d(hidden_channels)
        # self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=2 * hidden_channels, kernel_size=kernel_size,
        #                        stride=stride, padding=padding)
        # self.bn2 = nn.BatchNorm2d(2 * hidden_channels)
        # self.pool2 = nn.MaxPool2d(2, 2)
        # self.conv3 = nn.Conv2d(in_channels=2 * hidden_channels, out_channels=3 * hidden_channels,
        #                        kernel_size=kernel_size,
        #                        stride=stride, padding=padding)
        # self.bn3 = nn.BatchNorm2d(3 * hidden_channels)
        # self.conv4 = nn.Conv2d(in_channels=3 * hidden_channels, out_channels=4 * hidden_channels,
        #                        kernel_size=kernel_size,
        #                        stride=stride, padding=padding)
        # self.pool4 = nn.MaxPool2d(2, 2)
        # self.bn4 = nn.BatchNorm2d(4 * hidden_channels)
        #
        # self.conv5 = nn.Conv2d(in_channels=4 * hidden_channels, out_channels=4 * hidden_channels, kernel_size=3,
        #                        padding=1)
        # self.bn5 = nn.BatchNorm2d(4 * hidden_channels)
        # self.pool5 = nn.MaxPool2d(2, 2)
        #
        # self.dropout = nn.Dropout(dropout_prob)
        #
        # self.global_avg_pool = nn.AdaptiveAvgPool2d((8, 8))
        #
        # self.fc1 = nn.Linear(8 * 8 * 4 * hidden_channels, 512)
        # self.bn_fc1 = nn.BatchNorm1d(512)
        #
        # self.fc_dropout = nn.Dropout(dropout_prob)
        #
        # self.fc2 = nn.Linear(512, number_of_classes)


    # def forward(self, x):
    #
    #     if isinstance(x, np.ndarray):
    #         x = torch.from_numpy(x).float()
    #         device = next(self.parameters()).device
    #         x = x.to(device)

        # # Convolutional Block 1
        # x = F.relu(self.bn1(self.conv1(x)))
        # # Convolutional Block 2
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = self.pool2(x)
        # # Convolutional Block 3
        # x = F.relu(self.bn3(self.conv3(x)))
        # # Convolutional Block 4
        # x = F.relu(self.bn4(self.conv4(x)))
        # x = self.pool4(x)
        # # Convolutional Block 5
        # x = F.relu(self.bn5(self.conv5(x)))
        # x = self.pool5(x)
        #
        # x = self.dropout(x)
        #
        # x = self.global_avg_pool(x)
        # # Flattening x
        # x = x.view(x.size(0), -1)
        #
        # x = F.relu(self.bn_fc1(self.fc1(x)))
        # x = self.fc_dropout(x)
        #
        # x = self.fc2(x)

        # Convolutional Block 1

    def __init__(self, num_classes, hidden_channels) -> None:
        super().__init__()
        self.hidden_size = hidden_channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=hidden_channels, out_channels=2*hidden_channels, kernel_size=5, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(2*hidden_channels)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=2*hidden_channels, out_channels=2*hidden_channels, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(2*hidden_channels)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(2*hidden_channels * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.fc_dropout = nn.Dropout(0.3)
        # self.fc2 = nn.Linear(256,128)
        # self.fc3 = nn.Linear(128,100)
        self.sm = nn.LogSoftmax(1)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        # output = self.dropout(output)
        output = self.pool1(output)
        output = F.relu(self.bn2(self.conv2(output)))
        # output = self.dropout(output)
        output = self.pool2(output)
        output = F.relu(self.bn3(self.conv3(output)))
        # output = self.dropout(output)
        # output = self.dropout(output)
        output = output.view(-1, 4 * 4 * 2*self.hidden_size)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        # output = self.fc_dropout(output)
        # output = self.fc2(output)
        # output = self.fc3(output)
        # output = self.sm(output)
        return output
