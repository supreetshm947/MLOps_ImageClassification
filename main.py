import utils
import torch.nn as nn
from torch import optim
from model import Network

dev_type = utils.get_device_type()
num_gpu = utils.get_num_gpus()
train_loader, test_loader = utils.load_data(200, num_gpu*4)

model = Network()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr = 0.03)

utils.trainandplot(train_loader=train_loader, test_loader=test_loader, device=dev_type, optimizer=optimizer, criterion=criterion, model=model)