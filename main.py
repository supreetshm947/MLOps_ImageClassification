import utils
import torch.nn as nn
from torch import optim
from model import ImageClassifier
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == '__main__':
    dev_type = utils.get_device_type()
    
    num_gpu = utils.get_num_gpus(dev_type)
    train_loader, test_loader, classes = utils.load_data(128, num_gpu*4)

    model = ImageClassifier(classes)
    model = model.to(dev_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-2)

    utils.train(train_loader=train_loader, test_loader=test_loader, device=dev_type, optimizer=optimizer, criterion=criterion, model=model, plot_loss=True)
        
