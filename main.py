import utils
import torch.nn as nn
from torch import optim
from model import Network
import ssl
import torch

TRAINING = False

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == '__main__':
    dev_type = utils.get_device_type()
    if TRAINING:
        num_gpu = utils.get_num_gpus()
        train_loader, test_loader = utils.load_data(128, num_gpu*4)

        model = Network()
        model = model.to(dev_type)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr = 1e-2)

        utils.train(train_loader=train_loader, test_loader=test_loader, device=dev_type, optimizer=optimizer, criterion=criterion, model=model, plot_loss=True)
    else:
        model = utils.load_model(dev_type)
        labels = utils.get_labels()
        image = utils.image_loader("tele.jpg", dev_type)
        #out = torch.exp(model(image))
        out = model(image)
        out_prob = torch.max(out).item()/torch.sum(out).item()
        cl_exp = out.argmax(dim=1)
        cl = model(image).argmax(dim=1)
        print(cl, cl_exp, out_prob)
        print(labels[cl])
        
