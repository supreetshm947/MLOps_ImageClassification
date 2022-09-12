from torch.autograd import Variable
from torchvision import datasets
import numpy as np
import torch
from tensorflow.python.eager import context
from model import Network
from torchvision import transforms
import matplotlib.pyplot as plt
import time 
import datetime
from IPython import display
import pickle 
import imageio
from skimage.transform import resize
from PIL import Image

plt.ion()

def load_data(batch_size, num_workers, pin_memory=False):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train = datasets.CIFAR100("data", train=True, download=True, transform=transform)
    test = datasets.CIFAR100("data", train=False, transform=transform)

    # train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, sampler=SubsetRandomSampler(
    #     range(len(train))), num_workers=num_workers, pin_memory=pin_memory)

    # test_loader = torch.utils.data.DataLoader(test, batch_size=len(test), sampler=SubsetRandomSampler(
    #     range(len(test))), num_workers=num_workers, pin_memory=pin_memory)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    


    return train_loader, test_loader


def get_device_type():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_num_gpus():
    return context.num_gpus()

def train(train_loader, test_loader, model, device, criterion, optimizer, epochs=100, plot_loss =False):
    train_losses = []
    test_losses = []
    test_accuracies = []
    for epoch in range(epochs):
        start = time.time()
        loss_sum = 0
        correct = 0
        sample_count = 0
        n_batches = len(train_loader)
        model.train()
        for i, (input, label) in enumerate(train_loader):
            input = input.to(device)
            label = label.to(device)
            out = model(input)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_sum += loss.item() * input.shape[0]
            _, pred = torch.max(out, 1)
            correct += (pred == label).sum().item()
            sample_count += input.shape[0]
            print(f"Train epoch {epoch+1}, step{i+1}/{n_batches}", end="    \r") 
        
        train_loss = loss_sum/sample_count
        train_accuracy = correct/sample_count
        train_losses.append(train_loss)
        with torch.no_grad():
            n_batches = len(test_loader)
            loss_sum = 0
            model.eval()
            for i, (input, label) in enumerate(test_loader):
                input = input.to(device)
                label = label.to(device)

                out = model(input)
                loss = criterion(out, label)
                loss_sum += loss.item() * input.shape[0]
                _, pred = torch.max(out, 1)
                correct += (pred == label).sum().item()
                sample_count += input.shape[0]
                print(f"Test epoch {epoch+1}, step{i+1}/{n_batches}", end="    \r") 
            
            test_loss = loss_sum/sample_count
            test_accuracy = correct/sample_count
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            print(
            f'Epoch {epoch+1} | train loss: {train_loss:.3f}, train accuracy: {train_accuracy:.3f}, ' + \
            f'test loss: {test_loss:.3f}, test accuracy: {test_accuracy:.3f}, ' + \
            f'time: {str(datetime.timedelta(seconds=int(time.time()-start)))}'
        )
        if plot:
            plot(train_losses=train_losses, test_losses=test_losses, test_accuracy=test_accuracies)
    model.save_model("model.pth")

def plot(train_losses, test_losses, test_accuracy):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.plot(test_accuracy, label='Validation Accuracy')
    plt.legend(frameon=False)
    plt.show(block=False)
    plt.pause(.1)

def get_labels():
    with open("data/cifar-100-python/meta", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return [x.decode('utf-8') for x in list(dict.values())[0]]

def load_model(device, filename="model.pth"):
    model = Network()
    model.load_state_dict(torch.load("model/"+filename, map_location=device))
    return model

def load_image(path):
    return imageio.imread(path)

def resize_image(image):
    return torch.tensor(resize(image, (32,32), anti_aliasing=True)).permute(2,0,1).type(torch.FloatTensor)[None,:,:]

def image_loader(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    image = transform(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image.to(device)
