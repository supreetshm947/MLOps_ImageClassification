from unittest import TestLoader
from torchvision import datasets
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from tensorflow.python.eager import context
from model import Network
from torchvision import transforms
import matplotlib.pyplot as plt
import time 
import datetime
from IPython import display
plt.ion()

def load_data(batch_size, num_workers, pin_memory=False):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train = datasets.CIFAR100("data", train=True, download=True, transform=transform)
    test = datasets.CIFAR100("data", train=False, transform=transform)

   

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, sampler=SubsetRandomSampler(
        range(len(train))), num_workers=num_workers, pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(test, batch_size=len(test), sampler=SubsetRandomSampler(
        range(len(test))), num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader


def get_device_type():
    return torch.device("gpu" if torch.cuda.is_available() else "cpu")


def get_num_gpus():
    return context.num_gpus()


def train_model(train_loader, test_loader, device,model: Network, optimizer, criterion, epochs=10, print_every=10):
    train_losses = []
    test_losses = []
    running_loss = 0
    steps = 0
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            log_output = model.forward(inputs)
            loss = criterion(log_output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # if steps % print_every == 0:
            #     test_loss = 0
            #     accuracy = 0
            #     model.eval()
            #     with torch.no_grad():
            #         for test_inputs, test_labels in test_loader:
            #             test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            #             test_log_output = model.forward(test_inputs)
            #             batch_loss = criterion(test_log_output, test_labels)
            #             test_loss += batch_loss.item()

            #             test_output = torch.exp(test_log_output)
            #             top_out, top_class = test_output.topk(1, dim=1)
            #             equals = top_class == test_labels.view(top_class.shape)
            #             accuracy += torch.mean(equals.type(torch.float)).item()
            #         train_losses.append(running_loss/len(train_loader))
            #         test_losses.append(test_loss/len(test_loader))
            #         print(f"Epoch {epoch+1}/{epochs}.. ",
            #         f"Train loss: {running_loss/print_every:.3f}.. "
            #         f"Test loss: {test_loss/len(test_loader):.3f}.. ",
            #         f"Test accuracy: {accuracy/len(test_loader):.3f}")
            #         running_loss=0
            #         plot(train_losses, test_losses)
            #     model.train()

        test_loss, accuracy = run_validate(model=model, test_loader=test_loader, device=device, criterion=criterion)
        train_losses.append(running_loss/len(train_loader))
        test_losses.append(test_loss/len(test_loader))
        print(f"Epoch {epoch+1}/{epochs}.. ",
        f"Train loss: {running_loss/print_every:.3f}.. "
        f"Test loss: {test_loss/len(test_loader):.3f}.. ",
        f"Test accuracy: {accuracy/len(test_loader):.3f}")
        running_loss=0
        plot(train_losses, test_losses)
        
    model.save_model("model.pth")

def run_validate(model, test_loader, device, criterion):
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_log_output = model.forward(test_inputs)
            batch_loss = criterion(test_log_output, test_labels)
            test_loss += batch_loss.item()

            test_output = torch.exp(test_log_output)
            top_out, top_class = test_output.topk(1, dim=1)
            equals = top_class == test_labels.view(top_class.shape)
            accuracy += torch.mean(equals.type(torch.float)).item()
    model.train()
    return test_loss, accuracy

def train(train_loader, test_loader, model, device, criterion, optimizer, epochs=100,):
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

            print(
            f'Epoch {epoch+1} | train loss: {train_loss:.3f}, train accuracy: {train_accuracy:.3f}, ' + \
            f'test loss: {test_loss:.3f}, test accuracy: {test_accuracy:.3f}, ' + \
            f'time: {str(datetime.timedelta(seconds=int(time.time()-start)))}'
        )
    model.save_model("model.pth")

def plot(train_losses, test_losses):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show(block=False)
    plt.pause(.1)