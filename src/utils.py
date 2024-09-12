from torch.autograd import Variable
from torchvision import datasets
import numpy as np
import torch
# from tensorflow.python.eager import context
# from model import ImageClassifier
from torchvision import transforms
# import matplotlib.pyplot as plt
import time
import datetime
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from constants import BATCH_SIZE


# from IPython import display
# import pickle
# import imageio
# from skimage.transform import resize
# from PIL import Image

# plt.ion()

def load_data(batch_size, num_workers, device_type="cpu", pin_memory=False):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = datasets.CIFAR100("data", train=True, download=True, transform=transform)
    test_data = datasets.CIFAR100("data", train=False, transform=transform)

    classes = len(train_data.classes)

    if device_type == "gpu":
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                                                   pin_memory=pin_memory)

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers,
                                                  pin_memory=pin_memory)
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers)

    return train_loader, test_loader, classes


def get_device_type():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_num_gpus(device):
    if device == "cuda":
        return torch.cuda.device_count()
    return 1

def read_classes(class_file_path):
    with open(class_file_path, "r") as file:
        return file.readline().split(" ")

def get_label_from_prediction(prediction, classes):
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    indices = np.argmax(prediction, axis=1)
    predicted_labels = [classes[idx] for idx in indices]
    return predicted_labels

# def train(train_loader, test_loader, model, device, criterion, optimizer, epochs=100, plot_loss=False):
#     train_losses = []
#     test_losses = []
#     test_accuracies = []
#     for epoch in range(epochs):
#         start = time.time()
#         loss_sum = 0
#         correct = 0
#         sample_count = 0
#         n_batches = len(train_loader)
#         model.train()
#         for i, (input, label) in enumerate(train_loader):
#             input = input.to(device)
#             label = label.to(device)
#             out = model(input)
#             loss = criterion(out, label)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#
#             loss_sum += loss.item() * input.shape[0]
#             _, pred = torch.max(out, 1)
#             correct += (pred == label).sum().item()
#             sample_count += input.shape[0]
#             print(f"Train epoch {epoch + 1}, step{i + 1}/{n_batches}", end="    \r")
#
#         train_loss = loss_sum / sample_count
#         train_accuracy = correct / sample_count
#         train_losses.append(train_loss)
#         with torch.no_grad():
#             n_batches = len(test_loader)
#             loss_sum = 0
#             model.eval()
#             for i, (input, label) in enumerate(test_loader):
#                 input = input.to(device)
#                 label = label.to(device)
#
#                 out = model(input)
#                 loss = criterion(out, label)
#                 loss_sum += loss.item() * input.shape[0]
#                 _, pred = torch.max(out, 1)
#                 correct += (pred == label).sum().item()
#                 sample_count += input.shape[0]
#                 print(f"Test epoch {epoch + 1}, step{i + 1}/{n_batches}", end="    \r")
#
#             test_loss = loss_sum / sample_count
#             test_accuracy = correct / sample_count
#             test_losses.append(test_loss)
#             test_accuracies.append(test_accuracy)
#             print(
#                 f'Epoch {epoch + 1} | train loss: {train_loss:.3f}, train accuracy: {train_accuracy:.3f}, ' + \
#                 f'test loss: {test_loss:.3f}, test accuracy: {test_accuracy:.3f}, ' + \
#                 f'time: {str(datetime.timedelta(seconds=int(time.time() - start)))}'
#             )
#     if plot:
#         plot(train_losses=train_losses, test_losses=test_losses, test_accuracy=test_accuracies)
# saved_model.save_model("saved_model.pth")

# def plot(train_losses, test_losses, test_accuracy):
#     display.clear_output(wait=True)
#     display.display(plt.gcf())
#     plt.clf()
#     plt.plot(train_losses, label='Training loss')
#     plt.plot(test_losses, label='Validation loss')
#     plt.plot(test_accuracy, label='Validation Accuracy')
#     plt.legend(frameon=False)
#     plt.show(block=False)
#     plt.pause(.1)
#
# def get_labels():
#     with open("data/cifar-100-python/meta", 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return [x.decode('utf-8') for x in list(dict.values())[0]]
#
# def load_model(device, filename="saved_model.pth"):
#     saved_model = ImageClassifier()
#     saved_model.load_state_dict(torch.load("saved_model/"+filename, map_location=device))
#     return saved_model
#
# def load_image(path):
#     return imageio.imread(path)
#
# def resize_image(image):
#     return torch.tensor(resize(image, (32,32), anti_aliasing=True)).permute(2,0,1).type(torch.FloatTensor)[None,:,:]
#
# def image_loader(image_path, device):
#     transform = transforms.Compose([
#         transforms.Resize((32,32)),
#         transforms.ToTensor()
#     ])
#     image = Image.open(image_path)
#     image = transform(image).float()
#     image = Variable(image, requires_grad=True)
#     image = image.unsqueeze(0)
#     return image.to(device)
