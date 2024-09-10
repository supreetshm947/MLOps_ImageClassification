
# from PIL import ImageTk, Image
# import utils
# from torchvision.utils import save_image
#
#
# from torchvision import transforms
# transform = transforms.Compose([
#         transforms.Resize((32,32)),
#         transforms.ToTensor()
#     ])
# image = Image.open("index.jpg")
# image = transform(image)
#
# save_image(image, "fdsf.jpg")

import utils

dev_type = utils.get_device_type()

num_gpu = utils.get_num_gpus(dev_type)
train_loader, test_loader = utils.load_data(128, num_gpu * 4)
