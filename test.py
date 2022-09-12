
from PIL import ImageTk, Image
import utils
from torchvision.utils import save_image


from torchvision import transforms
transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])
image = Image.open("index.jpg")
image = transform(image)

save_image(image, "fdsf.jpg")


