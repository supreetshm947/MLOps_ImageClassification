from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
from torchvision import transforms
from constants import LABEL_FILE_PATH
from pipelines.deployment_pipeline import prediction_service_loader
import uvicorn
import torch

from src.utils import read_classes, get_label_from_prediction

app = FastAPI()
service = prediction_service_loader(
    pipeline_name="continuous_deployment_pipeline",
    pipeline_step_name="mlflow_model_deployer_step",
    running=True,
)

if not service.is_running:
    service.start(timeout=60)

classes = read_classes(LABEL_FILE_PATH)


@app.post("/classify/")
async def classify(image_file: UploadFile = File(...)):
    try:
        image = Image.open(image_file.file)
        image = image.resize((32, 32), Image.Resampling.LANCZOS)
        image.save("img.jpeg")
        transform = transforms.Compose([
                transforms.ToTensor(),  # Convert image to a PyTorch tensor
            ])
        image_tensor = transform(image)
        image_tensor = torch.unsqueeze(image_tensor, 0)
        image_array = image_tensor.numpy()

        # image_array = np.array(image)
        # image_array = np.transpose(image_array, (2, 0, 1))
        # image_array = np.expand_dims(image_array, axis=0)

        # from torchvision import transforms
        # import torch
        # image_path = 'charlie.jpg'
        # image = Image.open(image_path)
        # transform = transforms.Compose([
        #     transforms.Resize((32, 32)),  # Resize to 32x32
        #     transforms.ToTensor(),  # Convert image to a PyTorch tensor
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # ])
        #
        # # Apply the transform to the image
        # image_tensor = transform(image)
        # image_tensor = torch.unsqueeze(image_tensor, 0)
        # image_array = image_tensor.numpy()

        prediction = service.predict(image_array)
        prediction = get_label_from_prediction(prediction, classes)

        return JSONResponse(content={"prediction": prediction[0]})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=9000)
