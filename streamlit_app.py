import streamlit as st
from PIL import Image

from constants import CLASSIFICATION_API_PORT, TEST_DATA_ROOT
from pipelines.deployment_pipeline import prediction_service_loader
import numpy as np
import requests
from torchvision import transforms
import torch
from io import BytesIO

from src.data_ingester import ImageFolderIngestor

st.title("Image Classification")

# Upload image widget
uploaded_image = st.file_uploader("Upload your image:", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    image = np.array(image)
    # Classify button
    if st.button("CLASSIFY!"):

        with st.spinner("Classifying..."):
            files = {"image_file": uploaded_image.getvalue()}  # Send image as bytes


            # image_ingestor = ImageIngester(TEST_DATA_ROOT, 64)
            # loader, num_classes = image_ingestor.get_data()
            # image, _ = next(iter(loader))
            # image = image[0]
            # to_pil = transforms.ToPILImage()
            # image_pil = to_pil(image)
            # buffer = BytesIO()
            # image_pil.save(buffer, format="PNG")
            # buffer.seek(0)
            # files = {'image_file': buffer.getvalue()}


            response = requests.post(f"http://localhost:{CLASSIFICATION_API_PORT}/classify/", files=files)

        if response.status_code == 200:
            # Display the result
            result = response.json().get("prediction", "No result")
            st.write("## Results")
            st.write(result)
        else:
            st.error(f"Error: {response.content}")
