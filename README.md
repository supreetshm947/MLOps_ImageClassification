# 🖼️ Image Classification with PyTorch | MLOps using ZenML
This work implements a Machine Learning (ML) workflow and deployment for an image classification model built using PyTorch and trained on CIFAR-100 dataset. The project implements a complete MLOps workflow with **ZenML** for orchestration, **Streamlit** for UI, **FastAPI** for backend services, and connects to a deployed model on an MLOps server for inference. The goal is to demonstrate how to integrate machine learning pipelines with MLOps tools and serve a deep learning model using a modern full-stack setup.

# 🛠️ Technologies Used
- **PyTorch**: Deep learning framework for building the CNN model for classification.
- **ZenML**: MLOps framework for creating reproducible ML pipelines.
- **MLFlow**: For model logging and tracking (integrated via ZenML).
- **FastAPI**: Backend API service for handling inference requests.
- **Streamlit**: Frontend interface for user interactions.
- **Conda**: Used to manage the project’s virtual environment and dependencies.

## 🎯 Project Overview

The implemented Machine Learning workflow consists of several key stages:

![pipeline](pipeline.png)

1. **Data Ingestion**: Load and preprocess (apply transformations) the CIFAR-100 dataset.
2. **DataLoader**: Fetch loader objects for ingested and processed data.
3. **Model Training**: Train a convolutional neural network (CNN) on the training dataset.
4. **Model Evaluation**: Evaluate the performance of the model on the training dataset.
5. **Model Deployment**: Deploy the model to a ZenML-based MLOps server for serving predictions.
6. **Inference Service**: Build a FastAPI service for communication between the frontend and backend.
7. **Frontend Interface**: Build a Streamlit app to provide a user-friendly interface for image uploads and classification results.


### 🚀 How to Run the Project

1. Clone the repository
   git clone [https://github.com/your_username/image-classification-mlops.git](https://github.com/supreetshm947/MLOps_ImageClassification.git)
   cd MLOps_ImageClassification


