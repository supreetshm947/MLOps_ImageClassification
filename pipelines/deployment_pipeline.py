import torch
from fontTools.ttx import process
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from constants import MIN_ACCURACY, TEST_DATA_ROOT, BATCH_SIZE, LABEL_FILE_PATH, TRAIN_VAL_SPLIT_RATIO
from src.utils import read_classes, get_label_from_prediction
from steps.data_splitter import split_data
from steps.evaluate_model import evaluate_model
from steps.ingest_data import ingest_data
from steps.process_data import process_data
from steps.train_model import train_model
from torch.utils.data import DataLoader
import numpy as np

docker_settings = DockerSettings(required_integrations=[MLFLOW])


class DeploymentTriggerConfig(BaseParameters):
    min_accuracy: float = MIN_ACCURACY


@step
def deployment_trigger(test_accuracy: float, min_accuracy: float) -> bool:
    return test_accuracy > min_accuracy


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
        data_path: str,
        min_accuracy: float = 0,
        workers: int = 1,
        timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    train_dataset, num_classes = ingest_data(data_path)
    # train_dataset, val_dataset = split_data(train_dataset, TRAIN_VAL_SPLIT_RATIO)
    train_loader = process_data(train_dataset, BATCH_SIZE, True)
    # val_loader = process_data(val_dataset, BATCH_SIZE, True)
    model, criterion = train_model(train_loader, num_classes)
    test_dataset, _ = ingest_data(TEST_DATA_ROOT)
    test_loader = process_data(test_dataset, BATCH_SIZE)
    test_avg_loss, test_accuracy = evaluate_model(model, criterion, test_loader)
    deployment_decision = deployment_trigger(test_accuracy, min_accuracy)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout
    )


@step(enable_cache=False)
def prediction_service_loader(
        pipeline_name: str,
        pipeline_step_name: str,
        running: bool = True,
        model_name: str = "model"):
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    service = existing_services[0]
    if not service.is_running:
        service.start(timeout=60)

    return service


@step
def predictor(service: MLFlowDeploymentService,
              loader: DataLoader) -> np.ndarray:
    image, _ = next(iter(loader))
    image = torch.unsqueeze(image[0],0)
    image = image.numpy()
    prediction = service.predict(image)
    return prediction




@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    test_loader, num_classes = ingest_data(TEST_DATA_ROOT, BATCH_SIZE)
    service = prediction_service_loader(pipeline_name, pipeline_step_name)
    predictor(service, test_loader)