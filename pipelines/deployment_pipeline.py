from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters

from constants import MIN_ACCURACY
from steps.evaluate_model import evaluate_model
from steps.ingest_data import ingest_data
from steps.train_model import train_model

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
    train_loader, test_loader, num_classes = ingest_data(data_path)
    model, criterion = train_model(train_loader, num_classes)
    test_avg_loss, test_accuracy = evaluate_model(model, criterion, test_loader)
    deployment_decision = deployment_trigger(test_accuracy, min_accuracy)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout
    )
