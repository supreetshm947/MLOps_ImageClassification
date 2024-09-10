install_mlflow:
	@zenml integration install mlflow -y
	@echo Installed mlflow

register_experiment_tracker:
	@zenml experiment-tracker register mlflow_tracker_image_classification --flavor=mlflow
	@echo Registered experiment-tracker

register_deployer:
	@zenml model-deployer register mlflow --flavor=mlflow
	@echo Registered register_deployer

register_stack:
	@zenml stack register stack_image_classification -a default -o default -d mlflow -e mlflow_tracker_image_classification --set
	@echo Registered ZenML stack

setup: install_mlflow register_experiment_tracker register_deployer register_stack
	@echo Setup complete

start_zenml:
	@echo Starting ZenML
	@zenml up --blocking

start: start_zenml

stack_list:
	zenml stack list

list: stack_list
