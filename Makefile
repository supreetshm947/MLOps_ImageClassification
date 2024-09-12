install_mlflow:
	@zenml integration install mlflow -y
	@echo Installed mlflow

install_pyenv:
	@echo Installing pyenv...
	curl https://pyenv.run | bash
	@echo "export PYENV_ROOT=\"$${HOME}/.pyenv\"" >> ~/.bashrc
	@echo '[[ -d $${PYENV_ROOT}/bin ]] && export PATH="$${PYENV_ROOT}/bin:$${PATH}"' >> ~/.bashrc
	@echo 'eval "$$(pyenv init -)"' >> ~/.bashrc
	@echo 'eval "$$(pyenv virtualenv-init -)"' >> ~/.bashrc
	@echo "Pyenv installed. Please run 'exec bash' to reload your shell or restart your terminal."

install_dependencies:
	@echo Installing dependencies...
	sudo apt update
	sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
	libreadline-dev libsqlite3-dev wget llvm libncurses5-dev \
	libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl

setup: install_dependencies

register_experiment_tracker:
	@zenml experiment-tracker register mlflow_tracker_image_classification --flavor=mlflow
	@echo Registered experiment-tracker

register_deployer:
	@zenml model-deployer register mlflow --flavor=mlflow
	@echo Registered register_deployer

register_stack:
	@zenml stack register stack_image_classification -a default -o default -d mlflow -e mlflow_tracker_image_classification --set
	@echo Registered ZenML stack

create: register_experiment_tracker register_deployer register_stack
	@echo Setup complete


start_zenml:
	@echo Starting ZenML
	@zenml up --blocking

start: start_zenml

stack_list:
	zenml stack list

list: stack_list
