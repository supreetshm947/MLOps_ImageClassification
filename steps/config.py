from zenml.steps import BaseParameters

from constants import LEARNING_RATE, EPOCHS, HIDDEN_SIZE


class ModelConfig(BaseParameters):
    hidden_size: int = HIDDEN_SIZE
    learning_rate: float = LEARNING_RATE
    epochs: int = EPOCHS
