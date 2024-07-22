from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class AbstractConfig(ABC):
    """Abstract class for model configuration.
    """
    input_shape:tuple  	# Dimension of input feature (data format: channel last)
    # batch_size:int = 256
    # epochs:int = 100
    # training_steps:int = 1000

    @classmethod
    def from_dict(cls, obj: dict):
        return cls(**obj)

    # @abstractmethod
    # def optimizer(self):
    #     pass
