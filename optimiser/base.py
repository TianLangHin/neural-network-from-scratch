import numpy as np

from abc import ABC, abstractmethod


# Any optimiser must abide by this abstract class.
class Optimiser(ABC):
    # Upon initialisation, the optimiser must be given access to
    # the learnable parameters and the gradients of these parameters.
    @abstractmethod
    def __init__(self, parameters: list[np.ndarray],
                 gradients: list[np.ndarray]):
        pass

    # Conducts a single optimisation step, where the parameters are updated.
    @abstractmethod
    def step(self):
        pass

    # Sets the learning rate of the optimiser,
    # and can be called at any point during the training.
    @abstractmethod
    def set_lr(self, lr: float):
        pass
