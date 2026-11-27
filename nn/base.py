import numpy as np

from abc import ABC, abstractmethod


# Every layer that can be used to construct a neural network is written as
# a class that abides by the `Layer` abstract class.
# These layers must support forward computation, backpropagation, and
# methods to return the learnable parameters (and their respective gradients)
# so that they can be updated by an external optimiser.
class Layer(ABC):
    # The forward pass requires an input NumPy array, as well as
    # an explicit indicator of whether this is during training or inference.
    @abstractmethod
    def forward(self, x: np.ndarray, *, training: bool = True) -> np.ndarray:
        pass

    # The backward pass sets the internal state to record the gradients with
    # respect to all its learnable parameters. It requires the loss w.r.t.
    # its previous output as an argument, and outputs the loss w.r.t. the
    # previous input (`x` in `forward`) to support backpropagation.
    @abstractmethod
    def backward(self, dl: np.ndarray) -> np.ndarray:
        pass

    # Returns a list of all the learnable parameters of the layer.
    # References to the actual attributes (not copies) are returned so that
    # an external optimiser can modify the values returned in-place.
    @abstractmethod
    def parameters(self) -> list[np.ndarray]:
        pass

    # Returns a list of the gradients of all the learnable parameters.
    # Again, references to the actual gradients being stored are returned.
    @abstractmethod
    def gradients(self) -> list[np.ndarray]:
        pass


type RandomState = int | np.random.Generator


# Returns a matrix randomly initialised using Kaiming initialisation.
# Either an existing RNG or a seed for a new temporary RNG can be used.
def kaiming_init(fan_in: int, shape: list[int],
                 *, random_state: RandomState) -> np.ndarray:
    if isinstance(random_state, int):
        rng = np.random.Generator(np.random.MT19937(random_state))
    else:
        rng = random_state
    return rng.normal(0, np.sqrt(2 / fan_in), size=shape)


# Returns a matrix randomly initialised using Glorot initialisation.
# Either an existing RNG or a seed for a new temporary RNG can be used.
def glorot_init(fan_in: int, fan_out: int, shape: list[int],
                *, random_state: RandomState) -> np.ndarray:
    if isinstance(random_state, int):
        rng = np.random.Generator(np.random.MT19937(random_state))
    else:
        rng = random_state
    bound = np.sqrt(6 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, size=shape)
