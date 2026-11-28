from .activations import ReLU, Sigmoid
from .base import Layer, RandomState, glorot_init, kaiming_init, np
from .linear import Linear


# Generic class for neural networks composed of subclasses of `Layer`.
class NeuralNetwork:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, x: np.ndarray, *, training: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, dl: np.ndarray):
        # The result of the last iteration is ignored,
        # since there is no meaningful layer to pass gradients back to.
        for layer in reversed(self.layers):
            dl = layer.backward(dl)

    # The parameters of the entire neural network are laid out and unpacked
    # in the order of the layers themselves.
    def parameters(self) -> list[np.ndarray]:
        return [
            param
            for layer in self.layers
            for param in layer.parameters()]

    # The order in which the gradients are returned is exactly the same as
    # the corresponding parameters outputted by `.parameters()`,
    # so that they can be zipped and iterated over during optimisation.
    def gradients(self) -> list[np.ndarray]:
        return [
            gradient
            for layer in self.layers
            for gradient in layer.gradients()]
