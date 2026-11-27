from .base import Layer, np


class ReLU(Layer):
    def __init__(self):
        # Cache to store the previous output of the layer.
        # This is needed for backpropagation.
        self.fx = np.array([])

    def forward(self, x: np.ndarray, *, training: bool = True) -> np.ndarray:
        # First, we compute the output of the activation function.
        fx = np.maximum(0, x)

        # If the network is in training mode, we cache this output to prepare
        # for backpropagation to follow after this call.
        if training:
            self.fx = fx

        return fx

    def backward(self, dl_dz: np.ndarray) -> np.ndarray:
        # By expressing its gradient in terms of the original function,
        # we calculate the gradient using the output of the forward pass.
        return dl_dz * np.sign(self.fx)

    # The ReLU layer is an activation function without learnable parameters.
    # Hence, `.parameters()` returns an empty list.
    def parameters(self) -> list[np.ndarray]:
        return []

    def gradients(self) -> list[np.ndarray]:
        return []


class Sigmoid(Layer):
    def __init__(self):
        self.fx = np.array([])

    def forward(self, x: np.ndarray, *, training: bool = True) -> np.ndarray:
        # First, we compute the output of the activation function.
        # For numerical stability, we split the sigmoid into two cases
        # to ensure the values passed to `np.exp` are always negative.
        fx = np.where(x >= 0, 1 / (1 + np.exp(-x)),
                      np.exp(x) / (1 + np.exp(x)))

        # If the network is in training mode, we cache this output to prepare
        # for backpropagation to follow after this call.
        if training:
            self.fx = fx

        return fx

    def backward(self, dl_dz: np.ndarray) -> np.ndarray:
        # By expressing its gradient in terms of the original function,
        # we calculate the gradient using the output of the forward pass.
        return dl_dz * self.fx * (1 - self.fx)

    # The ReLU layer is an activation function without learnable parameters.
    # Hence, `.parameters()` returns an empty list.
    def parameters(self) -> list[np.ndarray]:
        return []

    def gradients(self) -> list[np.ndarray]:
        return []
