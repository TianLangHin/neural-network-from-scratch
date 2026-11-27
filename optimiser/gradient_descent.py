from .base import Optimiser, np


# Conducts naive gradient descent.
class GradientDescent(Optimiser):
    def __init__(self, parameters: list[np.ndarray],
                 gradients: list[np.ndarray], *, lr: float = 0.01):
        self.parameters = parameters
        self.gradients = gradients
        self.lr = lr

    # After the backward step in the neural network is called,
    # every parameter is updated with its corresponding gradient
    # by subtracting the learning rate by its gradient value.
    def step(self):
        for theta_t, g_t in zip(self.parameters, self.gradients):
            theta_t -= self.lr * g_t

    def set_lr(self, lr: float):
        self.lr = lr
