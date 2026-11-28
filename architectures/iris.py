import numpy as np

from nn import Linear, NeuralNetwork, ReLU


def iris_nn(seed: int) -> NeuralNetwork:
    rng = np.random.Generator(np.random.MT19937(seed))

    # A sample neural network with two hidden layers.
    # The final layer has no softmax since the loss function
    # will already account for it (for numerical stability).
    return NeuralNetwork([
        Linear(4, 100, random_state=rng),
        ReLU(),
        Linear(100, 50, random_state=rng),
        ReLU(),
        Linear(50, 3, random_state=rng),
    ])
