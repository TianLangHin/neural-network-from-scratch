import numpy as np

from nn import Linear, NeuralNetwork, ReLU, Sigmoid


def wisconsin_nn(seed: int) -> NeuralNetwork:
    rng = np.random.Generator(np.random.MT19937(seed))

    # A sample neural network with three hidden layers.
    return NeuralNetwork([
        Linear(30, 100, random_state=rng),
        ReLU(),
        Linear(100, 50, random_state=rng),
        ReLU(),
        Linear(50, 10, random_state=rng),
        ReLU(),
        Linear(10, 1, random_state=rng),
        Sigmoid()
    ])
