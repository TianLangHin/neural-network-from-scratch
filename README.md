# Neural Networks From Scratch

This repository implements neural network layers from scratch in NumPy,
as an exploration of the underlying components used in various neural network architectures.

In this repository, a modular structure is used to emphasise the flexibility and commonalities
between the layers of each architecture being explored.

## Directory Structure

The base of the folder contains the training and running scripts of the architectures themselves,
while the inner folders contain the underlying implementations for each component.

* The `architectures` folder contains preset concrete setups for various chosen data learning tasks.
* The `data` folder contains the data used for training and scripts to collect them.
  Some of the data is extracted from `torchvision.datasets`,
  but PyTorch itself is not used anywhere in the implementation of the neural network layers.
* The `loss` folder contains implementations for the loss functions used to train the neural networks.
* The `nn` folder contains implementations for each neural network layer used to compose the architectures.
* The `optimiser` folder contains implementations for the optimisers that improve the neural networks.
