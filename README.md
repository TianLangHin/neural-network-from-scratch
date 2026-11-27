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
  The following files contain the datasets used for training the networks
  (where `[partition]` is either `train` or `test`, and `[var]` is either `x` or `y`)
  but are not provided directly in this repository:
  * `cifar10-[partition]-[var].npy` containing the CIFAR-10 dataset.
    The `train` partition contains 50,000 images of dimensions 32x32x3
    while the `test` partition contains another 10,000 images.
  * `fashion_mnist-[partition]-[var].npy` containing the Fashion-MNIST dataset.
    The `train` partition contains 50,000 images of dimensions 28x28 (with 1 channel implicitly)
    while the `test` partition contains another 10,000 images.
  * `mnist-[partition]-[var].npy` containing the MNIST handwritten digit dataset.
    The `train` partition contains 50,000 images of dimensions 28x28 (with 1 channel implicitly)
    while the `test` partition contains another 10,000 images.
  * `tiny-shakespeare.txt` is not downloaded via a script,
    but taken directly from [Andrej Karpathy's char-nn](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt).
* The `loss` folder contains implementations for the loss functions used to train the neural networks.
* The `nn` folder contains implementations for each neural network layer used to compose the architectures.
* The `optimiser` folder contains implementations for the optimisers that improve the neural networks.
