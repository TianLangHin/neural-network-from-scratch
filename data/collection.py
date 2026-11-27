# The `torchvision` module is used here only for downloading standard
# well-known datasets for image classification.
from torchvision import datasets
import numpy as np
import sys


def save_dataset(dataset_name: str):
    # This helper function currently supports CIFAR-10, MNIST, and
    # Fashion-MNIST datasets.
    match dataset_name:
        case 'cifar10':
            dataset = datasets.CIFAR10
        case 'mnist':
            dataset = datasets.MNIST
        case 'fashion_mnist':
            dataset = datasets.FashionMNIST
        case _:
            return

    # First, the training and testing portions of the dataset are downloaded.
    training_set = dataset(root=f'./{dataset_name}', train=True, download=True)
    testing_set = dataset(
        root=f'./{dataset_name}_test',
        train=False,
        download=True)

    # Then, the data features and the labels are converted to NumPy arrays.
    x_train = training_set.data
    y_train = training_set.targets.numpy()
    x_test = testing_set.data
    y_test = testing_set.targets.numpy()

    # Finally, the NumPy arrays are saved in `.npy` files.
    np.save(f'{dataset_name}-train-x', x_train)
    np.save(f'{dataset_name}-train-y', y_train)
    np.save(f'{dataset_name}-test-x', x_test)
    np.save(f'{dataset_name}-test-y', y_test)


if __name__ == '__main__':
    # A single argument (dataset name) is passed via command line to indicate
    # which to download.
    if len(sys.argv) > 1:
        save_dataset(sys.argv[1])
