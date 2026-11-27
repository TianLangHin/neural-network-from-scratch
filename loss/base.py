import numpy as np

from abc import ABC, abstractmethod


# Abstract class that all loss functions must abide by.
class Loss(ABC):
    # The true labels and the predicted labels are required as arguments.
    # If `mean` is True, a single scalar value is returned.
    # If `mean` is False, then it outputs the unaggregated NumPy array,
    # allowing collection of statistics across multiple batches.
    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, *,
                mean: bool = True) -> np.float64 | np.ndarray:
        pass

    # The derivative of the scalar loss (compute where `mean` is True)
    # with respect to all values in `y_pred` is computed and returned.
    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass
