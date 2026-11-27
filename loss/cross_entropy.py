from .base import Loss, np


# Calculates cross-entropy from raw logits,
# accounting for the softmax calculation itself
# and thus requiring the softmax activation function to not be included in
# the last layer of the network.
class CrossEntropyWithSoftmax(Loss):

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, *,
                mean: bool = True) -> np.float64 | np.ndarray:
        # The inner product component uses the raw logits first.
        inner_product = -(y_true * y_pred).sum(axis=1, keepdims=True)

        # For the log-sum-exp component, the maximum element is selected
        # and then used to subtract from all elements for numerical stability.
        c = y_pred.max(axis=1, keepdims=True)
        logsumexp = np.log(np.exp(y_pred - c).sum(axis=1, keepdims=True)) + c

        loss = inner_product + logsumexp

        return loss.mean() if mean else loss

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # The maximum element is selected and then used to subtract from all
        # elements for numerical stability.
        c = y_pred.max(axis=1, keepdims=True)
        s = np.exp(y_pred - c) / np.exp(y_pred - c).sum(axis=1, keepdims=True)

        # We scale by the number of data points
        # to return the gradient of the mean loss.
        n = y_pred.shape[0]

        return (s - y_true) / n
