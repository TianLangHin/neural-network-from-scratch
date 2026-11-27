from .base import Loss, np


class BinaryCrossEntropy(Loss):
    def __init__(self, *, eps: float = 1e-8):
        # Used for numerical stability.
        self.eps = eps

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, *,
                mean: bool = True) -> np.float64 | np.ndarray:
        # The addition of `eps` to the argument of each log is for numerical
        # stability when `y_pred` is near 0 or 1.
        pos_term = y_true * np.log(y_pred + self.eps)
        neg_term = (1 - y_true) * np.log(1 - y_pred + self.eps)

        # The binary cross-entropy is the negative of the sum of both.
        loss = -(pos_term + neg_term)

        return loss.mean() if mean else loss

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # We scale by the number of data points
        # to return the gradient of the mean loss.
        n = y_pred.shape[0]

        # The formula for the gradient actively considers the inclusion of
        # `eps` as well.
        pos_term = y_true / (y_pred + self.eps)
        neg_term = (1 - y_true) / (1 - y_pred + self.eps)

        return -(pos_term - neg_term) / n
