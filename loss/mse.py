from .base import Loss, np


class MSE(Loss):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, *,
                mean: bool = True) -> np.float64 | np.ndarray:
        loss = (y_pred - y_true) * (y_pred - y_true)
        return loss.mean() if mean else loss

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n = y_pred.shape[0]
        return 2 * (y_pred - y_true) / n
