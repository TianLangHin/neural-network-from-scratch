from .base import Layer, RandomState, kaiming_init, np


class Linear(Layer):

    def __init__(self, in_features: int, out_features: int, *,
                 random_state: RandomState, bias: bool = True):
        # Learnable parameters (weight matrix and bias vector).
        self.w = kaiming_init(
            in_features,
            [in_features, out_features],
            random_state=random_state)
        self.b = np.zeros([1, out_features])

        # Loss gradients for the learnable parameters, initialised to zero.
        self.dl_dw = np.zeros([in_features, out_features])
        self.dl_db = np.zeros([1, out_features])

        # Input data cache for backpropagation calculation.
        self.x = np.array([])

        # Whether or not this layer will attempt to learn its bias vector.
        self.use_bias = bias

    # The input for the forward computation has dimensions `[b, m]`,
    # where `b` is the batch size and `m` is the number of features.
    # The weight matrix has dimensions `[m, n]` where `n` is the number
    # of output features (i.e., number of neurons).
    # The output has dimensions `[b, n]`.
    def forward(self, x: np.ndarray, *, training: bool = True) -> np.ndarray:
        # If in training mode, the input must be cached for backpropagation.
        if training:
            self.x = x

        # Every data point must be a row in `x`,
        # and the height of `x` is the size of the batch.
        assert self.w.shape[0] == x.shape[1]

        # The bias vector is broadcasted along the batch dimension.
        return np.matmul(x, self.w) + self.b

    # The attributes `dl_dw` and `dl_db` are used to store the gradients.
    # New values are written in-place via the `[:]` slicing notation.
    def backward(self, dl_dz: np.ndarray) -> np.ndarray:
        # The input data is used in gradient calculation for the weights.
        self.dl_dw[:] = np.matmul(self.x.T, dl_dz)
        # The gradients for the bias vector are summed across the batch.
        self.dl_db[:] = dl_dz.sum(axis=0, keepdims=True)

        return np.matmul(dl_dz, self.w.T)

    def parameters(self) -> list[np.ndarray]:
        # If the bias is learnt, then it is exposed as learnable.
        return [self.w, self.b] if self.use_bias else [self.w]

    def gradients(self) -> list[np.ndarray]:
        # If the bias is learnt, its gradients are exposed as learnable.
        return [self.dl_dw, self.dl_db] if self.use_bias else [self.dl_dw]
