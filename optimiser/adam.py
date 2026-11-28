from .base import Optimiser, np


class AdamW(Optimiser):
    def __init__(self, parameters: list[np.ndarray],
                 gradients: list[np.ndarray],
                 *, lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, weight_decay=1e-3):
        # First, references to the parameters and gradients are stored.
        self.parameters = parameters
        self.gradients = gradients

        # Sets the learning rate of the optimiser.
        self.lr = lr

        # Static settings specific to the AdamW optimiser:
        # momentum, weight decay, and epsilon (for numerical stability).
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.eps = 1e-8

        # Stored dynamic values for the AdamW optimiser:
        # time step counter and estimated moment gradients.
        self.t = 0
        self.m = [np.zeros(gradient.shape) for gradient in gradients]
        self.v = [np.zeros(gradient.shape) for gradient in gradients]

    # Conducts a single optimisation step and modifies parameters.
    def step(self):
        # First, the time step counter is incremented.
        self.t += 1

        for m_t, v_t, theta_t, g_t in zip(
                self.m, self.v, self.parameters, self.gradients):
            # First, conduct the weight decay step.
            theta_t -= self.lr * self.weight_decay * theta_t

            # Calculate and update the biased estimators of the two moments.
            m_t[:] = self.beta1 * m_t + (1 - self.beta1) * g_t
            v_t[:] = self.beta2 * v_t + (1 - self.beta2) * g_t * g_t

            # Calculate the unbiased estimators.
            m_hat = m_t / (1 - self.beta1 ** self.t)
            v_hat = v_t / (1 - self.beta2 ** self.t)

            # Finally, update the parameters using the gradient estimates.
            theta_t -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    # Allows changing the learning rate via a learning rate schedule.
    def set_lr(self, lr: float):
        self.lr = lr
