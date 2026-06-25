import numpy as np

class LogisticMixer:
    """Per-position logistic mixing of model distributions in the log domain.
    Weights adapt by online gradient descent on -log p[true] (LMS in logit space).
    Initialized to weight[0]=1, others 0 -> output starts equal to model 0."""
    def __init__(self, n_positions, n_models, lr=0.01):
        self.n_models = int(n_models)
        self.lr = float(lr)
        self.w = np.zeros((n_positions, n_models), dtype=np.float64)
        self.w[:, 0] = 1.0
        self._L = None
        self._p = None
        self._pos = None

    def mix(self, dists, pos):
        L = np.stack([np.log(np.clip(d, 1e-12, None)) for d in dists])  # (n_models, V)
        logits = self.w[pos] @ L                                        # (V,)
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        self._L, self._p, self._pos = L, p, pos
        return p

    def update(self, sym):
        # d(-log p[sym])/dw_m = sum_x p[x] L[m,x] - L[m,sym]
        grad = self._L @ self._p - self._L[:, sym]                      # (n_models,)
        self.w[self._pos] -= self.lr * grad
