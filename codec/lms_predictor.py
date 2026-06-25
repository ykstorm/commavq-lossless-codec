import numpy as np

class LMSPredictor:
    """Adaptive linear predictor over codebook-embedding space (Widrow-Hoff LMS),
    mapped to a categorical distribution by distance-softmax over the codebook.

    Scalar weight per tap, shared across embedding dimensions (classic LMS combiner).
    """
    def __init__(self, codebook, n_taps=3, mu=0.5, tau=1.0):
        C = np.asarray(codebook, dtype=np.float64)        # (V, d)
        # standardize to O(1) magnitude so tau is meaningful and NLMS is well-scaled
        self._mean = C.mean(axis=0, keepdims=True)
        self._scale = float(C.std()) + 1e-8
        self.C = (C - self._mean) / self._scale
        self._C2 = np.sum(self.C ** 2, axis=1)            # ||C_k||^2 precomputed
        self.V, self.d = self.C.shape
        self.n_taps = int(n_taps)
        self.mu = float(mu)                                # NLMS normalized step (0<mu<2)
        self.tau = float(tau)
        self.w = np.zeros(self.n_taps, dtype=np.float64)
        self.history = []                                  # recent embeddings, oldest..newest
        self._last_taps = None
        self._last_ehat = None

    def _taps(self):
        h = self.history[-self.n_taps:]
        pad = self.n_taps - len(h)
        rows = [np.zeros(self.d)] * pad + h
        return np.stack(rows)                              # (n_taps, d)

    def predict(self):
        taps = self._taps()
        ehat = self.w @ taps                               # (d,)
        # ||C_k - ehat||^2 = ||C_k||^2 - 2 C_k·ehat + ||ehat||^2 (avoids (V,d) temp)
        d2 = self._C2 - 2.0 * (self.C @ ehat) + float(ehat @ ehat)
        logits = -d2 / self.tau
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        self._last_taps = taps
        self._last_ehat = ehat
        return p

    def update(self, sym):
        e_actual = self.C[sym]
        err = e_actual - self._last_ehat                   # (d,)
        grad = -2.0 * (self._last_taps @ err)              # (n_taps,)
        power = float(np.sum(self._last_taps ** 2)) + 1e-6  # NLMS: normalize by input power
        self.w -= (self.mu / power) * grad
        self.history.append(e_actual.copy())
        if len(self.history) > self.n_taps:
            self.history = self.history[-self.n_taps:]
