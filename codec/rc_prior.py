import numpy as np

class RCPrior:
    """Per-position first-order leaky integrator (RC low-pass) over the vocab.
    update: freq <- (1-alpha)*freq + alpha*onehot(sym). Sum stays 1."""
    def __init__(self, n_positions, vocab, alpha=0.02):
        self.alpha = float(alpha)
        self.freq = np.full((n_positions, vocab), 1.0 / vocab, dtype=np.float64)

    def predict(self, pos):
        return self.freq[pos].copy()

    def update(self, pos, sym):
        self.freq[pos] *= (1.0 - self.alpha)
        self.freq[pos, sym] += self.alpha
