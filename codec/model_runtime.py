import numpy as np

class ModelRuntime:
    """Interface for a global-prior model. Plan B implements this with gpt2m.
    probs(context, pos) -> float64 distribution over `vocab` content codes (BOS excluded)."""
    def probs(self, context, pos):
        raise NotImplementedError

class MockModel(ModelRuntime):
    """Deterministic stand-in: biases toward repeating the last context token, plus a
    fixed per-position skew. Pure function of (context tail, pos) -> reproducible at decode."""
    def __init__(self, vocab, seed=0, repeat_weight=6.0):
        self.vocab = int(vocab)
        self.repeat_weight = float(repeat_weight)
        rs = np.random.default_rng(seed)
        # fixed per-position bias table (deterministic, not learned)
        self.bias = rs.standard_normal((128, self.vocab)) * 0.3

    def probs(self, context, pos):
        logits = self.bias[pos % 128].copy()
        if len(context) > 0:
            logits[context[-1] % self.vocab] += self.repeat_weight
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        return p
