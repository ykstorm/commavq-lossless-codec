import numpy as np
from .model_runtime import ModelRuntime

def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

class SyntheticDrive:
    """Generative process matching commaVQ structure: stationary per-position bias,
    temporal copies, spatial-neighbor (embedding) pull, plus a slow non-stationary
    per-position drift. The paired model knows only the stationary part."""
    def __init__(self, grid, vocab, codebook, seed=0, p_copy=0.35, drift_rate=0.01, neigh=0.5):
        self.grid = int(grid); self.vocab = int(vocab)
        self.C = np.asarray(codebook, dtype=np.float64)
        self.p_copy = float(p_copy); self.drift_rate = float(drift_rate); self.neigh = float(neigh)
        rng = np.random.default_rng(seed)
        self.bias = rng.standard_normal((self.grid, self.vocab)) * 0.5
        self._seed = seed

    def generate(self, n_frames, seed=None):
        rng = np.random.default_rng(self._seed if seed is None else seed)
        drift = np.zeros((self.grid, self.vocab))
        toks = np.zeros((n_frames, self.grid), dtype=np.int64)
        for f in range(n_frames):
            drift += rng.standard_normal(drift.shape) * self.drift_rate
            for pos in range(self.grid):
                if f > 0 and rng.random() < self.p_copy:
                    toks[f, pos] = toks[f-1, pos]; continue
                logits = self.bias[pos] + drift[pos]
                if pos > 0:
                    left = self.C[toks[f, pos-1]]
                    logits = logits - self.neigh * np.sum((self.C - left) ** 2, axis=1)
                toks[f, pos] = rng.choice(self.vocab, p=_softmax(logits))
        return toks

class SyntheticDriveModel(ModelRuntime):
    """Stationary predictor: per-position bias + temporal-copy bonus, blind to drift and
    spatial-neighbor structure -> leaves measurable headroom for the adaptive components."""
    def __init__(self, drive, copy_bonus=2.0):
        self.bias = drive.bias; self.grid = drive.grid; self.vocab = drive.vocab
        self.copy_bonus = float(copy_bonus)

    def probs(self, context, pos):
        logits = self.bias[pos].copy()
        if len(context) >= self.grid:
            prev = int(context[-self.grid])
            logits[prev] += self.copy_bonus
        return _softmax(logits)
