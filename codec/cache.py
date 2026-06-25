import numpy as np
from .model_runtime import ModelRuntime

def precompute_distributions(model, tokens, grid, context_frames, vocab):
    """Run `model` over the token stream once, caching each per-token distribution.
    This is the expensive step (real gpt2m in Plan B); cache enables GPU-free sweeps."""
    flat = tokens.reshape(-1).astype(np.int64)
    out = np.empty((flat.shape[0], vocab), dtype=np.float64)
    for t in range(flat.shape[0]):
        lo = max(0, t - context_frames * grid)
        out[t] = model.probs(flat[lo:t], t % grid)
    return out

class CachedModel(ModelRuntime):
    """Replays precomputed distributions in call order. The codec calls probs() exactly
    once per token in stream order, so an internal counter aligns the cache. Fresh instance
    (or reset()) per compress/decompress pass."""
    def __init__(self, dists):
        self.dists = np.asarray(dists, dtype=np.float64)
        self.vocab = self.dists.shape[1]
        self.i = 0
    def reset(self):
        self.i = 0
    def probs(self, context, pos):
        d = self.dists[self.i]; self.i += 1; return d
