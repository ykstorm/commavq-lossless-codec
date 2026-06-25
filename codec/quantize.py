import numpy as np

PRECISION_BITS = 16  # total frequency mass = 2**16; must satisfy total <= range_coder BOT

def quantize(probs, precision_bits=PRECISION_BITS):
    """Map a probability vector to int64 frequencies summing to exactly 2**precision_bits,
    every entry >= 1. Deterministic for a given input."""
    total = 1 << precision_bits
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    n = p.shape[0]
    assert total >= n, "precision too small for vocab"
    freqs = np.floor(p * (total - n)).astype(np.int64) + 1  # each >= 1
    diff = total - int(freqs.sum())
    if diff != 0:
        order = np.argsort(-p, kind="stable")  # deterministic tie-break by index
        step = 1 if diff > 0 else -1
        remaining = abs(diff)
        i = 0
        while remaining > 0:
            idx = order[i % n]
            if step > 0 or freqs[idx] > 1:
                freqs[idx] += step
                remaining -= 1
            i += 1
    return freqs
