import numpy as np
from codec.range_coder import RangeEncoder, RangeDecoder

def _cum(freqs, sym):
    return int(freqs[:sym].sum()), int(freqs[sym]), int(freqs.sum())

def test_roundtrip_uniform():
    rng = np.random.default_rng(0)
    V = 1024
    freqs = np.ones(V, dtype=np.int64) * 64  # sum 65536
    syms = rng.integers(0, V, size=5000)
    enc = RangeEncoder()
    for s in syms:
        lo, f, tot = _cum(freqs, int(s)); enc.encode(lo, f, tot)
    data = enc.finish()
    dec = RangeDecoder(data)
    out = []
    cumarr = np.concatenate([[0], np.cumsum(freqs)])
    for _ in range(len(syms)):
        dv = dec.get_freq(int(freqs.sum()))
        s = int(np.searchsorted(cumarr, dv, side="right") - 1)
        lo, f, tot = _cum(freqs, s); dec.decode(lo, f, tot)
        out.append(s)
    assert out == [int(x) for x in syms]

def test_roundtrip_skewed():
    rng = np.random.default_rng(7)
    V = 256
    raw = rng.random(V) ** 4
    freqs = np.maximum((raw / raw.sum() * (65536 - V)).astype(np.int64) + 1, 1)
    # force exact total 65536
    freqs[0] += 65536 - int(freqs.sum())
    assert freqs.sum() == 65536 and (freqs >= 1).all()
    syms = rng.integers(0, V, size=3000)
    enc = RangeEncoder()
    for s in syms:
        lo, f, tot = _cum(freqs, int(s)); enc.encode(lo, f, tot)
    data = enc.finish()
    dec = RangeDecoder(data)
    cumarr = np.concatenate([[0], np.cumsum(freqs)])
    out = []
    for _ in range(len(syms)):
        dv = dec.get_freq(65536)
        s = int(np.searchsorted(cumarr, dv, side="right") - 1)
        lo, f, tot = _cum(freqs, s); dec.decode(lo, f, tot)
        out.append(s)
    assert out == [int(x) for x in syms]
