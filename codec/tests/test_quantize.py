import numpy as np
from codec.quantize import quantize, PRECISION_BITS

def test_sums_to_total_and_min_one():
    total = 1 << PRECISION_BITS
    p = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
    f = quantize(p)
    assert f.dtype == np.int64
    assert int(f.sum()) == total
    assert (f >= 1).all()

def test_deterministic():
    p = np.array([0.1, 0.2, 0.7])
    assert np.array_equal(quantize(p), quantize(p.copy()))

def test_monotone_with_prob():
    p = np.array([0.7, 0.2, 0.1])
    f = quantize(p)
    assert f[0] >= f[1] >= f[2]

def test_handles_large_vocab():
    rng = np.random.default_rng(0)
    p = rng.random(1024); p /= p.sum()
    f = quantize(p)
    assert int(f.sum()) == (1 << PRECISION_BITS)
    assert (f >= 1).all()
