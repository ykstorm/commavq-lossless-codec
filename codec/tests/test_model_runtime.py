import numpy as np
from codec.model_runtime import MockModel

def test_returns_normalized_distribution():
    m = MockModel(vocab=16, seed=0)
    p = m.probs(context=[3, 4, 5], pos=2)
    assert p.shape == (16,)
    assert abs(p.sum() - 1.0) < 1e-9
    assert (p > 0).all()

def test_deterministic_given_context():
    m = MockModel(vocab=16, seed=0)
    a = m.probs(context=[1, 2, 3], pos=0)
    b = m.probs(context=[1, 2, 3], pos=0)
    assert np.array_equal(a, b)

def test_favors_repeat_of_last_token():
    # Mock biases toward repeating the previous token (mimics commaVQ temporal copies).
    m = MockModel(vocab=16, seed=0)
    p = m.probs(context=[7], pos=0)
    assert p.argmax() == 7
