import numpy as np
from codec.rc_prior import RCPrior

def test_predict_is_normalized():
    rc = RCPrior(n_positions=4, vocab=8)
    p = rc.predict(2)
    assert p.shape == (8,)
    assert abs(p.sum() - 1.0) < 1e-9

def test_learns_repeated_symbol():
    rc = RCPrior(n_positions=1, vocab=8, alpha=0.1)
    for _ in range(200):
        rc.update(0, 3)
    p = rc.predict(0)
    assert p.argmax() == 3
    assert p[3] > 0.5

def test_positions_independent():
    rc = RCPrior(n_positions=2, vocab=4, alpha=0.2)
    for _ in range(50):
        rc.update(0, 1)
        rc.update(1, 2)
    assert rc.predict(0).argmax() == 1
    assert rc.predict(1).argmax() == 2
