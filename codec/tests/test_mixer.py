import numpy as np
from codec.mixer import LogisticMixer

def test_initial_equals_first_model():
    mx = LogisticMixer(n_positions=1, n_models=3)
    p1 = np.array([0.7, 0.2, 0.1]); p2 = np.array([0.1, 0.1, 0.8]); p3 = np.ones(3)/3
    out = mx.mix([p1, p2, p3], pos=0)
    assert np.allclose(out, p1, atol=1e-9)   # init weights = [1,0,0]

def test_normalized():
    mx = LogisticMixer(n_positions=2, n_models=2)
    out = mx.mix([np.array([0.5,0.5]), np.array([0.9,0.1])], pos=1)
    assert abs(out.sum() - 1.0) < 1e-9

def test_learns_to_favor_better_model():
    # model 0 always wrong, model 1 always right -> mixer should shift weight to model 1.
    mx = LogisticMixer(n_positions=1, n_models=2, lr=0.1)
    bad = np.array([0.9, 0.05, 0.05]); good = np.array([0.05, 0.05, 0.9])
    truth = 2
    for _ in range(300):
        mx.mix([bad, good], pos=0); mx.update(truth)
    out = mx.mix([bad, good], pos=0)
    assert out.argmax() == truth
    assert mx.w[0, 1] > mx.w[0, 0]

def test_deterministic_replay():
    seq = [0, 1, 2, 1, 0]
    a = np.array([0.6,0.3,0.1]); b = np.array([0.2,0.3,0.5])
    def run():
        mx = LogisticMixer(n_positions=1, n_models=2, lr=0.1)
        outs = []
        for s in seq:
            outs.append(mx.mix([a, b], pos=0)); mx.update(s)
        return np.stack(outs)
    assert np.array_equal(run(), run())
