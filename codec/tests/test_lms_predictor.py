import numpy as np
from codec.lms_predictor import LMSPredictor

def test_predict_normalized(small_codebook):
    lms = LMSPredictor(small_codebook, n_taps=2)
    p = lms.predict()
    assert p.shape == (small_codebook.shape[0],)
    assert abs(p.sum() - 1.0) < 1e-9

def test_learns_constant_sequence(small_codebook):
    # Feeding the same token repeatedly: predictor should converge to predicting it.
    lms = LMSPredictor(small_codebook, n_taps=3, mu=0.05, tau=0.5)
    target = 5
    xent = []
    for _ in range(300):
        p = lms.predict()
        xent.append(-np.log(max(p[target], 1e-12)))
        lms.update(target)
    assert xent[-1] < xent[0]            # cross-entropy improves
    assert lms.predict().argmax() == target

def test_stable_with_large_magnitude_codebook():
    # 256-D codebook with large values (like the real VQ embeddings) must NOT diverge.
    rng = np.random.default_rng(0)
    codebook = rng.standard_normal((1024, 256)) * 50.0   # large magnitude
    lms = LMSPredictor(codebook, n_taps=3, mu=0.5, tau=1.0)
    syms = rng.integers(0, 1024, size=500)
    for s in syms:
        p = lms.predict()
        assert np.isfinite(p).all(), "distribution went non-finite (LMS diverged)"
        assert abs(p.sum() - 1.0) < 1e-6
        lms.update(int(s))

def test_deterministic_replay(small_codebook):
    seq = [1, 2, 3, 2, 1, 4, 5]
    def run():
        lms = LMSPredictor(small_codebook, n_taps=3, mu=0.05, tau=0.5)
        ps = []
        for s in seq:
            ps.append(lms.predict()); lms.update(s)
        return np.stack(ps)
    a, b = run(), run()
    assert np.array_equal(a, b)          # bit-identical replay (lockstep requirement)
