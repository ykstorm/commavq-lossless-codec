import numpy as np
from codec.synthetic import SyntheticDrive, SyntheticDriveModel

def test_model_normalized(small_codebook):
    drive = SyntheticDrive(grid=4, vocab=small_codebook.shape[0], codebook=small_codebook, seed=0)
    m = SyntheticDriveModel(drive)
    p = m.probs(context=[0,1,2,3], pos=2)
    assert p.shape == (small_codebook.shape[0],)
    assert abs(p.sum() - 1.0) < 1e-9

def test_generate_deterministic(small_codebook):
    drive = SyntheticDrive(grid=4, vocab=small_codebook.shape[0], codebook=small_codebook, seed=0)
    a = drive.generate(30, seed=1); b = drive.generate(30, seed=1)
    assert np.array_equal(a, b)

def test_copy_rate_near_target(small_codebook):
    drive = SyntheticDrive(grid=4, vocab=small_codebook.shape[0], codebook=small_codebook,
                           seed=0, p_copy=0.4)
    t = drive.generate(400, seed=2)
    copy = (t[1:] == t[:-1]).mean()
    assert 0.3 < copy < 0.6   # at least the injected temporal-copy structure is present

def test_model_beats_uniform(small_codebook):
    vocab = small_codebook.shape[0]
    drive = SyntheticDrive(grid=4, vocab=vocab, codebook=small_codebook, seed=0)
    m = SyntheticDriveModel(drive)
    t = drive.generate(200, seed=3)
    flat = t.reshape(-1); grid = 4
    xent = 0.0
    for i in range(grid, flat.size):
        p = m.probs(flat[max(0,i-20*grid):i], i % grid); xent += -np.log(max(p[flat[i]],1e-12))
    bpt = xent / np.log(2) / (flat.size - grid)
    assert bpt < np.log2(vocab)   # model is informative
