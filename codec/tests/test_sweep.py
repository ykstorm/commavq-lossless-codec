import numpy as np
from codec.sweep import sweep, codec_bits
from codec.synthetic import SyntheticDrive, SyntheticDriveModel
from codec.cache import precompute_distributions

def _setup(small_codebook):
    vocab = small_codebook.shape[0]; grid = 4
    drive = SyntheticDrive(grid=grid, vocab=vocab, codebook=small_codebook, seed=0)
    model = SyntheticDriveModel(drive)
    tokens = drive.generate(120, seed=5)
    dists = precompute_distributions(model, tokens, grid, 20, vocab)
    return tokens, dists, grid, vocab

def test_codec_bits_positive(small_codebook):
    tokens, dists, grid, vocab = _setup(small_codebook)
    bpt = codec_bits(tokens, dists, grid, vocab, small_codebook, rc_alpha=0.02, mixer_lr=0.01)
    assert 0 < bpt < np.log2(vocab)

def test_sweep_orders_by_bits(small_codebook):
    tokens, dists, grid, vocab = _setup(small_codebook)
    grid_params = {"rc_alpha": [0.01, 0.05], "mixer_lr": [0.005, 0.02]}
    res = sweep(tokens, dists, grid, vocab, small_codebook, grid_params)
    assert len(res) == 4
    assert res[0][1] <= res[-1][1]   # sorted ascending by bits/token
    assert isinstance(res[0][0], dict)
