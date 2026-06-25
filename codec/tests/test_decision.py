import numpy as np
from codec.decision import model_only_bits, projected_ratio, decision_report
from codec.synthetic import SyntheticDrive, SyntheticDriveModel
from codec.cache import precompute_distributions
from codec.sweep import sweep

def _setup(small_codebook):
    vocab = small_codebook.shape[0]; grid = 4
    drive = SyntheticDrive(grid=grid, vocab=vocab, codebook=small_codebook, seed=0,
                           drift_rate=0.03, neigh=0.8)   # more recoverable structure
    model = SyntheticDriveModel(drive)
    tokens = drive.generate(300, seed=9)
    dists = precompute_distributions(model, tokens, grid, 20, vocab)
    return tokens, dists, grid, vocab

def test_fusion_beats_model_only(small_codebook):
    tokens, dists, grid, vocab = _setup(small_codebook)
    base = model_only_bits(tokens, dists, grid, vocab, small_codebook)
    res = sweep(tokens, dists, grid, vocab, small_codebook,
                {"rc_alpha": [0.02, 0.08], "mixer_lr": [0.01, 0.05], "lms_mu": [0.0, 0.02]})
    best = res[0][1]
    assert best < base    # adaptive fusion recovers the drift/neighbor headroom

def test_projected_ratio(small_codebook):
    tokens, dists, grid, vocab = _setup(small_codebook)
    base = model_only_bits(tokens, dists, grid, vocab, small_codebook)
    r = projected_ratio(base, raw_bits_per_token=np.log2(vocab))
    assert r > 1.0

def test_decision_report_keys(small_codebook):
    tokens, dists, grid, vocab = _setup(small_codebook)
    rep = decision_report(tokens, dists, grid, vocab, small_codebook,
                          {"rc_alpha": [0.02, 0.08], "mixer_lr": [0.01, 0.05]},
                          raw_bits_per_token=np.log2(vocab))
    assert {"model_only_bpt","best_bpt","best_config","model_only_ratio","best_ratio"} <= set(rep)
    assert rep["best_bpt"] <= rep["model_only_bpt"]
