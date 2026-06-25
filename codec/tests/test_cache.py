import numpy as np
from codec.cache import precompute_distributions, CachedModel
from codec.synthetic import SyntheticDrive, SyntheticDriveModel
from codec.pipeline import Codec, CodecConfig

def test_cache_matches_live_model(small_codebook):
    vocab = small_codebook.shape[0]; grid = 4
    drive = SyntheticDrive(grid=grid, vocab=vocab, codebook=small_codebook, seed=0)
    model = SyntheticDriveModel(drive)
    tokens = drive.generate(15, seed=1)
    dists = precompute_distributions(model, tokens, grid, context_frames=20, vocab=vocab)
    assert dists.shape == (tokens.size, vocab)
    # replaying the cache reproduces the live model's distributions in order
    flat = tokens.reshape(-1); cm = CachedModel(dists)
    for t in range(flat.size):
        live = model.probs(flat[max(0,t-20*grid):t], t % grid)
        assert np.allclose(cm.probs(None, None), live)

def test_cached_roundtrip_lossless(small_codebook):
    vocab = small_codebook.shape[0]; grid = 4
    drive = SyntheticDrive(grid=grid, vocab=vocab, codebook=small_codebook, seed=0)
    model = SyntheticDriveModel(drive)
    tokens = drive.generate(20, seed=2)
    dists = precompute_distributions(model, tokens, grid, 20, vocab)
    cfg = CodecConfig(vocab=vocab, grid=grid, context_frames=20)
    enc_model = CachedModel(dists); dec_model = CachedModel(dists)
    data = Codec(cfg, enc_model, small_codebook).compress(tokens)
    out = Codec(cfg, dec_model, small_codebook).decompress(data, n_frames=tokens.shape[0])
    assert np.array_equal(out, tokens)
