import numpy as np
import pytest
from codec.pipeline import Codec, CodecConfig
from codec.model_runtime import MockModel
from codec.synthetic import SyntheticDrive, SyntheticDriveModel
from codec.cache import precompute_distributions, CachedModel
from codec.quantize import quantize


def _roundtrip(tokens, cfg, codebook, model_factory):
    data = Codec(cfg, model_factory(), codebook).compress(tokens)
    out = Codec(cfg, model_factory(), codebook).decompress(data, n_frames=tokens.shape[0])
    return data, out


# ---------- degenerate shapes ----------

def test_single_token(small_codebook):
    vocab = small_codebook.shape[0]
    tokens = np.array([[3]], dtype=np.int64)              # 1 frame, grid 1
    cfg = CodecConfig(vocab=vocab, grid=1, context_frames=5)
    _, out = _roundtrip(tokens, cfg, small_codebook, lambda: MockModel(vocab, seed=0))
    assert np.array_equal(out, tokens)

def test_single_frame(small_codebook):
    vocab = small_codebook.shape[0]
    tokens = np.array([[1, 2, 3, 4]], dtype=np.int64)
    cfg = CodecConfig(vocab=vocab, grid=4, context_frames=5)
    _, out = _roundtrip(tokens, cfg, small_codebook, lambda: MockModel(vocab, seed=0))
    assert np.array_equal(out, tokens)

def test_all_same_token(small_codebook):
    vocab = small_codebook.shape[0]
    tokens = np.full((30, 4), 2, dtype=np.int64)
    cfg = CodecConfig(vocab=vocab, grid=4, context_frames=20)
    data, out = _roundtrip(tokens, cfg, small_codebook, lambda: MockModel(vocab, seed=0))
    assert np.array_equal(out, tokens)
    # near-constant stream must compress hard (well under raw)
    assert len(data) * 8 < 0.5 * tokens.size * np.log2(vocab)

def test_extreme_token_values(small_codebook):
    vocab = small_codebook.shape[0]
    # only the two extreme code indices, alternating
    tokens = np.tile(np.array([0, vocab - 1, 0, vocab - 1]), (15, 1)).astype(np.int64)
    cfg = CodecConfig(vocab=vocab, grid=4, context_frames=20)
    _, out = _roundtrip(tokens, cfg, small_codebook, lambda: MockModel(vocab, seed=0))
    assert np.array_equal(out, tokens)


# ---------- determinism ----------

def test_compress_is_byte_deterministic(rng, small_codebook):
    vocab = small_codebook.shape[0]
    tokens = rng.integers(0, vocab, size=(25, 4)).astype(np.int64)
    cfg = CodecConfig(vocab=vocab, grid=4, context_frames=20)
    a = Codec(cfg, MockModel(vocab, seed=0), small_codebook).compress(tokens)
    b = Codec(cfg, MockModel(vocab, seed=0), small_codebook).compress(tokens)
    assert a == b


# ---------- losslessness must hold for ANY hyperparameter config ----------

@pytest.mark.parametrize("hp", [
    dict(rc_alpha=0.001, mixer_lr=0.0,  lms_mu=0.0,  lms_tau=0.5, lms_taps=1, precision_bits=12),
    dict(rc_alpha=0.2,   mixer_lr=0.1,  lms_mu=0.1,  lms_tau=2.0, lms_taps=5, precision_bits=16),
    dict(rc_alpha=0.05,  mixer_lr=0.03, lms_mu=0.02, lms_tau=1.0, lms_taps=3, precision_bits=14),
])
def test_lossless_across_configs(rng, small_codebook, hp):
    vocab = small_codebook.shape[0]
    tokens = rng.integers(0, vocab, size=(20, 4)).astype(np.int64)
    cfg = CodecConfig(vocab=vocab, grid=4, context_frames=20, **hp)
    _, out = _roundtrip(tokens, cfg, small_codebook, lambda: MockModel(vocab, seed=0))
    assert np.array_equal(out, tokens)


# ---------- larger vocab + realistic cached distributions ----------

def test_lossless_large_vocab_cached():
    rng = np.random.default_rng(3)
    vocab, grid, dim = 128, 16, 8
    codebook = rng.standard_normal((vocab, dim))
    drive = SyntheticDrive(grid=grid, vocab=vocab, codebook=codebook, seed=1)
    model = SyntheticDriveModel(drive)
    tokens = drive.generate(40, seed=2)
    dists = precompute_distributions(model, tokens, grid, 20, vocab)
    cfg = CodecConfig(vocab=vocab, grid=grid, context_frames=20)
    data = Codec(cfg, CachedModel(dists), codebook).compress(tokens)
    out = Codec(cfg, CachedModel(dists), codebook).decompress(data, n_frames=tokens.shape[0])
    assert np.array_equal(out, tokens)


# ---------- quantizer hardening ----------

def test_quantize_extreme_skew():
    p = np.zeros(1024); p[7] = 1.0                       # all mass on one symbol
    f = quantize(p)
    assert int(f.sum()) == (1 << 16)
    assert (f >= 1).all()
    assert f.argmax() == 7

def test_quantize_low_precision_many_symbols():
    # precision just above vocab: total = 4096 >= 1024
    p = np.ones(1024) / 1024
    f = quantize(p, precision_bits=12)
    assert int(f.sum()) == (1 << 12)
    assert (f >= 1).all()
