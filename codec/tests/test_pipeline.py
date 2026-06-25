import numpy as np
from codec.pipeline import Codec, CodecConfig
from codec.model_runtime import MockModel

def _make_codec(vocab, codebook):
    return Codec(CodecConfig(vocab=vocab, grid=4, context_frames=3),
                 model=MockModel(vocab=vocab, seed=0),
                 codebook=codebook)

def test_lossless_roundtrip_small(rng, small_codebook):
    vocab = small_codebook.shape[0]
    tokens = rng.integers(0, vocab, size=(10, 4)).astype(np.int64)  # 10 frames, grid=4
    codec = _make_codec(vocab, small_codebook)
    data = codec.compress(tokens)
    codec2 = _make_codec(vocab, small_codebook)                     # fresh state for decode
    out = codec2.decompress(data, n_frames=tokens.shape[0])
    assert np.array_equal(out, tokens)

def test_lossless_on_structured_sequence(small_codebook):
    # Highly repetitive (temporal copies) -> must still be exactly lossless.
    vocab = small_codebook.shape[0]
    base = np.array([1, 2, 3, 4])
    tokens = np.tile(base, (20, 1)).astype(np.int64)
    tokens[5:, 0] = 7  # a change partway through
    codec = _make_codec(vocab, small_codebook)
    data = codec.compress(tokens)
    codec2 = _make_codec(vocab, small_codebook)
    out = codec2.decompress(data, n_frames=tokens.shape[0])
    assert np.array_equal(out, tokens)

def test_compresses_below_raw(small_codebook):
    vocab = small_codebook.shape[0]               # 16 -> 4 raw bits/token
    base = np.array([1, 2, 3, 4])
    tokens = np.tile(base, (50, 1)).astype(np.int64)  # very predictable
    codec = _make_codec(vocab, small_codebook)
    data = codec.compress(tokens)
    raw_bits = tokens.size * np.log2(vocab)
    assert len(data) * 8 < raw_bits               # actual compression happened
