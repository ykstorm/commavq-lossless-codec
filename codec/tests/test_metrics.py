import numpy as np
from codec.metrics import bits_per_token, ablation

def test_bits_per_token_value(small_codebook):
    vocab = small_codebook.shape[0]
    base = np.array([1, 2, 3, 4])
    tokens = np.tile(base, (40, 1)).astype(np.int64)
    bpt = bits_per_token(tokens, vocab=vocab, grid=4, codebook=small_codebook)
    assert 0.0 < bpt < np.log2(vocab)     # below raw, above zero

def test_ablation_reports_each_variant(small_codebook):
    vocab = small_codebook.shape[0]
    tokens = np.tile(np.array([1, 2, 3, 4]), (40, 1)).astype(np.int64)
    table = ablation(tokens, vocab=vocab, grid=4, codebook=small_codebook)
    assert set(table) == {"model_only", "model+rc", "model+rc+lms"}
    for v in table.values():
        assert v > 0
