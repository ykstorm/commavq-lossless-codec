"""Guarded test for the real gpt2m ONNX backend. Skips when the model/onnxruntime
are absent (keeps the core suite GPU/model-free and fast). Runs only 2 frames."""
import numpy as np
import pytest

from codec.paths import EXAMPLE_TOKENS as TOKENS, GPT2M_ONNX as ONNX

ort = pytest.importorskip("onnxruntime")
pytestmark = pytest.mark.skipif(not (ONNX.exists() and TOKENS.exists()),
                                reason="gpt2m.onnx or examples/tokens.npy missing")

def test_onnx_deterministic_across_sessions():
    # Losslessness requires decompress (a separate process re-running the model) to get
    # bit-identical logits. Verify two independent sessions agree exactly on the same input.
    import numpy as np
    def run():
        s = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
        ids = np.arange(129, dtype=np.int32)[None, :] % 1024
        past = {f"past_{i}": np.zeros((2, 1, 16, 0, 64), dtype=np.float16) for i in range(24)}
        return s.run(["logits"], {"input_ids": ids, **past})[0]
    assert np.array_equal(run(), run())

def test_two_frame_distributions_valid():
    from codec.gpt2m_onnx import precompute_gpt2m_distributions
    tok = np.load(TOKENS).reshape(-1, 128).astype(np.int64)
    d = precompute_gpt2m_distributions(str(ONNX), tok, n_frames=2)
    assert d.shape == (2 * 128, 1024)
    assert np.allclose(d.sum(axis=1), 1.0, atol=1e-6)   # normalized, BOS slot dropped
    assert (d >= 0).all()
    # gpt2m must be informative: bits/token well below raw 10
    flat = tok[:2].reshape(-1)
    bpt = float(-np.log2(np.clip(d[np.arange(flat.size), flat], 1e-12, None)).mean())
    assert bpt < 10.0
