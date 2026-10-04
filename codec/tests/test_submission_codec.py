"""Guarded end-to-end test for the real gpt2m+AC submission codec.

Skips when gpt2m.onnx / onnxruntime are absent (keeps the core suite fast/model-free).
Runs a tiny round-trip. Frames >= 19 (the full 2580-token window) are too slow for the
suite at roughly 30 s/frame on CPU and have not been round-tripped (docs/findings.md 4.4)."""
import numpy as np
import pytest

from codec.paths import EXAMPLE_TOKENS as TOKENS, GPT2M_ONNX as ONNX

pytest.importorskip("onnxruntime")
pytestmark = pytest.mark.skipif(not (ONNX.exists() and TOKENS.exists()),
                                reason="gpt2m.onnx or examples/tokens.npy missing")

@pytest.mark.slow
def test_real_codec_roundtrip_small():
    from codec.submission_codec import (Gpt2mStepper, compress_segment,
                                                    decompress_segment)
    tok = np.load(TOKENS).reshape(-1, 128).astype(np.int64)[:3]
    data = compress_segment(Gpt2mStepper(str(ONNX)), tok)
    out = decompress_segment(Gpt2mStepper(str(ONNX)), data, n_frames=tok.shape[0])
    assert np.array_equal(out, tok)            # real autoregressive decode is lossless
    assert len(data) * 8 / tok.size < 10.0     # actually compresses
