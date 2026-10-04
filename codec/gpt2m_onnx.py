"""Real gpt2m distributions via the local ONNX export, for building measurement caches.

Produces per-content-token distributions over the 1024 codebook indices, aligned
with the codec's raster stream order (frame-by-frame, 128 tokens/frame). The BOS
slot (vocab index 1024) is deterministic at frame boundaries, so it is dropped and
the distribution renormalized over 0..1023.

Strategy: one forward pass per frame over a sliding <=context_frames window, each
frame BOS-prefixed (129 tokens/frame). Positions reset per window (0..L-1), which
matches feeding a standalone 20-frame sequence to the learned absolute-position model.
"""
import numpy as np
import onnxruntime as ort

BOS = 1024
TOKENS_PER_FRAME = 129   # 1 BOS + 128 content
GRID = 128
N_LAYERS = 24
N_HEAD = 16
HEAD_DIM = 64

class Gpt2mOnnx:
    def __init__(self, onnx_path, context_frames=20, providers=None):
        self.context_frames = int(context_frames)
        providers = providers or ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self._empty_past = {f"past_{i}": np.zeros((2, 1, N_HEAD, 0, HEAD_DIM), dtype=np.float16)
                            for i in range(N_LAYERS)}

    def _frame_logits(self, window_ids):
        """window_ids: 1-D int array of BOS-prefixed tokens, target frame is the last
        129. Returns logits (L, 1025) float32."""
        ids = window_ids.astype(np.int32)[None, :]
        feeds = {"input_ids": ids, **self._empty_past}
        logits = self.sess.run(["logits"], feeds)[0]      # (1, L, 1025) float16
        return logits[0].astype(np.float64)

    def frame_distributions(self, frames_window):
        """frames_window: list of frames (each a (128,) int array), oldest..target.
        Returns (128, 1024) distributions for the target (last) frame's content tokens."""
        toks = []
        for fr in frames_window:
            toks.append(BOS)
            toks.extend(int(x) for x in fr)
        window = np.array(toks, dtype=np.int64)
        L = window.shape[0]
        logits = self._frame_logits(window)               # (L, 1025)
        # distributions predicting the target frame's c0..c127 are at positions L-129..L-2
        sel = logits[L - TOKENS_PER_FRAME: L - 1, :]       # (128, 1025)
        sel = sel[:, :BOS]                                 # drop BOS slot -> (128, 1024)
        sel = sel - sel.max(axis=1, keepdims=True)
        p = np.exp(sel)
        p /= p.sum(axis=1, keepdims=True)
        return p

def precompute_gpt2m_distributions(onnx_path, tokens, n_frames=None, context_frames=20,
                                   providers=None, progress=None):
    """tokens: (F, 128) int array (raster content tokens, NO BOS).
    Returns (n_frames*128, 1024) float64 distributions in codec stream order."""
    tokens = np.asarray(tokens).reshape(tokens.shape[0], GRID).astype(np.int64)
    F = tokens.shape[0] if n_frames is None else min(n_frames, tokens.shape[0])
    model = Gpt2mOnnx(onnx_path, context_frames=context_frames, providers=providers)
    out = np.empty((F * GRID, 1024), dtype=np.float64)
    for f in range(F):
        lo = max(0, f - (context_frames - 1))
        window = [tokens[g] for g in range(lo, f + 1)]
        out[f * GRID:(f + 1) * GRID] = model.frame_distributions(window)
        if progress:
            progress(f + 1, F)
    return out
