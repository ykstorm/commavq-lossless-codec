"""gpt2m + arithmetic coding for one segment. This module ships in the submission zip.

Unlike the measurement cache (teacher-forced, tokens known), the decompressor cannot see
the tokens it is recovering, so it regenerates gpt2m's distributions autoregressively
from already-decoded tokens: per frame, prefill the <=20-frame context window once
through the onnx KV-cache, then step token by token.

Compress and decompress run the same stepping code on the same inputs, so they get the
same logits as long as onnxruntime is deterministic on that provider and hardware. Each
distribution is quantized to an integer frequency table before coding.

gpt2m only: the fusion experiments in this repo made it worse.
"""
import numpy as np
import onnxruntime as ort

from .quantize import PRECISION_BITS, quantize
from .range_coder import RangeEncoder, RangeDecoder

# gpt2m geometry, also used by gpt2m_onnx.py; it lives here because this module ships in the zip.
VOCAB = 1024             # content codes 0..1023
BOS = 1024               # frame separator, the 1025th symbol gpt2m predicts
GRID = 128               # tokens per frame (8 x 16, raster order)
CONTEXT_FRAMES = 20      # gpt2m block_size: 20 frames of BOS + 128 tokens = 2580
CTX_FRAMES = CONTEXT_FRAMES - 1   # previous frames; the current frame's BOS + 128 complete the window
N_LAYERS, N_HEAD, HEAD_DIM = 24, 16, 64

DATASET = "commaai/commavq"
SPLIT_FILES = ["data-0000.tar.gz", "data-0001.tar.gz"]  # challenge splits 0+1; encode and decode must agree

def empty_past():
    """KV cache with no past positions, for a fresh forward pass."""
    return {f"past_{i}": np.zeros((2, 1, N_HEAD, 0, HEAD_DIM), dtype=np.float16)
            for i in range(N_LAYERS)}

class Gpt2mStepper:
    """onnx gpt2m with incremental KV-cache. reset() -> prefill(ids) -> step(token)..."""
    def __init__(self, onnx_path, providers=None):
        self.sess = ort.InferenceSession(onnx_path, providers=providers or ["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.past = empty_past()

    def _run(self, ids):
        feeds = {"input_ids": np.array([ids], dtype=np.int32), **self.past}
        outs = self.sess.run(None, feeds)          # [logits, present_0..present_23]
        self.past = {f"past_{i}": outs[1 + i] for i in range(N_LAYERS)}
        return outs[0]                             # logits (1, L, 1025)

    @staticmethod
    def _dist(logit_row):
        x = logit_row[:BOS].astype(np.float64)     # drop the deterministic BOS slot
        x -= x.max()
        e = np.exp(x)
        return e / e.sum()

    def prefill(self, ids):
        return self._dist(self._run(ids)[0, -1])

    def step(self, token):
        return self._dist(self._run([int(token)])[0, -1])

def _frame_prefill_ids(tokens, f):
    """[BOS, prev_frame...] * up to CTX_FRAMES, then BOS of the current frame f."""
    lo = max(0, f - CTX_FRAMES)
    ids = []
    for pf in tokens[lo:f]:
        ids.append(BOS); ids.extend(int(x) for x in pf)
    ids.append(BOS)
    return ids

def _cum(freqs, sym):
    return int(freqs[:sym].sum()), int(freqs[sym]), int(freqs.sum())

def compress_segment(stepper, tokens, precision_bits=PRECISION_BITS):
    tokens = np.asarray(tokens).reshape(-1, GRID).astype(np.int64)
    F = tokens.shape[0]
    enc = RangeEncoder()
    for f in range(F):
        stepper.reset()
        dist = stepper.prefill(_frame_prefill_ids(tokens, f))
        for k in range(GRID):
            sym = int(tokens[f, k])
            cum, freq, tot = _cum(quantize(dist, precision_bits), sym)
            enc.encode(cum, freq, tot)
            if k < GRID - 1:
                dist = stepper.step(sym)
    return enc.finish()

def decompress_segment(stepper, data, n_frames, precision_bits=PRECISION_BITS):
    dec = RangeDecoder(data)
    tokens = np.zeros((n_frames, GRID), dtype=np.int64)
    for f in range(n_frames):
        stepper.reset()
        dist = stepper.prefill(_frame_prefill_ids(tokens, f))
        for k in range(GRID):
            freqs = quantize(dist, precision_bits)
            cumarr = np.concatenate([[0], np.cumsum(freqs)])
            tot = int(freqs.sum())
            v = dec.get_freq(tot)
            sym = int(np.searchsorted(cumarr, v, side="right") - 1)
            cum, freq, _ = _cum(freqs, sym)
            dec.decode(cum, freq, tot)
            tokens[f, k] = sym
            if k < GRID - 1:
                dist = stepper.step(sym)
    return tokens
