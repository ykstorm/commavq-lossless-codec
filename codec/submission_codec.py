"""Real gpt2m + arithmetic-coding submission codec (the actual compressor).

Unlike the measurement cache (teacher-forced, tokens known), a real DEcompressor
cannot peek at the tokens it is recovering. So it must regenerate gpt2m's per-token
distribution autoregressively from already-decoded tokens. We use onnx KV-cache:
per frame, prefill the <=20-frame context window once, then step token-by-token.

Losslessness: compress and decompress call the IDENTICAL stepping code on the IDENTICAL
inputs, so the float16 logits -> float64 distribution are bit-identical both directions.
The fused distribution is quantized to an integer frequency table (deterministic) before
arithmetic coding.

This is gpt2m-only (no fusion -- fusion was shown to hurt). It implements the
leaderboard's 4.0 method correctly. Encoding the full dataset is a heavy GPU job;
this module is validated end-to-end on a small sample locally.
"""
import numpy as np
import onnxruntime as ort

from .quantize import quantize
from .range_coder import RangeEncoder, RangeDecoder

BOS = 1024
GRID = 128
CTX_FRAMES = 19          # previous frames; +current frame's BOS+128 = 20 frames = 2580 tokens
N_LAYERS, N_HEAD, HEAD_DIM = 24, 16, 64

class Gpt2mStepper:
    """onnx gpt2m with incremental KV-cache. reset() -> prefill(ids) -> step(token)..."""
    def __init__(self, onnx_path, providers=None):
        self.sess = ort.InferenceSession(onnx_path, providers=providers or ["CPUExecutionProvider"])
        self.past = None
        self.reset()

    def reset(self):
        self.past = {f"past_{i}": np.zeros((2, 1, N_HEAD, 0, HEAD_DIM), dtype=np.float16)
                     for i in range(N_LAYERS)}

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

def compress_segment(stepper, tokens, precision_bits=16):
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

def decompress_segment(stepper, data, n_frames, precision_bits=16):
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
