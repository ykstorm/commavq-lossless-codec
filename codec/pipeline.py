from dataclasses import dataclass
import numpy as np

from .quantize import quantize
from .range_coder import RangeEncoder, RangeDecoder
from .rc_prior import RCPrior
from .lms_predictor import LMSPredictor
from .mixer import LogisticMixer

@dataclass
class CodecConfig:
    vocab: int
    grid: int            # tokens per frame (128 in production; small in tests)
    context_frames: int  # how many past frames the model sees
    # hyperparameters swept by sweep.py; the defaults are starting points
    rc_alpha: float = 0.02
    mixer_lr: float = 0.01
    lms_mu: float = 0.01
    lms_tau: float = 1.0
    lms_taps: int = 3
    precision_bits: int = 16

class Codec:
    """Lossless adaptive-fusion codec for one token example of shape (n_frames, grid).
    Encoder and decoder construct identical predictor state and step it in lockstep."""
    def __init__(self, config: CodecConfig, model, codebook):
        self.cfg = config
        self.model = model
        self.rc = RCPrior(n_positions=config.grid, vocab=config.vocab, alpha=config.rc_alpha)
        self.lms = LMSPredictor(codebook, n_taps=config.lms_taps, mu=config.lms_mu, tau=config.lms_tau)
        self.mixer = LogisticMixer(n_positions=config.grid, n_models=3, lr=config.mixer_lr)

    def _context(self, flat, t):
        lo = max(0, t - self.cfg.context_frames * self.cfg.grid)
        return flat[lo:t]

    def _distributions(self, flat, t, pos):
        p1 = self.model.probs(self._context(flat, t), pos)
        p2 = self.lms.predict()
        p3 = self.rc.predict(pos)
        return [p1, p2, p3]

    def _advance(self, pos, sym):
        self.lms.update(sym)
        self.rc.update(pos, sym)
        self.mixer.update(sym)

    def compress(self, tokens):
        grid = self.cfg.grid
        flat = tokens.reshape(-1).astype(np.int64)
        enc = RangeEncoder()
        seen = []
        for t in range(flat.shape[0]):
            pos = t % grid
            dists = self._distributions(np.array(seen, dtype=np.int64), t, pos)
            fused = self.mixer.mix(dists, pos)
            freqs = quantize(fused, self.cfg.precision_bits)
            sym = int(flat[t])
            cum = int(freqs[:sym].sum()); f = int(freqs[sym]); tot = int(freqs.sum())
            enc.encode(cum, f, tot)
            seen.append(sym)
            self._advance(pos, sym)
        return enc.finish()

    def decompress(self, data, n_frames):
        grid = self.cfg.grid
        dec = RangeDecoder(data)
        seen = []
        out = []
        total_tokens = n_frames * grid
        for t in range(total_tokens):
            pos = t % grid
            dists = self._distributions(np.array(seen, dtype=np.int64), t, pos)
            fused = self.mixer.mix(dists, pos)
            freqs = quantize(fused, self.cfg.precision_bits)
            cumarr = np.concatenate([[0], np.cumsum(freqs)])
            tot = int(freqs.sum())
            dv = dec.get_freq(tot)
            sym = int(np.searchsorted(cumarr, dv, side="right") - 1)
            cum = int(freqs[:sym].sum()); f = int(freqs[sym])
            dec.decode(cum, f, tot)
            out.append(sym)
            seen.append(sym)
            self._advance(pos, sym)
        return np.array(out, dtype=np.int64).reshape(n_frames, grid)
