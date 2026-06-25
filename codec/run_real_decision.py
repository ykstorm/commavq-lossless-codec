"""Run the Stage-0 decision harness on the REAL gpt2m cache (Plan B gate).

Loads the cached gpt2m distributions + VQ codebook, measures model-only vs tuned
fusion bits/token at the production setting (vocab=1024, grid=128), and reports the
projected compression ratio vs the leaderboard's 4.0.

Usage:
  python -m codec.run_real_decision --cache compression/codec/cache_real_300.npz
"""
import argparse
from pathlib import Path
import numpy as np
from .sweep import sweep, codec_bits
from .decision import model_only_bits, projected_ratio

HERE = Path(__file__).resolve().parents[2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(HERE / "compression" / "codec" / "cache_real_300.npz"))
    ap.add_argument("--codebook", default=str(HERE / "compression" / "codec" / "codebook.npy"))
    ap.add_argument("--tune-frames", type=int, default=80,
                    help="sweep hyperparameters on this many leading frames, then score winner on all")
    args = ap.parse_args()

    npz = np.load(args.cache)
    dists = npz["dists"].astype(np.float64)
    tokens = npz["tokens"].astype(np.int64)
    grid = int(npz["grid"]); vocab = int(npz["vocab"])
    codebook = np.load(args.codebook).astype(np.float64)
    F = tokens.shape[0]

    # tune on a leading subset (cheap), then score the winner on the full cache
    tf = min(args.tune_frames, F)
    sub_tok = tokens[:tf]
    sub_dists = dists[:tf * grid]
    param_grid = {
        "rc_alpha": [0.01, 0.05],
        "mixer_lr": [0.01, 0.05],
        "lms_mu":   [0.0, 0.5],
        "lms_tau":  [1.0],
    }
    print(f"tuning on first {tf} frames over {2*2*2} configs ...", flush=True)
    ranked = sweep(sub_tok, sub_dists, grid, vocab, codebook, param_grid)
    best_cfg = ranked[0][0]
    print(f"best config (subset)  : {best_cfg}  (subset bits/tok {ranked[0][1]:.4f})", flush=True)

    # prove lossless on REAL distributions with the chosen config (not just synthetic tests)
    from .pipeline import Codec, CodecConfig
    from .cache import CachedModel
    cfg = CodecConfig(vocab=vocab, grid=grid, context_frames=20, **best_cfg)
    blob = Codec(cfg, CachedModel(dists), codebook).compress(tokens)
    rt = Codec(cfg, CachedModel(dists), codebook).decompress(blob, n_frames=tokens.shape[0])
    assert np.array_equal(rt, tokens), "REAL-DATA ROUND-TRIP NOT LOSSLESS"
    print(f"lossless round-trip   : OK ({len(blob)} bytes for {tokens.size} tokens)", flush=True)

    # score on the FULL cache
    base = model_only_bits(tokens, dists, grid, vocab, codebook)
    best = codec_bits(tokens, dists, grid, vocab, codebook, **best_cfg)
    gain = base - best
    print(f"=== REAL gpt2m decision (examples/tokens.npy, {F} frames) ===")
    print(f"gpt2m-only bits/token : {base:.4f}  (ratio {projected_ratio(base,10.0):.3f})")
    print(f"BEST fusion bits/token: {best:.4f}  (ratio {projected_ratio(best,10.0):.3f})")
    print(f"fusion gain           : {gain:.4f} bits/token ({100*gain/base:.2f}% smaller)")
    print(f"leaderboard SOTA ratio: 4.0")
    print(f"VERDICT (this subset) : {'BEATS 4.0' if projected_ratio(best,10.0) > 4.0 else 'below 4.0'}")

if __name__ == "__main__":
    main()
