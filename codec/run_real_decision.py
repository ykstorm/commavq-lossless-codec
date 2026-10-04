"""Compare tuned fusion against gpt2m alone on a real gpt2m distribution cache.

Loads the cached gpt2m distributions and the VQ codebook, measures model-only and tuned
fusion bits/token at the production setting (vocab=1024, grid=128), and prints the
implied compression ratio next to the leaderboard's 4.0.

Usage:
  python -m codec.run_real_decision
"""
import argparse, math
import numpy as np
from .cache import CachedModel
from .decision import RAW_BITS_PER_TOKEN, model_only_bits, projected_ratio
from .paths import CODEBOOK, REAL_CACHE
from .pipeline import Codec, CodecConfig, DEFAULT_CONTEXT_FRAMES
from .sweep import sweep, codec_bits

LEADERBOARD_TOP = 4.0  # best commaVQ leaderboard score, held by gpt2m + arithmetic coding entries

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(REAL_CACHE))
    ap.add_argument("--codebook", default=str(CODEBOOK))
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
    n_configs = math.prod(len(v) for v in param_grid.values())
    print(f"tuning on first {tf} frames over {n_configs} configs ...", flush=True)
    ranked = sweep(sub_tok, sub_dists, grid, vocab, codebook, param_grid)
    best_cfg = ranked[0][0]
    print(f"best config (subset)  : {best_cfg}  (subset bits/tok {ranked[0][1]:.4f})", flush=True)

    # check the round trip on real distributions with the chosen config, not just synthetic tests
    cfg = CodecConfig(vocab=vocab, grid=grid, context_frames=DEFAULT_CONTEXT_FRAMES, **best_cfg)
    blob = Codec(cfg, CachedModel(dists), codebook).compress(tokens)
    rt = Codec(cfg, CachedModel(dists), codebook).decompress(blob, n_frames=tokens.shape[0])
    assert np.array_equal(rt, tokens), "REAL-DATA ROUND-TRIP NOT LOSSLESS"
    print(f"lossless round-trip   : OK ({len(blob)} bytes for {tokens.size} tokens)", flush=True)

    base = model_only_bits(tokens, dists, grid, vocab, codebook)
    best = codec_bits(tokens, dists, grid, vocab, codebook, **best_cfg)
    gain = base - best
    best_ratio = projected_ratio(best, RAW_BITS_PER_TOKEN)
    print(f"=== REAL gpt2m decision (examples/tokens.npy, {F} frames) ===")
    print(f"gpt2m-only bits/token : {base:.4f}  (ratio {projected_ratio(base, RAW_BITS_PER_TOKEN):.3f})")
    print(f"BEST fusion bits/token: {best:.4f}  (ratio {best_ratio:.3f})")
    print(f"fusion gain           : {gain:.4f} bits/token ({100*gain/base:.2f}% smaller)")
    print(f"leaderboard SOTA ratio: {LEADERBOARD_TOP}")
    print(f"VERDICT (this subset) : {'BEATS' if best_ratio > LEADERBOARD_TOP else 'below'} {LEADERBOARD_TOP}")

if __name__ == "__main__":
    main()
