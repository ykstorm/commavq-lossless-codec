"""Build a real gpt2m distribution cache from examples/tokens.npy (Plan B Stage-2).

One-time expensive step. Saves an .npz the GPU-free sweep/decision harness replays.

Usage:
  python -m codec.build_real_cache --frames 300 --out compression/codec/cache_real_300.npz
"""
import argparse, os, time
from pathlib import Path
import numpy as np
from .gpt2m_onnx import Gpt2mOnnx, GRID

HERE = Path(__file__).resolve().parents[2]

def _save(out, dists, tokens, done):
    """Atomic checkpoint (tmp + replace) so a kill mid-write never corrupts the file."""
    tmp = str(out) + ".tmp.npz"
    np.savez_compressed(tmp, dists=dists[:done * GRID].astype(np.float32),
                        tokens=tokens[:done], frames=done, grid=GRID, vocab=1024)
    os.replace(tmp, out)   # atomic on same filesystem

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default=str(HERE / "examples" / "tokens.npy"))
    ap.add_argument("--onnx", default=str(HERE / "gpt2m" / "gpt2m.onnx"))
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--context-frames", type=int, default=20)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--out", default=str(HERE / "compression" / "codec" / "cache_real_300.npz"))
    args = ap.parse_args()

    tokens = np.load(args.tokens).reshape(-1, 128).astype(np.int64)
    F = min(args.frames, tokens.shape[0])
    model = Gpt2mOnnx(args.onnx, context_frames=args.context_frames)
    out = np.empty((F * GRID, 1024), dtype=np.float64)
    t0 = time.time()
    for f in range(F):
        lo = max(0, f - (args.context_frames - 1))
        window = [tokens[g] for g in range(lo, f + 1)]
        out[f * GRID:(f + 1) * GRID] = model.frame_distributions(window)
        done = f + 1
        if done % args.checkpoint_every == 0 or done == F:
            _save(args.out, out, tokens, done)
            el = time.time() - t0
            eta = el / done * (F - done)
            print(f"[{done}/{F}] {el:.0f}s, ETA {eta:.0f}s, checkpoint saved", flush=True)

    flat = tokens[:F].reshape(-1)
    p = out[np.arange(flat.size), flat]
    bpt = float(-np.log2(np.clip(p, 1e-12, None)).mean())
    print(f"DONE {args.out}  shape={out.shape}  gpt2m-only bits/token={bpt:.4f}  "
          f"(ratio {10.0/bpt:.3f})", flush=True)

if __name__ == "__main__":
    main()
