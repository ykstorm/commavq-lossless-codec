import itertools
from .pipeline import Codec, CodecConfig, DEFAULT_CONTEXT_FRAMES
from .cache import CachedModel

def codec_bits(tokens, dists, grid, vocab, codebook, context_frames=DEFAULT_CONTEXT_FRAMES, **hp):
    cfg = CodecConfig(vocab=vocab, grid=grid, context_frames=context_frames, **hp)
    codec = Codec(cfg, CachedModel(dists), codebook)
    data = codec.compress(tokens)
    return len(data) * 8.0 / tokens.size

def sweep(tokens, dists, grid, vocab, codebook, param_grid, context_frames=DEFAULT_CONTEXT_FRAMES):
    keys = list(param_grid)
    results = []
    for combo in itertools.product(*[param_grid[k] for k in keys]):
        hp = dict(zip(keys, combo))
        results.append((hp, codec_bits(tokens, dists, grid, vocab, codebook,
                                       context_frames=context_frames, **hp)))
    results.sort(key=lambda x: x[1])
    return results
