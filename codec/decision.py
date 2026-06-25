from .pipeline import Codec, CodecConfig
from .cache import CachedModel
from .sweep import sweep, codec_bits

def model_only_bits(tokens, dists, grid, vocab, codebook, context_frames=20):
    """Bits/token using the model alone: mixer keeps init weights [1,0,0] and we freeze
    adaptation by setting mixer_lr=0, so the adaptive models contribute nothing."""
    cfg = CodecConfig(vocab=vocab, grid=grid, context_frames=context_frames, mixer_lr=0.0)
    codec = Codec(cfg, CachedModel(dists), codebook)
    data = codec.compress(tokens)
    return len(data) * 8.0 / tokens.size

def projected_ratio(bits_per_token, raw_bits_per_token):
    return raw_bits_per_token / bits_per_token

def decision_report(tokens, dists, grid, vocab, codebook, param_grid, raw_bits_per_token,
                    context_frames=20):
    base = model_only_bits(tokens, dists, grid, vocab, codebook, context_frames)
    res = sweep(tokens, dists, grid, vocab, codebook, param_grid, context_frames)
    best_cfg, best_bpt = res[0]
    return {
        "model_only_bpt": base,
        "best_bpt": best_bpt,
        "best_config": best_cfg,
        "model_only_ratio": projected_ratio(base, raw_bits_per_token),
        "best_ratio": projected_ratio(best_bpt, raw_bits_per_token),
    }
