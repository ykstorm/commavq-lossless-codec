from .pipeline import Codec, CodecConfig
from .model_runtime import MockModel

def bits_per_token(tokens, vocab, grid, codebook, model=None):
    model = model or MockModel(vocab=vocab, seed=0)
    codec = Codec(CodecConfig(vocab=vocab, grid=grid, context_frames=20),
                  model=model, codebook=codebook)
    data = codec.compress(tokens)
    return len(data) * 8.0 / tokens.size

def ablation(tokens, vocab, grid, codebook):
    """Measure bits/token as adaptive components are switched on.
    Disabling = freezing a predictor's mixer weight at 0 (still stepped for lockstep)."""
    results = {}
    for name, use_rc, use_lms in [
        ("model_only", False, False),
        ("model+rc", True, False),
        ("model+rc+lms", True, True),
    ]:
        codec = Codec(CodecConfig(vocab=vocab, grid=grid, context_frames=20),
                      model=MockModel(vocab=vocab, seed=0), codebook=codebook)
        # zero (and freeze) the mixer weights of disabled models; lr already adapts the rest
        if not use_rc:
            codec.mixer.w[:, 2] = 0.0
        if not use_lms:
            codec.mixer.w[:, 1] = 0.0
        data = codec.compress(tokens)
        results[name] = len(data) * 8.0 / tokens.size
    return results
