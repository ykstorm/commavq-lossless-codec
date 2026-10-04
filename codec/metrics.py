from .pipeline import Codec, CodecConfig, DEFAULT_CONTEXT_FRAMES
from .model_runtime import MockModel

def bits_per_token(tokens, vocab, grid, codebook, model=None):
    model = model or MockModel(vocab=vocab, seed=0)
    codec = Codec(CodecConfig(vocab=vocab, grid=grid, context_frames=DEFAULT_CONTEXT_FRAMES),
                  model=model, codebook=codebook)
    data = codec.compress(tokens)
    return len(data) * 8.0 / tokens.size

def ablation(tokens, vocab, grid, codebook):
    """Bits/token for model only, +rc, and +rc+lms.

    This does not isolate the components: the mixer weights it zeroes are already zero
    at init and the mixer keeps adapting them, so all three variants score the same.
    No reported result uses it."""
    results = {}
    for name, use_rc, use_lms in [
        ("model_only", False, False),
        ("model+rc", True, False),
        ("model+rc+lms", True, True),
    ]:
        codec = Codec(CodecConfig(vocab=vocab, grid=grid, context_frames=DEFAULT_CONTEXT_FRAMES),
                      model=MockModel(vocab=vocab, seed=0), codebook=codebook)
        if not use_rc:
            codec.mixer.w[:, 2] = 0.0
        if not use_lms:
            codec.mixer.w[:, 1] = 0.0
        data = codec.compress(tokens)
        results[name] = len(data) * 8.0 / tokens.size
    return results
