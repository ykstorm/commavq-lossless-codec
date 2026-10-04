"""Default file locations for the tooling scripts and the model-gated tests."""
from pathlib import Path

CODEC_DIR = Path(__file__).resolve().parent
# gpt2m/ and examples/ come from a commavq checkout that holds this folder at compression/codec.
COMMAVQ_ROOT = CODEC_DIR.parents[1]

GPT2M_ONNX = COMMAVQ_ROOT / "gpt2m" / "gpt2m.onnx"
GPT2M_DECODER_ONNX = COMMAVQ_ROOT / "gpt2m" / "decoder.onnx"
EXAMPLE_TOKENS = COMMAVQ_ROOT / "examples" / "tokens.npy"
REAL_CACHE = CODEC_DIR / "cache_real_300.npz"
CODEBOOK = CODEC_DIR / "codebook.npy"
