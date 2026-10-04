"""Extract the VQ-VAE codebook embedding matrix (1024 x 256) from the local ONNX.

The embedding-space LMS predictor needs the continuous code vectors. Found at
initializer `model.quantize._embedding.weight` in gpt2m/decoder.onnx.

Usage:
  python -m codec.extract_codebook
"""
import numpy as np
from .paths import CODEBOOK, GPT2M_DECODER_ONNX

EMB_NAME = "model.quantize._embedding.weight"

def extract(decoder_onnx=None, out=None):
    import onnx
    from onnx import numpy_helper
    decoder_onnx = decoder_onnx or str(GPT2M_DECODER_ONNX)
    out = out or str(CODEBOOK)
    m = onnx.load(decoder_onnx)
    for init in m.graph.initializer:
        if init.name == EMB_NAME:
            arr = numpy_helper.to_array(init).astype(np.float64)
            np.save(out, arr)
            return out, arr.shape
    raise KeyError(f"{EMB_NAME} not found in {decoder_onnx}")

if __name__ == "__main__":
    path, shape = extract()
    print(f"SAVED {path} shape={shape}")
