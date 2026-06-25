"""Shipped decompressor for the commaVQ submission zip.

The evaluator unzips and runs this with OUTPUT_DIR set. It reproduces gpt2m's per-token
distributions autoregressively (onnx KV-cache) and arithmetic-decodes each segment back
to its exact tokens. gpt2m.onnx is fetched from HuggingFace (free, doesn't count toward
the score). The codec modules ship alongside (in ./codec).

DETERMINISM: must run gpt2m with the same execution provider/precision used at encode time
(see docs/superpowers/SUBMISSION.md). Default CPU provider is the safe, portable choice.
"""
import os, sys, struct
from pathlib import Path
import numpy as np
import multiprocessing
from datasets import load_dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                 # make ./codec importable
from codec.submission_codec import Gpt2mStepper, decompress_segment

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", HERE / "decompressed"))
SPLIT_FILES = ["data-0000.tar.gz", "data-0001.tar.gz"]

def _gpt2m_onnx():
    local = HERE / "gpt2m.onnx"
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download
    return hf_hub_download("commaai/commavq-gpt2m", "gpt2m.onnx")  # verify filename on HF

def main():
    onnx = _gpt2m_onnx()
    stepper = Gpt2mStepper(onnx, providers=["CPUExecutionProvider"])
    ds = load_dataset("commaai/commavq", num_proc=multiprocessing.cpu_count(),
                      data_files={"train": SPLIT_FILES})["train"]
    for ex in ds:
        name = ex["json"]["file_name"]
        with open(OUTPUT_DIR / name, "rb") as fh:
            raw = fh.read()
        n_frames = struct.unpack("<I", raw[:4])[0]
        tokens = decompress_segment(stepper, raw[4:], n_frames=n_frames)  # (F, 128)
        out = tokens.reshape(-1, 8, 16)
        np.save(OUTPUT_DIR / name, out)
        gt = np.array(ex["token.npy"])
        assert np.array_equal(out, gt), f"mismatch for {name}"

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
