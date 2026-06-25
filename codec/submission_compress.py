"""Build the commaVQ submission: compress splits 0+1 with gpt2m+AC, package the zip.

Run on a GPU box (set providers=['CUDAExecutionProvider']). Maps the validated
`submission_codec` over every segment, writes one compressed file per segment, copies
the decompressor + codec modules, and zips. Mirrors the structure of the repo's
baseline compress.py so `./compression/evaluate.sh` scores it.

This is a heavy full-dataset job (~768M tokens) — intended for the cloud GPU run, not
the local 4GB machine.
"""
import os, shutil, struct
from pathlib import Path
import numpy as np
from datasets import load_dataset

from .submission_codec import Gpt2mStepper, compress_segment

HERE = Path(__file__).resolve().parents[2]
ONNX = str(HERE / "gpt2m" / "gpt2m.onnx")
OUT = HERE / "compression" / "submission_out"
SPLIT_FILES = ["data-0000.tar.gz", "data-0001.tar.gz"]
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]  # falls back to CPU if no GPU

def main():
    os.makedirs(OUT, exist_ok=True)
    ds = load_dataset("commaai/commavq", data_files={"train": SPLIT_FILES})["train"]
    stepper = Gpt2mStepper(ONNX, providers=PROVIDERS)

    total_tokens = 0
    for i in range(ds.num_rows):
        ex = ds[i]
        name = ex["json"]["file_name"]
        tokens = np.array(ex["token.npy"]).reshape(-1, 128).astype(np.int64)
        blob = compress_segment(stepper, tokens)
        with open(OUT / name, "wb") as fh:
            fh.write(struct.pack("<I", tokens.shape[0]) + blob)  # 4-byte frame count header
        total_tokens += tokens.size
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{ds.num_rows}] segments compressed", flush=True)

    # ship the decompressor + codec package
    pkg = OUT / "codec"
    pkg.mkdir(exist_ok=True)
    for m in ["__init__.py", "quantize.py", "range_coder.py", "submission_codec.py"]:
        shutil.copy(HERE / "compression" / "codec" / m, pkg / m)
    shutil.copy(HERE / "compression" / "codec" / "decompress.py", OUT / "decompress.py")

    shutil.make_archive(str(HERE / "compression" / "commavq_submission"), "zip", OUT)
    zip_path = HERE / "compression" / "commavq_submission.zip"
    rate = (total_tokens * 10 / 8) / zip_path.stat().st_size
    print(f"SUBMISSION built: {zip_path}  rate={rate:.2f}", flush=True)

if __name__ == "__main__":
    main()
