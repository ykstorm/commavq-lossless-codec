"""Build the commaVQ submission: compress splits 0+1 with gpt2m + AC and package the zip.

Maps `submission_codec` over every segment, writes one compressed file per segment,
copies the decompressor and the codec modules it needs, and zips. Mirrors the layout of
commavq's baseline compress.py so `./compression/evaluate.sh` can score it.

This is the full-dataset job (about 768M tokens) and has not been run. PROVIDERS prefers
CUDA but the shipped decompress.py decodes on CPU, so check that pairing round-trips on a
few segments first (see docs/submission.md).
"""
import os, shutil, struct
from pathlib import Path
import numpy as np
from datasets import load_dataset

from .decision import RAW_BITS_PER_TOKEN
from .paths import CODEC_DIR, GPT2M_ONNX
from .submission_codec import DATASET, GRID, SPLIT_FILES, Gpt2mStepper, compress_segment

OUT = CODEC_DIR.parent / "submission_out"
ZIP_BASE = CODEC_DIR.parent / "commavq_submission"
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]  # falls back to CPU if no GPU
# what decompress.py imports; extend this if submission_codec gains an import
SHIPPED_MODULES = ["__init__.py", "quantize.py", "range_coder.py", "submission_codec.py"]

def main():
    os.makedirs(OUT, exist_ok=True)
    ds = load_dataset(DATASET, data_files={"train": SPLIT_FILES})["train"]
    stepper = Gpt2mStepper(str(GPT2M_ONNX), providers=PROVIDERS)

    total_tokens = 0
    for i in range(ds.num_rows):
        ex = ds[i]
        name = ex["json"]["file_name"]
        tokens = np.array(ex["token.npy"]).reshape(-1, GRID).astype(np.int64)
        blob = compress_segment(stepper, tokens)
        with open(OUT / name, "wb") as fh:
            fh.write(struct.pack("<I", tokens.shape[0]) + blob)  # 4-byte frame count header
        total_tokens += tokens.size
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{ds.num_rows}] segments compressed", flush=True)

    pkg = OUT / "codec"
    pkg.mkdir(exist_ok=True)
    for m in SHIPPED_MODULES:
        shutil.copy(CODEC_DIR / m, pkg / m)
    shutil.copy(CODEC_DIR / "decompress.py", OUT / "decompress.py")

    zip_path = Path(shutil.make_archive(str(ZIP_BASE), "zip", OUT))
    rate = (total_tokens * RAW_BITS_PER_TOKEN / 8) / zip_path.stat().st_size
    print(f"SUBMISSION built: {zip_path}  rate={rate:.2f}", flush=True)

if __name__ == "__main__":
    main()
