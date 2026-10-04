# commaVQ: submission and GPU notes

## Status
- `codec/submission_codec.py` is a gpt2m + arithmetic coding codec. A 6-frame round trip was lossless, and a 22-frame sample that crosses the 2580-token context boundary compressed to ratio 4.04 (2.474 bits/token) without error. The 22-frame decompress ran out of memory before finishing ([findings.md](findings.md), section 4.4).
- `codec/submission_compress.py` builds the zip and `codec/decompress.py` unpacks it. Neither has been run on the real dataset.
- The full run was not attempted on the local machine, which ran out of memory on sustained onnx runs.

## When to set up a GPU
Only for the full submission: compressing all of splits 0+1 (5,000 segments, about 768M tokens). The codec has only been checked on samples, so the full run would also be its first test at scale.

Cost: the codec runs one onnx forward pass per token. On the local CPU that took about 26 to 40 s per frame, which for 6,000,000 frames is roughly 5 to 8 years in a single process. A GPU should be far faster, but no GPU timing was measured, so the cost of the full run is unknown. Decompression costs about the same as compression, and the evaluator runs it once.

Hosts: any CUDA machine works, for example a rented consumer GPU, Colab, or an AWS g5.xlarge (one NVIDIA A10G, 24 GB). The AWS account used here was free-tier only and could not launch g5 instances without a paid plan.

## Setup on the GPU box
```bash
pip install onnxruntime-gpu numpy datasets huggingface_hub
python -m codec.submission_compress   # maps the codec over splits 0+1 and builds the zip
```
`submission_compress.py` already asks onnxruntime for `CUDAExecutionProvider` first and falls back to CPU. It compresses one segment at a time; batching segments would use the GPU better.

## Determinism (must hold for lossless)
Encoding and the shipped `decompress.py` must run gpt2m with the same execution provider and precision. The fp16 onnx model gave bit-identical logits run to run on one machine, which is no guarantee across CPU and GPU. As written, `submission_compress.py` encodes with CUDA when it can while `decompress.py` decodes on CPU. Before a full run, either encode on CPU or check a CUDA-encode, CPU-decode round trip on a few segments.

## How to submit
1. Submit a single zip (the compressed data plus `decompress.py`) through the form linked in the [commavq README](https://github.com/commaai/commavq#readme). comma scores it with `./compression/evaluate.sh your.zip` ([compression/README.md](https://github.com/commaai/commavq/blob/master/compression/README.md)).
2. The prize deadline was July 1, 2024 and the prize is marked claimed, but the leaderboard table has kept gaining entries since (three new rows between June and August 2026), so late submissions appear to still be scored.
3. The README's submission channel is the form. A pull request to `commaai/commavq` is not how entries reach the leaderboard.

## Checklist before any GPU spend
- [ ] Pick a GPU host.
- [ ] Settle the provider question above and round-trip about 5 segments under it.
- [ ] Run `python -m codec.submission_compress`, check the zip with `./compression/evaluate.sh` from a commavq checkout, then submit it through the form.
