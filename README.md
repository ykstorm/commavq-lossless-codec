# commaVQ lossless codec

[![tests](https://github.com/ykstorm/commavq-lossless-codec/actions/workflows/tests.yml/badge.svg)](https://github.com/ykstorm/commavq-lossless-codec/actions/workflows/tests.yml)

Lossless compression of the driving-video tokens in comma.ai's [commaVQ](https://github.com/commaai/commavq) challenge. The codec takes each token's next-token distribution from comma's pretrained `gpt2m` world model and codes the token with a range coder. The repo also holds two attempts to improve on plain gpt2m + arithmetic coding, an adaptive-filter fusion and a LoRA finetune. Neither helped.

## The task

The challenge data is the first two splits of commaVQ: 5,000 one-minute segments, each 1200 frames of 8 x 16 tokens from a 1024-entry VQ codebook, about 768M tokens in all. The score is the raw size at 10 bits per token divided by the size of a zip holding the compressed data and a `decompress.py` that must restore every token exactly ([compression/README.md](https://github.com/commaai/commavq/blob/master/compression/README.md), [evaluate.py](https://github.com/commaai/commavq/blob/master/compression/evaluate.py)). The rules count the commavq repo and PyPI as available, and gpt2m is part of the commavq repo, so its weights do not go in the zip.

## How it works

1. gpt2m reads each frame as a BOS token followed by the frame's 128 tokens in raster order, with a context of 20 frames (2580 tokens).
2. For each token, gpt2m gives a distribution over 1025 symbols. The BOS slot is dropped and the rest renormalised.
3. `quantize.py` turns that distribution into integer frequencies that sum to 2^16, each at least 1. Both sides code against this integer table, never the floats.
4. `range_coder.py`, a carryless range coder in Subbotin's style, codes the true token under that table.
5. To decode, `submission_codec.py` rebuilds each distribution from tokens it has already decoded. For each frame it prefills up to 19 previous frames through the onnx KV-cache, then steps one token at a time. Compression runs the same code on the same inputs, so both sides see the same tables as long as onnxruntime returns identical logits on that provider and hardware.

## Results

| Measurement | Data | bits/token | ratio |
|---|---|---:|---:|
| gpt2m + range coder (`submission_codec.py`) | 22 frames (2,816 tokens) of one clip | 2.474 | 4.04 |
| gpt2m cross-entropy, no coder | 4 held-out segments, non-overlapping 20-frame blocks | 2.92 | 3.43 |

The 4.04 comes from compressing a single 22-frame sample to 871 bytes ([docs/findings.md](docs/findings.md), section 4.4). It is not a full-dataset result: a run over all 5,000 segments was not performed, and the figure leaves out the zip container and the shipped `decompress.py`. The other gpt2m measurements in findings.md (section 2.2) range from 2.08 to 2.24 bits/token on commavq's example clip to 2.92 on four held-out segments, measured with different context and windowing, so a 22-frame sample says little about the full-set score.

A lossless round trip was confirmed on 6 frames. The 22-frame sample, which crosses the point where the context fills to 20 frames, compressed without error, but its decompression ran out of memory on the local machine before finishing. A round trip past that boundary has not been confirmed.

For reference, the top score in the leaderboard table of the [commavq README](https://github.com/commaai/commavq#readme) is 4.0, held by two entries listed as "arithmetic coding with commavq-gpt2m", the same method as this codec; the lzma baseline scores 1.6 (table checked 2026-10-04). Those are full-dataset scores and the 4.04 here is one short sample, so the comparison is indicative only. This repo is not a leaderboard entry.

## What did not work

Both experiments, with all their numbers, are in [docs/findings.md](docs/findings.md).

Adaptive-filter fusion. gpt2m's distribution was mixed with two online predictors: a normalised LMS predictor in the VQ codebook's embedding space and a per-position leaky-integrator frequency prior, combined by a logistic mixer with weights per grid position. On a 60-frame gpt2m cache of the example clip, built with a 10-frame context, gpt2m alone scored 2.3146 bits/token. Every fusion setting in the sweep did worse. The best scored 2.3292, and the setting tuned on a leading subset scored 2.3500, 1.5% worse than gpt2m alone.

In-distribution LoRA. LoRA at rank 16 on every Conv1D layer of gpt2m, trained for 600 steps on 40 segments and scored on 4 held-out segments ([notebook](notebooks/commavq_lora_colab.ipynb)). The base model scored 2.9194 bits/token and the finetuned one 2.9181, a 0.04% gain, while the training loss stayed flat near 2.0. At that gain the 12.6 MB adapter would cost far more bytes than it saves. A likely reason is that gpt2m was trained on 3,000,000 minutes of driving video ([commavq README](https://github.com/commaai/commavq#readme)), so an adapter trained on a few more segments of the same kind of footage has little left to learn.

Another public gpt2m + arithmetic coding codec, [meetr1912/commavq-neural-compression](https://github.com/meetr1912/commavq-neural-compression), reports 2.415 bits/token on 30 segments (about 4.1 by its own estimate) and says eight further levers it measured, including context mixing and test-time training, gave no gain. Those are related to the experiments here, not the same.

## Run it

```bash
pip install -r requirements.txt
python -m pytest codec/tests -q
```

The fusion codec, quantizer and range coder are numpy only, and their tests need no model. Three tests also need `gpt2m.onnx` and commavq's `examples/tokens.npy`, and skip when those are missing.

The scripts that use the real model read `gpt2m/` and `examples/` from a commavq checkout, with this repo's `codec/` folder at `<commavq>/compression/codec` (defaults in `codec/paths.py`). Run them from the folder that contains `codec/`:

```bash
python -m codec.extract_codebook                # gpt2m/decoder.onnx -> codec/codebook.npy (needs the onnx package)
python -m codec.build_real_cache --frames 300   # examples/tokens.npy -> codec/cache_real_300.npz
python -m codec.run_real_decision               # fusion against gpt2m alone on that cache
```

`build_real_cache` and `run_real_decision` also take explicit paths as flags. Building the submission zip needs `datasets` and a GPU-sized run; see [docs/submission.md](docs/submission.md).

## Layout

```
codec/
  quantize.py              float distribution -> integer frequency table
  range_coder.py           carryless range coder
  submission_codec.py      gpt2m + range coder with KV-cache decode (ships in the zip)
  decompress.py            the decompressor that ships in the zip
  submission_compress.py   builds the zip (the full run has not been done)
  gpt2m_onnx.py            teacher-forced gpt2m distributions, for measurement
  build_real_cache.py      saves those distributions to an .npz
  extract_codebook.py      pulls the VQ codebook out of decoder.onnx
  pipeline.py              fusion codec, encoder and decoder in lockstep
  lms_predictor.py, rc_prior.py, mixer.py    the fusion components
  cache.py, sweep.py, decision.py, metrics.py   replay a cache and sweep fusion settings
  run_real_decision.py     fusion against gpt2m alone on a real cache
  model_runtime.py, synthetic.py   mock and synthetic models for the tests
  paths.py                 default file locations
  train_lora.py            peft LoRA script for a GPU box
  tests/
docs/
  findings.md              results, numbers, other public solutions
  design.md                the design written before the experiments
  submission.md            GPU and submission notes
notebooks/
  commavq_lora_colab.ipynb   the LoRA run reported above
```

## License

MIT, see [LICENSE](LICENSE).
