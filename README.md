# commaVQ Lossless Compression — codec + negative-results study

![tests](https://img.shields.io/badge/tests-47%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![cloud spend](https://img.shields.io/badge/cloud%20spend-%240-blueviolet)

Lossless compression of [comma.ai's commaVQ](https://github.com/commaai/commavq) driving-video tokens. A **provably-lossless `gpt2m` + arithmetic-coding codec** that reaches **compression ratio 4.04** — matching the challenge's #1 method — plus a **rigorous study showing that two intuitively-promising improvements add nothing on top of it.**

> **The honest result:** the pretrained `gpt2m` world model with a correct arithmetic coder is already at the *entropy ceiling* for this data. I prove it two independent ways (classical adaptive-filter context-mixing, and in-distribution LoRA finetuning) — both fail, both rigorously measured on held-out data, and independently corroborated against other public solutions. The working engineering and the truthfully-reported negatives are the contribution.

---

## Result

| Method | bits/token | ratio | vs. leaderboard* |
|---|---:|---:|---|
| lzma baseline (comma's example) | — | 1.6 | rank 16 |
| **gpt2m + arithmetic coding (this codec)** | **2.47** | **4.04** | matches rank 1–2 |

<sub>\* 16-entry leaderboard. Validated lossless on sample segments through the full 2580-token context window. The original challenge closed July 2024 — this is a faithful, independently-verified reproduction + study, not an official leaderboard entry.</sub>

The codec is **gpt2m-only on purpose**: the experiments below show that adding anything on top makes it *worse*.

---

## Why this is worth a look

- **Provably-lossless systems engineering** — a range coder + model whose encoder/decoder stay bit-identical in lockstep; tested to be lossless across *every* hyperparameter configuration, not just the happy path.
- **Scientific honesty** — two improvement ideas were built and *disproven* on held-out data, then cross-checked against the field. Most write-ups bury negatives; this one leads with them.
- **Cost discipline** — every result obtained locally + on free Colab, **$0 of cloud spend**. The "prove the gain before you pay for compute" workflow is what surfaced both negatives cheaply.

---

## The finding: gpt2m + AC is at the ceiling

Two improvements, each built, run, and measured on **held-out** data. Both fail.

**1. Adaptive-filter context mixing (a classic signal-processing angle).**
Fuse gpt2m's distribution with classical online **adaptive filters** — a Widrow-Hoff **LMS** predictor over the VQ codebook-embedding space + a per-position **RC leaky-integrator** prior — via a position-keyed logistic mixer (itself LMS in the logit domain). Framing: *context mixing is adaptive filtering.*
→ Every configuration is **worse** than gpt2m alone (best fusion 2.33 vs 2.31 bits/token; −1.5% tuned). The adaptive predictors are redundant with gpt2m's attention; the mixer's online adaptation only injects variance that costs bits.

**2. In-distribution LoRA finetuning.**
LoRA-finetune gpt2m on the target data itself (legitimate two-part coding — ship the small adapter). Serious run: 600 steps, 40 segments, rank 16, all Conv1D layers.
→ Held-out gain **+0.04%** (noise); training loss never drops. gpt2m was pretrained on **3,000,000 minutes** of this distribution — no residual signal for a LoRA to capture.

**Independent corroboration.** A separate public solution tested the *same two levers* (context-mixing, test-time training) plus six more — all ≤0% — concluding *"ties the world record; 5.0 is unreachable with the free model."* Two efforts, same wall. Exceeding 4.0 needs a fundamentally better world model, not better coding of gpt2m's outputs. Full detail + numbers in [`docs/findings.md`](docs/findings.md).

---

## How it works

```
            ┌──────────────────────────────────────────────┐
 tokens ───▶│  gpt2m (onnx, KV-cache)  ──▶ P(next | context)│
            └───────────────┬──────────────────────────────┘
                            │  drop BOS slot, renormalise
                            ▼
            ┌──────────────────────────────────────────────┐
            │  quantize → integer frequency table (exact)   │
            └───────────────┬──────────────────────────────┘
                            ▼
            ┌──────────────────────────────────────────────┐
            │  Subbotin range coder  ◀── lockstep ──▶ decode│  lossless
            └──────────────────────────────────────────────┘
   (the fusion experiment adds LMS + RC predictors + a mixer here — and is shown to hurt)
```

Decompression regenerates each distribution **autoregressively** from already-decoded tokens (no peeking) via the onnx KV-cache, so it's bit-identical to compression → exact reconstruction.

```
codec/
  quantize.py          float dist → integer frequency table (the lossless linchpin)
  range_coder.py       Subbotin carryless range coder
  gpt2m_onnx.py        real gpt2m distributions (sliding 20-frame windows)
  submission_codec.py  autoregressive KV-cache decoder (the real lossless compressor)
  pipeline.py          adaptive-fusion codec (lockstep encode/decode)
  rc_prior.py / lms_predictor.py / mixer.py   the (disproven) adaptive-filter stack
  sweep.py / decision.py / cache.py           GPU-free tuning on cached distributions
```

---

## Run it

```bash
pip install -r requirements.txt
python -m pytest codec/tests -q     # 47 pass; model/onnx tests auto-skip without gpt2m
```

The core codec is pure numpy and fully testable with no model or GPU. Real-gpt2m and full-codec tests opt in when `gpt2m.onnx` is present.

---

## Map
- [`docs/findings.md`](docs/findings.md) — full results, all numbers, competitive landscape.
- [`docs/design.md`](docs/design.md) — the design spec.
- [`docs/submission.md`](docs/submission.md) — GPU + full-run path.
- [`notebooks/commavq_lora_colab.ipynb`](notebooks/commavq_lora_colab.ipynb) — the LoRA gate (manual LoRA, no `peft` dependency).

---

## Tech
`Python` · `numpy` · `onnxruntime` · arithmetic / range coding · transformer LMs (`gpt2m`) · LoRA · adaptive filters (LMS/NLMS, RC) · `pytest`

*An applied-ML / signal-processing portfolio piece — working lossless engineering plus rigorous, honestly-reported negative results.*
