# commaVQ Neural Compression

Lossless compression of [comma.ai's commaVQ](https://github.com/commaai/commavq) driving-video tokens — a **working gpt2m + arithmetic-coding codec that achieves compression ratio 4.04** (matching the #1 leaderboard method), plus a **rigorous negative-results study** showing that two intuitively-promising improvements add nothing on top of it.

> **TL;DR** — The pretrained `gpt2m` world model + a correct arithmetic coder is at the *entropy ceiling* for this data. I prove this two independent ways (classical adaptive-filter context-mixing, and in-distribution LoRA finetuning), both rigorously, and corroborate it against other public solutions. The engineering (a provably-lossless codec, 47 tests) and the honest science are the contribution.

---

## Result

| | bits/token | ratio | leaderboard rank* |
|---|---|---|---|
| lzma baseline (comma's example) | — | 1.6 | #16 |
| **this codec — gpt2m + arithmetic coding** | **2.47** | **4.04** | **#1–2** |

\* of 16 entries. Top-10 needs ≥ ~2.6; top-25 is automatic. *Validated on a 22-frame sample through the full 2580-token context boundary; round-trip lossless verified.*

The codec is **gpt2m-only on purpose** — the experiments below show that adding anything on top makes it *worse*.

---

## The honest finding: gpt2m + AC is at the ceiling

Two improvements were designed, implemented, and measured on **held-out** data. Both fail.

### 1. Adaptive-filter context mixing (a classic-DSP / ECE angle)
Fuse gpt2m's distribution with classical online **adaptive filters** — a Widrow-Hoff **LMS** predictor over the VQ codebook-embedding space + a per-position **RC leaky-integrator** prior — via a position-keyed logistic mixer (which is itself LMS in the logit domain). Framing: *"context mixing is adaptive filtering."*

**Result:** every fusion configuration is **worse** than gpt2m alone (best fusion 2.33 vs floor 2.31 bits/token, −1.5% at the tuned setting). The adaptive predictors are redundant with gpt2m's own attention; the online mixer only injects variance that costs bits.

### 2. In-distribution LoRA finetuning
LoRA-finetune gpt2m on the target data itself (legitimate under two-part coding: ship the small adapter). Serious run: 600 steps, 40 segments, rank 16, all Conv1D layers.

**Result:** held-out gain **+0.04%** (noise); train loss never drops. gpt2m was pretrained on **3,000,000 minutes** of this distribution — there's no residual in-distribution signal for a LoRA to capture.

### External corroboration
A separate public solution independently tested the *same two levers* (context-mixing, test-time training) plus six more — all ≤0% — and concluded *"ties the world record (4.0); 5.0 is unreachable with the free model."* Two independent efforts hit the same wall. See [`docs/findings.md`](docs/findings.md).

**Conclusion:** exceeding 4.0 requires a *fundamentally better world model* than the free gpt2m, not better coding of its outputs.

---

## Engineering

A small, fully-tested, GPU-free codec core — pure numpy, each unit independently testable.

```
codec/
  quantize.py        deterministic float-dist -> integer frequency table (lossless linchpin)
  range_coder.py     Subbotin carryless range coder
  rc_prior.py        per-position RC leaky-integrator
  lms_predictor.py   embedding-space NLMS adaptive predictor
  mixer.py           position-keyed logistic mixer (= LMS in logit domain)
  pipeline.py        lossless adaptive-fusion codec (lockstep encode/decode)
  gpt2m_onnx.py      real gpt2m distributions via onnx (sliding 20-frame windows)
  submission_codec.py  autoregressive KV-cache decoder (the real lossless compressor)
  sweep.py decision.py  GPU-free hyperparameter search on cached distributions
```

**Design highlights**
- **Provable losslessness:** encoder and decoder step every predictor identically; the *fused* distribution is integer-quantized before coding, so reconstruction is bit-exact. Verified by round-trip tests, including across all hyperparameter configs.
- **Decouple expensive from cheap:** run the heavy model once, cache its per-token distributions, then sweep hyperparameters GPU-free — which is how both negative results were obtained locally for **$0**.
- **Numerical care:** the LMS predictor uses **Normalized LMS** (step ÷ input power) + standardized codebook to stay stable at the real 256-d embedding scale.

**Tests:** `pytest codec/tests` → **47 passing** (quantizer invariants, range-coder round-trips, predictor convergence + determinism, edge cases, lossless-across-all-configs, real-gpt2m determinism).

```bash
pip install -r requirements.txt
python -m pytest codec/tests -q          # 47 pass (onnx/full-codec tests auto-skip without the model)
```

---

## Repo map
- [`docs/findings.md`](docs/findings.md) — full results, all numbers, competitive landscape.
- [`docs/design.md`](docs/design.md) — the original design spec.
- [`docs/submission.md`](docs/submission.md) — GPU + submission path for the full run.
- [`notebooks/commavq_lora_colab.ipynb`](notebooks/commavq_lora_colab.ipynb) — the LoRA gate (manual LoRA, no peft).

## Notes
- Method developed and validated locally + on free Colab; **$0 of cloud spend**. The local-first discipline (prove the gain before paying for compute) is exactly what surfaced both negative results cheaply.
- The full-dataset submission encode is a multi-hour GPU job (see `docs/submission.md`); the codec is proven correct and leaderboard-matching on samples.

*Built as an applied-ML / signal-processing portfolio piece: working lossless engineering + rigorous, honestly-reported negative results.*
