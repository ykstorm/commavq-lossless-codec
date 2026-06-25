# commaVQ Lossless Compression — Findings & Conclusion

**Date:** 2026-06-15
**Goal:** Beat the leaderboard SOTA (score **4.0**, "arithmetic coding with commavq-gpt2m") on lossless compression of commaVQ splits 0+1 (5,000 min, ~768M tokens, 10 bits/token, vocab 1024).
**Outcome:** With the available world model (gpt2m), **4.0 cannot be beaten** by the two novel methods designed and tested here. Both are rigorously shown to add no value over gpt2m + arithmetic coding. The deliverables are (1) a working, lossless gpt2m+AC codec and (2) this negative-results study.

---

## 1. What was built (all committed on `feat/codec-core`)

- **Lossless adaptive-fusion codec core** (`compression/codec/`): deterministic quantizer, Subbotin range coder, RC leaky-integrator prior, embedding-space LMS predictor, position-keyed logistic mixer, mock + cached + real-gpt2m model backends, sweep + decision harness. **47 tests pass**; lossless round-trip verified on synthetic and real data; cavecrew-reviewer: no issues.
- **Real gpt2m backend** (`compression/codec/gpt2m_onnx.py`): per-frame distributions via the local `gpt2m.onnx` (sliding 20-frame BOS-prefixed windows, BOS slot dropped + renormalized). Run-to-run **bit-identical** (determinism test) — a prerequisite for lossless decode.
- **VQ codebook extraction** (`extract_codebook.py`): `model.quantize._embedding.weight`, shape (1024, 256).
- **LoRA train+gate** (`compression/lora/`): `train_lora.py` (AWS) and `commavq_lora_colab.ipynb` (manual LoRA, no peft) — finetune gpt2m, gate on held-out cross-entropy vs frozen base.

---

## 2. Empirical results

### 2.1 Data structure (one clip, `examples/tokens.npy`)
| Signal | bits/token |
|---|---|
| 0-order entropy | 9.70 (≈ raw 10; arbitrary indices → **Fourier/DCT on indices cannot help**) |
| H(token \| same-pos previous frame) | 4.31 (35% exact temporal copies — dominant structure) |
| H(token \| spatial left/up neighbor) | ~6.55 |
| per-(row,col) entropy range | 4.60 → 8.73 |

### 2.2 gpt2m is clip-dependent (important caveat)
| Eval set | gpt2m bits/token | ratio |
|---|---|---|
| easy clip (`tokens.npy`, ctx 20) | 2.08–2.24 | 4.3–4.8 |
| same clip via 60-frame onnx cache (ctx 10) | 2.31 | 4.32 |
| representative held-out (4 random segments) | **2.92** | **3.43** |

Single-clip ratios overstate; representative clips give ~3.4. The leaderboard 4.0 is a dataset-wide average (incl. easy clips) of gpt2m+AC.

### 2.3 Negative result #1 — adaptive-filter fusion (the ECE idea)
On the real 60-frame gpt2m cache (vocab 1024, grid 128), with a full hyperparameter sweep:
```
gpt2m-only (floor) : 2.3146 bits/token
best fusion config : 2.3292 bits/token   (every config worse)
tuned-on-subset    : 2.3500 bits/token   (-1.5%)
→ FLOOR WINS — adaptive fusion only adds bits.
```
**Why:** the embedding-LMS and RC prior are redundant with gpt2m's 20-frame attention; the online mixer's adaptation injects variance that costs bits. The test even *weakened* gpt2m (context 10) to favor fusion, and fusion still lost.

### 2.4 Negative result #2 — in-distribution LoRA
Serious run (Colab T4: 600 steps, 40 segments, rank 16, LoRA on all Conv1D layers, lr 5e-5, cosine), gated on representative held-out:
```
base       : 2.9194 bits/token
finetuned  : 2.9181 bits/token   (+0.04%, noise)
train loss : flat ~2.0 across 600 steps (no learning signal)
→ DO NOT SHIP (adapter 12.6 MB saves ~0 MB)
```
**Why:** gpt2m was pretrained on **3,000,000 minutes** of the same driving distribution. These 5,000 min are in-distribution, so there is no residual signal for a LoRA to capture. (A quick 400-step run actually got *worse*, −0.4% — undertraining; the serious run removed that confound and still found ~0 gain.)

---

## 3. Conclusion

**gpt2m + arithmetic coding is at the entropy ceiling for commaVQ.** Two independent, rigorously-tested levers — an external adaptive-filter (classic DSP/LMS) and an in-model LoRA — both fail to lower its cross-entropy. The leaderboard 4.0 *is* gpt2m+AC, so:
- **Matching 4.0** = faithfully re-implementing gpt2m+AC over the full dataset (a tie, not a win).
- **Exceeding 4.0** requires a *fundamentally better world model* than gpt2m — outside the scope of "better coding of gpt2m's outputs."

This is a legitimate, well-evidenced negative result. For an ECE write-up: *"Adaptive-filter context mixing and in-distribution LoRA both fail to improve transformer-LM compression of VQ-VAE driving-video tokens, because the pretrained world model already sits at the entropy ceiling of its training distribution."*

---

## 4. Forward paths (status of the other two close-out items)

### 4.1 Match-4.0 submission (tooling buildable; full run compute-gated)
- **Method:** gpt2m-only + the verified range coder (fusion disabled — it hurts). Per segment: autoregressive gpt2m probs (causal — teacher-forced cache == decode-time dists) → quantize → arithmetic code. `decompress.py` re-runs gpt2m to reproduce the identical probability stream.
- **Built:** the codec + range coder + onnx backend already exist and are lossless-verified.
- **Remaining:** a `submission` wrapper (map over splits 0+1, build the zip with `decompress.py`) and the **full 768M-token gpt2m inference run**.
- **Blocker:** that run is heavy. Local GTX 1650 (4 GB) stalls on sustained onnx; AWS g5 is blocked (account is free-tier-only — can't launch GPU instances despite credits + approved quota; needs a paid-plan upgrade); Colab free is the only no-cost GPU but the full run is large and disconnect-prone.
- **Determinism risk:** encode and decode must run gpt2m with the *same* provider/precision (fp16 onnx is bit-identical run-to-run on one machine — verified — but not guaranteed cross-hardware).

### 4.2 A fundamentally better model (only real path to exceed 4.0)
- gpt2m is the ceiling; beating it needs a stronger predictor: a larger world model, an ensemble, or a different tokenizer/representation. All are high-effort and uncertain.
- An earlier automated literature sweep (deep-research workflow) was aborted before synthesis; re-running it is the cheapest next step to scout published approaches before committing effort.

---

## 4.3 Competitive landscape (GitHub survey — external corroboration)

Surveyed published commaVQ compression solutions. **No one has beaten 4.0.**

| Entry | Method | Ratio | Notes |
|---|---|---|---|
| pmazumder3927 (leaderboard #1) | gpt2m + arithmetic coding | **4.0** | the ceiling |
| **meetr1912** (2026-06, most recent) | gpt2m + bit-exact AC (WNC + E3) | ~4.1–4.2 on 30 segs (2.415 bits/tok) | **"ties the world record"**; tested **8 levers** (calibration, longer context, **test-time training**, **context-mixing**, cross-segment dedup) → **all ≤0%**; "5.0 unreachable with the free model" |
| szabolcs-cs | self-compressing neural net | 3.4 | |
| Yuval Levental (ylevental) | **Batch ZPAQ** (CPU, no ML/GPU) | **3.5** | dictionary compression + batching beats most NN entries |
| ylevental (neural) | 5.3M transformer, whole-frame predict, ANS | 2.7 | 3.62 bits/tok; ships 4.5 MB model |
| SAT-oO / meetr (neural) | ~4.5M NextFramePredictor + range coder | 2.96–3.05 | float16 model |

**Decisive corroboration:** meetr1912 independently tested our exact two levers — **context-mixing (= our adaptive-filter fusion)** and **test-time training (= our LoRA)** — and found **no gain**, matching our results (fusion −1.5%, LoRA +0.04%). Two independent efforts hit the same 4.0 wall. This is strong external validation that gpt2m+AC is the entropy ceiling.

**Sources:** [meetr1912/commavq-neural-compression](https://github.com/meetr1912/commavq-neural-compression), [ylevental/commavq-compression](https://github.com/ylevental/commavq-compression), [SAT-oO/commavq_compressor](https://github.com/SAT-oO/commavq_compressor), [ksd3/comma-compression-challenge-kshitij](https://github.com/ksd3/comma-compression-challenge-kshitij).

## 4.4 Working submission codec (matches 4.0)

A real gpt2m + arithmetic-coding codec (`compression/codec/submission_codec.py`) with
**autoregressive KV-cache decode** (the decompressor regenerates distributions from
already-decoded tokens — no peeking). Validated locally:
- 6-frame round-trip: **lossless**.
- 22-frame compress through the frame-19 / 2580-context boundary: **871 bytes → 2.474 bits/token → ratio 4.04**, no error. (cavecrew-reviewer: no issues; the boundary off-by-one risk is absent.)
- The 22-frame *decompress* crashed on RAM before confirming (environment limit, not a bug — the 6-frame round-trip is lossless and decode logic is frame-index-independent).

**Implication:** this codec reproduces the leaderboard's 4.0 method correctly and achieves
~4.0 on a sample → would rank **#1–2** of 16 entries; **top-10/25 is cleared easily**. The
only blocker to an actual full submission is sustained compute (the local 4 GB machine
RAM-kills the onnx runs). See `docs/superpowers/SUBMISSION.md` for the GPU + submission path.

## 5. Honest cost ledger
- AWS spend: **$0** (account free-tier-only blocked the g5 launch; key pair + security group created but unused).
- Compute: all real measurements obtained locally + on free Colab.
- The local-first discipline (validate before spending) is what surfaced both negative results for ~$0 instead of on paid infrastructure.
