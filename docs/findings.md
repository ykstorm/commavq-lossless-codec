# commaVQ lossless compression: findings

Date: 2026-06-15. Section 4.3 was surveyed later in June 2026, and the leaderboard figures were rechecked on 2026-10-04.

Goal: beat the leaderboard's top score (4.0, "arithmetic coding with commavq-gpt2m") on lossless compression of commaVQ splits 0+1 (5,000 minutes, about 768M tokens, 10 bits per token, vocab 1024).

Outcome: neither of the two methods tested here improved on gpt2m + arithmetic coding. What remains is a gpt2m + arithmetic coding codec, lossless on the samples it was run on, and the negative results below.

---

## 1. What was built

- Adaptive-fusion codec (`codec/pipeline.py` and the modules it imports): deterministic quantizer, Subbotin range coder, RC leaky-integrator prior, embedding-space LMS predictor, position-keyed logistic mixer, mock, cached and real-gpt2m model backends, and a sweep and decision harness. The test suite checks lossless round trips on synthetic data across hyperparameter settings, and `codec/run_real_decision.py` asserts a lossless round trip on the real gpt2m cache.
- Real gpt2m backend (`codec/gpt2m_onnx.py`): per-frame distributions from the local `gpt2m.onnx` (sliding 20-frame BOS-prefixed windows, BOS slot dropped and the rest renormalised). Two onnx sessions gave bit-identical logits on the same input (`test_onnx_deterministic_across_sessions`), which lossless decoding depends on.
- VQ codebook extraction (`codec/extract_codebook.py`): initializer `model.quantize._embedding.weight`, shape (1024, 256).
- LoRA train and gate: `codec/train_lora.py` (peft, for a GPU box) and `notebooks/commavq_lora_colab.ipynb` (manual LoRA, no peft; its configuration is the one reported in 2.4). Both finetune gpt2m and compare held-out cross-entropy with the frozen base.

---

## 2. Empirical results

### 2.1 Data structure (one clip, `examples/tokens.npy`)
| Signal | bits/token |
|---|---|
| 0-order entropy | 9.70 (close to the raw 10; the indices are arbitrary labels, so a Fourier or DCT transform of them cannot help) |
| H(token \| same position, previous frame) | 4.31 (35% of tokens repeat the previous frame exactly, the dominant structure) |
| H(token \| spatial left or up neighbour) | about 6.55 |
| per-(row, col) entropy range | 4.60 to 8.73 |

### 2.2 gpt2m bits/token depends on the clip and the setup
| Eval set | gpt2m bits/token | ratio |
|---|---|---|
| example clip (`examples/tokens.npy`), 20-frame context | 2.08 to 2.24 | 4.5 to 4.8 |
| same clip, 60-frame onnx cache at 10-frame context (the cache used in 2.3) | 2.31 | 4.32 |
| 4 held-out segments, non-overlapping 20-frame blocks (the base model in 2.4) | 2.92 | 3.43 |

The rows differ in clip, context length and windowing, so none of them predicts the full-dataset score. For scale, the leaderboard's two gpt2m + arithmetic coding entries score 4.0 (2.5 bits/token) over all of splits 0 and 1.

### 2.3 Negative result 1: adaptive-filter fusion
On the real 60-frame gpt2m cache (vocab 1024, grid 128), with a hyperparameter sweep:
```
gpt2m only (floor) : 2.3146 bits/token
best fusion config : 2.3292 bits/token   (every config worse)
tuned on subset    : 2.3500 bits/token   (1.5% worse)
result: gpt2m alone is best; every fusion setting adds bits.
```
Likely reason: the embedding LMS and the RC prior are redundant with gpt2m's 20-frame attention, and the online mixer's adaptation adds variance that costs bits. The cache gave gpt2m only a 10-frame context, which should favour fusion, and fusion still lost.

What was tested: as built, the LMS taps are the previous `n_taps` tokens in raster order (left neighbours), not the up and previous-frame taps in the design, and the sweep in `codec/run_real_decision.py` leaves `n_taps` at its default of 3.

### 2.4 Negative result 2: in-distribution LoRA
600 steps on Colab (T4), 40 training segments, rank 16, LoRA on all Conv1D layers, lr 5e-5 with a cosine schedule, scored on 4 held-out segments:
```
base       : 2.9194 bits/token
finetuned  : 2.9181 bits/token   (0.04% better, within noise)
train loss : flat around 2.0 across 600 steps
verdict    : do not ship (the 12.6 MB adapter would save about 0.1 MB over the full dataset)
```
Likely reason: gpt2m was trained on 3,000,000 minutes of driving video ([commavq README](https://github.com/commaai/commavq#readme)) and the challenge data is the same kind of footage, so there may be little left for a small adapter to learn. An earlier 400-step run came out 0.4% worse, probably undertrained; the 600-step run found essentially no gain.

---

## 3. Conclusion

Neither lever lowered gpt2m's cross-entropy: the adaptive-filter fusion made it worse and the in-distribution LoRA gained 0.04%. Two failed levers do not prove that gpt2m + arithmetic coding is optimal, but they are consistent with the leaderboard, where the top entries use exactly that method.
- Matching 4.0 means running this codec over the full dataset, which was not done.
- Doing better probably needs a stronger predictor than gpt2m rather than better coding of its outputs.

---

## 4. Forward paths

### 4.1 Full-dataset submission (tooling built, run not done)
- Method: gpt2m only with the range coder (fusion off, since it hurts). Per segment, gpt2m distributions are computed autoregressively, quantized and arithmetic coded, and `decompress.py` re-runs gpt2m to rebuild the same distributions.
- Built: the codec, the range coder and the onnx backend; `codec/submission_compress.py` (maps the codec over splits 0+1 and builds the zip) and `codec/decompress.py`.
- Remaining: the full 768M-token gpt2m run.
- Blocker: compute. The local machine (GTX 1650, 4 GB) stalls on sustained onnx runs. AWS g5 was blocked because the account is free-tier only and cannot launch GPU instances without a paid plan. Free Colab is the only no-cost GPU, but the full run is long and Colab sessions disconnect.
- Determinism risk: encode and decode must run gpt2m with the same provider and precision. The fp16 onnx model was bit-identical run to run on one machine, which says nothing about other hardware. As written, `submission_compress.py` prefers CUDA while `decompress.py` decodes on CPU, and that pairing has not been tested.

### 4.2 A stronger model
- Beating gpt2m + arithmetic coding most likely needs a better predictor: a larger world model, an ensemble, or a different tokenizer. All are large and uncertain efforts.
- A literature search for published approaches was started but not finished.

---

## 4.3 Other public solutions

Official scores are from the leaderboard table in the [commavq README](https://github.com/commaai/commavq#readme), checked 2026-10-04. Other figures are each project's own README.

| Entry | Method | Score | Notes |
|---|---|---|---|
| pmazumder3927 | arithmetic coding with commavq-gpt2m | 4.0 (official) | |
| JPL11 | arithmetic coding with commavq-gpt2m | 4.0 (official) | |
| mune-io | arithmetic coding with commavq-gpt2m | 3.7 (official) | |
| meetr1912 | gpt2m + 32-bit arithmetic coder (WNC + E3) | about 4.1 to 4.2, self-estimated | not on the leaderboard; 2.415 bits/token measured on 30 segments; reports eight further levers (calibration, longer context, better coding, three test-time-training variants, context mixing, cross-segment dedup), none helping |
| szabolcs-cs | self-compressing neural network | 3.4 (official) | |
| SAT-oO | NextFramePredictor transformer (about 4.48M parameters, 8-frame context) + constriction range coder | 3.0 (official) | README: 2.96 for the whole submission, 3.05 for the data alone |
| ylevental | 5.3M-parameter transformer predicting a whole frame at once + ANS | 2.7 (official) | 3.62 bits/token; ships a 4.5 MB 8-bit model; a separate zpaq entry scores 2.2 |
| ksd3 | 4M-parameter frame-level transformer + ANS | 2.7 (official) | README: 2.75, 3.56 bits/token |
| baseline | lzma | 1.6 (official) | |

No official score is above 4.0. meetr1912's README reports measuring context mixing and test-time training on top of gpt2m + arithmetic coding, with no gain. Those are related to the fusion and LoRA experiments here, not identical, but they point the same way.

Sources: [commaai/commavq](https://github.com/commaai/commavq), [meetr1912/commavq-neural-compression](https://github.com/meetr1912/commavq-neural-compression), [SAT-oO/commavq_compressor](https://github.com/SAT-oO/commavq_compressor), [ylevental/commavq-compression](https://github.com/ylevental/commavq-compression), [ksd3/comma-compression-challenge-kshitij](https://github.com/ksd3/comma-compression-challenge-kshitij).

## 4.4 Submission codec

`codec/submission_codec.py` implements gpt2m + arithmetic coding with an autoregressive KV-cache decoder that rebuilds each distribution from already-decoded tokens. Checked locally:
- 6-frame round trip: lossless.
- 22-frame compress, which crosses frame 19, where the context reaches the full 2580 tokens: 871 bytes, 2.474 bits/token, ratio 4.04, no error.
- The 22-frame decompress ran out of memory before finishing, so the round trip past that boundary is unconfirmed. The decoder runs the same code for every frame index, which makes a boundary bug unlikely, but that is reasoning, not a test.

This is the method behind the leaderboard's 4.0 entries, and one 22-frame sample came out at 4.04. That is not a leaderboard score: it is one short sample, it leaves out the zip overhead, and the full-dataset run was not performed (sustained onnx runs exhausted memory on the local machine). See `docs/submission.md` for the GPU and submission path.

## 5. Compute used
- No paid compute. The AWS account was free-tier only, which blocked the planned g5 launch.
- All measurements ran on the local machine and on free Colab.
