# commaVQ lossless compression: design

Date: 2026-06-14. Written before the experiments and kept as the plan of record. Where the plan and the results disagree, [findings.md](findings.md) has the measured outcome.

Goal: beat the commaVQ leaderboard's top score (4.0, "arithmetic coding with commavq-gpt2m") on lossless compression of the first two dataset splits (5,000 minutes, about 915 MB of tokens).

Approach: fuse the frozen gpt2m prior with classical online adaptive filters (Widrow-Hoff LMS or RLS, RC leaky integrators), treating context mixing as a form of adaptive filtering.

File references below are to the upstream [commaai/commavq](https://github.com/commaai/commavq) repo at commit 623a27f.

---

## 1. Problem and scoring

- Data: splits `data-0000` and `data-0001` of `commaai/commavq`. Each example is a `token.npy` of shape `(1200, 8, 16)`: 128 VQ-VAE codebook indices per frame at 10 bits per token (vocab 1024, plus BOS makes 1025). About 768M tokens in total.
- Token layout: the frame grid is 8 rows by 16 columns, flattened row-major (raster order) to 128 tokens ([utils/vqvae.py:248][vqvae], [compression/decompress.py:15][decomp]). For the world model each frame is prefixed by `BOS=1024`, giving 129 tokens per frame ([nanogpt/prepare.py:16,30][prepare], [utils/gpt.py:33][gpt33]). The model context is `block_size = 20*129 = 2580` (20 frames, [utils/gpt.py:27][gpt27]) with learned absolute positions ([utils/gpt.py:117][gpt117]), so BOS must land at positions 0, 129, 258, and so on.
- Score ([compression/evaluate.py:25][evaluate]): `score = raw_bytes / zip_size` with `raw_bytes = num_rows * 1200 * 128 * 10 / 8`. The zip holds the compressed payload plus `decompress.py`, and anything shipped counts against the score.
- Target: `score > 4.0`, that is, below 2.5 bits per token.
- Lossless gate: `decompress.py` must reconstruct every `token.npy` exactly; one mismatch fails the submission.

Free and paid bytes (two-part coding): the rules treat everything in the commavq repo and on PyPI as available ([compression/README.md][rules]). gpt2m and the VQ-VAE weights are in the commavq repo (a HuggingFace submodule), so they cost nothing; custom-trained weights would count. This design ships no learned weights: every adaptive component is recomputed deterministically at decode time (section 5).

---

## 2. Empirical grounding (measured on `examples/tokens.npy`, one 1200-frame clip)

| Signal | bits/token | Implication |
|---|---|---|
| 0-order entropy | 9.70 | close to the full 10 bits with all 1024 codes used, so no gain from marginal frequencies and no point in a Fourier or DCT transform of the indices |
| H(token \| same position, previous frame) | 4.31 | 35% of tokens are exact copies of the previous frame, the dominant structure |
| H(token \| left or up neighbour) | about 6.55 | spatial neighbours help, less than temporal ones |
| per-(row, col) entropy | 4.60 to 8.73 (mean 7.68) | strongly non-uniform across positions, which motivates a per-position adaptive prior |

Headroom: the leaderboard's 4.0 implies gpt2m's 20-frame context already reaches about 2.5 bits per token, far below every single-variable entropy above, so gpt2m does most of the work. The hope was that adaptive fusion would add a few tenths of a bit per token, a modest gain rather than a step change; a larger jump would need a better world model, which is out of scope. The design measures each component's contribution separately. (Measured: fusion made things worse; see findings.md, section 2.3.)

---

## 3. Approach: gpt2m prior plus classical adaptive filters

For every content token, form three predictive distributions over the 1024 codebook indices, fuse them, and arithmetic-code the true symbol under the fused distribution.

### Predictor 1: gpt2m global prior (free)
Condition gpt2m on the true preceding context (a sliding 20-frame, 2580-token BOS-prefixed window), take the 1025-way softmax and renormalise away the deterministic BOS slot, giving a distribution over 1024 indices. This alone is the method behind the leaderboard's 4.0.

### Predictor 2: embedding-space adaptive linear predictor
- Extract the VQ codebook embedding matrix `C` (1024 x d) from the VQ-VAE weights.
- Map recent tokens to their embeddings. An LMS or RLS adaptive linear predictor (classical adaptive LPC) predicts the next embedding `e_hat` from a tapped delay line of past embeddings. The taps were meant to cover the temporal predecessor (same grid position, previous frame) and the in-frame spatial neighbours (left, up), the structure section 2 shows is informative. As built, `lms_predictor.py` is NLMS over the previous `n_taps` tokens in raster order (left neighbours only), and no RLS variant was written.
- Convert to a categorical distribution by a distance softmax over the codebook: `P2(k) ~ exp(-||e_hat - C_k||^2 / tau)`.
- This uses the continuous codebook geometry, which gpt2m's index softmax never sees directly. The filter taps adapt online; `tau` is a fixed hyperparameter.

### Predictor 3: per-position RC leaky-integrator prior
One adaptive frequency estimate per (row, col) of the 8 x 16 grid, updated as an exponential moving average, which is a first-order RC low-pass filter: `p <- (1 - alpha) * p + alpha * onehot(symbol)`. Motivated by the 4.6 to 8.7 bit spread across positions.

### Fusion: a logistic mixer, which is LMS in the logit domain
Combine `log P1, log P2, log P3` with weights updated online by gradient descent on the coding (log) loss, which is a Widrow-Hoff LMS filter on a linear combiner. The mixer weights are keyed by grid position, so each cell learns its own trust in gpt2m against the adaptive predictors. The fused distribution is quantized to integers (section 5) and fed to the arithmetic coder.

### Dropped from the original proposal
- Fourier or DCT on token indices: the indices are arbitrary labels with a near-flat 9.70-bit 0-order entropy (section 2), so they have no meaningful spectrum.
- A custom Perceiver context compressor: custom weights cost zip bytes, and gpt2m already supplies the 20-frame conditional.

---

## 4. Components and interfaces

Small, single-purpose and independently testable. The pure units have no GPU dependency and can be built and tested on their own.

| Unit | Responsibility | Key interface | Depends on | Pure? |
|------|----------------|---------------|------------|-------|
| `model_runtime` | gpt2m forward pass to a per-token softmax, batched, BOS renormalised. | `probs1(window) -> f32[1024]` | torch, gpt2m | no (GPU) |
| `codebook` | Load the VQ-VAE codebook `C` (1024 x d). | `embeddings() -> f32[1024,d]` | VQ-VAE weights | no |
| `lms_predictor` | Adaptive embedding linear predictor with a distance-softmax output. | `predict(history)->f32[1024]`, `update(sym)` | `codebook` | yes |
| `rc_prior` | Per-position leaky-integrator frequency model. | `predict(pos)->f32[1024]`, `update(pos,sym)` | none | yes |
| `mixer` | Position-keyed logistic mixer (LMS logit fusion). | `mix([p1,p2,p3],pos,state)->(f32[1024],state)` | none | yes |
| `range_coder` | Lossless arithmetic or rANS encode and decode on integer frequencies. | `encode(sym,freqs)`, `decode(freqs)->sym` | none | yes |
| `codec` | Orchestrate predictors, mixer and coder for one example. | `compress(tokens)->bytes`, `decompress(bytes)->tokens` | all | no |
| `submission` | Map the codec over the dataset, build the zip with `decompress.py`. | CLI | datasets, codec | no |

As built, the real model runs through onnxruntime (`gpt2m_onnx.py`, `submission_codec.py`) rather than torch, the codebook loader became `extract_codebook.py`, and the range coder is a Subbotin-style range coder rather than rANS.

---

## 5. Lossless invariants (critical)

Lossless decoding requires the decoder to rebuild exactly the fused probability stream the encoder used.
- gpt2m runs with fixed weights and dtype, deterministic kernels and no sampling (full distributions only).
- The fused distribution is converted to an integer frequency table by a fixed quantization rule, identically on both sides. The coder consumes only the integer table, so encoder and decoder agree as long as they compute the same floats; small float differences across hardware can still change the table, which is what the cross-machine test was for.
- All three predictors and the mixer are stepped in the same order with the same online updates on both sides. Updates use the decoded symbol (causal), so encoder and decoder stay in lockstep.
- A cross-machine test (encode on AWS, decode locally) was planned to catch float drift (section 6). It was not run.

---

## 6. Verification

- Round trip (gating): `decompress(compress(x)) == x` exactly, synthetic data first, then real clips.
- Floor check: Predictor-1-only bits/token reproduces about 4.0 on a subset, as a sanity check of the method.
- Ablation: add Predictor 3, then Predictor 2, then the position-keyed mixer, and record the bits/token change of each. Confirm a projected full-dataset score above 4.0 before any expensive full run. (`metrics.ablation` does not isolate the components as written; the fusion result in findings.md comes from the sweep.)
- Determinism: encode on AWS, decode on a different machine, check exact reconstruction. Not run.
- Final: `./compression/evaluate.sh` on the produced zip; the printed rate is the score of record. Not run.

---

## 6.5 Local validation harness (stage 0, before any cloud spend)

No AWS spend until a local, GPU-free harness gives a trustworthy projected bits/token and tuned hyperparameters. The principle: run the heavy model once, cache its per-token distributions, then sweep cheap hyperparameters by replaying the cache.

- Two model backends behind one `ModelRuntime` interface: (a) a realistic synthetic model (no torch) for the machinery and tuning; (b) a cached-real backend that replays gpt2m distributions precomputed once on a small subset.
- Realistic synthetic generator and model: generate tokens whose structure matches section 2 (per-position bias, about 35% temporal copies, spatial-neighbour correlation) plus a slow non-stationary per-position drift the base model cannot see. The synthetic model knows only the stationary part, so the RC prior can track the drift and the LMS can recover the spatial residual, which gives fusion a measurable gain to find. This tests the fusion machinery without gpt2m; on this synthetic data fusion does beat the model (`test_fusion_beats_model_only`), as designed.
- Sweep engine: grid search over `{rc.alpha, mixer.lr, lms.(mu, tau, n_taps), quantize precision}`, replaying a cached distribution set, and reporting the best bits/token and the projected score against 4.0.
- Decision gate: stage 0 must show fusion bits/token below model-only bits/token on cached data, and a projected ratio that justifies the AWS run. The subset size for the cached-real run is chosen after the machinery is validated on synthetic data. (On the real cache, fusion failed this gate; findings.md, section 2.3.)

## 7. AWS execution (planned, blocked)

- Instance: `g5.xlarge`, one NVIDIA A10G with 24 GB of GPU memory ([AWS](https://aws.amazon.com/ec2/instance-types/g5/)). gpt2m has about 307M parameters (24 layers, width 1024, from the `GPTConfig` in [utils/gpt.py][gpt27]), about 0.6 GB in fp16.
- Cost: to be measured by a roughly 50-clip prototype (throughput and bits/token) before any full run. No estimate is given here because GPU throughput was never measured.
- Local role: the GTX 1650 (4 GB) machine for code and correctness, plus the cross-machine decode test.
- Prerequisites: AWS CLI installed and configured, the spend accepted, and possibly a g5 vCPU quota increase. The account turned out to be free-tier only, which blocked GPU launches (findings.md, section 4.1).

---

## 8. Build sequence (checkpoints)

1. Environment on AWS g5: torch and CUDA, load gpt2m, reproduce a forward pass matching `utils/gpt.py`. Extract the VQ codebook `C`.
2. `range_coder` and a Predictor-1-only `codec`; round trip passes; measure the floor (about 4.0).
3. Add `rc_prior` and the position-keyed `mixer`; re-measure.
4. Add `lms_predictor` (embedding LMS); ablate; confirm a projected score above 4.0 on a subset.
5. Full-dataset encode on AWS; build the zip; cross-machine decode; `evaluate.sh`.

The pure units (`range_coder`, `rc_prior`, `mixer`, `lms_predictor`) share no state, so they can be built and tested independently; `model_runtime` and `codec` integrate them.

Status: steps 2 to 4 were done locally, with onnxruntime in place of the AWS environment. Fusion did not beat gpt2m, and the gpt2m-only full run (step 5) has not been done.

---

## 9. Risks and open items

- Beating 4.0 was not guaranteed. Predictor 1 only matches it, so any win depended on the adaptive components adding a few tenths of a bit per token. Expected beforehand: the position-keyed mixer and RC prior were a likely small win, and the embedding LMS was uncertain and might be subsumed by gpt2m's attention. (Outcome: neither helped; findings.md, section 2.3.)
- Float non-determinism across encode and decode hardware: the integer table (section 5) makes the coder exact, but the floats feeding it must still match. The cross-machine test was not run.
- Codebook extraction from the VQ-VAE weights: low risk; `d` turned out to be 256 (findings.md, section 1).
- Throughput on an A10G was unknown; the prototype was meant to measure it.
- Literature search: planned, not completed.

[vqvae]: https://github.com/commaai/commavq/blob/623a27f85689de4db3822cc81654cdf4db1b3da0/utils/vqvae.py#L248
[decomp]: https://github.com/commaai/commavq/blob/623a27f85689de4db3822cc81654cdf4db1b3da0/compression/decompress.py#L15
[prepare]: https://github.com/commaai/commavq/blob/623a27f85689de4db3822cc81654cdf4db1b3da0/nanogpt/prepare.py#L16-L30
[gpt27]: https://github.com/commaai/commavq/blob/623a27f85689de4db3822cc81654cdf4db1b3da0/utils/gpt.py#L27
[gpt33]: https://github.com/commaai/commavq/blob/623a27f85689de4db3822cc81654cdf4db1b3da0/utils/gpt.py#L33
[gpt117]: https://github.com/commaai/commavq/blob/623a27f85689de4db3822cc81654cdf4db1b3da0/utils/gpt.py#L117
[evaluate]: https://github.com/commaai/commavq/blob/623a27f85689de4db3822cc81654cdf4db1b3da0/compression/evaluate.py#L25
[rules]: https://github.com/commaai/commavq/blob/623a27f85689de4db3822cc81654cdf4db1b3da0/compression/README.md
