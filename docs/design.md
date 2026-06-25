# commaVQ Lossless Compression — Design

**Date:** 2026-06-14
**Goal:** Beat the commaVQ leaderboard SOTA (score **4.0**, "arithmetic coding with commavq-gpt2m") on lossless compression of the first two dataset splits (5,000 minutes, ~915 MB of tokens).
**Contribution identity (ECE):** a unique angle grounded in *classic adaptive signal processing* — fuse the frozen GPT world-model prior with classical online **adaptive filters** (Widrow-Hoff LMS / RLS, RC leaky-integrators), framed via the equivalence "context mixing = adaptive filtering."

---

## 1. Problem & Scoring

- **Data:** splits `data-0000` + `data-0001` of `commaai/commavq`. Each example is a `token.npy` of shape `(1200, 8, 16)` int — 128 VQ-VAE codebook indices per frame, **10 bits/token** (vocab 1024, +1 BOS = 1025). Total ≈ **768M tokens**.
- **Token layout (verified, `vqvae.py:248`, `decompress.py:15`):** frame grid is 8 rows × 16 cols, flattened **row-major (raster)** to 128 tokens. For the world model each frame is prefixed by `BOS=1024`, giving **129 tokens/frame** (`prepare.py:16,30`; `gpt.py:33`). Model context `block_size = 20*129 = 2580` (20 frames), learned absolute positions — BOS must land at positions 0,129,258,…
- **Score** (`compression/evaluate.py:25`): `score = raw_bytes / zip_size`, `raw_bytes = num_rows · 1200 · 128 · 10 / 8`. The zip contains the compressed payload **plus `decompress.py`** — anything shipped counts against the score.
- **Target:** `score > 4.0`, i.e. effective **< 2.5 bits/token**.
- **Lossless gate:** `decompress.py` must reconstruct every `token.npy` exactly; one mismatch fails the submission.

**Free vs. paid bytes (two-part coding):** `gpt2m` and the VQ-VAE codebook are downloadable from HuggingFace / `pip`-installable, so their weights are **free** (0 bytes against score). Any *custom-trained* weights would count. The chosen design ships **no learned weights** — all adaptive components are recomputed deterministically at decode (see §5).

---

## 2. Empirical Grounding (measured on `examples/tokens.npy`, one 1200-frame clip)

| Signal | bits/token | Implication |
|---|---|---|
| 0-order entropy | **9.70** | ≈ full 10 bits, all 1024 codes used → no marginal-frequency win; **kills Fourier/DCT on indices** |
| H(token \| same-pos prev frame) | **4.31** | 35% of tokens are exact temporal copies — dominant structure |
| H(token \| left / up neighbor) | ~6.55 | spatial neighbors help, but less than temporal |
| per-(row,col) entropy | **4.60 → 8.73** (mean 7.68) | strong spatial non-uniformity → per-position adaptive prior is justified |

**Headroom (honest):** gpt2m's 20-frame joint context already operates near ~2.5 bits/token; every *single-variable* entropy above sits higher, so gpt2m is doing the heavy lifting. Realistic incremental gain from adaptive fusion ≈ **0.1–0.4 bits/token → ratio ~4.2–4.6**, not a step change. A larger jump would require a better world model, which is out of scope. The design is built to *measure* each component's marginal contribution (ablation), which is also the evidence an ECE writeup needs.

---

## 3. Approach — GPT prior × classic adaptive filters (three predictors → one fused distribution)

For every content token we form three predictive distributions over the 1024 codebook indices, fuse them, and arithmetic-code the true symbol under the fused distribution.

### Predictor 1 — gpt2m global prior (free)
Condition gpt2m on the true preceding context (sliding 20-frame / 2580-token BOS-prefixed window), take the 1025-way softmax, **renormalize away the deterministic BOS slot** → distribution over 1024 indices. This alone is the published ~4.0 floor.

### Predictor 2 — embedding-space adaptive linear predictor (the novel ECE core)
- Extract the VQ codebook embedding matrix `C ∈ ℝ^{1024×d}` from the VQ-VAE weights (new dependency, §7).
- Map recent tokens to their embeddings; an **LMS/RLS adaptive linear predictor** (classical adaptive LPC / Wiener filter) predicts the next embedding `ê` from a tapped-delay line of past embeddings — taps cover the temporal predecessor (same grid position, previous frame) and in-frame spatial neighbors (left, up), the structure §2 shows is informative.
- Convert to a categorical distribution by distance-softmax over the codebook: `P₂(k) ∝ exp(−‖ê − C_k‖² / τ)`.
- Exploits the **continuous codebook geometry** gpt2m's index-softmax never uses directly. Filter taps adapt online via LMS; `τ` is a fixed hyperparameter.

### Predictor 3 — per-position RC leaky-integrator prior (cheap)
One adaptive frequency estimate per (row,col) of the 8×16 grid, updated as an exponential moving average — a **first-order RC low-pass filter**: `p ← (1−α)·p + α·onehot(symbol)`. Justified by the 4.6→8.7-bit per-position spread.

### Fusion — logistic mixer = LMS in the logit domain
Combine `logit(P₁), logit(P₂), logit(P₃)` with weights updated online by gradient descent on the coding (log) loss — mathematically a **Widrow-Hoff LMS adaptive filter** on a linear combiner. Mixer weights are **keyed by spatial position** so each grid cell learns its own gpt2m-vs-adaptive trust. The fused distribution is integer-quantized (§5) and fed to the arithmetic coder.

### Dropped from the original proposal
- **Fourier/DCT on token indices** — refuted by the 9.70-bit flat 0-order entropy (§2); arbitrary labels have no spectrum.
- **Custom Perceiver context compressor** — custom weights cost zip bytes; gpt2m already supplies the 20-frame conditional.

---

## 4. Components & Interfaces

Small, single-purpose, independently testable. Pure units have no GPU dependency and are the parallel-build targets.

| Unit | Responsibility | Key interface | Depends on | Pure? |
|------|----------------|---------------|------------|-------|
| `model_runtime` | gpt2m forward → per-token softmax, batched, BOS-renormalized. | `probs1(window) -> f32[1024]` | torch, gpt2m | no (GPU) |
| `codebook` | Load VQ-VAE codebook `C` (1024×d). | `embeddings() -> f32[1024,d]` | VQ-VAE weights | no |
| `lms_predictor` | Adaptive embedding linear predictor → distance-softmax dist. | `predict(history)->f32[1024]`, `update(sym)` | `codebook` | yes |
| `rc_prior` | Per-position leaky-integrator frequency model. | `predict(pos)->f32[1024]`, `update(pos,sym)` | none | yes |
| `mixer` | Position-keyed logistic mixer (LMS logit fusion). | `mix([p1,p2,p3],pos,state)->(f32[1024],state)` | none | yes |
| `range_coder` | Lossless arithmetic/rANS encode & decode on integer freqs. | `encode(sym,freqs)`, `decode(freqs)->sym` | none | yes |
| `codec` | Orchestrate predictors → mixer → coder for one example. | `compress(tokens)->bytes`, `decompress(bytes)->tokens` | all | no |
| `submission` | Map codec over dataset, build zip with `decompress.py`. | CLI | datasets, codec | no |

---

## 5. Lossless Invariants (critical)

Lossless ⇔ decoder reconstructs the **bit-identical** fused-probability stream the encoder used.
- gpt2m run with fixed weights/dtype, deterministic kernels, **no sampling** (full distributions only).
- The **fused** distribution is converted to an integer frequency table by a fixed quantization rule, identically on both sides — float softmax is not bit-reproducible across hardware, so the coder consumes only the integer table.
- All three predictors and the mixer are stepped in the exact same order with the exact same online updates on both sides; updates use the *decoded* symbol (causal), so encoder and decoder stay in lockstep.
- Cross-machine test (encode on AWS, decode locally) guards against float drift (§6).

---

## 6. Verification

- **Round-trip (gating):** `decompress(compress(x)) == x` exactly — synthetic first, then real clips.
- **Floor check:** Predictor-1-only bits/token reproduces ≈4.0 on a subset (method sanity).
- **Ablation / edge check:** add Predictor 3, then Predictor 2, then position-keyed mixer; record bits/token delta of each. Confirm projected full-dataset score > 4.0 before any expensive full run. (Ablation table = ECE writeup evidence.)
- **Determinism:** encode on AWS, decode on a different machine; exact reconstruction.
- **Final:** `./compression/evaluate.sh` on the produced zip; printed rate is the score of record.

---

## 6.5 Pre-AWS Local Validation Harness (Stage 0 — gate before any cloud spend)

No AWS spend until a local, GPU-free harness produces a trustworthy projected bits/token and tuned hyperparameters. Built on the decoupling principle: **run the heavy model once, cache its per-token distributions, then sweep cheap hyperparameters by replaying the cache.**

- **Two model backends behind one `ModelRuntime` interface:** (a) a *realistic synthetic* model (no torch) for machinery + tuning; (b) a *cached-real* backend that replays gpt2m distributions precomputed once on a small subset.
- **Realistic synthetic generator + model:** generate tokens whose structure matches §2 — per-position bias, temporal copies (~35%), spatial-neighbor correlation — **plus a slow non-stationary per-position drift the base model cannot see**. The synthetic *model* knows the stationary part only; the RC leaky-integrator tracks the drift and the LMS recovers the spatial-embedding residual → a measurable, interpretable fusion gain. This validates the adaptive-fusion thesis without gpt2m.
- **Sweep engine:** grid/random search over `{rc.alpha, mixer.lr, lms.(mu, tau, n_taps), quantize precision}`, replaying a cached distribution set; reports best bits/token and projected score vs 4.0.
- **Decision gate:** Stage-0 must show fusion bits/token < model-only bits/token on cached data and a projected ratio that justifies the AWS run. Subset size for the cached-real run is chosen after Stage-1 machinery is validated on synthetic data.

## 7. AWS Execution

- **Instance:** `g5.xlarge` (1× NVIDIA A10G, 24 GB), ~$1/hr. Fits gpt2m (~350M params) + batched 1-min clips.
- **Cost estimate:** full 768M-token encode ≈ **20–50 GPU-hours ≈ $20–60** + storage/egress; validated by a ~50-clip prototype measuring throughput and bits/token before the full run.
- **Local role:** GTX 1650 (4 GB) for code/correctness + cross-machine decode test only.
- **User prerequisites:** install AWS CLI, `aws configure` (assistant will not handle secrets), accept the spend, possibly request a g5 vCPU quota increase.

---

## 8. Build Sequence (checkpoints)

1. Env on AWS g5: torch+CUDA, load gpt2m, reproduce a forward pass matching `utils/gpt.py`. Extract VQ codebook `C`.
2. `range_coder` + Predictor-1-only `codec`; round-trip passes; measure floor (~4.0).
3. Add `rc_prior` + position-keyed `mixer`; re-measure (safe ratio win).
4. Add `lms_predictor` (embedding LMS); ablate; confirm >4.0 projected on subset.
5. Full-dataset encode on AWS; build zip; cross-machine decode; `evaluate.sh`.

Pure units (`range_coder`, `rc_prior`, `mixer`, `lms_predictor`) are built in parallel (one agent each) since they share no state; `model_runtime`/`codec` integrate them.

---

## 9. Risks & Open Items

- **Beating 4.0 is not guaranteed.** Predictor 1 only matches it; the win rides on the adaptive components delivering 0.1–0.4 bits/token. Honest confidence: position-keyed mixer + RC prior = "likely small win"; embedding-LMS = "novel but marginal contribution uncertain — may be subsumed by gpt2m's internal attention."
- **Float non-determinism** across encode/decode hardware — mitigated by integer-frequency quantization (§5), tested cross-machine.
- **Codebook extraction** from VQ-VAE weights — low risk, must confirm `d` and exact index→embedding map.
- **Throughput** on A10G may exceed the cost estimate; the prototype gates this.
- **Literature pass pending:** deep-research workflow (`whyd3l4lq`) to be folded in — may surface prior art on neural-LM + adaptive-filter / DSP fusion and refine hyperparameters; citations appended on completion.
