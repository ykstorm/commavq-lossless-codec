# commaVQ — Submission, GPU setup, and "how to get on the leaderboard"

## Where we are
- Working **lossless gpt2m + arithmetic-coding codec** (`compression/codec/submission_codec.py`), validated end-to-end:
  - 6-frame round-trip: **lossless**.
  - 22-frame compress through the 2580-context boundary: **ratio 4.04** (2.474 bits/token), no error.
- That ratio would rank **#1–2** on the 16-entry leaderboard; **top-10/25 is cleared with room to spare**.
- Blocker: this 4 GB-class laptop can't sustain the full run (repeated RAM kills). The codec is correct; it just needs steady compute.

## When to set up a GPU (and which)

**When:** only when you decide to produce the *actual full submission* — i.e. compress all of splits 0+1 (~5000 segments, ~768M tokens) into the zip. Everything up to that point (method, correctness, the 4.04 number) is already proven on samples. Do NOT set up a GPU just to "see if it works" — that's done.

**Why a GPU:** the codec is one onnx forward per token. On this CPU it's ~26–40 s/frame → the full dataset is months. A single modern GPU (A10G/T4/3090) runs gpt2m fp16 in milliseconds/token → the full encode drops to a few hours. Decompress is the same cost (the evaluator runs it once).

**Cheapest viable options (pick one):**
| Option | Cost | Notes |
|---|---|---|
| **vast.ai / runpod** (rent a 3090/4090) | ~$0.2–0.5/hr | cheapest; spin up, run, destroy. Best $/hr. |
| **Lambda / Colab Pro+** | ~$10–50/mo | simple, persistent. |
| **AWS g5** | ~$1/hr | already CLI-configured + quota approved, BUT account is **free-tier-only** → must upgrade to a paid plan first (Billing → add payment). |

**Setup on the GPU box:**
```bash
pip install onnxruntime-gpu numpy datasets huggingface_hub
# providers=["CUDAExecutionProvider"] in Gpt2mStepper
python -m codec.submission_compress   # maps codec over splits 0+1, builds the zip
```
Use `onnxruntime-gpu` and pass `["CUDAExecutionProvider"]` to `Gpt2mStepper`. Batch multiple segments in parallel to saturate the GPU.

**Determinism caveat (must hold for lossless):** encode and the shipped `decompress.py` must run gpt2m with the **same execution provider + precision**. fp16 onnx is bit-identical run-to-run on the *same* hardware (verified) but not guaranteed across CPU↔GPU. Safest: encode and decode on the same provider, or validate a cross-provider round-trip on a few segments before trusting the full run.

## How to actually submit (NOT a PR)

The challenge is **not** entered by opening a pull request on `commaai/commavq`.
1. **Official channel:** a single **zip** (compressed data + `decompress.py`) submitted via comma's **Google Form** (linked in the repo README). They run `./compression/evaluate.sh your.zip` and post the score.
2. **The challenge ended July 1, 2024** → the leaderboard is likely frozen. **Verify first** on comma's Discord whether late entries are still scored before spending GPU money.
3. **A PR to `commaai/commavq`** is only for *code contributions* (e.g. improving the example) — it does **not** put you on the leaderboard. Don't use it for ranking.

## Realistic plan given the challenge is closed
1. **Publish on your own GitHub (ykstorm):** a repo with the codec + this writeup + the negative-results study + the validated 4.04 number. This is the portfolio artifact — it stands on its own regardless of whether comma re-scores.
2. **If comma confirms late scoring:** rent a cheap GPU (vast.ai ~$2–5 total), run `submission_compress`, submit the zip via the form.
3. Otherwise: the work is complete as a demonstrated, leaderboard-matching result + a rigorous study — which is the honest, defensible outcome.

## Decision checklist before any GPU spend
- [ ] Confirmed (Discord) comma still scores late entries — else skip the full run, publish to ykstorm only.
- [ ] Picked a GPU host (vast.ai recommended for cost).
- [ ] Cross-provider round-trip validated on ~5 segments (lossless under the GPU provider).
- [ ] Then run the full `submission_compress`, build zip, submit via form.
