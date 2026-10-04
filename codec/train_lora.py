"""LoRA-finetune gpt2m on commaVQ and gate on held-out cross-entropy.

For a GPU box. Self-contained (no codec imports): downloads the HF model and a data
subset, LoRA-finetunes, then compares held-out bits/token of the finetuned model with
the frozen base and prints a ship / do-not-ship verdict. Saves the adapter only if it
helps. The LoRA run reported in docs/findings.md used the peft-free variant in
notebooks/commavq_lora_colab.ipynb.

Usage:
  pip install torch transformers peft datasets
  python codec/train_lora.py --train-segments 8 --eval-segments 2 --steps 400 --rank 8
"""
import argparse, math, time
import numpy as np

# Copies of codec.submission_codec values, so this runs on a GPU box without onnxruntime.
BOS = 1024
TPF = 129          # tokens per frame incl BOS
GRID = 128
BLOCK_FRAMES = 20  # gpt2m context = 20 frames
BLOCK = BLOCK_FRAMES * TPF  # 2580
RAW_BITS = 10      # bits per raw token; a ratio is RAW_BITS / bits-per-token

def log(*a): print(*a, flush=True)

def load_segments(n, split_files):
    from datasets import load_dataset
    ds = load_dataset("commaai/commavq", data_files={"train": split_files})["train"]
    segs = []
    for i in range(min(n, ds.num_rows)):
        t = np.array(ds[i]["token.npy"]).reshape(-1, GRID).astype(np.int64)  # (frames,128)
        segs.append(t)
    return segs

def to_blocks(segs):
    """Each segment -> BOS-prefixed flat stream -> non-overlapping BLOCK-length chunks."""
    blocks = []
    for t in segs:
        flat = np.concatenate([np.concatenate([[BOS], fr]) for fr in t])  # 129/frame
        n = (flat.shape[0] // BLOCK) * BLOCK
        for i in range(0, n, BLOCK):
            blocks.append(flat[i:i + BLOCK])
    return np.stack(blocks) if blocks else np.zeros((0, BLOCK), dtype=np.int64)

def content_bits(model, blocks, device, batch=4):
    """Mean bits/token over CONTENT positions (exclude BOS slots) on given blocks."""
    import torch
    model.eval()
    tot_nll, tot_n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(blocks), batch):
            ids = torch.tensor(blocks[i:i + batch], dtype=torch.long, device=device)
            logits = model(ids).logits[:, :-1, :]          # predict next
            tgt = ids[:, 1:]
            logp = torch.log_softmax(logits.float(), dim=-1)
            nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
            mask = (tgt != BOS)                            # score content tokens only
            tot_nll += float(nll[mask].sum()); tot_n += int(mask.sum())
    return tot_nll / tot_n / math.log(2)

def main():
    import torch
    from transformers import GPT2LMHeadModel
    from peft import LoraConfig, get_peft_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="commaai/commavq-gpt2m")
    ap.add_argument("--train-segments", type=int, default=8)
    ap.add_argument("--eval-segments", type=int, default=2)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--split-files", nargs="+", default=["data-0000.tar.gz", "data-0001.tar.gz"])
    ap.add_argument("--out", default="commavq_lora_adapter")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}")

    n = args.train_segments + args.eval_segments
    segs = load_segments(n, args.split_files)
    train_blocks = to_blocks(segs[:args.train_segments])
    eval_blocks = to_blocks(segs[args.train_segments:])
    log(f"train blocks={len(train_blocks)} eval blocks={len(eval_blocks)} (block={BLOCK})")

    base = GPT2LMHeadModel.from_pretrained(args.model).to(device)
    base_bits = content_bits(base, eval_blocks, device, args.batch)
    log(f"BASE held-out bits/token: {base_bits:.4f}  (ratio {RAW_BITS/base_bits:.3f})")

    lcfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                      target_modules=["c_attn"], task_type="CAUSAL_LM")
    model = get_peft_model(base, lcfg)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    model.train()
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(args.steps):
        idx = rng.integers(0, len(train_blocks), size=args.batch)
        ids = torch.tensor(train_blocks[idx], dtype=torch.long, device=device)
        out = model(ids, labels=ids)       # causal LM loss
        out.loss.backward(); opt.step(); opt.zero_grad()
        if (step + 1) % 50 == 0:
            log(f"  step {step+1}/{args.steps} loss {out.loss.item():.4f} "
                f"({(time.time()-t0)/(step+1):.2f}s/step)")

    ft_bits = content_bits(model, eval_blocks, device, args.batch)
    # adapter ship size estimate
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter_mb = trainable * 2 / 1e6   # fp16 adapter
    gain = base_bits - ft_bits
    log("=== LoRA GATE (held-out) ===")
    log(f"base bits/token     : {base_bits:.4f}  (ratio {RAW_BITS/base_bits:.3f})")
    log(f"finetuned bits/token: {ft_bits:.4f}  (ratio {RAW_BITS/ft_bits:.3f})")
    log(f"gain                : {gain:.4f} bits/token ({100*gain/base_bits:.2f}%)")
    log(f"adapter (~fp16)     : {adapter_mb:.2f} MB")
    # break-even over full dataset: bits saved across 768M tokens vs adapter bytes
    saved_mb = gain * 768e6 / 8 / 1e6
    log(f"full-dataset bytes saved est: {saved_mb:.1f} MB  vs adapter {adapter_mb:.2f} MB")
    verdict = "SHIP (net win)" if saved_mb > adapter_mb and gain > 0 else "DO NOT SHIP (no net gain)"
    log(f"VERDICT: {verdict}")
    if gain > 0:
        model.save_pretrained(args.out)
        log(f"adapter saved to {args.out}")

if __name__ == "__main__":
    main()
