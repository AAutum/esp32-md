#!/usr/bin/env python3
"""Generate router training data using the REAL BPE tokenizer + domain-sampled
queries from the actual expert training corpora (not just hand-written pools).

For each query: compute loss under all 3 experts, label = argmin loss.
Matches the format train_router.py expects (JSONL: text, best_expert, expert_losses, soft_labels).
"""
import argparse, json, os, random, sys
import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, os.path.join(ROOT, "firmware", "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "firmware", "tools"))
from model import Config, TinyLM
from tokenizers import Tokenizer as TK

RUNS = os.path.join(ROOT, "checkpoints")
BPE = os.path.join(ROOT, "data", "bpe32768.json")
DOMAIN_FILES = {
    "code": os.path.join(ROOT, "data", "code_train.bin"),
    "reasoning": os.path.join(ROOT, "data", "reasoning_train.bin"),
    "general": os.path.join(ROOT, "data", "general_train.bin"),
}
DOMAINS = ["general", "code", "reasoning"]  # index order = expert id order


def load_expert(domain, seed=0):
    path = os.path.join(RUNS, f"expert-{domain}-s{seed}.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**ckpt["cfg"])
    model = TinyLM(cfg)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, cfg


@torch.no_grad()
def compute_expert_loss(model, token_ids, device="cpu"):
    """CE loss of the expert on this token sequence (mean over positions)."""
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    logits, _ = model(ids)  # TinyLM.forward returns (logits, loss=None)
    # predict next token: logits[:, :-1] vs ids[:, 1:]
    ce = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)).float(),
        ids[:, 1:].reshape(-1), reduction="mean")
    return float(ce)


def decode_window(tok, binpath, nbytes):
    """Read a random uint16 window from a domain train bin, decode to text."""
    b = np.memmap(binpath, dtype=np.uint16, mode="r")
    lo = random.randint(0, max(0, len(b) - nbytes - 1))
    ids = [int(x) for x in b[lo:lo + nbytes]]
    # decode ids -> text via BPE (skip out-of-vocab gracefully)
    try:
        text = tok.decode(ids)
    except Exception:
        text = ""
    if not text or len(text) < 10:
        return None
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=800)
    ap.add_argument("--win", type=int, default=48, help="query token window size")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "router_train.jsonl"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--handwritten-frac", type=float, default=0.3,
                    help="fraction of samples from hand-written query pools (realistic prompts)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    tok = TK.from_file(BPE)
    print(f"[bpe] vocab {tok.get_vocab_size()}")

    experts = {}
    for d in DOMAINS:
        experts[d] = load_expert(d)
        print(f"[expert] {d} loaded")

    # hand-written pools from original generator (import)
    import generate_router_data as grd
    pools = {"code": grd.CODE_QUERIES, "reasoning": grd.REASONING_QUERIES, "general": grd.GENERAL_QUERIES}

    samples = []
    n_hw = int(args.n_samples * args.handwritten_frac)
    n_corpus = args.n_samples - n_hw
    for i in range(args.n_samples):
        if i < n_hw:
            domain = random.choice(DOMAINS)
            query = random.choice(pools[domain])
        else:
            domain = random.choice(DOMAINS)
            query = decode_window(tok, DOMAIN_FILES[domain], args.win)
            if query is None:
                continue
        ids = tok.encode(query).ids[:256]
        if len(ids) < 4:
            continue
        losses = [compute_expert_loss(experts[d][0], ids) for d in DOMAINS]  # (model, cfg) tuple
        best = int(np.argmin(losses))
        inv = [1.0 / (l + 1e-8) for l in losses]
        soft = [x / sum(inv) for x in inv]
        samples.append({"text": query, "best_expert": best,
                        "expert_losses": losses, "soft_labels": soft})
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{args.n_samples}] last losses " +
                  " ".join(f"{l:.3f}" for l in losses))

    random.shuffle(samples)
    n_val = int(len(samples) * args.val_frac)
    val_s, train_s = samples[:n_val], samples[n_val:]
    with open(args.out, "w") as f:
        for s in train_s:
            f.write(json.dumps(s) + "\n")
    val_path = args.out.replace("train", "val")
    with open(val_path, "w") as f:
        for s in val_s:
            f.write(json.dumps(s) + "\n")
    # label balance
    from collections import Counter
    print("train labels:", Counter(s["best_expert"] for s in train_s))
    print(f"wrote {len(train_s)} train -> {args.out}")
    print(f"wrote {len(val_s)} val   -> {val_path}")


if __name__ == "__main__":
    main()