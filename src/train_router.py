"""Train the dispatch model for the ESP32 model-dispatch cluster.

The router learns to assign queries to the best expert(s) based on
evaluation data. For each query in the training set, we know which expert
produced the lowest loss — that becomes the router's target label.

Training data format: JSONL with fields:
    {"text": "query text", "best_expert": 0, "expert_losses": [2.1, 1.3, 2.8]}

The router is trained on (tokenized_query, best_expert_label) pairs.

Usage:
    python train_router.py --epochs 50 --batch-size 16
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Import the vendored PLE toolkit for tokenizer compatibility
TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "firmware", "tools")
sys.path.insert(0, TOOLS)

from router import RouterModel, RouterConfig, param_count


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
RUNS = os.path.join(PROJECT, "runs")
DATA = os.path.join(PROJECT, "data")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─── Simple tokenizer (byte-pair or char-level fallback) ─────────────────────

def load_tokenizer():
    """Load the same tokenizer used by the experts.

    The project uses a BPE tokenizer with vocab 32768 (data/bpe32768.json).
    We reuse the same vocab so the router sees the same token IDs.
    """
    # Try to load the shipped tokenizer
    tok_path = os.path.join(ESP32_AI, "..", "data", "tokenizer.json")
    if os.path.exists(tok_path):
        # TODO: load actual tokenizer when available
        pass

    # Fallback: simple char-level tokenizer mapped into vocab space
    # Placeholder path — gen_router_data_v2.py handles the real BPE path
    return None


def fnv1a(word: str, vocab_size: int) -> int:
    """Deterministic 32-bit FNV-1a word hash, folded into vocab space.

    Replaces Python's hash() (per-process randomized) so the SAME token ids are
    reproducible in training, on-host eval, AND in C on the ESP32 router node.
    Must stay bit-identical to fnv1a() in firmware/md_router/ and
    md_server.py::simple_tokenize.
    """
    h = 0x811C9DC5
    for byte in word.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h % vocab_size


def tokenize(text, tokenizer=None, vocab_size=32768, seq_len=64):
    """Tokenize text for the router.

    Word-level FNV-1a hash into vocab space (deterministic across processes,
    languages, and hosts — see fnv1a docstring).
    """
    if tokenizer is not None:
        ids = tokenizer.encode(text)
    else:
        words = text.lower().split()
        ids = [fnv1a(w, vocab_size) for w in words]

    # Truncate or pad to seq_len
    if len(ids) > seq_len:
        ids = ids[:seq_len]
    else:
        ids = ids + [0] * (seq_len - len(ids))  # pad with 0
    return ids


# ─── Data loading ────────────────────────────────────────────────────────────

class RouterBatcher:
    """Yields batches of (tokenized_query, best_expert_label)."""

    def __init__(self, data_path, batch_size, seq_len, n_experts, device, is_val=False):
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Router data not found: {data_path}\n"
                f"Run generate_router_data.py first to create expert eval data."
            )

        self.examples = []
        with open(data_path) as f:
            for line in f:
                ex = json.loads(line)
                self.examples.append(ex)

        self.bs = batch_size
        self.sl = seq_len
        self.n_experts = n_experts
        self.device = device
        self.rng = np.random.default_rng(1234 if is_val else None)

        # Pre-tokenize all queries
        self.tokenized = []
        for ex in self.examples:
            ids = tokenize(ex["text"], seq_len=seq_len)
            label = ex["best_expert"]
            self.tokenized.append((ids, label))

        print(f"Loaded {len(self.tokenized)} examples from {data_path}")

    def __len__(self):
        return len(self.tokenized)

    def __call__(self):
        idxs = self.rng.integers(0, len(self.tokenized), self.bs)
        batch_x = np.array([self.tokenized[i][0] for i in idxs], dtype=np.int64)
        batch_y = np.array([self.tokenized[i][1] for i in idxs], dtype=np.int64)
        return (torch.from_numpy(batch_x).to(self.device),
                torch.from_numpy(batch_y).to(self.device))


# ─── Training loop ───────────────────────────────────────────────────────────

def train_router(epochs, batch_size, lr, n_experts, seq_len, seed):
    cfg = RouterConfig(
        vocab_size=32768,
        d_model=64,
        n_layers=2,
        n_heads=2,
        n_experts=n_experts,
        seq_len=seq_len,
        pool="mean",
    )

    print(f"\n{'='*60}")
    print(f"Training Router")
    print(f"  params: {param_count(cfg):,}")
    print(f"  n_experts: {n_experts}")
    print(f"  seq_len: {seq_len}")
    print(f"  epochs: {epochs}, batch: {batch_size}, lr: {lr}")
    print(f"{'='*60}\n")

    torch.manual_seed(seed)
    device = get_device()
    os.makedirs(RUNS, exist_ok=True)

    model = RouterModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Router params: {n_params:,}")

    # Load data
    train_path = os.path.join(DATA, "router_train.jsonl")
    val_path = os.path.join(DATA, "router_val.jsonl")

    if not os.path.exists(train_path):
        print(f"❌ Router training data not found: {train_path}")
        print(f"   Run generate_router_data.py first.")
        return

    train_b = RouterBatcher(train_path, batch_size, seq_len, n_experts, device, is_val=False)
    val_b = RouterBatcher(val_path, batch_size, seq_len, n_experts, device, is_val=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_b))

    name = f"router-{n_experts}exp-s{seed}"
    best_val = float("inf")
    t0 = time.time()
    history = []

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for _ in range(len(train_b)):
            x, y = train_b()
            logits, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for _ in range(max(1, len(val_b) // batch_size)):
                x, y = val_b()
                logits, loss = model(x, y)
                val_losses.append(loss.item())
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        val_acc = correct / total

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        best_val = min(best_val, val_loss)
        print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| acc {val_acc:.1%} | best {best_val:.4f} | {time.time()-t0:.0f}s",
              flush=True)

    # Save
    result = {
        "n_experts": n_experts,
        "seed": seed,
        "config": cfg.__dict__,
        "params": n_params,
        "best_val": best_val,
        "final_val_acc": history[-1]["val_acc"],
        "epochs": epochs,
        "wall_seconds": time.time() - t0,
        "history": history,
    }
    with open(os.path.join(RUNS, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    torch.save({"cfg": cfg.__dict__, "state": model.state_dict()},
               os.path.join(RUNS, f"{name}.pt"))
    print(f"\n{name} DONE — val_acc={result['final_val_acc']:.1%} "
          f"({time.time()-t0:.0f}s)")
    print(f"Saved: {os.path.join(RUNS, f'{name}.pt')}")

    return result


def main():
    ap = argparse.ArgumentParser(description="Train the dispatch model")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-experts", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_router(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_experts=args.n_experts,
        seq_len=args.seq_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()