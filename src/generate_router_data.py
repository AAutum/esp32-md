"""Generate router training data by evaluating each expert on sample queries.

For each query in a sample set:
1. Run all 3 experts (compute loss on the query)
2. The expert with the lowest loss = best_expert (the label)
3. Save (query, best_expert, all_losses) as JSONL

This creates the labeled data needed to train the router.

Usage:
    python generate_router_data.py --n-samples 500 --output router_train.jsonl
"""

import argparse
import json
import os
import sys
import random

import torch
import numpy as np

TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "firmware", "tools")
sys.path.insert(0, ESP32_AI)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import Config, TinyLM
from router import RouterConfig


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
RUNS = os.path.join(PROJECT, "runs")
DATA = os.path.join(PROJECT, "data")


def load_expert(domain, seed=0):
    """Load a trained expert model."""
    path = os.path.join(RUNS, f"expert-{domain}-s{seed}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expert checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**ckpt["cfg"])
    model = TinyLM(cfg)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, cfg


# ─── Sample queries for each domain ──────────────────────────────────────────

# These are short text prompts that should clearly belong to one expert domain.
# The router learns from which expert handles each query best.

CODE_QUERIES = [
    "Write a function to sort an array",
    "Implement binary search in Python",
    "Create a hash table lookup",
    "Write a recursive factorial function",
    "Implement a linked list",
    "Write a function to reverse a string",
    "Create a simple HTTP server",
    "Implement quicksort algorithm",
    "Write a function to check if a number is prime",
    "Implement a stack data structure",
    "Write a function to merge two sorted arrays",
    "Create a binary tree traversal function",
    "Implement a queue using arrays",
    "Write a function to find the max element",
    "Create a simple calculator function",
    "Implement depth-first search",
    "Write a function to count words in a string",
    "Create a function to validate email format",
    "Implement a basic encryption function",
    "Write a function to parse JSON",
    "def fibonacci(n): return",
    "for i in range(len(items)): if items[i]",
    "class Node: def __init__(self, value):",
    "try: result = int(text) except ValueError:",
    "import os; path = os.path.join(dir, file)",
    "async def fetch_data(url): response = await",
    "SELECT * FROM users WHERE age > 25 ORDER BY",
    "docker build -t myapp . && docker run -p 8080",
    "git commit -m fix bug in parser && git push origin",
    "lambda x: x * 2 if x > 0 else -x",
]

REASONING_QUERIES = [
    "If A is greater than B and B is greater than C, what is the relation between A and C?",
    "Solve: 3x + 7 = 22. What is x?",
    "All cats are mammals. Whiskers is a cat. What is Whiskers?",
    "If it rains, the ground is wet. The ground is wet. Did it rain?",
    "What is 15% of 240?",
    "If a train travels 60 mph for 2.5 hours, how far does it go?",
    "There are 5 apples. You take away 2. How many do you have?",
    "If today is Monday, what day is it 100 days from now?",
    "A rectangle has length 8 and width 5. What is the area?",
    "If 3 workers can build a wall in 12 days, how long would 6 workers take?",
    "What is the next number: 2, 4, 8, 16, ...?",
    "If you flip 3 coins, what is the probability of all heads?",
    "Solve for x: x^2 - 5x + 6 = 0",
    "A clock shows 3:15. What is the angle between the hands?",
    "If A implies B and B implies C, then A implies what?",
    "What is 7 factorial divided by 5 factorial?",
    "There are 20 people at a party. Everyone shakes hands with everyone. How many handshakes?",
    "If the temperature drops 2 degrees every hour for 5 hours, what is the total drop?",
    "What is the sum of all even numbers from 1 to 100?",
    "If a = 3 and b = 4, what is the length of the hypotenuse?",
    "Therefore we can conclude that the answer must be",
    "Step 1: First we note that. Step 2: Then we observe",
    "The logical conclusion follows from the premises that",
    "By transitive property, if A = B and B = C then",
    "The probability of this event is calculated as follows:",
]

GENERAL_QUERIES = [
    "Once upon a time there was a little robot who",
    "The sun was setting over the quiet village when",
    "She opened the old book and discovered that",
    "The friendly dragon lived in a cave near the",
    "On a bright morning, a child decided to",
    "The old wizard smiled and said to the young apprentice",
    "In a small town by the river, everyone knew that",
    "The cat jumped onto the fence and watched as",
    "A long time ago in a distant kingdom, the king",
    "The little girl found a magical stone that could",
    "It was a cold winter night when the stranger",
    "The garden was full of colorful flowers and",
    "The teacher told the students about how",
    "After many years of traveling, the explorer",
    "The two friends decided to go on an adventure to",
    "Every morning the baker would wake up early and",
    "The storm was approaching and the sailors",
    "In the deep forest, there lived a wise old owl who",
    "The princess looked out her window and saw",
    "The musician picked up his guitar and began to play a",
    "Hello, how are you today? I hope you are",
    "Thank you for your help with the project. It was",
    "The weather has been really nice this week, with",
    "I think the best thing about weekends is that you can",
    "My favorite season is autumn because the leaves change",
]


def compute_expert_loss(model, cfg, token_ids, device="cpu"):
    """Compute the loss of an expert on a query (as next-token prediction)."""
    if len(token_ids) < 2:
        return float("inf")

    # Truncate to model's seq_len
    seq_len = min(len(token_ids) - 1, cfg.seq_len)
    x = torch.tensor([token_ids[:seq_len]], dtype=torch.long, device=device)
    y = torch.tensor([token_ids[1:seq_len + 1]], dtype=torch.long, device=device)

    with torch.no_grad():
        _, loss = model(x, y)
    return loss.item()


def simple_tokenize(text, vocab_size=4096, max_len=256):
    """Simple word-level tokenization mapped into vocab space.

    NOTE: hash-based placeholder — use the real BPE tokenizer (gen_router_data_v2.py).
    """
    words = text.lower().split()
    ids = [(hash(w) % (vocab_size - 1)) + 1 for w in words]  # avoid 0 (pad token)
    return ids[:max_len]


def generate_data(domains, n_samples, output_path, seed=42):
    """Generate router training data by evaluating experts on queries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cpu"  # Can use cuda if available

    # Load all experts
    experts = {}
    for domain in domains:
        try:
            model, cfg = load_expert(domain)
            model = model.to(device)
            experts[domain] = (model, cfg)
            print(f"✅ Loaded expert: {domain}")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return

    # Domain query pools
    query_pools = {
        "code": CODE_QUERIES,
        "reasoning": REASONING_QUERIES,
        "general": GENERAL_QUERIES,
    }

    # Generate samples
    domain_list = list(domains)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for i in range(n_samples):
            # Pick a random domain's query pool, but also mix in some random queries
            domain = random.choice(domain_list)
            pool = query_pools.get(domain, GENERAL_QUERIES)
            query = random.choice(pool)

            # Tokenize
            token_ids = simple_tokenize(query, vocab_size=experts[domain][1].vocab_size)

            # Compute loss for each expert
            losses = []
            for d in domain_list:
                model, cfg = experts[d]
                loss = compute_expert_loss(model, cfg, token_ids, device)
                losses.append(loss)

            # Best expert = lowest loss
            best_expert = int(np.argmin(losses))

            # Soft labels: use inverse loss as weight (optional, for distillation)
            inv_losses = [1.0 / (l + 1e-8) for l in losses]
            total_inv = sum(inv_losses)
            soft_labels = [il / total_inv for il in inv_losses]

            sample = {
                "text": query,
                "best_expert": best_expert,
                "expert_losses": losses,
                "soft_labels": soft_labels,
                "source_domain": domain,
            }
            f.write(json.dumps(sample) + "\n")

        print(f"✅ Generated {n_samples} samples → {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Generate router training data")
    ap.add_argument("--n-samples", type=int, default=500,
                    help="Number of training samples to generate")
    ap.add_argument("--n-val", type=int, default=100,
                    help="Number of validation samples")
    ap.add_argument("--output", default=None,
                    help="Output path (default: data/router_train.jsonl)")
    ap.add_argument("--domains", nargs="+", default=["code", "reasoning", "general"],
                    help="Expert domains to evaluate")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_path = args.output or os.path.join(DATA, "router_train.jsonl")
    val_path = train_path.replace("train", "val")

    # Generate training data
    generate_data(args.domains, args.n_samples, train_path, seed=args.seed)

    # Generate validation data with different seed
    generate_data(args.domains, args.n_val, val_path, seed=args.seed + 1000)


if __name__ == "__main__":
    main()