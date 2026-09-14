"""Host-side model-dispatch orchestrator — routes queries to ESP32 experts over serial.

This runs on the Pi 5 (or any host connected to multiple ESP32s) and:
1. Tokenizes the input query
2. Runs the router to determine top-k experts
3. Sends the query to each selected ESP32 over serial
4. Collects outputs and merges them weighted by router confidence

Usage:
    python md_server.py --ports /dev/ttyACM_EXPERT0 /dev/ttyACM_EXPERT2
"""

import argparse
import json
import os
import sys
import time
import serial

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router import RouterModel, RouterConfig


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
RUNS = os.path.join(PROJECT, "runs")


class ExpertDevice:
    """Wrapper for an ESP32 expert connected over serial."""

    def __init__(self, port, domain, baud=921600):
        self.port = port
        self.domain = domain
        self.baud = baud
        self.serial = None

    def connect(self):
        try:
            self.serial = serial.Serial(self.port, self.baud, timeout=10)
            # Wait for ready
            time.sleep(2)
            line = self.serial.readline().decode(errors="ignore").strip()
            print(f"  {self.domain} on {self.port}: {line or 'connected'}")
            return True
        except Exception as e:
            print(f"  ❌ {self.domain} on {self.port}: {e}")
            return False

    def query(self, token_ids, max_tokens=64):
        """Send token IDs to the ESP32 and get generated tokens back.

        Protocol (simple, newline-delimited):
            Send: "GEN <token_ids space-separated> <max_tokens>\n"
            Recv: "<generated_token_ids space-separated>\n"
        """
        if self.serial is None:
            raise RuntimeError(f"{self.domain} not connected")

        # Send query
        cmd = f"GEN {' '.join(map(str, token_ids))} {max_tokens}\n"
        self.serial.write(cmd.encode())
        self.serial.flush()

        # Read response
        response = self.serial.readline().decode(errors="ignore").strip()
        try:
            tokens = [int(t) for t in response.split()]
            return tokens
        except ValueError:
            print(f"  ⚠️ Bad response from {self.domain}: {response!r}")
            return []

    def close(self):
        if self.serial:
            self.serial.close()


class DispatchOrchestrator:
    """Routes queries to experts and merges outputs."""

    def __init__(self, router_path, expert_ports, top_k=2):
        """
        Args:
            router_path: Path to trained router checkpoint
            expert_ports: List of (port, domain) tuples
            top_k: Number of experts to query per request
        """
        self.top_k = top_k

        # Load router
        ckpt = torch.load(router_path, map_location="cpu", weights_only=False)
        cfg = RouterConfig(**ckpt["cfg"])
        self.router = RouterModel(cfg)
        self.router.load_state_dict(ckpt["state"])
        self.router.eval()
        print(f"✅ Router loaded: {sum(p.numel() for p in self.router.parameters()):,} params")

        # Connect experts
        self.experts = []
        for port, domain in expert_ports:
            dev = ExpertDevice(port, domain)
            if dev.connect():
                self.experts.append(dev)

        if len(self.experts) < top_k:
            print(f"⚠️  Only {len(self.experts)} experts connected, "
                  f"reducing top_k from {top_k} to {len(self.experts)}")
            self.top_k = len(self.experts)

    def route(self, token_ids):
        """Run the router to determine which experts to activate."""
        # Truncate to router's seq_len
        seq_len = self.router.cfg.seq_len
        ids = token_ids[:seq_len]
        # Pad to seq_len
        if len(ids) < seq_len:
            ids = ids + [0] * (seq_len - len(ids))

        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            topk_idx, topk_probs = self.router.route(x, top_k=self.top_k)

        return topk_idx[0].tolist(), topk_probs[0].tolist()

    def query(self, text, max_tokens=64):
        """Full pipeline: tokenize → route → query experts → merge."""
        # Tokenize (same simple scheme as generate_router_data.py)
        token_ids = simple_tokenize(text, vocab_size=self.router.cfg.vocab_size)

        # Route
        expert_indices, weights = self.route(token_ids)

        print(f"\n📝 Query: {text}")
        print(f"🔀 Routed to experts: {expert_indices} with weights: {weights}")

        # Query each selected expert
        results = []
        for idx, weight in zip(expert_indices, weights):
            if idx < len(self.experts):
                expert = self.experts[idx]
                tokens = expert.query(token_ids, max_tokens=max_tokens)
                results.append({
                    "expert": expert.domain,
                    "weight": weight,
                    "tokens": tokens,
                })
                print(f"  {expert.domain} (w={weight:.3f}): {len(tokens)} tokens generated")

        # Merge: weighted token voting (simplified — for real implementation,
        # we'd merge at the logit level, but since ESP32s return tokens not logits,
        # we do weighted token selection)
        merged = self._merge_results(results, max_tokens)
        return merged, results

    def _merge_results(self, results, max_tokens):
        """Merge outputs from multiple experts.

        Simple approach: for each position, pick the token from the
        highest-weighted expert. A more sophisticated approach would
        compare logits, but ESP32s return tokens not logits.

        TODO: If experts return logit distributions, do proper weighted
        logit merging. For now, we do confidence-weighted selection.
        """
        if not results:
            return []

        # Sort by weight (highest first)
        results.sort(key=lambda r: r["weight"], reverse=True)

        # Use the highest-weighted expert's output as primary
        primary = results[0]["tokens"]

        # If we have a second expert, use it to fill gaps where primary
        # produced fewer tokens
        if len(results) > 1 and len(primary) < max_tokens:
            secondary = results[1]["tokens"]
            needed = max_tokens - len(primary)
            primary = primary + secondary[:needed]

        return primary[:max_tokens]

    def close(self):
        for expert in self.experts:
            expert.close()


def fnv1a(word: str, vocab_size: int) -> int:
    """Deterministic FNV-1a word hash — must match train_router.py::fnv1a and
    the C implementation in firmware/md_router/md_router.ino."""
    h = 0x811C9DC5
    for byte in word.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h % vocab_size


def simple_tokenize(text, vocab_size=4096, max_len=256):
    """Word-level FNV-1a tokenization (matches train_router.py)."""
    words = text.lower().split()
    ids = [fnv1a(w, vocab_size) for w in words]
    return ids[:max_len]


def main():
    ap = argparse.ArgumentParser(description="model dispatch orchestrator")
    ap.add_argument("--router", default=os.path.join(RUNS, "router-3exp-s0.pt"),
                    help="Path to router checkpoint")
    ap.add_argument("--ports", nargs="+", required=True,
                    help="Serial ports for ESP32 experts (in expert index order)")
    ap.add_argument("--domains", nargs="+", default=["code", "reasoning", "general"],
                    help="Domain names for each port (same order)")
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    if len(args.ports) != len(args.domains):
        print("❌ Number of ports must match number of domains")
        return

    expert_ports = list(zip(args.ports, args.domains))
    orch = DispatchOrchestrator(args.router, expert_ports, top_k=args.top_k)

    # Interactive loop
    print(f"\n{'='*60}")
    print(f"Dispatch cluster ready — {len(orch.experts)} experts, top_k={orch.top_k}")
    print(f"{'='*60}")
    print(f"Type a query and press Enter. Type 'quit' to exit.\n")

    try:
        while True:
            text = input("query> ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue

            t0 = time.time()
            tokens, results = orch.query(text, max_tokens=args.max_tokens)
            elapsed = time.time() - t0

            print(f"\n✨ Merged output: {len(tokens)} tokens in {elapsed:.2f}s")
            for r in results:
                print(f"  {r['expert']}: w={r['weight']:.3f}, {len(r['tokens'])} tokens")
            print()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        orch.close()


if __name__ == "__main__":
    main()