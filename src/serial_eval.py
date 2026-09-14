#!/usr/bin/env python3
"""Serial eval for a flashed dispatch expert: PING + GEN samples with TTFT/tok-s.

Sends the firmware's serial contract (firmware/md_expert/md_expert.ino):
  PING -> "PONG <domain> V= D= L= heap="
  GEN <ids...> <max> -> token id line + "// N tok, X ms/tok, Y tok/s"

Measures host-side TTFT (write -> first byte), parses device-reported
ms/tok + tok/s, decodes generated ids back to text via the shared BPE.

Usage:
  python3 serial_eval.py --port /dev/ttyACM_EXPERT [--out results.json]
"""
import argparse
import json
import os
import time

import serial
from tokenizers import Tokenizer

DEFAULT_BPE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "bpe32768.json"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM_EXPERT")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--bpe", default=os.path.abspath(DEFAULT_BPE))
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--val-bin", default="data/general_val.bin")
    ap.add_argument("--val-prompt-len", type=int, default=32)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.bpe)
    print(f"[bpe] {args.bpe}")

    # Named text prompts + one real val-seed prompt for a realistic TTFT.
    prompts = [
        ("tinystories", "Once upon a time there was a little"),
        ("knowledge", "The capital of France is"),
        ("code", "def add(a, b):"),
        ("reasoning", "If John has 3 apples and buys 2 more, he has"),
    ]
    if os.path.exists(args.val_bin):
        raw = open(args.val_bin, "rb").read()
        import struct
        ids = list(struct.unpack(f"<{args.val_prompt_len}H", raw[: args.val_prompt_len * 2]))
        prompts.append(("val-seed-32", ids))
        print(f"[val-seed] first ids: {ids[:12]}...")

    s = serial.Serial(args.port, args.baud, timeout=2)
    s.dtr = False
    time.sleep(0.7)
    banner = drain(s, 1.0)
    print(f"[banner] {banner!r}")

    # PING
    s.reset_input_buffer()
    t0 = time.time()
    s.write(b"PING\n")
    pong_raw = drain(s, 2.0)
    print(f"[PING] {pong_raw!r} ({(time.time()-t0)*1000:.0f} ms)")

    results = {"port": args.port, "banner": banner, "pong": pong_raw.strip(), "gens": []}

    for name, prompt in prompts:
        ids = prompt if isinstance(prompt, list) else tok.encode(prompt).ids
        prompt_text = prompt if isinstance(prompt, str) else tok.decode(ids)
        line = "GEN " + " ".join(str(i) for i in ids) + f" {args.max_tokens}\n"
        s.reset_input_buffer()
        t0 = time.time()
        s.write(line.encode())
        # read until the "// " profile line appears or timeout
        buf = bytearray()
        ttft = None
        deadline = time.time() + 30.0
        while time.time() < deadline:
            n = s.in_waiting
            if n:
                chunk = s.read(n)
                if ttft is None and chunk.strip():
                    ttft = time.time() - t0
                buf += chunk
                if b"// " in bytes(buf):
                    time.sleep(0.05)
                    buf += s.read(s.in_waiting)
                    break
            else:
                time.sleep(0.01)
        text = bytes(buf).decode("utf-8", "replace")
        gen_ids, profile = parse_gen(text)
        out_text = ""
        if gen_ids:
            try:
                out_text = tok.decode(gen_ids)
            except Exception as e:
                out_text = f"<decode error: {e}>"
        rec = {
            "name": name,
            "prompt": prompt_text,
            "n_prompt_tokens": len(ids),
            "ttft_s": round(ttft, 3) if ttft else None,
            "profile": profile,
            "gen_ids": gen_ids[: args.max_tokens],
            "output": out_text,
            "raw": text.strip()[:400],
        }
        results["gens"].append(rec)
        print(f"\n=== {name} ===")
        print(f"  prompt : {prompt_text!r} ({len(ids)} tok)")
        print(f"  ttft   : {rec['ttft_s']} s")
        print(f"  device : {profile}")
        print(f"  output : {out_text!r}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[saved] {args.out}")


def drain(s, timeout):
    end = time.time() + timeout
    buf = bytearray()
    while time.time() < end:
        n = s.in_waiting
        if n:
            buf += s.read(n)
        else:
            time.sleep(0.02)
    return bytes(buf).decode("utf-8", "replace").strip()


def parse_gen(text):
    """Extract the token-id line and the '// N tok, X ms/tok, Y tok/s' profile."""
    gen_ids, profile = [], None
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.startswith("//"):
            profile = ln.lstrip("/")
            continue
        parts = ln.split()
        if parts and all(p.lstrip("-").isdigit() for p in parts) and len(parts) >= 1:
            gen_ids.extend(int(p) for p in parts)
    return gen_ids, profile


if __name__ == "__main__":
    main()