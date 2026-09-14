#!/usr/bin/env python3
"""End-to-end demo: text → dispatch node (on-device) → expert GEN → decoded text.

Wiring as of 2026-09-14:
  router : /dev/ttyACM_DISPATCH  (T3S3, ROUTE contract)
  expert : /dev/ttyACM_EXPERT    (expert board, GEN contract)
  code/reasoning experts: checkpoints trained, boards not attached yet —
  when routed there, this demo reports the miss and (optionally) falls back.

Usage:
  python3 md_demo.py "Once upon a time ..." [--max-tokens 32] [--fallback]
"""
import argparse, sys, time
from tokenizers import Tokenizer

ROUTER_PORT = "/dev/ttyACM_DISPATCH"
EXPERT_PORTS = {0: "/dev/ttyACM_EXPERT0", 2: "/dev/ttyACM_EXPERT2"}  # expert id → port
BAUD = 921600
BPE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "bpe32768.json")
DOMAINS = {0: "general", 1: "code", 2: "reasoning"}
# Device map (verified 9/14 by PING + GEN signature):
#   ACM1 = N16R8 #2  → general   (fresh flash 9/14, coherent TinyStories)
#   ACM2 = N16R8 #1  → reasoning (flashed 9/13, babi-style; general was
#                      flashed there first then OVERWRITTEN by reasoning eval)
#   code: checkpoint ready, no board attached yet.
AVAILABLE = {0: "/dev/ttyACM_EXPERT0", 2: "/dev/ttyACM_EXPERT2"}


def send_line(ser, line, chunk=16, gap=0.002):
    data = line.encode() + b"\n"
    for i in range(0, len(data), chunk):
        ser.write(data[i:i + chunk])
        time.sleep(gap)


def read_line(ser, timeout=10.0):
    t0, buf = time.time(), b""
    while time.time() - t0 < timeout:
        b = ser.read(1)
        if not b:
            continue
        if b == b"\n":
            return buf.decode(errors="replace").strip()
        buf += b
    raise TimeoutError("device silent")


def sync(ser, want="PONG", tries=6):
    ser.reset_input_buffer()
    for _ in range(tries):
        send_line(ser, "PING")
        line = read_line(ser, 3.0)
        if line.startswith(want):
            return line
    raise RuntimeError(f"no {want}")


def route(ser_r, text):
    send_line(ser_r, "ROUTE " + text[:1024])
    line = read_line(ser_r, 8.0)
    parts = line.split()
    pairs = [p.split(":") for p in parts[1].split(",")]
    us = int(parts[2]) if len(parts) > 2 else -1
    return [(int(e), float(w)) for e, w in pairs], us


def generate(ser_e, text, tok, max_tokens):
    ids = tok.encode(text).ids[:128]
    send_line(ser_e, "GEN " + " ".join(map(str, ids)) + f" {max_tokens}")
    # response: one line of ids, then a "// stats" comment line.
    # Generations can outpace per-byte reads (270ms/tok, line arrives in one
    # burst) — use a chunked raw read, fall back to line mode.
    deadline = time.time() + max_tokens * 0.6 + 12
    out_ids, stats = [], ""
    buf = b""
    while time.time() < deadline:
        chunk = ser_e.read(256)
        if chunk:
            buf += chunk
            if b"//" in buf and buf.split(b"//")[-1].endswith(b"\n"):
                break
        elif b"\n" in buf:
            # got at least one full line but no stats yet — keep waiting only
            # if it looks incomplete; ids line alone with no stats = retry below
            pass
    text_buf = buf.decode(errors="replace")
    for line in text_buf.splitlines():
        line = line.strip()
        if line.startswith("//"):
            stats = line
        elif line and all(t.lstrip("-").isdigit() for t in line.split()):
            try:
                out_ids = [int(t) for t in line.split()]
            except ValueError:
                pass
    if not out_ids:
        raise TimeoutError(f"expert produced no ids; raw={text_buf[:120]!r}")
    return out_ids, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--fallback", action="store_true",
                    help="generate on the available expert even if top pick is unflashed")
    args = ap.parse_args()

    tok = Tokenizer.from_file(BPE)

    import serial
    ser_r = serial.Serial(ROUTER_PORT, BAUD, timeout=1)
    time.sleep(1.0)
    pong = sync(ser_r)
    print(f"router: {pong}")

    top, us = route(ser_r, args.query)
    names = ", ".join(f"{DOMAINS[e]}({e}) {w:.4f}" for e, w in top)
    print(f"route : {names}   [{us/1000:.1f} ms]")

    eid = top[0][0]
    if eid not in AVAILABLE:
        note = (f"top expert '{DOMAINS[eid]}' is not flashed yet "
                f"(checkpoints ready, board not attached).")
        if not args.fallback:
            print(f"stop  : {note} use --fallback to generate on an available expert.")
            ser_r.close()
            return
        print(f"note  : {note} falling back.")
        eid = 0 if 0 in AVAILABLE else 2

    port = AVAILABLE[eid]
    ser_e = serial.Serial(port, BAUD, timeout=1)
    time.sleep(1.0)
    epong = sync(ser_e)
    print(f"expert: {epong}  ({port})")

    print(f"gen   : {DOMAINS[eid]} expert ← {args.max_tokens} tokens...")
    out_ids, stats = generate(ser_e, args.query, tok, args.max_tokens)
    text_out = tok.decode(out_ids)
    print(f"text  : {text_out!r}")
    if stats:
        print(f"perf  : {stats[3:]}")

    ser_r.close()
    ser_e.close()


if __name__ == "__main__":
    main()