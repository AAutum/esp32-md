#!/usr/bin/env python3
"""HW verification: send LOGITS commands for every golden vector to the router
node over serial, compare against export_router.py golden (int8 sim).

Gate: top-1 expert match on every val row + cosine >= 0.999 on logits.
Writes hw_verify_report.json next to the golden.

Usage: python3 hw_verify.py [--port /dev/ttyACM_DISPATCH] [--baud 921600]
"""
import argparse, json, os, time
import numpy as np
import serial

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "runs", "router-export")


def read_line(ser, timeout=10.0):
    t0 = time.time()
    buf = b""
    while time.time() - t0 < timeout:
        b = ser.read(1)
        if not b:
            continue
        if b == b"\n":
            return buf.decode(errors="replace").strip()
        buf += b
    raise TimeoutError("no line within timeout")


def send_line(ser, line: str):
    """Write a command in small chunks. The ESP32-S3 USB-Serial-JTAG RX ring is
    only ~128B and silently drops overflow; one-shot 150B writes can lose the
    tail while the device is mid-response. 16B chunks with 2ms gaps always drain."""
    data = line.encode() + b"\n"
    for i in range(0, len(data), 16):
        ser.write(data[i:i + 16])
        time.sleep(0.002)


def sync(ser):
    """Drain stale CDC bytes, then PING until PONG. Returns banner lines seen."""
    seen = []
    t0 = time.time()
    while time.time() - t0 < 1.0:            # drain silence window
        b = ser.read(256)
        if b:
            seen.append(b.decode(errors="replace"))
            t0 = time.time()
    for _ in range(6):
        ser.write(b"PING\n")
        line = read_line(ser, timeout=3.0)
        if line.startswith("PONG"):
            return line, seen
        seen.append(line)
    raise RuntimeError(f"no PONG; saw {seen[-3:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM_DISPATCH")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--limit", type=int, default=0, help="cap vectors (0=all)")
    args = ap.parse_args()

    g = np.load(os.path.join(OUT, "router_golden.npz"))
    ids, labels = g["ids"], g["labels"]
    logits_ref = g["logits_int8"]
    n = len(ids) if not args.limit else min(args.limit, len(ids))

    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(1.5)
    pong, stale = sync(ser)
    if stale:
        print(f"drained {len(stale)} stale line(s): {stale[-1][:60]!r}")
    print("ping  :", pong)

    cosines, matches, rows = [], [], []
    for i in range(n):
        cmd = "LOGITS " + " ".join(str(int(t)) for t in ids[i])
        got = None
        for attempt in range(3):
            send_line(ser, cmd)
            try:
                line = read_line(ser)
            except TimeoutError:
                continue
            if not line.startswith("LOGITS"):
                print(f"[{i}] unexpected: {line!r}")
                continue
            got = line
            break
        if got is None:
            print(f"[{i}] no response after retries, aborting")
            break
        parts = got.split()
        hw = np.array([float(x) for x in parts[1:1 + logits_ref.shape[1]]])
        ref = logits_ref[i]
        cos = float(np.dot(hw, ref) /
                    (np.linalg.norm(hw) * np.linalg.norm(ref) + 1e-12))
        cosines.append(cos)
        matches.append(int(np.argmax(hw)) == int(np.argmax(ref)))
        if labels[i] >= 0:
            rows.append((int(np.argmax(hw)), int(labels[i])))
    ser.close()

    cosines = np.array(cosines)
    top1_val = np.mean([m and (r[0] == r[1]) for m, r in zip(matches, rows)]) if rows else 0.0
    report = {
        "vectors": n,
        "cos_min": float(cosines.min()) if len(cosines) else None,
        "cos_mean": float(cosines.mean()) if len(cosines) else None,
        "top1_vs_ref": float(np.mean(matches)) if len(matches) else None,
        "val_top1_vs_label": float(top1_val),
        "gate_pass": bool(len(cosines) and cosines.min() >= 0.999
                          and np.mean(matches) == 1.0),
    }
    print(json.dumps(report, indent=2))
    with open(os.path.join(OUT, "hw_verify_report.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()