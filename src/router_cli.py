#!/usr/bin/env python3
"""Interactive CLI for the ESP32 dispatch node (T3S3).

Usage:
  python3 router_cli.py [--port /dev/ttyACM_DISPATCH] [--baud 921600]

Commands (type at the prompt):
  <any text>            route it — prints top-2 experts + weights + latency
  :ids <n n n ...>      route raw token ids (skips on-device tokenizer)
  :logits <n n n ...>   raw classifier logits (HW verification)
  :stat                 device stats
  :ping                 device identity
  :quit

Or pipe a single query:  echo "some text" | python3 router_cli.py --once
"""
import argparse, sys, time
import serial

MAX_SEQ = 64


def send_line(ser, line: str):
    """Chunked write — the ESP32-S3 USB-JTAG RX ring is ~128B and silently
    drops overflow from one-shot bursts (see DESIGN.md §8)."""
    data = line.encode() + b"\n"
    for i in range(0, len(data), 16):
        ser.write(data[i:i + 16])
        time.sleep(0.002)


def read_line(ser, timeout=8.0):
    t0, buf = time.time(), b""
    while time.time() - t0 < timeout:
        b = ser.read(1)
        if not b:
            continue
        if b == b"\n":
            return buf.decode(errors="replace").strip()
        buf += b
    raise TimeoutError("device silent")


def sync(ser):
    ser.reset_input_buffer()
    for _ in range(6):
        send_line(ser, "PING")
        line = read_line(ser, 3.0)
        if line.startswith("PONG"):
            return line
    raise RuntimeError("no PONG from router node")


DOMAINS = {0: "general", 1: "code", 2: "reasoning"}


def pretty(line: str):
    parts = line.split()
    if parts[0] == "ROUTE":
        pairs = parts[1].split(",")
        names = ", ".join(f"{DOMAINS.get(int(e), '?')}({e}) {float(w):.4f}"
                          for e, w in (p.split(":") for p in pairs))
        us = int(parts[2]) if len(parts) > 2 else -1
        print(f"  → {names}   [{us/1000:.1f} ms]")
    elif parts[0] == "LOGITS":
        print(f"  → logits {parts[1:-1]}   [{int(parts[-1])/1000:.1f} ms]")
    else:
        print(f"  → {line}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM_DISPATCH")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--once", action="store_true",
                    help="read one query from stdin, print, exit")
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(1.0)
    pong = sync(ser)
    print(f"router: {pong}")

    if args.once:
        text = sys.stdin.read().strip()
        if text:
            send_line(ser, "ROUTE " + text[:1024])
            pretty(read_line(ser))
        ser.close()
        return

    print("type text to route, :help for device commands, :quit\n")
    while True:
        try:
            text = input("route> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text == ":quit":
            break
        try:
            if text.startswith(":ids "):
                ids = text[5:].split()[:MAX_SEQ]
                ids += ["0"] * (MAX_SEQ - len(ids))
                send_line(ser, "TOKENS " + " ".join(ids))
                pretty(read_line(ser))
            elif text.startswith(":logits "):
                ids = text[8:].split()[:MAX_SEQ]
                ids += ["0"] * (MAX_SEQ - len(ids))
                send_line(ser, "LOGITS " + " ".join(ids))
                pretty(read_line(ser))
            elif text == ":stat":
                send_line(ser, "STAT")
                print(" ", read_line(ser))
            elif text == ":ping":
                send_line(ser, "PING")
                print(" ", read_line(ser))
            elif text == ":help":
                print("  <text> | :ids <ids> | :logits <ids> | :stat | :ping | :quit")
            else:
                send_line(ser, "ROUTE " + text[:1024])
                pretty(read_line(ser))
        except (TimeoutError, RuntimeError) as e:
            print(f"  ! {e} — resyncing")
            try:
                print(" ", sync(ser))
            except RuntimeError:
                print("  ! device not answering; check USB")
    ser.close()


if __name__ == "__main__":
    main()