"""Prepare expert training corpora (uint16 bins in the BPE-32768 vocab).

Outputs (data/):
  code_train.bin       — Python/shell/C from local repos + workspace scripts
  code_val.bin
  reasoning_train.bin  — synthetic word problems, arithmetic chains, logic
  reasoning_val.bin
  general_train.bin    — slice of the TinyStories train bin (same tokenizer, direct
                         uint16 copy of a contiguous slice)
  general_val.bin

All experts share the BPE-32768 vocab shipped in data/bpe32768.json,
so token IDs are directly compatible with the flashed vocab.h / PLE table.

Usage: python3 prepare_expert_data.py [--max-code-mb 60]
"""
import argparse, glob, json, os, random, re, sys
import numpy as np
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
ESP32_AI_DATA = os.environ.get("ESP32_AI_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
BPE = os.path.join(ESP32_AI_DATA, "bpe32768.json")

random.seed(1337)

# ---- code corpus sources: local repos + workspace scripts -------------------
CODE_ROOTS = [
    # Point this at your own code repos — the corpus is built from local source.
    os.environ.get("MD_CODE_ROOTS", os.path.expanduser("~/code")).rstrip("/"),
]
CODE_EXTS = (".py", ".sh", ".c", ".h", ".cpp", ".ino", ".csv")  # csv: tiny configs
SKIP_DIRS = {"runs", "runs_dlr", "__pycache__", ".git", "node_modules", "build",
             "data", "runs2m", "stage_v2s", ".venv", "venv"}
MAX_FILE_KB = 64

def iter_code_files():
    for root in CODE_ROOTS:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if fn.endswith(CODE_EXTS):
                    p = os.path.join(dirpath, fn)
                    try:
                        if os.path.getsize(p) > MAX_FILE_KB * 1024: continue
                        with open(p, "r", errors="ignore") as f:
                            txt = f.read()
                        if len(txt) < 80: continue
                        yield p, txt
                    except OSError:
                        pass

def chunk_text(txt, max_chars=1800):
    """Split file into ~logical chunks (functions/classes/blocks)."""
    lines = txt.split("\n")
    out, cur = [], []
    n = 0
    for ln in lines:
        cur.append(ln); n += len(ln) + 1
        if (n > max_chars) or (ln.strip() == "" and n > 400):
            out.append("\n".join(cur)); out, cur, n = [], [], 0
    if cur and len("\n".join(cur)) > 60:
        out.append("\n".join(cur))
    return out

# ---- reasoning corpus: synthetic generators ---------------------------------
def gen_arith(n):
    out = []
    for _ in range(n):
        a = random.randint(2, 99); b = random.randint(2, 99)
        op = random.choice(["+", "-", "*"])
        if op == "*": a = random.randint(2, 15); b = random.randint(2, 15)
        q = f"Question: What is {a} {op} {b}?\n"
        ans = a + op if False else (a + b if op == "+" else a - b if op == "-" else a * b)
        out.append(q + f"Answer: {ans}\n")
        # two-step chain
        c = random.randint(2, 20)
        q2 = f"Question: What is {a} {op} {b} {random.choice(['+','-'])} {c}?\n"
        mid = ans
        out.append(q2 + f"Answer: {mid + c if random.random()<0.5 else mid - c}\n")
    return out

def gen_word_problems(n):
    names = ["Ana","Ben","Cara","Dev","Eli","Fay","Gus","Hal","Ivy","Jo"]
    items = [("apples",3),("marbles",5),("cards",4),("coins",2),("sticks",7)]
    out = []
    for _ in range(n):
        nm = random.choice(names); it, base = random.choice(items)
        add = random.randint(2, 12)
        q = f"Question: {nm} has {base} {it}. A friend gives {nm} {add} more. How many {it} does {nm} have?\n"
        out.append(q + f"Answer: {base + add}\n")
        if random.random() < 0.5:
            take = random.randint(1, base)
            q = f"Question: {nm} has {base + add} {it} and gives away {take}. How many are left?\n"
            out.append(q + f"Answer: {base + add - take}\n")
    return out

def gen_logic(n):
    out = []
    for _ in range(n):
        a, b = random.sample(["red","blue","green","yellow","black","white"], 2)
        x, y = random.sample(["box","bag","drawer","pocket","shelf"], 2)
        # simple transitive: A is in x. x is bigger than y. So A is bigger than y? yes.
        q = (f"Question: The {a} object is in the {x}. The {x} is larger than the {y}. "
             f"Is the {a} object larger than the {y}? Answer yes or no.\n")
        out.append(q + "Answer: yes\n")
        q = (f"Question: The {a} object is in the {x}. The {y} is larger than the {x}. "
             f"Is the {a} object larger than the {y}? Answer yes or no.\n")
        out.append(q + "Answer: no\n")
    return out

def gen_sort_seqs(n):
    out = []
    for _ in range(n):
        k = random.randint(3, 5)
        nums = [random.randint(0, 99) for _ in range(k)]
        q = f"Question: Sort these numbers from smallest to largest: {', '.join(map(str, nums))}\n"
        out.append(q + "Answer: " + ", ".join(map(str, sorted(nums))) + "\n")
    return out

# ---- encode -----------------------------------------------------------------
def encode_chunks(tok, chunks, desc):
    ids = []
    B = 2000
    for i in range(0, len(chunks), B):
        enc = tok.encode_batch(chunks[i:i+B])
        for e in enc:
            ids.extend(e.ids)
    arr = np.array(ids, dtype=np.uint16)
    print(f"  {desc}: {len(chunks)} chunks -> {len(arr):,} tokens ({arr.nbytes/1e6:.1f} MB)")
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-code-mb", type=int, default=60)
    ap.add_argument("--reasoning-docs", type=int, default=40000)
    args = ap.parse_args()

    tok = Tokenizer.from_file(BPE)
    os.makedirs(DATA, exist_ok=True)

    # ---- code
    print("Collecting code files...")
    chunks = []
    for p, txt in iter_code_files():
        chunks.extend(chunk_text(txt))
    random.shuffle(chunks)
    print(f"  {len(chunks)} chunks total")
    # budget cap: encode in batches until max size
    n_all = len(chunks)
    n_train = int(n_all * 0.97)
    train_chunks = chunks[:n_train]
    val_chunks = chunks[n_train:]
    arr = encode_chunks(tok, train_chunks, "code_train")
    # cap
    if len(arr) > args.max_code_mb * 1_000_000 // 2:
        arr = arr[: args.max_code_mb * 1_000_000 // 2]
    arr.tofile(os.path.join(DATA, "code_train.bin"))
    val_arr = encode_chunks(tok, val_chunks, "code_val") if val_chunks else arr[:100_000]
    val_arr.tofile(os.path.join(DATA, "code_val.bin"))

    # ---- reasoning (synthetic)
    print("Generating reasoning corpus...")
    docs = (gen_arith(args.reasoning_docs // 4) + gen_word_problems(args.reasoning_docs // 4)
            + gen_logic(args.reasoning_docs // 4) + gen_sort_seqs(args.reasoning_docs // 4))
    random.shuffle(docs)
    n_tr = int(len(docs) * 0.97)
    tr = encode_chunks(tok, docs[:n_tr], "reasoning_train")
    tr.tofile(os.path.join(DATA, "reasoning_train.bin"))
    va = encode_chunks(tok, docs[n_tr:], "reasoning_val")
    va.tofile(os.path.join(DATA, "reasoning_val.bin"))

    # ---- general: contiguous slice of the existing TinyStories bin (same vocab)
    src = os.path.join(ESP32_AI_DATA, "train_v32768.bin")
    big = np.memmap(src, dtype=np.uint16, mode="r")
    print(f"TinyStories source: {len(big):,} tokens")
    g_tr = np.array(big[:30_000_000], dtype=np.uint16)   # 30M tokens ≈ 60MB slice
    g_tr.tofile(os.path.join(DATA, "general_train.bin"))
    g_va = np.array(big[30_000_000:30_500_000], dtype=np.uint16)
    g_va.tofile(os.path.join(DATA, "general_val.bin"))
    print(f"  general_train: {len(g_tr):,} tok ({g_tr.nbytes/1e6:.0f} MB), val {len(g_va):,}")

    print("\nDone. Bins in", os.path.abspath(DATA))
    for f in sorted(glob.glob(os.path.join(DATA, "*.bin"))):
        print(f"  {os.path.basename(f):24s} {os.path.getsize(f)/1e6:8.1f} MB")

if __name__ == "__main__":
    main()