# ESP32 Model Dispatch — specialized language models routed on $8 microcontrollers

Run three domain-specialized language models (code / reasoning / general) across
ESP32-S3 microcontrollers -- no cloud, no API keys, no datacenter. A tiny
**dispatch model** (a classifier) looks at your query and sends it to the model
best suited for it. Text in, routed to hardware, generated text out.

> **Terminology:** this is a *model dispatch* system (BTX family: learned-routing
> expert ensemble, sequence-level routing). Each expert is a complete standalone
> language model — not a sparse Mixture-of-Experts with per-token expert mixing.

```
                 query text ("What is 77 - 79?")
                      │
       ┌──────────────▼───────────────┐
       │ DISPATCH NODE (LilyGo T3S3)  │
       │  FNV-1a word tokenize        │
       │  int8 transformer (2.2M)     │
       │  → top-2 experts + weights   │      ~377 ms/route
       └──────────────┬───────────────┘
                      │ USB serial (ESP-NOW later)
       ┌──────────────▼───────────────┐
       │ ORCHESTRATOR (any host)      │   src/md_server.py / md_demo.py
       └───┬──────────────────────┬───┘
           ▼                      ▼
    ┌─────────────┐        ┌──────────────┐
    │ EXPERT board│        │ EXPERT board │    GEN <prompt ids> <max>
    │ general     │        │ reasoning    │    → generated token ids
    │ 3M, int4    │        │ 3M, int4     │    ~268 ms/tok (3.7 tok/s)
    └─────────────┘        └──────────────┘
           └────────┬─────────────┘
                  weighted merge → decoded text
```

## What's in the box

| Piece | What it is | Size on flash |
|---|---|---|
| Experts ×3 | PLE TinyLMs (~3M params), int4 group-wise quantized | 9.44 MB each |
| Dispatch model | 2.17M-param int8 transformer classifier | 2.2 MB |
| Expert firmware | Serial `GEN` contract, PLE1 mmap runtime | 860 KB app |
| Dispatch firmware | RTR1 format, flash-mmap streaming, SRAM-only hot path | 856 KB app |
| Orchestrator | Host-side serial orchestrator + end-to-end demo | Python |

- **Experts** share a BPE-32768 tokenizer, trained 20k steps each on
  TinyStories / real code / synthetic reasoning corpora.
- **Dispatch model** classifies query domain (93.3% val accuracy) using word-level
  FNV-1a hashing — deterministic, ~10 lines of C, bit-identical across the
  Python training, host orchestrator, and device firmware.

## Hardware

- **Expert boards:** ESP32-S3 N16R8 (16 MB flash / 8 MB PSRAM). Model blob lives
  in a `model` flash partition at `0x110000`; int8-staged head in PSRAM; dual-core
  head matvec.
- **Dispatch node:** LilyGo T3S3 (4 MB flash / 2 MB PSRAM SiP). The 2.2 MB int8
  dispatch model streams from flash-mmap every pass — ~2.6 routes/s, never the
  bottleneck next to a 3.7 tok/s expert.
- One expert per board; the orchestrator opens only the port that PINGs with the
  domain it needs. Boards are identified by USB serial MAC (ports re-enumerate).

## Quick start

```bash
# Route-only (classification): interactive CLI against the dispatch node
python3 src/router_cli.py --port /dev/ttyACM_DISPATCH

# End-to-end: text → dispatch node → expert generation → decoded text
python3 src/md_demo.py "Once upon a time there was a little dragon"
python3 src/md_demo.py "Question: What is 77 - 79?"

# Host-side orchestrator (torch router, for fleet testing without the T3S3)
python3 src/md_server.py --ports /dev/ttyACM_EXPERT0 /dev/ttyACM_EXPERT2
```

Training / reproduction (Pi 5-class CPU is enough):

```bash
python3 firmware/tools/prepare_data.py          # TinyStories slice + BPE-32768
python3 src/prepare_expert_data.py              # domain corpora → token bins
python3 src/train_expert.py --domain general    # ~2 h/board on Pi 5 CPU (x3)
python3 firmware/tools/export.py ...            # PLE1 int4 model.bin
python3 src/gen_router_data_v2.py               # queries labeled by expert loss
python3 src/train_router.py --epochs 30         # ~20 min on Pi 5 CPU
python3 src/export_router.py                    # RTR1 int8 + golden vectors
# verify on device BEFORE trusting it:
python3 src/hw_verify.py --port /dev/ttyACM_DISPATCH  # 184 golden vectors
```

See **`docs/DESIGN.md`** for the full manual: model formats (PLE1/RTR1),
quantization contract, tokenizer story (including the random-hash bug that
forced FNV-1a), verification ladder, and hardware specifics.

## Repository layout

```
src/                 training, export, verification, orchestration (Python)
firmware/md_router/  dispatch-node firmware (.ino, partitions, flash script)
firmware/md_expert/  expert firmware (serial GEN contract)
firmware/common/     PLE1 single-header runtime (mmap'd flash inference)
firmware/tools/      upstream PLE training/export toolkit (vendored, MIT)
firmware/model/      pre-built general-expert model.bin (int4)
data/                tokenized domain corpora + dispatch training data
checkpoints/         dispatch model checkpoint + golden export artifacts
evals/               on-device bring-up eval records (JSON)
docs/                DESIGN.md (the manual), PORT-PLAN.md (build history)
```

Checkpoints for the three experts are not included (70 MB torch files); the
training pipeline reproduces them from `data/`.

## Credits & license

- Expert runtime, PLE architecture, and training toolkit are vendored from
  [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai) (MIT) — an excellent
  project that runs a 28.9M-param LM on an ESP32-S3. See `firmware/tools/`
  and `firmware/common/llm.h`.
- General expert trained on [TinyStories](https://arxiv.org/abs/2305.07759)
  (roneneldan/TinyStories).
- Everything else (dispatch model, expert serial firmware, orchestration,
  verification tooling, domain corpora) — MIT, © 2026.
