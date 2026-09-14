# ESP32 Model Dispatch — Design & Operations Manual

*Last updated: 2026-09-14. Read this before touching anything.*

## 1. What this project is

A **model-dispatch language system that runs across ESP32 microcontrollers** —
no data center, no API keys, no cloud. Three small language models, each
specialized in one domain (code / reasoning / general), live as int4-quantized
blobs in ESP32-S3 flash. A tiny dispatch model (a classifier) decides which expert(s) should
handle a given query. An orchestrator (currently host-side, moving to-device)
routes queries and merges outputs.

This is the **BTX (Branch-Train-MiX) family**: *learned-routing expert ensemble
with sequence-level routing* — not a sparse Mixture-of-Experts (no per-token
expert mixing inside one model). Each expert is a complete standalone LM; the
dispatch model picks 1-2 of them per query. Use this terminology consistently.

### Why it matters (thesis)

Same architecture, different data → measurably different domain behavior.
That was validated in Stage 1: three experts with identical PLE TinyLM
architectures diverged exactly along their training domains (general val ppl
9.00 with coherent TinyStories, code ppl 39.5 with real code output, reasoning
ppl 1.57 with correct arithmetic — and *off-domain failure by design*). Domain
specialization is real, and a 93.3%-accurate dispatch model can tell them apart.

### Where each piece physically lives

| Node | Hardware | Runs | Status |
|---|---|---|---|
| Expert: general | ESP32-S3 N16R8 | `firmware/md_expert` + general model | flashed 9/13 |
| Expert: code | ESP32-S3 N16R8 #2 (pending flash) | same firmware + code model | pending |
| Expert: reasoning | ESP32-S3 N16R8 (pending reflash) | same + reasoning model | pending |
| **Dispatch node** | **LilyGo T3S3 (4MB flash / 2MB PSRAM)** | `firmware/md_router` + dispatch int8 blob | **verified on HW** |
| Orchestrator | any host with USB serial | `src/md_server.py` over serial | works host-side |

The T3S3 was reserved for exactly this role during planning. Its variant has only 4MB embedded flash
(verified via esptool: `Embedded Flash 4MB (XMC)`, `Embedded PSRAM 2MB
(AP_3v3)`), so the 9.44MB experts cannot live there — but the 2.17M-param
dispatch model int8-quantizes to ~2.2MB, which fits.

## 2. Architecture

```
                query text ("If A > B and B > C, what is A vs C?")
                     │
        ┌────────────▼─────────────┐
        │ DISPATCH NODE (T3S3)     │
        │  FNV-1a word tokenize    │   ← deterministic, see §4
        │  int8 transformer (2.17M)│
        │  → top-2 experts+weights │
        └────────────┬─────────────┘
                     │ serial (or ESP-NOW later)
        ┌────────────▼─────────────┐
        │ ORCHESTRATOR (host)      │  md_server.py
        │  picks experts, queries  │
        └──┬───────────────┬───────┘
           ▼               ▼
    ┌────────────┐  ┌────────────┐
    │ EXPERT 0   │  │ EXPERT 2   │     GEN <prompt ids> <max>
    │ code       │  │ reasoning  │     → generated token ids
    │ 3M int4    │  │ 3M int4    │
    └────────────┘  └────────────┘
           └───────┬───────┘
                   ▼
        weighted merge (host, token level)
```

### Dispatch model (the new piece)

A *classifier*, not a generator. Architecture (`src/router.py`):

- token embedding 32768×64 (int8-quantized per row), positional embedding 64×64 (fp16)
- 2 transformer blocks, d_model=64, 2 heads, FFN hidden 128, pre-RMSNorm,
  causal attention, GELU (exact erf form)
- mean-pool over **all 64 positions** (including padding — matches training)
- RMSNorm → linear classifier → 3 logits → softmax → top-2, renormalized

2,167,299 params. int8 per-row quantization ≈ 2.2MB. Trained: val acc 93.3%
on the full val set (30 epochs, ~20 min on a Pi-5-class CPU). Label source: for
each training query, the expert with the lowest CE loss (`gen_router_data_v2.py`
— real BPE-tokenized queries from the actual expert corpora, losses computed
under all 3 experts).

### Experts

PLE TinyLM ~3M params each (general/code/reasoning seeds s0), trained 20k
steps on domain corpora, int4 group-wise quantized (llm.h PLE1 format), 9.44MB
model.bin flashed at offset 0x110000 in the `model` partition. On-device
inference 270ms/tok. Firmware in `firmware/md_expert/` (uses the shared
`firmware/common/llm.h` PLE1 runtime).

Serial contract (md_server.py ⇄ experts): `GEN <ids...> <max>\n` →
`<ids...>\n`, plus `PING` → `PONG <domain> ...`

## 3. The two model formats

| | PLE1 (experts) | RTR1 (router) |
|---|---|---|
| Defined in | `firmware/common/llm.h` | `src/export_router.py` ⇄ `firmware/md_router/*.ino` |
| Purpose | int4 PLE TinyLM generator | int8 transformer classifier |
| Weight quant | group-wise int4 (group=128) + fp16 scales | per-output-row int8 + fp16 scales |
| Activation quant | dynamic per-tensor int8 (head) | dynamic per-tensor int8 (every linear) |
| Magic | `PLE1` 0x504C4531 | `RTR1` 0x52545231 |
| Loader | mmap + stream decode in llm.h | mmap + pointer walk in router .ino |

RTR1 layout (fixed order, header = magic + V,D,L,H,F,S,E as i32):

```
tok_codes V*D i8 | tok_scales V fp16 | pos_emb S*D fp16
per layer: qkv_w8 3D*D i8 + qkv_s 3D fp16 | proj_w8 D*D i8 + proj_s D fp16
         | up_w8 F*D i8 + up_s F fp16    | dn_w8 D*F i8 + dn_s D fp16
         | attn_norm D fp16 | ffn_norm D fp16
out_norm D fp16 | cls_w8 E*D i8 + cls_s E fp16 | cls_bias E fp32
```

### The quantization contract (memorize this)

- **Weights**: symmetric per-output-row int8. `scale = max|row|/127`,
  **fp16-rounded before dividing** (so device dequant matches golden bit-for-bit),
  codes clamped ±127.
- **Activations**: dynamic per-tensor symmetric int8, exactly `quantize_act()`
  semantics: `xmax=max(|x|,1e-8); inv=127/xmax; q=rint(x·inv) clamp ±127;
  xscale=xmax/127`.
- **Linear output**: `y[r] = (int32-dot(act_q, w8[r]) · wscale[r]) · act_s`
  — int32 accumulation, then TWO left-associated float multiplies in exactly
  that order. Order matters; both sides (export sim + C) must match.
- **tok_emb dequant**: `code · fp16scale` in fp32, then `+ pos_emb` (fp16→fp32).
- **RMSNorm**: `(w·x)·inv` — w first. fp32 math, eps 1e-6, mean over D.
- **Softmax/attention/pooling**: fp32 on device, mean over all S positions.

Golden references exist at two levels: `runs/router-export/router_golden.npz`
(int8-sim logits per vector, computed in float64-exact integer accumulation)
and `src/hw_verify.py` compares device `LOGITS` output against it.

## 4. The tokenizer story (read this, it burned us)

The dispatch input pipeline is **word-level FNV-1a hashing**, NOT the BPE
tokenizer the experts use. `train_router.py` originally tokenized with
Python's `hash(word)` — which is **seed-randomized per interpreter process**.
The original checkpoint was therefore keyed to a hash seed that stopped
existing when that process died; any deployment would route garbage. Worse,
`md_server.py` used a slightly different formula (`%(V-1)+1` vs `%(V)`), so
even same-seed runs disagreed.

Fix: `fnv1a()` — 32-bit FNV-1a over UTF-8 bytes, `h % V` —
implemented **identically in three places**:

1. `src/train_router.py::fnv1a` (training)
2. `src/md_server.py::simple_tokenize` (host orchestrator)
3. `firmware/md_router/md_router.ino::fnv1a` (on-device)

If you change one, change all three, and re-verify with `hw_verify.py`.

Why word-hash instead of BPE for the dispatch model: it only needs to
*classify domain*, and domain signal (code tokens, math phrasing, story
syntax) survives crude tokenization. BPE on-device would need a merge-table
engine (big, slow); FNV-1a is ~10 lines of C and bit-exact everywhere.
The experts still use real BPE-32768 (`data/bpe32768.json`) on the host side
when generating prompts — the orchestrator tokenizes text→ids with
BPE for experts and FNV-1a for dispatching. Two tokenizers, two purposes; the
dispatch model never needs to match the experts' ids.

Padding: word ids are truncated/padded **with id 0 to seq_len=64** and padded
positions flow through the whole pipeline (positional embeddings, causal
attention, mean-pool over all 64) — identical in training, host sim, and
device code. Do not "optimize" padding out; the model was trained that way.

## 5. T3S3 hardware specifics (LilyGo T3S3)

- esptool probe: ESP32-S3 QFN56 rev v0.2, **4MB embedded XMC flash, 2MB
  embedded PSRAM (AP_3v3 = QSPI)**. This is the SiP variant — NOT the
  N16R8 external-flash boards used for experts.
- USB: two ports enumerate — the ROM USB-Serial-JTAG port (use for flashing)
  and the native USB CDC port from the running firmware. Ports move after
  reflash; identify boards by USB serial MAC, not port number.
- Build FQBN: `esp32:esp32:lilygo_t3s3:CDCOnBoot=cdc,PSRAM=disabled,USBMode=hwcdc`
  — **CDCOnBoot=cdc is mandatory** or Serial is silent (fleet lesson, hit twice).
  PSRAM is DISABLED in the router build deliberately: the hot path uses SRAM
  only, so the QSPI/OPI question never arises. (Other fleet boards: PSRAM=opi.)
- Partitions (`firmware/md_router/partitions.csv`): nvs 0x9000,
  **factory app 0x10000–0x140000 (1.2MB)**, **model (data, subtype 0x40)
  0x140000–0x3F0000 (2.75MB)**, coredump 0x3F0000. Sketch compiles to 856KB
  (65% of app slot).
- Repurposed hardware: if your T3S3 previously ran other firmware, take a
  full 4MB flash backup before flashing (esptool read-back), and erase the
  model region so no stale bytes survive underneath.
- Flash procedure (`firmware/md_router/flash_router.sh`): compile →
  **erase-region 0x140000 0x2B0000** (so no stale RNode bytes survive under the
  model) → bootloader 0x0, partitions 0x8000, boot_app0 0xe000, app 0x10000 →
  model.bin 0x140000. esptool v5.3.1 syntax (subcommand style).
- Bootloader flash header was checked against the RNode image before flashing:
  both DIO, identical size/freq nibbles — embedded-flash SiP happy.

## 6. Session history (what happened when)

- **8/22-24**: original dispatch design + orchestration code written (README-era).
- **9/12**: Stage 1 execution starts. Data prep (`prepare_expert_data.py`),
  expert firmware written (`md_expert.ino`), PORT-PLAN gap analysis.
- **9/13**: all three experts trained (20k steps each, Pi 5 CPU) →
  general ppl 9.00 / code 39.51 / reasoning 1.57, int4 model.bin 9.44MB @
  0x110000, 270ms/tok on device; expert board flashed + validated end-to-end.
  Dispatch data v2 (real BPE + domain-sampled corpora), dispatch model trained
  30 epochs, 97.3% train-subset val acc. Stage 1 complete.
- **9/14 (this session)**: T3S3 plugged in as router node. RNode flash backed
  up (md5-verified). **Random-hash tokenizer bug found + fixed** (FNV-1a in all
  three implementations), router retrained: val acc 96.4% (training-subset) /
  **93.3% on the full val set** — that's the honest number. RTR1 export format
  + `export_router.py` (int8 gate PASS: int8 == fp32 acc, cos ≥ 0.9991),
  firmware written (flash-mmap model, SRAM-only hot path).
  **Two firmware bugs caught by verification**: (1) attention softmax lacked
  max-subtraction → scores >88 overflowed expf → NaN (sim used torch.softmax
  which max-subtracts — device must too); (2) header walk skipped 36 bytes
  instead of 32 (magic misread as 8 bytes) → misaligned weights, output
  collapsed to shifted bias + NaN. The 36-byte bug was diagnosed by simulating
  the pointer walk in numpy — skip=36 reproduced the device signature exactly
  ([-0.090425, -0.153402, nan]); skip=32 clean.
  **HW VERIFY PASS**: 184 golden vectors (120 val + 64 fuzz), device vs int8
  sim cos_min 0.99999984, top-1 match 100%, val top-1 vs label 93.3% (== sim).
  Latency ~377ms/route — the 2.14MB model streams from flash-mmap each pass
  (PSRAM is only 2MB, can't cache it). Fine for a router; don't expect more.
  Smoke: ROUTE(text) == TOKENS(host-fnv-ids) on 6 real texts — C tokenizer
  bit-identical to Python. OOD caveat: fibonacci/A>B>C prose routes general@1.00
  (model overconfidence, known limitation §8). RNode backup retained; project
  files reorganized into `scripts/ logs/ evals/ docs/ fleet/`.

## 7. Verification ladder (never skip)

1. **Export gate** (`export_router.py`): int8 sim val acc ≥ fp32 − 3pt and ≥90%
   absolute; cosine int8-vs-fp32 reported. Refuses nothing — read the report.
2. **Golden sim** (`runs/router-export/router_golden.npz`): 120 val rows + 64
   random fuzz rows, int8-exact logits.
3. **HW verify** (`src/hw_verify.py`): streams every golden vector through the
   device `LOGITS` command. Gate: top-1 match 100%, cosine ≥0.999. Writes
   `runs/router-export/hw_verify_report.json`.
4. **Smoke routes**: a few `ROUTE <text>` commands — code-looking text →
   expert 1, arithmetic/relational → expert 0, story text → expert 2.
5. **Latency**: STAT last_us. Measured: ~377ms/route — dominated by streaming
   the 2.14MB blob from SPI flash-mmap each pass (PSRAM is 2MB < model, can't
   cache; SRAM can't either). ≈2.6 routes/s — vastly faster than any expert
   (270ms/token), so the router is never the bottleneck. If it ever matters:
   shrink vocab embedding (int4) or stream only the needed blocks.

For experts the equivalent ladder is in the eval scripts (`serial_eval.py`,s
(`serial_eval.py`, `evals/*.json` bring-up evals).

## 8. Current limitations / known sharp edges

- Router trained/evaluated on 680/120 queries — tiny but the domains are highly
  separable (ep2 hit 100%). **Overconfidence is the real weakness**: ad-hoc OOD
  texts (fibonacci snippet, A>B>C prose) route general@1.00 with p=1.0. Training
  corpus-sampled queries hit 93.3%, but anything stylistically novel gets
  confidently wrong. Fix = harder, more-diverse router training data (roadmap).
- Orchestrator merge is token-level weighted-vote, not logit-level (experts
  return ids, not logits). Fine for v1; logit merging would need a GEN-LOGITS
  variant of the expert contract.
- `md_server.py` currently hardcodes `simple_tokenize` vocab default 4096 vs
  the dispatch model's 32768 — must pass vocab_size=router.cfg.vocab_size (fixed in the
  call path this session; if you touch simple_tokenize, keep the call sites
  explicit).
- Experts' `GEN` prompt capacity: 256 ids max in firmware parser; seq_len caps
  longer contexts.
- The T3S3's 4MB flash means OTA updates are impossible with the current
  partition table (single factory slot, no otadata). Serial flashing only.
  Accepted: this is a router node, not a production-OTA device.
- esptool flash-read to this board dropped twice at high baud ("packet content
  transfer stopped") — succeeded at 460800 with --no-stub. If a big transfer
  fails, drop baud before suspecting hardware.

## 9. Plans

### Next (Stage 4 completion)
1. Finish router retrain (deterministic FNV-1a) → `export_router.py` gate
2. Flash T3S3: `scripts/... flash_router.sh` → `hw_verify.py` PASS
3. Smoke routes over serial
4. Wire md_server to use the on-device dispatch node (`--router-host <port>`
   mode: orchestrator asks the T3S3 for ROUTE instead of running torch).
   **Proven end-to-end via src/md_demo.py** — text→ROUTE→GEN→decode,
   377ms route + 268ms/tok gen. md_server still needs the flag wired in.
5. Flash code expert to the next free N16R8 board, then reasoning expert
6. Full 3-expert + on-device-router demo: text in → routed, generated,
   merged text out

**⚠ Board-state gotcha:** an expert board was found holding the REASONING
expert, not the expected GENERAL — an earlier bring-up flashed reasoning over
general during that expert's eval, and no log recorded it. Detected by ppl
cross-check (ckpt story-ppl 1.74 vs device emitting babi-format text) plus
two eval JSONs listing the same port. Lesson: **after any expert flash,
immediately md5-record which model is on the board** (a line in evals/ or a
`flash-state.json`), else USB history silently lies.

### Then
- **ESP-NOW between router and experts** (kill the USB tether; sub-ms, no AP).
  Needs the GEN contract to move from serial lines to ESP-NOW frames — bigger
  firmware lift, only after serial version proves the routing logic.
- Harder mixed-domain router data (fix overconfidence).
- Top-2 logit-level merge (add LOGITS output to expert firmware; merge in
  orchestrator before decode).
- Router-on-T-Deck curiosity: the T-Deck (ESP32-S3+SX1262) could host both a
  router and a LoRa uplink — route queries from the mesh.
- Battery/enclosure for the router node once ESP-NOW cuts the cord.

### Fleet roles (stable)
- One expert per N16R8 board; the sensor/telemetry agent (if you run one)
  keeps its own board and is never reflashed.
- Dispatch node = T3S3. Boards are identified by USB serial MAC, not port.

## 10. File map

```
esp32-md/
├── README.md                 ← quick architecture (this file is the deep one)
├── docs/DESIGN.md            ← this document
├── docs/PORT-PLAN.md         ← 9/12 port plan + gap analysis (historical)
├── src/
│   ├── router.py             ← router model def (RouterConfig/RouterModel)
│   ├── train_router.py       ← router training (FNV-1a tokenize, batcher)
│   ├── gen_router_data_v2.py ← builds router_train/val.jsonl (BPE + expert losses)
│   ├── export_router.py      ← RTR1 export + int8 golden sim + gate
│   ├── hw_verify.py          ← device-vs-golden verification (LOGITS)
│   ├── md_server.py          ← host orchestrator (FNV-1a now)
│   ├── md_demo.py            ← end-to-end demo (device dispatch path)
│   ├── router_cli.py         ← interactive CLI for the dispatch node
│   ├── prepare_expert_data.py← domain corpora → token bins
│   ├── generate_router_data.py (v1, superseded by v2)
│   └── serial_eval.py        ← expert bring-up eval driver
├── scripts/                  ← shell utilities (autorun, boot_app, run_eval, flash_experts)
├── firmware/md_router/       ← dispatch firmware (.ino + partitions.csv + flash_router.sh)
├── data/                     ← expert corpora token bins + dispatch jsonl
├── checkpoints/              ← dispatch checkpoint + router-export/ (bin+golden+report)
├── evals/                    ← expert bring-up eval JSONs
└── firmware/tools/           ← vendored PLE training/export toolkit (MIT, see firmware/VENDORED.md)
```

Expert-side files that matter: `firmware/md_expert/` (firmware),
`firmware/common/llm.h` (PLE1 runtime), `firmware/tools/export.py`
(PLE1 exporter), `data/bpe32768.json` (shared BPE).