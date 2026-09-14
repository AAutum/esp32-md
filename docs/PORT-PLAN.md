# model dispatch on ESP32 fleet — port plan & gap analysis (2026-09-12)

**Goal:** 3-device model dispatch (2× N16R8 experts + host orchestrator) per the 8/22-24 design, leaving
the production sensor agent out. Later: more experts, maybe the T-Deck.

## What already exists (built 8/22-24, first draft of this repo)

| Component | Status | Notes |
|---|---|---|
| `router.py` (332K params) | ✅ tested | PyTorch, sequence-level routing |
| `train_expert.py` (PLE TinyLM per domain) | ✅ imports clean | falls back to TinyStories shared data |
| `generate_router_data.py` | ✅ | eval experts → route labels |
| `train_router.py` | ✅ | needs BPE tokenizer shared w/ experts (flagged) |
| `md_server.py` (orchestrator) | ✅ | serial-based |
| `flash_experts.sh` | ✅ | CPU cap + export + flash |
| Expert ESP32 firmware | ❌ **missing** | serial `GEN <tokens> <max>` sketch — the actual gap |
| Router training DATA | ❌ | need domain corpora (code/reasoning/general) |
| BPE-32768 tokenizer | ✅ shipped in `data/` | wire into dispatch-model training |

## Port plan (what "port it over" actually means)

**Stage 0 — fleet sink (done today):** sensor data centralized; irrelevant to model dispatch compute
but validates the WiFi+HTTP path we'll reuse for the orchestrator's LAN mode.

**Stage 1 — expert firmware (the missing piece, ~1-2 sessions):**
Port the upstream PLE runtime + loader into a serial-command sketch:
- `GEN <n_tokens> <prompt_tokens...>` → streams generated token IDs back
- Reuse the upstream runtime's proven pieces: PLE mmap loader, int8 head worker task, PSRAM scratch
- Partition table: keep `model` partition (N16R8: model ~5MB fits 3M-param expert at Q4)
- Flash via the workstation (both boards already attached there) using flash_experts.sh
- **Deliverable:** an ESP32 that responds to serial GEN — identical contract for every expert

**Stage 2 — single-expert bring-up on a workstation USB hub:**
- md_server.py already speaks serial; verify round-trip latency per token
- Budget check: 2 experts on same USB hub → ensure no bandwidth serialization issues
- Expected: ~9.5 tok/s per expert (proven), so top-2 routing ≈ 19 tok/s aggregate ceiling,
  realistically ~5-8 tok/s wall-clock with serial + routing overhead

**Stage 3 — router data + training:**
- 3 domain corpora: code (we have tons locally), general TinyStories, "reasoning"
  (scratch: small math/word-problem sets — weakest corpus, can synthesize)
- generate_router_data.py on the 2 flashed experts + a CPU-held third
- train_router.py (332K params, minutes even on Pi)

**Stage 4 — true-edge router (stretch):**
- Quantize router to int8 (332K → ~330KB, fits SRAM easily)
- Port to a 3rd ESP32 as standalone router node using the upstream matvec path
- ESP-NOW between router and experts (sub-ms, no AP dependency) — bigger firmware lift,
  only after serial version proves the routing logic

## Key engineering risks

1. **Specialization quality** — experts trained on different corpora can drift in format.
   Mitigation: shared tokenizer + shared base checkpoint, then domain fine-tune (proven pattern).
2. **Merging top-2 logits** — softmax-weighted sum at router; needs calibrated router confidences.
   Cheap first version: argmax-router (top-1), add top-2 later.
3. **USB hub bandwidth** — 2 ESP32s on the workstation's hub is fine (CDC-ACM is low-rate), but
   md_server must use async reads, which it already does (threaded).
4. **Model partition size** — 3M-param PLE at Q4 ≈ 1.9MB (25M-table was for 28.9M model).
   N16R8 model partition is 5MB → room for ~8M-param experts. Don't undersell this.

## Immediate next actions (when ready to execute)

1. Write `firmware/md_expert/md_expert.ino` (serial GEN contract) — port from
   the upstream PLE runtime + sensor-agent's task structure
2. Domain data prep on the workstation host (code/reasoning/general corpora → token IDs)
3. `train_expert.py --domain code|reasoning|general` on the workstation **when it's next powered on**
   (the workstation is intentionally OFF for power savings — train on the workstation host CPU if impatient,
   ~3× slower but works)
4. Dispatch data gen + dispatch train (fast, any CPU)
5. Flash both N16R8s from the workstation → md_server.py smoke test

**Honest effort estimate:** Stage 1-2 = one focused session; Stage 3 = data prep dominates
(~half a day); Stage 4 = separate session. Full pipeline ≈ 2-3 focused sessions.