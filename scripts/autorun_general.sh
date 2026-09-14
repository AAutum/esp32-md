#!/usr/bin/env bash
# Auto-run: wait for expert-general-s0 training to finish -> export -> verify ->
# host ppl (fp32 + int8 numerics) -> flash model.bin @0x110000 -> bring-up eval.
#
# Board: whichever expert port you pass/expect — set PORT below.
# Model partition: 0x110000 size 0xEE0000 (read from board 9/13 — NOT 0x420000)
set -uo pipefail

WS="${MD_WORKSPACE:-$HOME/model-dispatch-ws}"
MD="${MD_ROOT:-$WS/esp32-md}"
PORT="${1:-/dev/ttyACM_EXPERT}"
MODEL_OFF="0x110000"
LOG="${MD}/autorun_general.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG}"; }

log "=== general-expert autorun started (pid $$) ==="

# ── 1. Wait for training to finish ──────────────────────────────────────────
while true; do
  if ! pgrep -f "train_expert.py --domain general" >/dev/null 2>&1; then
    # double-check via ps (pgrep -f self-match guard)
    if ! ps aux | grep "train_expert.py --domain general" | grep -v grep >/dev/null; then
      log "training process gone"
      break
    fi
  fi
  sleep 60
done

sleep 5
log "checkpoint:"
ls -l "${MD}/checkpoints/expert-general-s0.pt" 2>/dev/null || ls -l "${MD}/checkpoints/"*.pt >> "${LOG}" 2>&1
tail -3 "${MD}/logs/runs_general.log" >> "${LOG}" 2>/dev/null

# ── 2. Export (int4-packed model.bin + golden reference) ────────────────────
cd "${MD}/firmware/tools"
log "export..."
python3 export.py expert-general-s0 >> "${LOG}" 2>&1 || { log "EXPORT FAILED"; exit 1; }
ls -l "${MD}/firmware/model/general.bin" "${MD}/firmware/model/general_golden.txt" >> "${LOG}"
log "export OK"

# ── 3. Host correctness: C port vs PyTorch golden ──────────────────────────
cd "${MD}/firmware/host_verify"
gcc -O2 -o verify verify.c ../common/llm.h 2>>"${LOG}" || { log "verify build FAILED"; exit 1; }
log "host verify (C vs golden)..."
./verify "${MD}/firmware/model/general.bin" "${MD}/firmware/model/golden.txt" >> "${LOG}" 2>&1 || log "VERIFY MISMATCH (continuing — check log)"

# ── 4. Host ppl on the val bin, fp32 + int8 activation numerics ─────────────
gcc -O2 -o ppl_f32 ppl.c ../common/llm.h 2>>"${LOG}"
gcc -O2 -DllM_INT8_ACT=1 -o ppl_i8 ppl.c ../common/llm.h 2>>"${LOG}"
log "host ppl fp32-activations..."
./ppl_f32 "${MD}/firmware/model/general.bin" "${MD}/data/general_val.bin" 16 >> "${LOG}" 2>&1
log "host ppl int8-activations..."
./ppl_i8 "${MD}/firmware/model/general.bin" "${MD}/data/general_val.bin" 16 >> "${LOG}" 2>&1

# ── 5. Flash model.bin at the model partition offset ────────────────────────
log "flashing model.bin (${MODEL_OFF})..."
esptool --port "${PORT}" --baud 921600 --chip esp32s3 \
  write_flash "${MODEL_OFF}" "${MD}/firmware/model/general.bin" >> "${LOG}" 2>&1 \
  || { log "FLASH FAILED"; exit 1; }
log "flash OK"

# ── 6. Bring-up: hard reset via esptool (reliable boot path), then eval ─────
sleep 2
esptool --port "${PORT}" --baud 115200 --chip esp32s3 run >> "${LOG}" 2>&1 || true
sleep 3

cd "${MD}"
python3 src/serial_eval.py --port "${PORT}" --max-tokens 48 \
  --val-bin data/general_val.bin --out "${MD}/eval_general.json" >> "${LOG}" 2>&1 \
  || log "serial eval FAILED (see log)"

log "=== autorun complete ==="
tail -40 "${LOG}"