#!/bin/bash
# Flash the GENERAL expert to an N16R8 expert board.
# Mirrors autorun_general.sh's proven 9/13 flow, minus the training wait.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
MD="$(cd "${HERE}/.." && pwd)"   # repo root
PORT="${1:-/dev/ttyACM_EXPERT}"
OFF="0x110000"
B2="${MD}/firmware/md_expert/build2"  # build dir: compile md_expert.ino first
LOG="${MD}/logs/flash_general.log"
exec > >(tee -a "${LOG}") 2>&1
echo "=== [$(date '+%F %T')] flash general -> ${PORT} ==="

# 1. fresh export so model.bin is unambiguous (domain-agnostic fw in build2)
cd "${MD}/firmware/tools" && python3 export.py expert-general-s0
md5sum "${MD}/firmware/model/general.bin"

# 2. full firmware: bootloader + partitions + app + otadata
python3 -m esptool --port "${PORT}" --baud 460800 --chip esp32s3 \
  write_flash 0x0 "${B2}/md_expert.ino.bootloader.bin" \
  0x8000 "${B2}/md_expert.ino.partitions.bin" \
  0xe000 "${B2}/boot_app0.bin" \
  0x10000 "${B2}/md_expert.ino.bin"

# 3. model into the model partition
python3 -m esptool --port "${PORT}" --baud 460800 --chip esp32s3 \
  write_flash "${OFF}" "${MD}/firmware/model/general.bin"

# 4. reset into app
python3 -m esptool --port "${PORT}" --baud 115200 --chip esp32s3 run || true
echo "=== flash done ==="