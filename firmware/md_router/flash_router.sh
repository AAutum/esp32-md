#!/bin/bash
# Build + flash the dispatch firmware and model to the LilyGo T3S3 dispatch node.
#
# Usage: ./flash_router.sh [serial_port] [model_bin]
#   serial_port  default: your T3S3 USB-Serial-JTAG ROM port
#   model_bin    default <workspace>/esp32-md/runs/router-export/router_model.bin
#
# Order matters: erase model region FIRST (RNode firmware still occupies
# 0x140000+ until we overwrite it — erase zeroes it so stale RNode data can
# never be read as a model), then bootloader/partitions/app, then model blob.
#
# Backup of the original RNode flash lives in
# fleet/t3s3-router/backup/t3s3-rnode-flash-2026-09-14.bin (md5 verified).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$HERE/../.."                                 # esp32-md/ (script is in firmware/md_router/)
FIRMWARE="$WS/firmware/md_router"
PORT="${1:-/dev/ttyACM_DISPATCH}"
MODEL="${2:-$WS/runs/router-export/router_model.bin}"
BUILD="/tmp/rtr_build"

FQBN="esp32:esp32:lilygo_t3s3:CDCOnBoot=cdc,PSRAM=disabled,USBMode=hwcdc"
PARTITION_CSV="md_router"

MODEL_OFFSET=0x140000
MODEL_SIZE=0x2B0000

[ -f "$MODEL" ] || { echo "model not found: $MODEL (run export_router.py first)"; exit 1; }
MSIZE=$(stat -c %s "$MODEL")
[ "$MSIZE" -le "$((MODEL_SIZE))" ] || { echo "model $MSIZE bytes > partition $MODEL_SIZE ($((MODEL_SIZE)))"; exit 1; }

echo "== compile =="
rm -rf "$BUILD"; mkdir -p "$BUILD"
arduino-cli compile -b "$FQBN" --build-path "$BUILD" \
  --build-property "build.partitions=$PARTITION_CSV" "$FIRMWARE"

echo "== erase stale model region (0x140000..) =="
python3 -m esptool -p "$PORT" -b 921600 \
  erase-region 0x140000 0x2B0000

echo "== flash firmware =="
python3 -m esptool -p "$PORT" -b 921600 \
  write-flash 0x0    "$BUILD/${FIRMWARE##*/}.ino.bootloader.bin" \
              0x8000 "$BUILD/${FIRMWARE##*/}.ino.partitions.bin" \
              0xe000 "$BUILD/boot_app0.bin" \
              0x10000 "$BUILD/${FIRMWARE##*/}.ino.bin"

echo "== flash model ($MSIZE bytes @ $MODEL_OFFSET) =="
python3 -m esptool -p "$PORT" -b 921600 \
  write-flash "$MODEL_OFFSET" "$MODEL"

echo "done. open serial at 921600; expect banner + READY."