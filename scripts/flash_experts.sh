#!/bin/bash
# Flash trained expert models to ESP32-S3 boards connected to this machine.
#
# Usage:
#   ./flash_experts.sh /dev/ttyACM_EXPERT0 /dev/ttyACM_EXPERT1 /dev/ttyACM_EXPERT2
#
# This script:
# 1. Sets CPU frequency to 2.5GHz (thermal cap for workstation)
# 2. Exports each expert model to ESP32 flash format
# 3. Flashes each model to the corresponding ESP32 board

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${PROJECT_DIR}/runs"

# Expert domains in order (must match port order)
DOMAINS=("code" "reasoning" "general")

# ─── CPU thermal cap ──────────────────────────────────────────────────────────

echo "🔧 Setting CPU frequency cap to 2.5GHz (thermal management for workstation)..."
if [ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq ]; then
    echo 2500000 | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq > /dev/null
    echo "✅ CPU capped at 2.5GHz"
else
    echo "⚠️  Cannot set CPU freq (need root). Continuing anyway..."
fi

# ─── Check ports ──────────────────────────────────────────────────────────────

if [ $# -lt 1 ]; then
    echo "Usage: $0 /dev/ttyACM_EXPERT0 [/dev/ttyACM_EXPERT1 ...]"
    echo ""
    echo "Provide serial ports in expert index order:"
    for i, domain in "${DOMAINS[@]}"; do
        echo "  Port $i: expert-$domain"
    done
    exit 1
fi

PORTS=("$@")

if [ ${#PORTS[@]} -gt ${#DOMAINS[@]} ]; then
    echo "⚠️  More ports than domains. Only first ${#DOMAINS[@]} will be used."
fi

# ─── Export and flash ─────────────────────────────────────────────────────────

for i in "${!DOMAINS[@]}"; do
    domain="${DOMAINS[$i]}"
    port="${PORTS[$i]:-}"

    if [ -z "$port" ]; then
        echo "⚠️  No port for expert-$domain, skipping"
        continue
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 Expert $i: $domain → $port"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Check if checkpoint exists
    ckpt="${RUNS}/expert-${domain}-s0.pt"
    if [ ! -f "$ckpt" ]; then
        echo "❌ Checkpoint not found: $ckpt"
        echo "   Run train_expert.py --domain $domain first"
        continue
    fi

    # Export to C header + binary (using the vendored export script)
    echo "📤 Exporting $domain model..."
    cd "${ROOT}/firmware/tools"
    python3 src/export.py --checkpoint "$ckpt" --output "${PROJECT_DIR}/firmware/expert_${domain}" || {
        echo "❌ Export failed for $domain"
        continue
    }

    # Flash to ESP32
    echo "🔥 Flashing to $port..."
    cd "${PROJECT_DIR}/firmware/expert_${domain}"

    # Use esptool to flash
    esptool.py --port "$port" --chip esp32s3 \
        --baud 921600 \
        write_flash 0x10000 model.bin || {
        echo "❌ Flash failed for $domain on $port"
        continue
    }

    echo "✅ expert-$domain flashed to $port"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All experts flashed. Ready to run md_server.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"