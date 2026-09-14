#!/bin/bash
cd "$(dirname "$0")/.."
python3 src/serial_eval.py --port /dev/ttyACM_EXPERT --max-tokens 48 --val-bin data/general_val.bin --out eval_general.json