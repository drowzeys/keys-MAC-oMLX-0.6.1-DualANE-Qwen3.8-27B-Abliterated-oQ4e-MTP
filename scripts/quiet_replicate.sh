#!/bin/bash
# Quiesce Mac co-tenants, let it settle, replicate the tie-break protocol, restore.
set -u
export PATH=/opt/homebrew/bin:$PATH
VP=/opt/homebrew/Cellar/omlx/0.6.1/libexec/bin/python
cd ~/omlx-qwen38

PIDS=$(pgrep -f "EXO.app|macmon|openclaw-codex|@openclaw/codex" | tr '\n' ' ')
restore() { for p in $PIDS; do kill -CONT $p 2>/dev/null; done; echo "[restored co-tenants: $PIDS]"; }
trap restore EXIT INT TERM

echo "=== co-tenants before ==="; ps -Ao pid,pcpu,comm -r | head -5
echo "=== suspending: $PIDS ==="
for p in $PIDS; do kill -STOP $p 2>/dev/null; done

echo "=== fresh engine state ==="
omlx restart >/dev/null 2>&1
echo "=== settling 120s (thermal + idle) ==="
sleep 120
echo "=== load check ==="; ps -Ao pcpu,comm -r | head -4

OQ=$(ls -d ~/.cache/huggingface/hub/models--Jundot--Qwen3.8-27B-oQ4e-mtp/snapshots/*/ | head -1)
AB=$(ls -d ~/.cache/huggingface/hub/models--root4k--Huihui-Qwen3.8-27B-abliterated-oQ4e-mtp/snapshots/*/ | head -1)

echo "=== STOCK k=2, 5 reps (original tie-break protocol) ==="
BENCH_MODEL=qwen38-27b-oq4e  $VP bench_mtp.py --tag quiet_stock --model-path "$OQ" --out quiet_stock.json --repeats 5 2>&1 | grep '^{'
echo "=== ABLIT k=2, 5 reps ==="
BENCH_MODEL=qwen38-27b-ablit $VP bench_mtp.py --tag quiet_ablit --model-path "$AB" --out quiet_ablit.json --repeats 5 2>&1 | grep '^{'
echo QUIET_COMPLETE
