#!/bin/bash
# k sweep with VERIFIED effective k: tok/cycle must be <= k+1, and we report it.
set -u
export PATH=/opt/homebrew/bin:$PATH
VP=/opt/homebrew/Cellar/omlx/0.6.1/libexec/bin/python
cd ~/omlx-qwen38
OQ=$(ls -d ~/.cache/huggingface/hub/models--Jundot--Qwen3.8-27B-oQ4e-mtp/snapshots/*/ | head -1)

for k in 2 3 5 8; do
  echo "########## k=$k ##########"
  python3 set_mtp.py $k
  omlx restart 2>&1 | tail -1              # login-shell PATH: restart actually happens
  sleep 18
  MARK=$(wc -l < ~/.omlx/logs/server.log)
  BENCH_MODEL=qwen38-27b-oq4e $VP bench_mtp.py --tag k${k}v --model-path "$OQ" --out kv_$k.json --repeats 4 2>&1 | grep '^{'
  echo "--- effective k evidence (tok/cycle, ceiling $((k+1))) ---"
  tail -n +$MARK ~/.omlx/logs/server.log | grep -a "MTP\[" | grep "tokens=256" | tail -4 | grep -oE "cycles=[0-9]+ tok/cycle=[0-9.]+ accept=[0-9/]+ \([0-9.]+%\)"
done
echo KSWEEP_COMPLETE
