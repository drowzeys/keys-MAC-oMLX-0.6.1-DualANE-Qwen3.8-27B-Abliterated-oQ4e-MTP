#!/usr/bin/env bash
# =============================================================================
# One-shot setup: Qwen3.8-27B abliterated oQ4e-MTP on oMLX 0.6.1
#                 with dual-ANE/GPU prefill + Lightning MTP k=3
#
#   bash oneshot-setup.sh              # full setup
#   bash oneshot-setup.sh --verify     # verify an existing install only
#   MODEL_REPO=... bash oneshot-setup.sh   # use a different oQ4e checkpoint
#
# Designed to be run unattended (by a human or an agent). Every step is
# idempotent and every failure is fatal with an explanation rather than a
# silent fallback — the whole point of this pack is that the ANE path fails
# QUIETLY when anything is off.
# =============================================================================
set -uo pipefail

MODEL_REPO="${MODEL_REPO:-root4k/Huihui-Qwen3.8-27B-abliterated-oQ4e-mtp}"
MODEL_ID="${MODEL_ID:-qwen38-27b-ablit}"
MTP_K="${MTP_K:-3}"
PORT="${PORT:-11500}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY_ONLY=0; [ "${1:-}" = "--verify" ] && VERIFY_ONLY=1

die() { echo "FATAL: $*" >&2; exit 1; }
ok()  { echo "  ✓ $*"; }
step(){ echo; echo "==> $*"; }

# --- 0. preflight ------------------------------------------------------------
step "Preflight"
[ "$(uname -s)" = "Darwin" ] || die "macOS only"
[ "$(uname -m)" = "arm64" ]  || die "Apple silicon only"
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)
ok "host: $CHIP, macOS $(sw_vers -productVersion)"
case "$CHIP" in *"M3 Ultra"*) ok "M3 Ultra: dual-ANE path supported" ;;
  *) echo "  ! $CHIP is not M3 Ultra — only ONE physical ANE, so dual-ANE gives less (or nothing)." ;;
esac
command -v brew >/dev/null || die "Homebrew required"

# --- 1. omlx -----------------------------------------------------------------
step "oMLX"
if ! command -v omlx >/dev/null 2>&1; then
  [ $VERIFY_ONLY -eq 1 ] && die "omlx not installed"
  brew tap jundot/omlx 2>/dev/null || true
  brew install omlx || die "brew install omlx failed"
fi
OMLX_VER=$(omlx --version 2>/dev/null | tr -d '[:space:]')
ok "omlx $OMLX_VER"
CELLAR="/opt/homebrew/Cellar/omlx/$OMLX_VER/libexec"
PY="$CELLAR/bin/python"
[ -x "$PY" ] || die "venv python not at $PY"
SITE="$CELLAR/lib/python3.11/site-packages"
[ -d "$SITE" ] || SITE=$(dirname "$($PY -c 'import omlx,os;print(os.path.dirname(omlx.__file__))')")
DST="$SITE/omlx/custom_kernels/qwen35_prefill"
[ -d "$DST" ] || die "kernel package dir missing: $DST"

# --- 2. native ANE kernel ----------------------------------------------------
step "Native ANE kernel"
kernel_ok() {
  ( cd "$CELLAR" && "$PY" -c "
from omlx.custom_kernels.qwen35_prefill import fast
import sys
sys.exit(0 if (fast.is_native_available() and fast.qwen35_ane_available()) else 1)" ) 2>/dev/null
}
if kernel_ok; then
  ok "already present and ANE-available"
else
  [ $VERIFY_ONLY -eq 1 ] && die "native kernel NOT available — ANE prefill would silently no-op"
  echo "  installing prebuilt artifacts (brew's bottle ships none)"
  ( cd "$HERE/prebuilt/qwen35_prefill" && shasum -a 256 -c SHA256SUMS >/dev/null 2>&1 ) \
    || die "prebuilt checksum mismatch"
  cp "$HERE"/prebuilt/qwen35_prefill/{_ext.cpython-311-darwin.so,*.metallib,libomlx_qwen35_prefill_kernel_ops.dylib} "$DST/" \
    || die "copy failed"
  MLXLIB=$("$PY" -c 'import os,mlx.core;print(os.path.join(os.path.dirname(mlx.core.__file__),"lib"))')
  [ -d "$MLXLIB" ] || die "mlx lib dir not found"
  for f in "$DST"/_ext*.so "$DST"/lib*_kernel_ops.dylib; do
    otool -l "$f" 2>/dev/null | grep -q "$MLXLIB" || {
      install_name_tool -add_rpath "$MLXLIB" "$f" 2>/dev/null
      codesign --force --sign - "$f" >/dev/null 2>&1
    }
  done
  kernel_ok || die "kernel still unavailable after install — ABI mismatch? Build from source: scripts/setup_qwen_only.py"
  ok "installed, rpath stamped, ANE available"
fi

# --- 3. model ----------------------------------------------------------------
step "Model: $MODEL_REPO"
if [ $VERIFY_ONLY -eq 0 ]; then
  command -v hf >/dev/null || pip install -q huggingface_hub 2>/dev/null || true
  hf download "$MODEL_REPO" --max-workers 8 >/dev/null || die "download failed"
fi
CACHE_DIR="$HOME/.cache/huggingface/hub/models--$(echo "$MODEL_REPO" | tr '/' '-' | sed 's/-/--/')"
SNAP=$(ls -d "$HOME"/.cache/huggingface/hub/models--*/snapshots/*/ 2>/dev/null \
       | grep -i "$(basename "$MODEL_REPO" | tr 'A-Z' 'a-z')" | head -1)
[ -n "$SNAP" ] || SNAP=$(ls -d "$HOME"/.cache/huggingface/hub/models--$(echo "${MODEL_REPO%%/*}")--$(echo "${MODEL_REPO##*/}")/snapshots/*/ 2>/dev/null | head -1)
[ -n "$SNAP" ] && [ -f "$SNAP/config.json" ] || die "snapshot not found for $MODEL_REPO"
ok "snapshot: $SNAP"

# ANE eligibility gate: gate/up MUST be 4-bit affine
"$PY" - "$SNAP" <<'PYEOF' || die "checkpoint is NOT ANE-eligible (needs 4-bit affine, group_size 64/128)"
import json, sys
q = json.load(open(sys.argv[1] + "/config.json")).get("quantization", {})
g = {k: v for k, v in q.items() if not isinstance(v, dict)}
okk = g.get("bits") == 4 and g.get("mode") == "affine" and g.get("group_size") in (64, 128)
print(f"  quant: {g}  -> ANE-eligible: {okk}")
sys.exit(0 if okk else 1)
PYEOF

mkdir -p "$HOME/.omlx/models"
ln -sfn "${SNAP%/}" "$HOME/.omlx/models/$MODEL_ID"
ok "registered as $MODEL_ID"

# --- 4. settings -------------------------------------------------------------
step "Settings (dual-ANE MLP-only + MTP k=$MTP_K)"
"$PY" - "$MODEL_ID" "$MTP_K" <<'PYEOF'
import json, pathlib, sys
mid, k = sys.argv[1], int(sys.argv[2])
p = pathlib.Path.home() / ".omlx/model_settings.json"
d = json.loads(p.read_text()) if p.exists() else {"version": "1.0", "models": {}}
d.setdefault("models", {})[mid] = {
    "qwen35_ane_prefill_enabled": True,
    "qwen35_ane_prefill_sequence_length": 2048,
    "qwen35_ane_prefill_fraction": 0.53,
    "qwen35_ane_prefill_max_layers": 64,
    "qwen35_ane_prefill_dual_ane": True,
    "qwen35_ane_prefill_gdn": False,      # OFF — see README §3, recurrent state + INT8 = bad
    "qwen35_ane_prefill_gdn_max_layers": 0,
    "mtp_enabled": True,
    "mtp_num_draft_tokens": k,
}
p.write_text(json.dumps(d, indent=2))
print(f"  wrote {p}")
PYEOF

# --- 5. restart (login shell! brew is not on PATH otherwise) -----------------
step "Restart server"
bash -lc "omlx restart" >/dev/null 2>&1 || die "restart failed"
sleep 20
ok "restarted"

# --- 6. verify ---------------------------------------------------------------
step "Verify"
curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q "\"$MODEL_ID\"" \
  || die "$MODEL_ID not served — model discovery only happens at STARTUP; re-run restart"
ok "served on :$PORT"
curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"hi\",\"max_tokens\":4,\"temperature\":0}" >/dev/null
LOG="$HOME/.omlx/logs/server.log"
if grep -aq "instance-pinned ANE programs" "$LOG"; then
  ok "$(grep -a 'instance-pinned ANE programs' "$LOG" | tail -1 | sed 's/.*- //')"
else
  die "ANE programs never compiled — check $LOG"
fi
grep -a "MTP\[" "$LOG" >/dev/null 2>&1 && ok "MTP active"

echo
echo "DONE. $MODEL_ID on :$PORT — dual-ANE prefill (MLP-only) + MTP k=$MTP_K"
echo "Benchmark:  BENCH_MODEL=$MODEL_ID $PY scripts/bench_mtp.py --tag t --model-path $SNAP --out /tmp/b.json --repeats 3"
echo "Effective k: tail ~/.omlx/logs/server.log for 'tok/cycle=' — it can never exceed k+1."
