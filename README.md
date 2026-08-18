# oMLX 0.6.1 · Dual-ANE/GPU Prefill + Lightning MTP
## Qwen3.8-27B **Abliterated** oQ4e-MTP on Mac Studio M3 Ultra

Reproduction pack for the upstream [oMLX 0.6.1 release](https://github.com/jundot/omlx/releases/tag/v0.6.1)
recipe, with the stock Qwen3.8-27B swapped for the **abliterated** build:

| | |
|---|---|
| Host | Mac Studio **M3 Ultra**, 256 GB unified, macOS 26.6.1, Xcode 26.6 |
| Engine | `omlx` 0.6.1 (brew tap `jundot/omlx`), served on `:11500` |
| Model | [`root4k/Huihui-Qwen3.8-27B-abliterated-oQ4e-mtp`](https://huggingface.co/root4k/Huihui-Qwen3.8-27B-abliterated-oQ4e-mtp) — 17.0 GB |
| Quant | oQ4e: 4-bit affine gs64, **166 tensors promoted to 5-bit** (all `linear_attn.in_proj_a/b` — the GDN inputs) |
| Accel | dual-ANE prefill (**MLP-only**) + Lightning **MTP k=3** |

> **Why oQ4e specifically.** The ANE kernel gates on `mode == "affine"` **and gate/up at exactly
> 4 bits** (`_affine_spec(gate, allowed_bits=(4,))`). oQ5e/oQ6e/oQ8e and every mxfp8/int8 build fail
> that gate, so the dual-ANE path silently no-ops on them. oQ4e is the only level that keeps it.


---

## Results — measured 2026-08-18, Mac Studio M3 Ultra

**Config: `root4k/Huihui-Qwen3.8-27B-abliterated-oQ4e-mtp` · dual-ANE prefill (MLP-only, GDN off) · Lightning MTP k=3**

| | | |
|---|---:|---|
| **Decode — code-edit** | **81.1 tok/s** | reps 77.4 / 83.9 / 81.9 · `tok/cycle` 3.56–3.82 · accept 99.5–100% |
| **Decode — prose** | **59.4 tok/s** | reps 61.3 / 59.9 / 57.0 · accept 65–78% |
| **Prefill 16K** | **394.7 tok/s** | +21.6% vs GPU (upstream claims +17.7%) |
| **Prefill 32K** | **371.0 tok/s** | +22.5% vs GPU (upstream claims +18.9%) |
| **Prefill 4K** | **368.0 tok/s** | +10.3% vs GPU (upstream claims +1.3–3.4%) |
| **Concurrency** | **78.0 tok/s** agg @ C=4 | saturates ~84 by C=8 |
| **vs stock (non-ablit)** | **parity** | 71.0 vs 71.0 code-edit, 53.0 vs 55.2 prose, identical prompts |
| **Baseline, no accel** | 35.4 / 35.8 tok/s | code-edit / prose, MTP off |

**Net: 2.3× decode on code-heavy work and +22% prefill at long context, at zero quality cost from the
abliteration.** Single stream, 2K padded context, `bench_mtp.py`.

Setup is one command — see [`oneshot-setup.sh`](oneshot-setup.sh); the prebuilt ANE kernel ships in
[`prebuilt/`](prebuilt/) because Homebrew's bottle does not include it and macOS 26 cannot rebuild it
without the workaround in §1. Credits in [CREDITS.md](CREDITS.md).

---

## 1. Install — and the trap that makes or breaks this

```bash
brew tap jundot/omlx && brew install omlx      # or: brew upgrade omlx
```

**The poured bottle ships NO native kernels.** Verify before trusting any ANE result:

```bash
PY=/opt/homebrew/Cellar/omlx/<ver>/libexec/bin/python
cd /opt/homebrew/Cellar/omlx/<ver>/libexec
$PY -c 'from omlx.custom_kernels.qwen35_prefill import fast; \
        print(fast.is_native_available(), fast.qwen35_ane_available())'
```

`False` means ANE prefill will never engage — no error, just the slow path.

`brew reinstall --with-custom-kernel omlx` is the documented fix, **but it fails on macOS 26**:
`xcrun: unable to find utility "metal"`. Root cause: `xcode-select` points at CommandLineTools and
**Homebrew filters `DEVELOPER_DIR`**, so exporting it does not help (`HOMEBREW_NO_ENV_FILTERING=1`
does not help either). Without passwordless sudo you cannot `xcode-select -s`.

### Fix that works, no sudo — build the one kernel out-of-band

```bash
curl -sSL -o src.tgz https://github.com/jundot/omlx/archive/refs/tags/v0.6.1.tar.gz && tar xf src.tgz
cd omlx-0.6.1
$PY -m pip install "nanobind==2.13.0"
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export OMLX_WITH_CUSTOM_KERNEL=1
$PY scripts/setup_qwen_only.py build_ext --inplace          # see scripts/
SITE=/opt/homebrew/Cellar/omlx/<ver>/libexec/lib/python3.11/site-packages
DST=$SITE/omlx/custom_kernels/qwen35_prefill
cp omlx/custom_kernels/qwen35_prefill/{_ext*.so,*.metallib,lib*_kernel_ops.dylib} $DST/
MLXLIB=$($PY -c 'import os,mlx.core;print(os.path.join(os.path.dirname(mlx.core.__file__),"lib"))')
for f in $DST/_ext*.so $DST/lib*_kernel_ops.dylib; do
  install_name_tool -add_rpath "$MLXLIB" "$f"; codesign --force --sign - "$f"
done
```

The rpath + re-codesign step is **mandatory** — Homebrew's post-install clean pass rewrites libmlx's
`LC_ID_DYLIB`, which breaks dlopen and silently drops you back to the GPU path (upstream #2233).

> ⚠️ **Every `brew upgrade omlx` drops the kernel again.** Re-run this after any upgrade.

---

## 2. Configure

`~/.omlx/model_settings.json` (see `docs/model_settings.reference.json`):

```json
"qwen38-27b-ablit": {
  "qwen35_ane_prefill_enabled": true,
  "qwen35_ane_prefill_sequence_length": 2048,
  "qwen35_ane_prefill_fraction": 0.53,
  "qwen35_ane_prefill_max_layers": 64,
  "qwen35_ane_prefill_dual_ane": true,
  "qwen35_ane_prefill_gdn": false,
  "qwen35_ane_prefill_gdn_max_layers": 0,
  "mtp_enabled": true,
  "mtp_num_draft_tokens": 2
}
```

Register the checkpoint and restart:

```bash
ln -sfn <hf-snapshot-dir> ~/.omlx/models/qwen38-27b-ablit
omlx restart      # NOTE: needs a login shell — non-login shells lack brew on PATH
```

**Model discovery only happens at startup.** A new symlink added while the server runs will 404
forever, even though settings changes hot-reload. Confirm activation in `~/.omlx/logs/server.log`:

```
Eagerly compiled 64 MLP and 0 GDN procedures into two instance-pinned ANE programs (sequence_length=2048)
Qwen ANE prefill scheduler aligned to fixed shape 2048
```

Two programs = one pinned per physical ANE, GPU running the quantized suffix concurrently.

---

## 3. ⚠️ Turn the GDN offload OFF

Upstream ships `qwen35_ane_prefill_gdn` and its own docs claim hidden/logit cosine ≈ 0.9999 with the
top token unchanged. **That does not reproduce here.** Measured on the 2048-token 64-layer body:

| Config | body PP | speedup | hidden cos | logit cos | RMSE | top token |
|---|---:|---:|---:|---:|---:|---|
| GPU baseline | 339.2 | 1.000× | — | — | — | 220 |
| MLP + GDN | 472.0 | 1.391× | 0.978 | 0.901 | 0.357 | **146 ✗** |
| **MLP-only** | 431.8 | **1.273×** | **0.9996** | **0.99996** | **0.047** | 220 ✓ |

GDN accelerates the Gated DeltaNet `z+qkv` input projections, which feed a **recurrent** state — INT8
error there compounds along the sequence instead of staying local like an MLP block's does. The
quantizer independently agrees: oQ4e promotes exactly those 166 `linear_attn.in_proj_*` tensors to
5-bit. Dropping the GDN half costs ~6 points of server-level PP and buys back vendor-grade numerics.

---

## 4. Measured (this pack's `results/`)

### Prefill — dual-ANE vs stock GPU (PP = prompt_tokens / TTFT, cold cache)

| Prompt | GPU | MLP-only ANE | Δ | upstream claim |
|---:|---:|---:|---:|---:|
| 4K | 333.7 | 368.0 | +10.3% | +1.3–3.4% |
| 16K | 324.5 | 394.7 | **+21.6%** | +17.7% |
| 32K | 302.9 | 371.0 | **+22.5%** | +18.9% |

TTFT −9/−18/−18%; end-to-end −7/−17/−17%; decode unchanged (ANE is prefill-only).
ANE costs ~28 s eager compile at load and ~8 GB resident (23.4 → 24.7 GB).

**ANE only fires on prefill chunks of exactly 2048 tokens.** Short prompts get zero benefit; long
agent system prompts get the most (a 20.5 K-token hermes preamble = ten full ANE blocks).

### Lightning MTP sweep — **verified k** (decode tok/s, 2K ctx, single stream, 4 reps)

| k | prose | code-edit | code-edit acceptance |
|---:|---:|---:|---|
| off | 35.8 | 35.4 | — |
| 2 | **58.2** | 71.4 | ~100%, but `tok/cycle` pinned at the k+1 = 3 ceiling |
| **3** | 55.2 | **81.5** | 98–100%, `tok/cycle` ~4.2 |
| 5 | 53.0 | **82.5** | 98–100%, `tok/cycle` ~4.6–4.8 |
| 8 | 48.8 | 77.9 | regresses |

**Optimal k is per-task.** Code-edit runs at 98–100% draft acceptance, so deeper drafts keep paying
up to k=5; prose sits at 65–78% acceptance, where the extra draft slots are wasted work and k=2 wins.
**Standing config = k=3** — within 1% of k=5 on code, cheaper on prose, and fewer draft slots per
request under concurrency.

> #### Verify effective k from telemetry, not from the config file
>
> `~/.omlx/logs/server.log` emits one line per request:
>
> ```
> MTP[8] finish=length tokens=256 cycles=87 tok/cycle=2.94 accept=167/167 (100.0%)
> ```
>
> `tok/cycle` **cannot exceed k+1**. If it does, the k you think you set is not the k that ran.
>
> This cost us hours. An earlier sweep called `omlx restart` from a **non-login shell**, where brew
> is not on PATH — so the restart silently no-op'd, the k change lagged, and every arm measured at an
> effectively higher k. That produced a bogus "k=2/3/5 all tie at ~83" result and a wrong k=2
> recommendation. Always run the restart through `bash -lc`, and read `tok/cycle` back afterwards.
>
> Same root cause bites model registration: **new model directories are only discovered at startup**,
> so a symlink added while the server runs will 404 forever even though settings changes hot-reload.

### Abliterated vs stock — identical prompts, back-to-back

| task | stock oQ4e | **ablit oQ4e** |
|---|---:|---:|
| code-edit | 71.0 | **71.0** |
| prose | 53.0 | **55.2** |

**The abliteration costs nothing.** Identical on code-edit, marginally ahead on prose.

### Concurrency — ablit @ k=3, warm (aggregate tok/s, per-request in parens)

| C | prose | coding | edit |
|---:|---|---|---|
| 1 | 48.7 (48.7) | 60.9 (60.9) | 56.6 (56.6) |
| 2 | 55.7 (27.9) | 55.6 (27.9) | 47.4 (23.7) |
| 3 | 68.6 (22.9) | 69.4 (23.2) | 56.7 (18.9) |
| 4 | 76.2 (19.2) | 78.0 (19.6) | 62.8 (15.8) |

Coding/edit dip at C=2 — batching overhead exceeds the gain at two streams — then scale cleanly from
C=3. A wider sweep at k=2 saturated at ~84 aggregate by C=8 and gained nothing to C=16.

**Always warm the model before benchmarking.** The first request after a reload pays the model load:
a cold C=1 prose run measured 10.2 tok/s (25 s wall) versus 48.7 warm. Fire one throwaway request
first, and discard rep 0.

---

## 5. Honest caveats

- **Decode varies ~±10% run to run** on a desktop machine with background load (Spotlight/XProtect
  indexing freshly downloaded weights is a real contributor). Always A/B within one session, and
  discard rep 0 after any reload. Suspending co-tenants (EXO, agent workers) via SIGSTOP/SIGCONT
  changed nothing measurable here — a ~15% swing we initially blamed on co-tenancy turned out to be
  the effective-k bug above.
- **Neither ANE nor MTP is bitwise-lossless.** Even MLP-only at logit cosine 0.99996, one early
  near-tie flip diverges a greedy generation. Coherent and correct, but not identical text — **do not
  use output hashes as a regression gate.**
- **Thinking defaults ON at `reasoning_effort: xhigh`** via the chat template. MMLU-style harnesses
  with small token budgets will truncate mid-`<think>` and score near chance (measured 52% until
  fixed). Pass `chat_template_kwargs: {"enable_thinking": false}` or raise the budget (17 k is
  comfortable; the cap costs nothing measurable at conc 1 or 4).
- The over-refusal probe (10 benign-but-commonly-refused prompts) scored **0/10 refusals for both
  stock and ablit** — a floor effect, so it cannot detect this abliteration. Not evidence either way.

## Layout

```
scripts/   setup_qwen_only.py (kernel build) · set_cfg.py · bench_mtp.py · bench_pp.py
           mac_conc_bench.py (concurrency) · refusal_probe.py · eval_ab.py
results/   raw JSON + offline kernel bench logs for every table above
docs/      model_settings.reference.json
```
