# Prebuilt `qwen35_prefill` ANE kernel

These are the compiled artifacts that Homebrew's poured `omlx` bottle **does not ship**, and that
`brew reinstall --with-custom-kernel` **cannot build on macOS 26** (Homebrew filters `DEVELOPER_DIR`,
so `xcrun` never finds `metal`). Dropping these in is the difference between the dual-ANE path
running and silently never engaging.

Built from the Apache-2.0 [jundot/omlx](https://github.com/jundot/omlx) v0.6.1 source
(`omlx/custom_kernels/qwen35_prefill/csrc`) with `OMLX_WITH_CUSTOM_KERNEL=1`.

## Build provenance — these are ABI-tied, check before using

| | |
|---|---|
| oMLX | **0.6.1** (brew `jundot/omlx`) |
| Python | **3.11** (Homebrew `python@3.11`, the formula's venv) |
| MLX | **0.32.0** |
| nanobind | **2.13.0** (must match — a mismatch isolates the `mlx` NB_DOMAIN and every `mlx.core.array` is rejected at the type caster) |
| macOS | 26.6.1 |
| Xcode | 26.6 (Metal toolchain component installed) |
| Target | Apple silicon arm64; the **dual**-ANE path needs M3 Ultra (two physical ANE instances) |

If your oMLX/MLX/Python versions differ from the above, **build from source instead** —
`scripts/setup_qwen_only.py` does exactly that. Loading an ABI-mismatched extension will fail at
import, not silently, so it is safe to try.

Verify integrity: `shasum -a 256 -c SHA256SUMS`

## Why the rpath step matters

After copying these in you MUST add the mlx lib dir as an rpath and re-codesign. Homebrew's
post-install clean pass rewrites libmlx's `LC_ID_DYLIB` to an absolute Cellar path, which breaks the
install-name match the extension was linked against — the kernels then fail to `dlopen` at runtime
and prefill silently falls back to the GPU path (upstream issue #2233). `oneshot-setup.sh` does this
for you.
