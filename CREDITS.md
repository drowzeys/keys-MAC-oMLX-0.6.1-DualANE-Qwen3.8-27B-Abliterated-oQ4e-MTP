# Credits

This pack is an integration and measurement effort. Essentially all of the engineering it depends on
is other people's work. Credit where it belongs:

## Engine, quantization and the ANE path

**[jundot/omlx](https://github.com/jundot/omlx)** (Apache-2.0) — the oMLX inference server, the oQ
mixed-precision quantization system, Lightning MTP, and the experimental dual-ANE/GPU prefill that
this whole recipe exists to exercise. The prebuilt kernel in `prebuilt/` is compiled unmodified from
their v0.6.1 source.

From the [0.6.1 release](https://github.com/jundot/omlx/releases/tag/v0.6.1) specifically:

| Contribution | Author |
|---|---|
| Experimental dual-ANE/GPU prompt processing (#2756, hardened in #2760) | **@onthehub97** |
| Concurrent Qwen3.8 VLM throughput / batching over Lightning MTP (#2752) | **@DiscoStew6082** |
| Qwen3.8 vision patch-embedding load fix (#2754) | **@frank-beans** |
| Prefix-cache reuse across tool-adjacent system messages (#2753) | **@q-p** |
| Distributed version-mismatch fix (#2758) | **@hellodk** |
| DeepSeek V4 reasoning-effort aliases (#2724) | **@jonathan308** |

The ANE work is theirs; our contribution is independent verification on a second M3 Ultra, the
GDN-offload accuracy finding, and the verified-k methodology.

## Model lineage

- **[Qwen team / Alibaba](https://huggingface.co/Qwen/Qwen3.8-27B)** — the Qwen3.8-27B base model
  and its hybrid Gated DeltaNet + full-attention architecture.
- **[huihui-ai](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated)** — the abliteration.
- **[root4k](https://huggingface.co/root4k/Huihui-Qwen3.8-27B-abliterated-oQ4e-mtp)** — the oQ4e-mtp
  quantization of it, which is the checkpoint this recipe serves.
- **[Jundot](https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp)** — the stock oQ4e-mtp build used as
  the comparison baseline.

## Framework

- **[ml-explore/MLX](https://github.com/ml-explore/mlx)** (Apple) — the array framework and Metal
  backend everything here runs on, plus `mlx-lm` / `mlx-vlm`.
- **Apple** — the Neural Engine, and the private `AppleNeuralEngine.framework` runtime that oMLX's
  experimental path drives. Note that path uses undocumented APIs and can break on any macOS update.

## Prior art referenced in the analysis

- The upstream 0.6.1 release notes' own measurements, which we reproduce and in places exceed.
- oQ's published quality benchmarks (MMLU/TruthfulQA/HumanEval/MBPP across bit widths), used to
  reason about why oQ4 beats naive 4-bit.

## What is actually ours

The measurements, the failure analysis, and these specific findings:

1. Homebrew's poured bottle ships no native kernels, and the ANE path then fails **silently**.
2. `brew reinstall --with-custom-kernel` cannot work on macOS 26 because Homebrew filters
   `DEVELOPER_DIR` — plus the no-sudo out-of-band build that does work.
3. The GDN offload degrades numerics far past what upstream documents (logit cosine 0.901, top-token
   flip) while MLP-only matches their claim (0.99996) — and the mechanism: GDN feeds a recurrent
   state, so INT8 error compounds along the sequence.
4. oQ4e is the only quant level that preserves ANE eligibility.
5. Effective k must be verified from `tok/cycle` telemetry, not the config file.

Bugs are reported upstream where they are upstream's; items 1–2 are environment traps rather than
defects, and item 3 is a discrepancy we would welcome being told we measured wrong.
