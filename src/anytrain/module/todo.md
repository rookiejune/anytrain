# Module TODO

## Qwen3

- Done: add thin Qwen3 helpers that reuse Hugging Face `transformers` modules instead of local transformer implementations.
- Done: keep `anytrain.module` import lightweight; resolve Qwen3 classes lazily only when a Qwen3 helper is used.
- Done: add per-module minimal builders for RMSNorm, RoPE, attention, MLP, decoder layer, and full `Qwen3Model`.
- Later: expose SDPA/flash-attention knobs only after a real downstream training need.
- Later: add a codec-specific token-forward wrapper only if `anycodec` needs a narrower contract than HF `Qwen3Model`.
- Later: design KV cache semantics separately from training forward.

## Adaptive Dirichlet Tempering

### P0: training stability

- Done: add `temperature_warmup_steps` so ADT can collect router statistics while keeping temperature at 1.0 during early training.
- Done: add `min_temperature` and `max_temperature` bounds to avoid over-sharpening or over-flattening router probabilities.
- Done: add optional `temperature_smoothing_decay` to reduce batch-to-batch temperature jitter.

### P1: real training semantics

- Done: add mask support for sequence/token logits so padding or invalid positions do not affect expert utilization statistics.
- Done: add optional DDP stats synchronization. When enabled, all-reduce router means and squared means before EMA updates.
- Done: add explicit stats lifecycle controls: `freeze_stats()`, `unfreeze_stats()`, `reset_stats()`, and a `collect_stats` forward override.

### P2: richer routing behavior

- Support non-uniform priors for expert capacities or intentionally biased routing targets.
- Add optional top-k or hard routing support, including gumbel top-k and straight-through variants.
- Expose optional auxiliary losses such as KL-to-prior, entropy bonus, or load-balance loss without automatically wiring them into task loss.

### P3: monitoring

- Extend `diagnostics()` with entropy, perplexity, min/max load, dead expert count, and load imbalance ratio.
- Add tests for diagnostic values after skewed routing batches.

## Quantization

设计计划见 `../../../docs/quantization-migration.md`。

### P1: FSQ

- Done: add `anytrain.module.quantization` package.
- Done: add unified quantize output dataclasses.
- Done: port FSQ with identity/linear projection only.
- Done: rename FSQ config field from `levels_per_codebook` to `levels`.
- Done: make FSQ `indices` single-codebook flat ids and expose levels conversion helpers.
- Done: make FSQ `codebook_size` equal to `prod(levels)`.
- Done: remove FSQ `num_codebooks` config so its output shapes match vanilla VQ.
- Done: warn on even FSQ `levels` and recommend odd levels for symmetric scalar grids.
- Done: make FSQ default level presets odd-only so the default config stays warning-free.
- Done: add FSQ `bound_scale` to reduce tanh saturation from oversized projected latents.
- Done: add shape and round-trip tests.

### P2: VQ

- Done: port vanilla VQ after fixing output naming and EMA/non-EMA training branches.
- Done: name the implementation `EmbeddingVectorQuantizer` to avoid protocol/class ambiguity.
- Done: add explicit `normalize_latents` config for l2-normalized lookup.
- Done: add scalar loss and gradient-flow tests.

### P2.5: GVQ

- Done: add `GVQConfig` and `GroupedVectorQuantizer`.
- Done: represent learned product codebooks with `group_sizes`, such as `(90, 90)` for a flat `codebook_size` of 8100.
- Done: keep GVQ output shapes aligned with FSQ/VQ.
- Done: add group/flat index round-trip, shape, gradient, and lookup tests.

### P3: RVQ

- Done: port RVQ with `QuantizeOutput`.
- Done: prefer uniform codebook dim in the first version.
- Done: add train/eval dropout and `num_active_codebooks` tests.
