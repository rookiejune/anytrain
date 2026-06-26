# Module TODO

## Qwen3

- Expose SDPA/flash-attention knobs only after a real downstream training need.
- Add a codec-specific token-forward wrapper only if `anycodec` needs a narrower contract than HF `Qwen3Model`.
- Design KV cache semantics separately from training forward.

## Adaptive Dirichlet Tempering

- Support non-uniform priors for expert capacities or intentionally biased routing targets.
- Add optional top-k or hard routing support, including gumbel top-k and straight-through variants.
- Expose optional auxiliary losses such as KL-to-prior, entropy bonus, or load-balance loss without automatically wiring them into task loss.

- Extend `diagnostics()` with entropy, perplexity, min/max load, dead expert count, and load imbalance ratio.
- Add tests for diagnostic values after skewed routing batches.
