# Tokenizer TODO

## CodecBPE Batch Support

批量输入单独设计，避免第一版把 ragged 行为写乱：

1. `repeat_interleave_packed(x, token_ids, dim=1)`：
   - 返回 concat 后的 `expanded_x`、`expanded_unit_ids`、`offsets`。
   - 适合后续接 pack/pad 或自定义 loss。
2. `repeat_interleave_padded(x, token_ids, dim=1, pad_id=0)`：
   - 返回 `padded_x`、`padded_unit_ids`、`mask`。
   - 每个 batch item pad 到本 batch 最大展开长度。
