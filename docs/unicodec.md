# UniCodec

`anytrain.codec.unicodec` 是对 `rookiejune/UniCodec` fork 的 optional 集成层。
它不进入 core import，只在用户显式安装 `unicodec` extra 并调用相关 API 时使用。

## 安装

UniCodec 源码维护在可安装 fork：

```bash
python -m pip install -e ../UniCodec
```

fork 推送后，下游可以直接安装 anytrain extra：

```bash
python -m pip install -e ".[unicodec]"
```

`unicodec` extra 依赖：

- `unicodec @ git+https://github.com/rookiejune/UniCodec.git`
- `huggingface-hub`

## 缓存路径

checkpoint 自动下载到缓存目录。路径优先级：

1. 显式传入 `cache_dir=...`
2. `HF_HOME/unicodec`
3. 如果 `HF_HOME` 未设置，anytrain 会设置
   `HF_HOME=${ANYTRAIN_HOME:-~/.anytrain}/huggingface`，再使用
   `$HF_HOME/unicodec`

远程服务器上推荐：

```bash
export ANYTRAIN_HOME=/mnt/pami202/zhuyin/.anytrain
```

默认 checkpoint 来自 Hugging Face 仓库 `Yidiii/UniCodec_ckpt` 的
`unicode.ckpt`。默认 config 使用 UniCodec fork 内打包的
`unicodec_frame75_10s_nq1_code16384_dim512_acousitic.yaml`。

## 使用

```python
from anytrain.codec.unicodec import UniCodec

codec = UniCodec.from_pretrained(device="cuda")

codes = codec.encode(
    audio,
    domain="0",
    bandwidth_id=0,
)
audio_out = codec.decode(codes, bandwidth_id=0)

features, codes = codec.encode_features(
    audio,
    domain="0",
    bandwidth_id=0,
)
audio_from_features = codec.decode_features(features, bandwidth_id=0)
```

`audio` 需要是 `[batch, time]` 的 24 kHz 单声道 waveform。`domain` 直接沿用
UniCodec 上游约定：`"0"` 表示 speech，`"1"` 表示 music，`"2"` 表示 general
audio。它选择的是单一 codebook 内的 domain-adaptive partition，不是
semantic/acoustic 分支。可以传单个 domain，也可以传长度等于 batch size 的
domain sequence。

`bandwidth_id` 沿用上游参数名，但默认 config 里的 `bandwidths` 都是 `6.6`：

```yaml
bandwidths: [6.6, 6.6, 6.6, 6.6]
adanorm_num_embeddings: 4
```

因此第一版只把它视为上游 decoder/backbone 的条件 id，默认固定传 `0`。不要把它理解为
已经验证可用的可变码率接口；后续如果确认 checkpoint 中 `0/1/2/3` 有明确语义，再把
它提升成更清楚的枚举。

UniCodec 不拆分 semantic / acoustic code。`encode()` 返回单一路径的离散
codes；`decode()` 接收同一组 codes 并重建 waveform。

上游实现内部还有连续 quantized features 边界。需要直接操作这个边界时使用
`encode_features()` / `decode_features()`；普通 codec roundtrip 应优先使用
`encode()` / `decode()`。

`local_files_only=True` 可以在离线环境中只使用已有缓存。
