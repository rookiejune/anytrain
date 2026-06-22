# TODO

utils 只放跨模块稳定复用的小工具。

1. 能放在具体模块旁边的 helper 不放进 `utils`。
2. 不迁移 deepaudio 的 audio/io/checker/rpc/patchifier/overlap 等领域工具。
3. 可考虑放通用 dict flatten/prefix、optional dependency error helper。
4. 保持 `registry.py`、`types.py` 作为明确支撑层，不把它们塞回 utils。
