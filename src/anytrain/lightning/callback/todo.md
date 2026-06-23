# Callback TODO

## ModelCheckpoint

目标：提供一个和 Lightning 原生 `ModelCheckpoint` 参数基本一致的 callback，并增加一个显式开关 `async_save`。默认开启异步保存，用于目标 checkpoint 目录位于 NFS 等慢文件系统的训练场景。

### 接口

- 类名：`ModelCheckpoint`，和 Lightning 原生 callback 同名。
- 继承：`lightning.pytorch.callbacks.ModelCheckpoint`。
- 构造参数沿用原生 `ModelCheckpoint.__init__` 的顺序和默认值，在末尾增加 keyword-only 参数：
  - `async_save: bool = True`
- `async_save=False` 时不改变原生保存、删除、`save_last`、top-k 和 logger 通知行为。

### 异步保存链路

开启 `async_save` 时：

1. Lightning 仍在所有 rank 上调用 callback，checkpoint state 仍由原生 Trainer 负责构建。
2. rank 0 先把 checkpoint 写入本机临时目录，`trainer.save_checkpoint()` 的同步边界只覆盖本地写入。
3. 本地写入完成后，把“从本地临时文件复制到目标路径”的任务放入单线程后台队列。
4. 后台队列负责创建目标父目录，先复制到目标目录下的临时 `.part` 文件，再用 `os.replace()` 替换最终路径，避免暴露半写入文件。
5. callback 在 `on_fit_end`、异常保存结束和用户显式等待时检查后台任务；后台错误必须抛出，不能静默吞掉。

### 删除

top-k 删除进入同一个后台队列：

- 保存和删除按 Lightning 触发顺序串行执行，避免旧 checkpoint 的后台复制晚于删除完成并把文件写回来。
- `save_last="link"` 保持 Lightning 原生逻辑，不在第一版里额外接管。

### 边界

- 第一版只支持本地文件系统路径，包括挂载到本机的 NFS 路径；不支持 `s3://`、`gs://` 等远端 URI。
- 本地临时目录使用 Python `tempfile` 默认位置，依赖运行环境把 `TMPDIR` 指向本机磁盘。
- 只改变 checkpoint 文件落盘方式，不接管下游训练入口、resume 选择、logger 配置或分布式策略。

### 任务

1. Done: 新增 `ModelCheckpoint` 实现和公开导出。
2. Done: 增加同步兼容、异步复制和删除排队测试。
3. Done: 更新 Lightning 文档里的 callback 能力说明。
