# TinyStories 100 MB 实验

已下载并训练，退出码为 0，模型可正常重载和生成。

## 数据和命令

来源：[官方 TinyStories 原版训练集](https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStories-train.txt)；许可证 CDLA-Sharing-1.0。下载按源文件顺序取完整故事，以 `<|endoftext|>` 分隔并移除故事边界空白，转换为 JSONL 字符串。正文达到 100 MB 后保留最后一篇完整故事。

- 输入：`data/tiny-story-100mb.jsonl`，113,035 篇，100,000,007 字节正文；JSONL 文件为 101,326,996 字节。
- 验证：现有 `data/validation.jsonl`，1,286 篇，1,000,375 字节。
- 下载期间对边界去空白的故事做 SHA-256 比较，未发现与验证集完全重复的故事；未检测近似重复。
- 下载耗时约 12.94 秒，不计入训练耗时。

```bash
cargo run train --input data/tiny-story-100mb.jsonl --validation data/validation.jsonl --output output-tiny-story-100mb.model --rounds 12 --max-nodes 250000 --no-match
```

关闭匹配复制，全部训练正文用于图结构学习，不留复制校准集。节点上限从默认 60,000 提高至 250,000。其他设置保持默认，包括候选上限 100,000、每轮新增上限 10,000、每轮合并上限 64、上下文深度上限 12。

## 实测结果

| 项目 | 结果 |
|---|---:|
| 各轮结构修改 seconds 之和（不含逐轮验证） | 110.64 秒 |
| 程序 training_seconds（含逐轮验证及最终空校准调用） | 112.22 秒 |
| 完整命令墙钟时间 | 139.54 秒 |
| 完整命令峰值内存（macOS ru_maxrss） | 1.98 GiB |
| 模型文件大小 | 10,662,954 字节 / 10.17 MiB |
| 节点数 / 合并类别数 / 深度 | 96,729 / 577 / 12 |
| 独立验证 graph_only BPB | 1.27081008 |
| 相同训练数据的固定 5-gram BPB | 1.39063686 |

完整命令包括 cargo 启动、输入读取、训练、逐轮验证、模型保存和重载、最终验证及 5-gram 对照，不含下载。因未校准复制，with_match 与 graph_only 指标相同；本实验以 graph_only 为主要指标。

训练在第 12 轮上限停止，最后一轮仍新增 9,665 个节点，且候选数量达到上限，并非收敛。相比旧 10 MB 实验，还改变了复制校准划分和节点上限，因此不是严格单变量对照；BPB 不代表语法和长文本连贯性。

## 文件和验证

- `output-tiny-story-100mb.model`：已训练模型。
- `output-tiny-story-100mb.train.log`：完整日志。
- `output-tiny-story-100mb.report.json`：训练设置、每轮指标与最终评估。
- `output-tiny-story-100mb.run.json`：开始/结束时间、耗时、内存、模型校验值。
- `output-tiny-story-100mb.source-sha256.json`：代码及数据的校验值。
- `output-tiny-story-100mb.sample.txt`：以下固定提示词的生成结果。
- `scripts/download_tinystories_100mb.py`：数据下载和转换脚本。

训练自动验证了保存后重载的模型与内存模型完全一致。以下生成命令已成功执行：

```bash
cargo run infer --model output-tiny-story-100mb.model --prompt "Once upon a time," --no-match --seed 42 --n 400
```

模型 SHA-256：`89f97a08efa9569b6eff8340e1ca82ff77acbce2ea7d4916c166a54ce4e63230`。
