# Exp4 v2：最多 36 轮，验证 BPB 首次上升即停止

沿用 [v1](../README.md) 的 WikiText-103 raw 数据：5,648 篇训练文章、100,007,723 正文字节，以及 60 篇官方验证文章、1,145,236 正文字节。脚本检查两个文件的 SHA-256，未重新抽样或改变预处理。

本次从头训练。相对 v1 仅把轮数上限从 12 改为 36，并启用 `--stop-on-bpb-increase`。其余参数相同，包括最大深度 12、节点上限 250,000、关闭复制。36 轮不代表深度上限为 36。

停止规则：第 1 轮建立基准；之后每轮比较验证集的 `graph_only` BPB，首次严格大于上一轮就停止，并保存上一轮模型。使用完整浮点精度，相等继续，不设置容忍度或 patience。日志和 JSON 保留触发停止的那轮指标，报告的 `stopping.selected_round` 标明实际保存轮次。原有“本轮没有增长、合并或裁剪”停止条件仍有效。

验证集现在参与轮次选择，是开发集；其 BPB 不能再视为独立测试成绩。BLiMP 在模型确定后才运行，不用于选择 checkpoint。评分协议与 v1 相同：全部 67,000 对句子，比较整句字节概率，不做长度归一化，不使用匹配复制。

## 复现

在 `rust-version/` 执行，数据下载方式见 [v1 README](../README.md)：

```sh
sh exp1/scripts/run_exp4_v2.sh
```

脚本先构建，再记录训练命令的时间；拒绝覆盖已存在的 v2 模型。核心训练命令：

```sh
cargo run --offline --manifest-path exp1/Cargo.toml -- train \
  --input data/wikitext-103-raw-100mb.jsonl \
  --validation data/wikitext-103-raw-validation.jsonl \
  --output exp4/v2/output-wikitext-103-raw-100mb-v2.model \
  --rounds 36 --max-nodes 250000 --no-match --stop-on-bpb-increase
```

模型、逐轮报告、训练日志、计时、完整 BLiMP 报告和逐题分数、固定种子生成样例均保存在本目录。模型和 BLiMP 逐题分数由 gitignore 排除。`training_seconds` 包含逐轮验证和 checkpoint 复制；完整命令 real 还包含读取数据、保存/重载、最终验证和固定 5-gram 基线，不包含预先构建、下载或 BLiMP。

结果见 [RESULTS.md](RESULTS.md)。源代码测试、Clippy 记录分别为 `tests.log`、`clippy.log`；`sha256.txt` 记录本次代码、数据和主要产物，v1 的校验清单保留为当时的记录。
