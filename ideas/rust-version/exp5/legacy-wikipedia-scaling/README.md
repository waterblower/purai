# Exp5：英文 Wikipedia 100 MB / 1 GB 数据规模对照

迁移说明：本目录保留原有 exp5 实验，现位于 `exp5/legacy-wikipedia-scaling/`。新的程序学习实现位于 `exp5/code/`。历史报告、日志和校验清单保留原始路径；复现脚本中的新路径已调整，但历史产物的路径/二进制校验仍可能阻止直接续跑。本次迁移没有重新训练或覆盖产物。

## 预先确定的实验设计

以 exp4 v1 的固定 12 轮为主训练协议：不改变学习算法或容量/搜索预算，比较同源嵌套 100 MB 与 1 GB 训练数据。另报告 exp4 v1/v2 的 BLiMP 分数作为跨语料参考。v2 的提前停止策略不用于本轮。

数据选择：[wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)，`20231101.en`，固定 revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`。英文百科领域接近 WikiText，但不是相同的文章选择或格式，不能把 exp4→exp5 的变化只归因于数据规模。WikiText-103 本身不足以提供约 1 GB 独立正文。[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 与 [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) 也是可用候选，但会引入更大的领域变化。

数据卡列出的许可为 CC BY-SA 3.0 / GFDL。原始元数据存于 `source-info.json`；原始和转换后的训练数据、文档来源清单均放在 `../../data/`。本文的 MB/GB 为十进制 UTF-8 正文字节数，不包含 JSONL 转义开销。

### 数据划分

- 按固定 shard 和文章顺序读取，仅取 `text` 并去除首尾空白；不拼接标题，不改写标点或单词。
- 对处理后的全文计算 SHA-256，去除完全相同的重复正文。
- SHA-256 前 8 字节按大端无符号整数取模 100：0 为 dev，1 为 test，其余为 train。相同正文必在同一集合。
- 保留完整文章，train 累积至至少 1,000,000,000 字节，dev/test 各至少 5,000,000 字节。
- 100 MB 训练集是 1 GB 训练集的完整文章前缀，至少 100,000,000 字节。
- 这是源文件前缀样本，不是整个英文 Wikipedia 的均匀随机样本。文章长度和顺序可能影响代表性；没有近似重复去重或 BLiMP 污染审计。
- Dev 用于逐轮报告，不选轮数和模型；test 在训练完成后评估。BLiMP 不参与训练或参数选择。

### 训练与评测

两组相同参数：12 轮，max-depth 12，max-nodes 250000，candidates 100000，forks 10000，merges 64，min-count 4，structure-weight 0.25，merge-top 16，discount 0.75，theta 2；关闭匹配复制，不启用提前停止。程序在没有结构变化时仍可能自然结束。

保留同数据固定 byte 5-gram 的 dev BPB 基线。两组各评估独立 test BPB、完整 BLiMP、固定 seed 的 400 字节生成样本。BLiMP 比较完整句子的字节概率，无长度归一化，无复制，完全同分计错。主结论看同源两组的质量、时间、内存、模型大小；本实验不隔离 Fold 的贡献，也不强迫模型文件随数据量扩大十倍。

训练程序来自本轮冻结的 `source/`，开发 profile opt-level=2，与 exp4 一致；单线程 CPU。`run.py` 在计时前构建，直接调用二进制。内部 `training_seconds` 含逐轮 dev 验证；外部计时还含加载、保存/重载、最终 dev 验证与 byte 5-gram 基线，不含下载、预处理、编译、BLiMP 和独立 test。`measure.py` 在独立进程中运行一个子进程，用 monotonic 和 getrusage 记录墙钟、CPU 时间和峰值内存（macOS 字节）。100 MB 原先采用 `/usr/bin/time -l`，训练完成后计时工具因 sysctl 权限报错，保留有效墙钟/CPU 时间和训练结果，峰值内存缺失；原始记录一并保留。

## 复现

在 `rust-version/` 下执行：

```sh
sh exp5/legacy-wikipedia-scaling/prepare.sh
python3 exp5/legacy-wikipedia-scaling/validate_data.py
python3 exp5/legacy-wikipedia-scaling/run.py
python3 exp5/legacy-wikipedia-scaling/summarize.py
```

预处理与训练拒绝覆盖已存在的正式输出；训练脚本仅跳过已成功完成、命令及模型校验一致的训练。`prepare.sh` 的构建和下载日志位于本目录。需要 Rust/Cargo、curl、Python 3 标准库，无 Python 第三方依赖。

## 目录

- `prep/`：独立 Parquet 预处理工具，SHA-256 代码从本仓库 `lang/src/sha256.rs` 复制。
- `source/`：模型程序的本轮源代码快照。
- `100mb/`、`1gb/`：各自模型、训练日志、时间、BLiMP 汇总及逐题结果、test 评测和样本。
- `data-preparation.json`、`data-validation.json`、`sha256.json`：数据统计与校验。
- `comparison.json`、`RESULTS.md`：完成后生成的机器可读对照与报告。
- `build/`：忽略版本控制的编译产物。

实验已完成，见 [RESULTS.md](RESULTS.md)。100 MB / 1 GB 均完成 12 轮、独立测试和完整 67,000 对 BLiMP；核心模型代码未为本实验修改。

计时审计发现：1 GB 完整命令 UTC 起止跨度为 152.51 分钟，单调时钟为 33.03 分钟，CPU 时间为 32.62 分钟。差异原因未确认（可能涉及休眠或系统时钟调整）。报告保留三种时间，不能将单调计时当成本次实际日历等待时间。训练阶段的 23.98 分钟来自程序内部 `Instant`。
