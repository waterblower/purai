# Exp4：100 MB WikiText-103 raw → BLiMP

目的：把前一轮的约 100 MB TinyStories 换成等体积的 WikiText-103 raw，保持学习算法和训练参数不变，比较完整 BLiMP 语法判断准确率。

后续 [v2 实验](v2/README.md) 沿用相同数据，最多训练 36 轮，在验证 BPB 首次上升时停止并保存上一轮模型；所有 v2 产物放在 `v2/`，本目录原有 v1 结果保留。

## 下载和预处理

在 `rust-version/` 目录执行：

```sh
sh exp1/scripts/download_wikitext_100mb.sh
```

使用现有 Rust/Cargo、curl、shasum、unzip 等命令行工具，无 Python 依赖。第一次预处理需要 Cargo 下载独立转换工具使用的 Parquet reader；依赖固定在 `exp1/tools/wikitext-prep/Cargo.lock`，不改变模型程序的依赖。

数据来源为 [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)，配置 `wikitext-103-raw-v1`，固定 revision `b08601e04326c79dfdd32d625aee71d232d685c3`。下载第一份训练 shard 和完整官方 validation shard。原始 ZIP 地址当前返回 403，所以使用该官方 Parquet 版本。来源和许可见 [固定版本数据卡](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md)（Wikipedia 派生数据，数据卡列有 CC BY-SA / GFDL）。

数据位于 `data/`，全部新数据文件由 gitignore 排除：

| 文件 | 用途 |
|---|---|
| `data/wikitext-103-raw-source/train-00000-of-00002.parquet` | 固定版本的原始训练 shard，约 157 MB |
| `data/wikitext-103-raw-source/validation-00000-of-00001.parquet` | 原始官方验证集 |
| `data/wikitext-103-raw-100mb.jsonl` | 5,648 篇文章，100,007,723 字节正文 |
| `data/wikitext-103-raw-validation.jsonl` | 60 篇文章，1,145,236 字节正文 |

预处理按原始顺序，以单层 `= Title =` 标题划分文章，保留子标题。每篇文章一个 JSONL 字符串；行尾缺换行时补换行，仅去除整篇首尾空白。累计正文达到 100,000,000 字节后，在完整文章边界停止，因此多出 7,723 字节。大小按解码后的 UTF-8 正文计算，不含 JSON 编码开销；MB 为十进制单位。

**raw 不代表完全未加工的网页原文**：该版本仍有 WikiText 自带的空格、标点和标题格式。我们没有去 tokenization 痕迹，也没有把单词替换成 `<unk>`。保留格式本身也是本次语料变化的一部分。文章清单、字节数、转换协议分别记录在 `data-train.json`、`data-validation.json`，脚本校验原始和生成文件的 SHA-256。

## 训练和评测

先确保 BLiMP 数据存在，再执行：

```sh
sh exp1/scripts/download_blimp.sh
sh exp1/scripts/run_exp4.sh
```

运行脚本拒绝覆盖现有模型，避免破坏本次结果。手动命令为：

```sh
cargo run --manifest-path exp1/Cargo.toml -- train \
  --input data/wikitext-103-raw-100mb.jsonl \
  --validation data/wikitext-103-raw-validation.jsonl \
  --output exp4/output-wikitext-103-raw-100mb.model \
  --rounds 12 --max-nodes 250000 --no-match

cargo run --manifest-path exp1/Cargo.toml -- benchmark \
  --model exp4/output-wikitext-103-raw-100mb.model \
  --input data/benchmarks/blimp/data \
  --reference exp1/benchmarks/references/blimp-models-summary.jsonl \
  --report exp4/blimp.report.json \
  --samples exp4/blimp.samples.jsonl
```

与 TinyStories 100 MB 实验相同：12 轮、最大深度 12、节点上限 250,000、候选上限 100,000、每轮新增上限 10,000、每轮合并上限 64、min-count 4、结构代价权重 0.25、discount 0.75、theta 2。复制关闭，无复制校准集，全部训练输入用于图结构学习。仍计算同数据的固定 5-gram 对照。

BLiMP 仍为 67,000 对句子、整句原始概率比较、无长度归一化、无匹配复制。语料验证集只用于报告，没有选择训练轮数或 checkpoint。训练、模型格式、预测和 BLiMP 评分代码均未为本次实验修改。

WikiText 使用自己的官方验证集，因此它的训练日志 BPB 与 TinyStories 日志 BPB **不能直接比较**。主要可比结果是相同 BLiMP 上的准确率；语料领域、文章长度分布和文本格式差异不能进一步拆开归因。没有对 BLiMP 做全文/近似重复污染审计。

## 产物

本次实验的模型、日志、报告均放在本目录：

- `output-wikitext-103-raw-100mb.model`：二进制模型（忽略 git）。
- `output-wikitext-103-raw-100mb.report.json`：训练参数、逐轮指标、模型大小和验证结果。
- `training.log`、`training.time.txt`、`training-*-utc.txt`：日志、完整命令实测 real/user/sys、起止时间。
- `blimp.report.json`、`blimp.samples.jsonl`、`blimp.log`：完整 BLiMP 汇总、逐题分数、日志。逐题文件忽略 git。
- `download-*.log`、`preparation*.log`、`data-*.json`：下载与转换记录。
- `sample.txt`、`inference.log`：固定提示词生成检查。
- `RESULTS.md`、`comparison.json`、`sha256.txt`：结果分析、对照和校验值。

`training_seconds` 包含逐轮验证；`training.time.txt` 的 real 还包含读取数据、保存/重载模型、最终验证和 5-gram 基线，不包含下载和数据转换。
