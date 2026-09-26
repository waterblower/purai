# BLiMP 评测

使用 Rust 直接评估本项目模型，参考分数来自作者公开结果；不安装 Python，不运行或下载参考模型。

BLiMP 是 Warstadt 等人在 TACL 2020 发表的英语语法最小对基准，覆盖 67 个子任务、12 类现象，共 67,000 对句子。每对包含正确句和错误句，模型应给正确句更高的**整句概率**。

## 复现

以下命令从 `rust-version/exp1/` 执行。

```sh
sh scripts/download_blimp.sh

cargo run benchmark \
  --model output-tiny-story-100mb.model \
  --input ../data/benchmarks/blimp/data \
  --reference benchmarks/references/blimp-models-summary.jsonl \
  --report benchmarks/results/tiny-story-100mb.blimp.json \
  --samples benchmarks/results/tiny-story-100mb.blimp.samples.jsonl

cargo run benchmark \
  --model output.model \
  --input ../data/benchmarks/blimp/data \
  --reference benchmarks/references/blimp-models-summary.jsonl \
  --report benchmarks/results/tiny-story-10mb.blimp.json \
  --samples benchmarks/results/tiny-story-10mb.blimp.samples.jsonl
```

`--samples` 可省略；逐题结果较大，已被 gitignore 排除。`--reference` 可省略。默认要求完整 67 个子任务，每个包含不重复的 pairID 0–999；诊断小样本须显式加 `--allow-partial`，报告会标记 `full_suite=false`。

## 评分协议

- 每句话从空历史开始，用原始 UTF-8 字节计算 `bits = -sum(log2 P(byte | previous bytes))`；第一字节也参与评分。
- 正确句 bits 小于错误句时计对；完全相同计错，同时单独记录 ties。报告还记录相差小于 1e-10 bits 的 near ties，主准确率仍按严格比较计算。
- 不按字节数或单词数归一化；不加 BOS/EOS；不使用采样、temperature、匹配复制或提示词指令。模型参数不更新。
- 官方数据中的 `s-selection` 两个子任务，在作者分数表中属于 `argument_structure`。仅汇总标签按作者分类映射，句子和评分不改动。
- 参考分数按 UID 对齐，只聚合 67 个子任务原始行，跳过作者文件中已有的 `overall` 汇总行，避免重复计算。
- 主结果为 67,000 对的准确率；每个子任务均有 1,000 对，所以也等于子任务宏平均。另报告分类结果和字节长度诊断。

## 公开来源和限制

- [作者仓库与协议](https://github.com/alexwarstadt/blimp)。数据和分数固定到 revision `3e56b06fcabca9b30822fc66435fca6b1aa40bb1`，下载脚本检查 SHA-256。
- [作者公开分数](https://github.com/alexwarstadt/blimp/blob/3e56b06fcabca9b30822fc66435fca6b1aa40bb1/raw_results/summary/models_summary.jsonl)。使用修正后的分数；作者提醒过早期论文表格存在误报。
- [修订后的论文](https://arxiv.org/pdf/1912.00582)，表 3 和第 4 节解释结果与参考模型。公开 GPT-2 分数对应 **Large，774M 参数**，不能标作 TinyStories 或 GPT-2 Small。
- BLiMP 数据由 Alex Warstadt、Alicia Parrish、Haokun Liu、Anhad Mohananey、Wei Peng、Sheng-Fu Wang、Samuel R. Bowman 发布，遵循 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本目录保留原始参考数据；评测没有修改原始句子。

这些是外部能力参照，不是控制变量实验。论文的 5-gram 使用 Gigaword（3.1B tokens），LSTM 使用 Wikipedia（83M tokens），Transformer-XL 使用 WikiText-103（103M tokens），GPT-2 Large 使用 WebText（约 40 GB）。它们的训练域、数据量、tokenization、句子边界处理不同于我们的 byte 模型。

TinyStories 和 BLiMP 也存在领域与词汇差异；低分不能单独归因为某一种架构设计。当前模型没有 EOS 概率，评测的是给定长度文本的逐字节概率。按总概率比较对长度敏感，因此另附字节长度诊断，但不将诊断子集当作新的标准榜单成绩。

没有用该测试集训练、校准或选择新模型。后续若据此反复调参，应额外保留独立评测集。这里尚未运行 HellaSwag、LAMBADA 或生成质量评测，BLiMP 不能代表完整语言能力。

本次实测见 [RESULTS.md](RESULTS.md)。
