# TinyStories 10 MB 实验

数据来自 [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)，固定版本
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`，使用原始 TinyStories（非 V2-GPT4）。

## 数据与评估协议

- 按原文件顺序读取完整故事，移除 `<|endoftext|>` 和故事两端空白；按 UTF-8 文本的 SHA-256 在训练集内、验证集内及跨集合去除完全重复故事。
- 训练：11,459 篇，故事正文共 10,000,592 字节；用两个换行连接后，实际训练输入为 10,023,508 字节。
- 验证：来自独立官方验证文件，1,286 篇，共 1,000,375 字节。
- 这是文件前缀子集，未随机抽样；不是全量 TinyStories 结果。未做近重复检测。
- AFRN 与 n-gram 使用完全相同的连续训练文本。验证时逐故事重置上下文；AFRN 的 match 状态也逐故事重置。验证数据不更新持久模型计数。
- 指标为总负对数概率除以 UTF-8 字节数，即 bits per byte（BPB，越低越好），按字节加权，非按故事取平均。不能直接与采用不同 tokenization 的 token perplexity 比较。
- 默认 AFRN 训练 20 轮；关闭 Fold 的对照仅将 `foldsPerRound` 改为 0。默认压缩开启、`structWeight=0`。两者均额外评估关闭 match 的情况。
- n-gram 测试上下文长度 0–10 字节，使用相同的 Pitman–Yor 平滑参数 `discount=0.75, theta=2`。AFRN 的混合参数 `kappa=2`。初次测试到 6 阶时最优结果位于范围上限，因此进一步扩展阶数。
- 生成样本采用 3 个固定提示词、温度 0.8/1.0、种子 42，每个续写 600 字节，所有样本都保存。
- 没有依据本次验证结果调参。两组 AFRN 并行训练，运行耗时受资源竞争影响，不能当作严格性能对比。

最初尝试逐故事训练，但原实现为每篇文档分配并累加完整上下文计数矩阵，随结构增大产生很高开销，因此中止并改用项目 CLI 的连续文本训练方式。中止日志单独保存，不混入正式结果。核心模型源码没有修改。

## 复现

从 `deno-version/` 目录执行（Python 仅使用标准库，下载需要网络）：

```sh
python3 bench/prepare_tinystories.py
deno run --allow-read --allow-write bench/tinystories.ts ngram
deno run --allow-read --allow-write bench/tinystories.ts default
deno run --allow-read --allow-write bench/tinystories.ts no-fold
python3 bench/summarize_tinystories.py
```

下载的 JSONL 在 `bench/data/tinystories10mb/`，已加入 gitignore。数据版本、文件 SHA-256、字节数和故事数记录在 `manifest.json`。

输出在 `bench/results/tinystories10mb/`：

- `default.data`、`no-fold.data`：模型，遵循项目原有 `*.data` 忽略规则。
- `default.json`、`no-fold.json`、`ngram.json`：最终指标和参数。
- `*-rounds.json`、`*.log`：逐轮训练记录；每轮验证值在该轮结构改动之前计算，最终结果以最终指标文件为准。
- `*-per-story.json`：逐故事的字节数和编码长度。
- `*-samples.json`：全部生成样本。
- `source-sha256.json`：模型源码及实验脚本的 SHA-256。
- `RESULTS.md`、`fold-bootstrap.json`：两组运行完成后生成的汇总、全部样本和 Fold 配对故事重采样结果。

脚本发现同名模型时会直接加载并重新评估；若需从头训练，请先将对应模型和结果移到其他目录。重用缓存评估时，`trainSeconds` 会为 `null`。
