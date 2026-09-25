# Rust 字节级结构学习原型

从一个根预测节点开始，根据训练数据增加上下文分支、合并预测相近的叶子、裁剪没有足够收益的分支。保存的模型包含学到的拓扑、整数证据和复制预测的校准表。推理时根据输入遍历不同路径，内部查找不要求产生额外输出。

这是一个可运行、可测量的语言建模实验。它学习的是可变深度的字节上下文及字节类，尚不具备任意程序归纳、算术算法发现或通用多步骤推理能力。

## 使用

需要 Rust 2024 edition 工具链（Rust 1.85 或更新）。从本目录运行：

```sh
cargo run train --input data.txt --output output.model
cargo run infer --model output.model --prompt "Once upon a time,"
```

使用本目录提供的数据：

```sh
cargo run train --input data/train.jsonl --output output.model
cargo run infer --model output.model --prompt "Once upon a time,"
```

`train.jsonl` 同目录下存在 `validation.jsonl` 时会自动使用它。也可以明确指定：

```sh
cargo run train --input data/train.jsonl --validation data/validation.jsonl --output output.model
cargo run eval --model output.model --input data/validation.jsonl
cargo run eval --model output.model --input data/validation.jsonl --no-match
cargo run inspect --model output.model --top 20
```

`cargo run train ...` 和 `cargo run -- train ...` 两种形式都支持。开发 profile 已启用优化，普通 `cargo run` 就能用于训练；不需要 GPU。首次构建需要取得 `serde_json` 及其依赖，之后可用 `--offline`。`Cargo.lock` 已固定此次构建的依赖版本。

完整参数：`cargo run -- --help`。

## 输入与数据隔离

- `.jsonl`：每行是一个 JSON 字符串，或者含字符串 `text` 字段的对象。JSON 解码后的 UTF-8 字节进入模型，每条记录是独立文档。
- 其他扩展名：按原始字节读取，整个文件是一篇文档；不要求 UTF-8。
- 没有独立验证文件时，默认保留末尾 10% 作为验证数据。多文档按完整文档数量划分，单文件按字节划分；边界两侧不共享上下文。
- 训练部分再保留末尾 5% 文档用于复制预测的可靠度校准。这部分不参与拓扑搜索和上下文计数。单文档输入则按字节划分。
- 验证集只报告指标，不选择候选结构、训练轮数或最佳 checkpoint，也不更新校准表。
- 每篇文档重新开始上下文和复制状态。验证的 bits/byte 按全部字节加权，不按故事取平均。

`--calibration-fraction 0` 不留校准集，全部训练部分用于结构学习。没有校准证据时，复制预测不会改变上下文分布。`--no-match` 同时关闭训练时的校准留出；推理和评估也有独立的 `--no-match` 开关。

## 训练如何改变拓扑

一个节点表示一类上下文，并保存其后续字节的稀疏整数直方图。根节点不看历史；向下走一条边，就向左多查看一个历史字节。边上的标签既可以是单字节，也可以是训练合并得到的字节集合。同一父节点下的集合不重叠，因此每次只需要走一条匹配路径。

每轮训练执行：

1. **提议分支**：扫描训练文档，寻找当前图尚未区分的前置字节条件。先统计支持次数，再为最常见的候选收集完整后续分布。
2. **评估并增加**：使用 Pitman–Yor 边际码长比较新增节点与父节点解释这些位置的成本。候选先验排除候选自身的计数，以免拿自己的数据证明自己有效。只有负成本变化的候选进入新增集合。
3. **共享叶子**：在同一父节点下，为预测最常见字节相同的叶子提出配对；通过边际码长和结构成本评估后，共享一个分布并合并边标签。合并后可在后续轮次继续学习更深的条件。
4. **裁剪**：重新检查叶子；不再值得单独保存的细化分支被删除，推理退回父节点。

这是有候选预算的贪心局部结构搜索，不保证找到全局最优图，也不是对任意程序进行搜索。每轮独立候选的估计收益不能直接当作整个模型的全局 MDL 改善。

拓扑修改不要求从头重新估计所有已有节点：新增节点精确计数，合并节点相加，裁剪不改变保留节点的计数。测试会在每一轮独立扫描文档，验证所有节点计数与重新计数一致。

默认限制：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--rounds` | 12 | 最大训练轮数 |
| `--max-depth` | 12 | 最多读取的历史字节数 |
| `--max-nodes` | 60000 | 根和上下文节点的总上限 |
| `--candidates` | 100000 | 每轮精确评分的新增候选上限 |
| `--forks` | 10000 | 每轮接受的新增分支上限 |
| `--merges` | 64 | 每轮接受的叶子合并上限，0 关闭 |
| `--min-count` | 4 | 新候选最少支持次数 |
| `--structure-weight` | 0.25 | 结构成本权重 |
| `--discount` / `--theta` | 0.75 / 2 | Pitman–Yor 平滑参数 |
| `--baseline-order` | 5 | 同条件固定 n-gram 对照，-1 跳过 |

模型可以增长也可以缩小；达到节点上限后仍可进行合并和裁剪。限额是明确的资源约束，不是事先规定每个上下文都必须有节点。

## 推理与生成

```sh
cargo run infer --model output.model --prompt "Once upon a time," --n 400 --seed 42 --temperature 0.85 --stats
cargo run infer --model output.model --prompt "Once upon a time," --n 20 --trace
cargo run infer --model output.model --prompt "Once upon a time," --n 400 --max-steps 4 --no-match
```

- `--n` 是生成**字节**数，默认 400；温度 0 使用确定性的最大概率选择。
- `--seed` 固定采样随机数，默认 42；`--top-p` 默认 1，可以额外限制采样分布。
- 沿图向左匹配历史，使用最深命中节点的平滑分布；节点证据少时，由父分布提供回退概率。
- 可选复制预测只读取本次输入中已出现的字节。匹配器是固定机制，是否可信由训练侧留出数据中的整数校准表决定；它不是从数据中归纳出的复制程序。
- `--max-steps` 限制每个字节的上下文遍历工作，最低为 1（只用根分布）。它**不是整个推理过程的 CPU 指令预算**，也不限制复制匹配器；匹配器的新匹配回看最多 64 字节。
- `--stats` 报告根访问与槽位测试次数；`--trace` 显示实际命中的条件、支持次数和复制状态。它们均写入 stderr。
- 遍历可以执行多步而不输出，最后才产生一个字节。当前模型的路径深度仍有上限，并不具备学习出的任意循环或搜索程序。

默认 stdout 原样输出提示词、生成字节及一个换行。字节采样不保证结果是合法 UTF-8。处理二进制数据时可用：

```sh
cargo run infer --model output.model --prompt-file prefix.bin --n 1024 --output continuation.bin
```

`continuation.bin` 只包含生成的原始字节，不含提示词和额外换行。

## 模型文件

自定义格式 v1，全部整数和浮点按小端编码：

1. 8 字节 magic `GBGRAPH\n` 和 `u32` 版本；
2. 两个 `f64` 平滑参数、`u64` 结构训练与校准字节数；
3. 256 个 `u64` 命中计数及 256 个 `u64` 总计数；
4. `u32` 节点数；每个节点保存父索引、256-bit 边标签、稀疏 `(u8 字节, u64 次数)` 直方图；
5. 全部前置字节的 64-bit FNV-1a 校验值。

父节点先于子节点写入，根父索引为 `u32::MAX`。加载时校验版本、长度、计数、拓扑、边标签互斥和校验值。FNV 用于检测意外损坏，不提供密码学认证。派生的路由表、平滑分布、生成历史和输入原文不保存在模型里。

保存使用同目录临时文件和 rename，避免写入失败破坏已有模型。本格式不兼容 Deno 版本；目前没有模型间合并、CRDT 同步、文档遗忘或从保存模型继续训练的命令。

## 已完成的验证

详见 [RESULTS.md](RESULTS.md)、[output.report.json](output.report.json)、[validation-run.json](validation-run.json) 和 [samples.json](samples.json)。

```sh
cargo test --offline
cargo clippy --offline --all-targets -- -D warnings
cargo fmt --all -- --check
```

测试覆盖实际 CLI 工作流、结构增长及未见句子预测、合并和裁剪、每轮精确计数、概率归一化、模型往返与损坏检测、随机种子复现、字节输入、文档隔离、验证数据不影响模型、未来字节不可见，以及执行预算与输出长度分离。

当前验证支持的是一个受限的结构学习原型。长期语义一致性、程序发现、可学习停止策略、持续学习中的遗忘控制，以及相对 Transformer 的能力或效率优势，都尚未得到证明。
