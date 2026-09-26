# Rust 实验

```text
rust-version/
  exp1/       原始字节模型、代码、工具、早期实验和 BLiMP 报告
  exp4/       WikiText-103 100 MB 实验及 v2
  exp5/
    code/       新的离散程序学习 Rust 实现
    experiment/ 新实现的模型、日志、报告（尚未训练）
    readme.md   新实现说明
    legacy-wikipedia-scaling/ 原有 Wikipedia 规模实验，完整保留
  data/       共享训练、验证及 benchmark 数据，Git 忽略
```

没有已有的独立 exp2、exp3，因此暂不创建空实验。后续实验按 expN 添加。

从本目录使用旧模型程序：

```sh
cargo run --manifest-path exp1/Cargo.toml -- train --input data/train.jsonl --output exp1/new.model
```

新实现见 [exp5/readme.md](exp5/readme.md)。原有实验说明见 [exp1](exp1/README.md) 和 [exp4](exp4/README.md)。

数据下载脚本位于 `exp1/scripts/`：`download_blimp.sh`、`download_wikitext_100mb.sh`、`download_tinystories_100mb.py`。脚本已调整为写入共享 `data/`；具体数据来源与校验值见各实验 README。本次整理未下载数据、未运行训练。

迁移对应关系：原根目录代码及产物 → `exp1/`；`exp4-wikitext-103-raw-100mb/` → `exp4/`；原 `exp5/` → `exp5/legacy-wikipedia-scaling/`；`benchmarks/data/` → `data/benchmarks/`。历史日志、报告和校验清单中的旧路径保留为当时记录，不代表当前路径；历史二进制路径校验可能需要按此映射处理，不能直接覆盖原实验重新运行。
