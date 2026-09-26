# Exp5 结果：英文 Wikipedia 100 MB → 1 GB

同源嵌套语料、相同模型程序与固定训练设置下，完整 BLiMP 从 **53.9463%** 到 **54.6030%**，变化 **+0.6567 个百分点**。

## 同源规模对照

| 指标 | 100 MB | 1 GB |
|---|---:|---:|
| 训练正文 bytes | 100,031,616 | 1,000,005,849 |
| 训练文章 | 12,384 | 234,404 |
| 平均文章正文字节 | 8077.5 | 4266.2 |
| 完成轮数 | 12 | 12 |
| 节点数 | 105,177 | 109,503 |
| 共享字节类别 | 469 | 410 |
| 模型 MiB | 16.360 | 30.613 |
| 训练阶段 Instant 秒数（含逐轮 dev 验证） | 139.220 | 1438.940 |
| 完整命令单调时钟秒数 | 188.988 | 1981.587 |
| 完整命令 UTC 起止跨度秒数 | 188.991 | 9150.308 |
| 完整命令 CPU 秒数（user + sys） | 185.720 | 1957.192 |
| 峰值内存 GiB（完整训练命令） | 未取得 | 5.072 |
| dev graph BPB | 2.109804593 | 2.041043895 |
| dev byte 5-gram BPB | 2.103073997 | 1.981633496 |
| 独立 test graph BPB（CLI 六位小数） | 2.124154 | 2.041662 |
| BLiMP 正确 / 67,000 | 36,144 | 36,584 |
| BLiMP 准确率 | 53.9463% | 54.6030% |

数据量为 9.997 倍；训练阶段 Instant 计时为 10.336 倍；完整命令单调计时为 10.485 倍；模型文件为 1.871 倍。这是一次运行，不是重复计时的稳定性估计。

**计时异常（1gb）**：原始 UTC 起止时间跨度为 152.51 分钟，单调时钟记录 33.03 分钟，差 119.48 分钟。进程 CPU 时间为 32.62 分钟。未确认是否发生休眠或系统时钟调整。性能比较使用程序 Instant、外部 monotonic 和 CPU 时间，不能把单调计时直接称为本次实际日历等待时间。原始起止时间和计时值均保留在 training.run.json，未回填或修改。

BLiMP 逐题：错→对 4,407，对→错 3,967，净变化 +440。未做统计显著性检验。

## 与 exp4 和公开模型对照

| 模型 | BLiMP |
|---|---:|
| Exp4 WikiText 100 MB，12 轮 | 54.8627% |
| Exp4 v2 WikiText 100 MB，选第 16 轮 | 55.0657% |
| Exp5 Wikipedia 100mb | 53.9463% |
| Exp5 Wikipedia 1gb | 54.6030% |
| Published word 5-gram (Gigaword) | 61.2403% |
| Published Gulordava LSTM (Wikipedia) | 69.8119% |
| Published Transformer-XL Large (WikiText-103) | 69.6254% |
| Published GPT-2 Large, 774M parameters (WebText) | 83.0239% |

Exp4 和 Exp5 的语料来源、文章选择、文本格式不同；v2 还改变了停止规则。只有 Exp5 内部 100 MB/1 GB 是同源嵌套规模对照。公开模型是作者发布的参考结果，本机未重训，训练数据与预算不同。不同语料的 BPB 不直接比较。

## BLiMP 类别与长度诊断

| 类别 | 100 MB | 1 GB | 变化（百分点） |
|---|---:|---:|---:|
| anaphor_agreement | 40.05% | 47.25% | +7.20 |
| argument_structure | 62.01% | 63.06% | +1.04 |
| binding | 60.67% | 59.14% | -1.53 |
| control_raising | 59.04% | 59.64% | +0.60 |
| determiner_noun_agreement | 49.74% | 48.98% | -0.76 |
| ellipsis | 29.50% | 29.10% | -0.40 |
| filler_gap_dependency | 65.46% | 65.23% | -0.23 |
| irregular_forms | 59.00% | 61.45% | +2.45 |
| island_effects | 49.31% | 49.19% | -0.12 |
| npi_licensing | 45.56% | 50.03% | +4.47 |
| quantifiers | 47.08% | 48.27% | +1.20 |
| subject_verb_agreement | 53.58% | 52.98% | -0.60 |

| 正确句字节长度 | 100 MB | 1 GB |
|---|---:|---:|
| good_shorter_bytes | 70.63% | 72.80% |
| same_bytes | 45.61% | 46.71% |
| good_longer_bytes | 47.54% | 46.34% |

各长度子集类别构成不同，长度诊断不能单独确定偏差原因。BLiMP 是语法接受度任务，不是通用语义/推理评测。

## 协议和边界

训练固定最多 12 轮，不按 dev/test/BLiMP 选模型；最大上下文仍为 12 字节。两组使用同一冻结二进制，均关闭匹配复制。100 MB 为 1 GB 的完整文章前缀；两者共享独立 dev/test。源文件顺序抽样没有控制文章长度/主题分布：100 MB 的平均文章明显更长，因此这是扩展该语料前缀的效果，不是全 Wikipedia 随机抽样的普遍缩放规律。Python 独立验证了预处理字节数、SHA-256 分组、组内无完全重复和 train/dev/test 零完全重复交集。未做近似重复或 BLiMP 污染审计。

- 100mb：停止原因 `round_limit`；最后一轮 candidates=100,000、新增=9,467、合并=64。预算结束不代表收敛。
- 1gb：停止原因 `round_limit`；最后一轮 candidates=100,000、新增=10,000、合并=64。预算结束不代表收敛。

模型磁盘大小不等于训练/推理内存。本轮固定了结构增长预算，因此数据增大不要求模型文件同比增大。没有 Fold 消融，不能把质量变化归因于某一个机制。

时间口径：内部 training_seconds 含每轮 dev 验证；完整训练命令还含数据加载、保存/重载、最终 dev 评测和同数据 byte 5-gram 基线。下载、转换、编译、BLiMP 与独立 test 不计入。100 MB 首次计时工具 `/usr/bin/time -l` 在训练完成后因 sandbox 禁止 sysctl kern.clockrate 而返回 1；实际训练报告已完整写出，保存/重载验证已通过，保留模型继续评测，没有重新训练。该组墙钟和 CPU 时间可用，峰值内存未取得。后续改用独立 Python 进程中的 monotonic/getrusage 计时；原始失败记录保留。

## 产物

- [实验设计和复现](README.md)、[数据转换统计](data-preparation.json)、[独立数据验证](data-validation.json)
- [100 MB 训练](100mb/output.report.json)、[100 MB BLiMP](100mb/blimp.report.json)、[100 MB 样本](100mb/sample.txt)
- [1 GB 训练](1gb/output.report.json)、[1 GB BLiMP](1gb/blimp.report.json)、[1 GB 样本](1gb/sample.txt)
- [机器可读对照](comparison.json)、[数据与源代码校验](sha256.json)、[产物校验](artifact-sha256.json)
