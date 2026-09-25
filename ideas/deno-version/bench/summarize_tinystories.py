"""Summarize completed runs without retraining or selecting generated samples."""
import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent / "results" / "tinystories10mb"


def read(name):
    return json.loads((BASE / name).read_text())


def main():
    default = read("default.json")
    no_fold = read("no-fold.json")
    baselines = read("ngram.json")["results"]
    best = min(baselines, key=lambda row: row["bpb"])
    rows = [f"| n-gram，{best['order']} 字节上下文 | {best['bpb']:.6f} | {best['contexts']:,}（该阶） | — |"]
    for label, result in [("AFRN 默认", default), ("AFRN 关闭 Fold", no_fold)]:
        for evaluation in result["evaluations"]:
            name = label + ("" if evaluation["useMatch"] else "，关闭 match")
            rows.append(f"| {name} | {evaluation['bpb']:.6f} | {result['structure']['ctxs']:,} | {evaluation['stepsPerByte']:.1f} |")

    # Paired resampling of stories: positive delta means Fold reduced BPB.
    a = read("default-match-per-story.json")
    b = read("no-fold-match-per-story.json")
    assert len(a) == len(b)
    assert all(x["bytes"] == y["bytes"] for x, y in zip(a, b))
    deltas = [y["bits"] - x["bits"] for x, y in zip(a, b)]
    lengths = [x["bytes"] for x in a]
    rng = random.Random(42)
    estimates = []
    for _ in range(2000):
        indices = [rng.randrange(len(a)) for _ in a]
        estimates.append(sum(deltas[i] for i in indices) / sum(lengths[i] for i in indices))
    estimates.sort()
    bootstrap = {"metric": "no-fold BPB minus default BPB; positive favors Fold",
                 "observed": sum(deltas) / sum(lengths),
                 "percentile_95_interval": [estimates[49], estimates[1949]],
                 "resamples": 2000, "seed": 42,
                 "unit": "paired validation stories; fixed trained models"}
    (BASE / "fold-bootstrap.json").write_text(json.dumps(bootstrap, indent=2) + "\n")
    report = [
        "# TinyStories 10 MB：实测结果", "",
        "训练输入 10,023,508 字节（11,459 篇故事及空行分隔符），独立官方验证集 1,000,375 字节（1,286 篇）。",
        "使用固定文件前缀子集，非全量数据；实验协议和复现命令见 [TINYSTORIES.md](../../TINYSTORIES.md)。", "",
        "## 验证集结果", "",
        "BPB 是按 UTF-8 字节加权的负对数概率，越低越好。n-gram 从 0–10 阶中比较；本验证集也用于展示逐轮指标，因此这些是验证结果，不是独立最终测试集结果。", "",
        "| 模型 | bits/byte | 上下文数量 | 计数步数/字节 |", "|---|---:|---:|---:|", *rows, "",
        "步数是解释器定义的操作计数，未计入全部标量运算，不等于 CPU 指令数。n-gram 表中数量仅为选中阶的上下文数，实际预测也使用低阶表。", "",
        "## Fold 对照", "",
        f"关闭 Fold − 默认模型 = {bootstrap['observed']:.6f} bits/byte；正值表示 Fold 更好。",
        f"按验证故事配对重采样的 95% 百分位区间为 [{estimates[49]:.6f}, {estimates[1949]:.6f}]（2,000 次，种子 42）。",
        "该区间仅描述固定模型在这些验证故事上的抽样波动，不包含更换训练子集带来的不确定性。默认压缩对两组分别执行，因此这是完整训练配置的对照。", "",
        "## 模型与运行", "",
    ]
    for name, result in [("default", default), ("no-fold", no_fold)]:
        elapsed = result["trainSeconds"]
        duration = f"{elapsed / 60:.2f} 分钟" if elapsed is not None else "缓存模型，未重新计时"
        report.append(f"- `{name}.data`：{result['modelBytes']:,} 字节；训练 {duration}；结构 {json.dumps(result['structure'])}。")
    report += ["", "两组训练并行运行，耗时受资源竞争影响。模型源码未改动；仅新增数据准备、实验与报告脚本。", "",
               "## 固定提示词的全部生成样本", "",
               "每个样本生成 600 字节，随机种子 42；以下包含所有预设提示词与温度，没有根据流畅程度筛选。", ""]
    for name in ["default", "no-fold"]:
        for sample in read(f"{name}-samples.json"):
            report += [f"### {name}，温度 {sample['temperature']}", "", "```text",
                       sample["prompt"] + sample["continuation"], "```", ""]
    (BASE / "RESULTS.md").write_text("\n".join(report))
    print("\n".join(rows))
    print(json.dumps(bootstrap, indent=2))


if __name__ == "__main__":
    main()
