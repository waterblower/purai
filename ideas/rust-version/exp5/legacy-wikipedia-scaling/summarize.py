import hashlib
import json
import pathlib
import re
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
EXP = pathlib.Path(__file__).resolve().parent

def read(path): return json.loads(path.read_text())

def paired(a, b):
    result = dict(both_correct=0, both_wrong=0, wrong_to_correct=0, correct_to_wrong=0)
    left = {}
    with a.open() as f:
        for line in f:
            x = json.loads(line); left[x["uid"], x["pair_id"]] = x
    seen = set()
    with b.open() as f:
        for line in f:
            y = json.loads(line); key = (y["uid"], y["pair_id"]); x = left[key]
            assert key not in seen and x["sentence_good"] == y["sentence_good"] and x["sentence_bad"] == y["sentence_bad"]
            seen.add(key)
            field = "both_correct" if x["correct"] and y["correct"] else "both_wrong" if not x["correct"] and not y["correct"] else "wrong_to_correct" if y["correct"] else "correct_to_wrong"
            result[field] += 1
    assert len(seen) == len(left) == 67000
    result["net_correct_change"] = result["wrong_to_correct"] - result["correct_to_wrong"]
    return result

def main():
    data = read(EXP / "data-preparation.json")
    rows = {}
    for size in ("100mb", "1gb"):
        directory = EXP / size
        training = read(directory / "output.report.json")
        blimp = read(directory / "blimp.report.json")
        timing = read(directory / "training.run.json")
        utc_elapsed = (datetime.fromisoformat(timing['finished_at_utc']) - datetime.fromisoformat(timing['started_at_utc'])).total_seconds()
        test_match = re.search(r"bits/byte=([\d.]+)", (directory / "test.log").read_text())
        assert blimp["full_suite"] and blimp["overall"]["pairs"] == 67000 and timing["training_completed"]
        assert len(training["rounds"]) <= 12 and training["structure_bytes"] >= (100_000_000 if size == "100mb" else 1_000_000_000)
        rows[size] = dict(training=training, blimp=blimp, timing=timing, test_bpb_six_decimal_precision=float(test_match[1]),
                          timing_audit=dict(utc_elapsed_seconds=utc_elapsed, monotonic_elapsed_seconds=timing['wall_seconds'], utc_minus_monotonic_seconds=utc_elapsed-timing['wall_seconds']))
    a, b = rows["100mb"], rows["1gb"]
    assert a["training"]["options"] == b["training"]["options"]
    assert a["training"]["parameters"] == b["training"]["parameters"]
    assert a["timing"]["binary_sha256"] == b["timing"]["binary_sha256"]
    pair = paired(EXP / "100mb/blimp.samples.jsonl", EXP / "1gb/blimp.samples.jsonl")
    cross = paired(ROOT / "exp4/blimp.samples.jsonl", EXP / "1gb/blimp.samples.jsonl")
    ratios = {"data": b["training"]["structure_bytes"] / a["training"]["structure_bytes"],
              "training_time": b["training"]["training_seconds"] / a["training"]["training_seconds"],
              "full_command_time": b["timing"]["wall_seconds"] / a["timing"]["wall_seconds"],
              "model_bytes": b["training"]["model_bytes"] / a["training"]["model_bytes"]}
    delta = 100 * (b["blimp"]["overall"]["accuracy"] - a["blimp"]["overall"]["accuracy"])
    comparison = dict(data=data, runs=rows, same_training_options=True, paired_100mb_to_1gb=pair, paired_exp4v1_to_exp5_1gb=cross, ratios=ratios, blimp_change_percentage_points=delta)
    (EXP / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Exp5 结果：英文 Wikipedia 100 MB → 1 GB", "",
             f"同源嵌套语料、相同模型程序与固定训练设置下，完整 BLiMP 从 **{a['blimp']['overall']['accuracy']*100:.4f}%** 到 **{b['blimp']['overall']['accuracy']*100:.4f}%**，变化 **{delta:+.4f} 个百分点**。", "",
             "## 同源规模对照", "", "| 指标 | 100 MB | 1 GB |", "|---|---:|---:|"]
    metrics = [("训练正文 bytes", lambda r: f"{r['training']['structure_bytes']:,}"),
               ("训练文章", lambda r: f"{r['training']['structure_documents']:,}"),
               ("平均文章正文字节", lambda r: f"{r['training']['structure_bytes']/r['training']['structure_documents']:.1f}"),
               ("完成轮数", lambda r: str(len(r['training']['rounds']))),
               ("节点数", lambda r: f"{r['training']['nodes']:,}"),
               ("共享字节类别", lambda r: str(r['training']['classes'])),
               ("模型 MiB", lambda r: f"{r['training']['model_bytes']/2**20:.3f}"),
               ("训练阶段 Instant 秒数（含逐轮 dev 验证）", lambda r: f"{r['training']['training_seconds']:.3f}"),
               ("完整命令单调时钟秒数", lambda r: f"{r['timing']['wall_seconds']:.3f}"),
               ("完整命令 UTC 起止跨度秒数", lambda r: f"{r['timing_audit']['utc_elapsed_seconds']:.3f}"),
               ("完整命令 CPU 秒数（user + sys）", lambda r: f"{r['timing']['user_cpu_seconds']+r['timing']['system_cpu_seconds']:.3f}"),
               ("峰值内存 GiB（完整训练命令）", lambda r: f"{r['timing']['peak_rss_bytes_macos']/2**30:.3f}" if r['timing']['peak_rss_bytes_macos'] is not None else "未取得"),
               ("dev graph BPB", lambda r: f"{r['training']['graph_only']['bits_per_byte']:.9f}"),
               ("dev byte 5-gram BPB", lambda r: f"{r['training']['ngram_baseline']['bits_per_byte']:.9f}"),
               ("独立 test graph BPB（CLI 六位小数）", lambda r: f"{r['test_bpb_six_decimal_precision']:.6f}"),
               ("BLiMP 正确 / 67,000", lambda r: f"{r['blimp']['overall']['correct']:,}"),
               ("BLiMP 准确率", lambda r: f"{100*r['blimp']['overall']['accuracy']:.4f}%")]
    for label, f in metrics: lines.append(f"| {label} | {f(a)} | {f(b)} |")
    lines += ["", f"数据量为 {ratios['data']:.3f} 倍；训练阶段 Instant 计时为 {ratios['training_time']:.3f} 倍；完整命令单调计时为 {ratios['full_command_time']:.3f} 倍；模型文件为 {ratios['model_bytes']:.3f} 倍。这是一次运行，不是重复计时的稳定性估计。", ""]
    for size, r in rows.items():
        audit = r['timing_audit']
        if abs(audit['utc_minus_monotonic_seconds']) > 1:
            lines += [f"**计时异常（{size}）**：原始 UTC 起止时间跨度为 {audit['utc_elapsed_seconds']/60:.2f} 分钟，单调时钟记录 {audit['monotonic_elapsed_seconds']/60:.2f} 分钟，差 {audit['utc_minus_monotonic_seconds']/60:.2f} 分钟。进程 CPU 时间为 {(r['timing']['user_cpu_seconds']+r['timing']['system_cpu_seconds'])/60:.2f} 分钟。未确认是否发生休眠或系统时钟调整。性能比较使用程序 Instant、外部 monotonic 和 CPU 时间，不能把单调计时直接称为本次实际日历等待时间。原始起止时间和计时值均保留在 training.run.json，未回填或修改。", ""]
    lines += [
              f"BLiMP 逐题：错→对 {pair['wrong_to_correct']:,}，对→错 {pair['correct_to_wrong']:,}，净变化 {pair['net_correct_change']:+,}。未做统计显著性检验。", "",
              "## 与 exp4 和公开模型对照", "", "| 模型 | BLiMP |", "|---|---:|"]
    for label, path in [("Exp4 WikiText 100 MB，12 轮", ROOT / "exp4/blimp.report.json"), ("Exp4 v2 WikiText 100 MB，选第 16 轮", ROOT / "exp4/v2/blimp.report.json")]:
        old = read(path); lines.append(f"| {label} | {100*old['overall']['accuracy']:.4f}% |")
    for size, r in rows.items(): lines.append(f"| Exp5 Wikipedia {size} | {100*r['blimp']['overall']['accuracy']:.4f}% |")
    refs = b['blimp']['published_references']
    for key in ('ngram','lstm','txl','gpt2'): lines.append(f"| {refs['labels'][key]} | {100*refs['overall_accuracy'][key]:.4f}% |")
    lines += ["", "Exp4 和 Exp5 的语料来源、文章选择、文本格式不同；v2 还改变了停止规则。只有 Exp5 内部 100 MB/1 GB 是同源嵌套规模对照。公开模型是作者发布的参考结果，本机未重训，训练数据与预算不同。不同语料的 BPB 不直接比较。", "",
              "## BLiMP 类别与长度诊断", "", "| 类别 | 100 MB | 1 GB | 变化（百分点） |", "|---|---:|---:|---:|"]
    for key, x in a['blimp']['categories'].items():
        y = b['blimp']['categories'][key]; lines.append(f"| {key} | {100*x['accuracy']:.2f}% | {100*y['accuracy']:.2f}% | {100*(y['accuracy']-x['accuracy']):+.2f} |")
    lines += ["", "| 正确句字节长度 | 100 MB | 1 GB |", "|---|---:|---:|"]
    for key in ('good_shorter_bytes','same_bytes','good_longer_bytes'):
        lines.append(f"| {key} | {100*a['blimp']['byte_length_diagnostics'][key]['accuracy']:.2f}% | {100*b['blimp']['byte_length_diagnostics'][key]['accuracy']:.2f}% |")
    lines += ["", "各长度子集类别构成不同，长度诊断不能单独确定偏差原因。BLiMP 是语法接受度任务，不是通用语义/推理评测。", "", "## 协议和边界", "",
              "训练固定最多 12 轮，不按 dev/test/BLiMP 选模型；最大上下文仍为 12 字节。两组使用同一冻结二进制，均关闭匹配复制。100 MB 为 1 GB 的完整文章前缀；两者共享独立 dev/test。源文件顺序抽样没有控制文章长度/主题分布：100 MB 的平均文章明显更长，因此这是扩展该语料前缀的效果，不是全 Wikipedia 随机抽样的普遍缩放规律。Python 独立验证了预处理字节数、SHA-256 分组、组内无完全重复和 train/dev/test 零完全重复交集。未做近似重复或 BLiMP 污染审计。", ""]
    for size, r in rows.items():
        last = r['training']['rounds'][-1]
        lines.append(f"- {size}：停止原因 `{r['training']['stopping']['reason']}`；最后一轮 candidates={last['candidates']:,}、新增={last['added']:,}、合并={last['merged']}。预算结束不代表收敛。")
    lines += ["", "模型磁盘大小不等于训练/推理内存。本轮固定了结构增长预算，因此数据增大不要求模型文件同比增大。没有 Fold 消融，不能把质量变化归因于某一个机制。", "",
              "时间口径：内部 training_seconds 含每轮 dev 验证；完整训练命令还含数据加载、保存/重载、最终 dev 评测和同数据 byte 5-gram 基线。下载、转换、编译、BLiMP 与独立 test 不计入。100 MB 首次计时工具 `/usr/bin/time -l` 在训练完成后因 sandbox 禁止 sysctl kern.clockrate 而返回 1；实际训练报告已完整写出，保存/重载验证已通过，保留模型继续评测，没有重新训练。该组墙钟和 CPU 时间可用，峰值内存未取得。后续改用独立 Python 进程中的 monotonic/getrusage 计时；原始失败记录保留。", "",
              "## 产物", "", "- [实验设计和复现](README.md)、[数据转换统计](data-preparation.json)、[独立数据验证](data-validation.json)",
              "- [100 MB 训练](100mb/output.report.json)、[100 MB BLiMP](100mb/blimp.report.json)、[100 MB 样本](100mb/sample.txt)",
              "- [1 GB 训练](1gb/output.report.json)、[1 GB BLiMP](1gb/blimp.report.json)、[1 GB 样本](1gb/sample.txt)",
              "- [机器可读对照](comparison.json)、[数据与源代码校验](sha256.json)、[产物校验](artifact-sha256.json)"]
    (EXP / "RESULTS.md").write_text("\n".join(lines) + "\n")
    checks = {}
    for size in ('100mb','1gb'):
        for path in sorted((EXP / size).iterdir()):
            if path.is_file():
                h=hashlib.sha256()
                with path.open('rb') as f:
                    for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
                checks[str(path.relative_to(EXP))] = h.hexdigest()
    (EXP / "artifact-sha256.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(dict(ratios=ratios, blimp_change_percentage_points=delta, paired=pair), indent=2))

if __name__ == '__main__': main()
