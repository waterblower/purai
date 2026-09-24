import { fail, parseArgs } from "./args.ts";
import { defaultOptions, stats, train, TrainOptions } from "../learn.ts";
import { emptyImage } from "../image.ts";
import { load, save } from "../model_io.ts";
import { evalBits, predictorFor } from "../predict.ts";
import { DEFAULT_PARAMS, Params } from "../types.ts";

const USAGE = `usage: train <doc1.txt> [doc2.txt ...] [选项]
  每个文件是一篇独立的文档：计数按文档累加，所以之后可以精确地遗忘其中任何一篇。
  -o <file>              输出模型文件（默认 model.data）
  --init <file>          从已有模型继续训练（沿用它的解释器参数）
  --rounds <n>           最多训练轮数（默认 ${defaultOptions.rounds}）
  --valid-frac <f>       每篇文档末尾留作验证集的比例（默认 0.1）
  --discount <d>         Pitman-Yor 折扣（默认 ${DEFAULT_PARAMS.discount}）
  --theta <t>            Pitman-Yor 强度（默认 ${DEFAULT_PARAMS.theta}）
  --kappa <k>            混合权重锐度（默认 ${DEFAULT_PARAMS.kappa}）

  --forks <k>            每轮最多接受的 Fork 数（默认 ${defaultOptions.forksPerRound}）
  --folds <k>            每轮最多接受的 Fold 数（默认 ${defaultOptions.foldsPerRound}）
  --chunks <k>           每轮最多接受的 Chunk 数（默认 ${defaultOptions.chunksPerRound}）
  --min-count <n>        新上下文的最少出现次数（默认 ${defaultOptions.minCount}）
  --max-depth <n>        上下文最多槽位数（默认 ${defaultOptions.maxDepth}）
  --struct-weight <w>    结构码长的权重，1 = 标准 MDL（默认 ${defaultOptions.structWeight}）
  --candidates <n>       每轮精确评估的 Fork 候选数上限（默认 ${defaultOptions.maxCandidates}）
  --no-compact           训练结束时不做观察等价压缩（bpc 略好，但计算步数约 3 倍）
  --compact-min <n>      只合并证据至少为 n 的等价组（默认 ${defaultOptions.compactMin}）`;

const a = parseArgs(Deno.args, ["help", "h", "no-compact"]);
if (a.bool("help") || a.bool("h") || a.positional.length === 0) fail(USAGE);

const out = a.str("o", "model.data")!;
const validFrac = a.num("valid-frac", 0.1);
const opts: TrainOptions = {
  ...defaultOptions,
  rounds: a.num("rounds", defaultOptions.rounds),

  forksPerRound: a.num("forks", defaultOptions.forksPerRound),
  foldsPerRound: a.num("folds", defaultOptions.foldsPerRound),
  chunksPerRound: a.num("chunks", defaultOptions.chunksPerRound),
  minCount: a.num("min-count", defaultOptions.minCount),
  minChunkCount: a.num("min-chunk-count", defaultOptions.minChunkCount),
  maxDepth: a.num("max-depth", defaultOptions.maxDepth),
  structWeight: a.num("struct-weight", defaultOptions.structWeight),
  maxCandidates: a.num("candidates", defaultOptions.maxCandidates),
  compact: !a.bool("no-compact"),
  compactMin: a.num("compact-min", defaultOptions.compactMin),
};
const params: Params = {
  discount: a.num("discount", DEFAULT_PARAMS.discount),
  theta: a.num("theta", DEFAULT_PARAMS.theta),
  kappa: a.num("kappa", DEFAULT_PARAMS.kappa),
};

const docs: Uint8Array[] = [], valid: Uint8Array[] = [];
for (const p of a.positional) {
  const data = await Deno.readFile(p);
  const cut = Math.floor(data.length * (1 - validFrac));
  docs.push(data.subarray(0, cut));
  if (cut < data.length) valid.push(data.subarray(cut));
}
const init = a.str("init") ? await load(a.str("init")!) : emptyImage(params);

const bytes = (xs: Uint8Array[]) => xs.reduce((s, d) => s + d.length, 0);
console.log(
  `train: ${docs.length} docs, ${bytes(docs)} bytes; valid: ${
    bytes(valid)
  } bytes`,
);
console.log(`params: ${JSON.stringify(init.params)}`);
console.log(
  "round   ctxs  seqs  alts class   +chunk +fork +fold   model_bits  valid_bpc   secs",
);
const pad = (x: string | number, n: number) => String(x).padStart(n);
const model = train(init, docs, valid, opts, (r) => {
  console.log(
    `${pad(r.round, 5)} ${pad(r.ctxs, 6)} ${pad(r.seqs, 5)} ${pad(r.alts, 5)} ${
      pad(r.classed, 6)
    }  ` +
      `${pad(r.added.chunk, 6)} ${pad(r.added.fork, 5)} ${
        pad(r.added.fold, 5)
      }  ` +
      `${pad(Math.round(r.modelBits).toLocaleString("en"), 11)}  ${
        pad(r.validBpc.toFixed(4), 9)
      }  ${pad(r.seconds.toFixed(1), 5)}`,
  );
}, (removed) => console.log(`compact: 删除 ${removed} 个观察等价的上下文`));

const pr = predictorFor(model);
let bits = 0, steps = 0;
for (const t of valid) {
  const r = evalBits(pr, t);
  bits += r.bits;
  steps += r.steps;
}
const bpc = bytes(valid) ? bits / bytes(valid) : NaN;
const size = await save(out, model);
const s = stats(model);
console.log(
  `final valid_bpc=${bpc.toFixed(4)}  steps/byte=${
    (steps / Math.max(1, bytes(valid))).toFixed(1)
  }  saved ${out} (${s.ctxs} ctx, ${s.seqs} seq, ${s.alts} alt, ${
    (size / 1024).toFixed(0)
  } KB)`,
);
