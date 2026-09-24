import { fail, parseArgs } from "./args.ts";
import { load } from "../model_io.ts";
import { describeStep, generate, predictorFor } from "../predict.ts";
import { decoder, encoder, renderBytes } from "../text.ts";

import { rng } from "../hash.ts";

const USAGE = `usage: infer --model <file> --input <text> [选项]
  --n <k>              生成多少个字节（默认 200）
  --temperature <t>    采样温度（默认 1.0）
  --seed <s>           随机种子（默认 42）
  --no-match           关掉 match 模型
  --trace              打印每一步参与混合的上下文及其权重
  --stats              打印每一步的计算步数与不确定性类型`;

const a = parseArgs(Deno.args, ["trace", "stats", "no-match", "help", "h"]);
if (a.bool("help") || a.bool("h")) fail(USAGE);
const modelPath = a.str("model") ?? fail(USAGE);
const input = a.str("input", "")!;

const img = await load(modelPath);
const pr = predictorFor(img, !a.bool("no-match"));
const trace = a.bool("trace");
const { bytes, stats } = generate(pr, encoder.encode(input), a.num("n", 200), {
  temperature: a.num("temperature", 1.0),
  random: rng(a.num("seed", 42)),
  onStep: trace
    ? (step, out, st) => {
      const p = out[st.byte];
      console.log(
        `→ ${renderBytes([st.byte])}  p=${
          p.toFixed(3)
        }  steps=${st.steps}  ${st.uncertainty}`,
      );
      console.log(describeStep(pr, step));
    }
    : undefined,
});

console.log(input + decoder.decode(bytes));

if (a.bool("stats")) {
  const steps = stats.map((s) => s.steps);
  const kinds = { confident: 0, aleatoric: 0, epistemic: 0 };
  stats.forEach((s) => kinds[s.uncertainty]++);
  const avg = steps.reduce((x, y) => x + y, 0) / Math.max(1, steps.length);
  console.log(
    `\n--- stats: ${stats.length} bytes, steps/byte avg=${avg.toFixed(1)} min=${
      Math.min(...steps)
    } max=${Math.max(...steps)}`,
  );
  console.log(
    `    confident=${kinds.confident} aleatoric=${kinds.aleatoric} epistemic=${kinds.epistemic}`,
  );
  const hist = new Map<number, number>();
  for (const s of steps) {
    const b = Math.min(10, Math.floor(s / 10));
    hist.set(b, (hist.get(b) ?? 0) + 1);
  }
  for (const [b, n] of [...hist].sort((x, y) => x[0] - y[0])) {
    const label = b === 10 ? "100+" : `${b * 10}-${b * 10 + 9}`;
    console.log(
      `    steps ${label.padStart(7)}: ${
        "#".repeat(Math.ceil((n / stats.length) * 60))
      } ${n}`,
    );
  }
}
