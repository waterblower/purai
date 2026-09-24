// 推理 = 解释执行镜像。每个字节的计算：
//   1. 沿上下文树匹配（指针跳转 + 字节比较）
//   2. 叶子分布按计数表给出的权重线性混合
//   3. match 模型（哈希表 + 指针）给出一个候选字节，由计数表校准它的概率
// 没有矩阵乘法；每一步都可以计数。
import { compile, Compiled, Matched, newMatched, traverse } from "./compile.ts";
import { Image, Params, Tables } from "./types.ts";
import { renderCtx } from "./image.ts";
import { renderBytes } from "./text.ts";
import {
  apmBucket,
  apmHitProb,
  MatchModel,
  mixBucket,
  mixWeight,
} from "./calib.ts";

const UNIFORM = new Float64Array(256).fill(1 / 256);

export interface Predictor {
  readonly comp: Compiled;
  readonly params: Params;
  readonly tables: Tables;
  readonly P: Float64Array[]; // 每个 ctx 平滑后的预测分布
  readonly weight: Float64Array; // 每个 ctx 作为叶子时的混合权重
  readonly useMatch: boolean;
}

// Pitman-Yor 式平滑（与 calib.countDoc 中的预序版本完全一致）：
//   P_c(x) = (max(c(x) - d, 0) + (θ + d·u) · P_parent(x)) / (N + θ)，u 为见过的不同字节数
export function buildPredictor(
  comp: Compiled,
  params: Params,
  tables: Tables,
  useMatch = true,
): Predictor {
  const n = comp.ctx.length;
  const P: Float64Array[] = new Array(n);
  const weight = new Float64Array(n);
  const { discount: d, theta } = params;
  for (let k = 0; k < n; k++) {
    const c = comp.ctx[k];
    const parent = c.parentIdx >= 0 ? P[c.parentIdx] : UNIFORM;
    if (c.total === 0) {
      P[k] = parent;
    } else {
      let u = 0;
      for (let x = 0; x < 256; x++) if (c.counts[x] > 0) u++;
      const den = c.total + theta, g = (theta + d * u) / den;
      const p = new Float64Array(256);
      for (let x = 0; x < 256; x++) {
        p[x] = Math.max(c.counts[x] - d, 0) / den + g * parent[x];
      }
      P[k] = p;
    }
    weight[k] = mixWeight(tables, mixBucket(c.depth, c.total), params.kappa);
  }
  return { comp, params, tables, P, weight, useMatch };
}

export function predictorFor(img: Image, useMatch = true): Predictor {
  return buildPredictor(compile(img), img.params, img.tables, useMatch);
}

export interface Step {
  leaves: number[]; // 参与混合的 ctx
  steps: number; // 这一次预测的计算量：匹配测试次数 + 混合的叶子数 + match 查询
  matchLen: number; // 0 = 没有匹配
  matchByte: number; // -1 = 没有匹配
}

// 一次推理会话：持有 match 模型的状态（它只看当前输入的前文）。
export class Session {
  private m: Matched;
  private mark: Int32Array;
  private nonLeaf: Int32Array;
  private stamp = 0;
  private mm = new MatchModel();

  constructor(readonly pr: Predictor) {
    this.m = newMatched(pr.comp);
    this.mark = new Int32Array(pr.comp.ctx.length);
    this.nonLeaf = new Int32Array(pr.comp.ctx.length);
  }

  // 预测 text[i]（只读 text[0..i)）
  predict(text: Uint8Array, i: number, out: Float64Array): Step {
    const { pr, m } = this;
    const { comp } = pr;
    traverse(comp, text, i, m);
    const st = ++this.stamp;
    for (let k = 0; k < m.n; k++) this.mark[m.idx[k]] = st;
    for (let k = 0; k < m.n; k++) {
      const par = comp.ctx[m.idx[k]].parentIdx;
      if (par >= 0) this.nonLeaf[par] = st;
    }
    const leaves: number[] = [];
    for (let k = 0; k < m.n; k++) {
      const ci = m.idx[k];
      if (this.nonLeaf[ci] !== st) leaves.push(ci);
    }
    out.fill(0);
    let W = 0;
    for (const ci of leaves) {
      const w = pr.weight[ci], p = pr.P[ci];
      W += w;
      for (let x = 0; x < 256; x++) out[x] += w * p[x];
    }
    for (let x = 0; x < 256; x++) out[x] /= W;

    let matchByte = -1, matchLen = 0;
    if (pr.useMatch) {
      matchByte = this.mm.predicted(text);
      if (matchByte >= 0) {
        matchLen = this.mm.len;
        const q = out[matchByte];
        if (q < 1 - 1e-9) {
          const ph = apmHitProb(pr.tables, apmBucket(matchLen, q), q);
          const f = (1 - ph) / (1 - q);
          for (let x = 0; x < 256; x++) out[x] *= f;
          out[matchByte] = ph;
        }
      }
    }
    return {
      leaves,
      steps: m.steps + leaves.length + (pr.useMatch ? 1 : 0),
      matchLen,
      matchByte,
    };
  }

  // text[i] 已确定以后调用
  observe(text: Uint8Array, i: number): void {
    if (this.pr.useMatch) this.mm.observe(text, i);
  }
}

export function evalBits(
  pr: Predictor,
  text: Uint8Array,
): { bits: number; steps: number } {
  const s = new Session(pr);
  const out = new Float64Array(256);
  let bits = 0, steps = 0;
  for (let i = 0; i < text.length; i++) {
    steps += s.predict(text, i, out).steps;
    bits -= Math.log2(out[text[i]]);
    s.observe(text, i);
  }
  return { bits, steps };
}

export type Uncertainty = "confident" | "aleatoric" | "epistemic";

// 用计数区分两种不确定性（见 topic.md 3.4）：
//   认知不确定：最有力的叶子数据太少，或叶子之间意见不一致 -> 该"多想"
//   偶然不确定：数据充足但分布平坦 -> 直接采样
export function classify(
  pr: Predictor,
  step: Step,
  out: Float64Array,
): Uncertainty {
  let best = step.leaves[0];
  for (const ci of step.leaves) if (pr.weight[ci] > pr.weight[best]) best = ci;
  const argmax = (p: Float64Array) => {
    let a = 0;
    for (let x = 1; x < 256; x++) if (p[x] > p[a]) a = x;
    return a;
  };
  const top = new Set(step.leaves.map((ci) => argmax(pr.P[ci])));
  if (pr.comp.ctx[best].total < 8 || top.size > 1) return "epistemic";
  let h = 0;
  for (let x = 0; x < 256; x++) if (out[x] > 0) h -= out[x] * Math.log2(out[x]);
  return h > 2.5 ? "aleatoric" : "confident";
}

export interface GenStat {
  byte: number;
  steps: number;
  leaves: number;
  matchLen: number;
  uncertainty: Uncertainty;
}

export function generate(
  pr: Predictor,
  prompt: Uint8Array,
  n: number,
  opts: {
    temperature: number;
    random: () => number;
    onStep?: (s: Step, out: Float64Array, stat: GenStat) => void;
  },
): { bytes: Uint8Array; stats: GenStat[] } {
  const buf = new Uint8Array(prompt.length + n);
  buf.set(prompt);
  const s = new Session(pr);
  for (let i = 0; i < prompt.length; i++) s.observe(buf, i);
  const out = new Float64Array(256);
  const q = new Float64Array(256);
  const stats: GenStat[] = [];
  for (let t = 0; t < n; t++) {
    const i = prompt.length + t;
    const step = s.predict(buf, i, out);
    let Z = 0;
    for (let x = 0; x < 256; x++) {
      q[x] = Math.pow(out[x], 1 / opts.temperature);
      Z += q[x];
    }
    let r = opts.random() * Z, b = 255;
    for (let x = 0; x < 256; x++) {
      r -= q[x];
      if (r <= 0) {
        b = x;
        break;
      }
    }
    buf[i] = b;
    s.observe(buf, i);
    const stat: GenStat = {
      byte: b,
      steps: step.steps,
      leaves: step.leaves.length,
      matchLen: step.matchLen,
      uncertainty: classify(pr, step, out),
    };
    stats.push(stat);
    opts.onStep?.(step, out, stat);
  }
  return { bytes: buf.subarray(prompt.length), stats };
}

export function describeStep(pr: Predictor, step: Step): string {
  const img = pr.comp.img;
  let W = 0;
  for (const ci of step.leaves) W += pr.weight[ci];
  const lines = step.leaves.map((ci) => {
    const c = pr.comp.ctx[ci];
    const p = pr.P[ci];
    const top = [...p.keys()].sort((a, b) => p[b] - p[a]).slice(0, 3)
      .map((x) => `${renderBytes([x])}:${p[x].toFixed(2)}`).join(" ");
    return `    ${renderCtx(img, c.slots)}  w=${
      (pr.weight[ci] / W).toFixed(2)
    } N=${c.total}  ${top}`;
  });
  if (step.matchByte >= 0) {
    lines.push(
      `    match: len=${step.matchLen} → ${renderBytes([step.matchByte])}`,
    );
  }
  return lines.join("\n");
}
