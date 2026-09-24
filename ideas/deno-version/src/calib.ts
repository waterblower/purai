// 推理时的两个 CPU 友好机制，以及它们的预序统计：
//   1. Pitman-Yor 平滑：每个 ctx 只用自己的计数和父分布，O(1) 标量递推。
//   2. Match 模型：哈希表 + 指针，在当前输入的前文里找最长匹配，预测"上次后面跟的字节"。
//      这是离散版的"归纳头"（in-context copy），每个字节 O(1)。
// 混合权重和 match 置信度都来自整数计数表，按文档预序统计：
// 每个位置只用它之前的数据来"考"模型，所以没有训练集内的偏差，而且结果只依赖这一篇文档。
import { Compiled, Matched, newMatched, traverse } from "./compile.ts";
import { emptyTables, Params, Tables } from "./types.ts";

// ---------- 表的分桶 ----------

export function mixBucket(depth: number, n: number): number {
  return Math.min(depth, 15) * 16 + Math.min(15, Math.floor(Math.log2(n + 1)));
}

export function mixWeight(t: Tables, bucket: number, kappa: number): number {
  const n = t.mixN[bucket];
  const avgBits = n > 0 ? t.mixBits[bucket] / 1024 / n : 8;
  return Math.pow(2, -kappa * avgBits);
}

export function apmBucket(len: number, q: number): number {
  const lb = Math.min(15, Math.floor(2 * Math.log2(len)));
  const qb = Math.min(15, Math.floor(-2 * Math.log2(Math.max(q, 1e-9))));
  return lb * 16 + qb;
}

// match 预测的字节真正出现的概率。按匹配长度逐级回退：
//   p(L) = (hit[L] + k · p(L-1)) / (tot[L] + k)，p(-1) = 上下文模型自己的概率 q
// 这样训练中没见过的长匹配，至少继承较短匹配的可靠度。
export function apmHitProb(t: Tables, bucket: number, q: number): number {
  const k = 2;
  const lb = bucket >> 4, qb = bucket & 15;
  let p = q;
  for (let l = 0; l <= lb; l++) {
    const b = l * 16 + qb;
    p = (t.apmHit[b] + k * p) / (t.apmTot[b] + k);
  }
  return p;
}

// ---------- Match 模型 ----------

export const MATCH_MIN = 6;
const HASH_BITS = 20;

export class MatchModel {
  private table = new Int32Array(1 << HASH_BITS); // 哈希 -> 上次出现位置 + 1
  ptr = 0; // 预测字节所在的位置
  len = 0; // 当前匹配长度，0 表示没有匹配

  predicted(text: Uint8Array): number {
    return this.len > 0 ? text[this.ptr] : -1;
  }

  // text[i] 已知以后调用
  observe(text: Uint8Array, i: number): void {
    if (this.len > 0) {
      if (text[this.ptr] === text[i]) {
        this.ptr++;
        this.len++;
      } else this.len = 0;
    }
    if (i + 1 < MATCH_MIN) return;
    let h = 0;
    for (let k = i - MATCH_MIN + 1; k <= i; k++) {
      h = Math.imul(h ^ text[k], 0x9e3779b1);
    }
    h >>>= 32 - HASH_BITS;
    if (this.len === 0) {
      const cand = this.table[h];
      if (cand > 0) {
        let L = 0;
        while (
          L < 64 && cand - 1 - L >= 0 && text[cand - 1 - L] === text[i - L]
        ) L++;
        if (L >= MATCH_MIN) {
          this.ptr = cand;
          this.len = L;
        }
      }
    }
    this.table[h] = i + 1;
  }
}

// ---------- 在一篇文档上计数（预序） ----------

export interface DocCounts {
  counts: Int32Array[]; // 按 comp.ctx 下标
  tables: Tables;
}

// 对一篇文档：沿用当前代码匹配上下文，累加计数，并在"看到答案之前"记录
// 叶子的码长与 match 的命中情况。onMatched 用于训练时收集 Fork 候选。
export function countDoc(
  comp: Compiled,
  text: Uint8Array,
  params: Params,
  onMatched?: (m: Matched, i: number) => void,
): DocCounts {
  const n = comp.ctx.length;
  const counts = comp.ctx.map(() => new Int32Array(256));
  const tot = new Int32Array(n), dist = new Int32Array(n);
  const tables = emptyTables();
  const m = newMatched(comp);
  const mm = new MatchModel();
  const mark = new Int32Array(n),
    nonLeaf = new Int32Array(n),
    memo = new Int32Array(n);
  const pA = new Float64Array(n), pB = new Float64Array(n);
  const { discount: d, theta } = params;
  let st = 0;
  let xa = 0, xb = 0;

  // 用"到目前为止"的计数，沿父链递推 P(xa) 和 P(xb)
  const chain = (ci: number): void => {
    if (memo[ci] === st) return;
    const par = comp.ctx[ci].parentIdx;
    let a0 = 1 / 256, b0 = 1 / 256;
    if (par >= 0) {
      chain(par);
      a0 = pA[par];
      b0 = pB[par];
    }
    const N = tot[ci];
    if (N === 0) {
      pA[ci] = a0;
      pB[ci] = b0;
    } else {
      const c = counts[ci], den = N + theta, g = (theta + d * dist[ci]) / den;
      pA[ci] = Math.max(c[xa] - d, 0) / den + g * a0;
      pB[ci] = Math.max(c[xb] - d, 0) / den + g * b0;
    }
    memo[ci] = st;
  };

  const leaves: number[] = [];
  for (let i = 0; i < text.length; i++) {
    traverse(comp, text, i, m);
    onMatched?.(m, i);
    st++;
    for (let k = 0; k < m.n; k++) mark[m.idx[k]] = st;
    for (let k = 0; k < m.n; k++) {
      const par = comp.ctx[m.idx[k]].parentIdx;
      if (par >= 0) nonLeaf[par] = st;
    }
    leaves.length = 0;
    for (let k = 0; k < m.n; k++) {
      if (nonLeaf[m.idx[k]] !== st) leaves.push(m.idx[k]);
    }

    const x = text[i];
    const xh = mm.predicted(text);
    xa = x;
    xb = xh >= 0 ? xh : x;
    let W = 0, q = 0;
    for (const ci of leaves) {
      chain(ci);
      const b = mixBucket(comp.ctx[ci].depth, tot[ci]);
      const w = mixWeight(tables, b, params.kappa);
      W += w;
      q += w * pB[ci];
      tables.mixBits[b] += Math.round(-Math.log2(pA[ci]) * 1024);
      tables.mixN[b]++;
    }
    if (xh >= 0) {
      const b = apmBucket(mm.len, q / W);
      tables.apmTot[b]++;
      if (xh === x) tables.apmHit[b]++;
    }
    for (let k = 0; k < m.n; k++) {
      const ci = m.idx[k], c = counts[ci];
      if (c[x] === 0) dist[ci]++;
      c[x]++;
      tot[ci]++;
    }
    mm.observe(text, i);
  }
  void mark;
  return { counts, tables };
}

export function addTables(into: Tables, t: Tables, sign = 1): void {
  for (let k = 0; k < into.mixBits.length; k++) {
    into.mixBits[k] += sign * t.mixBits[k];
    into.mixN[k] += sign * t.mixN[k];
    into.apmHit[k] += sign * t.apmHit[k];
    into.apmTot[k] += sign * t.apmTot[k];
  }
}
