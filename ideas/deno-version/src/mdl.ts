// 码长计算。数据码长用 Pitman-Yor 边际似然：它与预测器是同一个概率模型，
// 只取决于计数（与数据顺序无关），并且自带 Occam 惩罚，
// 所以 Fork（分出新上下文）和 Fold（合并相似上下文）可以用同一个判据比较。
import { Node } from "./types.ts";

const LN2 = Math.LN2;
const G = 7;
const LANCZOS = [
  0.99999999999980993,
  676.5203681218851,
  -1259.1392167224028,
  771.32342877765313,
  -176.61502916214059,
  12.507343278686905,
  -0.13857109526572012,
  9.9843695780195716e-6,
  1.5056327351493116e-7,
];

export function lgamma(x: number): number {
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  let a = LANCZOS[0];
  const t = x + G + 0.5;
  for (let i = 1; i < G + 2; i++) a += LANCZOS[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t +
    Math.log(a);
}

// -log2 P(counts)，Pitman-Yor 过程（每种字节一张"桌子"），基分布为 base：
//   P = Π_x base(x) · Π_{k<u}(θ + d·k) · Π_x Γ(c_x - d)/Γ(1 - d) · Γ(θ)/Γ(θ + N)
export function pyBits(
  counts: ArrayLike<number>,
  base: Float64Array,
  d: number,
  theta: number,
): number {
  let n = 0, u = 0, s = 0;
  const g1d = lgamma(1 - d);
  for (let x = 0; x < 256; x++) {
    const c = counts[x];
    if (c > 0) {
      s += Math.log(base[x]) + Math.log(theta + d * u) + lgamma(c - d) - g1d;
      u++;
      n += c;
    }
  }
  if (n === 0) return 0;
  return -(s + lgamma(theta) - lgamma(theta + n)) / LN2;
}

// Pitman-Yor 预测分布（与 predict.ts 相同的公式），写入 out。
export function pyPredictive(
  counts: ArrayLike<number>,
  base: Float64Array,
  d: number,
  theta: number,
  out: Float64Array,
): void {
  let n = 0, u = 0;
  for (let x = 0; x < 256; x++) {
    if (counts[x] > 0) {
      n += counts[x];
      u++;
    }
  }
  if (n === 0) {
    out.set(base);
    return;
  }
  const den = n + theta, g = (theta + d * u) / den;
  for (let x = 0; x < 256; x++) {
    out[x] = Math.max(counts[x] - d, 0) / den + g * base[x];
  }
}

// 结构码长：种类标签 8 比特 + 每个引用 log2(可引用对象数) 比特。
// ctx 按"父 ctx + 新槽位"描述，只需要 2 个引用，与深度无关。
export function nodeBits(n: Node, totalNodes: number): number {
  const refBits = Math.log2(totalNodes + 256);
  switch (n.kind) {
    case "seq":
      return 8 + 8 * n.bytes.length;
    case "alt":
      return 8 + n.items.length * refBits;
    case "ctx":
      return 8 + Math.min(n.slots.length, 2) * refBits;
  }
}
