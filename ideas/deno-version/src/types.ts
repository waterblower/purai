import { hash64 } from "./hash.ts";

export type Sym = number; // 字节终结符 0..255
export type Id = string; // 内容 hash

export type Ref =
  | { readonly tag: "term"; readonly sym: Sym }
  | { readonly tag: "ref"; readonly id: Id };

// 学出来的"代码"只有三种节点，构成一个 Merkle DAG：
//   seq：词首片段（chunk），如 "the"。只有从词边界开始时才匹配。
//   alt：可替换类，如 {king|queen}。成员全是字节，或全是 seq（保证匹配唯一）。
//   ctx：上下文模式。slots[0] 是最左（最早）的槽位，最后一个紧挨着要预测的位置。
//        Ctx [] 是根。ctx 的计数挂在 Image.ctxCounts 上。
export type Node =
  | { readonly kind: "seq"; readonly bytes: readonly Sym[] }
  | { readonly kind: "alt"; readonly items: readonly Ref[] }
  | { readonly kind: "ctx"; readonly slots: readonly Ref[] };

export const term = (sym: Sym): Ref => ({ tag: "term", sym });
export const ref = (id: Id): Ref => ({ tag: "ref", id });

export function refKey(r: Ref): string {
  return r.tag === "term" ? `t${r.sym}` : `r${r.id}`;
}

export function nodeKey(n: Node): string {
  switch (n.kind) {
    case "seq":
      return `S:${n.bytes.join(",")}`;
    case "alt":
      return `A:${n.items.map(refKey).join(",")}`;
    case "ctx":
      return `C:${n.slots.map(refKey).join(",")}`;
  }
}

export const nodeId = (n: Node): Id => hash64(nodeKey(n));

export const seqNode = (bytes: readonly Sym[]): Node => ({
  kind: "seq",
  bytes,
});
export const ctxNode = (slots: readonly Ref[]): Node => ({
  kind: "ctx",
  slots,
});

// alt 的规范形式：成员按 key 排序并去重，所以同一个类在任何机器上 id 相同。
export function altNode(items: readonly Ref[]): Node {
  const byKey = new Map<string, Ref>();
  for (const r of items) byKey.set(refKey(r), r);
  const keys = [...byKey.keys()].sort();
  return { kind: "alt", items: keys.map((k) => byKey.get(k)!) };
}

// 镜像 = 代码（只增的节点集合）+ 数据库（可加的计数）。
// 所有计数都只取决于"节点本身 + 数据"，与其他节点无关，所以对文档是可加的。
export interface Image {
  readonly nodes: ReadonlyMap<Id, Node>;
  readonly ctxCounts: ReadonlyMap<Id, Int32Array>; // 每个 ctx 之后下一个字节的计数
  readonly seqFreq: ReadonlyMap<Id, number>; // 每个 seq 作为词首片段出现的次数
  readonly params: Params;
  readonly tables: Tables;
}

// 解释器的超参数。它们决定了 tables 的含义，所以只有参数相同的镜像才能合并。
// 文件里按千分之一存成整数，保证完全确定。
export interface Params {
  readonly discount: number; // Pitman-Yor 折扣 d
  readonly theta: number; // Pitman-Yor 强度 θ
  readonly kappa: number; // 混合权重的锐度：w = 2^(-κ · 平均码长)
}

export const DEFAULT_PARAMS: Params = { discount: 0.75, theta: 2, kappa: 2 };

// 校准表。全是整数计数（用 Float64Array 存，2^53 以内精确），按预序方式在每篇文档内统计，
// 所以它们和 ctx 计数一样对文档可加。
//   mixBits / mixN：每类叶子（深度 × 证据量）的累计码长（×1024 取整）与次数 -> 混合权重
//   apmHit / apmTot：match 模型在（匹配长度 × 上下文模型概率）每一格里的命中次数与总次数
export interface Tables {
  readonly mixBits: Float64Array;
  readonly mixN: Float64Array;
  readonly apmHit: Float64Array;
  readonly apmTot: Float64Array;
}

export const TABLE_SIZE = 256;

export function emptyTables(): Tables {
  return {
    mixBits: new Float64Array(TABLE_SIZE),
    mixN: new Float64Array(TABLE_SIZE),
    apmHit: new Float64Array(TABLE_SIZE),
    apmTot: new Float64Array(TABLE_SIZE),
  };
}

export const TABLE_KEYS = ["mixBits", "mixN", "apmHit", "apmTot"] as const;

const milli = (x: number) => Math.round(x * 1000);
export function sameParams(a: Params, b: Params): boolean {
  return milli(a.discount) === milli(b.discount) &&
    milli(a.theta) === milli(b.theta) &&
    milli(a.kappa) === milli(b.kappa);
}
