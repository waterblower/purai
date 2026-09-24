import {
  ctxNode,
  DEFAULT_PARAMS,
  emptyTables,
  Id,
  Image,
  Node,
  nodeId,
  Params,
  Ref,
  refKey,
  sameParams,
  TABLE_KEYS,
  Tables,
} from "./types.ts";
import { renderBytes } from "./text.ts";

export const ROOT: Node = ctxNode([]);
export const ROOT_ID: Id = nodeId(ROOT);

export function emptyImage(params: Params = DEFAULT_PARAMS): Image {
  return {
    nodes: new Map([[ROOT_ID, ROOT]]),
    ctxCounts: new Map([[ROOT_ID, new Int32Array(256)]]),
    seqFreq: new Map(),
    params,
    tables: emptyTables(),
  };
}

function combineTables(a: Tables, b: Tables, sign: 1 | -1): Tables {
  const out = emptyTables();
  for (const k of TABLE_KEYS) {
    for (let i = 0; i < out[k].length; i++) {
      const v = a[k][i] + sign * b[k][i];
      if (v < 0) {
        throw new Error(
          `unlearn: 校准表 ${k} 会变成负数，说明被删除的数据并不在这个模型里`,
        );
      }
      out[k][i] = v;
    }
  }
  return out;
}

function checkParams(a: Image, b: Image) {
  if (!sameParams(a.params, b.params)) {
    throw new Error(
      `参数不同的模型不能合并：${JSON.stringify(a.params)} vs ${
        JSON.stringify(b.params)
      }`,
    );
  }
}

export function total(c: Int32Array): number {
  let n = 0;
  for (let i = 0; i < c.length; i++) n += c[i];
  return n;
}

// 合并：节点取并集（G-Set），计数逐项相加（G-Counter）。
// 两者都满足交换律、结合律，所以 merge(a,b) 与 merge(b,a) 逐字节相同。
// 限制：只在一边存在的 ctx，只带着那一边的证据（另一边没有为它计过数）。
export function merge(a: Image, b: Image): Image {
  checkParams(a, b);
  const nodes = new Map(a.nodes);
  for (const [id, n] of b.nodes) nodes.set(id, n);
  const ctxCounts = new Map<Id, Int32Array>();
  for (const src of [a, b]) {
    for (const [id, c] of src.ctxCounts) {
      const acc = ctxCounts.get(id) ?? new Int32Array(256);
      for (let i = 0; i < 256; i++) acc[i] += c[i];
      ctxCounts.set(id, acc);
    }
  }
  const seqFreq = new Map<Id, number>();
  for (const src of [a, b]) {
    for (const [id, f] of src.seqFreq) {
      seqFreq.set(id, (seqFreq.get(id) ?? 0) + f);
    }
  }
  return {
    nodes,
    ctxCounts,
    seqFreq,
    params: a.params,
    tables: combineTables(a.tables, b.tables, 1),
  };
}

// 精确遗忘：减去某份数据贡献的计数（PN-Counter），再裁掉失去全部证据的节点。
export function unlearn(m: Image, d: Image): Image {
  checkParams(m, d);
  const ctxCounts = new Map<Id, Int32Array>();
  for (const [id, c] of m.ctxCounts) ctxCounts.set(id, c.slice());
  for (const [id, c] of d.ctxCounts) {
    const acc = ctxCounts.get(id);
    if (!acc) continue; // 模型里没有的知识，不需要删除
    for (let i = 0; i < 256; i++) {
      acc[i] -= c[i];
      if (acc[i] < 0) {
        throw new Error(
          `unlearn: ctx ${id} 的计数会变成负数，说明被删除的数据并不在这个模型里`,
        );
      }
    }
  }
  const seqFreq = new Map(m.seqFreq);
  for (const [id, f] of d.seqFreq) {
    const cur = seqFreq.get(id);
    if (cur === undefined) continue;
    if (cur < f) throw new Error(`unlearn: seq ${id} 的频数会变成负数`);
    seqFreq.set(id, cur - f);
  }
  const tables = combineTables(m.tables, d.tables, -1);
  return prune({
    nodes: m.nodes,
    ctxCounts,
    seqFreq,
    params: m.params,
    tables,
  });
}

// 规范化：删除没有任何证据的节点。
//   ctx：总计数为 0（根除外）
//   seq：频数为 0 且没被引用
//   alt：没被任何 ctx 引用
export function prune(img: Image): Image {
  const nodes = new Map<Id, Node>();
  const ctxCounts = new Map<Id, Int32Array>();
  const used = new Set<Id>();
  const markRef = (r: Ref) => {
    if (r.tag !== "ref" || used.has(r.id)) return;
    used.add(r.id);
    const n = img.nodes.get(r.id);
    if (n?.kind === "alt") n.items.forEach(markRef);
  };
  for (const [id, n] of img.nodes) {
    if (n.kind !== "ctx") continue;
    const c = img.ctxCounts.get(id) ?? new Int32Array(256);
    if (id !== ROOT_ID && total(c) === 0) continue;
    nodes.set(id, n);
    ctxCounts.set(id, c);
    n.slots.forEach(markRef);
  }
  const seqFreq = new Map<Id, number>();
  for (const [id, n] of img.nodes) {
    if (n.kind === "seq") {
      const f = img.seqFreq.get(id) ?? 0;
      if (f > 0 || used.has(id)) {
        nodes.set(id, n);
        seqFreq.set(id, f);
      }
    } else if (n.kind === "alt" && used.has(id)) {
      nodes.set(id, n);
    }
  }
  return { nodes, ctxCounts, seqFreq, params: img.params, tables: img.tables };
}

export function renderRef(img: Image, r: Ref): string {
  if (r.tag === "term") return renderBytes([r.sym]);
  const n = img.nodes.get(r.id);
  if (!n) return `?${r.id}`;
  switch (n.kind) {
    case "seq":
      return renderBytes(n.bytes);
    case "alt":
      return `{${n.items.map((x) => renderRef(img, x)).join("|")}}`;
    case "ctx":
      return `ctx#${r.id}`;
  }
}

export function renderCtx(img: Image, slots: readonly Ref[]): string {
  if (slots.length === 0) return "ε";
  return slots.map((s) => `[${renderRef(img, s)}]`).join("");
}

// alt 类的成员 key（字节或 seq 自身算一个成员）
export function members(img: Image, r: Ref): string[] {
  if (r.tag === "ref") {
    const n = img.nodes.get(r.id);
    if (n?.kind === "alt") return n.items.map(refKey);
  }
  return [refKey(r)];
}

export function slotKind(img: Image, r: Ref): "byte" | "seq" {
  if (r.tag === "term") return "byte";
  const n = img.nodes.get(r.id);
  if (n?.kind === "alt") return n.items[0].tag === "term" ? "byte" : "seq";
  return "seq";
}
