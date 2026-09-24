// 把镜像编译成可执行的索引（派生视图）。编译是纯函数：相同镜像得到相同结构。
import { ctxNode, Id, Image, nodeId, Ref } from "./types.ts";
import { members, ROOT_ID } from "./image.ts";
import { isLetter } from "./text.ts";

// 槽位的一个可选匹配：一段字节，boundary 表示要求左侧是词边界。
export interface SlotAlt {
  readonly bytes: Uint8Array;
  readonly boundary: boolean;
}

export interface CtxInfo {
  readonly id: Id;
  readonly slots: readonly Ref[];
  readonly depth: number;
  readonly first: SlotAlt[] | null; // slots[0] 的匹配器
  readonly tailIdx: number; // slots[1..] 对应的 ctx（结构上的父节点）
  parentIdx: number; // 预测时的先验：有类（alt ctx）覆盖时是最小的那个类，否则是 tail
  classed: boolean; // 是否被某个类覆盖（Fold 的结果）
  counts: Int32Array;
  total: number;
  readonly childIndex: Map<number, number[]>; // 子节点首槽位的末字节 -> 子节点
}

export interface Compiled {
  readonly img: Image;
  readonly ctx: CtxInfo[]; // 按 (深度, 首槽位成员数降序, id) 排序，parentIdx 一定排在前面
  readonly rootIdx: number;
  readonly idxById: Map<Id, number>;
}

export function slotAlts(img: Image, r: Ref): SlotAlt[] {
  if (r.tag === "term") {
    return [{ bytes: Uint8Array.of(r.sym), boundary: false }];
  }
  const n = img.nodes.get(r.id);
  if (!n) throw new Error(`dangling ref ${r.id}`);
  if (n.kind === "seq") {
    return [{ bytes: Uint8Array.from(n.bytes), boundary: true }];
  }
  if (n.kind === "alt") return n.items.flatMap((x) => slotAlts(img, x));
  throw new Error("ctx 不能作为槽位");
}

// 槽位必须恰好在 end 处结束；返回起点，不匹配返回 -1。
// alt 的成员同类（全是字节或全是带词边界的 seq），所以最多一个成员能匹配。
export function matchSlot(
  alts: SlotAlt[],
  text: Uint8Array,
  end: number,
): number {
  outer: for (const a of alts) {
    const s = end - a.bytes.length;
    if (s < 0) continue;
    for (let k = 0; k < a.bytes.length; k++) {
      if (text[s + k] !== a.bytes[k]) continue outer;
    }
    if (a.boundary && s > 0 && isLetter(text[s - 1])) continue;
    return s;
  }
  return -1;
}

export function compile(img: Image): Compiled {
  const entries: { id: Id; slots: readonly Ref[]; size: number }[] = [];
  for (const [id, n] of img.nodes) {
    if (n.kind === "ctx") {
      const size = n.slots.length ? members(img, n.slots[0]).length : 0;
      entries.push({ id, slots: n.slots, size });
    }
  }
  entries.sort((a, b) =>
    a.slots.length - b.slots.length || b.size - a.size ||
    (a.id < b.id ? -1 : a.id > b.id ? 1 : 0)
  );
  const idxById = new Map<Id, number>();
  entries.forEach((e, i) => idxById.set(e.id, i));

  const ctx: CtxInfo[] = entries.map((e) => {
    const counts = img.ctxCounts.get(e.id) ?? new Int32Array(256);
    let t = 0;
    for (let i = 0; i < 256; i++) t += counts[i];
    const tailIdx = e.slots.length === 0
      ? -1
      : idxById.get(nodeId(ctxNode(e.slots.slice(1)))) ?? -1;
    return {
      id: e.id,
      slots: e.slots,
      depth: e.slots.length,
      first: e.slots.length ? slotAlts(img, e.slots[0]) : null,
      tailIdx,
      parentIdx: tailIdx,
      classed: false,
      counts,
      total: t,
      childIndex: new Map(),
    };
  });

  for (let k = 0; k < ctx.length; k++) {
    const c = ctx[k];
    if (c.tailIdx < 0 || !c.first) continue;
    const idx = ctx[c.tailIdx].childIndex;
    for (const a of c.first) {
      const key = a.bytes[a.bytes.length - 1];
      const list = idx.get(key);
      if (!list) idx.set(key, [k]);
      else if (list[list.length - 1] !== k) list.push(k);
    }
  }

  // 类层次：同一个 tail 下，若存在首槽位是 alt X 的 ctx，且 X 严格包含另一个 ctx 首槽位的成员，
  // 就把 X 作为它的先验（选最小的覆盖类，形成 [king] -> [{king|queen}] -> [{king|queen|duke}] -> tail）。
  // 具体上下文不会被替换，所以这个视图是单调的：加节点只会多出信息。
  const groups = new Map<number, number[]>();
  for (let k = 0; k < ctx.length; k++) {
    const t = ctx[k].tailIdx;
    if (t < 0) continue;
    const g = groups.get(t);
    if (g) g.push(k);
    else groups.set(t, [k]);
  }
  for (const g of groups.values()) {
    const mem = g.map((k) => new Set(members(img, ctx[k].slots[0])));
    for (let a = 0; a < g.length; a++) {
      let best = -1;
      for (let b = 0; b < g.length; b++) {
        if (a === b || mem[b].size <= mem[a].size) continue;
        let sup = true;
        for (const x of mem[a]) {
          if (!mem[b].has(x)) {
            sup = false;
            break;
          }
        }
        if (!sup) continue;
        if (best < 0 || mem[b].size < mem[best].size) best = b;
      }
      if (best >= 0) {
        ctx[g[a]].classed = true;
        ctx[g[a]].parentIdx = g[best];
      }
    }
  }

  return { img, ctx, rootIdx: idxById.get(ROOT_ID)!, idxById };
}

// 一次匹配的结果：所有被激活的 ctx，以及每个 ctx 匹配段的起点。
export interface Matched {
  idx: Int32Array;
  start: Int32Array;
  n: number;
  steps: number; // 做了多少次槽位匹配测试：这次预测的计算代价
  stackIdx: Int32Array;
  stackStart: Int32Array;
}

export function newMatched(comp: Compiled): Matched {
  const cap = comp.ctx.length + 1;
  return {
    idx: new Int32Array(cap),
    start: new Int32Array(cap),
    n: 0,
    steps: 0,
    stackIdx: new Int32Array(cap),
    stackStart: new Int32Array(cap),
  };
}

// 在位置 i 处（预测 text[i]，历史为 text[0..i)）沿上下文树向左匹配。
// 每个 ctx 只有一个 tail，且槽位匹配唯一，所以每个 ctx 至多被访问一次。
export function traverse(
  comp: Compiled,
  text: Uint8Array,
  i: number,
  m: Matched,
): void {
  m.n = 0;
  m.steps = 0;
  let sp = 0;
  m.stackIdx[sp] = comp.rootIdx;
  m.stackStart[sp] = i;
  sp++;
  while (sp > 0) {
    sp--;
    const ci = m.stackIdx[sp], j = m.stackStart[sp];
    m.idx[m.n] = ci;
    m.start[m.n] = j;
    m.n++;
    if (j === 0) continue;
    const kids = comp.ctx[ci].childIndex.get(text[j - 1]);
    if (!kids) continue;
    for (const k of kids) {
      m.steps++;
      const s = matchSlot(comp.ctx[k].first!, text, j);
      if (s >= 0) {
        m.stackIdx[sp] = k;
        m.stackStart[sp] = s;
        sp++;
      }
    }
  }
}
