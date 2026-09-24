// 训练 = 反复执行：计数 -> 提议（Chunk / Fork / Fold）-> 按 ΔMDL 筛选 -> 改写镜像。
import {
  altNode,
  ctxNode,
  Id,
  Image,
  Node,
  nodeId,
  Ref,
  ref,
  refKey,
  seqNode,
  term,
} from "./types.ts";
import { compile, Compiled, newMatched, traverse } from "./compile.ts";
import { members, prune, slotKind, total } from "./image.ts";
import { nodeBits, pyBits, pyPredictive } from "./mdl.ts";
import { buildPredictor, evalBits } from "./predict.ts";
import { isLetter } from "./text.ts";
import { addTables, countDoc } from "./calib.ts";
import { compact } from "./compact.ts";
import { emptyTables, Tables } from "./types.ts";

const UNIFORM = new Float64Array(256).fill(1 / 256);

export interface TrainOptions {
  rounds: number;

  forksPerRound: number;
  foldsPerRound: number;
  chunksPerRound: number;
  minCount: number; // 新 ctx 至少要有多少次出现
  minChunkCount: number;
  maxDepth: number; // ctx 最多几个槽位
  maxChunkLen: number;
  maxCandidates: number; // 每轮进入精确评估的 Fork 候选数上限
  foldTopK: number; // 每组兄弟节点里参与 Fold 配对的数量
  structWeight: number; // 结构码长的权重（1 = 标准 MDL）
  foldWordsOnly: boolean; // 只对词（seq）做 Fold
  compact: boolean; // 训练结束时做观察等价压缩
  compactMin: number; // 只合并证据至少这么多的组（证据少时"计数相同"常常只是巧合）
}

export const defaultOptions: TrainOptions = {
  rounds: 20,

  forksPerRound: 50000,
  foldsPerRound: 50,
  chunksPerRound: 100,
  minCount: 2,
  minChunkCount: 8,
  maxDepth: 12,
  maxChunkLen: 16,
  maxCandidates: 200000,
  foldTopK: 24,
  // 只用数据项（Pitman-Yor 边际似然，自带 Occam 惩罚）验证集最好。
  // 调到 0.1～1 模型更小但 bpc 更差，见 README。
  structWeight: 0,
  foldWordsOnly: true,
  // 实测：bpc 2.2070 -> 2.2167，每字节计算步数 99 -> 35。见 README
  compact: true,
  compactMin: 20,
};

export interface RoundReport {
  round: number;
  ctxs: number;
  seqs: number;
  alts: number;
  classed: number;
  modelBits: number;
  validBpc: number;
  added: { chunk: number; fork: number; fold: number };
  seconds: number;
}

// ---------- 词首片段的 trie（用于 seq 匹配与 Chunk 提议）----------

class Trie {
  next: Map<number, number>[] = [new Map()];
  slot: number[] = [-1];
  depth: number[] = [0];
  insert(bytes: readonly number[], slot: number) {
    let n = 0;
    for (const b of bytes) {
      let nx = this.next[n].get(b);
      if (nx === undefined) {
        nx = this.next.length;
        this.next.push(new Map());
        this.slot.push(-1);
        this.depth.push(this.depth[n] + 1);
        this.next[n].set(b, nx);
      }
      n = nx;
    }
    this.slot[n] = slot;
  }
  step(n: number, b: number): number {
    return this.next[n].get(b) ?? -1;
  }
}

interface Vocab {
  seqIds: Id[];
  trie: Trie;
}

function buildVocab(img: Image): Vocab {
  const seqIds = [...img.nodes].filter(([, n]) => n.kind === "seq").map((
    [id],
  ) => id).sort();
  const trie = new Trie();
  seqIds.forEach((id, k) => {
    const n = img.nodes.get(id)!;
    if (n.kind === "seq") trie.insert(n.bytes, k);
  });
  return { seqIds, trie };
}

// 统计每个 seq 的词首频数；同时统计"把某个 seq 再延长一个字母"的候选频数。
function scanWords(
  text: Uint8Array,
  v: Vocab,
  maxLen: number,
  seqFreq: Int32Array,
  cand: Map<string, number> | null,
) {
  const n = text.length;
  let i = 0;
  while (i < n) {
    if (!isLetter(text[i])) {
      i++;
      continue;
    }
    const ws = i;
    while (i < n && isLetter(text[i])) i++;
    const len = Math.min(i - ws, maxLen);
    let node = v.trie.step(0, text[ws]);
    for (let L = 2; L <= len; L++) {
      const prevOk = L === 2 || (node >= 0 && v.trie.slot[node] >= 0);
      const nx = node >= 0 ? v.trie.step(node, text[ws + L - 1]) : -1;
      if (nx >= 0 && v.trie.slot[nx] >= 0) {
        seqFreq[v.trie.slot[nx]]++;
        node = nx;
        continue;
      }
      if (prevOk && cand) {
        const key = String.fromCharCode(...text.subarray(ws, ws + L));
        cand.set(key, (cand.get(key) ?? 0) + 1);
      }
      if (nx < 0) break;
      node = nx;
    }
  }
}

// seqEnd[j] = 以 j 结尾（不含）、从词首开始的片段对应的 seq 编号，没有则为 -1。
function computeSeqEnd(text: Uint8Array, v: Vocab): Int32Array {
  const out = new Int32Array(text.length + 1).fill(-1);
  let node = -1;
  for (let p = 0; p < text.length; p++) {
    const b = text[p];
    if (!isLetter(b)) {
      node = -1;
      continue;
    }
    node = p === 0 || !isLetter(text[p - 1])
      ? v.trie.step(0, b)
      : node >= 0
      ? v.trie.step(node, b)
      : -1;
    if (node >= 0 && v.trie.slot[node] >= 0) out[p + 1] = v.trie.slot[node];
  }
  return out;
}

// ---------- 重新计数（权威状态的"视图"：计数只取决于节点与数据）----------

export function recount(
  img: Image,
  texts: Uint8Array[],
  maxChunkLen = defaultOptions.maxChunkLen,
): Image {
  const comp = compile(img);
  const { counts, tables } = countDocs(comp, texts, img);
  const v = buildVocab(img);
  const freq = new Int32Array(v.seqIds.length);
  for (const text of texts) scanWords(text, v, maxChunkLen, freq, null);
  const ctxCounts = new Map<Id, Int32Array>();
  comp.ctx.forEach((c, k) => ctxCounts.set(c.id, counts[k]));
  const seqFreq = new Map<Id, number>();
  v.seqIds.forEach((id, k) => seqFreq.set(id, freq[k]));
  return { nodes: img.nodes, ctxCounts, seqFreq, params: img.params, tables };
}

// 逐篇文档计数再相加：每篇文档的贡献只取决于"代码 + 这篇文档"，这是精确遗忘的前提。
function countDocs(
  comp: Compiled,
  texts: Uint8Array[],
  img: Image,
  onMatched?: (
    doc: number,
    m: import("./compile.ts").Matched,
    i: number,
  ) => void,
): { counts: Int32Array[]; tables: Tables } {
  const counts = comp.ctx.map(() => new Int32Array(256));
  const tables = emptyTables();
  texts.forEach((text, d) => {
    const r = countDoc(
      comp,
      text,
      img.params,
      onMatched && ((m, i) => onMatched(d, m, i)),
    );
    for (let k = 0; k < counts.length; k++) {
      const a = counts[k], b = r.counts[k];
      for (let x = 0; x < 256; x++) a[x] += b[x];
    }
    addTables(tables, r.tables);
  });
  return { counts, tables };
}

// ---------- 一轮训练 ----------

interface Slots {
  refs: Ref[]; // 0..255 字节，然后是 seq，然后是 alt
  altsContaining: (number[] | undefined)[];
  altItems: (number[] | undefined)[]; // alt 槽位 -> 成员槽位
}

function buildSlots(img: Image, v: Vocab): Slots {
  const refs: Ref[] = [];
  for (let b = 0; b < 256; b++) refs.push(term(b));
  for (const id of v.seqIds) refs.push(ref(id));
  const altIds = [...img.nodes].filter(([, n]) => n.kind === "alt").map((
    [id],
  ) => id).sort();
  const slotOf = new Map<string, number>();
  refs.forEach((r, k) => slotOf.set(refKey(r), k));
  const altsContaining: (number[] | undefined)[] = [];
  const altItems: (number[] | undefined)[] = [];
  for (const id of altIds) {
    const s = refs.length;
    refs.push(ref(id));
    const n = img.nodes.get(id)!;
    if (n.kind !== "alt") continue;
    altItems[s] = [];
    for (const it of n.items) {
      const k = slotOf.get(refKey(it));
      if (k === undefined) continue;
      altItems[s]!.push(k);
      (altsContaining[k] ??= []).push(s);
    }
  }
  return { refs, altsContaining, altItems };
}

interface Accepted {
  kind: "chunk" | "fork" | "fold";
  delta: number;
  key: string;
  nodes: [Id, Node][];
  counts?: [Id, Int32Array];
  seqFreq?: [Id, number];
  uses?: number[]; // fold 用到的 ctx，同一轮内不能重复使用
}

function modelBitsOf(img: Image, comp: Compiled): number {
  const N = img.nodes.size;
  let bits = 0;
  for (const c of comp.ctx) {
    if (!c.classed) bits += nodeBits(ctxNode(c.slots), N);
  }
  for (const n of img.nodes.values()) {
    if (n.kind !== "ctx") bits += nodeBits(n, N);
  }
  return bits;
}

export function train(
  init: Image,
  docs: Uint8Array[],
  validDocs: Uint8Array[],
  opts: TrainOptions,
  onRound?: (r: RoundReport) => void,
  onCompact?: (removed: number) => void,
): Image {
  let img = recount(init, docs, opts.maxChunkLen);
  const validBytes = validDocs.reduce((s, d) => s + d.length, 0);
  for (let round = 1; round <= opts.rounds; round++) {
    const t0 = performance.now();
    const comp = compile(img);
    const v = buildVocab(img);
    const S = buildSlots(img, v);
    const SL = S.refs.length;
    const seqEnds = docs.map((t) => computeSeqEnd(t, v));
    const m = newMatched(comp);

    // 第 1 遍：预序计数 + 校准表，同时统计每个 (父 ctx, 左侧槽位) 候选的出现次数
    const candTotals = new Map<number, number>();
    const inc = (k: number) => candTotals.set(k, (candTotals.get(k) ?? 0) + 1);
    const pass1 = countDocs(comp, docs, img, (d, mm) => {
      const text = docs[d], seqEnd = seqEnds[d];
      for (let k = 0; k < mm.n; k++) {
        const ci = mm.idx[k], c = comp.ctx[ci];
        const j = mm.start[k];
        if (c.depth >= opts.maxDepth || j === 0) continue;
        const base = ci * SL;
        const lb = text[j - 1];
        inc(base + lb);
        S.altsContaining[lb]?.forEach((a) => inc(base + a));
        const s = seqEnd[j];
        if (s >= 0) {
          inc(base + 256 + s);
          S.altsContaining[256 + s]?.forEach((a) => inc(base + a));
        }
      }
    });
    comp.ctx.forEach((c, k) => {
      c.counts = pass1.counts[k];
      c.total = total(c.counts);
    });
    img = { ...img, tables: pass1.tables };

    const pr = buildPredictor(comp, img.params, img.tables);
    const N = img.nodes.size;

    // 每个父节点下已有的字面子节点成员。类是在别处学到的假设，
    // 只在"还没有具体证据"的位置扩散，否则会把别处的归类强加到这里。
    const literalKids = new Map<number, Set<string>>();
    for (const c of comp.ctx) {
      if (c.tailIdx < 0) continue;
      const first = c.slots[0];
      if (first.tag === "ref" && img.nodes.get(first.id)?.kind === "alt") {
        continue;
      }
      let set = literalKids.get(c.tailIdx);
      if (!set) literalKids.set(c.tailIdx, set = new Set());
      set.add(refKey(first));
    }

    // Fork 候选：过滤、去重，取出现次数最多的一批
    type ForkCand = {
      key: number;
      ci: number;
      node: Node;
      id: Id;
      total: number;
    };
    let forks: ForkCand[] = [];
    for (const [key, tot] of candTotals) {
      if (tot < opts.minCount) continue;
      const ci = Math.floor(key / SL), slot = key % SL;
      const parent = comp.ctx[ci];
      const r = S.refs[slot];
      // 类槽位只有在至少两个成员真正出现过时才有泛化意义，否则与字面上下文重复
      const items = S.altItems[slot];
      if (items) {
        if (items.some((it) => candTotals.get(ci * SL + it) === tot)) continue;
        const lit = literalKids.get(ci);
        if (lit && items.some((it) => lit.has(refKey(S.refs[it])))) continue;
      }
      const node = ctxNode([r, ...parent.slots]);
      const id = nodeId(node);
      if (img.nodes.has(id)) continue;
      forks.push({ key, ci, node, id, total: tot });
    }
    forks.sort((a, b) => b.total - a.total || (a.id < b.id ? -1 : 1));
    forks = forks.slice(0, opts.maxCandidates);

    // 第 2 遍：为候选收集下一个字节的分布
    const dist = new Map<number, Int32Array>();
    for (const f of forks) dist.set(f.key, new Int32Array(256));
    for (let d = 0; forks.length && d < docs.length; d++) {
      const trainText = docs[d], seqEnd = seqEnds[d];
      for (let i = 0; i < trainText.length; i++) {
        traverse(comp, trainText, i, m);
        const b = trainText[i];
        for (let k = 0; k < m.n; k++) {
          const ci = m.idx[k], c = comp.ctx[ci];
          const j = m.start[k];
          if (c.depth >= opts.maxDepth || j === 0) continue;
          const base = ci * SL;
          const add = (key: number) => {
            const d = dist.get(key);
            if (d) d[b]++;
          };
          const lb = trainText[j - 1];
          add(base + lb);
          S.altsContaining[lb]?.forEach((a) => add(base + a));
          const s = seqEnd[j];
          if (s >= 0) {
            add(base + 256 + s);
            S.altsContaining[256 + s]?.forEach((a) => add(base + a));
          }
        }
      }
    }

    const acc: Accepted[] = [];

    // Fork 的 ΔMDL。码长用与预测器相同的 Pitman-Yor 模型，且与数据顺序无关：
    //   之前：这些位置由父上下文编码 = PY(父) - PY(父 - 子)
    //   之后：由子上下文编码，基分布是"去掉子数据以后"的父预测（避免用自己的数据当先验）
    const { discount: pd, theta: pt } = img.params;
    const rest = new Int32Array(256), Prest = new Float64Array(256);
    for (const f of forks) {
      const c = dist.get(f.key)!;
      const parent = comp.ctx[f.ci];
      const G = parent.parentIdx >= 0 ? pr.P[parent.parentIdx] : UNIFORM;
      for (let x = 0; x < 256; x++) rest[x] = parent.counts[x] - c[x];
      pyPredictive(rest, G, pd, pt, Prest);
      const before = pyBits(parent.counts, G, pd, pt) - pyBits(rest, G, pd, pt);
      const delta = pyBits(c, Prest, pd, pt) - before +
        opts.structWeight * nodeBits(f.node, N);

      if (delta < 0) {
        acc.push({
          kind: "fork",
          delta,
          key: f.id,
          nodes: [[f.id, f.node]],
          counts: [f.id, c],
        });
      }
    }

    // Fold 候选：同一父节点下、同类槽位、还没有归入任何类的兄弟 ctx 两两比较。
    // 类本身也可以再和别的 ctx 合并，所以这是自底向上的层次聚类（类似 Brown clustering）。
    const groups = new Map<string, number[]>();
    for (let k = 0; k < comp.ctx.length; k++) {
      const c = comp.ctx[k];
      if (c.tailIdx < 0 || c.classed) continue;
      if (c.total < opts.minCount) continue;
      if (opts.foldWordsOnly && slotKind(img, c.slots[0]) !== "seq") continue;
      const g = `${c.tailIdx}:${slotKind(img, c.slots[0])}`;
      (groups.get(g) ?? groups.set(g, []).get(g)!).push(k);
    }
    const folds: Accepted[] = [];
    const sum = new Int32Array(256);
    const QA = new Float64Array(256), QB = new Float64Array(256);
    for (const g of groups.values()) {
      g.sort((a, b) => comp.ctx[b].total - comp.ctx[a].total || a - b);
      const top = g.slice(0, opts.foldTopK);
      for (let x = 0; x < top.length; x++) {
        for (let y = x + 1; y < top.length; y++) {
          const A = comp.ctx[top[x]], B = comp.ctx[top[y]];
          const mA = members(img, A.slots[0]),
            mB = new Set(members(img, B.slots[0]));
          if (mA.some((z) => mB.has(z))) continue;
          const P = pr.P[A.tailIdx];
          for (let z = 0; z < 256; z++) sum[z] = A.counts[z] + B.counts[z];
          const items: Ref[] = [];
          for (const r of [A.slots[0], B.slots[0]]) {
            const n = r.tag === "ref" ? img.nodes.get(r.id) : undefined;
            if (n?.kind === "alt") items.push(...n.items);
            else items.push(r);
          }
          const alt = altNode(items);
          const altId = nodeId(alt);
          const cn = ctxNode([ref(altId), ...A.slots.slice(1)]);
          const cid = nodeId(cn);
          if (img.nodes.has(cid)) continue;
          // 之前：A、B 各自以 tail 的预测为先验。
          // 之后：类成为它们的先验。为避免用自己的数据当先验，用留一法：
          //   A 的先验只含 B 的数据，B 的先验只含 A 的数据。
          pyPredictive(B.counts, P, pd, pt, QA);
          pyPredictive(A.counts, P, pd, pt, QB);
          const dData = pyBits(A.counts, QA, pd, pt) +
            pyBits(B.counts, QB, pd, pt) -
            pyBits(A.counts, P, pd, pt) - pyBits(B.counts, P, pd, pt);
          const dModel = nodeBits(cn, N) +
            (img.nodes.has(altId) ? 0 : nodeBits(alt, N));
          const delta = dData + opts.structWeight * dModel;
          if (delta < 0) {
            folds.push({
              kind: "fold",
              delta,
              key: cid,
              nodes: [[altId, alt], [cid, cn]],
              counts: [cid, sum.slice()],
              uses: [top[x], top[y]],
            });
          }
        }
      }
    }

    // Chunk 候选：高频的词首片段延长（频率启发式，不是 MDL；见 README）
    const freq = new Int32Array(v.seqIds.length);
    const chunkCand = new Map<string, number>();
    for (const t of docs) scanWords(t, v, opts.maxChunkLen, freq, chunkCand);
    const chunks = [...chunkCand].filter(([, n]) => n >= opts.minChunkCount)
      .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
      .slice(0, opts.chunksPerRound)
      .map(([s, n]): Accepted => {
        const node = seqNode([...s].map((ch) => ch.charCodeAt(0)));
        const id = nodeId(node);
        return {
          kind: "chunk",
          delta: -n,
          key: id,
          nodes: [[id, node]],
          seqFreq: [id, n],
        };
      });

    const byDelta = (a: Accepted, b: Accepted) =>
      a.delta - b.delta || (a.key < b.key ? -1 : 1);
    const chosen: Accepted[] = [];
    acc.sort(byDelta);
    chosen.push(...acc.slice(0, opts.forksPerRound));
    folds.sort(byDelta);
    const used = new Set<number>(), seen = new Set<string>();
    let nFold = 0;
    for (const f of folds) {
      if (nFold >= opts.foldsPerRound) break;
      if (f.uses!.some((u) => used.has(u)) || seen.has(f.key)) continue;
      f.uses!.forEach((u) => used.add(u));
      seen.add(f.key);
      chosen.push(f);
      nFold++;
    }
    chosen.push(...chunks);

    const report: RoundReport = {
      round,
      ctxs: comp.ctx.length,
      seqs: v.seqIds.length,
      alts: [...img.nodes.values()].filter((n) => n.kind === "alt").length,
      classed: comp.ctx.filter((c) => c.classed).length,
      modelBits: modelBitsOf(img, comp),
      validBpc: validBytes
        ? validDocs.reduce((s, t) => s + evalBits(pr, t).bits, 0) / validBytes
        : NaN,
      added: {
        chunk: chunks.length,
        fork: Math.min(acc.length, opts.forksPerRound),
        fold: nFold,
      },
      seconds: 0,
    };

    if (chosen.length === 0) {
      report.seconds = (performance.now() - t0) / 1000;
      onRound?.(report);
      break;
    }

    // 改写镜像：只增加节点；新节点的计数在上面已精确算出，无需重新扫描
    const nodes = new Map(img.nodes);
    const ctxCounts = new Map(img.ctxCounts);
    const seqFreq = new Map(img.seqFreq);
    for (const a of chosen) {
      for (const [id, n] of a.nodes) nodes.set(id, n);
      if (a.counts) ctxCounts.set(a.counts[0], a.counts[1]);
      if (a.seqFreq) seqFreq.set(a.seqFreq[0], a.seqFreq[1]);
    }
    img = { nodes, ctxCounts, seqFreq, params: img.params, tables: img.tables };
    report.seconds = (performance.now() - t0) / 1000;
    onRound?.(report);
  }
  if (opts.compact) {
    const r = compact(img, opts.compactMin);
    img = r.img;
    onCompact?.(r.removed);
  }
  // 校准表要对应最终的结构，所以最后再完整计数一遍
  return prune(recount(img, docs, opts.maxChunkLen));
}

export function stats(img: Image) {
  let ctxs = 0, seqs = 0, alts = 0;
  for (const n of img.nodes.values()) {
    if (n.kind === "ctx") ctxs++;
    else if (n.kind === "seq") seqs++;
    else alts++;
  }
  let tokens = 0;
  for (const c of img.ctxCounts.values()) tokens = Math.max(tokens, total(c));
  return { ctxs, seqs, alts, tokens };
}
