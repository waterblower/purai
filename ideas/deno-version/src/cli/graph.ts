// 把模型的完整拓扑画成一个交互式 HTML（canvas + 少量 JS）。布局在这里用 Deno 预先算好。
//
// 布局：上下文树是一棵真正的树（每个 ctx 恰好有一个 tail = 去掉最左槽位后的 ctx），
// 用径向树布局：根在圆心，深度 d 的 ctx 在第 d 圈，每棵子树分到的角度正比于它的叶子数。
// seq / alt 节点放在最外面两圈，角度取引用它们的 ctx 的平均方向。
//
// 用法：graph <model.data> [-o model.graph.html] [--explain "文本1|||文本2"]
import { fail, parseArgs } from "./args.ts";
import { load } from "../model_io.ts";
import { compile, newMatched, traverse } from "../compile.ts";
import { renderRef } from "../image.ts";
import { encoder, renderBytes } from "../text.ts";
import { Id } from "../types.ts";

const USAGE = `usage: graph <model.data> [选项]
  -o <file>          输出 HTML（默认 <模型名>.graph.html）
  --explain <text>   预置几次预测的激活路径，多个用 ||| 分隔`;

const a = parseArgs(Deno.args, ["help", "h"]);
if (a.bool("help") || a.bool("h") || a.positional.length !== 1) fail(USAGE);
const modelPath = a.positional[0];
const outPath = a.str("o", modelPath.replace(/\.data$/, "") + ".graph.html")!;
const examples =
  (a.str("explain") ??
    "and to the king|||JULIET:\nO Romeo, Rom|||I pray thee, my lo|||ROMEO:\nBut, soft! what light")
    .split("|||");

const raw = await Deno.readFile(modelPath);
const img = await load(modelPath);
const comp = compile(img);
const n = comp.ctx.length;
const R = 100;
const r1 = (v: number) => Math.round(v * 10) / 10;

// ---------- 上下文树的径向布局 ----------

const kids: number[][] = comp.ctx.map(() => []);
comp.ctx.forEach((c, k) => {
  if (c.tailIdx >= 0) kids[c.tailIdx].push(k);
});
kids.forEach((ks) =>
  ks.sort((x, y) => comp.ctx[y].total - comp.ctx[x].total || x - y)
);

// compile 按深度排序，所以子节点下标一定大于父节点：倒序即可自底向上累计叶子数
const leaves = new Float64Array(n);
for (let k = n - 1; k >= 0; k--) {
  leaves[k] = kids[k].length ? kids[k].reduce((s, c) => s + leaves[c], 0) : 1;
}
const a0 = new Float64Array(n),
  a1 = new Float64Array(n),
  ang = new Float64Array(n);
a0[comp.rootIdx] = -Math.PI;
a1[comp.rootIdx] = Math.PI;
for (let k = 0; k < n; k++) {
  ang[k] = (a0[k] + a1[k]) / 2;
  let cur = a0[k];
  const w = a1[k] - a0[k];
  for (const c of kids[k]) {
    const cw = (w * leaves[c]) / leaves[k];
    a0[c] = cur;
    a1[c] = cur + cw;
    cur += cw;
  }
}
const maxD = Math.max(...comp.ctx.map((c) => c.depth));

// ---------- seq / alt 节点 ----------

const symIds: Id[] = [];
for (const kind of ["alt", "seq"] as const) {
  for (const [id, nd] of img.nodes) if (nd.kind === kind) symIds.push(id);
}
const symIdx = new Map<Id, number>();
symIds.forEach((id, s) => symIdx.set(id, s));
const NS = symIds.length;
const vx = new Float64Array(NS), vy = new Float64Array(NS);

const ctxRef = new Int32Array(n).fill(-1);
const ctxKind = new Uint8Array(n);
comp.ctx.forEach((c, k) => {
  if (c.depth === 0) return void (ctxKind[k] = 3);
  const f = c.slots[0];
  if (f.tag === "term") return void (ctxKind[k] = 0);
  ctxKind[k] = img.nodes.get(f.id)?.kind === "alt" ? 2 : 1;
  const s = symIdx.get(f.id)!;
  ctxRef[k] = s;
  const w = Math.log2(c.total + 2);
  vx[s] += w * Math.cos(ang[k]);
  vy[s] += w * Math.sin(ang[k]);
});
const items: (number[] | null)[] = symIds.map(() => null);
symIds.forEach((id, s) => {
  const nd = img.nodes.get(id)!;
  if (nd.kind !== "alt") return;
  items[s] = nd.items.filter((r) => r.tag === "ref").map((r) =>
    symIdx.get((r as { id: Id }).id)!
  );
  const L = Math.hypot(vx[s], vy[s]) || 1;
  for (const t of items[s]!) {
    vx[t] += vx[s] / L;
    vy[t] += vy[s] / L;
  }
});
// 每一圈按平均方向排序后等距摆放：保持方位，又不会重叠
const symX = new Float64Array(NS), symY = new Float64Array(NS);
const symKind = symIds.map((id) => (img.nodes.get(id)!.kind === "alt" ? 1 : 0));
for (const kind of [1, 0]) {
  const ring = [...Array(NS).keys()].filter((s) => symKind[s] === kind)
    .sort((x, y) => Math.atan2(vy[x], vx[x]) - Math.atan2(vy[y], vx[y]));
  const rad = (maxD + (kind === 1 ? 1.6 : 2.8)) * R;
  ring.forEach((s, i) => {
    const t = -Math.PI + (2 * Math.PI * (i + 0.5)) / ring.length;
    symX[s] = rad * Math.cos(t);
    symY[s] = rad * Math.sin(t);
  });
}

// ---------- 数据 ----------

const topCounts = (c: Int32Array) =>
  [...c.keys()].filter((x) => c[x] > 0).sort((x, y) => c[y] - c[x]).slice(0, 3)
    .map((x) => `${renderBytes([x])}:${c[x]}`).join(" ");

const data = {
  R,
  maxD,
  model: modelPath.split("/").pop(),
  bytes: raw.length,
  params: img.params,
  ctx: {
    x: comp.ctx.map((c, k) => r1(c.depth * R * Math.cos(ang[k]))),
    y: comp.ctx.map((c, k) => r1(c.depth * R * Math.sin(ang[k]))),
    d: comp.ctx.map((c) => c.depth),
    n: comp.ctx.map((c) => c.total),
    tail: comp.ctx.map((c) => c.tailIdx),
    prior: comp.ctx.map((c) => (c.parentIdx !== c.tailIdx ? c.parentIdx : -1)),
    kind: [...ctxKind],
    ref: [...ctxRef],
    lab: comp.ctx.map((
      c,
    ) => (c.depth === 0 ? "ε" : renderRef(img, c.slots[0]))),
    top: comp.ctx.map((c) => topCounts(c.counts)),
  },
  sym: {
    x: [...symX].map(r1),
    y: [...symY].map(r1),
    kind: symKind,
    lab: symIds.map((id) => renderRef(img, { tag: "ref", id })),
    freq: symIds.map((id) => img.seqFreq.get(id) ?? 0),
    items,
  },
  examples: examples.map((text) => {
    const m = newMatched(comp);
    traverse(comp, encoder.encode(text), encoder.encode(text).length, m);
    const matched = [...Array(m.n).keys()].map((k) => m.idx[k]).sort((x, y) =>
      x - y
    );
    const priors = new Set(matched.map((k) => comp.ctx[k].parentIdx));
    return {
      text,
      matched,
      leaves: matched.filter((k) => !priors.has(k)),
      steps: m.steps,
    };
  }),
};

const client = await Deno.readTextFile(
  new URL("./graph_client.js", import.meta.url),
);
const json = JSON.stringify(data).replace(/</g, "\\u003c");
const esc = (s: string) =>
  s.replace(
    /[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!),
  );
const name = esc(data.model ?? "model");

const html = `<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>AFRN 拓扑 · ${name}</title>
<style>
:root { --bg:#fbfbfa; --fg:#1d1d1f; --sub:#6b6b72; --line:#ececea; --card:#fff; --edge:#6b6b72;
  --k-byte:#8a8f98; --k-seq:#2aa36b; --k-alt:#3a6df0; --k-root:#e0553d; --hl:#f08a24;
  font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Helvetica Neue",sans-serif; }
@media (prefers-color-scheme: dark) { :root { --bg:#151517; --fg:#e8e8ea; --sub:#9a9aa3; --line:#26262b;
  --card:#1d1d20; --edge:#a0a0aa; --k-byte:#9aa0aa; --k-seq:#3cc585; --k-alt:#7ea2ff; --hl:#ffa94d; } }
html, body { margin:0; height:100%; background:var(--bg); color:var(--fg); font-size:13px; overflow:hidden; }
header { display:flex; flex-wrap:wrap; gap:14px; align-items:center; padding:8px 14px; border-bottom:1px solid var(--line); }
header h1 { font-size:15px; margin:0; font-weight:600; }
header .sub { color:var(--sub); }
header label { color:var(--sub); cursor:pointer; user-select:none; }
select, input[type=search], button { font:inherit; background:var(--card); color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:3px 6px; }
main { display:flex; height:calc(100% - 46px); }
#wrap { position:relative; flex:1; min-width:0; }
canvas { width:100%; height:100%; display:block; cursor:grab; }
canvas.drag { cursor:grabbing; }
#tip { position:absolute; pointer-events:none; background:var(--card); border:1px solid var(--line); border-radius:6px;
  padding:6px 8px; font-family:ui-monospace,Menlo,monospace; font-size:12px; white-space:pre; display:none; max-width:520px; overflow:hidden;
  box-shadow:0 2px 10px rgba(0,0,0,.12); }
#legend { position:absolute; left:10px; bottom:10px; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px 10px; line-height:1.7; }
.sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }
.ln { display:inline-block; width:18px; height:0; border-top:2px solid; margin-right:6px; vertical-align:3px; }
aside { width:360px; border-left:1px solid var(--line); overflow:auto; padding:12px 14px; background:var(--card); }
aside h3 { margin:4px 0 8px; font-size:14px; }
aside .mono { font-family:ui-monospace,Menlo,monospace; }
aside .kv { color:var(--sub); margin:2px 0; }
aside .kv b { color:var(--fg); font-weight:500; }
aside a { color:var(--k-alt); cursor:pointer; text-decoration:none; }
aside a:hover { text-decoration:underline; }
aside ul { padding-left:16px; margin:4px 0 10px; }
aside li { margin:2px 0; font-family:ui-monospace,Menlo,monospace; font-size:12px; }
.hint { color:var(--sub); line-height:1.6; }
.tag { font-size:11px; background:var(--k-alt); color:#fff; border-radius:4px; padding:0 4px; margin-left:4px; }
</style></head><body>
<header>
  <h1>AFRN 模型拓扑 · ${name}</h1>
  <span class="sub" id="stats"></span>
  <label><input type="checkbox" id="t-tree" checked> 树边</label>
  <label><input type="checkbox" id="t-prior" checked> 类先验链</label>
  <label><input type="checkbox" id="t-refs"> 引用边（ctx→seq/alt）</label>
  <label><input type="checkbox" id="t-syms" checked> seq/alt 圈</label>
  <label><input type="checkbox" id="t-labels" checked> 标签</label>
  <select id="examples"><option value="">— 高亮一次预测 —</option></select>
  <input type="search" id="search" placeholder="搜索：king / {do|kin} / ROMEO" size="22">
  <button id="reset">复位视图</button>
</header>
<main>
  <div id="wrap"><canvas id="cv"></canvas><div id="tip"></div>
    <div id="legend">
      <div><span class="sw" style="background:var(--k-root)"></span>根 ε（圆心）</div>
      <div><span class="sw" style="background:var(--k-byte)"></span>ctx，新槽位是字节</div>
      <div><span class="sw" style="background:var(--k-seq)"></span>ctx，新槽位是 seq / 外圈 seq 节点</div>
      <div><span class="sw" style="background:var(--k-alt)"></span>ctx，新槽位是类 / 内外圈之间的 alt 节点</div>
      <div><span class="ln" style="border-color:var(--edge)"></span>树边：tail（去掉最左槽位）</div>
      <div><span class="ln" style="border-color:var(--k-alt);border-top-style:dashed"></span>类先验链</div>
      <div class="hint">第 d 圈 = 有 d 个槽位的上下文 · 滚轮缩放 · 拖动平移 · 点击查看</div>
    </div>
  </div>
  <aside id="info"></aside>
</main>
<script type="application/json" id="graph-data">${json}</script>
<script>${client.replace(/<\/script/gi, "<\\/script")}</script>
</body></html>`;

await Deno.writeTextFile(outPath, html);
console.log(
  `wrote ${outPath} (${
    (html.length / 1024).toFixed(0)
  } KB; ${n} ctx, ${NS} seq/alt)`,
);
