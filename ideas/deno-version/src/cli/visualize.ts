// 把模型可视化成一个静态 HTML 文件。所有内容都在这里用 Deno 算好，页面本身不需要 JS。
// 用法：visualize <model.data> [-o model.html] [--text sample.txt] [--prompt "ROMEO:"] [--n 400]
import { fail, parseArgs } from "./args.ts";
import { load } from "../model_io.ts";
import { compile, newMatched, traverse } from "../compile.ts";
import { buildPredictor, classify, generate, Session } from "../predict.ts";
import { members, renderCtx, renderRef } from "../image.ts";
import { decoder, encoder, renderBytes } from "../text.ts";
import { rng } from "../hash.ts";
import { Id, Ref, ref } from "../types.ts";

const USAGE = `usage: visualize <model.data> [选项]
  -o <file>          输出 HTML（默认与模型同名 .html）
  --text <file>      用来做逐字节追踪的文本（默认内置一小段）
  --max-bytes <n>    追踪文本最多取多少字节（默认 1500）
  --prompt <text>    生成样本的开头（默认 "ROMEO:"）
  --n <k>            生成多少字节（默认 400）
  --temperature <t>  采样温度（默认 0.8）
  --seed <s>         随机种子（默认 42）
  --tree-kids <k>    上下文树每个节点最多展示的子节点数（默认 24）
  --tree-depth <d>   上下文树最多展开的深度（默认 4）
  --explain <text>   在"代码"部分展示哪几次预测涉及的节点，多个用 ||| 分隔
                     （默认 "and to the king|||JULIET:\nO Romeo, Rom|||I pray thee, my lo"）`;

const a = parseArgs(Deno.args, ["help", "h"]);
if (a.bool("help") || a.bool("h") || a.positional.length !== 1) fail(USAGE);
const modelPath = a.positional[0];
const outPath = a.str("o", modelPath.replace(/\.data$/, "") + ".html")!;

const DEFAULT_TEXT = `ROMEO:
But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.

JULIET:
O Romeo, Romeo! wherefore art thou Romeo?

ROMEO:
But, soft! what light through yonder window breaks?
`;

const raw = await Deno.readFile(modelPath);
const img = await load(modelPath);
const comp = compile(img);
const pr = buildPredictor(comp, img.params, img.tables, true);

const esc = (s: string) =>
  s.replace(
    /[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!),
  );
const fmt = (n: number) => n.toLocaleString("en");
const heat = (x: number, max: number) =>
  `hsl(${
    Math.round(120 * (1 - Math.max(0, Math.min(1, x / max))))
  } 75% var(--heat-l))`;
const topK = (p: Float64Array, k: number) =>
  [...p.keys()].sort((x, y) => p[y] - p[x]).slice(0, k);
const sym = (b: number) => esc(renderBytes([b]));
const charHtml = (b: number) =>
  b === 10 ? "↵<br>" : b === 32 ? "&nbsp;" : esc(String.fromCharCode(b));

const kids: number[][] = comp.ctx.map(() => []);
comp.ctx.forEach((c, k) => {
  if (c.tailIdx >= 0) kids[c.tailIdx].push(k);
});
kids.forEach((ks) => ks.sort((x, y) => comp.ctx[y].total - comp.ctx[x].total));
const seqs = [...img.nodes].filter(([, n]) => n.kind === "seq");
const alts = [...img.nodes].filter(([, n]) => n.kind === "alt");

// ---------- 概览 ----------

function overview(): string {
  const hist = new Array(16).fill(0);
  for (const c of comp.ctx) hist[Math.min(15, c.depth)]++;
  const maxH = Math.max(...hist);
  const w = 22, h = 70;
  const bars = hist.map((n, d) => {
    const bh = n ? Math.max(1, (n / maxH) * (h - 16)) : 0;
    return `<rect x="${d * w + 2}" y="${h - 14 - bh}" width="${
      w - 4
    }" height="${bh}" rx="2" fill="var(--accent)"><title>深度 ${d}：${
      fmt(n)
    } 个上下文</title></rect><text x="${d * w + w / 2}" y="${
      h - 2
    }" text-anchor="middle">${d}</text>`;
  }).join("");
  const stat = (v: string, label: string) =>
    `<div class="stat"><b>${v}</b>${label}</div>`;
  return `<div id="overview">
    ${stat(fmt(comp.ctx.length), "上下文 ctx")}
    ${stat(fmt(comp.ctx.filter((c) => c.classed).length), "被类覆盖的 ctx")}
    ${stat(fmt(seqs.length), "词首片段 seq")}
    ${stat(fmt(alts.length), "类 alt")}
    ${stat(`${(raw.length / 1024).toFixed(0)} KB`, "模型文件")}
    ${stat(fmt(comp.ctx[comp.rootIdx].total), "训练字节")}
    ${
    stat(
      `d=${img.params.discount} θ=${img.params.theta} κ=${img.params.kappa}`,
      "解释器参数",
    )
  }
    <div class="stat"><svg width="${
    16 * w
  }" height="${h}" class="hist">${bars}</svg>上下文深度分布（槽位数）</div>
  </div>`;
}

// ---------- 逐字节追踪 ----------

function traceSection(): string {
  const path = a.str("text");
  let bytes = path ? Deno.readFileSync(path) : encoder.encode(DEFAULT_TEXT);
  bytes = bytes.subarray(0, a.num("max-bytes", 1500));
  const s = new Session(pr);
  const out = new Float64Array(256);
  const rows: {
    b: number;
    bits: number;
    steps: number;
    tip: string;
    cls: string;
  }[] = [];
  for (let i = 0; i < bytes.length; i++) {
    const step = s.predict(bytes, i, out);
    const b = bytes[i];
    const bits = -Math.log2(out[b]);
    let W = 0;
    for (const ci of step.leaves) W += pr.weight[ci];
    const leaves = step.leaves.slice().sort((x, y) =>
      pr.weight[y] - pr.weight[x]
    ).slice(0, 4)
      .map((ci) => {
        const c = comp.ctx[ci];
        return `  ${renderCtx(img, c.slots)}  w=${
          ((pr.weight[ci] / W) * 100).toFixed(0)
        }% N=${c.total}`;
      });
    const guess = topK(out, 3).map((x) =>
      `${renderBytes([x])} ${(out[x] * 100).toFixed(0)}%`
    ).join("  ");
    const tip = [
      `实际 ${renderBytes([b])}：${(out[b] * 100).toFixed(1)}%，${
        bits.toFixed(2)
      } bits，${step.steps} 步`,
      `不确定性：${classify(pr, step, out)}`,
      step.matchByte >= 0
        ? `match：长度 ${step.matchLen}，建议 ${renderBytes([step.matchByte])}`
        : "match：无",
      `模型最想要：${guess}`,
      `叶子上下文（${step.leaves.length}）：`,
      ...leaves,
    ].join("\n");
    rows.push({
      b,
      bits,
      steps: step.steps,
      tip,
      cls: step.matchByte === b ? "m" : "",
    });
    s.observe(bytes, i);
  }
  const n = rows.length;
  const bpc = rows.reduce((x, r) => x + r.bits, 0) / n;
  const avgSteps = rows.reduce((x, r) => x + r.steps, 0) / n;
  const maxSteps = Math.max(...rows.map((r) => r.steps));
  const render = (by: "bits" | "steps") =>
    rows.map((r) =>
      `<span class="ch ${r.cls}" style="background:${
        heat(by === "bits" ? r.bits : r.steps, by === "bits" ? 8 : maxSteps)
      }" title="${esc(r.tip)}">${charHtml(r.b)}</span>`
    ).join("");
  const w = Math.max(1.5, Math.min(6, 900 / n));
  const chart = rows.map((r, i) => {
    const h = (r.steps / maxSteps) * 70;
    return `<rect x="${(i * w).toFixed(1)}" y="${(72 - h).toFixed(1)}" width="${
      (w * 0.85).toFixed(1)
    }" height="${h.toFixed(1)}" fill="${heat(r.bits, 8)}"><title>${
      sym(r.b)
    }：${r.steps} 步，${r.bits.toFixed(2)} bits</title></rect>`;
  }).join("");
  const worst = rows.map((r, i) => ({ ...r, i })).sort((x, y) =>
    y.bits - x.bits
  ).slice(0, 10);
  const ctxOf = (i: number) =>
    esc(renderBytes(bytes.subarray(Math.max(0, i - 12), i)));
  return `<section><h2>1. 逐字节追踪</h2>
    <p class="hint">模型逐字节预测这段文本（每个位置只看它之前的内容）。${
    fmt(n)
  } 字节，<b>${bpc.toFixed(3)}</b> bpc，
    平均 <b>${avgSteps.toFixed(1)}</b> 步/字节（最少 ${
    Math.min(...rows.map((r) => r.steps))
  }，最多 ${maxSteps}）。
    <b>鼠标悬停任意字符</b>，可以看到这一步用了哪些上下文、match 是否命中。带下划线的字符是 match 模型猜中的。</p>
    <h3>颜色 = 意外程度（绿 0 bits → 红 8+ bits）</h3><div class="text">${
    render("bits")
  }</div>
    <h3>颜色 = 计算步数（绿 少 → 红 多）</h3><div class="text">${
    render("steps")
  }</div>
    <h3>每个字节的计算步数（柱高 = 步数，颜色 = 意外程度）</h3>
    <div class="chart"><svg width="${
    (n * w).toFixed(0)
  }" height="74">${chart}</svg></div>
    <p class="hint">第二次出现的那句台词几乎全绿：match 模型在前文里找到了它，逐字照抄。计算步数则取决于上下文树匹配得有多深，和"意外程度"不是一回事。</p>
    <h3>最意外的 10 个位置</h3>
    <table><tr><th>位置</th><th>前文</th><th>实际</th><th>bits</th><th>步数</th></tr>${
    worst.map((r) =>
      `<tr><td>${r.i}</td><td class="mono">…${
        ctxOf(r.i)
      }</td><td class="mono">${sym(r.b)}</td><td>${
        r.bits.toFixed(2)
      }</td><td>${r.steps}</td></tr>`
    ).join("")
  }</table></section>`;
}

// ---------- 生成 ----------

function generateSection(): string {
  const prompt = a.str("prompt", "ROMEO:")!;
  const n = a.num("n", 400);
  const temperature = a.num("temperature", 0.8);
  const seed = a.num("seed", 42);
  const t = performance.now();
  const { bytes, stats } = generate(pr, encoder.encode(prompt), n, {
    temperature,
    random: rng(seed),
  });
  const ms = performance.now() - t;
  const maxSteps = Math.max(1, ...stats.map((s) => s.steps));
  const kinds = { confident: 0, aleatoric: 0, epistemic: 0 };
  stats.forEach((s) => kinds[s.uncertainty]++);
  const body = [...bytes].map((b, i) => {
    const s = stats[i];
    return `<span class="ch" style="background:${
      heat(s.steps, maxSteps)
    }" title="${s.steps} 步 · ${s.uncertainty}${
      s.matchLen ? ` · match 长度 ${s.matchLen}` : ""
    }">${charHtml(b)}</span>`;
  }).join("");
  void decoder;
  return `<section><h2>2. 生成样本</h2>
    <p class="hint">开头 <span class="mono">${
    esc(JSON.stringify(prompt))
  }</span>，温度 ${temperature}，种子 ${seed}。
    ${n} 字节用时 ${ms.toFixed(0)} ms。颜色 = 每个字节的计算步数。
    不确定性：确定 ${kinds.confident} / 偶然 ${kinds.aleatoric} / 认知 ${kinds.epistemic}。</p>
    <div class="text"><span class="prompt">${
    esc(prompt).replace(/\n/g, "<br>")
  }</span>${body}</div></section>`;
}

// ---------- 上下文树 ----------

function treeSection(): string {
  const maxKids = a.num("tree-kids", 24), maxDepth = a.num("tree-depth", 4);
  let budget = 5000;
  const node = (k: number, depth: number): string => {
    budget--;
    const c = comp.ctx[k];
    const P = pr.P[k];
    const label = c.depth === 0
      ? "ε（根：不看任何上下文）"
      : `[${renderRef(img, c.slots[0])}]`;
    const chips = topK(P, 5).map((x) =>
      `<span class="chip" style="opacity:${(0.35 + 0.65 * P[x]).toFixed(2)}">${
        sym(x)
      } ${(P[x] * 100).toFixed(0)}%</span>`
    ).join("");
    const head = `<span class="mono lbl">${esc(label)}</span>${
      c.classed ? '<span class="tag">类成员</span>' : ""
    }
      <span class="n">N=${fmt(c.total)} · ${kids[k].length} 子</span>${chips}`;
    const full = esc(renderCtx(img, c.slots));
    if (!kids[k].length || depth >= maxDepth || budget <= 0) {
      return `<div class="leaf" title="${full}">${head}</div>`;
    }
    const shown = kids[k].slice(0, maxKids);
    const more = kids[k].length - shown.length;
    return `<details${
      depth === 0 ? " open" : ""
    }><summary title="${full}">${head}</summary><div class="kids">${
      shown.map((ch) => node(ch, depth + 1)).join("")
    }${
      more > 0 ? `<div class="more">…还有 ${fmt(more)} 个子上下文</div>` : ""
    }</div></details>`;
  };
  return `<section><h2>3. 上下文树（存储结构）</h2>
    <p class="hint">每个上下文 = 父上下文左边再加一个槽位。槽位可以是一个字节、一个词首片段（seq，只在词边界匹配），或者一个类（alt）。
    点击展开（每层按证据量排序，最多 ${maxKids} 个，最深 ${maxDepth} 层）。芯片是这个上下文平滑后对下一个字节的预测。悬停查看完整上下文（左边更早）。</p>
    <div id="tree">${node(comp.rootIdx, 0)}</div></section>`;
}

// ---------- 类与词表 ----------

function classesSection(): string {
  const use = new Map<string, { ctx: number; n: number; example: string }>();
  for (const c of comp.ctx) {
    const f = c.slots[0];
    if (f?.tag === "ref" && img.nodes.get(f.id)?.kind === "alt") {
      const u = use.get(f.id) ??
        { ctx: 0, n: 0, example: renderCtx(img, c.slots) };
      u.ctx++;
      u.n += c.total;
      use.set(f.id, u);
    }
  }
  const rows = alts.map(([id]) => ({
    id,
    size: members(img, ref(id)).length,
    ...(use.get(id) ?? { ctx: 0, n: 0, example: "" }),
  }))
    .sort((x, y) => y.n - x.n).slice(0, 150);
  return `<section><h2>4. 类（Fold 学出的可替换类）</h2>
    <p class="hint">Fold 把"后面跟的内容相似"的词首片段合成一个类，作为成员上下文的先验，让成员之间共享统计。
    共 ${alts.length} 个，按证据量排序，显示前 ${rows.length} 个。</p>
    <table><tr><th>类</th><th>成员数</th><th>用在几个上下文</th><th>证据</th><th>例子</th></tr>${
    rows.map((r) =>
      `<tr><td class="mono">${
        esc(renderRef(img, ref(r.id)))
      }</td><td>${r.size}</td><td>${r.ctx}</td><td>${
        fmt(r.n)
      }</td><td class="mono small">${esc(r.example)}</td></tr>`
    ).join("")
  }</table></section>`;
}

function vocabSection(): string {
  const top = seqs.map(([id]) => ({ id, f: img.seqFreq.get(id) ?? 0 })).sort((
    x,
    y,
  ) => y.f - x.f).slice(0, 400);
  const maxF = top[0]?.f ?? 1;
  return `<section><h2>5. 词表（Chunk 学出的词首片段）</h2>
    <p class="hint">共 ${seqs.length} 个，显示最常见的 ${top.length} 个。字号 ∝ log(频数)，悬停查看次数。</p>
    <div class="cloud">${
    top.map((s) =>
      `<span style="font-size:${
        (10 + 16 * Math.log(1 + s.f) / Math.log(1 + maxF)).toFixed(1)
      }px" title="${fmt(s.f)} 次">${esc(renderRef(img, ref(s.id)))}</span>`
    ).join(" ")
  }</div></section>`;
}

// ---------- 校准表 ----------

function heatmap(
  title: string,
  rowLabel: string,
  colLabel: string,
  cell: (r: number, c: number) => [number, string] | null,
  max: number,
): string {
  let h =
    `<div class="hm"><h3>${title}</h3><div class="hmgrid"><div class="axis">${rowLabel}↓ ${colLabel}→</div>`;
  for (let c = 0; c < 16; c++) h += `<div class="axis">${c}</div>`;
  for (let r = 0; r < 16; r++) {
    h += `<div class="axis">${r}</div>`;
    for (let c = 0; c < 16; c++) {
      const v = cell(r, c);
      h += v
        ? `<div class="cell" style="background:${heat(v[0], max)}" title="${
          esc(v[1])
        }"></div>`
        : `<div class="cell empty"></div>`;
    }
  }
  return h + "</div></div>";
}

function tablesSection(): string {
  const t = img.tables;
  const mix = heatmap(
    "叶子上下文的实际表现（平均码长）",
    "深度",
    "log₂(N+1)",
    (r, c) => {
      const b = r * 16 + c, n = t.mixN[b];
      if (!n) return null;
      const avg = t.mixBits[b] / 1024 / n;
      return [
        avg,
        `深度 ${r}，证据约 2^${c}：平均 ${avg.toFixed(2)} bits（${
          fmt(n)
        } 次）→ 混合权重 2^(−${img.params.kappa}×${avg.toFixed(2)})`,
      ];
    },
    8,
  );
  const apm = heatmap(
    "match 建议的实际命中率",
    "2·log₂(匹配长度)",
    "−2·log₂(上下文模型概率)",
    (r, c) => {
      const b = r * 16 + c, n = t.apmTot[b];
      if (!n) return null;
      const hit = t.apmHit[b] / n;
      return [
        1 - hit,
        `匹配长度约 ${Math.pow(2, r / 2).toFixed(0)}，上下文模型给出约 ${
          (Math.pow(2, -c / 2) * 100).toFixed(1)
        }%：实际命中 ${(hit * 100).toFixed(1)}%（${fmt(n)} 次）`,
      ];
    },
    1,
  );
  return `<section><h2>6. 校准表</h2>
    <p class="hint">两张表都是整数计数，按文档预序统计：每个位置只用它之前的数据来"考"模型。合并两个模型时，表逐格相加。
    绿色 = 好（码长短 / 命中率高），灰色 = 没有数据。悬停查看数值。</p>
    <div class="cols">${mix}${apm}</div></section>`;
}

// ---------- 代码（计算图） ----------

const short = (id: Id) => `#${id.slice(0, 6)}`;
const refText = (r: Ref) =>
  r.tag === "term" ? `'${renderBytes([r.sym])}'` : short(r.id);
const topCounts = (c: Int32Array, k: number) =>
  [...c.keys()].filter((x) => c[x] > 0).sort((x, y) => c[y] - c[x]).slice(0, k)
    .map((x) => `'${renderBytes([x])}':${c[x]}`).join(" ");

function graphStats(): string {
  let ctxEdges = 0, altEdges = 0;
  const fanIn = new Map<Id, number>();
  for (const [, n] of img.nodes) {
    const refs = n.kind === "ctx" ? n.slots : n.kind === "alt" ? n.items : [];
    for (const r of refs) {
      if (r.tag !== "ref") continue;
      if (n.kind === "ctx") ctxEdges++;
      else altEdges++;
      fanIn.set(r.id, (fanIn.get(r.id) ?? 0) + 1);
    }
  }
  const terms = [...img.nodes.values()].reduce(
    (s, n) =>
      s +
      (n.kind === "ctx" ? n.slots : n.kind === "alt" ? n.items : []).filter((
        r,
      ) => r.tag === "term").length,
    0,
  );
  const top = [...fanIn].sort((x, y) => y[1] - x[1]).slice(0, 16)
    .map(([id, n]) =>
      `<span class="chip">${esc(renderRef(img, ref(id)))} ×${fmt(n)}</span>`
    ).join(" ");
  return `<table class="narrow">
    <tr><th>节点</th><td>${fmt(comp.ctx.length)} ctx · ${
    fmt(alts.length)
  } alt · ${fmt(seqs.length)} seq</td></tr>
    <tr><th>引用边（存储里的 DAG 边）</th><td>ctx→seq/alt ${
    fmt(ctxEdges)
  } 条 · alt→seq ${fmt(altEdges)} 条 · 另有 ${
    fmt(terms)
  } 个字节终结符（不是节点）</td></tr>
    <tr><th>DAG 最大深度</th><td>3（ctx → alt → seq）。ctx 之间没有存储边："父上下文 / 先验"是编译时从内容推导出来的</td></tr>
    <tr><th>被引用最多的节点</th><td>${top}</td></tr>
  </table>`;
}

function explainOne(input: string): string {
  const text = encoder.encode(input);
  const m = newMatched(comp);
  traverse(comp, text, text.length, m);
  const matched = [...Array(m.n).keys()].map((k) => m.idx[k]).sort((x, y) =>
    x - y
  );
  const isPrior = new Set(matched.map((ci) => comp.ctx[ci].parentIdx));
  const leaves = matched.filter((ci) => !isPrior.has(ci));
  let W = 0;
  for (const ci of leaves) W += pr.weight[ci];

  // 收集被引用的 alt / seq
  const altIds: Id[] = [], seqIds: Id[] = [];
  const visit = (r: Ref) => {
    if (r.tag !== "ref") return;
    const n = img.nodes.get(r.id)!;
    const list = n.kind === "alt" ? altIds : seqIds;
    if (list.includes(r.id)) return;
    if (n.kind === "alt") n.items.forEach(visit);
    list.push(r.id);
  };
  matched.forEach((ci) => comp.ctx[ci].slots.forEach(visit));

  // ---- 清单 ----
  const rows: string[] = [];
  for (const id of seqIds) {
    const n = img.nodes.get(id)!;
    if (n.kind === "seq") {
      rows.push(
        `<tr><td>${short(id)}</td><td>seq</td><td>"${
          esc(renderBytes(n.bytes))
        }"</td><td></td><td>freq=${
          fmt(img.seqFreq.get(id) ?? 0)
        }</td><td></td></tr>`,
      );
    }
  }
  for (const id of altIds) {
    const n = img.nodes.get(id)!;
    if (n.kind === "alt") {
      rows.push(
        `<tr><td>${short(id)}</td><td>alt</td><td>{ ${
          esc(n.items.map(refText).join(" | "))
        } }  = ${
          esc(renderRef(img, ref(id)))
        }</td><td></td><td></td><td></td></tr>`,
      );
    }
  }
  for (const ci of matched) {
    const c = comp.ctx[ci];
    const role = leaves.includes(ci)
      ? `<b>叶子 ${((pr.weight[ci] / W) * 100).toFixed(0)}%</b>`
      : "先验";
    rows.push(
      `<tr><td>${short(c.id)}</td><td>ctx</td><td>[${
        esc(c.slots.map(refText).join(", "))
      }]${c.classed ? ' <span class="tag">类成员</span>' : ""}</td><td>${
        c.parentIdx >= 0 ? short(comp.ctx[c.parentIdx].id) : "-"
      }</td><td>N=${fmt(c.total)}  ${
        esc(topCounts(c.counts, 4))
      }</td><td>${role}</td></tr>`,
    );
  }

  // ---- 图 ----
  const CH = 7.3, RH = 30, BH = 22;
  const ctxLabel = (ci: number) => renderCtx(img, comp.ctx[ci].slots);
  const wOf = (s: string) => Math.min(420, s.length * CH + 18);
  const ctxW = Math.max(
    ...matched.map((ci) =>
      wOf(ctxLabel(ci) + "  N=" + fmt(comp.ctx[ci].total))
    ),
  );
  const altW = Math.max(
    60,
    ...altIds.map((id) => wOf(renderRef(img, ref(id)))),
  );
  const seqW = Math.max(
    60,
    ...seqIds.map((id) => wOf(`"${renderRef(img, ref(id))}"`)),
  );
  const left = 30 + 14 * Math.min(12, matched.length);
  const xCtx = left, xAlt = xCtx + ctxW + 70, xSeq = xAlt + altW + 70;
  const pos = new Map<string, { x: number; y: number; w: number }>();
  matched.forEach((ci, i) =>
    pos.set(comp.ctx[ci].id, { x: xCtx, y: 24 + i * RH, w: ctxW })
  );
  altIds.forEach((id, i) => pos.set(id, { x: xAlt, y: 24 + i * RH, w: altW }));
  seqIds.forEach((id, i) => pos.set(id, { x: xSeq, y: 24 + i * RH, w: seqW }));
  const H = 34 + Math.max(matched.length, altIds.length, seqIds.length, 1) * RH;
  const Wsvg = xSeq + seqW + 10;

  const edges: string[] = [];
  const link = (from: string, to: string) => {
    const p = pos.get(from)!, q = pos.get(to)!;
    const x1 = p.x + p.w,
      y1 = p.y + BH / 2,
      x2 = q.x,
      y2 = q.y + BH / 2,
      mx = (x1 + x2) / 2;
    edges.push(
      `<path d="M${x1} ${y1} C${mx} ${y1} ${mx} ${y2} ${x2} ${y2}" class="e"/>`,
    );
  };
  for (const ci of matched) {
    for (const r of comp.ctx[ci].slots) {
      if (r.tag === "ref") link(comp.ctx[ci].id, r.id);
    }
  }
  for (const id of altIds) {
    const n = img.nodes.get(id)!;
    if (n.kind === "alt") {
      for (const r of n.items) if (r.tag === "ref") link(id, r.id);
    }
  }
  // 先验链：编译时推导的视图，画在左侧的虚线弧
  matched.forEach((ci, i) => {
    const par = comp.ctx[ci].parentIdx;
    if (par < 0) return;
    const p = pos.get(comp.ctx[ci].id)!, q = pos.get(comp.ctx[par].id);
    if (!q) return;
    const bulge = 16 + 12 * (i % Math.max(1, Math.min(12, matched.length)));
    const y1 = p.y + BH / 2, y2 = q.y + BH / 2;
    edges.push(
      `<path d="M${xCtx} ${y1} C${xCtx - bulge} ${y1} ${
        xCtx - bulge
      } ${y2} ${xCtx} ${y2}" class="prior"/>`,
    );
  });

  const box = (
    id: string,
    label: string,
    kind: string,
    tip: string,
    strong = false,
  ) => {
    const p = pos.get(id)!;
    return `<g><title>${
      esc(tip)
    }</title><rect x="${p.x}" y="${p.y}" width="${p.w}" height="${BH}" rx="5" class="n-${kind}${
      strong ? " strong" : ""
    }"/><text x="${p.x + 8}" y="${p.y + 15}">${esc(label)}</text></g>`;
  };
  const boxes = [
    ...matched.map((ci) => {
      const c = comp.ctx[ci];
      const leaf = leaves.includes(ci);
      return box(
        c.id,
        `${ctxLabel(ci)}  N=${fmt(c.total)}`,
        "ctx",
        `${short(c.id)} ctx  N=${c.total}  ${topCounts(c.counts, 5)}${
          leaf
            ? `\n叶子，混合权重 ${((pr.weight[ci] / W) * 100).toFixed(0)}%`
            : ""
        }`,
        leaf,
      );
    }),
    ...altIds.map((id) =>
      box(id, renderRef(img, ref(id)), "alt", `${short(id)} alt`)
    ),
    ...seqIds.map((id) =>
      box(
        id,
        `"${renderRef(img, ref(id))}"`,
        "seq",
        `${short(id)} seq  freq=${img.seqFreq.get(id)}`,
      )
    ),
  ];
  const head = `<text x="${xCtx}" y="14" class="colh">ctx（规则）</text>` +
    (altIds.length
      ? `<text x="${xAlt}" y="14" class="colh">alt（类）</text>`
      : "") +
    (seqIds.length
      ? `<text x="${xSeq}" y="14" class="colh">seq（词首片段）</text>`
      : "");
  const svg =
    `<div class="chart"><svg class="dag" width="${Wsvg}" height="${H}">${head}${
      edges.join("")
    }${boxes.join("")}</svg></div>`;

  return `<h3>预测 <span class="mono">${
    esc(JSON.stringify(input))
  }</span> 之后的字节</h3>
    <p class="hint">激活 ${matched.length} 个上下文，其中 ${leaves.length} 个叶子参与混合；匹配测试 ${m.steps} 次。
    实线 = 存储里的引用（hash 里包含对方的 id）；左侧虚线 = 编译时推导的先验链（数据少时向它借统计）；粗框 = 参与混合的叶子。</p>
    ${svg}
    <details><summary>节点清单（${rows.length}）</summary><table class="mono small">
      <tr><th>id</th><th>种类</th><th>内容</th><th>先验</th><th>计数</th><th>角色</th></tr>${
    rows.join("")
  }</table></details>`;
}

// 观察等价：计数完全相同的上下文，在训练数据上表现一致，是同一个"事实"的多种写法
function redundancy(): string {
  const groups = new Map<string, number[]>();
  comp.ctx.forEach((c, k) => {
    if (c.depth === 0 || c.total < 5) return;
    const key = c.counts.join(",");
    const g = groups.get(key);
    if (g) g.push(k);
    else groups.set(key, [k]);
  });
  const dup = [...groups.values()].filter((g) => g.length > 1);
  const extra = dup.reduce((s, g) => s + g.length - 1, 0);
  const top = dup.sort((x, y) =>
    y.length * comp.ctx[y[0]].total - x.length * comp.ctx[x[0]].total
  ).slice(0, 12);
  return `<p class="hint">有 ${
    fmt(dup.length)
  } 组上下文的计数<b>逐字节完全相同</b>（只统计 N ≥ 5）。它们匹配的是同一组位置，只是由 Fork 沿不同路径（逐字节 / 词首片段）长出来。
    每组只留一个，可以省掉 <b>${fmt(extra)}</b> 个上下文（${
    ((extra / comp.ctx.length) * 100).toFixed(1)
  }%）。这就是 topic.md 里说的"观察等价"，是一个现成的压缩机会。按规模排序的前 12 组：</p>
    <table class="mono small"><tr><th>N</th><th>等价的写法</th><th>后面跟什么</th></tr>${
    top.map((g) =>
      `<tr><td>${fmt(comp.ctx[g[0]].total)}</td><td>${
        g.map((k) => esc(renderCtx(img, comp.ctx[k].slots))).join("<br>")
      }</td><td>${esc(topCounts(comp.ctx[g[0]].counts, 4))}</td></tr>`
    ).join("")
  }</table>`;
}

function codeSection(): string {
  const inputs =
    (a.str("explain") ??
      "and to the king|||JULIET:\nO Romeo, Rom|||I pray thee, my lo").split(
        "|||",
      );
  return `<section><h2>7. 代码（模型本身的计算图）</h2>
    <p class="hint">学出来的"代码"是一大组<b>带计数的模式规则</b>：解释器是固定的（匹配、平滑、混合、match），学到的是规则本身。
    目前还没有过程、循环或变量（topic.md 里的 L0 层），更像一个巨大的 <span class="mono">match</span> 语句或一组 Datalog 事实。</p>
    <h3>指令集</h3>
    <table class="narrow">
      <tr><th>节点</th><th>含义</th><th>类比</th></tr>
      <tr><td class="mono">seq "king"</td><td>词首片段，只在词边界匹配</td><td>常量字符串</td></tr>
      <tr><td class="mono">alt { a | b | … }</td><td>可替换类，匹配任意一个成员</td><td>正则里的 (a|b)</td></tr>
      <tr><td class="mono">ctx [s1, s2, …] + 计数</td><td>如果前文以 s1 s2 … 结尾，下一个字节的计数是这些</td><td><span class="mono">if 模式 then 分布</span></td></tr>
    </table>
    <p class="hint">每个节点的 id = hash(内容)，内容里包含它引用的节点的 id，所以整个存储是一个 Merkle DAG。字节是终结符，不是节点。</p>
    <h3>全图结构</h3>${graphStats()}
    ${inputs.map(explainOne).join("\n")}
    <h3>冗余：观察等价的上下文</h3>${redundancy()}
  </section>`;
}

// ---------- 页面 ----------

const CSS = `
:root { --bg:#fbfbfa; --fg:#1d1d1f; --line:#e6e6e3; --card:#fff; --accent:#3a6df0; --heat-l:82%; --sub:#6b6b72;
  font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Helvetica Neue",sans-serif; }
@media (prefers-color-scheme: dark) { :root { --bg:#151517; --fg:#e8e8ea; --line:#2b2b30; --card:#1d1d20;
  --accent:#7ea2ff; --heat-l:30%; --sub:#9a9aa3; } }
body { margin:0 auto; max-width:1200px; padding:20px 28px 60px; background:var(--bg); color:var(--fg); font-size:14px; }
h1 { font-size:22px; font-weight:600; margin:0 0 4px; }
h2 { font-size:18px; font-weight:600; margin:36px 0 8px; padding-top:12px; border-top:1px solid var(--line); }
h3 { font-size:14px; font-weight:600; margin:16px 0 6px; }
nav a { color:var(--accent); margin-right:14px; text-decoration:none; }
.sub, .hint, .n { color:var(--sub); }
.hint { margin:4px 0 10px; line-height:1.6; }
.n { font-size:12px; }
.mono, .text, .cloud, .chip { font-family:ui-monospace,Menlo,monospace; }
.small { font-size:12px; }
#overview { display:flex; flex-wrap:wrap; gap:10px; margin:16px 0; }
.stat { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 14px; color:var(--sub); font-size:12px; }
.stat b { display:block; color:var(--fg); font-size:19px; font-weight:600; }
.hist text { font-size:9px; fill:var(--sub); }
.text { font-size:15px; line-height:1.75; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; word-break:break-all; }
.ch { border-radius:2px; padding:1px 0; cursor:help; }
.ch.m { text-decoration:underline; text-decoration-color:var(--accent); text-underline-offset:3px; }
.prompt { color:var(--sub); }
.chart { overflow-x:auto; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:6px; }
table { border-collapse:collapse; width:100%; }
th, td { text-align:left; padding:4px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--sub); font-weight:500; font-size:12px; }
.tag { font-size:11px; background:var(--accent); color:#fff; border-radius:4px; padding:0 4px; margin-left:4px; }
#tree summary, #tree .leaf { padding:3px 0; cursor:pointer; }
#tree .leaf { padding-left:14px; cursor:default; }
#tree summary:hover { background:color-mix(in srgb, var(--accent) 8%, transparent); }
#tree .kids { margin-left:14px; border-left:1px solid var(--line); padding-left:8px; }
.lbl { margin-right:6px; }
.chip { font-size:12px; background:color-mix(in srgb, var(--accent) 18%, transparent); border-radius:4px; padding:0 5px; margin-left:4px; }
.more { color:var(--sub); font-size:12px; padding:3px 0 3px 14px; }
.cloud { line-height:1.9; }
.cloud span { margin:0 5px; display:inline-block; }
.cols { display:flex; gap:28px; flex-wrap:wrap; }
.hm { flex:1 1 400px; }
.hmgrid { display:grid; grid-template-columns:90px repeat(16, 1fr); gap:2px; font-size:10px; }
.axis { color:var(--sub); text-align:center; align-self:center; }
.hmgrid > .axis:first-child { text-align:left; }
.cell { aspect-ratio:1; border-radius:2px; cursor:help; }
.cell.empty { background:var(--line); opacity:.4; }
table.narrow { width:auto; }
table.narrow th { white-space:nowrap; }
.dag text { font-family:ui-monospace,Menlo,monospace; font-size:12px; fill:var(--fg); }
.dag .colh { fill:var(--sub); font-family:inherit; }
.dag rect { fill:var(--card); stroke:var(--line); stroke-width:1; }
.dag rect.n-ctx { stroke:var(--sub); }
.dag rect.n-alt { fill:color-mix(in srgb, var(--accent) 16%, var(--card)); }
.dag rect.n-seq { fill:color-mix(in srgb, #2aa36b 16%, var(--card)); }
.dag rect.strong { stroke:var(--accent); stroke-width:2.5; }
.dag .e { fill:none; stroke:var(--sub); stroke-width:1; opacity:.7; }
.dag .prior { fill:none; stroke:var(--accent); stroke-width:1.2; stroke-dasharray:4 3; opacity:.8; }
details > summary { cursor:pointer; color:var(--sub); margin:6px 0; }
`;

const name = modelPath.split("/").pop()!;
const html = `<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>AFRN · ${
  esc(name)
}</title>
<style>${CSS}</style></head><body>
<h1>AFRN 模型可视化 · ${esc(name)}</h1>
<div class="sub">由 <span class="mono">visualize</span> 在 ${
  new Date().toISOString().slice(0, 16).replace("T", " ")
} 生成。
推理过程：上下文匹配（指针跳转）+ 计数平滑 + match 模型，没有矩阵乘法。</div>
<nav style="margin-top:10px"><a href="#s1">逐字节追踪</a><a href="#s2">生成</a><a href="#s3">上下文树</a><a href="#s4">类</a><a href="#s5">词表</a><a href="#s6">校准表</a><a href="#s7">代码（计算图）</a></nav>
${overview()}
${
  [
    traceSection(),
    generateSection(),
    treeSection(),
    classesSection(),
    vocabSection(),
    tablesSection(),
    codeSection(),
  ]
    .map((s, i) => s.replace("<section>", `<section id="s${i + 1}">`)).join(
      "\n",
    )
}
</body></html>`;

await Deno.writeTextFile(outPath, html);
console.log(`wrote ${outPath} (${(html.length / 1024).toFixed(0)} KB)`);
