// 拓扑图的浏览器端：只负责绘制与交互，所有布局与数据都由 graph.ts 预先算好。
"use strict";
(() => {
  const G = JSON.parse(document.getElementById("graph-data").textContent);
  const C = G.ctx, S = G.sym, N = C.x.length, NS = S.x.length, R = G.R;

  // ---------- 索引 ----------
  const kids = Array.from({ length: N }, () => []);
  for (let k = 0; k < N; k++) if (C.tail[k] >= 0) kids[C.tail[k]].push(k);
  const users = Array.from({ length: NS }, () => []);
  for (let k = 0; k < N; k++) if (C.ref[k] >= 0) users[C.ref[k]].push(k);
  const inAlts = Array.from({ length: NS }, () => []);
  for (let s = 0; s < NS; s++) {
    for (const t of S.items[s] || []) inAlts[t].push(s);
  }
  const byN = [...Array(N).keys()].sort((a, b) => C.n[b] - C.n[a]);
  const rad = new Float32Array(N);
  for (let k = 0; k < N; k++) rad[k] = 0.35 * Math.log2(C.n[k] + 1) + 0.3;
  const symRad = (s) => (S.kind[s] === 1 ? 3 : 2);
  const fullCache = new Array(N);
  const full = (k) => {
    if (fullCache[k] !== undefined) return fullCache[k];
    const parts = [];
    for (let j = k; j >= 0 && C.d[j] > 0; j = C.tail[j]) {
      parts.push("[" + C.lab[j] + "]");
    }
    return (fullCache[k] = parts.join("") || "ε（根）");
  };
  const esc = (s) =>
    String(s).replace(
      /[&<>"]/g,
      (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]),
    );
  const fmt = (x) => x.toLocaleString("en");
  const KINDS = ["字节", "seq", "类", "根"];

  // ---------- 颜色 ----------
  let COL = {};
  const readColors = () => {
    const css = getComputedStyle(document.documentElement);
    const v = (n) => css.getPropertyValue(n).trim();
    COL = {
      bg: v("--bg"),
      fg: v("--fg"),
      sub: v("--sub"),
      line: v("--line"),
      edge: v("--edge"),
      hl: v("--hl"),
      kind: [v("--k-byte"), v("--k-seq"), v("--k-alt"), v("--k-root")],
    };
  };
  readColors();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    readColors();
    schedule();
  });

  // ---------- 视图 ----------
  const cv = document.getElementById("cv"),
    g = cv.getContext("2d"),
    tip = document.getElementById("tip");
  let W = 0, H = 0, dpr = 1, scale = 1, ox = 0, oy = 0, fitted = false;
  const show = {
    tree: true,
    prior: true,
    refs: false,
    syms: true,
    labels: true,
  };
  let hl = null, hover = null;

  const fit = () => {
    const e = (G.maxD + 3.4) * R;
    scale = Math.min(W, H) / (2 * e);
    ox = W / 2;
    oy = H / 2;
  };
  const resize = () => {
    dpr = devicePixelRatio || 1;
    W = cv.clientWidth;
    H = cv.clientHeight;
    cv.width = W * dpr;
    cv.height = H * dpr;
    if (!fitted) {
      fit();
      fitted = true;
    }
    schedule();
  };
  let raf = 0;
  const schedule = () => {
    if (!raf) {
      raf = requestAnimationFrame(() => {
        raf = 0;
        draw();
      });
    }
  };
  const pos = (
    id,
  ) => (id >= 0 ? [C.x[id], C.y[id]] : [S.x[-1 - id], S.y[-1 - id]]);

  function draw() {
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = COL.bg;
    g.fillRect(0, 0, W, H);
    g.setTransform(dpr * scale, 0, 0, dpr * scale, dpr * ox, dpr * oy);
    const px = 1 / scale, dim = hl ? 0.3 : 1;

    g.strokeStyle = COL.line;
    g.lineWidth = px;
    for (let d = 1; d <= G.maxD; d++) {
      g.beginPath();
      g.arc(0, 0, d * R, 0, 2 * Math.PI);
      g.stroke();
    }

    if (show.tree) {
      g.globalAlpha = 0.2 * dim;
      g.strokeStyle = COL.edge;
      g.lineWidth = 0.7 * px;
      g.beginPath();
      for (let k = 0; k < N; k++) {
        const t = C.tail[k];
        if (t < 0) continue;
        g.moveTo(C.x[t], C.y[t]);
        g.lineTo(C.x[k], C.y[k]);
      }
      g.stroke();
    }
    if (show.prior) {
      g.globalAlpha = 0.75 * dim;
      g.strokeStyle = COL.kind[2];
      g.lineWidth = px;
      g.setLineDash([4 * px, 3 * px]);
      g.beginPath();
      for (let k = 0; k < N; k++) {
        const p = C.prior[k];
        if (p < 0) continue;
        const mx = (C.x[k] + C.x[p]) / 2, my = (C.y[k] + C.y[p]) / 2;
        g.moveTo(C.x[k], C.y[k]);
        g.quadraticCurveTo(mx * 1.06, my * 1.06, C.x[p], C.y[p]);
      }
      g.stroke();
      g.setLineDash([]);
    }
    if (show.refs && show.syms) {
      g.globalAlpha = 0.08 * dim;
      g.strokeStyle = COL.kind[1];
      g.lineWidth = 0.7 * px;
      g.beginPath();
      for (let k = 0; k < N; k++) {
        const s = C.ref[k];
        if (s < 0) continue;
        g.moveTo(C.x[k], C.y[k]);
        g.lineTo(S.x[s], S.y[s]);
      }
      for (let s = 0; s < NS; s++) {
        for (const t of S.items[s] || []) {
          g.moveTo(S.x[s], S.y[s]);
          g.lineTo(S.x[t], S.y[t]);
        }
      }
      g.stroke();
    }

    g.globalAlpha = dim;
    const minR = 0.9 * px;
    for (let kind = 0; kind < 4; kind++) {
      g.fillStyle = COL.kind[kind];
      g.beginPath();
      for (let k = 0; k < N; k++) {
        if (C.kind[k] !== kind) continue;
        const r = Math.max(rad[k], minR);
        g.rect(C.x[k] - r, C.y[k] - r, 2 * r, 2 * r);
      }
      g.fill();
    }
    if (show.syms) {
      for (const kind of [0, 1]) {
        g.fillStyle = COL.kind[kind === 1 ? 2 : 1];
        g.beginPath();
        for (let s = 0; s < NS; s++) {
          if (S.kind[s] !== kind) continue;
          const r = Math.max(symRad(s), 1.2 * px);
          g.moveTo(S.x[s] + r, S.y[s]);
          g.arc(S.x[s], S.y[s], r, 0, 2 * Math.PI);
        }
        g.fill();
      }
    }
    g.globalAlpha = 1;

    if (hl) {
      g.lineWidth = 2 * px;
      for (const [x1, y1, x2, y2, c, dash] of hl.edges) {
        g.strokeStyle = c;
        g.setLineDash(dash ? [5 * px, 3 * px] : []);
        g.beginPath();
        g.moveTo(x1, y1);
        g.lineTo(x2, y2);
        g.stroke();
      }
      g.setLineDash([]);
      for (const id of hl.nodes) {
        const [x, y] = pos(id);
        const r = Math.max(id >= 0 ? rad[id] : symRad(-1 - id), 2.5 * px) *
          (hl.strong.has(id) ? 1.6 : 1);
        g.fillStyle = hl.strong.has(id)
          ? COL.hl
          : id >= 0
          ? COL.kind[C.kind[id]]
          : COL.kind[S.kind[-1 - id] ? 2 : 1];
        g.beginPath();
        g.arc(x, y, r, 0, 2 * Math.PI);
        g.fill();
        g.strokeStyle = COL.fg;
        g.lineWidth = px;
        g.stroke();
      }
    }
    if (hover !== null) {
      const [x, y] = pos(hover);
      g.strokeStyle = COL.hl;
      g.lineWidth = 2 * px;
      g.beginPath();
      g.arc(
        x,
        y,
        Math.max(hover >= 0 ? rad[hover] : symRad(-1 - hover), 3 * px) + 2 * px,
        0,
        2 * Math.PI,
      );
      g.stroke();
    }

    if (show.labels) drawLabels();
  }

  function drawLabels() {
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.font = "11px ui-monospace, Menlo, monospace";
    g.textBaseline = "middle";
    let count = 0;
    const label = (x, y, r, text, strong) => {
      const sx = x * scale + ox, sy = y * scale + oy;
      if (sx < -50 || sx > W + 50 || sy < -20 || sy > H + 20) return false;
      const t = text.length > 28 ? text.slice(0, 27) + "…" : text;
      g.fillStyle = strong ? COL.fg : COL.sub;
      g.fillText(t, sx + r * scale + 3, sy);
      return true;
    };
    if (hl) {
      for (const id of hl.nodes) {
        const [x, y] = pos(id);
        label(
          x,
          y,
          id >= 0 ? rad[id] : symRad(-1 - id),
          id >= 0 ? C.lab[id] : S.lab[-1 - id],
          true,
        );
      }
    }
    for (const k of byN) {
      if (count > 700) break;
      if (rad[k] * scale < 2.2) continue;
      if (label(C.x[k], C.y[k], rad[k], C.lab[k], false)) count++;
    }
    if (show.syms && scale > 0.9) {
      for (let s = 0; s < NS && count < 1400; s++) {
        if (label(S.x[s], S.y[s], symRad(s), S.lab[s], false)) count++;
      }
    }
  }

  // ---------- 命中测试 ----------
  const CELL = 8, grid = new Map();
  const put = (id, x, y) => {
    const key = Math.floor(x / CELL) + "," + Math.floor(y / CELL);
    let l = grid.get(key);
    if (!l) grid.set(key, l = []);
    l.push(id);
  };
  for (let k = 0; k < N; k++) put(k, C.x[k], C.y[k]);
  for (let s = 0; s < NS; s++) put(-1 - s, S.x[s], S.y[s]);
  function pick(sx, sy) {
    const wx = (sx - ox) / scale, wy = (sy - oy) / scale, r = 7 / scale;
    let best = null, bd = Infinity;
    for (
      let i = Math.floor((wx - r) / CELL); i <= Math.floor((wx + r) / CELL); i++
    ) {
      for (
        let j = Math.floor((wy - r) / CELL);
        j <= Math.floor((wy + r) / CELL);
        j++
      ) {
        for (const id of grid.get(i + "," + j) || []) {
          if (id < 0 && !show.syms) continue;
          const [x, y] = pos(id);
          const d = Math.hypot(x - wx, y - wy) -
            (id >= 0 ? rad[id] : symRad(-1 - id));
          if (d < r && d < bd) {
            bd = d;
            best = id;
          }
        }
      }
    }
    return best;
  }

  // ---------- 高亮与信息面板 ----------
  const info = document.getElementById("info");
  const ctxLink = (k) => `<a data-id="${k}">${esc(full(k))}</a>`;
  const symLink = (s) => `<a data-id="${-1 - s}">${esc(S.lab[s])}</a>`;
  const edge = (a, b, c, dash) => {
    const [x1, y1] = pos(a), [x2, y2] = pos(b);
    return [x1, y1, x2, y2, c, dash];
  };

  function defaultInfo() {
    const depth = new Array(G.maxD + 1).fill(0);
    for (let k = 0; k < N; k++) depth[C.d[k]]++;
    const prior = C.prior.filter((p) => p >= 0).length;
    info.innerHTML = `<h3>整张图</h3>
      <p class="hint">这是 <span class="mono">${
      esc(G.model)
    }</span> 里的全部节点：${fmt(N)} 个 ctx，
      ${fmt(S.kind.filter((k) => k === 1).length)} 个 alt，${
      fmt(S.kind.filter((k) => k === 0).length)
    } 个 seq。</p>
      <p class="hint">每个 ctx 恰好有一个 <b>tail</b>（去掉最左边的槽位），所以上下文构成一棵树：根在圆心，第 d 圈是有 d 个槽位的上下文，
      每棵子树分到的角度正比于它的叶子数。蓝色虚线是 Fold 产生的<b>类先验链</b>（${
      fmt(prior)
    } 条）。
      外面两圈是 alt（内）和 seq（外）节点，放在引用它们的上下文的方向上。</p>
      <p class="hint">存储里真正的边只有"ctx → 它的槽位（seq/alt）"和"alt → 成员"。树边和先验链都是编译时从内容推导出的视图。</p>
      <h3>每圈的上下文数</h3><ul>${
      depth.map((c, d) => `<li>第 ${d} 圈：${fmt(c)}</li>`).join("")
    }</ul>
      <p class="hint">模型文件 ${
      (G.bytes / 1024).toFixed(0)
    } KB · d=${G.params.discount} θ=${G.params.theta} κ=${G.params.kappa}</p>`;
  }

  function selectCtx(k) {
    const nodes = new Set([k]), strong = new Set([k]), edges = [];
    for (let j = k; C.tail[j] >= 0; j = C.tail[j]) {
      nodes.add(C.tail[j]);
      edges.push(edge(C.tail[j], j, COL.hl, false));
    }
    for (let j = k; C.prior[j] >= 0; j = C.prior[j]) {
      nodes.add(C.prior[j]);
      edges.push(edge(j, C.prior[j], COL.kind[2], true));
    }
    const ks = kids[k].slice().sort((a, b) => C.n[b] - C.n[a]);
    for (const c of ks) {
      nodes.add(c);
      edges.push(edge(k, c, COL.edge, false));
    }
    const s = C.ref[k];
    if (s >= 0) {
      nodes.add(-1 - s);
      edges.push(edge(k, -1 - s, COL.kind[1], false));
      for (const t of S.items[s] || []) {
        nodes.add(-1 - t);
        edges.push(edge(-1 - s, -1 - t, COL.kind[1], false));
      }
    }
    const classedBy = [];
    for (let j = 0; j < N; j++) if (C.prior[j] === k) classedBy.push(j);
    hl = { nodes, strong, edges };
    info.innerHTML = `<h3>ctx #${k}${
      C.prior[k] >= 0 ? '<span class="tag">类成员</span>' : ""
    }</h3>
      <div class="mono" style="font-size:14px;margin-bottom:8px">${
      esc(full(k))
    }</div>
      <div class="kv">新槽位：<b>${esc(C.lab[k])}</b>（${KINDS[C.kind[k]]}）${
      s >= 0 ? " → " + symLink(s) : ""
    }</div>
      <div class="kv">槽位数（圈）：<b>${C.d[k]}</b> · 证据 N：<b>${
      fmt(C.n[k])
    }</b></div>
      <div class="kv">下一个字节：<b class="mono">${esc(C.top[k])}</b></div>
      <div class="kv">tail：${C.tail[k] >= 0 ? ctxLink(C.tail[k]) : "—"}</div>
      <div class="kv">先验：${
      C.prior[k] >= 0 ? ctxLink(C.prior[k]) : "= tail"
    }</div>
      ${
      classedBy.length
        ? `<h3>把它当先验的类成员（${classedBy.length}）</h3><ul>${
          classedBy.slice(0, 20).map((j) => `<li>${ctxLink(j)}</li>`).join("")
        }</ul>`
        : ""
    }
      <h3>子上下文（${fmt(ks.length)}）</h3>
      <ul>${
      ks.slice(0, 40).map((c) =>
        `<li>${ctxLink(c)} <span class="kv">N=${fmt(C.n[c])}</span></li>`
      ).join("")
    }${ks.length > 40 ? `<li>…还有 ${fmt(ks.length - 40)} 个</li>` : ""}</ul>
      <p class="hint">橙色 = 到根的路径（每一步去掉最左槽位）· 蓝色虚线 = 先验链 · 绿色 = 引用的 seq/alt</p>`;
  }

  function selectSym(s) {
    const id = -1 - s,
      nodes = new Set([id]),
      strong = new Set([id]),
      edges = [];
    for (const k of users[s]) {
      nodes.add(k);
      edges.push(edge(k, id, COL.kind[1], false));
    }
    for (const t of S.items[s] || []) {
      nodes.add(-1 - t);
      edges.push(edge(id, -1 - t, COL.kind[1], false));
    }
    for (const a of inAlts[s]) {
      nodes.add(-1 - a);
      edges.push(edge(-1 - a, id, COL.kind[2], false));
    }
    hl = { nodes, strong, edges };
    const us = users[s].slice().sort((a, b) => C.n[b] - C.n[a]);
    info.innerHTML = `<h3>${
      S.kind[s] === 1 ? "alt（类）" : "seq（词首片段）"
    }</h3>
      <div class="mono" style="font-size:15px;margin-bottom:8px">${
      esc(S.lab[s])
    }</div>
      ${
      S.kind[s] === 0
        ? `<div class="kv">词首频数：<b>${fmt(S.freq[s])}</b></div>`
        : ""
    }
      ${
      S.items[s]
        ? `<h3>成员（${S.items[s].length}）</h3><ul>${
          S.items[s].map((t) => `<li>${symLink(t)}</li>`).join("")
        }</ul>`
        : ""
    }
      ${
      inAlts[s].length
        ? `<h3>属于这些类</h3><ul>${
          inAlts[s].map((a) => `<li>${symLink(a)}</li>`).join("")
        }</ul>`
        : ""
    }
      <h3>以它为新槽位的上下文（${fmt(us.length)}）</h3>
      <ul>${
      us.slice(0, 40).map((k) =>
        `<li>${ctxLink(k)} <span class="kv">N=${fmt(C.n[k])}</span></li>`
      ).join("")
    }</ul>`;
  }

  function select(id, center) {
    if (id === null || id === undefined) {
      hl = null;
      defaultInfo();
    } else if (id >= 0) selectCtx(id);
    else selectSym(-1 - id);
    if (center && id !== null) {
      const [x, y] = pos(id);
      scale = Math.max(scale, 1.5);
      ox = W / 2 - x * scale;
      oy = H / 2 - y * scale;
    }
    schedule();
  }

  function showExample(i) {
    const ex = G.examples[i];
    if (!ex) return select(null);
    const nodes = new Set(ex.matched), strong = new Set(ex.leaves), edges = [];
    for (const k of ex.matched) {
      if (C.tail[k] >= 0 && nodes.has(C.tail[k])) {
        edges.push(edge(C.tail[k], k, COL.hl, false));
      }
      if (C.prior[k] >= 0 && nodes.has(C.prior[k])) {
        edges.push(edge(k, C.prior[k], COL.kind[2], true));
      }
    }
    hl = { nodes, strong, edges };
    info.innerHTML = `<h3>一次预测</h3>
      <p class="hint">预测 <span class="mono">${
      esc(JSON.stringify(ex.text))
    }</span> 之后的字节：沿树向左匹配，
      激活 ${ex.matched.length} 个上下文，做了 ${ex.steps} 次槽位匹配测试。橙色大点 = 参与混合的叶子。</p>
      <ul>${
      ex.matched.map((k) =>
        `<li>${ex.leaves.includes(k) ? "<b>叶子</b> " : ""}${
          ctxLink(k)
        } <span class="kv">N=${fmt(C.n[k])} → ${esc(C.top[k])}</span></li>`
      ).join("")
    }</ul>`;
    schedule();
  }

  function search(q) {
    q = q.trim();
    if (!q) return select(null);
    const syms = [], ctxs = [];
    for (let s = 0; s < NS; s++) if (S.lab[s].includes(q)) syms.push(s);
    syms.sort((a, b) =>
      (S.lab[a] === q ? -1 : 0) - (S.lab[b] === q ? -1 : 0) ||
      users[b].length - users[a].length
    );
    for (const k of byN) {
      if (ctxs.length >= 60) break;
      if (full(k).replace(/[\[\]]/g, "").includes(q)) ctxs.push(k);
    }
    const nodes = new Set([...ctxs, ...syms.slice(0, 60).map((s) => -1 - s)]);
    hl = { nodes, strong: new Set(), edges: [] };
    info.innerHTML = `<h3>搜索 "${esc(q)}"</h3>
      <h3>seq / alt（${syms.length}）</h3><ul>${
      syms.slice(0, 30).map((s) => `<li>${symLink(s)}</li>`).join("") ||
      "<li>无</li>"
    }</ul>
      <h3>上下文（按证据量，前 ${ctxs.length} 个）</h3><ul>${
      ctxs.map((k) =>
        `<li>${ctxLink(k)} <span class="kv">N=${fmt(C.n[k])}</span></li>`
      ).join("") || "<li>无</li>"
    }</ul>`;
    schedule();
  }

  // ---------- 事件 ----------
  info.addEventListener("click", (e) => {
    const a = e.target.closest("a[data-id]");
    if (a) select(Number(a.dataset.id), true);
  });
  let drag = null;
  cv.addEventListener("mousedown", (e) => {
    drag = { x: e.offsetX, y: e.offsetY, ox, oy, moved: false };
    cv.classList.add("drag");
  });
  addEventListener("mouseup", (e) => {
    if (drag && !drag.moved && e.target === cv) {
      select(pick(e.offsetX, e.offsetY), false);
    }
    drag = null;
    cv.classList.remove("drag");
  });
  cv.addEventListener("mousemove", (e) => {
    if (drag) {
      const dx = e.offsetX - drag.x, dy = e.offsetY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      ox = drag.ox + dx;
      oy = drag.oy + dy;
      tip.style.display = "none";
      return schedule();
    }
    const id = pick(e.offsetX, e.offsetY);
    if (id !== hover) {
      hover = id;
      schedule();
    }
    if (id === null) {
      tip.style.display = "none";
      return;
    }
    tip.textContent = id >= 0
      ? `${full(id)}\n${KINDS[C.kind[id]]} · ${C.d[id]} 槽位 · N=${
        fmt(C.n[id])
      }\n→ ${C.top[id]}`
      : `${S.kind[-1 - id] ? "alt" : "seq"}  ${S.lab[-1 - id]}\n被 ${
        users[-1 - id].length
      } 个上下文引用`;
    tip.style.display = "block";
    tip.style.left = Math.min(e.offsetX + 14, W - tip.offsetWidth - 4) + "px";
    tip.style.top = Math.min(e.offsetY + 14, H - tip.offsetHeight - 4) + "px";
  });
  cv.addEventListener("mouseleave", () => {
    hover = null;
    tip.style.display = "none";
    schedule();
  });
  cv.addEventListener("wheel", (e) => {
    e.preventDefault();
    const f = Math.exp(-e.deltaY * 0.0015);
    ox = e.offsetX - (e.offsetX - ox) * f;
    oy = e.offsetY - (e.offsetY - oy) * f;
    scale *= f;
    schedule();
  }, { passive: false });

  for (const key of Object.keys(show)) {
    document.getElementById("t-" + key).addEventListener("change", (e) => {
      show[key] = e.target.checked;
      schedule();
    });
  }
  const exSel = document.getElementById("examples");
  G.examples.forEach((ex, i) =>
    exSel.add(new Option(JSON.stringify(ex.text), String(i)))
  );
  exSel.addEventListener(
    "change",
    () => showExample(exSel.value === "" ? -1 : Number(exSel.value)),
  );
  const box = document.getElementById("search");
  box.addEventListener("keydown", (e) => {
    if (e.key === "Enter") search(box.value);
  });
  box.addEventListener("search", () => search(box.value));
  document.getElementById("reset").addEventListener("click", () => {
    fit();
    select(null);
    exSel.value = "";
    box.value = "";
  });

  document.getElementById("stats").textContent = `${fmt(N)} ctx · ${
    fmt(NS)
  } seq/alt · 最深 ${G.maxD} 槽位`;
  addEventListener("resize", resize);
  defaultInfo();
  resize();
})();
