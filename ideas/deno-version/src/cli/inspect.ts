import { fail, parseArgs } from "./args.ts";
import { load } from "../model_io.ts";
import { compile } from "../compile.ts";
import { renderCtx, renderRef } from "../image.ts";
import { ref } from "../types.ts";
import { renderBytes } from "../text.ts";

const a = parseArgs(Deno.args, ["help", "h"]);
if (a.bool("help") || a.bool("h") || a.positional.length !== 1) {
  fail("usage: inspect <model.data> [--top <k>]");
}
const top = a.num("top", 20);
const img = await load(a.positional[0]);
const comp = compile(img);

const seqs = [...img.nodes].filter(([, n]) => n.kind === "seq");
const alts = [...img.nodes].filter(([, n]) => n.kind === "alt");
console.log(
  `nodes: ${img.nodes.size}  ctx=${comp.ctx.length} (classed ${
    comp.ctx.filter((c) => c.classed).length
  })  seq=${seqs.length}  alt=${alts.length}`,
);

console.log(`\n== 最常见的 seq（词首片段）==`);
console.log(
  seqs.map(([id]) => [id, img.seqFreq.get(id) ?? 0] as const)
    .sort((x, y) => y[1] - x[1]).slice(0, top)
    .map(([id, f]) => `${renderRef(img, ref(id))}:${f}`).join("  "),
);

console.log(`\n== alt 类（Fold 学出的可替换类）==`);
const altUse = new Map<string, number>();
for (const c of comp.ctx) {
  for (const s of c.slots) {
    if (s.tag === "ref") altUse.set(s.id, (altUse.get(s.id) ?? 0) + c.total);
  }
}
alts.map(([id, n]) => ({ id, n, use: altUse.get(id) ?? 0 }))
  .sort((x, y) =>
    (y.n.kind === "alt" && x.n.kind === "alt"
      ? y.n.items.length - x.n.items.length
      : 0) || y.use - x.use
  )
  .slice(0, top)
  .forEach(({ id }) => console.log(`  ${renderRef(img, ref(id))}`));

console.log(`\n== 最具体的上下文（深度最大、证据最多）==`);
[...comp.ctx].filter((c) => c.depth > 0)
  .sort((x, y) => y.depth - x.depth || y.total - x.total).slice(0, top)
  .forEach((c) => {
    const nxt = [...c.counts.keys()].filter((x) => c.counts[x] > 0)
      .sort((x, y) => c.counts[y] - c.counts[x]).slice(0, 4)
      .map((x) => `${renderBytes([x])}:${c.counts[x]}`).join(" ");
    console.log(
      `  ${renderCtx(img, c.slots).padEnd(40)} N=${c.total}  → ${nxt}`,
    );
  });
