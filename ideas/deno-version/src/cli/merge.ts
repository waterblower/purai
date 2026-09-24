import { fail, parseArgs } from "./args.ts";
import { load, save } from "../model_io.ts";
import { merge } from "../image.ts";

const a = parseArgs(Deno.args);
if (a.positional.length < 2) {
  fail("usage: merge <a.data> <b.data> [...] -o <out.data>");
}
const out = a.str("o", "merged.data")!;
let img = await load(a.positional[0]);
for (const p of a.positional.slice(1)) img = merge(img, await load(p));
const size = await save(out, img);
console.log(
  `merged ${a.positional.length} models -> ${out} (${img.nodes.size} nodes, ${
    (size / 1024).toFixed(0)
  } KB)`,
);
