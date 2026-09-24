import { fail, parseArgs } from "./args.ts";
import { load, save } from "../model_io.ts";
import { unlearn } from "../image.ts";
import { recount } from "../learn.ts";

const USAGE = `usage:
  unlearn <model.data> <forget.data> -o <out.data>   减去另一个模型的全部证据（它必须曾被 merge 进来）
  unlearn <model.data> --text <doc.txt> -o <out.data> 用模型自己的代码给 doc 计数，再减去`;

const a = parseArgs(Deno.args);
const out = a.str("o", "unlearned.data")!;
const textPath = a.str("text");
if (a.positional.length !== (textPath ? 1 : 2)) fail(USAGE);

const img = await load(a.positional[0]);
const forget = textPath
  ? recount(img, [await Deno.readFile(textPath)])
  : await load(a.positional[1]);
const res = unlearn(img, forget);
const size = await save(out, res);
console.log(
  `unlearned -> ${out} (${res.nodes.size} nodes, ${
    (size / 1024).toFixed(0)
  } KB)`,
);
