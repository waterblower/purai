import { fail, parseArgs } from "./args.ts";
import { load } from "../model_io.ts";
import { evalBits, predictorFor } from "../predict.ts";

const a = parseArgs(Deno.args, ["no-match"]);
const modelPath = a.str("model");
const textPath = a.str("text");
if (!modelPath || !textPath) {
  fail(
    "usage: eval --model <file> --text <file> [--valid-frac 0.1] [--no-match]\n" +
      "  在文本末尾 valid-frac 部分上计算 bits-per-byte（设为 1 则用整份文本）\n" +
      "  --no-match  关掉 match 模型（用于对比）",
  );
}
const data = await Deno.readFile(textPath);
const frac = a.num("valid-frac", 0.1);
const text = data.subarray(Math.floor(data.length * (1 - frac)));
const pr = predictorFor(await load(modelPath), !a.bool("no-match"));
const t0 = performance.now();
const { bits, steps } = evalBits(pr, text);
const secs = (performance.now() - t0) / 1000;
console.log(
  `bpc=${(bits / text.length).toFixed(4)}  bytes=${text.length}  steps/byte=${
    (steps / text.length).toFixed(1)
  }  ${(text.length / 1024 / secs).toFixed(0)} KB/s`,
);
