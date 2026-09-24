// 基线：字节级 n-gram，保留所有出现过的上下文，用与 AFRN 相同的 Pitman-Yor 平滑。
// 它没有任何结构学习（没有 MDL、Chunk、Fold），也没有 match 模型，用来检验这些机制是否有效。
// 用法：deno run --allow-read bench/ngram.ts <file> [maxOrder=6] [discount=0.75] [theta=0.5] [validFrac=0.1]
const [path, maxOrderS = "6", dS = "0.75", thS = "0.5", fracS = "0.1"] =
  Deno.args;
const data = await Deno.readFile(path);
const cut = Math.floor(data.length * (1 - Number(fracS)));
const train = data.subarray(0, cut), valid = data.subarray(cut);
const d = Number(dS), theta = Number(thS);
const maxOrder = Number(maxOrderS);

const key = (t: Uint8Array, s: number, e: number) =>
  String.fromCharCode(...t.subarray(s, e));

// joint[k]: "上下文+下一个字节" -> 次数；ctxTotal[k]: 上下文 -> 次数；distinct[k]: 上下文 -> 不同后继数
const joint: Map<string, number>[] = [],
  ctxTotal: Map<string, number>[] = [],
  distinct: Map<string, number>[] = [];
for (let k = 0; k <= maxOrder; k++) {
  joint.push(new Map());
  ctxTotal.push(new Map());
  distinct.push(new Map());
}
for (let i = 0; i < train.length; i++) {
  for (let k = 0; k <= maxOrder && k <= i; k++) {
    const c = key(train, i - k, i), j = c + String.fromCharCode(train[i]);
    ctxTotal[k].set(c, (ctxTotal[k].get(c) ?? 0) + 1);
    const prev = joint[k].get(j) ?? 0;
    if (prev === 0) distinct[k].set(c, (distinct[k].get(c) ?? 0) + 1);
    joint[k].set(j, prev + 1);
  }
}

for (let K = 0; K <= maxOrder; K++) {
  let bits = 0;
  for (let i = 0; i < valid.length; i++) {
    let p = 1 / 256;
    for (let k = 0; k <= K && k <= i; k++) {
      const c = key(valid, i - k, i);
      const n = ctxTotal[k].get(c);
      if (n === undefined) break;
      const cx = joint[k].get(c + String.fromCharCode(valid[i])) ?? 0;
      const u = distinct[k].get(c)!;
      p = (Math.max(cx - d, 0) + (theta + d * u) * p) / (n + theta);
    }
    bits -= Math.log2(p);
  }
  console.log(
    `order ${K}: valid_bpc=${(bits / valid.length).toFixed(4)}  contexts=${
      ctxTotal[K].size
    }`,
  );
}
