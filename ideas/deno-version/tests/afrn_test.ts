// 验证 topic.md 里"模型即数据库"的几条定律。零依赖：只用 Deno.test。
import { defaultOptions, recount, train, TrainOptions } from "../src/learn.ts";
import { emptyImage, merge, unlearn } from "../src/image.ts";
import { decode, encode, sameImage } from "../src/model_io.ts";
import { evalBits, predictorFor } from "../src/predict.ts";
import { encoder } from "../src/text.ts";
import { rng } from "../src/hash.ts";
import { compact } from "../src/compact.ts";
import { Image } from "../src/types.ts";

function assert(cond: unknown, msg = "assertion failed"): asserts cond {
  if (!cond) throw new Error(msg);
}

// 一个小的合成语料：主语 + 动词 + 宾语，保证有可泛化的结构
function corpus(seed: number, sentences: number): Uint8Array {
  const r = rng(seed);
  const pick = <T>(xs: T[]) => xs[Math.floor(r() * xs.length)];
  const subj = [
    "the king",
    "the queen",
    "my lord",
    "the duke",
    "his father",
    "her brother",
  ];
  const verb = ["loves", "hates", "sees", "calls", "serves", "fears"];
  const obj = [
    "the people",
    "his son",
    "the crown",
    "my cousin",
    "the city",
    "her mother",
  ];
  const out: string[] = [];
  for (let i = 0; i < sentences; i++) {
    out.push(
      `${pick(subj)} ${pick(verb)} ${pick(obj)}${pick([".", ",", "!", ";"])}\n`,
    );
  }
  return encoder.encode(out.join(""));
}

const opts: TrainOptions = { ...defaultOptions, rounds: 8, minChunkCount: 4 };
const quick = (text: Uint8Array): Image =>
  train(emptyImage(), [text], [], opts);
const bitsOf = (img: Image, text: Uint8Array, useMatch = true) =>
  evalBits(predictorFor(img, useMatch), text).bits;

const A = corpus(1, 400), B = corpus(2, 400), C = corpus(3, 400);
const mA = quick(A), mB = quick(B), mC = quick(C);

Deno.test("序列化往返：逐字节相同", () => {
  const bytes = encode(mA);
  const again = encode(decode(bytes));
  assert(
    bytes.length === again.length && bytes.every((x, i) => x === again[i]),
  );
});

Deno.test("文件损坏会被校验和发现", () => {
  const bytes = encode(mA).slice();
  bytes[20] ^= 0xff;
  let threw = false;
  try {
    decode(bytes);
  } catch {
    threw = true;
  }
  assert(threw);
});

Deno.test("训练是确定性的：两次训练得到相同文件", () => {
  assert(sameImage(quick(A), mA));
});

Deno.test("训练中增量维护的计数 = 从头重新计数", () => {
  assert(sameImage(recount(mA, [A]), mA));
});

Deno.test("计数和校准表对文档可加：count(A) + count(B) = count(A, B)", () => {
  const sep = merge(recount(mA, [A]), recount(mA, [B]));
  assert(sameImage(sep, recount(mA, [A, B])));
});

Deno.test("merge 满足交换律", () => {
  assert(sameImage(merge(mA, mB), merge(mB, mA)));
});

Deno.test("merge 满足结合律", () => {
  assert(sameImage(merge(merge(mA, mB), mC), merge(mA, merge(mB, mC))));
});

Deno.test("精确遗忘：unlearn(merge(A, B), B) = A", () => {
  assert(sameImage(unlearn(merge(mA, mB), mB), mA));
  assert(sameImage(unlearn(merge(mA, mB), mA), mB));
});

Deno.test("按文本遗忘：删除一份文档的计数后，与只在其余文档上计数一致", () => {
  const both = recount(mA, [A, B]);
  const forgotten = unlearn(both, recount(both, [B]));
  assert(sameImage(forgotten, recount(mA, [A])));
});

Deno.test("结构学习有效：比只有根节点（order-0）的模型好", () => {
  const valid = corpus(9, 100);
  const root = recount(emptyImage(), [A]);
  const bRoot = bitsOf(root, valid, false);
  const bModel = bitsOf(mA, valid, false);
  assert(bModel < bRoot * 0.6, `model=${bModel} root=${bRoot}`);
});

Deno.test("match 模型：训练数据里重复是可靠的，模型就会在新文本里利用重复", () => {
  const x = corpus(4, 200);
  const xx = new Uint8Array(x.length * 2);
  xx.set(x);
  xx.set(x, x.length);
  const mR = quick(xx);
  const once = corpus(7, 60);
  const twice = new Uint8Array(once.length * 2);
  twice.set(once);
  twice.set(once, once.length);
  const second = (useMatch: boolean) =>
    bitsOf(mR, twice, useMatch) - bitsOf(mR, once, useMatch);
  assert(
    second(true) < second(false) * 0.3,
    `with=${second(true)} without=${second(false)}`,
  );
});

Deno.test("参数不同的模型拒绝合并", () => {
  const other = train(
    emptyImage({ discount: 0.5, theta: 1, kappa: 2 }),
    [B],
    [],
    opts,
  );
  let threw = false;
  try {
    merge(mA, other);
  } catch {
    threw = true;
  }
  assert(threw);
});

Deno.test("压缩只删节点、不改计数，且删的都是有等价替身的叶子", () => {
  const full = train(emptyImage(), [A], [], { ...opts, compact: false });
  const { img: c, removed } = compact(full, 2);
  assert(removed > 0, "合成语料上应该有可压缩的上下文");
  const sigs = new Set([...c.ctxCounts.values()].map((x) => x.join(",")));
  for (const [id, cnt] of full.ctxCounts) {
    const kept = c.ctxCounts.get(id);
    if (kept) assert(kept.join(",") === cnt.join(","), "保留节点的计数被改了");
    else assert(sigs.has(cnt.join(",")), "删掉的节点没有等价替身");
  }
  // ctx 计数只取决于节点本身，删掉别的节点不影响它（校准表则需要重新统计，训练流程会做）
  const again = recount(c, [A]);
  for (const [id, cnt] of c.ctxCounts) {
    assert(
      again.ctxCounts.get(id)!.join(",") === cnt.join(","),
      "压缩后的计数与重新计数不一致",
    );
  }
});

Deno.test("多文档训练后可以精确遗忘其中一篇", () => {
  const m = train(emptyImage(), [A, B], [], opts);
  const forgotten = unlearn(m, recount(m, [B]));
  assert(sameImage(forgotten, recount(m, [A])));
});

Deno.test("合并以后的模型在两边的数据上都不比单个模型差", () => {
  const m = merge(mA, mB);
  for (const [text, single] of [[A, mA], [B, mB]] as const) {
    const merged = bitsOf(m, text);
    const other = bitsOf(single === mA ? mB : mA, text);
    assert(merged < other, `merged=${merged} other=${other}`);
  }
});
