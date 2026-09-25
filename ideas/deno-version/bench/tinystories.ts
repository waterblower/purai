// Whole-story train/validation splits, same byte metric and PY smoothing for all models.
// deno run --allow-read --allow-write bench/tinystories.ts [default|no-fold|ngram] [maxOrder=10]
import { defaultOptions, stats, train } from "../src/learn.ts";
import { emptyImage } from "../src/image.ts";
import { load, save } from "../src/model_io.ts";
import { evalBits, generate, predictorFor } from "../src/predict.ts";
import { DEFAULT_PARAMS } from "../src/types.ts";
import { rng } from "../src/hash.ts";

const mode = Deno.args[0] ?? "default";
if (!["default", "no-fold", "ngram"].includes(mode)) {
  throw new Error("Unknown mode");
}
const base = new URL("./data/tinystories10mb/", import.meta.url);
const output = new URL("./results/tinystories10mb/", import.meta.url);
await Deno.mkdir(output, { recursive: true });
const encoder = new TextEncoder();
async function docs(split: string) {
  return (await Deno.readTextFile(new URL(`${split}.jsonl`, base))).trimEnd()
    .split("\n").map((line) => encoder.encode(JSON.parse(line)));
}
const trainStories = await docs("train"), validation = await docs("validation");
// Match the existing CLI's single-corpus training protocol. Keep complete stories,
// separated by two newlines. Validation sessions still reset for each story.
const decoder = new TextDecoder();
const training = [
  encoder.encode(trainStories.map((s) => decoder.decode(s)).join("\n\n")),
];
const total = (xs: Uint8Array[]) => xs.reduce((n, x) => n + x.length, 0);
const trainBytes = total(training), validBytes = total(validation);
const manifest = JSON.parse(
  await Deno.readTextFile(new URL("manifest.json", base)),
);
if (
  total(trainStories) !== manifest.splits.train.bytes ||
  validBytes !== manifest.splits.validation.bytes
) {
  throw new Error("Dataset size mismatch");
}
for (const split of ["train", "validation"]) {
  const bytes = await Deno.readFile(new URL(`${split}.jsonl`, base));
  const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((b) => b.toString(16).padStart(2, "0")).join("");
  if (hash !== manifest.splits[split].file_sha256) {
    throw new Error("Dataset hash mismatch");
  }
}
await Deno.writeTextFile(
  new URL("manifest.json", output),
  JSON.stringify(manifest, null, 2) + "\n",
);
console.log(
  JSON.stringify({
    mode,
    trainBytes,
    validBytes,
    trainStories: trainStories.length,
    validStories: validation.length,
  }),
);
const write = async (name: string, value: unknown) =>
  await Deno.writeTextFile(
    new URL(name, output),
    JSON.stringify(value, null, 2) + "\n",
  );

if (mode === "ngram") {
  // Build one order at a time to avoid retaining seven copies of all string keys.
  // Per-position probabilities retain the previous order's backoff distribution.
  const probabilities = validation.map((text) =>
    new Float64Array(text.length).fill(1 / 256)
  );
  const key = (t: Uint8Array, start: number, end: number) =>
    String.fromCharCode(...t.subarray(start, end));
  const results = [];
  const maxOrder = Number(Deno.args[1] ?? 10);
  if (!Number.isInteger(maxOrder) || maxOrder < 0 || maxOrder > 12) {
    throw new Error("maxOrder must be an integer between 0 and 12");
  }
  for (let order = 0; order <= maxOrder; order++) {
    const start = performance.now();
    const joint = new Map<string, number>(),
      totals = new Map<string, number>(),
      distinct = new Map<string, number>();
    for (const text of training) {
      for (let i = order; i < text.length; i++) {
        const context = key(text, i - order, i),
          event = context + String.fromCharCode(text[i]);
        totals.set(context, (totals.get(context) ?? 0) + 1);
        const previous = joint.get(event) ?? 0;
        if (!previous) distinct.set(context, (distinct.get(context) ?? 0) + 1);
        joint.set(event, previous + 1);
      }
    }
    let bits = 0;
    validation.forEach((text, doc) => {
      const p = probabilities[doc];
      for (let i = 0; i < text.length; i++) {
        if (i >= order) {
          const context = key(text, i - order, i), n = totals.get(context);
          if (n !== undefined) {
            const count = joint.get(context + String.fromCharCode(text[i])) ??
              0;
            const { discount, theta } = DEFAULT_PARAMS;
            p[i] = (Math.max(count - discount, 0) +
              (theta + discount * distinct.get(context)!) * p[i]) /
              (n + theta);
          }
        }
        bits -= Math.log2(p[i]);
      }
    });
    const result = {
      order,
      bpb: bits / validBytes,
      contexts: totals.size,
      seconds: (performance.now() - start) / 1000,
    };
    results.push(result);
    console.log(JSON.stringify(result));
    await write("ngram.json", { params: DEFAULT_PARAMS, results });
  }
} else {
  const options = {
    ...defaultOptions,
    ...(mode === "no-fold" ? { foldsPerRound: 0 } : {}),
  };
  const path = new URL(`${mode}.data`, output);
  let model;
  let trainSeconds: number | null = null;
  try {
    model = await load(path.pathname);
    console.log(`Loaded existing ${mode} model`);
  } catch (error) {
    if (!(error instanceof Deno.errors.NotFound)) throw error;
    const start = performance.now();
    const rounds: unknown[] = [];
    model = train(emptyImage(), training, validation, options, (report) => {
      rounds.push(report);
      console.log(JSON.stringify(report));
      Deno.writeTextFileSync(
        new URL(`${mode}-rounds.json`, output),
        JSON.stringify(rounds, null, 2) + "\n",
      );
    }, (removed) => console.log(`compact removed=${removed}`));
    trainSeconds = (performance.now() - start) / 1000;
    await save(path.pathname, model);
  }
  const evaluations = [];
  for (const useMatch of [true, false]) {
    const predictor = predictorFor(model, useMatch);
    const start = performance.now();
    let bits = 0, steps = 0;
    const perStory = validation.map((text) => {
      const result = evalBits(predictor, text); // Fresh match state for every story.
      bits += result.bits;
      steps += result.steps;
      return { bytes: text.length, bits: result.bits };
    });
    const seconds = (performance.now() - start) / 1000;
    const result = {
      useMatch,
      bpb: bits / validBytes,
      stepsPerByte: steps / validBytes,
      seconds,
      bytesPerSecond: validBytes / seconds,
    };
    evaluations.push(result);
    console.log(JSON.stringify(result));
    await write(
      `${mode}-${useMatch ? "match" : "no-match"}-per-story.json`,
      perStory,
    );
  }
  const samples = [];
  for (
    const prompt of [
      "Once upon a time, there was a little girl named Lily.",
      "Tom found a small red box in the garden.",
      "The little rabbit was afraid of the dark.",
    ]
  ) {
    for (const temperature of [0.8, 1.0]) {
      const { bytes } = generate(
        predictorFor(model),
        encoder.encode(prompt),
        600,
        { temperature, random: rng(42) },
      );
      samples.push({
        prompt,
        temperature,
        seed: 42,
        continuation: new TextDecoder().decode(bytes),
      });
    }
  }
  await write(`${mode}-samples.json`, samples);
  await write(`${mode}.json`, {
    mode,
    options,
    params: model.params,
    trainBytes,
    validBytes,
    trainSeconds,
    modelBytes: (await Deno.stat(path)).size,
    structure: stats(model),
    evaluations,
    deno: Deno.version,
    protocol:
      "Training stories concatenated with two newlines; validation state reset for every story.",
  });
}
