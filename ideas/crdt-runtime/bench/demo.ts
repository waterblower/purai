import { compile } from "../src/compile.ts";
import { SumClassifier } from "../src/example.ts";
import { cases, oracle, trainReplicas } from "./fixtures.ts";

const start = performance.now();
const replicas = await trainReplicas();
const trainingMs = performance.now() - start;
const compileStart = performance.now();
const view = compile(replicas.merged), classifier = new SumClassifier(view);
const compileMs = performance.now() - compileStart;
const dataset = cases();
const budgets = [8, 32, 128, 512, 2048].map((budget) => {
  let correct = 0, completed = 0, correctCompleted = 0, steps = 0, calls = 0;
  const lengths = new Set<number>();
  for (const c of dataset) {
    const r = classifier.predict(c.a, c.b, budget);
    correct += Number(r.prediction === c.expected);
    completed += Number(r.completed);
    correctCompleted += Number(r.completed && r.prediction === c.expected);
    steps += r.steps;
    calls += r.calls;
    lengths.add(r.prediction.length);
  }
  return {
    budget,
    accuracy: correct / dataset.length,
    completionRate: completed / dataset.length,
    completedAccuracy: completed ? correctCompleted / completed : null,
    meanSteps: steps / dataset.length,
    meanCalls: calls / dataset.length,
    predictionCharacters: [...lengths],
  };
});
if (budgets.at(-1)!.accuracy !== 1 || budgets.at(-1)!.completionRate !== 1) {
  throw new Error("Full-budget benchmark failed");
}

// Warm JIT; latency includes decimal parsing + workspace + VM, excludes compile/IO.
for (let pass = 0; pass < 3; pass++) {
  for (const c of dataset) classifier.predict(c.a, c.b, 2048);
}
const latencyUs: number[] = [];
let checksum = 0;
const runtimeStart = performance.now();
for (let pass = 0; pass < 5; pass++) {
  for (const c of dataset) {
    const t = performance.now();
    checksum += Number(classifier.predict(c.a, c.b, 2048).prediction);
    latencyUs.push((performance.now() - t) * 1000);
  }
}
const runtimeMs = performance.now() - runtimeStart;
latencyUs.sort((a, b) => a - b);
const nativeStart = performance.now();
let nativeChecksum = 0;
for (let pass = 0; pass < 5; pass++) {
  for (const c of dataset) nativeChecksum += Number(oracle(c.a, c.b));
}
const nativeMs = performance.now() - nativeStart;
if (checksum !== nativeChecksum) throw new Error("Reference mismatch");

await Deno.mkdir("artifacts", { recursive: true });
const snapshots = {
  "replica-a": replicas.a,
  "replica-b": replicas.b,
  "merged": replicas.merged,
};
const sizes: Record<string, number> = {};
for (const [name, state] of Object.entries(snapshots)) {
  const wire = state.encode();
  await Deno.writeTextFile(`artifacts/${name}.state.json`, wire);
  sizes[name] = new TextEncoder().encode(wire).length;
}
const composition = Object.fromEntries(
  Object.entries(snapshots).map(([name, state]) => {
    const r = new SumClassifier(compile(state)).predict(
      "99999999999999999999999999999999",
      "4",
      2048,
      true,
    );
    return [name, {
      status: r.status,
      completed: r.completed,
      prediction: r.prediction,
      steps: r.steps,
      calls: r.calls,
      expertsUsed: [...new Set(r.trace.map((x) => x.module))],
    }];
  }),
);
const report = {
  generatedAt: new Date().toISOString(),
  runtime: Deno.version,
  platform: Deno.build,
  training: {
    observations: 230,
    addition: 200,
    modulo: 30,
    program: "handwritten bytecode",
    trainingMs,
    compileMs,
  },
  evaluation: {
    cases: dataset.length,
    labels: "balanced within each length",
    decimalLengths: [1, 2, 4, 8, 16, 32, 64, 128],
    seed: 0x5eed,
    budgets,
  },
  composition,
  crdt: {
    repeatedMergeIdentical:
      replicas.merged.merge(replicas.a).merge(replicas.b).encode() ===
        replicas.merged.encode(),
    records: replicas.merged.size,
  },
  storage: {
    snapshotBytes: sizes,
    compiled: view.stats,
    typedArraysExclude:
      "JS maps, strings, objects, bytecode, and per-query workspace",
  },
  cpu: {
    timedQueries: latencyUs.length,
    wallMs: runtimeMs,
    queriesPerSecond: latencyUs.length / (runtimeMs / 1000),
    p50Us: latencyUs[Math.floor(latencyUs.length * .5)],
    p95Us: latencyUs[Math.floor(latencyUs.length * .95)],
    nativeBigIntReferenceWallMs: nativeMs,
    checksum,
  },
};
await Deno.writeTextFile(
  "artifacts/results.json",
  JSON.stringify(report, null, 2) + "\n",
);
const pct = (x: number) => `${(x * 100).toFixed(2)}%`;
const markdown =
  `# CPU / CRDT / internal-computation prototype\n\nGenerated: ${report.generatedAt}. Deno ${Deno.version.deno}, ${Deno.build.arch}, ${Deno.build.os}.\n\n` +
  `Training: 200 digit-addition transitions on replica A; 30 remainder transitions on replica B. Program control flow is handwritten. No full-task examples are trained. Evaluation: ${dataset.length} balanced queries, 1–128 digits per operand. The 1-digit evaluation overlaps primitive inputs; longer compositions are unseen.\n\n` +
  `| VM budget | Accuracy | Completed | Completed accuracy | Mean steps | Prediction length |\n|---:|---:|---:|---:|---:|---:|\n` +
  budgets.map((r) =>
    `| ${r.budget} | ${pct(r.accuracy)} | ${pct(r.completionRate)} | ${
      r.completedAccuracy === null ? "n/a" : pct(r.completedAccuracy)
    } | ${r.meanSteps.toFixed(2)} | 1 character |`
  ).join("\n") +
  `\n\nIncomplete computations return an explicitly marked constant prior guess. Completed accuracy refers only to computations reaching emit. This toy task demonstrates a compute/output separation, not general reasoning or language generation.\n\n` +
  `Replica A and replica B each stop with a missing capability. The merged model completes the composition without extra training. Repeated merge preserves the exact snapshot: ${report.crdt.repeatedMergeIdentical}.\n\n` +
  `CPU (warm, sequential, ${latencyUs.length} queries): ${
    report.cpu.queriesPerSecond.toFixed(0)
  } queries/s; p50 ${report.cpu.p50Us.toFixed(2)} µs; p95 ${
    report.cpu.p95Us.toFixed(2)
  } µs. Total VM query wall time ${
    runtimeMs.toFixed(2)
  } ms; native BigInt reference ${
    nativeMs.toFixed(2)
  } ms. This measures warm inference, including input parsing and workspace construction, excluding model loading/compilation. Single-run measurements vary with JIT, GC and system load; this is not an LLM comparison.\n\n` +
  `Authoritative snapshot: ${sizes.merged} bytes / ${replicas.merged.size} records. Compiled typed arrays: ${view.stats.typedArrayBytes} bytes (not total memory). Training/build ${
    trainingMs.toFixed(2)
  } ms; compilation ${
    compileMs.toFixed(2)
  } ms. The append-only evidence set grows with observations; bounded-storage synchronization and program learning remain unimplemented.\n`;
await Deno.writeTextFile("artifacts/RESULTS.md", markdown);
console.log(markdown);
