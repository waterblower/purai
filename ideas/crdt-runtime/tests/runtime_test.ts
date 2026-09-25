import { cases, oracle, trainReplicas } from "../bench/fixtures.ts";
import { compile } from "../src/compile.ts";
import { ADD, ADD_SPEC, digits, MOD3, SumClassifier } from "../src/example.ts";
import { Knowledge } from "../src/knowledge.ts";
import { canonical, encodeRow, idOf, type Payload } from "../src/schema.ts";
import { Executable } from "../src/vm.ts";

function assert(
  condition: unknown,
  message = "Assertion failed",
): asserts condition {
  if (!condition) throw new Error(message);
}
function equal(actual: unknown, expected: unknown) {
  assert(
    canonical(actual) === canonical(expected),
    `${canonical(actual)} != ${canonical(expected)}`,
  );
}
async function rejects(fn: () => unknown | Promise<unknown>) {
  let rejected = false;
  try {
    await fn();
  } catch {
    rejected = true;
  }
  assert(rejected, "Expected rejection");
}
const fixture = trainReplicas();

Deno.test("set union is associative, commutative, idempotent with overlapping histories", async () => {
  const { a, b, shared, merged } = await fixture;
  equal(a.merge(b).encode(), b.merge(a).encode());
  equal(a.merge(a).encode(), a.encode());
  equal(a.merge(b).merge(shared).encode(), a.merge(b.merge(shared)).encode());
  equal(merged.size, 233);
  // Deliver the same three snapshots in all six orders, then replay them.
  for (
    const order of [
      [a, b, shared],
      [a, shared, b],
      [b, a, shared],
      [b, shared, a],
      [shared, a, b],
      [shared, b, a],
    ]
  ) {
    let state = Knowledge.empty();
    for (const message of [...order, ...order, merged, a, b]) {
      state = state.merge(message);
    }
    equal(state.encode(), merged.encode());
    equal(compile(state).stats.acceptedEvidence, 230);
  }
});

Deno.test("serialization is canonical and hash-checked; committed data is immutable", async () => {
  const { merged } = await fixture;
  equal((await Knowledge.decode(merged.encode())).encode(), merged.encode());
  const tampered = JSON.parse(merged.encode());
  tampered.records.find((r: { payload: Payload }) => r.payload.kind === "table")
    .payload.name = "tampered";
  await rejects(() => Knowledge.decode(JSON.stringify(tampered)));
  const payload: Payload = {
    kind: "table",
    name: "original",
    port: { name: "test", inputs: [2], outputs: [2] },
  };
  const committing = Knowledge.empty().add([payload]);
  (payload as { name: string }).name = "mutated";
  const state = await committing;
  equal((state.entries()[0].payload as { name: string }).name, "original");
  await rejects(() => {
    (state.entries()[0].payload as { name: string }).name = "mutated again";
  });
});

Deno.test("event identity deduplicates replay, counts independent observations, quarantines contradictions", async () => {
  const module = await idOf(ADD_SPEC);
  const e: Payload = {
    kind: "evidence",
    module,
    event: "observation/1",
    input: [1, 2, 0],
    output: [3, 0],
  };
  const a = await Knowledge.empty().add([ADD_SPEC, e, e]);
  equal(compile(a.merge(a)).stats.acceptedEvidence, 1);
  const b = await a.add([{ ...e, event: "observation/2" }]);
  equal(compile(b).stats.acceptedEvidence, 2);
  const conflicting = await Knowledge.empty().add([{ ...e, output: [4, 0] }]);
  const view = compile(b.merge(conflicting));
  equal(view.stats.conflictingEvents, 1);
  equal(view.stats.acceptedEvidence, 1);
  equal(b.merge(conflicting).encode(), conflicting.merge(b).encode());
});

Deno.test("out-of-order evidence activates when its module arrives; invalid domains are diagnosed", async () => {
  const module = await idOf(ADD_SPEC);
  const orphan = await Knowledge.empty().add([{
    kind: "evidence",
    module,
    event: "early",
    input: [1, 2, 0],
    output: [3, 0],
  }]);
  equal(compile(orphan).stats.deferredEvidence, 1);
  const state = orphan.merge(await Knowledge.empty().add([ADD_SPEC]));
  equal(compile(state).stats.acceptedEvidence, 1);
  const invalid = await state.add([{
    kind: "evidence",
    module,
    event: "bad",
    input: [10, 0, 0],
    output: [0, 0],
  }]);
  equal(compile(invalid).stats.invalidEvidence, 1);
});

Deno.test("row routing chooses confidence then stable module id, independently of delivery order", async () => {
  const other: Payload = {
    kind: "table",
    name: "alternative expert",
    port: ADD,
  };
  const firstId = await idOf(ADD_SPEC), secondId = await idOf(other);
  const a = await Knowledge.empty().add([ADD_SPEC, {
    kind: "evidence",
    module: firstId,
    event: "a",
    input: [1, 2, 0],
    output: [3, 0],
  }]);
  const b = await Knowledge.empty().add([other, {
    kind: "evidence",
    module: secondId,
    event: "b",
    input: [1, 2, 0],
    output: [4, 0],
  }]);
  const row = encodeRow([1, 2, 0], ADD.inputs)!;
  for (const state of [a.merge(b), b.merge(a)]) {
    equal(
      compile(state).tables.get(canonical(ADD))!.owners[row],
      [firstId, secondId].sort()[0],
    );
  }
  const stronger = await b.add([{
    kind: "evidence",
    module: secondId,
    event: "b2",
    input: [1, 2, 0],
    output: [4, 0],
  }]);
  equal(
    compile(a.merge(stronger)).tables.get(canonical(ADD))!.owners[row],
    secondId,
  );
});

Deno.test("independent skills compose only after merging, without further training", async () => {
  const { a, b, merged, addId, modId } = await fixture;
  for (const state of [a, b]) {
    const result = new SumClassifier(compile(state)).predict("999", "3", 2048);
    equal(result.status, "missing-capability");
    assert(!result.completed);
    equal(result.prediction.length, 1);
  }
  const result = new SumClassifier(compile(merged)).predict(
    "999",
    "3",
    2048,
    true,
  );
  assert(result.completed);
  equal(result.prediction, "1");
  equal(
    [...new Set(result.trace.map((x) => x.module))].sort(),
    [addId, modId].sort(),
  );
});

Deno.test("learned transitions work on carries, leading zeros, unequal lengths and 4096 digits", async () => {
  const { merged } = await fixture;
  const before = merged.encode(),
    classifier = new SumClassifier(compile(merged));
  const inputs = [
    ["0", "0"],
    ["99", "1"],
    ["0099", "0001"],
    ["123456789", "3"],
    ["9".repeat(4096), "1"],
  ];
  for (const [a, b] of inputs) {
    const result = classifier.predict(a, b, 100_000);
    assert(result.completed);
    equal(result.prediction, oracle(a, b));
  }
  equal(merged.encode(), before);
  await rejects(() => classifier.predict("-1", "2", 100));
  await rejects(() => classifier.predict("1", "2", NaN));
});

Deno.test("budget limits loops; paused execution resumes with identical result and work", async () => {
  const { merged, programId } = await fixture;
  const executable = new Executable(compile(merged), programId);
  const machine = executable.start([digits("999999999"), digits("3")], true);
  equal(machine.advance(0).steps, 0);
  equal(machine.advance(7).status, "budget");
  const resumed = machine.advance(1000);
  equal(
    resumed,
    executable.start([digits("999999999"), digits("3")], true).advance(1007),
  );
  equal(machine.advance(10), resumed);
  const loop: Payload = {
    kind: "program",
    name: "loop",
    code: [{ op: "jump", target: 0 }],
  };
  const state = await Knowledge.empty().add([loop]);
  const result = new Executable(compile(state), await idOf(loop)).start([])
    .advance(123);
  equal(result.status, "budget");
  equal(result.steps, 123);
});

Deno.test("unseen transitions and invalid register arithmetic cannot silently succeed", async () => {
  const { shared } = await fixture;
  const state = await shared.add([ADD_SPEC]);
  equal(
    new SumClassifier(compile(state)).predict("1", "2", 2048).status,
    "unseen-input",
  );
  const overflow: Payload = {
    kind: "program",
    name: "overflow",
    code: [{ op: "const", dst: 0, value: 2147483647 }, {
      op: "increment",
      register: 0,
      by: 1,
    }, { op: "emit", src: 0 }],
  };
  const invalid = await Knowledge.empty().add([overflow]);
  equal(
    new Executable(compile(invalid), await idOf(overflow)).start([]).advance(10)
      .status,
    "invalid-state",
  );
});

Deno.test("more internal work improves balanced held-out accuracy with one-character outputs", async () => {
  const { merged } = await fixture;
  const classifier = new SumClassifier(compile(merged)), dataset = cases(8);
  let previous = 0;
  for (const budget of [0, 32, 128, 512, 2048]) {
    let correct = 0;
    for (const c of dataset) {
      const result = classifier.predict(c.a, c.b, budget);
      equal(result.prediction.length, 1);
      assert(result.steps <= budget);
      correct += Number(result.prediction === c.expected);
      if (budget === 2048) assert(result.completed);
    }
    assert(correct >= previous);
    previous = correct;
    if (budget === 0) equal(correct, dataset.length / 2);
  }
  equal(previous, dataset.length);
});

Deno.test("changing learned evidence changes answers: inference has no hidden arithmetic oracle", async () => {
  const { merged, addId } = await fixture;
  equal(
    new SumClassifier(compile(merged)).predict("1", "2", 2048).prediction,
    "1",
  );
  const changed = await merged.add(
    [0, 1, 2].map((n): Payload => ({
      kind: "evidence",
      module: addId,
      event: `counterexample/${n}`,
      input: [1, 2, 0],
      output: [4, 0],
    })),
  );
  const result = new SumClassifier(compile(changed)).predict("1", "2", 2048);
  assert(result.completed);
  equal(result.prediction, "0");
});

Deno.test("unrelated experts do not increase runtime work", async () => {
  const { merged } = await fixture;
  const larger = await merged.add(
    Array.from(
      { length: 100 },
      (_, n): Payload => ({
        kind: "table",
        name: `extra/${n}`,
        port: { name: `unrelated/${n}`, inputs: [2], outputs: [2] },
      }),
    ),
  );
  const a = new SumClassifier(compile(merged)).predict(
    "99999",
    "4",
    2048,
    true,
  );
  const b = new SumClassifier(compile(larger)).predict(
    "99999",
    "4",
    2048,
    true,
  );
  equal(a, b);
  assert(compile(merged).tables.has(canonical(MOD3)));
});
