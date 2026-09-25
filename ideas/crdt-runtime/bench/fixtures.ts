// Training/oracle code is intentionally separate from the inference interpreter.
import { ADD_SPEC, MOD_SPEC, PROGRAM } from "../src/example.ts";
import { Knowledge } from "../src/knowledge.ts";
import { idOf, type Payload } from "../src/schema.ts";

export async function trainReplicas() {
  const addId = await idOf(ADD_SPEC),
    modId = await idOf(MOD_SPEC),
    programId = await idOf(PROGRAM);
  const shared = await Knowledge.empty().add([PROGRAM]);
  const addition: Payload[] = [ADD_SPEC], modulo: Payload[] = [MOD_SPEC];
  for (let a = 0; a < 10; a++) {
    for (let b = 0; b < 10; b++) {
      for (let c = 0; c < 2; c++) {
        const sum = a + b + c;
        addition.push({
          kind: "evidence",
          module: addId,
          event: `addition/${a}/${b}/${c}`,
          input: [a, b, c],
          output: [sum % 10, Math.floor(sum / 10)],
        });
      }
    }
  }
  for (let r = 0; r < 3; r++) {
    for (let d = 0; d < 10; d++) {
      const next = (10 * r + d) % 3;
      modulo.push({
        kind: "evidence",
        module: modId,
        event: `modulo/${r}/${d}`,
        input: [r, d],
        output: [next, Number(next === 0)],
      });
    }
  }
  const a = await shared.add(addition), b = await shared.add(modulo);
  return { a, b, shared, merged: a.merge(b), addId, modId, programId };
}

export interface Case {
  a: string;
  b: string;
  expected: string;
  digits: number;
}
export function oracle(a: string, b: string): string {
  return (BigInt(a) + BigInt(b)) % 3n === 0n ? "1" : "0";
}

/** 50/50 labels within every input length, deterministic and never used for training. */
export function cases(perLabel = 64): Case[] {
  let seed = 0x5eed;
  const random = () => {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return seed >>> 0;
  };
  const result: Case[] = [];
  for (const length of [1, 2, 4, 8, 16, 32, 64, 128]) {
    const counts = [0, 0];
    while (counts[0] < perLabel || counts[1] < perLabel) {
      const number = () =>
        Array.from(
          { length },
          (_, i) => String(i === 0 ? 1 + random() % 9 : random() % 10),
        ).join("");
      const a = number(),
        b = number(),
        expected = oracle(a, b),
        label = Number(expected);
      if (counts[label] >= perLabel) continue;
      counts[label]++;
      result.push({ a, b, expected, digits: length });
    }
  }
  return result;
}
