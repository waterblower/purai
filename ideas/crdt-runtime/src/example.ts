import { type View } from "./compile.ts";
import { canonical, type Payload, type Port } from "./schema.ts";
import { Executable } from "./vm.ts";

export const ADD: Port = {
  name: "digit.add",
  inputs: [10, 10, 2],
  outputs: [10, 2],
};
export const MOD3: Port = {
  name: "digit.mod3",
  inputs: [3, 10],
  outputs: [3, 2],
};
export const ADD_SPEC: Payload = {
  kind: "table",
  name: "addition expert",
  port: ADD,
};
export const MOD_SPEC: Payload = {
  kind: "table",
  name: "modulo expert",
  port: MOD3,
};

/** Supplied control flow; transition tables are learned separately from observations. */
export const PROGRAM: Extract<Payload, { kind: "program" }> = {
  kind: "program",
  name: "sum-divisible-by-three",
  code: [
    { op: "length", dst: 2, vector: 0 },
    { op: "length", dst: 3, vector: 1 },
    { op: "max", dst: 4, a: 2, b: 3 },
    { op: "const", dst: 0, value: 0 },
    { op: "const", dst: 1, value: 0 },
    { op: "get", dst: 5, vector: 0, index: 0 },
    { op: "get", dst: 6, vector: 1, index: 0 },
    { op: "call", port: ADD, args: [5, 6, 1], dst: [7, 1] },
    { op: "push", vector: 2, src: 7 },
    { op: "increment", register: 0, by: 1 },
    { op: "jump-lt", a: 0, b: 4, target: 5 },
    { op: "jump-nonzero", register: 1, target: 5 },
    { op: "length", dst: 0, vector: 2 },
    { op: "increment", register: 0, by: -1 },
    { op: "const", dst: 8, value: 0 },
    { op: "get", dst: 9, vector: 2, index: 0 },
    { op: "call", port: MOD3, args: [8, 9], dst: [8, 10] },
    { op: "increment", register: 0, by: -1 },
    { op: "jump-nonnegative", register: 0, target: 15 },
    { op: "emit", src: 10 },
  ],
};

export function digits(text: string): number[] {
  if (!/^[0-9]{1,4096}$/.test(text)) {
    throw new Error("Expected 1..4096 unsigned decimal digits");
  }
  return Array.from(text, (c) => c.charCodeAt(0) - 48).reverse();
}

export class SumClassifier {
  readonly executable: Executable;
  readonly fallback: "0" | "1";
  constructor(view: View) {
    const programs = [...view.programs].filter(([, p]) =>
      p.name === PROGRAM.name
    );
    if (programs.length !== 1) {
      throw new Error("Expected exactly one sum-divisibility program");
    }
    this.executable = new Executable(view, programs[0][0]);
    // Explicit modal-transition prior. This is a guess, never a completed computation.
    this.fallback = view.tables.get(canonical(MOD3))?.prior[1] === 1
      ? "1"
      : "0";
  }
  predict(a: string, b: string, budget: number, trace = false) {
    const result = this.executable.start([digits(a), digits(b)], trace).advance(
      budget,
    );
    const completed = result.status === "halted";
    if (completed && result.value !== 0 && result.value !== 1) {
      throw new Error("Non-binary program result");
    }
    return {
      ...result,
      completed,
      prediction: completed ? String(result.value) : this.fallback,
      source: completed ? "computed" : "prior",
    };
  }
}
