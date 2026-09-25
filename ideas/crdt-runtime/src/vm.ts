import { type DenseTable, type View } from "./compile.ts";
import {
  canonical,
  encodeRow,
  type Instruction,
  REGISTERS,
  VECTORS,
} from "./schema.ts";

export type Status =
  | "budget"
  | "halted"
  | "missing-capability"
  | "unseen-input"
  | "invalid-state";
export interface TraceCall {
  module: string;
  input: number[];
  output: number[];
}
export interface Result {
  status: Status;
  value?: number;
  steps: number;
  calls: number;
  trace: readonly TraceCall[];
}

/** Bind ports once; execution never searches the knowledge library. */
export class Executable {
  readonly code: readonly Instruction[];
  readonly bindings: readonly (DenseTable | undefined)[];
  constructor(view: View, program: string) {
    const spec = view.programs.get(program);
    if (!spec) throw new Error("Unknown program");
    this.code = spec.code;
    this.bindings = spec.code.map((i) =>
      i.op === "call" ? view.tables.get(canonical(i.port)) : undefined
    );
  }
  start(inputs: readonly (readonly number[])[], trace = false): Machine {
    return new Machine(this, inputs, trace);
  }
}

/** Mutable memory belongs to one request, never to replicated knowledge. */
export class Machine {
  readonly #registers = new Int32Array(REGISTERS);
  readonly #vectors: number[][];
  readonly #trace: TraceCall[] = [];
  #pc = 0;
  #steps = 0;
  #calls = 0;
  #status: Status = "budget";
  #value: number | undefined;

  constructor(
    readonly executable: Executable,
    inputs: readonly (readonly number[])[],
    readonly tracing = false,
  ) {
    if (
      inputs.length > VECTORS || inputs.some((v) =>
        v.length > 65536 ||
        v.some((x) =>
          !Number.isSafeInteger(x) || x < -2147483648 || x > 2147483647
        )
      )
    ) throw new Error("Invalid input vectors");
    this.#vectors = Array.from(
      { length: VECTORS },
      (_, i) => [...(inputs[i] ?? [])],
    );
  }

  /** Fuel counts VM instructions, not parsing, compilation, or output serialization. */
  advance(fuel: number): Result {
    if (!Number.isSafeInteger(fuel) || fuel < 0 || fuel > 10_000_000) {
      throw new Error("Budget must be an integer in 0..10000000");
    }
    const r = this.#registers, vectors = this.#vectors;
    for (let used = 0; used < fuel && this.#status === "budget"; used++) {
      const i = this.executable.code[this.#pc];
      if (!i) {
        this.#status = "invalid-state";
        break;
      }
      const table = this.executable.bindings[this.#pc];
      this.#steps++;
      this.#pc++;
      switch (i.op) {
        case "const":
          r[i.dst] = i.value;
          break;
        case "length":
          r[i.dst] = vectors[i.vector].length;
          break;
        case "max":
          r[i.dst] = Math.max(r[i.a], r[i.b]);
          break;
        case "get":
          r[i.dst] = vectors[i.vector][r[i.index]] ?? 0;
          break;
        case "push":
          if (vectors[i.vector].length >= 65536) this.#status = "invalid-state";
          else vectors[i.vector].push(r[i.src]);
          break;
        case "increment": {
          const value = r[i.register] + i.by;
          if (value < -2147483648 || value > 2147483647) {
            this.#status = "invalid-state";
          } else r[i.register] = value;
          break;
        }
        case "call": {
          this.#calls++;
          if (!table) {
            this.#status = "missing-capability";
            break;
          }
          // Read all arguments before writing any destination (carry/remainder alias).
          const input = i.args.map((register) => r[register]);
          const row = encodeRow(input, i.port.inputs);
          if (row === undefined || !table.present[row]) {
            this.#status = "unseen-input";
            break;
          }
          const offset = row * i.dst.length;
          for (let d = 0; d < i.dst.length; d++) {
            r[i.dst[d]] = table.values[offset + d];
          }
          if (this.tracing) {
            this.#trace.push({
              module: table.owners[row],
              input,
              output: i.dst.map((register) => r[register]),
            });
          }
          break;
        }
        case "jump":
          this.#pc = i.target;
          break;
        case "jump-lt":
          if (r[i.a] < r[i.b]) this.#pc = i.target;
          break;
        case "jump-nonzero":
          if (r[i.register] !== 0) this.#pc = i.target;
          break;
        case "jump-nonnegative":
          if (r[i.register] >= 0) this.#pc = i.target;
          break;
        case "emit":
          this.#value = r[i.src];
          this.#status = "halted";
          break;
      }
    }
    return {
      status: this.#status,
      value: this.#value,
      steps: this.#steps,
      calls: this.#calls,
      trace: [...this.#trace],
    };
  }
}
