/** Finite categorical ports. Each number is a domain cardinality: values are 0..n-1. */
export interface Port {
  readonly name: string;
  readonly inputs: readonly number[];
  readonly outputs: readonly number[];
}

export type Instruction =
  | { readonly op: "const"; readonly dst: number; readonly value: number }
  | { readonly op: "length"; readonly dst: number; readonly vector: number }
  | {
    readonly op: "max";
    readonly dst: number;
    readonly a: number;
    readonly b: number;
  }
  | {
    readonly op: "get";
    readonly dst: number;
    readonly vector: number;
    readonly index: number;
  }
  | { readonly op: "push"; readonly vector: number; readonly src: number }
  | { readonly op: "increment"; readonly register: number; readonly by: number }
  | {
    readonly op: "call";
    readonly port: Port;
    readonly args: readonly number[];
    readonly dst: readonly number[];
  }
  | { readonly op: "jump"; readonly target: number }
  | {
    readonly op: "jump-lt";
    readonly a: number;
    readonly b: number;
    readonly target: number;
  }
  | {
    readonly op: "jump-nonzero";
    readonly register: number;
    readonly target: number;
  }
  | {
    readonly op: "jump-nonnegative";
    readonly register: number;
    readonly target: number;
  }
  | { readonly op: "emit"; readonly src: number };

export type Payload =
  | { readonly kind: "table"; readonly name: string; readonly port: Port }
  | {
    readonly kind: "program";
    readonly name: string;
    readonly code: readonly Instruction[];
  }
  | {
    readonly kind: "evidence";
    readonly module: string;
    readonly event: string;
    readonly input: readonly number[];
    readonly output: readonly number[];
  };

export interface Entry {
  readonly id: string;
  readonly payload: Payload;
}

export const REGISTERS = 16;
export const VECTORS = 4;
export const MAX_TABLE_ROWS = 65536;

/** Canonical JSON, also used for deterministic tie-breaking and wire encoding. */
export function canonical(value: unknown): string {
  if (
    value === null || typeof value === "boolean" || typeof value === "string"
  ) return JSON.stringify(value);
  if (typeof value === "number" && Number.isFinite(value)) {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (typeof value === "object" && value !== null) {
    const obj = value as Record<string, unknown>;
    return `{${
      Object.keys(obj).sort().map((key) =>
        `${JSON.stringify(key)}:${canonical(obj[key])}`
      ).join(",")
    }}`;
  }
  throw new Error("Not canonical JSON");
}

function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Expected object");
  }
  return value as Record<string, unknown>;
}
function keys(o: Record<string, unknown>, expected: string[]) {
  if (Object.keys(o).sort().join(",") !== expected.sort().join(",")) {
    throw new Error("Unexpected or missing fields");
  }
}
function integer(
  value: unknown,
  min: number,
  max: number,
): asserts value is number {
  if (
    typeof value !== "number" || !Number.isSafeInteger(value) || value < min ||
    value > max
  ) throw new Error("Invalid integer");
}
function name(value: unknown): asserts value is string {
  if (typeof value !== "string" || value.length < 1 || value.length > 256) {
    throw new Error("Invalid name");
  }
}
function numbers(
  value: unknown,
  min: number,
  max: number,
): asserts value is number[] {
  if (!Array.isArray(value) || !value.length || value.length > 8) {
    throw new Error("Expected 1..8 numbers");
  }
  for (const n of value) integer(n, min, max);
}
export const product = (values: readonly number[]) =>
  values.reduce((a, b) => a * b, 1);

export function validatePort(value: unknown): asserts value is Port {
  const p = object(value);
  keys(p, ["name", "inputs", "outputs"]);
  name(p.name);
  numbers(p.inputs, 1, MAX_TABLE_ROWS);
  numbers(p.outputs, 1, MAX_TABLE_ROWS);
  if (
    product(p.inputs) > MAX_TABLE_ROWS || product(p.outputs) > MAX_TABLE_ROWS
  ) throw new Error("Port domain too large");
}

export function validatePayload(value: unknown): asserts value is Payload {
  const p = object(value);
  if (p.kind === "table") {
    keys(p, ["kind", "name", "port"]);
    name(p.name);
    validatePort(p.port);
  } else if (p.kind === "evidence") {
    keys(p, ["kind", "module", "event", "input", "output"]);
    if (typeof p.module !== "string" || !/^[a-f0-9]{64}$/.test(p.module)) {
      throw new Error("Invalid module id");
    }
    name(p.event);
    numbers(p.input, 0, 2147483647);
    numbers(p.output, 0, 2147483647);
  } else if (p.kind === "program") {
    keys(p, ["kind", "name", "code"]);
    name(p.name);
    if (!Array.isArray(p.code) || !p.code.length || p.code.length > 4096) {
      throw new Error("Invalid code length");
    }
    const reg = (x: unknown) => integer(x, 0, REGISTERS - 1);
    const vec = (x: unknown) => integer(x, 0, VECTORS - 1);
    const target = (x: unknown) =>
      integer(x, 0, (p.code as unknown[]).length - 1);
    for (const raw of p.code) {
      const i = object(raw);
      switch (i.op) {
        case "const":
          keys(i, ["op", "dst", "value"]);
          reg(i.dst);
          integer(i.value, -2147483648, 2147483647);
          break;
        case "length":
          keys(i, ["op", "dst", "vector"]);
          reg(i.dst);
          vec(i.vector);
          break;
        case "max":
          keys(i, ["op", "dst", "a", "b"]);
          reg(i.dst);
          reg(i.a);
          reg(i.b);
          break;
        case "get":
          keys(i, ["op", "dst", "vector", "index"]);
          reg(i.dst);
          vec(i.vector);
          reg(i.index);
          break;
        case "push":
          keys(i, ["op", "vector", "src"]);
          vec(i.vector);
          reg(i.src);
          break;
        case "increment":
          keys(i, ["op", "register", "by"]);
          reg(i.register);
          integer(i.by, -2147483648, 2147483647);
          break;
        case "call":
          keys(i, ["op", "port", "args", "dst"]);
          validatePort(i.port);
          numbers(i.args, 0, REGISTERS - 1);
          numbers(i.dst, 0, REGISTERS - 1);
          if (
            i.args.length !== i.port.inputs.length ||
            i.dst.length !== i.port.outputs.length ||
            new Set(i.dst).size !== i.dst.length
          ) throw new Error("Call arity or destination alias mismatch");
          break;
        case "jump":
          keys(i, ["op", "target"]);
          target(i.target);
          break;
        case "jump-lt":
          keys(i, ["op", "a", "b", "target"]);
          reg(i.a);
          reg(i.b);
          target(i.target);
          break;
        case "jump-nonzero":
        case "jump-nonnegative":
          keys(i, ["op", "register", "target"]);
          reg(i.register);
          target(i.target);
          break;
        case "emit":
          keys(i, ["op", "src"]);
          reg(i.src);
          break;
        default:
          throw new Error("Unknown opcode");
      }
    }
  } else throw new Error("Unknown record kind");
}

export function deepFreeze<T>(value: T): T {
  if (value && typeof value === "object") {
    for (const child of Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

export async function idOf(payload: Payload): Promise<string> {
  validatePayload(payload);
  const hash = new Uint8Array(
    await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(canonical(payload)),
    ),
  );
  return [...hash].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export function encodeRow(
  values: readonly number[],
  domains: readonly number[],
): number | undefined {
  if (values.length !== domains.length) return undefined;
  let row = 0;
  for (let i = 0; i < values.length; i++) {
    if (
      !Number.isSafeInteger(values[i]) || values[i] < 0 ||
      values[i] >= domains[i]
    ) return undefined;
    row = row * domains[i] + values[i];
  }
  return row;
}
