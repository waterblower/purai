import { Knowledge } from "./knowledge.ts";
import {
  canonical,
  encodeRow,
  type Payload,
  type Port,
  product,
} from "./schema.ts";

type TableSpec = Extract<Payload, { kind: "table" }>;
type ProgramSpec = Extract<Payload, { kind: "program" }>;
type Evidence = Extract<Payload, { kind: "evidence" }>;
interface Vote {
  output: readonly number[];
  count: number;
}
interface Expert {
  id: string;
  spec: TableSpec;
  rows: Map<number, Map<string, Vote>>;
  votes: number;
}

export class DenseTable {
  readonly values: Int32Array;
  readonly present: Uint8Array;
  readonly owners: string[];
  readonly prior: readonly number[];
  constructor(readonly port: Port, prior: readonly number[]) {
    const rows = product(port.inputs);
    this.values = new Int32Array(rows * port.outputs.length);
    this.present = new Uint8Array(rows);
    this.owners = new Array(rows).fill("");
    this.prior = [...prior];
  }
  get typedArrayBytes(): number {
    return this.values.byteLength + this.present.byteLength;
  }
}

export interface View {
  readonly tables: ReadonlyMap<string, DenseTable>;
  readonly programs: ReadonlyMap<string, ProgramSpec>;
  readonly stats: {
    records: number;
    experts: number;
    ports: number;
    programs: number;
    acceptedEvidence: number;
    conflictingEvents: number;
    deferredEvidence: number;
    invalidEvidence: number;
    routedRows: number;
    typedArrayBytes: number;
  };
}

function winner(votes: Map<string, Vote>): Vote {
  return [...votes.entries()].sort((a, b) =>
    b[1].count - a[1].count || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0)
  )[0][1];
}

/** Pure deterministic compilation. No mutation, pruning or normalization of the G-set. */
export function compile(knowledge: Knowledge): View {
  const experts = new Map<string, Expert>();
  const programs = new Map<string, ProgramSpec>();
  const events = new Map<string, Evidence[]>();
  for (const { id, payload } of knowledge.entries()) {
    if (payload.kind === "table") {
      experts.set(id, { id, spec: payload, rows: new Map(), votes: 0 });
    } else if (payload.kind === "program") programs.set(id, payload);
    else {
      const key = canonical([payload.module, payload.event]);
      const group = events.get(key) ?? [];
      group.push(payload);
      events.set(key, group);
    }
  }
  const stats = {
    records: knowledge.size,
    experts: experts.size,
    ports: 0,
    programs: programs.size,
    acceptedEvidence: 0,
    conflictingEvents: 0,
    deferredEvidence: 0,
    invalidEvidence: 0,
    routedRows: 0,
    typedArrayBytes: 0,
  };
  for (const group of events.values()) {
    // Same logical observation with contradictory contents is retained in state,
    // but gets NO vote. It must not become two independent observations.
    if (group.length !== 1) {
      stats.conflictingEvents++;
      continue;
    }
    const e = group[0];
    const expert = experts.get(e.module);
    if (!expert) {
      stats.deferredEvidence++;
      continue;
    }
    const row = encodeRow(e.input, expert.spec.port.inputs);
    if (
      row === undefined ||
      encodeRow(e.output, expert.spec.port.outputs) === undefined
    ) {
      stats.invalidEvidence++;
      continue;
    }
    const votes = expert.rows.get(row) ?? new Map<string, Vote>();
    const key = canonical(e.output), old = votes.get(key);
    votes.set(key, { output: e.output, count: (old?.count ?? 0) + 1 });
    expert.rows.set(row, votes);
    expert.votes++;
    stats.acceptedEvidence++;
  }
  const groups = new Map<string, Expert[]>();
  for (const expert of experts.values()) {
    const key = canonical(expert.spec.port), group = groups.get(key) ?? [];
    group.push(expert);
    groups.set(key, group);
  }
  const tables = new Map<string, DenseTable>();
  for (const [key, candidates] of groups) {
    // The prior is a data-derived fallback from the most-supported expert.
    candidates.sort((a, b) => b.votes - a.votes || (a.id < b.id ? -1 : 1));
    const port = candidates[0].spec.port;
    const marginal = new Map<string, Vote>();
    for (const votes of candidates[0].rows.values()) {
      for (const [label, vote] of votes) {
        const old = marginal.get(label);
        marginal.set(label, {
          output: vote.output,
          count: (old?.count ?? 0) + vote.count,
        });
      }
    }
    const prior = marginal.size
      ? winner(marginal).output
      : port.outputs.map(() => 0);
    const table = new DenseTable(port, prior);
    const selected = new Map<
      number,
      { numerator: bigint; denominator: bigint; id: string }
    >();
    for (const expert of candidates) {
      for (const [row, votes] of expert.rows) {
        const best = winner(votes);
        const total = [...votes.values()].reduce(
          (n, vote) => n + vote.count,
          0,
        );
        // Laplace-smoothed confidence, compared exactly during compilation only.
        const numerator = BigInt(best.count + 1),
          denominator = BigInt(total + product(port.outputs));
        const old = selected.get(row);
        const comparison = old
          ? numerator * old.denominator - old.numerator * denominator
          : 1n;
        if (
          !old || comparison > 0n || (comparison === 0n && expert.id < old.id)
        ) {
          selected.set(row, { numerator, denominator, id: expert.id });
          table.present[row] = 1;
          table.values.set(best.output, row * port.outputs.length);
          table.owners[row] = expert.id;
        }
      }
    }
    stats.routedRows += selected.size;
    stats.typedArrayBytes += table.typedArrayBytes;
    tables.set(key, table);
  }
  stats.ports = tables.size;
  return { tables, programs, stats };
}
