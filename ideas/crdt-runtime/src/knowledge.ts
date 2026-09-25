import {
  canonical,
  deepFreeze,
  type Entry,
  idOf,
  type Payload,
  validatePayload,
} from "./schema.ts";

/** Authoritative state: a grow-only set of immutable, content-addressed records.
 * Local updates are inflationary; the only merge operation is set union.
 * Indexes, majority votes and runtime memory are derived, not replicated state.
 */
export class Knowledge {
  readonly #records: ReadonlyMap<string, Entry>;

  private constructor(records: ReadonlyMap<string, Entry>) {
    this.#records = records;
  }
  static empty(): Knowledge {
    return new Knowledge(new Map());
  }
  get size(): number {
    return this.#records.size;
  }
  entries(): readonly Entry[] {
    return [...this.#records.values()].sort((a, b) => a.id.localeCompare(b.id));
  }

  async add(payloads: readonly Payload[]): Promise<Knowledge> {
    // Clone BEFORE awaiting: mutation by a caller cannot alter committed records.
    const copies = payloads.map((payload) => {
      validatePayload(payload);
      return deepFreeze(JSON.parse(canonical(payload)) as Payload);
    });
    const additions = await Promise.all(
      copies.map(async (payload) =>
        deepFreeze({ id: await idOf(payload), payload })
      ),
    );
    const records = new Map(this.#records);
    for (const entry of additions) records.set(entry.id, entry);
    return new Knowledge(records);
  }

  merge(other: Knowledge): Knowledge {
    const records = new Map(this.#records);
    for (const entry of other.#records.values()) {
      const old = records.get(entry.id);
      if (old && canonical(old.payload) !== canonical(entry.payload)) {
        throw new Error("Content hash collision");
      }
      records.set(entry.id, entry);
    }
    return new Knowledge(records);
  }

  encode(): string {
    return canonical({ version: 1, records: this.entries() }) + "\n";
  }

  static async decode(text: string): Promise<Knowledge> {
    const value = JSON.parse(text);
    if (
      !value || value.version !== 1 || !Array.isArray(value.records) ||
      Object.keys(value).sort().join(",") !== "records,version"
    ) throw new Error("Invalid state envelope");
    const records: Entry[] = value.records;
    for (const entry of records) {
      if (!entry || Object.keys(entry).sort().join(",") !== "id,payload") {
        throw new Error("Invalid entry");
      }
      validatePayload(entry.payload);
      if (entry.id !== await idOf(entry.payload)) {
        throw new Error("Record checksum mismatch");
      }
    }
    return await Knowledge.empty().add(records.map((entry) => entry.payload));
  }
}
