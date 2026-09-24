// 模型文件格式（确定性：同一个镜像永远序列化成同样的字节）
//
//   "AFRN"  version:u16le  flags:u16le
//   params: discount theta kappa（千分之一，varint）
//   tables: mixBits mixN apmHit apmTot（各 256 个 varint）
//   node_count:varint
//   节点，按拓扑序（seq，然后 alt，然后 ctx；同类按 id 排序；ctx 按深度、id 排序）：
//     seq: 0  len  bytes...  freq
//     alt: 1  len  refs...
//     ctx: 2  len  refs...  nnz  (sym:u8  count:zigzag-varint)...
//   ref 编码：字节 -> sym*2，节点 -> 下标*2+1（只能引用前面的节点）
//   checksum: 8 字节（对前面所有内容做 hash64）
//
// 文件里不存 id：加载时按内容重新计算。
import {
  emptyTables,
  Id,
  Image,
  Node,
  nodeId,
  Params,
  Ref,
  TABLE_KEYS,
} from "./types.ts";
import { bytesToLatin1, hash64 } from "./hash.ts";

const MAGIC = [0x41, 0x46, 0x52, 0x4e]; // "AFRN"
const VERSION = 2;

class Writer {
  buf = new Uint8Array(1 << 16);
  n = 0;
  byte(b: number) {
    if (this.n === this.buf.length) {
      const nb = new Uint8Array(this.buf.length * 2);
      nb.set(this.buf);
      this.buf = nb;
    }
    this.buf[this.n++] = b;
  }
  varint(v: number) {
    if (v < 0 || !Number.isInteger(v)) throw new Error(`varint: ${v}`);
    while (v >= 0x80) {
      this.byte((v % 0x80) | 0x80);
      v = Math.floor(v / 0x80);
    }
    this.byte(v);
  }
  zigzag(v: number) {
    this.varint(v >= 0 ? v * 2 : -v * 2 - 1);
  }
  bytes() {
    return this.buf.subarray(0, this.n);
  }
}

class Reader {
  i = 0;
  constructor(readonly buf: Uint8Array) {}
  byte(): number {
    if (this.i >= this.buf.length) throw new Error("模型文件被截断");
    return this.buf[this.i++];
  }
  varint(): number {
    let v = 0, mul = 1;
    for (;;) {
      const b = this.byte();
      v += (b & 0x7f) * mul;
      if (b < 0x80) return v;
      mul *= 0x80;
    }
  }
  zigzag(): number {
    const v = this.varint();
    return v % 2 === 0 ? v / 2 : -(v + 1) / 2;
  }
}

function order(img: Image): Id[] {
  const rank = { seq: 0, alt: 1, ctx: 2 } as const;
  return [...img.nodes.keys()].sort((a, b) => {
    const na = img.nodes.get(a)!, nb = img.nodes.get(b)!;
    const da = rank[na.kind] - rank[nb.kind];
    if (da) return da;
    if (na.kind === "ctx" && nb.kind === "ctx") {
      const dd = na.slots.length - nb.slots.length;
      if (dd) return dd;
    }
    return a < b ? -1 : a > b ? 1 : 0;
  });
}

export function encode(img: Image): Uint8Array {
  const w = new Writer();
  MAGIC.forEach((b) => w.byte(b));
  w.byte(VERSION & 0xff);
  w.byte(VERSION >> 8);
  w.byte(0);
  w.byte(0);
  w.varint(Math.round(img.params.discount * 1000));
  w.varint(Math.round(img.params.theta * 1000));
  w.varint(Math.round(img.params.kappa * 1000));
  for (const k of TABLE_KEYS) img.tables[k].forEach((v) => w.varint(v));
  const ids = order(img);
  const index = new Map<Id, number>();
  ids.forEach((id, k) => index.set(id, k));
  const writeRef = (r: Ref) => {
    if (r.tag === "term") return w.varint(r.sym * 2);
    const k = index.get(r.id);
    if (k === undefined) throw new Error(`encode: 悬空引用 ${r.id}`);
    w.varint(k * 2 + 1);
  };
  w.varint(ids.length);
  for (const id of ids) {
    const n = img.nodes.get(id)!;
    switch (n.kind) {
      case "seq":
        w.byte(0);
        w.varint(n.bytes.length);
        n.bytes.forEach((b) => w.byte(b));
        w.varint(img.seqFreq.get(id) ?? 0);
        break;
      case "alt":
        w.byte(1);
        w.varint(n.items.length);
        n.items.forEach(writeRef);
        break;
      case "ctx": {
        w.byte(2);
        w.varint(n.slots.length);
        n.slots.forEach(writeRef);
        const c = img.ctxCounts.get(id) ?? new Int32Array(256);
        let nnz = 0;
        for (let x = 0; x < 256; x++) if (c[x] !== 0) nnz++;
        w.varint(nnz);
        for (let x = 0; x < 256; x++) {
          if (c[x] !== 0) {
            w.byte(x);
            w.zigzag(c[x]);
          }
        }
        break;
      }
    }
  }
  const body = w.bytes().slice();
  const sum = hash64(bytesToLatin1(body));
  const out = new Uint8Array(body.length + 8);
  out.set(body);
  for (let k = 0; k < 8; k++) {
    out[body.length + k] = parseInt(sum.slice(k * 2, k * 2 + 2), 16);
  }
  return out;
}

export function decode(buf: Uint8Array): Image {
  if (buf.length < 16) throw new Error("不是 AFRN 模型文件");
  for (let k = 0; k < 4; k++) {
    if (buf[k] !== MAGIC[k]) throw new Error("不是 AFRN 模型文件");
  }
  const body = buf.subarray(0, buf.length - 8);
  const sum = hash64(bytesToLatin1(body));
  for (let k = 0; k < 8; k++) {
    if (buf[body.length + k] !== parseInt(sum.slice(k * 2, k * 2 + 2), 16)) {
      throw new Error("模型文件校验和不匹配");
    }
  }
  const r = new Reader(body);
  r.i = 4;
  const version = r.byte() | (r.byte() << 8);
  if (version !== VERSION) throw new Error(`不支持的版本 ${version}`);
  r.byte();
  r.byte();
  const params: Params = {
    discount: r.varint() / 1000,
    theta: r.varint() / 1000,
    kappa: r.varint() / 1000,
  };
  const tables = emptyTables();
  for (const k of TABLE_KEYS) {
    for (let q = 0; q < tables[k].length; q++) tables[k][q] = r.varint();
  }
  const count = r.varint();
  const ids: Id[] = [];
  const nodes = new Map<Id, Node>();
  const ctxCounts = new Map<Id, Int32Array>();
  const seqFreq = new Map<Id, number>();
  const readRef = (): Ref => {
    const v = r.varint();
    if (v % 2 === 0) return { tag: "term", sym: v / 2 };
    const k = (v - 1) / 2;
    if (k >= ids.length) throw new Error("引用了后面的节点：不是拓扑序");
    return { tag: "ref", id: ids[k] };
  };
  for (let k = 0; k < count; k++) {
    const tag = r.byte();
    let n: Node;
    let counts: Int32Array | undefined, freq: number | undefined;
    if (tag === 0) {
      const len = r.varint();
      const bytes: number[] = [];
      for (let q = 0; q < len; q++) bytes.push(r.byte());
      n = { kind: "seq", bytes };
      freq = r.varint();
    } else if (tag === 1) {
      const len = r.varint();
      const items: Ref[] = [];
      for (let q = 0; q < len; q++) items.push(readRef());
      n = { kind: "alt", items };
    } else if (tag === 2) {
      const len = r.varint();
      const slots: Ref[] = [];
      for (let q = 0; q < len; q++) slots.push(readRef());
      n = { kind: "ctx", slots };
      counts = new Int32Array(256);
      const nnz = r.varint();
      for (let q = 0; q < nnz; q++) {
        const x = r.byte();
        counts[x] = r.zigzag();
      }
    } else throw new Error(`未知节点类型 ${tag}`);
    const id = nodeId(n);
    if (nodes.has(id)) throw new Error("重复节点");
    ids.push(id);
    nodes.set(id, n);
    if (counts) ctxCounts.set(id, counts);
    if (freq !== undefined) seqFreq.set(id, freq);
  }
  if (r.i !== body.length) throw new Error("模型文件末尾有多余数据");
  return { nodes, ctxCounts, seqFreq, params, tables };
}

export async function save(path: string, img: Image): Promise<number> {
  const bytes = encode(img);
  await Deno.writeFile(path, bytes);
  return bytes.length;
}

export async function load(path: string): Promise<Image> {
  return decode(await Deno.readFile(path));
}

// 仅用于测试：镜像内容是否完全相同
export function sameImage(a: Image, b: Image): boolean {
  const ea = encode(a), eb = encode(b);
  return ea.length === eb.length && ea.every((x, i) => x === eb[i]);
}
