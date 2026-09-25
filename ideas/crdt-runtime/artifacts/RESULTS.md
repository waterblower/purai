# CPU / CRDT / internal-computation prototype

Generated: 2026-09-25T04:50:05.474Z. Deno 2.9.6, aarch64, darwin.

Training: 200 digit-addition transitions on replica A; 30 remainder transitions on replica B. Program control flow is handwritten. No full-task examples are trained. Evaluation: 1024 balanced queries, 1–128 digits per operand. The 1-digit evaluation overlaps primitive inputs; longer compositions are unseen.

| VM budget | Accuracy | Completed | Completed accuracy | Mean steps | Prediction length |
|---:|---:|---:|---:|---:|---:|
| 8 | 50.00% | 0.00% | n/a | 8.00 | 1 character |
| 32 | 58.50% | 17.48% | 100.00% | 31.11 | 1 character |
| 128 | 75.00% | 50.00% | 100.00% | 90.93 | 1 character |
| 512 | 87.50% | 75.00% | 100.00% | 219.08 | 1 character |
| 2048 | 100.00% | 100.00% | 100.00% | 335.45 | 1 character |

Incomplete computations return an explicitly marked constant prior guess. Completed accuracy refers only to computations reaching emit. This toy task demonstrates a compute/output separation, not general reasoning or language generation.

Replica A and replica B each stop with a missing capability. The merged model completes the composition without extra training. Repeated merge preserves the exact snapshot: true.

CPU (warm, sequential, 5120 queries): 69411 queries/s; p50 7.21 µs; p95 51.67 µs. Total VM query wall time 73.76 ms; native BigInt reference 0.92 ms. This measures warm inference, including input parsing and workspace construction, excluding model loading/compilation. Single-run measurements vary with JIT, GC and system load; this is not an LLM comparison.

Authoritative snapshot: 55526 bytes / 233 records. Compiled typed arrays: 2070 bytes (not total memory). Training/build 9.12 ms; compilation 13.63 ms. The append-only evidence set grows with observations; bounded-storage synchronization and program learning remain unimplemented.
