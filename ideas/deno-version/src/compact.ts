// 观察等价压缩：计数逐字节完全相同的上下文，在训练数据上行为一致（通常是同一组位置的不同写法，
// 比如 [king]、[kin]['g']、[␣][k][i][n][g]），每组只留一个。
//
// 只删除"叶子"位置的上下文：
//   - 没有子上下文（否则子节点会失去匹配路径：子节点的内容里包含它的槽位）
//   - 不是任何类成员的先验（否则会改变别人的平滑）
// 删完以后父节点可能变成新的叶子，所以反复执行直到不动。
// 压缩只删节点、不改任何计数，所以 merge / unlearn 的性质不受影响。
import { compile } from "./compile.ts";
import { prune } from "./image.ts";
import { Id, Image, Node } from "./types.ts";

export function compact(
  img: Image,
  minTotal = 20,
  keep: "shallow" | "deep" = "deep",
): { img: Image; removed: number } {
  let removed = 0;
  for (;;) {
    const comp = compile(img);
    const hasKids = new Uint8Array(comp.ctx.length);
    const isPrior = new Uint8Array(comp.ctx.length);
    for (const c of comp.ctx) {
      if (c.tailIdx >= 0) hasKids[c.tailIdx] = 1;
      if (c.parentIdx >= 0) isPrior[c.parentIdx] = 1;
    }
    const groups = new Map<string, number[]>();
    comp.ctx.forEach((c, k) => {
      if (c.depth === 0 || c.total < minTotal) return;
      const key = c.counts.join(",");
      const g = groups.get(key);
      if (g) g.push(k);
      else groups.set(key, [k]);
    });
    const drop = new Set<Id>();
    for (const g of groups.values()) {
      if (g.length < 2) continue;
      const removable = g.filter((k) => !hasKids[k] && !isPrior[k]);
      const kept = g.filter((k) => hasKids[k] || isPrior[k]);
      if (!removable.length) continue;
      // 组里没有必须保留的，就留一个（平局按 id，保证确定性）。
      // 默认留最深的：浅层的词首片段上下文挂在根附近，每个位置都要被测试，删掉它们省下的步数最多。
      if (!kept.length) {
        removable.sort((x, y) =>
          (keep === "shallow" ? 1 : -1) *
            (comp.ctx[x].depth - comp.ctx[y].depth) ||
          (comp.ctx[x].id < comp.ctx[y].id ? -1 : 1)
        );
        removable.shift();
      }
      removable.forEach((k) => drop.add(comp.ctx[k].id));
    }
    if (!drop.size) break;
    removed += drop.size;
    const nodes = new Map<Id, Node>();
    const ctxCounts = new Map<Id, Int32Array>();
    for (const [id, n] of img.nodes) if (!drop.has(id)) nodes.set(id, n);
    for (const [id, c] of img.ctxCounts) {
      if (!drop.has(id)) ctxCounts.set(id, c);
    }
    img = prune({ ...img, nodes, ctxCounts });
  }
  return { img, removed };
}
