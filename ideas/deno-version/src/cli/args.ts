// 极简参数解析：--key value、--key=value、-o value、布尔开关、位置参数。
export interface Args {
  positional: string[];
  str(name: string, def?: string): string | undefined;
  num(name: string, def: number): number;
  bool(name: string): boolean;
}

export function parseArgs(argv: string[], booleans: string[] = []): Args {
  const flags = new Map<string, string>();
  const positional: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("-") || a === "-") {
      positional.push(a);
      continue;
    }
    const body = a.replace(/^--?/, "");
    const eq = body.indexOf("=");
    if (eq >= 0) flags.set(body.slice(0, eq), body.slice(eq + 1));
    else if (booleans.includes(body)) flags.set(body, "true");
    else {
      if (i + 1 >= argv.length) fail(`参数 ${a} 缺少取值`);
      flags.set(body, argv[++i]);
    }
  }
  return {
    positional,
    str: (n, d) => flags.get(n) ?? d,
    num: (n, d) => {
      const v = flags.get(n);
      if (v === undefined) return d;
      const x = Number(v);
      if (!Number.isFinite(x)) fail(`--${n} 需要数字，得到 ${v}`);
      return x;
    },
    bool: (n) => flags.get(n) === "true",
  };
}

export function fail(msg: string): never {
  console.error(msg);
  Deno.exit(1);
}
