import { compile } from "./compile.ts";
import { SumClassifier } from "./example.ts";
import { Knowledge } from "./knowledge.ts";

async function main(args: string[]) {
  const [command, ...rest] = args;
  if (command === "run") {
    const options = new Map<string, string>();
    if (rest.length % 2) throw new Error("Options need values");
    for (let i = 0; i < rest.length; i += 2) {
      if (
        !["--model", "--a", "--b", "--budget"].includes(rest[i]) ||
        options.has(rest[i])
      ) throw new Error("Unknown or repeated option");
      options.set(rest[i], rest[i + 1]);
    }
    const file = options.get("--model"),
      a = options.get("--a"),
      b = options.get("--b");
    if (!file || a === undefined || b === undefined) {
      throw new Error(
        "run --model FILE --a DECIMAL --b DECIMAL [--budget 2048]",
      );
    }
    const knowledge = await Knowledge.decode(await Deno.readTextFile(file));
    const classifier = new SumClassifier(compile(knowledge));
    console.log(
      JSON.stringify(
        classifier.predict(a, b, Number(options.get("--budget") ?? "2048")),
        null,
        2,
      ),
    );
  } else if (command === "merge") {
    const outputAt = rest.indexOf("-o");
    if (outputAt < 2 || outputAt !== rest.length - 2) {
      throw new Error("merge FILE FILE [FILE ...] -o OUTPUT");
    }
    let merged = Knowledge.empty();
    for (const path of rest.slice(0, outputAt)) {
      merged = merged.merge(
        await Knowledge.decode(await Deno.readTextFile(path)),
      );
    }
    // Read and validate all input states before writing output.
    await Deno.writeTextFile(rest[outputAt + 1], merged.encode());
    console.log(
      JSON.stringify({ output: rest[outputAt + 1], records: merged.size }),
    );
  } else throw new Error("Usage: run or merge (see README.md)");
}

if (import.meta.main) {
  try {
    await main(Deno.args);
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    Deno.exitCode = 1;
  }
}
