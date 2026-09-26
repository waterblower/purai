"""Run the preregistered pair sequentially; standard library only, macOS timing."""
import datetime
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
EXP = pathlib.Path(__file__).resolve().parent
BIN = EXP / "build/model/debug/growing-byte-model"

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def timed(command, directory, stem):
    command = [str(x) for x in command]
    with open(directory / (stem + ".log"), "w") as log:
        result = subprocess.run([sys.executable, str(EXP / "measure.py"), str(directory / (stem + ".run.json")), *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    record = json.loads((directory / (stem + ".run.json")).read_text())
    if result.returncode:
        raise RuntimeError(f"{stem} failed; see {directory}")
    return record

def main():
    with open(EXP / "model-build.log", "w") as log:
        subprocess.run(["cargo", "build", "--offline", "--locked", "--manifest-path", str(EXP / "source/Cargo.toml"), "--target-dir", str(EXP / "build/model")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    for size in ("100mb", "1gb"):
        directory = EXP / size
        directory.mkdir(exist_ok=True)
        model = directory / "output.model"
        command = [BIN, "train", "--input", f"data/wikipedia-20231101-en-{size}.jsonl", "--validation", "data/wikipedia-20231101-en-dev.jsonl", "--output", model,
                   "--rounds", "12", "--max-depth", "12", "--max-nodes", "250000", "--candidates", "100000", "--forks", "10000", "--merges", "64", "--min-count", "4", "--structure-weight", "0.25", "--merge-top", "16", "--discount", "0.75", "--theta", "2", "--baseline-order", "5", "--no-match"]
        if model.exists():
            rec = json.loads((directory / "training.run.json").read_text())
            if not rec.get("training_completed") or rec["command"] != [str(x) for x in command] or rec.get("model_sha256") != sha(model) or rec.get("binary_sha256") != sha(BIN):
                raise RuntimeError(f"refusing to overwrite or reuse incomplete/mismatched {model}")
        else:
            print(f"{utc()} training {size}", flush=True)
            rec = timed(command, directory, "training")
            rec["training_completed"] = True
            rec["model_sha256"] = sha(model)
            rec["binary_sha256"] = sha(BIN)
            rec["timing_scope"] = "Full binary train command: input load, graph learning, per-round dev evaluation, save/reload, final dev evaluations, fixed byte 5-gram baseline. Excludes download, prep, build, independent test, BLiMP."
            (directory / "training.run.json").write_text(json.dumps(rec, indent=2) + "\n")
        timed([BIN, "benchmark", "--model", model, "--input", "data/benchmarks/blimp/data", "--reference", "exp1/benchmarks/references/blimp-models-summary.jsonl", "--report", directory / "blimp.report.json", "--samples", directory / "blimp.samples.jsonl"], directory, "blimp")
        timed([BIN, "eval", "--model", model, "--input", "data/wikipedia-20231101-en-test.jsonl", "--no-match"], directory, "test")
        with open(directory / "sample.txt", "w") as out, open(directory / "inference.log", "w") as err:
            subprocess.run([str(BIN), "infer", "--model", str(model), "--prompt", "Once upon a time,", "--no-match", "--seed", "42", "--n", "400"], cwd=ROOT, stdout=out, stderr=err, check=True)
        print(f"{utc()} completed {size}", flush=True)

if __name__ == "__main__":
    main()
