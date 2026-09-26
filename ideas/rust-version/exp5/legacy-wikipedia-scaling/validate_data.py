"""Independent Python verification of Rust preprocessing and split isolation."""
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
EXP = pathlib.Path(__file__).resolve().parent

def main():
    prepared = json.loads((EXP / "data-preparation.json").read_text())
    sets, orders, results, checksums = {}, {}, [], {}
    for expected in prepared["splits"]:
        path = ROOT / "data" / expected["file"]
        corpus_hash = hashlib.sha256()
        seen, order, total = set(), [], 0
        with path.open("rb") as f:
            for line in f:
                corpus_hash.update(line)
                text = json.loads(line)
                assert isinstance(text, str) and text and text == text.strip()
                data = text.encode("utf-8")
                h = hashlib.sha256(data).digest()
                assert h not in seen, "duplicate document"
                bucket = int.from_bytes(h[:8], "big") % 100
                required = 0 if "-dev." in path.name else 1 if "-test." in path.name else None
                assert bucket == required if required is not None else bucket >= 2
                seen.add(h); order.append(h); total += len(data)
        assert total == expected["decoded_bytes"] and len(seen) == expected["documents"]
        assert total >= expected["target_bytes"]
        sets[path.name], orders[path.name] = seen, order
        checksums[str(path.relative_to(ROOT))] = corpus_hash.hexdigest()
        results.append(dict(file=path.name, decoded_bytes=total, documents=len(seen), file_bytes=path.stat().st_size, sha256=corpus_hash.hexdigest()))
    prefix = "wikipedia-20231101-en-"
    train, small, dev, test = [prefix + name + ".jsonl" for name in ("1gb", "100mb", "dev", "test")]
    assert orders[train][:len(orders[small])] == orders[small]
    assert not sets[train] & sets[dev] and not sets[train] & sets[test] and not sets[dev] & sets[test]
    sources = list((ROOT / "data/wikipedia-20231101-en-source").glob("*.parquet"))
    sources += list((EXP / "source").rglob("*.rs")) + [EXP / "source/Cargo.toml", EXP / "source/Cargo.lock"]
    sources += list((ROOT / "data/benchmarks/blimp/data").glob("*.jsonl"))
    for path in sources:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
        checksums[str(path.relative_to(ROOT))] = h.hexdigest()
    (EXP / "sha256.json").write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
    result = dict(verified=True, splits=results, nested_100mb_prefix=True, exact_train_dev_test_overlap=0,
                  limitations="No near-duplicate or benchmark contamination audit; source-prefix sampling.")
    (EXP / "data-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
