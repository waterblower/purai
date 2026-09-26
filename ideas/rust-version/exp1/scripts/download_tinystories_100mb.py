"""Download a document-aligned 100 MB subset of the official training split."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
URL = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REVISION}/TinyStories-train.txt"
DEST = DATA / "tiny-story-100mb.jsonl"
TARGET = 100_000_000


def digest(text):
    return hashlib.sha256(text.strip().encode("utf-8")).digest()


def main():
    if DEST.exists():
        raise SystemExit(f"Refusing to overwrite {DEST}")
    validation = DATA / "validation.jsonl"
    excluded = set()
    for line in validation.read_text().splitlines():
        item = json.loads(line)
        excluded.add(digest(item if isinstance(item, str) else item["text"]))
    start = time.perf_counter()
    docs = size = downloaded = skipped = 0
    pending = b""
    marker = b"<|endoftext|>"
    temp = DEST.with_suffix(".jsonl.part")
    with urllib.request.urlopen(URL, timeout=120) as response, temp.open("w", encoding="utf-8") as out:
        while size < TARGET:
            chunk = response.read(1024 * 1024)
            if not chunk:
                raise RuntimeError("Source ended before the requested size")
            downloaded += len(chunk)
            parts = (pending + chunk).split(marker)
            pending = parts.pop()
            for raw in parts:
                text = raw.decode("utf-8").strip()
                if not text:
                    continue
                if digest(text) in excluded:
                    skipped += 1
                    continue
                out.write(json.dumps(text, ensure_ascii=False) + "\n")
                size += len(text.encode("utf-8"))
                docs += 1
                if size >= TARGET:
                    break
            print(f"{size:,} text bytes; {docs:,} stories", flush=True)
    temp.rename(DEST)
    metadata = {
        "source_url": URL, "revision": REVISION,
        "dataset": "roneneldan/TinyStories", "source_split": "train",
        "license": "cdla-sharing-1.0",
        "selection": "First complete nonempty stories, stripping boundary whitespace; exact normalized validation overlaps excluded",
        "target_decoded_bytes": TARGET, "decoded_bytes": size,
        "documents": docs, "downloaded_bytes": downloaded,
        "validation_overlap_documents_skipped": skipped,
        "file": str(DEST.relative_to(ROOT)), "file_bytes": DEST.stat().st_size,
        "sha256": hashlib.sha256(DEST.read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
        "download_seconds": time.perf_counter() - start,
    }
    DEST.with_suffix(".source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
