"""Download a reproducible, whole-story TinyStories prefix (standard library only)."""
import hashlib
import json
from pathlib import Path
import urllib.request

OUT = Path(__file__).resolve().parent / "data" / "tinystories10mb"
REPO = "https://huggingface.co/datasets/roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
DELIMITER = b"<|endoftext|>"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    revision = REVISION
    seen = set()
    manifest = {"source": REPO, "revision": revision,
                "sampling": "First unique complete stories in file order; UTF-8; strip outer whitespace; remove delimiter; exact-text SHA-256 dedup within/across splits.",
                "splits": {}}
    for split, filename, target in [
        ("train", "TinyStories-train.txt", 10_000_000),
        ("validation", "TinyStories-valid.txt", 1_000_000),
    ]:
        url = f"{REPO}/resolve/{revision}/{filename}"
        stories, size, skipped, buffer = [], 0, 0, b""
        print(f"Downloading {split}: target {target:,} UTF-8 bytes", flush=True)
        # Only consume the prefix needed, even if the server ignores Range.
        request = urllib.request.Request(url, headers={"Range": f"bytes=0-{target * 3}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            while size < target:
                chunk = response.read(65536)
                if not chunk:
                    raise RuntimeError("Downloaded prefix did not contain enough unique stories")
                buffer += chunk
                while DELIMITER in buffer and size < target:
                    raw, buffer = buffer.split(DELIMITER, 1)
                    story = raw.decode("utf-8").strip()
                    encoded = story.encode("utf-8")
                    sha = digest(encoded)
                    if not story or sha in seen:
                        skipped += 1
                        continue
                    seen.add(sha)
                    stories.append(story)
                    size += len(encoded)
        payload = "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in stories).encode("utf-8")
        path = OUT / f"{split}.jsonl"
        path.write_bytes(payload)
        manifest["splits"][split] = {"url": url, "stories": len(stories),
            "bytes": size, "target_bytes": target, "skipped_duplicates_or_empty": skipped,
            "file_sha256": digest(payload)}
        print(f"Saved {len(stories):,} stories, {size:,} bytes", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
