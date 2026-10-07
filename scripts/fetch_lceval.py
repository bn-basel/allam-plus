"""Download HUMAIN's LC-Eval (one 1.9 GB JSONL) as a stream and keep only the Arabic items whose text is short
enough to matter for this project (contexts up to ~32K ALLaM tokens). The full file never touches the disk.

    uv run python scripts/fetch_lceval.py

Output: data/raw/lceval/lceval_ar.jsonl (+ a small stats.json). Exact token counts are taken later with ALLaM's
tokenizer; here a character limit is used (ALLaM: ~3 characters of Arabic per token, so 120K characters ~ 40K tokens).
"""

import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

URL = "https://huggingface.co/datasets/humain-ai/LC-Eval/resolve/main/LC-Eval_evaluation_datasets.jsonl"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "lceval"
MAX_CHARS = 120_000


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seen, kept = Counter(), Counter()
    n_bytes = 0
    with urllib.request.urlopen(URL, timeout=120) as r, open(OUT / "lceval_ar.part", "w", encoding="utf-8") as f:
        for i, line in enumerate(r):
            n_bytes += len(line)
            item = json.loads(line)
            key = (item.get("language"), item.get("task"))
            seen[key] += 1
            size = len(item.get("full_context") or "") + len(item.get("question") or "")
            if (item.get("language") or "").lower() == "arabic" and size <= MAX_CHARS:
                item["chars"] = size
                item["row"] = i
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept[key] += 1
            if i % 500 == 0:
                print(f"{i} rows, {n_bytes / 1e9:.2f} GB read, {sum(kept.values())} kept", flush=True)
    (OUT / "lceval_ar.part").replace(OUT / "lceval_ar.jsonl")
    stats = {"bytes_read": n_bytes, "max_chars": MAX_CHARS,
             "seen": {f"{k[0]}|{k[1]}": v for k, v in seen.items()},
             "kept": {f"{k[0]}|{k[1]}": v for k, v in kept.items()}}
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
