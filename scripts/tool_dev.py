"""Score Phase 3 checkpoints on the held-out tool conversations (data/tools/dev.jsonl).

    uv run python scripts/tool_dev.py --adapter runs/tools_v1/checkpoint-100
    uv run python scripts/tool_dev.py --adapter runs/long_v1/checkpoint-183 --name start   # the starting point
    uv run python scripts/tool_dev.py --report tools_v1

Every assistant turn of every dev conversation is asked at its point in the conversation (earlier turns as written),
greedy, up to 200 new tokens. A turn is right when:
- the reference calls tools: the reply's tool calls equal the reference calls (names and arguments after
  normalisation, order ignored);
- the reference replies in text: the reply makes no tool call.
Checkpoint rule, fixed before training (docs/DECISIONS.md D16): the highest share of right turns; on a tie, the
earlier checkpoint. The benchmarks are run only on the chosen one.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402

DEV = ROOT / "data" / "tools" / "dev.jsonl"
OUT = ROOT / "results" / "tool_dev"


def call_key(calls: list[dict]) -> list:
    from tools_bench import norm_value
    keys = []
    for c in calls:
        args = c.get("arguments")
        args = {k: norm_value(v) for k, v in args.items() if v not in (None, "", [])} if isinstance(args, dict) else args
        keys.append(json.dumps([c.get("name"), args], ensure_ascii=False, sort_keys=True, default=str))
    return sorted(keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter")
    ap.add_argument("--name")
    ap.add_argument("--rope", default="yarn4")
    ap.add_argument("--report")
    args = ap.parse_args()
    if args.report:
        return report(args.report)

    from common import generate, load_model
    from toolformat import parse

    name = args.name or args.adapter.replace("\\", "/").replace("runs/", "").replace("/", "__")
    tok, model = load_model(args.rope, args.adapter)
    rows = []
    for ex in map(json.loads, open(DEV, encoding="utf-8")):
        msgs = ex["messages"]
        for j, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            prompt = tok.apply_chat_template(msgs[:j], tokenize=False, add_generation_prompt=True)
            g = generate(tok, model, prompt, 200)
            _, ref_calls = parse(m["content"])
            _, got_calls = parse(g["text"])
            right = (call_key(got_calls) == call_key(ref_calls)) if ref_calls else not got_calls
            rows.append({"id": ex["id"], "source": ex["source"], "scenario": ex.get("scenario"), "turn": j,
                         "ref_calls": bool(ref_calls), "right": right, "reply": g["text"][:400]})
    by = defaultdict(list)
    for r in rows:
        by[r["source"]].append(r["right"])
        by["calls" if r["ref_calls"] else "replies"].append(r["right"])
    s = {"name": name, "turns": len(rows), "right": round(sum(r["right"] for r in rows) / len(rows), 4),
         **{f"right_{k}": round(sum(v) / len(v), 4) for k, v in sorted(by.items())}, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in s.items() if k != "rows"}, indent=1))


def report(run: str) -> None:
    files = sorted(OUT.glob(f"{run}__checkpoint-*.json"), key=lambda p: int(p.stem.split("-")[-1]))
    start = OUT / "start.json"
    rows = ([json.loads(start.read_text(encoding="utf-8"))] if start.exists() else []) + \
           [json.loads(p.read_text(encoding="utf-8")) for p in files]
    keys = [k for k in rows[0] if k.startswith("right")] if rows else []
    print("| checkpoint | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys))
    for r in rows:
        print(f"| {r['name'].split('__')[-1]} | " + " | ".join(f"{r.get(k, 0):.3f}" for k in keys) + " |")
    cands = [r for r in rows if "checkpoint" in r["name"]]
    if cands:
        best = max(cands, key=lambda r: (r["right"], -int(r["name"].split("-")[-1])))
        print(f"\nchosen by the rule: {best['name']}")


if __name__ == "__main__":
    main()
