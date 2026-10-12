"""Phase 3: assemble the tool-calling training and dev sets.

    uv run python scripts/toolbuild.py                      # -> data/tools/train.jsonl, data/tools/dev.jsonl
    uv run python scripts/toolbuild.py --focus              # -> data/tools/train_v2.jsonl (receptionist round)
    uv run python scripts/toolbuild.py --fix                # -> data/tools/train_v3.jsonl, dev_v3.jsonl (fix round)

Every conversation is rendered with the allam-plus tool format (scripts/toolformat.py) into plain
system/user/assistant messages, which train_long.py trains on (every assistant turn: tool-call lines and replies).

Mix (docs/DECISIONS.md D16):
- general tool conversations from our pipeline (data/tools/convos.jsonl, the ones that passed every check);
  5% (by a hash of the generation job) held out as dev;
- the receptionist conversations of the earlier project (data/tools/receptionist.jsonl: its train split here, its
  validation split in dev), now with the Arabic guide;
- Phase 2 replay, so long reading and short-text behaviour are not forgotten: 150 long question chats, 30 long
  articles and 400 short replay answers from data/long/train.jsonl.
"""

import hashlib
import json
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402
from toolformat import render  # noqa: E402

T = ROOT / "data" / "tools"
WEEKDAYS_AR = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]


def openai_messages(msgs: list[dict]) -> list[dict]:
    out = []
    for m in msgs:
        if m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or "",
                        "tool_calls": [{"name": c["name"], "arguments": c.get("arguments", {})} for c in m["tool_calls"]]})
        elif m["role"] == "tool":
            c = m.get("content")
            out.append({"role": "tool", "content": c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def main() -> None:
    rng = random.Random(53)
    lib = json.loads((T / "library.json").read_text(encoding="utf-8"))
    tools = {t["name"]: t for d in lib.values() for t in d}
    train, dev, stats = [], [], Counter()
    for r in map(json.loads, open(T / "convos.jsonl", encoding="utf-8")):
        if r["rejected"]:
            continue
        system = None
        if r["today"]:
            d = date.fromisoformat(r["today"])
            system = f"تاريخ اليوم: {WEEKDAYS_AR[d.weekday()]} {r['today']}"
        offered = [{"type": "function", "function": tools[n]} for n in r["tools"]]
        ex = {"id": r["id"], "kind": "tool", "source": "general", "scenario": r["kind"], "dialect": r["dialect"],
              "messages": render(system, openai_messages(r["messages"]), offered)}
        job = r["id"].split("-")[0]
        (dev if int(hashlib.sha256(job.encode()).hexdigest(), 16) % 100 < 5 else train).append(ex)
    for r in map(json.loads, open(T / "receptionist.jsonl", encoding="utf-8")):
        ex = {"id": "rec-" + r["id"], "kind": "tool", "source": "receptionist",
              "messages": render(r["system"], r["messages"], r["tools"])}
        (train if r["split"] == "train" else dev).append(ex)
    long_rows = [json.loads(l) for l in open(ROOT / "data" / "long" / "train.jsonl", encoding="utf-8")]
    for kind, n in (("longqa", 150), ("lm", 30), ("replay", 400)):
        pool = [e for e in long_rows if e["kind"] == kind]
        train += [{**e, "source": "phase2"} for e in rng.sample(pool, min(n, len(pool)))]
    rng.shuffle(train)
    for name, rows in (("train", train), ("dev", dev)):
        with open(T / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for e in rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        stats.update(f"{name}_{e.get('source')}_{e['kind']}" for e in rows)
    print(json.dumps(dict(sorted(stats.items())), indent=1))
    (T / "build.stats.json").write_text(json.dumps(dict(sorted(stats.items())), indent=1), encoding="utf-8")


FOCUS = {"receptionist": None, "general": 500, "longqa": 100, "lm": 20, "replay": 200}  # None = all of them


def focus() -> None:
    """The receptionist tuning round (tools_v2, docs/DECISIONS.md D19): a second pass over every receptionist
    conversation, with enough general tool conversations and Phase 2 replay that the rest is not forgotten. Drawn from
    data/tools/train.jsonl (already rendered, so word for word what tools_v1 trained on); dev stays the same."""
    rng = random.Random(59)
    rows = [json.loads(l) for l in open(T / "train.jsonl", encoding="utf-8")]
    out = []
    for key, n in FOCUS.items():
        if key in ("receptionist", "general"):
            pool = [e for e in rows if e.get("source") == key]
        else:
            pool = [e for e in rows if e.get("source") == "phase2" and e["kind"] == key]
        out += pool if n is None else rng.sample(pool, min(n, len(pool)))
    rng.shuffle(out)
    with open(T / "train_v2.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    stats = Counter(f"{e.get('source')}_{e['kind']}" for e in out)
    print(json.dumps(dict(sorted(stats.items())), indent=1), f"\n{len(out)} examples -> data/tools/train_v2.jsonl")


FIX_REPLAY = {"general": 700, "receptionist": 400, "longqa": 60, "lm": 10, "replay": 200}


def fix() -> None:
    """The fix round (tools_v3, docs/DECISIONS.md D23): the new conversations for three skills (a second call that waits
    for the first one's result, asking for missing details before an action, arithmetic on results), 5% held out by
    generation job, plus replay drawn word for word from the tools_v1 training file so nothing else is forgotten.
    Only conversations whose job gave today's date are used, so no date in them was guessed.
    -> data/tools/train_v3.jsonl, and data/tools/dev_v3.jsonl = dev.jsonl + the held-out new conversations."""
    rng = random.Random(61)
    lib = json.loads((T / "library.json").read_text(encoding="utf-8"))
    tools = {t["name"]: t for d in lib.values() for t in d}
    new_train, new_dev = [], []
    for r in map(json.loads, open(T / "convos_fix.jsonl", encoding="utf-8")):
        if r["rejected"] or not r["today"]:
            continue
        d = date.fromisoformat(r["today"])
        system = f"تاريخ اليوم: {WEEKDAYS_AR[d.weekday()]} {r['today']}"
        offered = [{"type": "function", "function": tools[n]} for n in r["tools"]]
        ex = {"id": r["id"], "kind": "tool", "source": "fix", "scenario": r["kind"], "dialect": r["dialect"],
              "messages": render(system, openai_messages(r["messages"]), offered)}
        job = r["id"].split("-")[0]
        (new_dev if int(hashlib.sha256(job.encode()).hexdigest(), 16) % 100 < 5 else new_train).append(ex)
    rows = [json.loads(l) for l in open(T / "train.jsonl", encoding="utf-8")]
    replay = []
    for key, n in FIX_REPLAY.items():
        if key in ("receptionist", "general"):
            pool = [e for e in rows if e.get("source") == key]
        else:
            pool = [e for e in rows if e.get("source") == "phase2" and e["kind"] == key]
        replay += rng.sample(pool, min(n, len(pool)))
    train = new_train + replay
    rng.shuffle(train)
    dev = [json.loads(l) for l in open(T / "dev.jsonl", encoding="utf-8")] + new_dev
    for name, out in (("train_v3", train), ("dev_v3", dev)):
        with open(T / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for e in out:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    stats = {name: dict(sorted(Counter(f"{e.get('source')}_{e.get('scenario') or e['kind']}" for e in out).items()))
             for name, out in (("train_v3", train), ("dev_v3", dev))}
    print(json.dumps(stats, indent=1))
    (T / "build_v3.stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")


if __name__ == "__main__":
    {"--focus": focus, "--fix": fix}.get(sys.argv[1] if sys.argv[1:] else "", main)()
