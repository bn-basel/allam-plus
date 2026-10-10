"""Score Phase 2 checkpoints on the held-out dev set (Wikipedia articles never used in training).

    uv run python scripts/long_dev.py --adapter runs/long_v1/checkpoint-40 [--rope yarn4]
    uv run python scripts/long_dev.py --base               # the starting point: base + YaRN, no adapter
    uv run python scripts/long_dev.py --report long_v1     # table of all scored checkpoints of a run

Long questions (dev.jsonl, kind longqa): each question is asked at the point of the chat where training asks it
(earlier questions answered with the reference answers), greedy, up to 64 new tokens.
  open questions: word-overlap F1 between the reply and the reference answer (normalised Arabic, content words);
  multiple choice: the first option letter in the reply (A-D followed by "." or ")") is the right one.
Replay (kind replay): mean loss of the original model's answers under the checkpoint (teacher forcing): how far
short-text behaviour has drifted from the original. Lower = closer.

Checkpoint rule, fixed before training (docs/DECISIONS.md D14): the checkpoint with the highest long-question score
(mean over all dev questions, an open question counting its F1 and a multiple-choice question 1 or 0) is chosen;
on a tie, the earlier one. The tests (needle, LC-Eval, ArabicMMLU, IFEval) are run only on the chosen checkpoint.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, normalize_ar  # noqa: E402

DEV = ROOT / "data" / "long" / "dev.jsonl"


def words(s: str) -> list[str]:
    return re.findall(r"[ء-ي0-9]+", normalize_ar(s))


def f1(pred: str, ref: str) -> float:
    p, r = words(pred), words(ref)
    if not p or not r:
        return 0.0
    common = sum(min(p.count(w), r.count(w)) for w in set(r))
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def mcq_letter(reply: str) -> str | None:
    m = re.search(r"(?<![A-Za-z])([ABCD])[.)]", reply)
    return m.group(1) if m else None


def score(kind: str, reply: str, ref: str) -> float:
    return float(mcq_letter(reply) == ref[:1]) if kind == "mcq" else f1(reply, ref)


def summarize(name: str, rope: str, rows: list[dict], losses: list[float]) -> dict:
    return {"name": name, "rope": rope, "long_score": round(sum(r["score"] for r in rows) / len(rows), 4),
            "open_f1": round(sum(r["score"] for r in rows if r["kind"] == "open") /
                             max(1, sum(r["kind"] == "open" for r in rows)), 4),
            "mcq_acc": round(sum(r["score"] for r in rows if r["kind"] == "mcq") /
                             max(1, sum(r["kind"] == "mcq" for r in rows)), 4),
            "n_questions": len(rows), "replay_loss": round(sum(losses) / max(1, len(losses)), 4),
            "n_replay": len(losses), "replay_losses": losses, "rows": rows}


def rescore() -> None:
    """Recompute scores from the saved replies (after a scoring fix), without the GPU."""
    for p in (ROOT / "results" / "long_dev").glob("*.json"):
        s = json.loads(p.read_text(encoding="utf-8"))
        for r in s["rows"]:
            r["score"] = score(r["kind"], r["reply"], r["ref"])
        losses = s.get("replay_losses") or [s["replay_loss"]] * s["n_replay"]
        new = summarize(s["name"], s["rope"], s["rows"], losses)
        p.write_text(json.dumps(new, indent=1, ensure_ascii=False), encoding="utf-8")
        print(p.name, {k: new[k] for k in ("long_score", "open_f1", "mcq_acc", "replay_loss")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--rope", default="yarn4")
    ap.add_argument("--report")
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report(args.report)
    if args.rescore:
        return rescore()

    import torch
    from common import generate, load_model

    name = "base_" + args.rope if args.base else args.adapter.replace("\\", "/").rstrip("/")
    out = ROOT / "results" / "long_dev" / (name.replace("runs/", "").replace("/", "__") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tok, model = load_model(args.rope, None if args.base else args.adapter)
    dev = [json.loads(l) for l in open(DEV, encoding="utf-8")]
    rows = []
    for ex in (e for e in dev if e["kind"] == "longqa"):
        for qi, kind in enumerate(ex["question_kinds"]):
            msgs = ex["messages"][: 2 * qi + 1]
            ref = ex["messages"][2 * qi + 1]["content"]
            g = generate(tok, model, tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), 64)
            rows.append({"id": ex["id"], "q": qi, "kind": kind, "tokens": g["prompt_tokens"],
                         "score": score(kind, g["text"], ref),
                         "reply": g["text"], "ref": ref})
    from train_long import encode
    losses = []
    for ex in (e for e in dev if e["kind"] == "replay"):
        enc = encode(ex, tok)
        ids = torch.tensor([enc["input_ids"]], device=model.device)
        labels = torch.tensor([enc["labels"]], device=model.device)
        with torch.no_grad():
            losses.append(float(model(input_ids=ids, labels=labels).loss))
    s = summarize(name, args.rope, rows, losses)
    out.write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in s.items() if k not in ("rows", "replay_losses")}, indent=1))


def report(run: str) -> None:
    files = sorted((ROOT / "results" / "long_dev").glob(f"{run}__checkpoint-*.json"),
                   key=lambda p: int(p.stem.split("-")[-1]))
    base = ROOT / "results" / "long_dev" / "base_yarn4.json"
    rows = ([json.loads(base.read_text(encoding="utf-8"))] if base.exists() else []) + \
           [json.loads(p.read_text(encoding="utf-8")) for p in files]
    print("| checkpoint | long score | open F1 | multiple choice | replay loss |\n|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['name'].split('/')[-1]} | {r['long_score']:.3f} | {r['open_f1']:.3f} | {r['mcq_acc']:.3f} | "
              f"{r['replay_loss']:.3f} |")
    cands = [r for r in rows if "checkpoint" in r["name"]]
    if cands:
        best = max(cands, key=lambda r: (r["long_score"], -int(r["name"].split("-")[-1])))
        print(f"\nchosen by the rule: {best['name']}")


if __name__ == "__main__":
    main()
