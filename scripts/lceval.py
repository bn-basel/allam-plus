"""LC-Eval (HUMAIN, EMNLP 2025 Findings), Arabic part, the two tasks that can be scored without an LLM judge:

- multiple choice: a question about a long book text with 4 options;
- claim verification: 5 short claims about a long book text, each true or false.

    uv run python scripts/lceval.py --rope shipped [--adapter ...] [--max-tokens 16384]

Only items whose prompt fits in --max-tokens (minus room for the answer) are used. Answers are read from the model's
next-token probabilities, not from generated text: for multiple choice the option letter (A-D) with the highest
probability; for each claim, "صحيح" (true) vs "خطأ" (false). So there are no parsing failures.

Position bias fix: in LC-Eval's Arabic multiple-choice items the correct option is the first one 62% of the time, so a
model that always says "A" would score 62%. Each item's options are shuffled with a fixed seed (the same order for
every model), which puts the correct answer in each position about equally often; chance is then ~25%.

The long text is shared by all questions about it, so it is read once and its KV cache reused for each question.
"""

import argparse
import ast
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402

DATA = ROOT / "data" / "raw" / "lceval" / "lceval_ar.jsonl"
LETTERS = ["A", "B", "C", "D"]
BUCKETS = [(0, 4096, "≤4K"), (4096, 8192, "4-8K"), (8192, 16384, "8-16K"), (16384, 32768, "16-32K")]
ANSWER_ROOM = 16


def mcq_items(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if not r["task"].startswith("Multiple"):
            continue
        q, _, opts = r["question"].partition("\n")
        options = [o.strip() for o in ast.literal_eval(opts.strip())]
        gold = int(r["answer"]) - 1
        order = list(range(len(options)))
        random.Random(f"lceval-{r['row']}").shuffle(order)
        out.append({"row": r["row"], "kind": "mcq", "context": r["full_context"], "question": q.strip(),
                    "options": [options[i] for i in order], "gold": order.index(gold)})
    return out


def claim_items(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if not r["task"].startswith("Claim"):
            continue
        body = r["question"].split("\n", 1)[1].strip()
        claims = [c.strip() for c in re.split(r"(?<=[.!؟])\s+", body) if c.strip()]
        if len(claims) != len(r["answer"]):  # cannot tell which label belongs to which claim: skip
            continue
        out.append({"row": r["row"], "kind": "claims", "context": r["full_context"], "claims": claims,
                    "gold": [a.strip().lower() == "true" for a in r["answer"]]})
    return out


def prompts_for(tok, item: dict) -> list[str]:
    from common import chat_prompt
    head = "اقرأ النص التالي ثم أجب.\n\n" + re.sub(r"[ \t]+", " ", item["context"]).strip() + "\n\n"
    if item["kind"] == "mcq":
        opts = "\n".join(f"{L}. {o}" for L, o in zip(LETTERS, item["options"]))
        return [chat_prompt(tok, head + f"السؤال: {item['question']}\n{opts}\n\nأجب بحرف الخيار الصحيح فقط (A أو B أو C أو D).")]
    return [chat_prompt(tok, head + f"الادعاء: {c}\nهل هذا الادعاء صحيح أم خطأ بحسب النص؟ أجب بكلمة واحدة: صحيح أو خطأ.")
            for c in item["claims"]]


def first_token(tok, word: str) -> int:
    ids = [i for i in tok(" " + word, add_special_tokens=False)["input_ids"] if tok.convert_ids_to_tokens(i) != "▁"]
    return ids[0]


def main() -> None:
    import torch
    from common import load_model, new_cache, prefill

    ap = argparse.ArgumentParser()
    ap.add_argument("--rope", choices=["shipped", "yarn4"], default="shipped")
    ap.add_argument("--adapter")
    ap.add_argument("--name")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    name = args.name or args.rope + (("__" + args.adapter.replace("/", "_")) if args.adapter else "")
    out_dir = ROOT / "results" / "lceval" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "cases.jsonl"

    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    items = mcq_items(rows) + claim_items(rows)
    tok, model = load_model(args.rope, args.adapter)
    letter_ids = [first_token(tok, L) for L in LETTERS]
    tf_ids = [first_token(tok, "صحيح"), first_token(tok, "خطأ")]
    assert len(set(tf_ids)) == 2

    done = {(json.loads(l)["kind"], json.loads(l)["row"]) for l in open(out, encoding="utf-8")} if out.exists() else set()
    work = []
    for it in items:
        ps = prompts_for(tok, it)
        ids = [tok(p, add_special_tokens=False)["input_ids"] for p in ps]
        n = max(len(i) for i in ids)
        if n + ANSWER_ROOM <= args.max_tokens and (it["kind"], it["row"]) not in done:
            work.append((n, it, ids))
    work.sort(key=lambda w: w[0])
    if args.limit:
        work = work[: args.limit]
    print(f"{len(work)} items to run (≤{args.max_tokens} tokens), {len(done)} done", flush=True)

    cache = new_cache(model, args.max_tokens)  # one fixed-size cache, overwritten from position 0 for each item
    with open(out, "a", encoding="utf-8", newline="\n") as f:
        for k, (n, it, ids) in enumerate(work):
            common = 0  # longest shared token prefix of this item's prompts: read once, reuse its KV cache
            while all(len(i) > common for i in ids) and len({i[common] for i in ids}) == 1:
                common += 1
            common = min(common, min(len(i) for i in ids) - 1)
            with torch.no_grad():
                prefill(model, ids[0][:common], cache)  # the shared text, read once
                scores = [prefill(model, i[common:], cache, start=common) for i in ids]  # each question after it
            if it["kind"] == "mcq":
                pred = int(scores[0][letter_ids[: len(it["options"])]].argmax())
                rec = {"pred": pred, "gold": it["gold"], "correct": pred == it["gold"]}
            else:
                preds = [bool(s[tf_ids[0]] > s[tf_ids[1]]) for s in scores]
                rec = {"pred": preds, "gold": it["gold"], "claims_correct": sum(p == g for p, g in zip(preds, it["gold"])),
                       "n_claims": len(preds), "correct": preds == it["gold"]}
            rec = {"kind": it["kind"], "row": it["row"], "prompt_tokens": n, "rope": args.rope} | rec
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{k + 1}/{len(work)}] {it['kind']} {n} tok {'OK' if rec['correct'] else 'no'}", flush=True)
    report(out_dir)


def report(out_dir: Path) -> dict:
    rows = [json.loads(l) for l in open(out_dir / "cases.jsonl", encoding="utf-8")]
    s = {}
    for kind in ("mcq", "claims"):
        for lo, hi, label in BUCKETS + [(0, 10 ** 9, "all")]:
            sel = [r for r in rows if r["kind"] == kind and lo < r["prompt_tokens"] <= hi]
            if not sel:
                continue
            if kind == "mcq":
                s[f"mcq|{label}"] = {"n": len(sel), "acc": round(sum(r["correct"] for r in sel) / len(sel), 4)}
            else:
                nc = sum(r["n_claims"] for r in sel)
                pairs = [(p, g) for r in sel for p, g in zip(r["pred"], r["gold"])]
                true_ok = sum(p and g for p, g in pairs) / max(1, sum(g for _, g in pairs))
                false_ok = sum(not p and not g for p, g in pairs) / max(1, sum(not g for _, g in pairs))
                s[f"claims|{label}"] = {"n": len(sel), "claim_acc": round(sum(r["claims_correct"] for r in sel) / nc, 4),
                                        # mean of "true claims called true" and "false claims called false": 50% for a
                                        # model that always gives the same answer, whatever that answer is
                                        "balanced_acc": round((true_ok + false_ok) / 2, 4),
                                        "said_true": round(sum(p for p, _ in pairs) / len(pairs), 4),
                                        "all_claims_right": round(sum(r["correct"] for r in sel) / len(sel), 4),
                                        "always_true_baseline": round(sum(sum(r["gold"]) for r in sel) / nc, 4)}
    (out_dir / "summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(s, indent=1, ensure_ascii=False))
    return s


if __name__ == "__main__":
    main()
