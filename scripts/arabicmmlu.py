"""ArabicMMLU, 0-shot: 14,455 multiple-choice school and university questions written in Arabic (MBZUAI, CC BY-NC 4.0,
used for evaluation only).

    uv run python scripts/arabicmmlu.py --rope shipped [--adapter runs/x/checkpoint-n] [--limit 200]

Format: exactly the LM Evaluation Harness task "arabicmmlu" (the harness ALLaM's model card reports with), so our base
number can be checked against the card's 67.78:
    This is a {Subject} question for {level} in {country}. Select the correct answer!

    Question: {context + question}
    A. ...
    B. ...

    Answer:
The model's choice is the option letter with the highest probability as the next token (" A", " B", ...). No chat
template, no generation: this measures what the model knows, not how it writes (Arabic IFEval covers that).
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402

DATA = ROOT / "data" / "raw" / "arabicmmlu" / "test.csv"
PROMPT = "This is a {}. Select the correct answer!\n\nQuestion: {}\n{}\n\nAnswer:"
LEVEL = {"Primary": "primary school", "Middle": "middle school", "High": "high school", "Univ": "university",
         "Prof": "professional"}
LETTERS = ["A", "B", "C", "D", "E"]


def doc_text(d: dict) -> tuple[str, int]:
    blank = lambda v: v is None or (isinstance(v, float) and math.isnan(v)) or v == ""  # noqa: E731
    level = "" if blank(d["Level"]) else " for " + LEVEL[d["Level"]]
    country = "" if blank(d["Country"]) else " in " + d["Country"]
    question = d["Question"] if blank(d["Context"]) else f"{d['Context']}\n\n{d['Question']}"
    options = []
    for i in range(5):
        v = d[f"Option {i + 1}"]
        if blank(v):
            break
        options.append(f"{LETTERS[i]}. {v}")
    return PROMPT.format(f"{d['Subject']} question{level}{country}", question, "\n".join(options)), len(options)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main() -> None:
    import pandas as pd
    import torch
    from common import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--rope", choices=["shipped", "yarn4"], default="shipped")
    ap.add_argument("--adapter")
    ap.add_argument("--name", help="results folder name (default: rope [+ adapter])")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    name = args.name or args.rope + (("__" + args.adapter.replace("/", "_")) if args.adapter else "")
    out_dir = ROOT / "results" / "arabicmmlu" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = pd.read_csv(DATA).to_dict("records")
    if args.limit:
        docs = docs[:: max(1, len(docs) // args.limit)][: args.limit]
    tok, model = load_model(args.rope, args.adapter)
    inner = model.get_base_model() if args.adapter else model
    letter_ids = []
    for L in LETTERS:
        ids = tok(" " + L, add_special_tokens=False)["input_ids"]
        ids = [i for i in ids if tok.convert_ids_to_tokens(i) != "▁"]
        assert len(ids) == 1, (L, ids)
        letter_ids.append(ids[0])

    items = [(d, *doc_text(d)) for d in docs]
    order = sorted(range(len(items)), key=lambda i: len(items[i][1]))
    preds = [None] * len(items)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    # Batches of up to --batch questions but at most ~6,000 tokens in total (long reading passages go a few at a
    # time): attention's temporary tables grow with batch × length², and 16 long questions filled the card.
    lengths = [len(tok(items[i][1])["input_ids"]) for i in order]
    batches, cur, cur_max = [], [], 0
    for i, n in zip(order, lengths):
        if cur and (len(cur) >= args.batch or max(cur_max, n) * (len(cur) + 1) > 6000):
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, n)
    batches.append(cur)
    done_n = 0
    for bi, idx in enumerate(batches):
        b = done_n
        done_n += len(idx)
        enc = tok([items[i][1] for i in idx], return_tensors="pt", padding=True, add_special_tokens=True).to(model.device)
        last = enc["attention_mask"].sum(1) - 1
        with torch.no_grad():
            # Only the last real position of each question matters. Running the output layer on every position
            # would make 16 × length × 64,000 scores (8 GB for long questions), so take the hidden states and apply
            # the output layer to the last positions only. (LoRA layers, if any, sit inside the inner model.)
            hidden = inner.model(**enc).last_hidden_state
            logits = inner.lm_head(hidden[torch.arange(len(idx), device=hidden.device), last])
        for row, i in enumerate(idx):
            n_opt = items[i][2]
            lp = torch.log_softmax(logits[row].float(), -1)[letter_ids[:n_opt]]
            preds[i] = LETTERS[int(lp.argmax())]
        if bi % 50 == 0:
            print(f"{b + len(idx)}/{len(order)}", flush=True)

    by = {"Group": defaultdict(list), "Subject": defaultdict(list)}
    rows = []
    for (d, _, n_opt), p in zip(items, preds):
        ok = p == d["Answer Key"]
        by["Group"][d["Group"]].append(ok)
        by["Subject"][d["Subject"]].append(ok)
        rows.append({"ID": int(d["ID"]), "pred": p, "gold": d["Answer Key"], "correct": ok})
    k, n = sum(r["correct"] for r in rows), len(rows)
    subj = {s: sum(v) / len(v) for s, v in by["Subject"].items()}
    summary = {"name": name, "rope": args.rope, "adapter": args.adapter, "n": n, "correct": k, "acc": round(k / n, 4),
               "ci95": [round(x, 4) for x in wilson(k, n)], "macro_subject_acc": round(sum(subj.values()) / len(subj), 4),
               "by_group": {g: round(sum(v) / len(v), 4) for g, v in sorted(by["Group"].items())},
               "by_subject": {s: round(a, 4) for s, a in sorted(subj.items())}}
    with open(out_dir / "cases.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k_: v for k_, v in summary.items() if k_ != "by_subject"}, indent=1))


if __name__ == "__main__":
    main()
