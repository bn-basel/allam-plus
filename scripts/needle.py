"""Arabic needle-in-a-haystack test.

A haystack is a long stretch of Arabic book text. A needle is one short invented fact hidden inside it (for example
"the code of the lab door is 9052"), and the question at the end asks for it. If the model answers correctly at every
length and depth, it can find information anywhere in its context; if it fails beyond some length, that length is
where its long-context reading breaks down.

    uv run python scripts/needle.py build                      # make the cases (deterministic)
    uv run python scripts/needle.py run --rope shipped         # run them with one position setting
    uv run python scripts/needle.py run --rope yarn4
    uv run python scripts/needle.py report

Design (v1): 10 needles (data/needle/needles_v1.json) × 4 lengths (2K/4K/8K/16K tokens, prompt + 64 tokens of
answer room) × 5 depths (10/30/50/70/90% of the haystack) = 200 cases, plus 10 controls at 2K with no needle (does the
model "guess" the answer?). Each needle has its own haystack: LC-Eval's Arabic book texts (claim-verification and
multiple-choice contexts), whitespace collapsed, cut at a sentence end. A haystack that already contains the answer is
skipped. Longer haystacks start with the shorter ones' text. Scoring: correct if the answer (normalised: no
diacritics, one alef, ة→ه, ى→ي, Hindi digits → Arabic digits) appears in the model's normalised reply.

The built cases quote licensed book text, so they stay in data/cache/ (not published); the needles, this script and
the SHA-256 of the cases file are published.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, chat_prompt, normalize_ar  # noqa: E402

LENGTHS = [2048, 4096, 8192, 16384]
DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]
ANSWER_ROOM = 64
NEEDLES = ROOT / "data" / "needle" / "needles_v1.json"
LCEVAL = ROOT / "data" / "raw" / "lceval" / "lceval_ar.jsonl"
CASES = ROOT / "data" / "cache" / "needle_cases_v1.jsonl"
RESULTS = ROOT / "results" / "phase1" / "needle"
HAYSTACK_TASKS = {"Claim Verification", "Multiple Choice Question Answering"}
SENTENCE_END = re.compile(r"[.!؟?]\s")


def user_message(haystack: str, question: str) -> str:
    return ("اقرأ النص التالي، ثم أجب عن السؤال الذي يليه.\n\n" + haystack +
            "\n\nالسؤال: " + question + "\nأجب من النص أعلاه فقط وباختصار.")


def book_texts() -> list[str]:
    seen, texts = set(), []
    for line in open(LCEVAL, encoding="utf-8"):
        item = json.loads(line)
        if item["task"] not in HAYSTACK_TASKS:
            continue
        text = re.sub(r"\s+", " ", item["full_context"]).strip()
        if text[:300] not in seen:
            seen.add(text[:300])
            texts.append(text)
    return texts


def sentence_ends(text: str) -> list[int]:
    return [m.start() + 1 for m in SENTENCE_END.finditer(text)]


def build(_args) -> None:
    from transformers import AutoTokenizer
    from common import MODEL_ID

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    needles = json.loads(NEEDLES.read_text(encoding="utf-8"))
    texts = book_texts()
    print(f"{len(texts)} distinct book texts")
    n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    cases, used = [], set()
    start = 0
    for nd in needles:
        # Find a haystack (one or more consecutive book texts, none used before) long enough for 16K tokens that does
        # not already contain the answer.
        while True:
            base, j = "", start
            while len(base) < 90_000 and j < len(texts):
                base += (" " if base else "") + texts[j]
                j += 1
            if len(base) < 90_000:
                sys.exit("not enough book text for the haystacks")
            if not any(a in normalize_ar(base) for a in nd["accept"]):
                break
            start += 1
        source = list(range(start, j))
        used.update(source)
        start = j
        enc = tok(base, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        ends = sentence_ends(base)
        for L in LENGTHS:
            overhead = n_tok(chat_prompt(tok, user_message(nd["needle"], nd["question"])))
            target = L - ANSWER_ROOM - overhead - 2  # 2 tokens of slack for the spaces around the needle
            cut = max(e for e in ends if e <= offsets[target - 1][1])
            while True:
                hay = base[:cut]
                prompt = chat_prompt(tok, user_message(hay + " " + nd["needle"], nd["question"]))
                if n_tok(prompt) <= L - ANSWER_ROOM:
                    break
                cut = max(e for e in ends if e < cut)
            hay_ends = [e for e in ends if e < cut]
            for d in DEPTHS:
                pos = min(hay_ends, key=lambda e: abs(e - d * cut))
                text = hay[:pos] + " " + nd["needle"] + hay[pos:]
                prompt = chat_prompt(tok, user_message(text, nd["question"]))
                p_tokens = n_tok(prompt)
                depth_tokens = n_tok(hay[:pos]) / max(1, n_tok(hay))
                cases.append({"id": f"{nd['id']}-{L}-{int(d * 100)}", "needle": nd["id"], "length": L, "depth": d,
                              "depth_actual": round(depth_tokens, 3), "prompt_tokens": p_tokens,
                              "source_texts": source, "accept": nd["accept"], "prompt": prompt})
            if L == LENGTHS[0]:  # control: same 2K haystack, no needle
                prompt = chat_prompt(tok, user_message(hay, nd["question"]))
                cases.append({"id": f"{nd['id']}-{L}-control", "needle": nd["id"], "length": L, "depth": None,
                              "control": True, "prompt_tokens": n_tok(prompt), "source_texts": source,
                              "accept": nd["accept"], "prompt": prompt})
    CASES.parent.mkdir(parents=True, exist_ok=True)
    with open(CASES, "w", encoding="utf-8", newline="\n") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(CASES.read_bytes()).hexdigest()
    lens = defaultdict(list)
    for c in cases:
        lens[c["length"]].append(c["prompt_tokens"])
    print(f"{len(cases)} cases -> {CASES}\nSHA-256 {sha}")
    for L, v in sorted(lens.items()):
        print(f"  {L}: prompt tokens {min(v)}-{max(v)}")
    meta = {"cases": len(cases), "sha256": sha, "book_texts_used": sorted(used),
            "prompt_tokens": {str(L): [min(v), max(v)] for L, v in lens.items()}}
    (ROOT / "data" / "needle" / "cases_v1.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def correct(reply: str, accept: list[str]) -> bool:
    r = normalize_ar(reply)
    return any(a in r for a in accept)


def run(args) -> None:
    from common import generate, load_model

    out_dir = RESULTS / args.rope
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "cases.jsonl"
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    cases = [json.loads(l) for l in open(CASES, encoding="utf-8")]
    cases = [c for c in cases if c["length"] <= args.max_length and c["id"] not in done]
    cases.sort(key=lambda c: c["length"])  # short first: a memory problem shows up last, after the rest is saved
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(done)} done, {len(cases)} to run, rope={args.rope}", flush=True)
    if not cases:
        return
    tok, model = load_model(args.rope)
    with open(out, "a", encoding="utf-8", newline="\n") as f:
        for i, c in enumerate(cases):
            g = generate(tok, model, c["prompt"], max_new_tokens=48)
            rec = {k: v for k, v in c.items() if k != "prompt"} | {
                "reply": g["text"], "correct": correct(g["text"], c["accept"]), "seconds": g["seconds"],
                "peak_gb": g["peak_gb"], "rope": args.rope}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(cases)}] {c['id']} {'OK ' if rec['correct'] else 'no '} {g['seconds']}s "
                  f"{g['peak_gb']}GB  {g['text'][:60]!r}", flush=True)


def report(_args) -> None:
    summary = {}
    for d in sorted(p for p in RESULTS.iterdir() if (p / "cases.jsonl").exists()):
        rows = [json.loads(l) for l in open(d / "cases.jsonl", encoding="utf-8")]
        grid, ctrl = defaultdict(list), []
        for r in rows:
            (ctrl.append(r["correct"]) if r.get("control") else grid[(r["length"], r["depth"])].append(r["correct"]))
        by_len = defaultdict(list)
        for (L, _), v in grid.items():
            by_len[L] += v
        summary[d.name] = {
            "grid": {f"{L}|{int(dp * 100)}": sum(v) / len(v) for (L, dp), v in sorted(grid.items())},
            "by_length": {str(L): {"correct": sum(v), "n": len(v), "acc": round(sum(v) / len(v), 3)}
                          for L, v in sorted(by_len.items())},
            "controls_answered": f"{sum(ctrl)}/{len(ctrl)}",
            "peak_gb": max(r["peak_gb"] for r in rows),
            "seconds_by_length": {str(L): round(sum(r["seconds"] for r in rows if r["length"] == L) /
                                                max(1, sum(1 for r in rows if r["length"] == L)), 1)
                                  for L in sorted({r["length"] for r in rows})}}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    lines = ["# Arabic needle-in-a-haystack (v1)", "",
             "Share of needles found (10 needles × 5 depths per length). Controls: no needle in the text; a "
             "'found' control means the model guessed the answer.", ""]
    for name, s in summary.items():
        lines += [f"## rope = {name}", "", "| depth \\ length | " + " | ".join(f"{int(L) // 1024}K" for L in s["by_length"]) + " |",
                  "|---|" + "---|" * len(s["by_length"])]
        for dp in DEPTHS:
            cells = [s["grid"].get(f"{L}|{int(dp * 100)}") for L in s["by_length"]]
            lines.append(f"| {int(dp * 100)}% | " + " | ".join("-" if c is None else f"{c:.0%}" for c in cells) + " |")
        lines.append("| **all** | " + " | ".join(f"**{v['acc']:.0%}** ({v['correct']}/{v['n']})"
                                                for v in s["by_length"].values()) + " |")
        lines += ["", f"Controls answered: {s['controls_answered']}. Peak GPU memory {s['peak_gb']} GB. "
                  f"Seconds per case by length: {s['seconds_by_length']}.", ""]
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(fn=build)
    r = sub.add_parser("run")
    r.add_argument("--rope", choices=["shipped", "yarn4"], required=True)
    r.add_argument("--max-length", type=int, default=max(LENGTHS))
    r.add_argument("--limit", type=int, default=0)
    r.set_defaults(fn=run)
    sub.add_parser("report").set_defaults(fn=report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
