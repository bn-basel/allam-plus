"""Phase 2 data: long-context training examples built from Arabic Wikipedia, written mostly by ALLaM itself.

    uv run python scripts/longdata.py articles                    # clean + split the Wikipedia shard
    uv run python scripts/longdata.py genqa --server URL --n 1500 # ALLaM writes Q&A pairs about single sections
    uv run python scripts/longdata.py replay --server URL --n 900 # the original ALLaM answers short tasks
    uv run python scripts/longdata.py build                       # assemble train / dev sets
    uv run python scripts/longdata.py stats

Why this design (docs/DECISIONS.md D13):
- Source text: Arabic Wikipedia (20231101.ar, shard 0, CC BY-SA 4.0). Never book text (LC-Eval's Arabic texts are
  Hindawi books) and never invented "needle" facts, so the long-context tests stay unseen.
- Long reading (the main part): ALLaM, given ONE short section (~100-400 tokens) and no long context, writes two
  question-answer pairs about it. The section's article is then placed among other articles to make a 6K-16K-token
  context, and the model learns to answer from the right place. Three questions per context, about sections from
  different articles, as a multi-turn chat. A quarter are multiple choice, with the wrong options taken from answers to
  other questions about the same article (same topic, so they are plausible).
- Replay (protects short-text quality): the ORIGINAL ALLaM (config as shipped, no YaRN) answers short tasks; the YaRN
  model is trained to give the same answers ("self-distillation"). This targets the 1.4-point ArabicMMLU drop.
- Plain long text: a few whole long articles as raw text, loss on every token, so the model gets used to long
  positions in ordinary prose.
Generation goes through llama-server with ALLaM's Q8_0 build (fast); prompts are rendered by ALLaM's own tokenizer.
"""

import argparse
import hashlib
import json
import random
import re
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL_ID, ROOT, chat_prompt, normalize_ar  # noqa: E402

WIKI = ROOT / "data" / "raw" / "wikipedia_ar" / "train-00000-of-00007.parquet"
OUT = ROOT / "data" / "long"
ARTICLES = OUT / "articles.jsonl"
TAIL = re.compile(r"^\s*(المراجع|مراجع|انظر أيضا|انظر أيضًا|وصلات خارجية|روابط خارجية|المصادر|مصادر|ملاحظات|"
                  r"الهوامش|هوامش|قراءات إضافية|قراءات أخرى|مراجع وملاحظات|المراجع والملاحظات)\s*$", re.M)
LATIN = re.compile(r"[A-Za-z]")


# ---------- 1. articles ----------

def clean(text: str) -> str:
    m = TAIL.search(text)
    if m:
        text = text[: m.start()]
    text = re.sub(r"<[^>]+>", "", text)
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith(("|", "{", "!")):
            lines.append("")
            continue
        lines.append(s)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def sections(text: str) -> list[str]:
    """Paragraph groups of ~300-1,400 characters: a heading line (short, no full stop) plus the paragraphs under it."""
    out, cur = [], []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        cur.append(para)
        body = "\n\n".join(cur)
        if len(body) >= 300:
            out.append(body[:1400])
            cur = []
    return out


def articles(_args) -> None:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    rows = pq.read_table(WIKI, columns=["id", "title", "text"]).to_pylist()
    OUT.mkdir(parents=True, exist_ok=True)
    kept, latin_chars, all_chars = 0, 0, 0
    with open(ARTICLES, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            if len(r["text"]) < 3000:  # stubs: too short to be useful as a section source or filler
                continue
            text = clean(r["text"])
            if len(text) < 2500:
                continue
            # The pool split is by article id hash, so it never changes: 90% train, 5% dev, 5% spare.
            h = int(hashlib.sha256(r["id"].encode()).hexdigest(), 16) % 100
            pool = "train" if h < 90 else "dev" if h < 95 else "spare"
            n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
            latin_chars += len(LATIN.findall(text))
            all_chars += len(text)
            f.write(json.dumps({"id": r["id"], "title": r["title"], "pool": pool, "tokens": n_tok, "text": text},
                               ensure_ascii=False) + "\n")
            kept += 1
    stats = {"articles_in_shard": len(rows), "kept": kept, "latin_letter_share": round(latin_chars / all_chars, 4)}
    (OUT / "articles.stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(stats)


def load_articles(pool: str | None = None) -> list[dict]:
    return [a for a in map(json.loads, open(ARTICLES, encoding="utf-8")) if pool is None or a["pool"] == pool]


# ---------- 2. ALLaM writes Q&A pairs about one section ----------

QA_PROMPT = ("اقرأ الفقرة التالية من مقالة في ويكيبيديا بعنوان «{title}»، ثم اكتب سؤالين مختلفين يمكن الإجابة عنهما من "
             "الفقرة وحدها، مع الإجابة الصحيحة لكل سؤال في جملة قصيرة مأخوذة من الفقرة. لا تسأل عن شيء غير مذكور في "
             "الفقرة. اكتب الناتج بصيغة JSON فقط، هكذا:\n"
             '[{{"سؤال": "...", "جواب": "..."}}, {{"سؤال": "...", "جواب": "..."}}]\n\n'
             "الفقرة:\n{section}")


def complete(server: str, tok, prompt: str, max_tokens: int, temperature: float = 0.0) -> str:
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    body = {"prompt": ids, "n_predict": max_tokens, "temperature": temperature, "top_k": 1 if temperature == 0 else 40,
            "cache_prompt": False}
    req = urllib.request.Request(f"{server}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["content"].strip()


def content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[ء-ي0-9]{3,}", normalize_ar(s))}


def check_qa(q: str, a: str, section: str) -> str | None:
    """None if the pair is usable, else the reason it is not."""
    if not (5 <= len(q) <= 200 and 1 <= len(a) <= 250):
        return "length"
    if LATIN.search(q + a):
        return "latin letters"
    if not q.rstrip().endswith(("؟", "?")):
        return "not a question"
    words = content_words(a)
    if words and len(words & content_words(section)) / len(words) < 0.6:
        return "answer not from the section"  # grounded: most answer words appear in the section
    if normalize_ar(a) in normalize_ar(q):
        return "answer inside the question"
    return None


def genqa(args) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    out = OUT / "qa.jsonl"
    done = {json.loads(l)["key"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    rng = random.Random(13)
    jobs = []
    for pool, share in (("train", 0.92), ("dev", 0.08)):
        arts = load_articles(pool)
        rng.shuffle(arts)
        for a in arts[: int(args.n * share)]:
            secs = sections(a["text"])
            if len(secs) < 2:
                continue
            for si in rng.sample(range(len(secs)), 2):  # two sections per article: questions spread over the article
                key = f"{a['id']}#{si}"
                if key not in done:
                    jobs.append((pool, a, si, secs[si], key))
    print(f"{len(jobs)} sections to ask about", flush=True)

    def work(job):
        pool, a, si, sec, key = job
        raw = complete(args.server, tok, chat_prompt(tok, QA_PROMPT.format(title=a["title"], section=sec)), 300)
        m = re.search(r"\[.*\]", raw, re.S)
        pairs, reasons = [], []
        try:
            items = json.loads(m.group()) if m else []
        except json.JSONDecodeError:
            items = []
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict):
                continue
            q, ans = str(it.get("سؤال", "")).strip(), str(it.get("جواب", "")).strip()
            why = check_qa(q, ans, sec)
            (reasons.append(why) if why else pairs.append({"q": q, "a": ans}))
        return {"key": key, "article": a["id"], "title": a["title"], "pool": pool, "section_index": si,
                "pairs": pairs, "rejected": reasons, "parsed": bool(items), "raw": raw}

    with open(out, "a", encoding="utf-8", newline="\n") as f, ThreadPoolExecutor(args.parallel) as ex:
        for k, rec in enumerate(ex.map(work, jobs)):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if k % 100 == 0:
                f.flush()
                print(f"{k + 1}/{len(jobs)}", flush=True)


# ---------- 3. replay: the original ALLaM answers short tasks ----------

REPLAY_TASKS = [
    "لخّص الفقرة التالية في جملتين.\n\n{section}",
    "ما الفكرة الرئيسة في الفقرة التالية؟\n\n{section}",
    "اشرح الفقرة التالية بأسلوب بسيط لطالب في المرحلة المتوسطة.\n\n{section}",
    "استخرج من الفقرة التالية ثلاث معلومات مهمة في نقاط.\n\n{section}",
    "اكتب عنوانًا مناسبًا للفقرة التالية، ثم اذكر سبب اختيارك له في جملة.\n\n{section}",
    "اقرأ الفقرة التالية ثم أجب عن السؤال.\n\n{section}\n\nالسؤال: {question}",
    "{question}",  # the same question with no text: the model answers from what it knows
]


def replay(args) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    out = OUT / "replay.jsonl"
    done = {json.loads(l)["key"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    rng = random.Random(29)
    qa = [json.loads(l) for l in open(OUT / "qa.jsonl", encoding="utf-8")]
    qa_by_article = {r["article"]: r for r in qa if r["pairs"]}
    used = {r["article"] for r in qa}  # replay uses articles that the long Q&A did not draw from
    jobs = []
    for pool, share in (("train", 0.9), ("dev", 0.1)):
        arts = [a for a in load_articles(pool) if a["id"] not in used]
        rng.shuffle(arts)
        for a in arts[: int(args.n * share)]:
            secs = sections(a["text"])
            if not secs:
                continue
            task = rng.choice(REPLAY_TASKS[:5])
            jobs.append((pool, a["id"], task.format(section=rng.choice(secs))))
    # plus questions ALLaM wrote about other sections, answered with and without the section
    for r in list(qa_by_article.values())[: args.n // 4]:
        q = r["pairs"][0]["q"]
        jobs.append((r["pool"], r["article"] + "#q", REPLAY_TASKS[6].format(question=q)))
    jobs = [(p, k, t) for p, k, t in jobs if f"{k}|{hashlib.md5(t.encode()).hexdigest()[:8]}" not in done]
    print(f"{len(jobs)} short tasks", flush=True)

    def work(job):
        pool, key, task = job
        ans = complete(args.server, tok, chat_prompt(tok, task), 512)
        return {"key": f"{key}|{hashlib.md5(task.encode()).hexdigest()[:8]}", "pool": pool, "prompt": task,
                "answer": ans, "latin": bool(LATIN.search(ans))}

    with open(out, "a", encoding="utf-8", newline="\n") as f, ThreadPoolExecutor(args.parallel) as ex:
        for k, rec in enumerate(ex.map(work, jobs)):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if k % 100 == 0:
                f.flush()
                print(f"{k + 1}/{len(jobs)}", flush=True)


# ---------- 4. assemble training and dev examples ----------

LONG_INTRO = "فيما يلي عدة مقالات من ويكيبيديا. اقرأها ثم أجب عن الأسئلة التي تليها.\n\n"
MCQ_ASK = "اختر الإجابة الصحيحة واكتب حرفها ثم نصها."  # worded differently from our LC-Eval prompt on purpose
LETTERS = ["A", "B", "C", "D"]


def doc_text(a: dict, tok, keep_section: str | None = None, max_tokens: int = 5000) -> str:
    """An article as a document; very long ones are cut to a window of ~max_tokens that keeps the asked-about section."""
    text = a["text"]
    if a["tokens"] > max_tokens:
        chars = int(len(text) * max_tokens / a["tokens"])
        start = 0
        if keep_section:
            pos = text.find(keep_section[:200])
            start = max(0, min(pos - chars // 2, len(text) - chars)) if pos >= 0 else 0
        text = text[start: start + chars]
        cut = text.find("\n\n")
        text = text[cut + 2:] if 0 <= cut < 1500 and start > 0 else text
    return f"المقالة: {a['title']}\n\n{text.strip()}"


def build(args) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    rng = random.Random(41)
    arts = {a["id"]: a for a in load_articles()}
    qa = [json.loads(l) for l in open(OUT / "qa.jsonl", encoding="utf-8")]
    by_article: dict[str, list] = {}
    for r in qa:
        for p in r["pairs"]:
            by_article.setdefault(r["article"], []).append({**p, "section_index": r["section_index"]})
    summary = Counter()
    for pool, n_long, n_lm in (("train", args.n_long, args.n_lm), ("dev", args.n_long_dev, 0)):
        sources = [aid for aid in by_article if arts[aid]["pool"] == pool]
        rng.shuffle(sources)
        fillers = [a for a in arts.values() if a["pool"] == pool and a["id"] not in by_article and a["tokens"] < 6000]
        examples = []
        # every pair is asked once; an article can come back in later examples with another of its questions
        remaining = {aid: rng.sample(by_article[aid], len(by_article[aid])) for aid in sources}
        for k in range(n_long):
            target = rng.randint(6000, 16000) - 300  # room for the three answers
            open_ids = [aid for aid in sources if remaining[aid]]
            if len(open_ids) < 3:
                break
            picked = rng.sample(open_ids, 3)
            docs, questions = [], []
            for aid in picked:
                a = arts[aid]
                p = remaining[aid].pop()
                sec = sections(a["text"])[p["section_index"]]
                docs.append(doc_text(a, tok, sec))
                q = f"بحسب مقالة «{a['title']}»، {p['q']}"
                others = [x["a"] for x in by_article[aid] if x["a"] != p["a"]]
                # multiple choice only when the same article gives three other answers: same topic, so the wrong
                # options are plausible (options from unrelated articles made the question trivially easy)
                wrong = list(dict.fromkeys(others))
                if len(wrong) >= 3 and rng.random() < 0.4:
                    opts = rng.sample(wrong, 3) + [p["a"]]
                    rng.shuffle(opts)
                    letter = LETTERS[opts.index(p["a"])]
                    q += "\n" + "\n".join(f"{L}. {o}" for L, o in zip(LETTERS, opts)) + "\n" + MCQ_ASK
                    questions.append((q, f"{letter}. {p['a']}", "mcq"))
                else:
                    questions.append((q, p["a"], "open"))
            used = sum(n_tok(d) for d in docs)
            fill = []
            for f in rng.sample(fillers, min(60, len(fillers))):
                if used >= target:
                    break
                t = n_tok(doc_text(f, tok))
                if used + t > target + 400:
                    continue
                fill.append(doc_text(f, tok))
                used += t
            all_docs = docs + fill
            rng.shuffle(all_docs)
            depth = [all_docs.index(d) / max(1, len(all_docs) - 1) for d in docs]
            messages = []
            for i, (q, ans, kind) in enumerate(questions):
                body = (LONG_INTRO + "\n\n---\n\n".join(all_docs) + "\n\nالسؤال: " + q) if i == 0 else "السؤال: " + q
                messages += [{"role": "user", "content": body}, {"role": "assistant", "content": ans}]
            n = n_tok(tok.apply_chat_template(messages, tokenize=False))
            if n > 16384:
                summary[f"{pool}_too_long"] += 1
                continue
            examples.append({"id": f"{pool}-long-{k}", "kind": "longqa", "tokens": n, "messages": messages,
                             "question_kinds": [x[2] for x in questions], "source_depths": [round(d, 2) for d in depth]})
        # plain long text: single long articles, or neighbours packed together, 8K-16K tokens
        long_arts = [a for a in arts.values() if a["pool"] == pool and a["id"] not in by_article and a["tokens"] >= 8000]
        rng.shuffle(long_arts)
        for k, a in enumerate(long_arts[:n_lm]):
            text = a["text"]
            ids = tok(text, add_special_tokens=False)["input_ids"][:16000]
            examples.append({"id": f"{pool}-lm-{k}", "kind": "lm", "tokens": len(ids) + 1, "text": tok.decode(ids)})
        # replay: the original model's own short answers (no Latin letters, not cut off at the 512-token limit)
        if (OUT / "replay.jsonl").exists():
            for r in map(json.loads, open(OUT / "replay.jsonl", encoding="utf-8")):
                if r["pool"] != pool or r["latin"] or not r["answer"] or n_tok(r["answer"]) >= 510:
                    summary[f"{pool}_replay_skipped"] += 1
                    continue
                msgs = [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["answer"]}]
                examples.append({"id": f"{pool}-replay-{r['key']}", "kind": "replay",
                                 "tokens": n_tok(tok.apply_chat_template(msgs, tokenize=False)), "messages": msgs})
        rng.shuffle(examples)
        with open(OUT / f"{pool}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for e in examples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        for e in examples:
            summary[f"{pool}_{e['kind']}"] += 1
            summary[f"{pool}_{e['kind']}_tokens"] += e["tokens"]
    print(json.dumps(summary, indent=1))
    (OUT / "build.stats.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


def stats(_args) -> None:
    qa = [json.loads(l) for l in open(OUT / "qa.jsonl", encoding="utf-8")]
    rej = Counter(x for r in qa for x in r["rejected"])
    print({"sections": len(qa), "parsed": sum(r["parsed"] for r in qa), "pairs_kept": sum(len(r["pairs"]) for r in qa),
           "rejected": dict(rej)})
    if (OUT / "replay.jsonl").exists():
        rp = [json.loads(l) for l in open(OUT / "replay.jsonl", encoding="utf-8")]
        print({"replay": len(rp), "with_latin": sum(r["latin"] for r in rp)})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("articles").set_defaults(fn=articles)
    for name, fn in (("genqa", genqa), ("replay", replay)):
        p = sub.add_parser(name)
        p.add_argument("--server", default="http://127.0.0.1:8095")
        p.add_argument("--n", type=int, default=1500, help="articles to draw from")
        p.add_argument("--parallel", type=int, default=4)
        p.set_defaults(fn=fn)
    b = sub.add_parser("build")
    b.add_argument("--n-long", type=int, default=450)
    b.add_argument("--n-lm", type=int, default=100)
    b.add_argument("--n-long-dev", type=int, default=40)
    b.set_defaults(fn=build)
    sub.add_parser("stats").set_defaults(fn=stats)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
