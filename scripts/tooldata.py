"""Phase 3 data: general Arabic tool-calling conversations from our rule-driven generation pipeline, whose base
model is DeepSeek V4.1 Flash (via OpenRouter, providers pinned, data collection denied, hard USD cap).

    uv run python scripts/tooldata.py library                 # tool definitions for 18 everyday domains
    uv run python scripts/tooldata.py convos --n 1800         # conversations, 4 per call, checked by rules
    uv run python scripts/tooldata.py stats

Everything that ends up in training text (tools, user requests, tool results, replies) is written by the pipeline;
this script only gives instructions, picks the mix and checks the output. Kept out:
- tools whose name matches a tool in any of our benchmarks (ArabFuncBench, AISA-ArabicFC dev/test), so the tests
  stay unseen;
- conversations whose user turns overlap a benchmark request (word-trigram overlap >= 0.5);
- any Latin letter in user or assistant text (the no-English, no-Arabizi rule); Latin is allowed only in tool names,
  JSON keys and enum codes the tool itself defines.
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, normalize_ar  # noqa: E402
from deepseek import Budget, DeepSeek  # noqa: E402

OUT = ROOT / "data" / "tools"
LIBRARY = OUT / "library.json"
CONVOS = OUT / "convos.jsonl"
LATIN = re.compile(r"[A-Za-z]")
CAP_USD = 4.0  # Basel's key holds $4.60 (2026-10-11); stay under it

DOMAINS = [
    "government e-services (IDs, residency permits, traffic fines, appointments)", "banking and payments",
    "healthcare and clinics", "education (schools and universities)", "online shopping and order tracking",
    "flights and travel", "hotels and holiday rentals", "taxis, buses and public transport", "weather and air quality",
    "Islamic services (prayer times, Quran, zakat, Hajj and Umrah)", "real estate and home rentals",
    "electricity, water and mobile/internet bills", "restaurants and food delivery",
    "events, cinema and entertainment tickets", "jobs and human resources", "sports and fitness",
    "personal productivity (calendar, reminders, notes, messages)", "news, encyclopedia lookups and unit conversion",
]

LIBRARY_PROMPT = """Design {k} different tools (functions) that an Arabic AI assistant for people in Saudi Arabia could call in this domain: {domain}.
Rules:
- "name": English snake_case, specific and natural (e.g. check_flight_status).
- "description" and every parameter "description": clear Modern Standard Arabic, one sentence.
- "parameters": a JSON schema object with "type": "object", "properties" and "required". Parameter types: string, integer, number or boolean; use "enum" (short English codes) only where a fixed set of options is natural; for dates say in the description that the format is YYYY-MM-DD.
- 1 to 5 parameters per tool; mix required and optional ones.
- Generic tools only: no real company, bank, app or brand names.
Output only a JSON array of {k} objects: [{{"name": ..., "description": ..., "parameters": {{...}}}}, ...]"""

KINDS = {  # scenario: (weight, instruction)
    "single": (30, "The user asks for something one of the tools does. The assistant calls exactly one tool with the right arguments, gets the result, then answers from it."),
    "parallel": (10, "The user asks for two independent things in one message. The assistant calls two tools (or the same tool twice) in ONE assistant turn, gets both results, then answers both."),
    "chain": (14, "The user's need takes two steps: the assistant calls a first tool, then uses a value from its result as an argument of a second call, then answers."),
    "clarify": (12, "The user asks for something a tool does but leaves out a REQUIRED detail. The assistant does not call any tool yet: it asks one short question for the missing detail. The user answers, then the assistant calls the tool and answers from the result."),
    "no_tool": (18, "None of the tools fits: the user asks a general-knowledge question, makes small talk, or asks for something no tool can do. The assistant makes NO tool call: it answers from general knowledge, or says politely that it cannot do that, briefly."),
    "error": (6, "The assistant calls a tool but the result is an error (e.g. not found, invalid number, service unavailable). The assistant explains the problem plainly and suggests what to do or asks for a corrected detail; no invented data."),
    "multi_turn": (10, "Two requests in sequence: the user asks, the assistant calls a tool and answers; then the user asks a follow-up that needs another call, and the assistant calls and answers again."),
}
DIALECTS = {"msa": (55, "Modern Standard Arabic"),
            "saudi": (45, "everyday Saudi Arabic as written in chat apps (e.g. أبي، وش، كم سعر، الحين، بكرة)")}

CONVO_PROMPT = """Write {n} different realistic conversations between a user and an Arabic AI assistant that can call the tools below.
Scenario for all {n}: {kind}
The user writes in {dialect}. The assistant always writes clear, polite Modern Standard Arabic, briefly.
{today}
Tools:
{tools}

Rules:
- Tool-call arguments follow the tool's parameters exactly: names, types, enum values; every required parameter given (except where the scenario says it is missing and the assistant asks). Never invent an argument value the user did not give; put values in Arabic as the user wrote them (names, cities, places), numbers as numbers, dates as YYYY-MM-DD.
- Tool results are realistic JSON objects with Arabic text values.
- The assistant's final answers use only facts from the tool results and the conversation: no invented numbers, prices, times or names.
- No Latin letters anywhere in user or assistant text: no English words, no Arabic written in Latin letters. Digits are fine.
- The {n} conversations differ in user, need, wording, length and details.
Output only a JSON array of {n} objects, each {{"messages": [...]}}, where every message is one of:
{{"role": "user", "content": "..."}}
{{"role": "assistant", "content": "..."}}
{{"role": "assistant", "tool_calls": [{{"name": "...", "arguments": {{...}}}}]}}
{{"role": "tool", "name": "...", "content": "<the result as a JSON string>"}}"""

WEEKDAYS_AR = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]


def parse_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = min([i for i in (text.find("["), text.find("{")) if i >= 0], default=-1)
    if start < 0:
        return None
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        try:
            return json.loads(text[start: max(text.rfind("]"), text.rfind("}")) + 1])
        except json.JSONDecodeError:
            return None


def benchmark_tool_names() -> set[str]:
    import pandas as pd
    names = set()
    afb = ROOT / "data" / "raw" / "arabfuncbench" / "arab_func_bench_tools.json"
    for tools in json.loads(afb.read_text(encoding="utf-8")).values():
        names |= {t["name"] for t in tools}
    for split in ("dev", "test"):
        for r in pd.read_parquet(ROOT / "data" / "raw" / "aisa" / f"{split}.parquet").to_dict("records"):
            names |= {t["function"]["name"] for t in r["tools_sampled"]}
    return names


def benchmark_trigrams() -> list[set]:
    import pandas as pd
    texts = [e["utterance"] for e in json.loads((ROOT / "data" / "raw" / "arabfuncbench" /
                                                 "arab_func_bench_examples.json").read_text(encoding="utf-8"))]
    for split in ("dev", "test"):
        for r in pd.read_parquet(ROOT / "data" / "raw" / "aisa" / f"{split}.parquet").to_dict("records"):
            texts += [m["content"] for m in r["messages"] if m["role"] == "user"]
    return [trigrams(t) for t in texts]


def trigrams(text: str) -> set:
    w = normalize_ar(text).split()
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


# ---------- 1. tool library ----------

def library(args) -> None:
    ds, budget = DeepSeek(), Budget(CAP_USD, OUT / "spend.jsonl")
    banned = benchmark_tool_names()
    lib = json.loads(LIBRARY.read_text(encoding="utf-8")) if LIBRARY.exists() else {}

    def work(domain):
        r = ds.chat("You design clean, realistic API tools.", LIBRARY_PROMPT.format(k=args.k, domain=domain), 4000, 0.7)
        if "error" in r:
            return domain, None, r["error"]
        budget.add(r["usd"], f"library {domain}")
        return domain, parse_json(r["text"]), None

    todo = [d for d in DOMAINS if d not in lib]
    with ThreadPoolExecutor(6) as ex:
        for domain, tools, err in ex.map(work, todo):
            if err or not isinstance(tools, list):
                print(f"{domain}: failed ({err or 'no JSON'})")
                continue
            keep = []
            for t in tools:
                ok = (isinstance(t, dict) and re.fullmatch(r"[a-z][a-z0-9_]{2,40}", str(t.get("name", ""))) and
                      isinstance(t.get("parameters"), dict) and t["parameters"].get("properties"))
                if ok and t["name"] not in banned and not LATIN.search(t.get("description", "")):
                    t["parameters"].setdefault("type", "object")
                    t["parameters"].setdefault("required", [])
                    keep.append(t)
            lib[domain] = keep
            print(f"{domain}: {len(keep)}/{len(tools)} kept")
    LIBRARY.write_text(json.dumps(lib, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{sum(map(len, lib.values()))} tools in {len(lib)} domains; spent ${budget.spent:.4f}")


# ---------- 2. conversations ----------

def nums(text: str) -> set[str]:
    """Numbers in a text, written the same way: Hindi digits as Arabic digits, no thousands separators."""
    t = re.sub(r"(?<=\d)[,،٬](?=\d{3})", "", normalize_ar(str(text)))
    return {n.rstrip(".") for n in re.findall(r"\d+(?:\.\d+)?", t)} | {n.split(".")[0] for n in re.findall(r"\d+\.0+(?!\d)", t)}


def type_ok(v, spec: dict) -> bool:
    t = spec.get("type")
    if "enum" in spec and v not in spec["enum"]:
        return False
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "boolean":
        return isinstance(v, bool)
    if t == "string":
        return isinstance(v, str) and v.strip() != ""
    return True


def check(conv: dict, tools: dict, kind: str, bench: list[set]) -> str | None:
    """None if the conversation is usable, else the first rule it breaks."""
    msgs = conv.get("messages") if isinstance(conv, dict) else None
    if not isinstance(msgs, list) or len(msgs) < 2 or msgs[0].get("role") != "user":
        return "structure"
    calls_per_turn, pending, results_text = [], 0, ""
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            if pending:
                return "call before results"
            calls = m["tool_calls"]
            calls_per_turn.append(len(calls))
            for c in calls:
                t = tools.get(c.get("name"))
                if not t:
                    return "unknown tool"
                props, args = t["parameters"].get("properties", {}), c.get("arguments")
                if not isinstance(args, dict) or set(args) - set(props):
                    return "bad argument names"
                if any(r not in args for r in t["parameters"].get("required", [])):
                    return "missing required argument"
                for k, v in args.items():
                    if not type_ok(v, props[k]):
                        return "argument type"
                    if isinstance(v, str) and LATIN.search(v) and v not in props[k].get("enum", []) and \
                            not re.fullmatch(r"[\w.+-]+@[\w.-]+", v):
                        return "latin argument"
            pending = len(calls)
        elif role == "tool":
            if pending <= 0:
                return "result without call"
            pending -= 1
            results_text += " " + str(m.get("content", ""))
        elif role in ("user", "assistant"):
            if pending:
                return "results missing"
            text = m.get("content")
            if not isinstance(text, str) or not text.strip():
                return "empty turn"
            # Latin: never in user text; in a reply only as a code copied from a tool result (e.g. ELC-4721)
            latin = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text)
            if latin and (role == "user" or any(w not in results_text for w in latin)):
                return "latin text"
            if role == "assistant" and results_text:  # grounding: every number stated was said or returned before
                said = nums(" ".join(str(x.get("content", "")) for x in msgs[:i]) + results_text)
                if any(n not in said and len(n) > 1 for n in nums(text)):
                    return "number not from results"
        else:
            return "bad role"
    if pending:
        return "results missing"
    if msgs[-1].get("role") != "assistant" or msgs[-1].get("tool_calls"):
        return "does not end with a reply"
    n_calls = sum(calls_per_turn)
    if kind == "no_tool" and n_calls:
        return "call in no-tool scenario"
    if kind != "no_tool" and not n_calls:
        return "no call"
    if kind == "parallel" and max(calls_per_turn, default=0) < 2:
        return "not parallel"
    if kind == "clarify" and (msgs[1].get("tool_calls") or len(msgs) < 4 or msgs[2].get("role") != "user"):
        return "no clarifying question"
    user = trigrams(" ".join(m["content"] for m in msgs if m["role"] == "user"))
    if user and any(len(user & b) / max(1, min(len(user), len(b))) >= 0.5 for b in bench if b):
        return "overlaps a benchmark request"
    return None


def convos(args) -> None:
    lib = json.loads(LIBRARY.read_text(encoding="utf-8"))
    domains = [d for d in lib if len(lib[d]) >= 3]
    all_tools = {t["name"]: t for d in lib.values() for t in d}
    ds, budget = DeepSeek(), Budget(CAP_USD, OUT / "spend.jsonl")
    bench = benchmark_trigrams()
    done = sum(1 for _ in open(CONVOS, encoding="utf-8")) if CONVOS.exists() else 0
    rng = random.Random(1000 + done)
    calls_needed = max(0, (args.n - done) // args.per_call)
    kinds, kw = list(KINDS), [KINDS[k][0] for k in KINDS]
    jobs = []
    for j in range(calls_needed):
        d = rng.choice(domains)
        offered = rng.sample(lib[d], min(len(lib[d]), rng.randint(2, 4)))
        other = rng.choice([x for x in domains if x != d])
        offered.append(rng.choice(lib[other]))  # one tool from another domain, as a distractor
        rng.shuffle(offered)
        kind = rng.choices(kinds, kw)[0]
        dialect = rng.choices(list(DIALECTS), [v[0] for v in DIALECTS.values()])[0]
        day = date(2026, 1, 1) + timedelta(days=rng.randint(0, 364))
        today = (f"Today is {day.isoformat()}; the assistant knows this (it is in its system prompt)."
                 if rng.random() < 0.4 else "")
        jobs.append({"job": done // args.per_call + j, "domain": d, "kind": kind, "dialect": dialect,
                     "today": day.isoformat() if today else None, "tools": [t["name"] for t in offered],
                     "prompt": CONVO_PROMPT.format(n=args.per_call, kind=KINDS[kind][1], dialect=DIALECTS[dialect][1],
                                                   today=today,
                                                   tools="\n".join(json.dumps(t, ensure_ascii=False) for t in offered))})

    def work(job):
        if not budget.ok() or budget.spent >= args.budget:
            return job, None, "budget"
        r = ds.chat("You write realistic, varied Arabic training conversations for a tool-calling assistant, "
                    "and you follow format rules exactly.", job["prompt"], 7000, 0.9)
        if "error" in r:
            return job, None, r["error"]
        budget.add(r["usd"], f"convos job {job['job']}")
        return job, r, None

    stats = Counter()
    with open(CONVOS, "a", encoding="utf-8", newline="\n") as f, ThreadPoolExecutor(args.parallel) as ex:
        for k, (job, r, err) in enumerate(ex.map(work, jobs)):
            if err:
                stats[f"call failed: {err}"] += 1
                continue
            items = parse_json(r["text"])
            if not isinstance(items, list):
                stats["no JSON"] += 1
                continue
            tools = {n: all_tools[n] for n in job["tools"]}
            for i, conv in enumerate(items):
                why = check(conv, tools, job["kind"], bench)
                stats[why or "kept"] += 1
                rec = {"id": f"t{job['job']:04d}-{i}", "domain": job["domain"], "kind": job["kind"],
                       "dialect": job["dialect"], "today": job["today"], "tools": job["tools"],
                       "messages": conv.get("messages") if isinstance(conv, dict) else conv, "rejected": why}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if k % 10 == 0:
                print(f"{k + 1}/{len(jobs)} calls, ${budget.spent:.3f} spent, {dict(stats)}", flush=True)
    print(f"done: ${budget.spent:.3f} spent in total; {dict(stats)}")


def recheck(_args) -> None:
    """Re-apply today's checks to every written conversation (after a rule change), without new API calls."""
    lib = json.loads(LIBRARY.read_text(encoding="utf-8"))
    all_tools = {t["name"]: t for d in lib.values() for t in d}
    bench = benchmark_trigrams()
    rows = [json.loads(l) for l in open(CONVOS, encoding="utf-8")]
    for r in rows:
        r["rejected"] = check({"messages": r["messages"]}, {n: all_tools[n] for n in r["tools"]}, r["kind"], bench)
    CONVOS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    stats(None)


def stats(_args) -> None:
    rows = [json.loads(l) for l in open(CONVOS, encoding="utf-8")]
    kept = [r for r in rows if not r["rejected"]]
    print({"written": len(rows), "kept": len(kept), "rejected": dict(Counter(r["rejected"] for r in rows if r["rejected"])),
           "kept_by_kind": dict(Counter(r["kind"] for r in kept)), "kept_by_dialect": dict(Counter(r["dialect"] for r in kept))})
    spend = OUT / "spend.jsonl"
    if spend.exists():
        print(f"spent ${sum(json.loads(l)['usd'] for l in open(spend, encoding='utf-8')):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("library")
    p.add_argument("--k", type=int, default=6)
    p.set_defaults(fn=library)
    p = sub.add_parser("convos")
    p.add_argument("--n", type=int, default=1800, help="conversations to write in total (kept + rejected)")
    p.add_argument("--per-call", type=int, default=4)
    p.add_argument("--parallel", type=int, default=6)
    p.add_argument("--budget", type=float, default=3.5, help="stop once the ledger reaches this many USD")
    p.set_defaults(fn=convos)
    sub.add_parser("stats").set_defaults(fn=stats)
    sub.add_parser("recheck").set_defaults(fn=recheck)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
