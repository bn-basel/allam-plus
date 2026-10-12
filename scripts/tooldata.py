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
CONVOS_FIX = OUT / "convos_fix.jsonl"  # the fix round (docs/DECISIONS.md D23)
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
# The fix round (docs/DECISIONS.md D23): three skills the Phase 4 agent test showed missing.
FIX_KINDS = {
    "dependent": (35, "The user asks, in ONE message, for two things where the second depends on the answer to the first (for example: find something and then book it; get a price and then convert or pay it; look up a booking and then cancel or change it). The assistant calls ONLY the first tool, waits for its result, then calls the second tool in a NEW turn using a value taken from that result (an id, a price, a date, a number), never calling both at once and never guessing the value; then it answers."),
    "clarify_action": (30, "The user asks the assistant to DO something (book, reserve, order, pay, schedule, renew, send, register) but leaves out two or more REQUIRED details, for example both the date and the time, or the place and the amount. The assistant makes NO tool call and NEVER fills in a missing detail itself: it asks one short question, ending with a question mark (؟), that names every missing detail. The user answers, then the assistant calls the tool and confirms from the result."),
    "compute": (35, "The answer needs simple arithmetic on tool results or on numbers the user gave: a total for several items, a quantity times a price, the difference between two results, what is left after paying an amount, the sum of two results. The assistant gets the numbers with the tools, then does the arithmetic ITSELF in its reply and states the correct result (it may show the calculation briefly); it never uses a tool to do arithmetic. In about a third of the conversations the arithmetic is asked in a follow-up message about an earlier result, and the assistant answers that follow-up without a new tool call."),
}
DIALECTS = {"msa":(55, "Modern Standard Arabic"),
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
- The assistant's final answers use only facts from the tool results and the conversation: no invented numbers, prices, times or names.{facts}
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


def computed(said: set[str]) -> list[float]:
    """Every sum, difference, product and quotient of two numbers in the conversation so far (the fix round lets a
    reply state a number it worked out, but only one that is exactly such a result: wrong arithmetic is rejected)."""
    vals = sorted({float(n) for n in said if len(n) < 12})[:60]
    out = []
    for i, a in enumerate(vals):
        for b in vals[i:]:
            out += [a + b, abs(a - b), a * b] + ([a / b] if b else []) + ([b / a] if a else [])
    return out


def is_computed(n: str, said: set[str]) -> bool:
    v = float(n)
    return any(abs(v - c) < 0.006 or (abs(c) >= 100 and abs(v - round(c)) < 0.5 and v == int(v)) for c in computed(said))


EQUATION = re.compile(r"((?:\d+(?:\.\d+)?\s*[-+×x*÷/]\s*)+\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)")


def checked_equations(text: str, said: set[str]) -> set[str]:
    """Results of equations written out in a reply (e.g. 1200 + 800 + 300 = 2300) whose terms were all said before
    (or are earlier results in the same reply) and whose arithmetic is right; these count as said from then on."""
    t = re.sub(r"(?<=\d)[,،٬](?=\d{3})", "", normalize_ar(text))
    known, ok = set(said), set()
    for left, right in EQUATION.findall(t):
        terms = re.findall(r"\d+(?:\.\d+)?", left)
        if not all(x in known or len(x) <= 1 for x in terms):
            continue
        try:
            value = eval(left.replace("×", "*").replace("x", "*").replace("÷", "/"), {"__builtins__": {}})
        except (SyntaxError, ZeroDivisionError):
            continue
        if abs(value - float(right)) < 0.006 or (abs(value) >= 100 and abs(value - float(right)) < 0.5):
            known.add(right.rstrip("."))
            ok.add(right.rstrip("."))
        else:  # written out with the wrong result (e.g. 40 - 105 = 65): the reply would teach bad arithmetic
            ok.add("WRONG")
    return ok


ACTION = re.compile(r"^(book|reserve|create|submit|pay|transfer|schedule|place|request|renew|send|add|cancel|set|order|"
                    r"apply|register|make|buy|update)_")


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
    fix = kind in FIX_KINDS
    any_computed, user_turns_before_second = False, None
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            if pending:
                return "call before results"
            calls = m["tool_calls"]
            calls_per_turn.append(len(calls))
            if len(calls_per_turn) == 2:
                user_turns_before_second = sum(x.get("role") == "user" for x in msgs[:i])
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
                if fix:
                    worked = checked_equations(text, said)
                    if "WRONG" in worked:
                        return "wrong arithmetic"
                    any_computed |= bool(worked - said)
                    said |= worked
                for n in nums(text):
                    if n in said or len(n) <= 1:
                        continue
                    if fix and is_computed(n, said):  # fix round: worked out exactly from numbers already there
                        any_computed = True
                        continue
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
    if kind in ("clarify", "clarify_action") and (msgs[1].get("tool_calls") or len(msgs) < 4 or
                                                 msgs[2].get("role") != "user"):
        return "no clarifying question"
    if kind == "clarify_action":
        if "؟" not in msgs[1].get("content", "") and "?" not in msgs[1].get("content", ""):
            return "no clarifying question"
        if not any(ACTION.match(c.get("name", "")) for m in msgs if m.get("tool_calls") for c in m["tool_calls"]):
            return "no action call"
    if kind == "dependent":  # two calls, the second after the first one's result, both for the first message
        if len(calls_per_turn) < 2:
            return "no second call"
        if user_turns_before_second != 1:
            return "not one request"
    if kind == "compute" and not any_computed:
        return "no computed number"
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
    fix = args.round == "fix"
    path, table = (CONVOS_FIX, FIX_KINDS) if fix else (CONVOS, KINDS)
    facts = " Numbers the assistant works out from them (totals, differences, quantities) must be exactly right." if fix else ""
    done = sum(1 for _ in open(path, encoding="utf-8")) if path.exists() else 0
    rng = random.Random((5000 if fix else 1000) + done)
    calls_needed = max(0, (args.n - done) // args.per_call)
    kinds, kw = list(table), [table[k][0] for k in table]
    jobs = []
    lookup = re.compile(r"^(search|list|get|check|find|track|estimate|lookup|calculate)_")
    for j in range(calls_needed):
        kind = rng.choices(kinds, kw)[0]
        need = []  # tools the scenario cannot do without
        if kind in ("clarify_action", "dependent"):  # an action to ask about, or a lookup that feeds an action
            d = rng.choice([x for x in domains if any(ACTION.match(t["name"]) for t in lib[x])])
            need = [rng.choice([t for t in lib[d] if ACTION.match(t["name"])])]
            if kind == "dependent" and any(lookup.match(t["name"]) for t in lib[d]):
                need.append(rng.choice([t for t in lib[d] if lookup.match(t["name"])]))
        elif kind == "compute":  # tools that return prices, amounts or counts
            d = rng.choice([x for x in domains if any(w in x for w in ("shopping", "flights", "hotels", "restaurants",
                                                                          "events", "bills", "banking", "real estate",
                                                                          "taxis", "jobs"))])
        else:
            d = rng.choice(domains)
        rest = [t for t in lib[d] if t not in need]
        offered = need + rng.sample(rest, min(len(rest), max(0, rng.randint(2, 4) - len(need))))
        other = rng.choice([x for x in domains if x != d])
        offered.append(rng.choice(lib[other]))  # one tool from another domain, as a distractor
        rng.shuffle(offered)
        how = table[kind][1]
        if kind == "dependent" and len(need) == 2:
            how += (f" In these conversations the two steps are: first `{need[1]['name']}`, then `{need[0]['name']}` "
                    f"(the user's message asks for both).")
        elif kind == "dependent":
            how += f" In these conversations the second step is `{need[0]['name']}`."
        elif kind == "clarify_action":
            how += f" In these conversations the action is `{need[0]['name']}`."
        dialect = rng.choices(list(DIALECTS), [v[0] for v in DIALECTS.values()])[0]
        day = date(2026, 1, 1) + timedelta(days=rng.randint(0, 364))
        today = (f"Today is {day.isoformat()}; the assistant knows this (it is in its system prompt)."
                 if fix or rng.random() < 0.4 else "")  # fix round: always, so no date is ever guessed
        jobs.append({"job": done // args.per_call + j, "domain": d, "kind": kind, "dialect": dialect,
                     "today": day.isoformat() if today else None, "tools": [t["name"] for t in offered],
                     "prompt": CONVO_PROMPT.format(n=args.per_call, kind=how, dialect=DIALECTS[dialect][1],
                                                   today=today, facts=facts,
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
    with open(path, "a", encoding="utf-8", newline="\n") as f, ThreadPoolExecutor(args.parallel) as ex:
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
                rec = {"id": f"{'f' if fix else 't'}{job['job']:04d}-{i}", "domain": job["domain"], "kind": job["kind"],
                       "dialect": job["dialect"], "today": job["today"], "tools": job["tools"],
                       "messages": conv.get("messages") if isinstance(conv, dict) else conv, "rejected": why}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if k % 10 == 0:
                print(f"{k + 1}/{len(jobs)} calls, ${budget.spent:.3f} spent, {dict(stats)}", flush=True)
    print(f"done: ${budget.spent:.3f} spent in total; {dict(stats)}")


def recheck(args) -> None:
    """Re-apply today's checks to every written conversation (after a rule change), without new API calls."""
    lib = json.loads(LIBRARY.read_text(encoding="utf-8"))
    all_tools = {t["name"]: t for d in lib.values() for t in d}
    bench = benchmark_trigrams()
    path = CONVOS_FIX if args.round == "fix" else CONVOS
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    for r in rows:
        r["rejected"] = check({"messages": r["messages"]}, {n: all_tools[n] for n in r["tools"]}, r["kind"], bench)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    stats(args)


def stats(args) -> None:
    rows = [json.loads(l) for l in open(CONVOS_FIX if args.round == "fix" else CONVOS, encoding="utf-8")]
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
    for name, fn in (("stats", stats), ("recheck", recheck)):
        sub.add_parser(name).set_defaults(fn=fn)
    for p in sub.choices.values():
        if p.prog.split()[-1] != "library":
            p.add_argument("--round", choices=["main", "fix"], default="main",
                           help="fix: the fix-round conversations (FIX_KINDS, data/tools/convos_fix.jsonl)")
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
