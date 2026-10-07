"""Arabic tool-calling benchmarks: AISA-ArabicFC (dev, five dialects) and ArabFuncBench (MSA).

    uv run python scripts/tools_bench.py --bench aisa_dev --protocol text [--adapter ...] [--limit 50]
    uv run python scripts/tools_bench.py --bench arabfuncbench --protocol paper
    uv run python scripts/tools_bench.py --bench arabfuncbench --protocol text

Each case: one Arabic user request, a few candidate tools, and either one expected call (name + arguments) or "no
call" (the request needs none of the tools). The model gets one turn and we read its first tool call.

Protocols:
  text   the plain-text tool format the receptionist ALLaM was trained on (scripts/models.py TEXT_TOOL_GUIDE): tools
         described in the system prompt, the model answers <tool_call>{"name": ..., "arguments": {...}}</tool_call>
         or a normal reply. Phase 3 re-runs the base with the new native template as well.
  paper  ArabFuncBench's own ALLaM protocol (their notebook): an Arabic instruction to answer with JSON
         {"function_called": ..., "arguments": ...}, tools and question in one user turn. Used only to reproduce
         their published ALLaM number; we decode greedily, they sampled at temperature 0.1.

Scores:
  fn_acc   right tool, or rightly no tool (AISA "FnAcc"; ArabFuncBench "TSA").
  arg_em   right tool and every argument right after normalisation (Arabic letters/diacritics, both digit sets,
           numbers as numbers, list order), extra empty arguments ignored. Our normalisation is simpler than the AISA
           leaderboard's (no alias tables like الأردن = Jordan), so it is a little stricter.
  aef1     ArabFuncBench's argument score: share of expected arguments whose value is ≥ 0.65 similar (difflib
           ratio) to the predicted one, over cases with the right tool.
  lcr      ArabFuncBench's language compliance: share of predicted argument values that are Arabic (or numbers).
  call_rate, wellformed: how often the model called a tool at all, and how many calls parsed.
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, normalize_ar  # noqa: E402
from models import parse_text_tool_calls, text_tool_messages  # noqa: E402

RAW = ROOT / "data" / "raw"
SYSTEM = "أنت مساعد ذكي يجيب باللغة العربية."
PAPER_PROMPT = (  # ArabFuncBench notebook, ALLaM cell, word for word
    "أنت مساعد ذكي يجيب على استفسارات المستخدمين باللغة العربية.\n"
    "لديك مجموعة من الأدوات. استخدمها فقط عند الحاجة.\n"
    "إذا لم تكن هناك أداة مناسبة، أجب بـ: {\"function_called\": null, \"arguments\": null}\n\n"
    "قواعد صارمة للإجابة:\n"
    "1. أجب بـ JSON فقط — لا نص إضافي\n"
    "2. في حقل arguments، ضع قيم المعاملات فقط — لا تنسخ مخطط الأداة\n"
    "3. مثال صحيح: {\"function_called\": \"get_prayer_times\", \"arguments\": {\"city\": \"الرياض\"}}\n"
    "4. مثال خاطئ: {\"function_called\": \"get_prayer_times\", \"arguments\": {\"type\": \"object\", \"properties\": {...}}}\n"
    "الشكل المطلوب:\n"
    "{\"function_called\": \"اسم_الأداة\" أو null, \"arguments\": {\"param\": \"قيمة\"} أو null}"
)


# ---------- loaders: -> {id, system, user, tools (OpenAI style), gold_name or None, gold_args, group} ----------

def _clean(v):
    """Parquet pads every tool with every property name set to None; drop those and turn arrays into lists."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items() if x is not None}
    if hasattr(v, "tolist"):
        return _clean(v.tolist())
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


def load_aisa(split: str) -> list[dict]:
    import pandas as pd
    cases = []
    for i, r in enumerate(pd.read_parquet(RAW / "aisa" / f"{split}.parquet").to_dict("records")):
        msgs = list(r["messages"])
        dev = next((m["content"] for m in msgs if m["role"] in ("developer", "system")), "")
        user = next(m["content"] for m in msgs if m["role"] == "user")
        gold_name = None if r["tool_called"] in (None, "none") else r["tool_called"]
        gold_args = {}
        for m in msgs:
            if m["role"] == "assistant" and m.get("tool_calls") is not None and len(m["tool_calls"]):
                gold_args = _clean(m["tool_calls"][0]["function"]["arguments"]) or {}
        tools = [{"type": "function", "function": _clean(t["function"])} for t in r["tools_sampled"]]
        cases.append({"id": f"{split}-{i}", "system": (SYSTEM + "\n" + dev.strip()).strip(), "user": user, "tools": tools,
                      "gold_name": gold_name, "gold_args": gold_args, "group": r["dialect"]})
    return cases


def load_arabfuncbench() -> list[dict]:
    d = RAW / "arabfuncbench"
    examples = json.loads((d / "arab_func_bench_examples.json").read_text(encoding="utf-8"))
    tools_by_domain = json.loads((d / "arab_func_bench_tools.json").read_text(encoding="utf-8"))
    cases = []
    for ex in examples:
        domain_tools = {t["name"]: t for t in tools_by_domain[ex["domain"]]}
        avail = [domain_tools[n] for n in ex["available_tools"] if n in domain_tools]
        cases.append({"id": ex["id"], "system": SYSTEM, "user": ex["utterance"],
                      "tools": [{"type": "function", "function": t} for t in avail], "paper_tools": avail,
                      "gold_name": None if ex["is_negative"] else ex["expected_function"],
                      "gold_args": ex["expected_arguments"] or {}, "group": ex["domain"]})
    return cases


BENCHES = {"aisa_dev": lambda: load_aisa("dev"), "aisa_test": lambda: load_aisa("test"),
           "arabfuncbench": load_arabfuncbench}


# ---------- prompts and parsing ----------

def build_prompt(tok, case: dict, protocol: str) -> str:
    if protocol == "text":
        msgs = text_tool_messages(case["system"], [{"role": "user", "content": case["user"]}], case["tools"])
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    tools_str = json.dumps(case.get("paper_tools") or [t["function"] for t in case["tools"]], ensure_ascii=False, indent=2)
    user = PAPER_PROMPT + "\n\n" + "الأدوات المتاحة:\n" + tools_str + "\n\nاستفسار المستخدم: " + case["user"]
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


def parse(raw: str, protocol: str) -> dict:
    if protocol == "text":
        _, calls = parse_text_tool_calls(raw)
        if not calls:
            return {"name": None, "args": {}, "called": False, "wellformed": None}
        c = calls[0]
        ok = c["name"] is not None and isinstance(c["arguments"], dict)
        return {"name": c["name"], "args": c["arguments"] if ok else {}, "called": True, "wellformed": ok}
    text = re.sub(r"```(json)?", "", raw).strip()
    m = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(m.group()) if m else None
    except json.JSONDecodeError:
        obj = None
    if not isinstance(obj, dict):
        return {"name": None, "args": {}, "called": False, "wellformed": False}
    name = obj.get("function_called")
    args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {}
    return {"name": name, "args": args, "called": name is not None, "wellformed": True}


# ---------- scoring ----------

def norm_value(v):
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return int(v) if float(v).is_integer() else round(float(v), 4)
    if isinstance(v, list):
        return sorted(str(norm_value(x)) for x in v)
    s = normalize_ar(str(v)).strip(" .،")
    try:
        f = float(s.replace(",", ""))
        return int(f) if f.is_integer() else round(f, 4)
    except ValueError:
        return s.lower()


def is_arabic(text: str) -> bool:  # ArabFuncBench's rule
    text = str(text).strip()
    if not text:
        return False
    if re.match(r"^[\d\-\./]+$", text):
        return True
    return sum(1 for c in text if "؀" <= c <= "ۿ") / max(len(text), 1) > 0.3


def score(case: dict, pred: dict) -> dict:
    gold, name = case["gold_name"], pred["name"]
    fn_ok = (name is None) if gold is None else (name == gold)
    gold_args = {k: v for k, v in case["gold_args"].items() if v not in (None, "", [])}
    pred_args = {k: v for k, v in (pred["args"] or {}).items() if v not in (None, "", [])}
    out = {"fn_ok": fn_ok, "arg_em": None, "aef1": None, "lcr": None}
    if gold is not None:
        out["arg_em"] = fn_ok and {k: norm_value(v) for k, v in gold_args.items()} == \
                        {k: norm_value(v) for k, v in pred_args.items()}
        if fn_ok and gold_args:
            matched = sum(SequenceMatcher(None, str(v), str(pred_args.get(k, ""))).ratio() >= 0.65
                          for k, v in gold_args.items())
            out["aef1"] = matched / len(gold_args)
            out["lcr"] = sum(is_arabic(pred_args.get(k, "")) for k in gold_args) / len(gold_args)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


def summarize(rows: list[dict]) -> dict:
    def block(sel):
        pos = [r for r in sel if r["gold_name"] is not None]
        neg = [r for r in sel if r["gold_name"] is None]
        aef = [r["aef1"] for r in sel if r["aef1"] is not None]
        lcr = [r["lcr"] for r in sel if r["lcr"] is not None]
        k = sum(r["fn_ok"] for r in sel)
        return {"n": len(sel), "fn_acc": round(k / len(sel), 4), "fn_acc_ci95": wilson(k, len(sel)),
                "arg_em": round(sum(r["arg_em"] for r in pos) / len(pos), 4) if pos else None,
                "no_call_acc": round(sum(r["fn_ok"] for r in neg) / len(neg), 4) if neg else None,
                "aef1": round(sum(aef) / len(aef), 4) if aef else None,
                "lcr": round(sum(lcr) / len(lcr), 4) if lcr else None,
                "call_rate": round(sum(r["called"] for r in sel) / len(sel), 4),
                "wellformed": f"{sum(1 for r in sel if r['wellformed'])}/{sum(1 for r in sel if r['called'])}"}
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    return {"all": block(rows), "by_group": {g: block(v) for g, v in sorted(groups.items())}}


def main() -> None:
    import torch
    from common import load_model

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", choices=list(BENCHES), required=True)
    ap.add_argument("--protocol", choices=["text", "paper"], default="text")
    ap.add_argument("--rope", choices=["shipped", "yarn4"], default="shipped")
    ap.add_argument("--adapter")
    ap.add_argument("--name")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    name = args.name or ("base" if not args.adapter else args.adapter.replace("/", "_").replace("\\", "_").replace(":", ""))
    name += f"__{args.protocol}" + ("" if args.rope == "shipped" else f"__{args.rope}")
    out_dir = ROOT / "results" / "tools" / args.bench / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "cases.jsonl"

    cases = BENCHES[args.bench]()
    if args.limit:
        cases = cases[: args.limit]
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    todo = [c for c in cases if c["id"] not in done]
    if todo:
        tok, model = load_model(args.rope, args.adapter)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.unk_token or tok.eos_token
        prompts = {c["id"]: build_prompt(tok, c, args.protocol) for c in todo}
        todo.sort(key=lambda c: len(prompts[c["id"]]))
        with open(out, "a", encoding="utf-8", newline="\n") as f:
            for b in range(0, len(todo), args.batch):
                batch = todo[b: b + args.batch]
                enc = tok([prompts[c["id"]] for c in batch], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=256, do_sample=False, top_p=None, top_k=None,
                                         temperature=None, pad_token_id=tok.pad_token_id)
                for c, row in zip(batch, gen[:, enc["input_ids"].shape[1]:]):
                    raw = tok.decode(row, skip_special_tokens=True).strip()
                    pred = parse(raw, args.protocol)
                    rec = {"id": c["id"], "group": c["group"], "gold_name": c["gold_name"], "pred_name": pred["name"],
                           "pred_args": pred["args"], "called": pred["called"], "wellformed": pred["wellformed"],
                           **score(c, pred), "raw": raw}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{b + len(batch)}/{len(todo)}", flush=True)
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    summary = {"bench": args.bench, "protocol": args.protocol, "rope": args.rope, "adapter": args.adapter,
               **summarize(rows)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["all"], indent=1))


if __name__ == "__main__":
    main()
