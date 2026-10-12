"""allam-plus as a LangGraph agent: Arabic requests, real tool calls, through a local OpenAI-compatible server.

1. Serve the GGUF (the template and 16K setting are inside the file; no extra flags):
       llama-server -m allam-plus-7b-Q4_K_M.gguf -ngl 99 -c 16384 --jinja --port 8095
2. Chat with the agent, or run the 20 scripted tasks:
       uv run python demo/agent.py --server http://127.0.0.1:8095 --chat
       uv run python demo/agent.py --server http://127.0.0.1:8095 --tasks --name q4

The graph is the usual tool loop, written out so each part is visible:

    START -> model --(the reply has tool_calls)--> tools -> model -> ... --(plain reply)--> END

"model" is LangChain's ChatOpenAI pointed at llama-server, with the tools bound (they travel in the standard OpenAI
"tools" field); "tools" is LangGraph's ToolNode, which runs every call in the reply and adds the results. Nothing here
knows about ALLaM's prompt format: the chat template inside the GGUF writes the Arabic tool guide, and llama-server
turns the model's <tool_call> lines back into standard tool_calls.

The tools are a small fake backend in three areas the model never saw in training (car servicing, money, parcel
shipping), so the tasks also test tools it has never met. Every answer is fixed, so runs are repeatable.
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

ROOT = Path(__file__).resolve().parent.parent
TODAY = date(2026, 10, 15)  # a Thursday; the system prompt gives the date, as in training
SYSTEM = "تاريخ اليوم: الخميس 2026-10-15"
HINDI = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _digits(s) -> str:
    return str(s).translate(HINDI).strip()


def _out(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


# ---------------------------------------------------------------- fake backend: three areas, fixed answers

CENTERS = {
    "الرياض": [{"center_id": "101", "name": "مركز النخبة لصيانة السيارات", "district": "العليا",
                "brands": ["تويوتا", "لكزس", "هيونداي"], "rating": 4.6},
               {"center_id": "102", "name": "ورشة الريان", "district": "النرجس", "brands": ["نيسان", "هيونداي", "كيا"],
                "rating": 4.2}],
    "جدة": [{"center_id": "201", "name": "مركز البحر للصيانة", "district": "الروضة",
             "brands": ["تويوتا", "هيونداي", "فورد"], "rating": 4.4}],
    "الدمام": [{"center_id": "301", "name": "مركز الخليج للسيارات", "district": "الفيصلية",
                "brands": ["تويوتا", "نيسان"], "rating": 4.1}],
}
SERVICES = {"تغيير زيت": 180, "فحص شامل": 350, "تبديل فرامل": 620, "تبديل إطارات": 1400}
GOLD = {18: 352.0, 21: 410.0, 22: 430.0, 24: 469.0}
TO_SAR = {"SAR": 1.0, "USD": 3.75, "EUR": 4.10, "AED": 1.02, "EGP": 1 / 13.0, "KWD": 12.2}
DAYS = {frozenset(["الرياض", "جدة"]): 2, frozenset(["الرياض", "الدمام"]): 1, frozenset(["جدة", "الدمام"]): 3}
SHIPMENTS = {"778812": {"status": "في الطريق", "location": "مركز الفرز في الرياض", "expected": "2026-10-17"},
             "554433": {"status": "تم التسليم", "location": "جدة", "delivered": "2026-10-13"}}
TAKEN = {("101", "2026-10-18", "10:00"): ["09:00", "11:30"]}


def _city(c: str) -> str:
    c = c.strip().removeprefix("مدينة ").strip()
    return {"رياض": "الرياض", "دمام": "الدمام"}.get(c, c)


@tool(parse_docstring=True)
def find_car_service_centers(city: str, brand: str = "") -> str:
    """البحث عن مراكز صيانة السيارات في مدينة، ويمكن تحديد ماركة السيارة.

    Args:
        city: اسم المدينة، مثل الرياض
        brand: ماركة السيارة (اختياري)، مثل تويوتا
    """
    found = [c for c in CENTERS.get(_city(city), []) if not brand or brand.strip() in c["brands"]]
    return _out({"centers": [{k: v for k, v in c.items() if k != "brands"} for c in found]})


@tool(parse_docstring=True)
def get_service_price(service: str, car_model: str) -> str:
    """سعر خدمة صيانة لطراز سيارة.

    Args:
        service: الخدمة: تغيير زيت أو فحص شامل أو تبديل فرامل أو تبديل إطارات
        car_model: طراز السيارة وسنة الصنع، مثل كامري 2020
    """
    base = SERVICES.get(service.strip())
    if base is None:
        return _out({"error": "خدمة غير معروفة", "services": list(SERVICES)})
    return _out({"service": service.strip(), "car_model": car_model, "price_sar": base, "duration_hours": 2})


@tool(parse_docstring=True)
def book_car_service(center_id: str, service: str, date: str, time: str) -> str:
    """حجز موعد صيانة في مركز.

    Args:
        center_id: رقم المركز من نتائج البحث
        service: الخدمة المطلوبة
        date: التاريخ بصيغة YYYY-MM-DD
        time: الوقت بصيغة HH:MM (24 ساعة)
    """
    cid, d, t = _digits(center_id), _digits(date), _digits(time)
    if not any(c["center_id"] == cid for cs in CENTERS.values() for c in cs):
        return _out({"error": "رقم المركز غير موجود"})
    if (cid, d, t) in TAKEN:
        return _out({"error": "الموعد محجوز", "available_times": TAKEN[(cid, d, t)]})
    return _out({"status": "confirmed", "booking_number": "88" + cid + t.replace(":", ""), "center_id": cid,
                 "service": service, "date": d, "time": t})


@tool(parse_docstring=True)
def get_gold_price(karat: int) -> str:
    """سعر غرام الذهب اليوم بالريال السعودي.

    Args:
        karat: العيار: 18 أو 21 أو 22 أو 24
    """
    k = int(_digits(karat))
    return _out({"karat": k, "price_per_gram_sar": GOLD[k]} if k in GOLD else {"error": "عيار غير متاح"})


@tool(parse_docstring=True)
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """تحويل مبلغ من عملة إلى أخرى بسعر اليوم.

    Args:
        amount: المبلغ
        from_currency: رمز العملة المحوَّل منها، مثل SAR أو USD
        to_currency: رمز العملة المحوَّل إليها
    """
    f, t = from_currency.strip().upper(), to_currency.strip().upper()
    if f not in TO_SAR or t not in TO_SAR:
        return _out({"error": "عملة غير مدعومة", "supported": list(TO_SAR)})
    value = round(float(_digits(amount)) * TO_SAR[f] / TO_SAR[t], 2)
    return _out({"amount": float(_digits(amount)), "from": f, "to": t, "result": value})


@tool(parse_docstring=True)
def estimate_shipping_cost(from_city: str, to_city: str, weight_kg: float) -> str:
    """تقدير تكلفة ومدة شحن طرد بين مدينتين داخل المملكة.

    Args:
        from_city: مدينة الإرسال
        to_city: مدينة الاستلام
        weight_kg: الوزن بالكيلوغرام
    """
    a, b = _city(from_city), _city(to_city)
    days = DAYS.get(frozenset([a, b]))
    if days is None:
        return _out({"error": "المسار غير مدعوم"})
    w = float(_digits(weight_kg))
    return _out({"from": a, "to": b, "weight_kg": w, "cost_sar": round(25 + 6 * w, 2), "days": days})


@tool(parse_docstring=True)
def track_shipment(tracking_number: str) -> str:
    """تتبع شحنة برقمها.

    Args:
        tracking_number: رقم الشحنة
    """
    n = _digits(tracking_number)
    return _out(SHIPMENTS.get(n, {"error": "رقم الشحنة غير موجود"}))


TOOLS = [find_car_service_centers, get_service_price, book_car_service, get_gold_price, convert_currency,
         estimate_shipping_cost, track_shipment]

# A fourth area, used by the larger agent test (scripts/agent_bench.py) together with the seven above.
TECHNICIANS = {
    "الرياض": [{"technician_id": "501", "name": "فني السباكة أبو خالد", "services": ["سباكة"], "rating": 4.7},
               {"technician_id": "502", "name": "مؤسسة النور للكهرباء", "services": ["كهرباء"], "rating": 4.5},
               {"technician_id": "503", "name": "فني التكييف السريع", "services": ["تكييف"], "rating": 4.3}],
    "جدة": [{"technician_id": "601", "name": "ورشة الساحل للتكييف", "services": ["تكييف", "كهرباء"], "rating": 4.6},
            {"technician_id": "602", "name": "فني الدهانات الحديثة", "services": ["دهان"], "rating": 4.4}],
    "الدمام": [{"technician_id": "701", "name": "مؤسسة الشرقية للصيانة", "services": ["سباكة", "دهان"],
                "rating": 4.2}],
}
VISIT_FEES = {"سباكة": 120, "كهرباء": 100, "تكييف": 150, "دهان": 200}
BUSY = {("501", "2026-10-17", "09:00"): ["11:00", "16:00"]}


@tool(parse_docstring=True)
def find_technicians(city: str, service: str) -> str:
    """البحث عن فنيي صيانة منزلية في مدينة لخدمة معينة.

    Args:
        city: اسم المدينة
        service: الخدمة: سباكة أو كهرباء أو تكييف أو دهان
    """
    found = [t for t in TECHNICIANS.get(_city(city), []) if service.strip() in t["services"]]
    return _out({"technicians": [{k: v for k, v in t.items() if k != "services"} for t in found]})


@tool(parse_docstring=True)
def get_visit_fee(service: str) -> str:
    """رسوم زيارة الفني لخدمة صيانة منزلية بالريال السعودي.

    Args:
        service: الخدمة: سباكة أو كهرباء أو تكييف أو دهان
    """
    fee = VISIT_FEES.get(service.strip())
    return _out({"service": service.strip(), "visit_fee_sar": fee} if fee else
                {"error": "خدمة غير معروفة", "services": list(VISIT_FEES)})


@tool(parse_docstring=True)
def book_technician_visit(technician_id: str, date: str, time: str, district: str) -> str:
    """حجز زيارة فني صيانة إلى المنزل.

    Args:
        technician_id: رقم الفني من نتائج البحث
        date: التاريخ بصيغة YYYY-MM-DD
        time: الوقت بصيغة HH:MM (24 ساعة)
        district: الحي الذي سيزوره الفني
    """
    tid, d, t = _digits(technician_id), _digits(date), _digits(time)
    if not any(x["technician_id"] == tid for xs in TECHNICIANS.values() for x in xs):
        return _out({"error": "رقم الفني غير موجود"})
    if (tid, d, t) in BUSY:
        return _out({"error": "الفني مشغول في هذا الوقت", "available_times": BUSY[(tid, d, t)]})
    return _out({"status": "confirmed", "visit_number": "77" + tid + t.replace(":", ""), "technician_id": tid,
                 "date": d, "time": t, "district": district})


ALL_TOOLS = TOOLS + [find_technicians, get_visit_fee, book_technician_visit]


# ---------------------------------------------------------------- the graph

def build_agent(server: str, max_tokens: int = 512, tools: list | None = None):
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    tools = tools or TOOLS
    llm = ChatOpenAI(base_url=server.rstrip("/") + "/v1", api_key="local", model="allam-plus", temperature=0,
                     max_tokens=max_tokens).bind_tools(tools)

    def model(state: MessagesState):
        return {"messages": [llm.invoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("model", model)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "model")
    g.add_conditional_edges("model", tools_condition)  # tool_calls -> "tools", otherwise END
    g.add_edge("tools", "model")
    return g.compile()


def run_turns(agent, turns: list[str], history: list | None = None) -> list:
    """Each user turn runs the graph to a plain reply; returns all messages."""
    msgs = list(history or [SystemMessage(SYSTEM)])
    for text in turns:
        msgs.append(HumanMessage(text))
        msgs = agent.invoke({"messages": msgs}, {"recursion_limit": 12})["messages"]
    return msgs


# ---------------------------------------------------------------- scripted tasks

def _norm(s) -> str:
    s = _digits(s)
    s = re.sub(r"(?<=\d)[,٬](?=\d{3})", "", s)  # thousands separators
    return re.sub(r"[ًٌٍَُِّْـ]", "", s).replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه")


def _arg_ok(want, got) -> bool:
    if isinstance(want, (int, float)) and not isinstance(want, bool):
        try:
            return abs(float(_digits(got)) - want) < 1e-6
        except (TypeError, ValueError):
            return False
    w, g = _norm(want).lower(), _norm(got).lower()  # currency codes: the tool takes "sar" as well as "SAR"
    return w == g or (len(w) > 2 and w in g)


def check(task: dict, msgs: list) -> dict:
    """Pass/fail for one task, with the reasons. Each call keeps the number of the model step that made it."""
    calls, step = [], 0
    for m in msgs:
        if m.type == "ai":
            step += 1
            calls += [{"name": c["name"], "args": c["args"], "step": step} for c in (m.tool_calls or [])]
    reply = next((m.content for m in reversed(msgs) if m.type == "ai" and m.content), "") or ""
    return check_parts(task, calls, reply)


def check_parts(task: dict, calls: list[dict], reply: str) -> dict:
    problems = []
    first, then = task.get("sequential", [None, None])
    if first:  # the second tool needs the first one's result: it must come in a later step, not alongside
        a = min((c.get("step", 0) for c in calls if c["name"] == first), default=None)
        b = [c.get("step", 0) for c in calls if c["name"] == then]
        if a is not None and b and min(b) <= a:
            problems.append(f"{then} called before {first}'s result came back")
    for want in task.get("expect_calls", []):
        if not any(c["name"] == want["name"] and all(_arg_ok(v, c["args"].get(k)) for k, v in want.get("args", {}).items())
                   for c in calls):
            problems.append(f"missing call {want['name']} {want.get('args', {})}")
    if task.get("no_calls") and calls:
        problems.append(f"called {[c['name'] for c in calls]} but no call was expected")
    for name in task.get("forbid_calls", []):
        if any(c["name"] == name for c in calls):
            problems.append(f"called {name}")
    for alts in task.get("reply_has", []):
        if not any(_norm(a) in _norm(reply) for a in alts):
            problems.append(f"reply lacks any of {alts}")
    for bad in task.get("reply_lacks", []):
        if _norm(bad) in _norm(reply):
            problems.append(f"reply has {bad!r}")
    return {"id": task["id"], "kind": task["kind"], "pass": not problems, "problems": problems,
            "calls": [{k: c[k] for k in ("name", "args", "step") if k in c} for c in calls], "reply": reply}


def run_tasks(server: str, name: str) -> None:
    tasks = json.loads((Path(__file__).parent / "tasks.json").read_text(encoding="utf-8"))
    agent = build_agent(server)
    rows = []
    for t in tasks:
        try:
            r = check(t, run_turns(agent, t["turns"]))
        except Exception as e:  # a run that breaks the loop (e.g. too many steps) is a failed task
            r = {"id": t["id"], "kind": t["kind"], "pass": False, "problems": [f"{type(e).__name__}: {e}"[:300]],
                 "calls": [], "reply": ""}
        rows.append(r)
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['id']:<22} {'; '.join(r['problems'])}", flush=True)
    n = sum(r["pass"] for r in rows)
    out = ROOT / "results" / "phase4" / "langgraph"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps({"name": name, "server": server, "passed": n, "tasks": len(rows),
                                                  "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{n} of {len(rows)} tasks passed -> results/phase4/langgraph/{name}.json")


def rescore(name: str) -> None:
    """Score a saved run again with the current checks (the calls and final reply are saved per task)."""
    tasks = {t["id"]: t for t in json.loads((Path(__file__).parent / "tasks.json").read_text(encoding="utf-8"))}
    path = ROOT / "results" / "phase4" / "langgraph" / f"{name}.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    broke = lambda r: not r["calls"] and not r["reply"] and r["problems"]  # noqa: E731  (the loop itself failed)
    rows = [r if broke(r) else check_parts(tasks[r["id"]], r["calls"], r["reply"]) for r in run["rows"]]
    run.update(rows=rows, passed=sum(r["pass"] for r in rows))
    path.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{name}: {run['passed']} of {run['tasks']} tasks passed")


def chat(server: str) -> None:
    agent = build_agent(server)
    msgs = [SystemMessage(SYSTEM)]
    print("اكتب رسالتك (سطر فارغ للخروج)")
    while True:
        text = input("> ").strip()
        if not text:
            return
        before = len(msgs)
        msgs = run_turns(agent, [text], msgs)
        for m in msgs[before + 1:]:
            if getattr(m, "tool_calls", None):
                for c in m.tool_calls:
                    print(f"  [tool] {c['name']} {json.dumps(c['args'], ensure_ascii=False)}")
            elif m.type == "tool":
                print(f"  [result] {m.content}")
            elif m.content:
                print(m.content)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Arabic on a Windows console
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8095")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--tasks", action="store_true")
    ap.add_argument("--name", default="run")
    ap.add_argument("--rescore", action="store_true", help="score the saved run --name again with the current checks")
    args = ap.parse_args()
    if args.rescore:
        rescore(args.name)
    elif args.tasks:
        run_tasks(args.server, args.name)
    else:
        chat(args.server)


if __name__ == "__main__":
    main()
