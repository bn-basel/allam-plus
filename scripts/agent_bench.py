"""Agent test: 90 scripted Arabic tasks run through LangGraph against a served model (frozen before the fix round).

    uv run python scripts/agent_bench.py --server http://127.0.0.1:8095 --name tools_v2_q4
    uv run python scripts/agent_bench.py --compare tools_v2_q4 tools_v3_q4

The agent is demo/agent.py's graph (model → tools → model → … → END) with all ten tools of its fake backend, in four
areas the model never saw in training: car servicing, gold and currency prices, parcel shipping, home-maintenance
visits. Tasks (data/agent_test/tasks_v1.json, SHA-256 checked at start):
- dependent (20): two steps where the second call needs a value from the first one's result (a price to convert, a
  centre or technician number to book with); the second call must come after the first result, not alongside it;
- missing (20): a request that leaves out a required detail (date, time, amount, city…): ask, do not call the action;
- arithmetic (20): an answer that needs a sum, difference or product of tool results, said in the reply;
- regression (30): one call, independent calls at once, no tool fits, tool errors, follow-up turns.
A task passes when every check holds (demo/agent.py check_parts). Comparisons are paired: the same 90 tasks.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
TASKS = ROOT / "data" / "agent_test" / "tasks_v1.json"
SHA = "60921984fdedd193cb8b8d71625b68ccd0908ca7761dc6af6c1cb2ba88d0e164"
OUT = ROOT / "results" / "agent_test"
SKILLS = ["dependent", "missing", "arithmetic"]


def load_tasks() -> list[dict]:
    raw = TASKS.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA:
        sys.exit(f"{TASKS} changed since it was frozen")
    return json.loads(raw)


def run(server: str, name: str) -> None:
    import agent as A

    tasks = load_tasks()
    out = OUT / f"{name}.json"
    done = {r["id"]: r for r in json.loads(out.read_text(encoding="utf-8"))["rows"]} if out.exists() else {}
    graph = A.build_agent(server, tools=A.ALL_TOOLS)
    rows = []
    for t in tasks:
        if t["id"] in done:
            rows.append(done[t["id"]])
            continue
        try:
            r = A.check(t, A.run_turns(graph, t["turns"]))
        except Exception as e:  # a run that breaks the loop (e.g. too many steps) is a failed task
            r = {"id": t["id"], "kind": t["kind"], "pass": False, "problems": [f"{type(e).__name__}: {e}"[:300]],
                 "calls": [], "reply": ""}
        rows.append(r)
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['id']:<30} {'; '.join(r['problems'])[:150]}", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"name": name, "server": server, "rows": rows}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    s = summary(rows)
    out.write_text(json.dumps({"name": name, "server": server, **s, "rows": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps(s, ensure_ascii=False))


def summary(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["kind"] if r["kind"] in SKILLS else "regression"].append(r["pass"])
    return {"passed": sum(r["pass"] for r in rows), "tasks": len(rows),
            "by_group": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by.items())},
            "skills": sum(sum(by[k]) for k in SKILLS), "regression": sum(by["regression"])}


def mcnemar_p(gained: int, lost: int) -> float:
    n, k = gained + lost, min(gained, lost)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(a: str, b: str) -> dict:
    ra = {r["id"]: r for r in json.loads((OUT / f"{a}.json").read_text(encoding="utf-8"))["rows"]}
    rb = {r["id"]: r for r in json.loads((OUT / f"{b}.json").read_text(encoding="utf-8"))["rows"]}
    res = {}
    print(f"| group | {a} | {b} | gained | lost | paired p |\n|---|---|---|---|---|---|")
    groups = SKILLS + ["skills (all three)", "regression", "all"]
    for g in groups:
        ids = [i for i in ra if i in rb and (g == "all" or (g == "skills (all three)" and ra[i]["kind"] in SKILLS)
                                            or (g == "regression" and ra[i]["kind"] not in SKILLS) or ra[i]["kind"] == g)]
        gained = sum(rb[i]["pass"] and not ra[i]["pass"] for i in ids)
        lost = sum(ra[i]["pass"] and not rb[i]["pass"] for i in ids)
        res[g] = {"n": len(ids), "a": sum(ra[i]["pass"] for i in ids), "b": sum(rb[i]["pass"] for i in ids),
                  "gained": gained, "lost": lost, "p": mcnemar_p(gained, lost)}
        x = res[g]
        print(f"| {g} | {x['a']}/{x['n']} | {x['b']}/{x['n']} | {gained} | {lost} | {x['p']:.3f} |")
    return res


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server")
    ap.add_argument("--name")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args.server, args.name)


if __name__ == "__main__":
    main()
