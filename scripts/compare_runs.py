"""Compare two trained versions case by case on every test, and apply the rule for replacing the release model.

    uv run python scripts/compare_runs.py --a tools_v1 --b tools_v2

For each test: the share right for A and B, how many cases B got right that A missed ("gained") and the reverse
("lost"), and an exact paired (McNemar) p-value: the chance of a split at least this uneven if neither version were
really better. Paired counts are much more sensitive than comparing two percentages, because both versions answer the
very same cases.

Rule for the receptionist tuning round (docs/DECISIONS.md D19, fixed before training): B replaces A as the release
model only if B reaches the 85% target on the receptionist test v2 and nothing else breaks: ArabFuncBench and AISA
right-tool not significantly lower (paired p < 0.05 with more lost than gained), needle at 16K >= 98%, LC-Eval
multiple choice >= 60%, ArabicMMLU >= 67.6%, Arabic IFEval not significantly lower than A.
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []


def cases(test: str, name: str) -> dict:
    """{case key: right or not} for one test and one version."""
    if test == "arabfuncbench" or test == "aisa_dev":
        return {r["id"]: bool(r["fn_ok"]) for r in rows(R / "tools" / test / f"{name}__native__yarn4" / "cases.jsonl")}
    if test == "receptionist":
        return {r["id"]: bool(r["passed"]) for r in rows(R / "receptionist" / f"{name}_native__test_v2" / "scores.jsonl")}
    if test == "needle_16k":
        return {r["id"]: r["correct"] for r in rows(R / "phase1" / "needle" / name / "cases.jsonl")
                if r["length"] == 16384 and not r.get("control")}
    if test == "lceval_mcq":
        return {r["row"]: r["correct"] for r in rows(R / "lceval" / name / "cases.jsonl") if r["kind"] == "mcq"}
    if test == "arabicmmlu":
        return {r["ID"]: r["correct"] for r in rows(R / "arabicmmlu" / name / "cases.jsonl")}
    if test == "ifeval":
        return {r["prompt"]: r["follow_all_instructions"] for r in
                rows(R / "ifeval" / f"{name}_q8_yarn4" / "checker" / "eval_results_strict_with_lang.jsonl")}
    raise ValueError(test)


def mcnemar_p(gained: int, lost: int) -> float:
    n, k = gained + lost, min(gained, lost)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


TESTS = ["receptionist", "arabfuncbench", "aisa_dev", "needle_16k", "lceval_mcq", "arabicmmlu", "ifeval"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="tools_v1")
    ap.add_argument("--b", default="tools_v2")
    args = ap.parse_args()
    res = {}
    print(f"| test | {args.a} | {args.b} | gained | lost | paired p |\n|---|---|---|---|---|---|")
    for t in TESTS:
        a, b = cases(t, args.a), cases(t, args.b)
        both = sorted(set(a) & set(b), key=str)
        if not both:
            print(f"| {t} | {len(a)} cases | {len(b)} cases | | | |")
            continue
        g = sum(b[k] and not a[k] for k in both)
        l = sum(a[k] and not b[k] for k in both)
        acc_a, acc_b = sum(a[k] for k in both) / len(both), sum(b[k] for k in both) / len(both)
        res[t] = {"n": len(both), "a": acc_a, "b": acc_b, "gained": g, "lost": l, "p": mcnemar_p(g, l),
                  "complete": len(both) == len(a) == len(b)}
        print(f"| {t} (n={len(both)}) | {acc_a:.1%} | {acc_b:.1%} | {g} | {l} | {res[t]['p']:.3f} |")
    worse = lambda t: t in res and res[t]["lost"] > res[t]["gained"] and res[t]["p"] < 0.05
    checks = {
        "receptionist v2 >= 85%": res.get("receptionist", {}).get("b", 0) >= 0.85,
        "ArabFuncBench not significantly lower": "arabfuncbench" in res and not worse("arabfuncbench"),
        "AISA not significantly lower": "aisa_dev" in res and not worse("aisa_dev"),
        "needle 16K >= 98%": res.get("needle_16k", {}).get("b", 0) >= 0.98,
        "LC-Eval MCQ >= 60%": res.get("lceval_mcq", {}).get("b", 0) >= 0.60,
        "ArabicMMLU >= 67.6%": res.get("arabicmmlu", {}).get("b", 0) >= 0.676,
        "IFEval not significantly lower": "ifeval" in res and not worse("ifeval"),
    }
    print()
    for k, v in checks.items():
        print(f"{'ok  ' if v else 'FAIL'} {k}")
    missing = [t for t in TESTS if t not in res or not res[t]["complete"]]
    if missing:
        print(f"\nnot complete yet: {', '.join(missing)}")
    verdict = args.b if all(checks.values()) and not missing else args.a
    print(f"\nrelease model by rule D19: {verdict}")
    out = R / "compare" / f"{args.a}__vs__{args.b}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tests": res, "checks": checks, "missing": missing, "release": verdict}, indent=1),
                   encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
