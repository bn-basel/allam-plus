"""The receptionist benchmark from the earlier project, run on an allam-plus model.

    uv run python scripts/receptionist_eval.py --name base_native                          # base ALLaM, native format
    uv run python scripts/receptionist_eval.py --adapter runs/tools_v1/checkpoint-N --rope yarn4 --name tools_v1
    uv run python scripts/receptionist_eval.py ... --split dev
    uv run python scripts/receptionist_eval.py --server http://127.0.0.1:8095 --name q8_served   # a GGUF, as apps see it

Cases, system prompts, tools and the scorer all come from saudi-dialect (scripts/run_eval.py and its imports), read in
place: the frozen test files are not copied. Only the model side is this project's: ALLaM in 4-bit with the chosen
position setting and adapter, the prompt rendered with this project's Arabic tool format (scripts/toolformat.py),
greedy decoding, up to 256 new tokens, the first <tool_call> lines parsed into tool calls. With --server the request
goes to a running server's OpenAI chat API instead (scripts/served.py): its chat template builds the prompt and its
parser returns the tool calls. Results go to
results/receptionist/<name>/ (predictions, scores, summary) in run_eval's own format.
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SD = HERE.parent.parent / "saudi-dialect"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(SD / "scripts"))

import run_eval  # noqa: E402  (saudi-dialect)
from prompt import load_spec  # noqa: E402  (saudi-dialect)


def main() -> None:
    from common import ROOT, generate, load_model
    from toolformat import parse, render

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", choices=list(run_eval.SPLITS), default="test_v2")
    ap.add_argument("--adapter")
    ap.add_argument("--rope", choices=["shipped", "yarn4"], default="shipped")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--server", help="OpenAI-compatible server URL: test a served GGUF through its chat API")
    ap.add_argument("--server-model", help="model name to send (Ollama needs it)")
    args = ap.parse_args()

    businesses, styles, tools = load_spec()
    for f in ("test_businesses_v2.json", "train_businesses.json"):
        businesses += json.loads((SD / "spec" / f).read_text(encoding="utf-8"))
    unseen = json.loads((SD / "spec" / "tools_unseen.json").read_text(encoding="utf-8"))
    spec = ({b["id"]: b for b in businesses}, styles, tools, unseen)

    out = ROOT / "results" / "receptionist" / f"{args.name}__{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    cases = run_eval.load_cases(args.split)
    if args.limit:
        cases = cases[: args.limit]
    pred_path = out / "predictions.jsonl"
    preds = {p["id"]: p for p in map(json.loads, open(pred_path, encoding="utf-8"))} if pred_path.exists() else {}
    todo = [c for c in cases if c["id"] not in preds]
    t0 = time.time()
    if todo:
        if args.server:
            from served import chat
        else:
            tok, model = load_model(args.rope, args.adapter)
        with open(pred_path, "a", encoding="utf-8", newline="\n") as f:
            for i, case in enumerate(todo, 1):
                system, case_tools = run_eval.case_inputs(case, *spec, "short")
                if args.server:
                    g = chat(args.server, system, case["messages"], case_tools, 256, model=args.server_model)
                    text, calls, raw = g["text"], g["tool_calls"], g["raw"]
                else:
                    prompt = tok.apply_chat_template(render(system, case["messages"], case_tools), tokenize=False,
                                                     add_generation_prompt=True)
                    g = generate(tok, model, prompt, 256)
                    raw = g["text"]
                    text, calls = parse(raw)
                rec = {"id": case["id"], "step": {"text": text, "tool_calls": calls, "raw": raw,
                                                  "tool_format": "served" if args.server else "allam-plus"},
                       "seconds": g["seconds"]}
                preds[case["id"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if i % 20 == 0:
                    print(f"[{i}/{len(todo)}]", flush=True)
    rows = run_eval.score_all(cases, preds, spec, "short")
    (out / "scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    model_name = f"served:{args.server}" if args.server else f"allam-plus:{args.adapter or 'base'}@{args.rope}"
    text, data = run_eval.summarize(rows, args.name, model_name, args.split, time.time() - t0 if todo else None, "short")
    (out / "summary.md").write_text(text, encoding="utf-8", newline="\n")
    (out / "summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
