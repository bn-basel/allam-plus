"""Arabic IFEval: 404 Arabic prompts with instructions a program can check ("no commas", "under 120 words", "write the
title between << >>"). Data: inception42/Arabic-IFEval (CC BY-NC 4.0, evaluation only); checker: Inception's
evaluation.py (MIT), run unchanged from data/raw/ifeval/checker.

    uv run python scripts/ifeval.py --rope shipped [--adapter ...] [--limit 20]
    uv run python scripts/ifeval.py --server http://127.0.0.1:8095 --name base_q8 [--limit 20]

--server: generate with a GGUF build behind llama.cpp's llama-server instead of transformers (4-bit decoding there
takes ~3 hours for the 404 prompts, because greedy decoding sometimes loops to the 1,280-token limit). The prompt is
rendered by ALLaM's own tokenizer exactly as above and sent as token ids, one request at a time, greedy, so a run is
repeatable. Start the server with the build and position setting to test, e.g. YaRN:
    llama-server -m <Q8_0.gguf> -ngl 99 -c 4096 --port 8095 --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 4096

The model gets each prompt as a single user turn through its own chat template (no system prompt, as ALLaM is meant
to be used), greedy decoding, up to 1,280 new tokens. Headline number: prompt-level strict accuracy with the language
check (every instruction in the prompt followed, and the reply in the requested language).
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402

CHECKER = ROOT / "data" / "raw" / "ifeval" / "checker"
INPUT = CHECKER / "input_data" / "ar_input_v3_lang_final.jsonl"


def main() -> None:
    import torch
    from common import MODEL_ID, chat_prompt, load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--rope", choices=["shipped", "yarn4"], default="shipped")
    ap.add_argument("--adapter")
    ap.add_argument("--name")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=1280)
    ap.add_argument("--server", help="llama-server URL: generate with the GGUF it serves")
    args = ap.parse_args()
    if args.server and not args.name:
        ap.error("--server needs --name (the server's model and settings are not known here)")
    name = args.name or args.rope + (("__" + args.adapter.replace("/", "_")) if args.adapter else "")
    out_dir = ROOT / "results" / "ifeval" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    resp_file = out_dir / "responses.jsonl"

    items = [json.loads(l) for l in open(INPUT, encoding="utf-8")]
    if args.limit:
        items = items[: args.limit]
    done = {json.loads(l)["prompt"] for l in open(resp_file, encoding="utf-8")} if resp_file.exists() else set()
    todo = [it for it in items if it["prompt"] not in done]
    if todo and args.server:
        from models import LlamaCppModel
        srv = LlamaCppModel(args.server, MODEL_ID, max_new_tokens=args.max_new)
        with open(resp_file, "a", encoding="utf-8", newline="\n") as f:
            for k, it in enumerate(todo):
                ids = srv.generate_ids(chat_prompt(srv.tok, it["prompt"]))
                text = srv.tok.decode(ids, skip_special_tokens=True).strip()
                f.write(json.dumps({"prompt": it["prompt"], "response": text, "new_tokens": len(ids)},
                                   ensure_ascii=False) + "\n")
                f.flush()
                if k % 20 == 0:
                    print(f"{k + 1}/{len(todo)}", flush=True)
    elif todo:
        tok, model = load_model(args.rope, args.adapter)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.unk_token or tok.eos_token
        prompts = {it["prompt"]: chat_prompt(tok, it["prompt"]) for it in todo}
        todo.sort(key=lambda it: len(prompts[it["prompt"]]))
        with open(resp_file, "a", encoding="utf-8", newline="\n") as f:
            for b in range(0, len(todo), args.batch):
                batch = todo[b: b + args.batch]
                enc = tok([prompts[it["prompt"]] for it in batch], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, top_p=None,
                                         top_k=None, temperature=None, pad_token_id=tok.pad_token_id)
                for it, row in zip(batch, out[:, enc["input_ids"].shape[1]:]):
                    text = tok.decode(row, skip_special_tokens=True).strip()
                    f.write(json.dumps({"prompt": it["prompt"], "response": text,
                                        "new_tokens": int((row != tok.pad_token_id).sum())}, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{b + len(batch)}/{len(todo)}", flush=True)

    inp = INPUT
    if args.limit:  # the checker scores every input prompt, so give it only the ones we ran
        inp = out_dir / "input_subset.jsonl"
        inp.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")
    res = subprocess.run([sys.executable, "evaluation.py", f"--input_data={inp}",
                          f"--input_response_data={resp_file}", f"--output_dir={out_dir / 'checker'}"],
                         cwd=CHECKER, capture_output=True, text=True, encoding="utf-8",
                         env={**os.environ, "PYTHONUTF8": "1"})  # the checker opens files without an encoding
    (out_dir / "checker.log").write_text(res.stdout + "\n" + res.stderr, encoding="utf-8")
    nums = {}
    for key, pat in {"language_check": r"Language-check accuracy: ([\d.]+)",
                     "prompt_strict_no_lang": r"Strict \(no language check\) Scores:\s*prompt-level: ([\d.]+)",
                     "instruction_strict_no_lang": r"Strict \(no language check\) Scores:\s*prompt-level: [\d.]+\s*instruction-level: ([\d.]+)",
                     "prompt_strict": r"strict_with_lang prompt-level: ([\d.]+)",
                     "instruction_strict": r"strict_with_lang instruction-level: ([\d.]+)"}.items():
        m = re.search(pat, res.stdout)
        nums[key] = round(float(m.group(1)), 4) if m else None
    lens = [json.loads(l)["new_tokens"] for l in open(resp_file, encoding="utf-8")]
    summary = {"name": name, "rope": None if args.server else args.rope, "adapter": args.adapter,
               "backend": f"llama-server {args.server}" if args.server else "transformers 4-bit", "n": len(items), **nums,
               "hit_max_new_tokens": sum(n >= args.max_new for n in lens)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    if res.returncode:
        print(res.stderr[-2000:])


if __name__ == "__main__":
    main()
