"""Release check: does a GGUF's built-in chat template give exactly the prompt the model was trained and tested on?

Apps such as llama-server (with --jinja) and LM Studio build the prompt from the template stored inside the GGUF file,
not from this repository's code. A difference of a few tokens can cost a lot (Gemma's empty thinking block cost about
20 points; README, Stage 7). This asks a running llama-server to render the test cases with its template
(/apply-template) and compares the result character by character with the benchmark's own prompt.

    uv run python scripts/check_gguf_template.py --url http://127.0.0.1:8082 --repo google/gemma-4-12B-it
    uv run python scripts/check_gguf_template.py --url http://127.0.0.1:8082 --repo humain-ai/ALLaM-7B-Instruct-preview

For Gemma the request uses the OpenAI-style messages and tools with chat_template_kwargs {"enable_thinking": false};
for ALLaM it sends the messages built by release/allam/allam_tools.py (ALLaM has no tool format of its own).
"""

import argparse
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import text_tool_messages, uses_text_tools  # noqa: E402
from prompt import build_system_prompt, load_spec  # noqa: E402
import run_eval  # noqa: E402


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--splits", default="test,v1_greeted")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.repo)
    text_tools = uses_text_tools(tok)
    businesses, styles, tools = load_spec()
    businesses = {b["id"]: b for b in businesses}
    unseen = json.loads((run_eval.ROOT / "spec" / "tools_unseen.json").read_text(encoding="utf-8"))
    same = total = 0
    first_diff = None
    for split in args.splits.split(","):
        for case in run_eval.load_cases(split):
            system, case_tools = run_eval.case_inputs(case, businesses, styles, tools, unseen)
            if text_tools:
                messages = text_tool_messages(system, case["messages"], case_tools)
                ours = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                body = {"messages": messages}
            else:
                messages = [{"role": "system", "content": system}] + case["messages"]
                ours = tok.apply_chat_template(messages, tools=case_tools, add_generation_prompt=True,
                                               enable_thinking=False, tokenize=False)
                body = {"messages": messages, "tools": case_tools, "chat_template_kwargs": {"enable_thinking": False}}
            theirs = post(args.url.rstrip("/") + "/apply-template", body)["prompt"]
            # the server adds the start-of-text token itself when it tokenizes, so its rendering may leave it out
            bos = tok.bos_token or ""
            ok = theirs == ours or (bos and bos + theirs == ours) or (bos and theirs.lstrip() == ours.removeprefix(bos).lstrip())
            same += bool(ok)
            total += 1
            if not ok and first_diff is None:
                i = next((k for k in range(min(len(ours), len(theirs))) if ours[k] != theirs[k]), min(len(ours), len(theirs)))
                first_diff = (case["id"], ours[max(0, i - 80): i + 80], theirs[max(0, i - 80): i + 80])
    print(f"{datetime.now():%Y-%m-%d} {args.repo}: the GGUF's template matches the benchmark prompt in {same} of {total} cases")
    if first_diff:
        print(f"first difference ({first_diff[0]}):\n  ours:   {first_diff[1]!r}\n  theirs: {first_diff[2]!r}")


if __name__ == "__main__":
    main()
