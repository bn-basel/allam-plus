"""Release check: does a running llama-server build exactly the prompt the model was trained and tested on?

Apps such as llama-server (with --jinja), LM Studio and Ollama build the prompt from the chat template stored in the
GGUF file, not from this repository's code, and a difference of a few tokens can cost a lot of accuracy. This sends
OpenAI-style requests (system message, messages with tool_calls and tool results, tools) to the server's
/apply-template and compares the text with render() + ALLaM's template, then compares token ids for a sample
(/tokenize), because the same text can still be split into different tokens.

    uv run python scripts/check_gguf_template.py --url http://127.0.0.1:8090
    uv run python scripts/check_gguf_template.py --url http://127.0.0.1:8090 --string-args

Cases: the prompt of every assistant turn in every Phase 3 tool conversation, and the edge cases of
scripts/make_template.py. (A request that ends on an assistant message is a "prefill" to llama-server: it continues
that message instead of closing it, so whole conversations, and the rare second assistant message right after another,
which training merges into one turn, are checked only in make_template.py.) With --string-args the tool-call arguments are sent as JSON strings with escaped Arabic
(\\u0627...), as LangChain and the OpenAI SDK send them; llama-server parses them back, so the prompt must not change.
"""

import argparse
import json
import random
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_template import edge_cases, source_conversations  # noqa: E402
from toolformat import render  # noqa: E402


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def stringify_args(msgs: list[dict]) -> list[dict]:
    out = []
    for m in msgs:
        if m.get("tool_calls"):
            m = {**m, "tool_calls": [{**c, "function": {**c["function"],
                                                        "arguments": json.dumps(c["function"]["arguments"])}}
                                     for c in m["tool_calls"]]}
        out.append(m)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--string-args", action="store_true")
    ap.add_argument("--token-sample", type=int, default=300)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from common import MODEL_ID

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    url = args.url.rstrip("/")
    cases = edge_cases()
    for system, msgs, tools in source_conversations():
        for i, m in enumerate(msgs):
            if m["role"] == "assistant" and msgs[i - 1]["role"] != "assistant":
                cases.append((system, msgs[:i], tools))
    same = 0
    first_diff = None
    rendered = []
    for system, msgs, tools in cases:
        ours = tok.apply_chat_template(render(system, msgs, tools), tokenize=False)
        req = ([{"role": "system", "content": system}] if system else []) + (stringify_args(msgs) if args.string_args
                                                                              else msgs)
        if tools:  # the OpenAI API needs {"type": "function", "function": {...}}; transformers also takes it bare
            tools = [t if "function" in t else {"type": "function", "function": t} for t in tools]
        body = {"messages": req, **({"tools": tools} if tools else {})}
        theirs = post(url + "/apply-template", body)["prompt"]
        same += ours == theirs
        rendered.append((ours, theirs))
        if ours != theirs and first_diff is None:
            k = next((j for j in range(min(len(ours), len(theirs))) if ours[j] != theirs[j]), min(len(ours), len(theirs)))
            first_diff = (ours[max(0, k - 80):k + 80], theirs[max(0, k - 80):k + 80])
    tok_same = 0
    sample = random.Random(1).sample(rendered, min(args.token_sample, len(rendered)))
    for ours, theirs in sample:
        hf = tok(ours, add_special_tokens=False)["input_ids"]
        srv = post(url + "/tokenize", {"content": theirs, "add_special": True, "parse_special": True})["tokens"]
        tok_same += hf == srv
    mode = " (arguments sent as JSON strings)" if args.string_args else ""
    print(f"{datetime.now():%Y-%m-%d} {url}{mode}: same prompt text in {same} of {len(cases)} cases; "
          f"same token ids in {tok_same} of {len(sample)} sampled")
    if first_diff:
        print(f"first difference:\n  ours:   {first_diff[0]!r}\n  server: {first_diff[1]!r}")
    sys.exit(0 if same == len(cases) and tok_same == len(sample) else 1)


if __name__ == "__main__":
    main()
