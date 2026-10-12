"""The model as an app sees it: one request to an OpenAI-compatible chat API (llama-server --jinja, LM Studio, Ollama).

The benchmarks' --server mode uses this instead of building the prompt themselves: the system message, the
conversation and the tools go out in the standard OpenAI shape, the server's chat template (release/chat_template.jinja,
inside the GGUF) builds the prompt, and its tool-call parser hands back standard "tool_calls". So a served score
measures everything a user gets: the GGUF weights, the template, the tokenizer and llama-server's parser and grammar.
"""

import json
import time
import urllib.request


def chat(url: str, system: str | None, messages: list[dict], tools: list[dict] | None, max_tokens: int = 256,
         timeout: int = 600, model: str | None = None) -> dict:
    """-> {"text": reply text or None, "tool_calls": [{"name", "arguments"}], "raw": the reply as text, "seconds"}.
    Arguments that are not valid JSON are kept as the string (name kept), like toolformat.parse()."""
    body = {"messages": ([{"role": "system", "content": system}] if system else []) + messages,
            "temperature": 0, "top_k": 1, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
    if model:  # Ollama serves several models and needs the name; llama-server ignores it
        body["model"] = model
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        msg = json.loads(r.read())["choices"][0]["message"]
    calls = []
    for tc in msg.get("tool_calls") or []:
        f = tc.get("function", {})
        args = f.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        calls.append({"name": f.get("name"), "arguments": args})
    text = (msg.get("content") or "").strip() or None
    raw = "\n".join(([text] if text else []) + ["<tool_call>" + json.dumps(c, ensure_ascii=False) + "</tool_call>"
                                                for c in calls])
    return {"text": text, "tool_calls": calls, "raw": raw, "seconds": round(time.time() - t0, 2)}
