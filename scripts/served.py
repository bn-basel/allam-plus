"""The model as an app sees it: one request to an OpenAI-compatible chat API (llama-server --jinja, LM Studio, Ollama).

The benchmarks' --server mode uses this instead of building the prompt themselves: the system message, the
conversation and the tools go out in the standard OpenAI shape, the server's chat template (release/chat_template.jinja,
inside the GGUF) builds the prompt, and its tool-call parser hands back standard "tool_calls". So a served score
measures everything a user gets: the GGUF weights, the template, the tokenizer and llama-server's parser and grammar.
"""

import json
import time
import urllib.request


def openai_shape(messages: list[dict]) -> list[dict]:
    """Past tool calls as the OpenAI API writes them: {"id", "type": "function", "function": {"name", "arguments": a
    JSON string}}, with content null. (llama-server also takes arguments as objects; Ollama answers 400.)"""
    out = []
    for i, m in enumerate(messages):
        if m.get("tool_calls"):
            calls = []
            for k, tc in enumerate(m["tool_calls"]):
                f = tc.get("function", tc)
                args = f.get("arguments", {})
                calls.append({"id": tc.get("id") or f"call_{i}_{k}", "type": "function",
                              "function": {"name": f["name"],
                                           "arguments": args if isinstance(args, str) else json.dumps(args)}})
            m = {**m, "content": m.get("content") or None, "tool_calls": calls}
        out.append(m)
    return out


def chat(url: str, system: str | None, messages: list[dict], tools: list[dict] | None, max_tokens: int = 256,
         timeout: int = 600, model: str | None = None, parse_text: bool = False) -> dict:
    """-> {"text": reply text or None, "tool_calls": [{"name", "arguments"}], "raw": the reply as text, "seconds"}.
    Arguments that are not valid JSON are kept as the string (name kept), like toolformat.parse().
    parse_text: also read <tool_call> lines left in the reply text (for servers that do not parse them, such as
    Ollama with a Go template)."""
    body = {"messages": ([{"role": "system", "content": system}] if system else []) + openai_shape(messages),
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
    if parse_text and not calls and text and "<tool_call>" in text:
        from toolformat import parse
        text, calls = parse(text)
    raw ="\n".join(([text] if text else []) + ["<tool_call>" + json.dumps(c, ensure_ascii=False) + "</tool_call>"
                                                for c in calls])
    return {"text": text, "tool_calls": calls, "raw": raw, "seconds": round(time.time() - t0, 2)}
