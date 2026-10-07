"""Tool calling for the fine-tuned ALLaM-7B Saudi receptionist (allam-7b-saudi-receptionist).

ALLaM-7B-Instruct's chat template has no tool format, so the model was trained on this plain-text one:
- the tools are described in the system prompt, after a short guide (TOOL_GUIDE below, word for word);
- the model calls a tool by replying with a line  <tool_call>{"name": ..., "arguments": {...}}</tool_call>
- your code runs the tool and sends the result back as a user turn:  <tool_result>{...}</tool_result>

Use the guide exactly as written: the model learned this wording, and a different one can lower accuracy.
No dependencies. Works with any server that applies ALLaM's own chat template: transformers, llama.cpp's
llama-server, LM Studio, or Ollama with the Modelfile in this folder.

    from allam_tools import to_allam_messages, parse_tool_calls
    msgs = to_allam_messages(system_prompt, conversation, tools)   # OpenAI-style conversation and tools
    reply = your_chat_completion(msgs)                              # one assistant message, plain text
    text, calls = parse_tool_calls(reply)                           # calls: [{"name": ..., "arguments": {...}}]
"""

import json
import re

TOOL_GUIDE = """You can use these tools. To call one, reply with only this line and nothing else:
<tool_call>{"name": "<tool name>", "arguments": {<arguments as JSON>}}</tool_call>
Tool results come back as <tool_result>...</tool_result>. Otherwise, reply to the customer normally.
Tools:
"""
TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def to_allam_messages(system_prompt: str, messages: list[dict], tools: list[dict]) -> list[dict]:
    """OpenAI-style messages and tools -> the system/user/assistant messages ALLaM was trained on.

    messages: {"role": "user"|"assistant"|"tool", "content": ..., "tool_calls": [...]} as in the OpenAI API.
    tools: [{"type": "function", "function": {"name", "description", "parameters"}}].
    Tool calls become <tool_call> lines, tool results become <tool_result> user turns, and consecutive turns of the
    same role are merged, because ALLaM's template needs user and assistant turns to alternate."""
    specs = "\n".join(json.dumps({"name": t["function"]["name"], "description": t["function"]["description"],
                                  "parameters": t["function"]["parameters"]}, ensure_ascii=False) for t in tools)
    out = [{"role": "system", "content": f"{system_prompt}\n\n{TOOL_GUIDE}{specs}"}]
    for m in messages:
        if m.get("tool_calls"):
            role, text = "assistant", "\n".join(
                "<tool_call>" + json.dumps({"name": tc["function"]["name"], "arguments": _args(tc["function"]["arguments"])},
                                           ensure_ascii=False) + "</tool_call>" for tc in m["tool_calls"])
        elif m["role"] == "tool":
            role, text = "user", f"<tool_result>{m['content']}</tool_result>"
        else:
            role, text = m["role"], m["content"]
        if out[-1]["role"] == role:
            out[-1]["content"] += "\n" + text
        else:
            out.append({"role": role, "content": text})
    return out


def parse_tool_calls(reply: str) -> tuple[str | None, list[dict]]:
    """The model's reply -> (text for the customer or None, tool calls). A call whose JSON does not parse is
    returned with name None, so your code can treat it as an error rather than silently drop it."""
    calls = []
    for body in TOOL_CALL.findall(reply):
        try:
            obj = json.loads(body)
            calls.append({"name": obj.get("name"), "arguments": obj.get("arguments", {})})
        except (json.JSONDecodeError, AttributeError):
            calls.append({"name": None, "arguments": body})
    rest = TOOL_CALL.sub("", reply).strip()
    return rest or None, calls


def _args(arguments):
    return json.loads(arguments) if isinstance(arguments, str) else arguments


if __name__ == "__main__":  # a self-check, no model needed
    tools = [{"type": "function", "function": {"name": "search_services", "description": "Services with price.",
                                               "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}]
    msgs = to_allam_messages("You are a receptionist.", [
        {"role": "user", "content": "السلام عليكم بكم التنظيف"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "search_services", "arguments": '{"query": "تنظيف"}'}}]},
        {"role": "tool", "content": '{"services": [{"name": "تنظيف أسنان", "price_sar": 250}]}'}], tools)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    text, calls = parse_tool_calls('<tool_call>{"name": "search_services", "arguments": {"query": "تنظيف"}}</tool_call>')
    assert text is None and calls == [{"name": "search_services", "arguments": {"query": "تنظيف"}}]
    print("ok")
