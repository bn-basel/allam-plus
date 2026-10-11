"""allam-plus tool format: how tools, tool calls and tool results are written into ALLaM's chat.

The same shape as the receptionist model's format (release/allam/allam_tools.py in the earlier project), with the
guide in Arabic (data/tools/guide.json, written by our DeepSeek pipeline from the project's spec):
- the tools are listed in the system prompt after the guide, one JSON object per line;
- the model calls a tool with a line  <tool_call>{"name": ..., "arguments": {...}}</tool_call>  (one line per call,
  several lines = several calls at once);
- each tool result comes back in a user turn as  <tool_result>{...}</tool_result>  (results of parallel calls in one
  turn, one per line), because ALLaM's template needs user and assistant turns to alternate.
Training (train_long.py via longdata/tooldata), every benchmark here and, in Phase 4, the server's chat template all
render through this one function, so the model is always tested exactly as it was trained.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_G = json.loads((ROOT / "data" / "tools" / "guide.json").read_text(encoding="utf-8"))
GUIDE, DEFAULT_SYSTEM = _G["guide"], _G["default_system"]
TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def spec_line(tool: dict) -> str:
    f = tool.get("function", tool)
    return json.dumps({"name": f["name"], "description": f.get("description", ""), "parameters": f.get("parameters", {})},
                      ensure_ascii=False)


def render(system: str | None, messages: list[dict], tools: list[dict] | None) -> list[dict]:
    """OpenAI-style conversation (+ tools) -> the plain system/user/assistant messages ALLaM is trained and tested on.

    messages: {"role": "user"|"assistant"|"tool", "content": ..., "tool_calls": [{"name", "arguments"} or OpenAI
    {"function": {...}}]}. With no tools and no system prompt there is no system turn (ALLaM is meant to work without
    one)."""
    out = []
    if tools:
        out.append({"role": "system", "content": f"{system or DEFAULT_SYSTEM}\n\n{GUIDE}\n" +
                    "\n".join(spec_line(t) for t in tools)})
    elif system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if m.get("tool_calls"):
            lines = []
            for tc in m["tool_calls"]:
                f = tc.get("function", tc)
                args = f.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                lines.append("<tool_call>" + json.dumps({"name": f["name"], "arguments": args}, ensure_ascii=False) +
                             "</tool_call>")
            text = ((m.get("content") or "").strip() + "\n" if (m.get("content") or "").strip() else "") + "\n".join(lines)
            role = "assistant"
        elif m["role"] == "tool":
            role, text = "user", f"<tool_result>{m['content']}</tool_result>"
        else:
            role, text = m["role"], m["content"]
        if out and out[-1]["role"] == role:  # merge, so user and assistant turns alternate
            out[-1]["content"] += "\n" + text
        else:
            out.append({"role": role, "content": text})
    return out


def parse(reply: str) -> tuple[str | None, list[dict]]:
    """A reply -> (text for the user or None, tool calls). A call whose JSON does not parse keeps name None."""
    calls = []
    for body in TOOL_CALL.findall(reply):
        try:
            obj = json.loads(body)
            calls.append({"name": obj.get("name"), "arguments": obj.get("arguments", {})})
        except (json.JSONDecodeError, AttributeError):
            calls.append({"name": None, "arguments": body})
    rest = TOOL_CALL.sub("", reply).strip()
    return rest or None, calls
