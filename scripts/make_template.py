"""Phase 4: write the chat template that puts allam-plus's tool format inside the model files.

    uv run python scripts/make_template.py   # writes release/chat_template.jinja (+ Ollama Modelfiles) and checks it

Apps build the prompt from the Jinja chat template stored in the model (tokenizer_config.json, or the GGUF file), not
from this repository's Python. So the template must turn an OpenAI-style request (system message, messages with
tool_calls and tool results, a tools list) into exactly the text that scripts/toolformat.py render() gives and the
model was trained on: the Arabic guide and the tool list in the system turn, <tool_call> lines in assistant turns,
<tool_result> lines in user turns, same-role turns merged, then ALLaM's own template on top. Without tools and without
tool messages it gives the same text as ALLaM's original template, so ordinary chat is unchanged.

llama.cpp's llama-server (--jinja) reads the template to learn the tool-call format and turns the model's
<tool_call> lines back into standard OpenAI "tool_calls" in its reply.

The check renders every Phase 3 tool conversation (each assistant turn as a prompt, and the whole conversation) and a
set of edge cases both ways (render() + ALLaM's template, and this template through transformers): identical text.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from toolformat import DEFAULT_SYSTEM, GUIDE, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "release" / "chat_template.jinja"

TEMPLATE = r"""{#- allam-plus chat template: ALLaM-7B-Instruct's own template, plus tools in Arabic.
    Tools go in the system turn after a short Arabic guide, one JSON object per line. The model calls a tool with a
    line <tool_call>{"name": ..., "arguments": {...}}</tool_call>; each tool result comes back in a user turn as
    <tool_result>...</tool_result>. Turns of the same role are merged, because ALLaM's template needs user and
    assistant turns to alternate. -#}
{%- set default_system = @DEFAULT_SYSTEM@ -%}
{%- set guide = @GUIDE@ -%}
{%- if messages and messages[0]['role'] == 'system' -%}
    {%- set system_text = messages[0]['content'] or '' -%}
    {%- set rest = messages[1:] -%}
{%- else -%}
    {%- set system_text = '' -%}
    {%- set rest = messages -%}
{%- endif -%}
{%- if tools -%}
    {%- set ns_tools = namespace(lines='') -%}
    {%- for t in tools -%}
        {%- set f = t['function'] if t['function'] is defined else t -%}
        {%- set line = '{"name": ' ~ (f['name'] | tojson) ~ ', "description": ' ~ ((f['description'] if f['description'] is defined else '') | tojson) ~ ', "parameters": ' ~ ((f['parameters'] if f['parameters'] is defined else {}) | tojson) ~ '}' -%}
        {%- set ns_tools.lines = ns_tools.lines ~ ('\n' if not loop.first else '') ~ line -%}
    {%- endfor -%}
    {%- set system_text = (system_text or default_system) ~ '\n\n' ~ guide ~ '\n' ~ ns_tools.lines -%}
{%- endif -%}
{%- set ns = namespace(turns=[], role='', text='') -%}
{%- for m in rest -%}
    {%- if m['tool_calls'] is defined and m['tool_calls'] -%}
        {%- set role = 'assistant' -%}
        {%- set before = (m['content'] or '') | trim -%}
        {%- set ns_calls = namespace(lines='') -%}
        {%- for tc in m['tool_calls'] -%}
            {%- set f = tc['function'] if tc['function'] is defined else tc -%}
            {%- set args = f['arguments'] if f['arguments'] is defined else {} -%}
            {%- set args_text = args if args is string else (args | tojson) -%}
            {%- set ns_calls.lines = ns_calls.lines ~ ('\n' if not loop.first else '') ~ '<tool_call>{"name": ' ~ (f['name'] | tojson) ~ ', "arguments": ' ~ args_text ~ '}</tool_call>' -%}
        {%- endfor -%}
        {%- set text = (before ~ '\n' if before else '') ~ ns_calls.lines -%}
    {%- elif m['role'] == 'tool' -%}
        {%- set role = 'user' -%}
        {%- set text = '<tool_result>' ~ m['content'] ~ '</tool_result>' -%}
    {%- else -%}
        {%- set role = m['role'] -%}
        {%- set text = m['content'] or '' -%}
    {%- endif -%}
    {%- if ns.role == role -%}
        {%- set ns.text = ns.text ~ '\n' ~ text -%}
    {%- else -%}
        {%- if ns.role -%}
            {%- set ns.turns = ns.turns + [{'role': ns.role, 'content': ns.text}] -%}
        {%- endif -%}
        {%- set ns.role = role -%}
        {%- set ns.text = text -%}
    {%- endif -%}
{%- endfor -%}
{%- if ns.role -%}
    {%- set ns.turns = ns.turns + [{'role': ns.role, 'content': ns.text}] -%}
{%- endif -%}
{%- for message in ns.turns -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{- raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') -}}
    {%- endif -%}
    {%- if loop.index0 == 0 and system_text -%}
        {%- set content = '<<SYS>>\n' ~ system_text ~ '\n<</SYS>>\n\n' ~ message['content'] -%}
    {%- else -%}
        {%- set content = message['content'] -%}
    {%- endif -%}
    {%- if message['role'] == 'user' -%}
        {{- bos_token ~ ' [INST] ' ~ (content | trim) ~ ' [/INST]' -}}
    {%- elif message['role'] == 'assistant' -%}
        {{- ' ' ~ (content | trim) ~ ' ' ~ eos_token -}}
    {%- endif -%}
{%- endfor -%}
"""


MODELFILE_SIMPLE = '''# Ollama setup for allam-plus-7b.
#   ollama create allam-plus -f Modelfile
#   then use Ollama's chat API (or its OpenAI-compatible /v1) with "tools" as usual.
# Ollama 0.40 and later build the prompt with the chat template inside the GGUF (the same one llama-server uses), so
# no TEMPLATE is needed here. For older Ollama, or with OLLAMA_GO_TEMPLATE=1, use Modelfile.go-template instead.

FROM ./@GGUF@

PARAMETER stop "</s>"
PARAMETER temperature 0
PARAMETER num_ctx 16384
'''

MODELFILE = r'''# Ollama setup for allam-plus-7b with a Go template: for Ollama before 0.40, or with OLLAMA_GO_TEMPLATE=1.
# (written by scripts/make_template.py; the Jinja template inside the GGUF, chat_template.jinja, is the reference)
#   ollama create allam-plus -f Modelfile.go-template
#
# It builds the same turns as the reference: the Arabic tool guide and one JSON line per tool in the system turn,
# <tool_call> lines in assistant turns, <tool_result> lines in user turns, same-role turns merged, ALLaM's [INST]
# format around them. One known difference: Go prints each tool's "parameters" (and past calls' arguments) as compact
# JSON in its own key order, where training used "key": value spacing; docs/PHASE4.md has the scores through it.
# Ollama adds the first <s> itself, so the first turn starts with " [INST]"; later turns write <s> explicitly.

FROM ./@GGUF@

TEMPLATE """{{- $sys := "" }}
{{- range .Messages }}{{ if eq .Role "system" }}{{ $sys = .Content }}{{ end }}{{ end }}
{{- if .Tools }}
{{- if not $sys }}{{ $sys = @DEFAULT_SYSTEM@ }}{{ end }}
{{- $sys = printf "%s\n\n%s\n" $sys @GUIDE@ }}
{{- range $i, $t := .Tools }}
{{- if $i }}{{ $sys = printf "%s\n" $sys }}{{ end }}
{{- $sys = printf "%s{\"name\": %s, \"description\": %s, \"parameters\": %s}" $sys (json $t.Function.Name) (json $t.Function.Description) (json $t.Function.Parameters) }}
{{- end }}
{{- end }}
{{- $prev := "" }}
{{- $first := true }}
{{- range .Messages }}
{{- if eq .Role "system" }}{{ continue }}{{ end }}
{{- $role := .Role }}
{{- $text := .Content }}
{{- if eq .Role "tool" }}{{ $role = "user" }}{{ $text = printf "<tool_result>%s</tool_result>" .Content }}{{ end }}
{{- if .ToolCalls }}
{{- $role = "assistant" }}
{{- if .Content }}{{ $text = printf "%s\n" .Content }}{{ else }}{{ $text = "" }}{{ end }}
{{- range $j, $c := .ToolCalls }}
{{- if $j }}{{ $text = printf "%s\n" $text }}{{ end }}
{{- $text = printf "%s<tool_call>{\"name\": %s, \"arguments\": %s}</tool_call>" $text (json $c.Function.Name) (json $c.Function.Arguments) }}
{{- end }}
{{- end }}
{{- if eq $role $prev }}
{{- "\n" }}{{ $text }}
{{- else }}
{{- if eq $prev "user" }} [/INST]{{ else if eq $prev "assistant" }} </s>{{ end }}
{{- if eq $role "user" }}{{ if not $first }}<s>{{ end }} [INST] {{ if and $first $sys }}<<SYS>>
{{ $sys }}
<</SYS>>

{{ end }}{{ $text }}
{{- else }} {{ $text }}
{{- end }}
{{- $first = false }}
{{- end }}
{{- $prev = $role }}
{{- end }}
{{- if eq $prev "user" }} [/INST]{{ end }}"""

PARAMETER stop "</s>"
PARAMETER temperature 0
PARAMETER num_ctx 16384
'''


def modelfile_text(gguf: str = "allam-plus-7b-Q4_K_M.gguf", go_template: bool = False) -> str:
    if not go_template:
        return MODELFILE_SIMPLE.replace("@GGUF@", gguf)
    lit = lambda s: json.dumps(s, ensure_ascii=False)  # also a valid Go string literal (\n, \" escapes)
    return (MODELFILE.replace("@DEFAULT_SYSTEM@", lit(DEFAULT_SYSTEM)).replace("@GUIDE@", lit(GUIDE))
            .replace("@GGUF@", gguf))


def template_text() -> str:
    lit = lambda s: json.dumps(s, ensure_ascii=False)  # a valid Jinja string literal (\n, \" escapes)
    return TEMPLATE.replace("@DEFAULT_SYSTEM@", lit(DEFAULT_SYSTEM)).replace("@GUIDE@", lit(GUIDE))


def edge_cases() -> list[tuple[str | None, list[dict], list[dict] | None]]:
    """(system, OpenAI-style messages, tools) cases the dev set may not cover."""
    tool = {"type": "function", "function": {"name": "get_weather", "description": "حالة الطقس في مدينة",
                                             "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                            "required": ["city"]}}}
    bare = {"name": "ping", "description": "", "parameters": {"type": "object", "properties": {}}}
    call = lambda n, a, i="c1": {"id": i, "type": "function", "function": {"name": n, "arguments": a}}
    u = {"role": "user", "content": "كيف الجو في الرياض وجدة؟"}
    return [
        (None, [u], None),                                    # plain chat, no system
        ("أنت مساعد.", [u], None),                             # plain chat with a system prompt
        (None, [u], [tool]),                                  # tools, default system prompt
        ("نظام", [u], [tool, bare]),                           # two tools, one without a "function" wrapper
        (None, [u, {"role": "assistant", "content": "", "tool_calls": [call("get_weather", {"city": "الرياض"}),
                                                                        call("get_weather", {"city": "جدة"}, "c2")]},
                {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 31}"},
                {"role": "tool", "tool_call_id": "c2", "content": "{\"temp\": 29}"},
                {"role": "assistant", "content": "الرياض ٣١ وجدة ٢٩."}, {"role": "user", "content": "شكرًا"}], [tool]),
        (None, [u, {"role": "assistant", "content": "لحظة أتحقق.", "tool_calls": [call("get_weather", {"city": "أبها"})]},
                {"role": "tool", "tool_call_id": "c1", "content": "خطأ"}], [tool]),  # text before a call; prompt ends on a result
        (None, [u, {"role": "assistant", "content": None, "tool_calls": [call("ping", {})]},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"}], [bare]),
    ]


def source_conversations():
    """The Phase 3 training and dev conversations before rendering, as an OpenAI-style request would carry them
    (tool calls wrapped as {"id", "type": "function", "function": {...}})."""
    from datetime import date
    from toolbuild import WEEKDAYS_AR, openai_messages

    T = ROOT / "data" / "tools"
    lib = json.loads((T / "library.json").read_text(encoding="utf-8"))
    by_name = {t["name"]: t for d in lib.values() for t in d}

    def wrap(msgs):
        out = []
        for m in msgs:
            if m.get("tool_calls"):
                m = {**m, "tool_calls": [c if "function" in c else
                                         {"id": f"call_{k}", "type": "function", "function": c}
                                         for k, c in enumerate(m["tool_calls"])]}
            out.append(m)
        return out

    for r in map(json.loads, open(T / "convos.jsonl", encoding="utf-8")):
        if r["rejected"]:
            continue
        system = None
        if r["today"]:
            d = date.fromisoformat(r["today"])
            system = f"تاريخ اليوم: {WEEKDAYS_AR[d.weekday()]} {r['today']}"
        yield system, wrap(openai_messages(r["messages"])), [{"type": "function", "function": by_name[n]}
                                                              for n in r["tools"]]
    for r in map(json.loads, open(T / "receptionist.jsonl", encoding="utf-8")):
        yield r["system"], wrap(r["messages"]), r["tools"]


def main() -> None:
    from transformers import AutoTokenizer
    from common import MODEL_ID

    text = template_text()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    (OUT.parent / "Modelfile").write_text(modelfile_text(), encoding="utf-8", newline="\n")
    (OUT.parent / "Modelfile.go-template").write_text(modelfile_text(go_template=True), encoding="utf-8", newline="\n")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    original = tok.chat_template
    cases = edge_cases()
    for system, msgs, tools in source_conversations():
        for i, m in enumerate(msgs):
            if m["role"] == "assistant":  # each assistant turn's prompt, and the whole conversation
                cases.append((system, msgs[:i], tools))
        cases.append((system, msgs, tools))
    same = 0
    first_diff = None
    for system, msgs, tools in cases:
        ours = tok.apply_chat_template(render(system, msgs, tools), tokenize=False, chat_template=original)
        req = ([{"role": "system", "content": system}] if system else []) + msgs
        theirs = tok.apply_chat_template(req, tools=tools, tokenize=False, chat_template=text)
        same += ours == theirs
        if ours != theirs and first_diff is None:
            k = next((j for j in range(min(len(ours), len(theirs))) if ours[j] != theirs[j]), min(len(ours), len(theirs)))
            first_diff = (ours[max(0, k - 80):k + 80], theirs[max(0, k - 80):k + 80])
    print(f"wrote {OUT.relative_to(ROOT)} ({len(text):,} characters)")
    print(f"template = render() + ALLaM's template in {same} of {len(cases)} prompts")
    if first_diff:
        print(f"first difference:\n  render():  {first_diff[0]!r}\n  template:  {first_diff[1]!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
