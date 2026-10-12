# Phase 4: serving and release

The goal: an app or agent sends a standard OpenAI-style request (messages and `tools`) to a local server and gets
standard `tool_calls` back, with the model reading exactly what it was trained on. Decisions and reasons:
[DECISIONS.md](DECISIONS.md) D20.

## How a request reaches the model

```
app / LangGraph ──OpenAI request──▶ llama-server ──chat template (inside the GGUF)──▶ prompt text ──tokenizer──▶ ALLaM
      ▲                                                                                                   │
      └──────────── standard tool_calls ◀── tool-call parser (+ grammar from the tool schemas) ◀── <tool_call> lines
```

| Piece | What we did | Check |
|---|---|---|
| Chat template | `release/chat_template.jinja`, written from the same guide as training (`scripts/make_template.py`) | identical text to training on 16,224 prompts (transformers) and 12,011 of 12,013 (llama-server; the 2: a past call's `250.0` comes back as `250`) |
| Tokenizer | GGUF flag `add_space_prefix = false` (llama.cpp otherwise adds a space token after every `<s>`) | token ids identical in 300 of 300 sampled prompts (0 of 300 without the flag) |
| Tool-call parser | llama-server's own, found from the template: no proxy | standard `tool_calls`; arguments sent back as JSON strings give the same prompt |
| 16K context | YaRN ×4 stored in the GGUF (`scripts/finalize_gguf.py`): no command-line flags | needle test through the chat API (below) |

## Release files

`allam-plus-7b-Q8_0.gguf` (7.4 GB) and `allam-plus-7b-Q4_K_M.gguf` (4.3 GB): the tools_v1 checkpoint 400 weights
merged into ALLaM-7B-Instruct, with the template and 16K setting inside. Ollama: `release/Modelfile` (Ollama 0.40+,
uses the template inside the GGUF) and `release/Modelfile.go-template` (older Ollama).

```
llama-server -m allam-plus-7b-Q4_K_M.gguf -ngl 99 -c 16384 --jinja --port 8095
```

## Served results (what an app gets)

_Pending: `scripts/steps_phase4.txt` (about 2 hours of GPU)._

| Test | transformers, 4-bit (Phase 3) | Q8_0 served | Q4_K_M served | Q4_K_M through Ollama |
|---|---|---|---|---|
| ArabFuncBench, right tool / every argument right | 96.8% / 48.5% | | | |
| AISA-ArabicFC dev, right tool / every argument right | 93.4% / 36.2% | | | |
| Receptionist test v2 | 83.1% | | | |
| Needle, 2K–16K | 100% | | | |
| LangGraph agent, 20 scripted tasks on unseen tools | | | | |

## Speed and memory

_Pending._ llama-bench on the RTX 5080: prompt reading and writing speed, and GPU memory with the full context
reserved, at 4K and 16K, 16-bit and 8-bit KV cache.

## LangGraph demo

[`demo/agent.py`](../demo/agent.py): the usual tool loop as a LangGraph graph (`model → tools → model → … → END`),
LangChain's `ChatOpenAI` pointed at llama-server. Seven tools in three areas the model never saw in training (car
servicing, gold and currency prices, parcel shipping) with a fixed fake backend, and 20 scripted tasks
([`demo/tasks.json`](../demo/tasks.json)): one call, parallel calls, chained calls (a search result feeding a
booking), a missing detail to ask for, no tool fits, a tool error, and follow-up turns.

```
uv run python demo/agent.py --server http://127.0.0.1:8095 --chat
```
