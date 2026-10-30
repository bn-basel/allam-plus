# allam-plus

HUMAIN's open Arabic model **ALLaM-7B-Instruct**, upgraded on one gaming GPU (RTX 5080, 16 GB):

- **Reads 4× more:** 16K tokens instead of 4K.
- **Uses tools in Arabic:** books, looks up and checks things, in Modern Standard Arabic and Saudi dialect.
- **Plugs into standard apps:** one GGUF file gives an OpenAI-style server (llama.cpp, Ollama, LangGraph).
- **Lost nothing:** general Arabic knowledge and instruction following stay the same.

**Download:** [huggingface.co/bnbasel/allam-plus-7b](https://huggingface.co/bnbasel/allam-plus-7b)

| | Before | **After** | After, 4-bit file through a server |
|---|---|---|---|
| Finds a hidden fact in 16K tokens of text | 0% | **100%** | 100% |
| Answers questions about long books (HUMAIN's LC-Eval) | 18.3% | **66.1%** | |
| ArabFuncBench, right tool / every argument right | 65.9% / 19.0% | **96.0% / 47.6%** | 99.3% / 48.0% |
| AISA-ArabicFC (five dialects), right tool / every argument right | 68.8% / 10.8% | **92.7% / 31.6%** | 97.6% / 34.2% |
| Saudi receptionist test (earlier project) | 36.3% | **85.0%** | 86.9% |
| ArabicMMLU | 68.6% | **69.2%** | |
| Arabic IFEval | 40.8% | **41.3%** | |
| LangGraph agent, 20 tasks on tools it never saw | | | 17 / 20 |

The last column is how an app uses it: the downloaded file in llama-server, asked through the standard chat API.

![Served release files](docs/charts/phase4_served.svg)

![Long context](docs/charts/phase2_long_context.svg)

## Try it

```
llama-server -m allam-plus-7b-Q4_K_M.gguf -ngl 99 -c 16384 -ctk q8_0 -ctv q8_0 -fa on --jinja --port 8095
uv run python demo/agent.py --server http://127.0.0.1:8095 --chat
```

Any OpenAI client works: send `messages` and `tools`, get `tool_calls` back. Ollama 0.40+: `ollama create allam-plus -f
release/Modelfile`. Q4_K_M with the 8-bit KV cache needs 8.6 GB of GPU memory at 16K ([speed and memory](docs/PHASE4.md)).

Details: [baselines](docs/BASELINES.md) · [long context](docs/PHASE2.md) · [tool calling](docs/PHASE3.md) · [serving](docs/PHASE4.md)

Decisions and reasons: [docs/DECISIONS.md](docs/DECISIONS.md) · Learning log: [LEARNING.md](LEARNING.md) ·
Terms: [GLOSSARY.md](GLOSSARY.md)

Builds on two Saudi-dialect receptionist fine-tunes with tool calling on Hugging Face:
[saudi-receptionist-gemma-4-12b](https://huggingface.co/bnbasel/saudi-receptionist-gemma-4-12b) and
[allam-7b-saudi-receptionist](https://huggingface.co/bnbasel/allam-7b-saudi-receptionist).

Licence: [Apache-2.0](LICENSE).
