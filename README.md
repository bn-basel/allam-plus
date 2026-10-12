# allam-plus

Upgrading HUMAIN's open **ALLaM-7B-Instruct** for real Arabic applications, on one 16 GB gaming GPU, with every
change measured against the untouched model:

1. **Long context**: 4K → 16K tokens (YaRN + a short long-text training stage).
2. **Arabic tool calling**: general tools, Modern Standard Arabic and Saudi dialect.
3. **Agent-ready serving**: an OpenAI-compatible server that returns standard `tool_calls` (LangGraph and others).
4. **Fast quantised releases**: GGUF 4/8-bit with a speed and memory table.
5. **No loss of Arabic quality**: ArabicMMLU and Arabic IFEval before and after.

**Status:** Phases 1–3 done; Phase 4 (serving and release) built, its GPU tests pending. **ALLaM-7B now reads 16K tokens (4× its original
4K) and calls tools in Arabic, with no loss on general Arabic tasks**, all trained on one RTX 5080.

| | As released | **allam-plus** |
|---|---|---|
| Needle at 16K tokens | 0% | **100%** |
| LC-Eval (HUMAIN), multiple choice on 8–16K-token books | 18.3% | **64.7%** |
| ArabFuncBench, right tool / every argument right | 65.9% / 19.0% | **96.8% / 48.5%** |
| AISA-ArabicFC (five dialects), right tool / every argument right | 68.8% / 10.8% | **93.4% / 36.2%** |
| Saudi receptionist test (earlier project) | 36.3% | **83.1%** |
| ArabicMMLU | 68.6% | **69.2%** |
| Arabic IFEval | 40.8% | **42.3%** |

![Tool calling](docs/charts/phase3_tools.svg)

![Long context](docs/charts/phase2_long_context.svg)

Details: [baselines](docs/BASELINES.md) · [long context](docs/PHASE2.md) · [tool calling](docs/PHASE3.md) · [serving](docs/PHASE4.md)

Decisions and reasons: [docs/DECISIONS.md](docs/DECISIONS.md) · Learning log: [LEARNING.md](LEARNING.md) ·
Terms: [GLOSSARY.md](GLOSSARY.md)

Builds on [mujawib-receptionist-finetune](https://github.com/bn-basel/mujawib-receptionist-finetune) (ALLaM-7B and
Gemma-4-12B fine-tuned as Saudi-dialect receptionists with tool calling).
