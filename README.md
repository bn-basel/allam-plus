# allam-plus

Upgrading HUMAIN's open **ALLaM-7B-Instruct** for real Arabic applications, on one 16 GB gaming GPU, with every
change measured against the untouched model:

1. **Long context**: 4K → 16K tokens (YaRN + a short long-text training stage).
2. **Arabic tool calling**: general tools, Modern Standard Arabic and Saudi dialect.
3. **Agent-ready serving**: an OpenAI-compatible server that returns standard `tool_calls` (LangGraph and others).
4. **Fast quantised releases**: GGUF 4/8-bit with a speed and memory table.
5. **No loss of Arabic quality**: ArabicMMLU and Arabic IFEval before and after.

**Status: Phase 1 (baselines) done.** Results: [docs/BASELINES.md](docs/BASELINES.md).

![Needle test](docs/charts/needle_phase1.svg)

![Baselines](docs/charts/baselines_phase1.svg)

Decisions and reasons: [docs/DECISIONS.md](docs/DECISIONS.md) · Learning log: [LEARNING.md](LEARNING.md) ·
Terms: [GLOSSARY.md](GLOSSARY.md)

Builds on [mujawib-receptionist-finetune](https://github.com/bn-basel/mujawib-receptionist-finetune) (ALLaM-7B and
Gemma-4-12B fine-tuned as Saudi-dialect receptionists with tool calling).
