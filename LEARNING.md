# Learning log

One entry per step: the concept in one paragraph, what we did, and what I learned. Terms are defined in
[GLOSSARY.md](GLOSSARY.md). Decisions and their reasons: [docs/DECISIONS.md](docs/DECISIONS.md).

---

## Step 0 — The plan, and checking the ground before building

**Concept.** "Make ALLaM better" only means something if every improvement is measured against the untouched model
on tests chosen *before* training. This project adds three things to ALLaM-7B-Instruct (16K context, Arabic tool
calling, a server that agent frameworks can use) and promises one thing it must not lose (Arabic quality). So the first
week trains nothing: it measures the base model on every test, and checks that the GPU can even train at 16K.

**What we did.**
- Checked every outside resource before planning around it:
  - **ArabFuncBench** exists: 1,000 native-MSA tool-calling cases, CC BY 4.0, gated. The authors even published their
    ALLaM-7B evaluation notebook, so we can reproduce their number.
  - **No usable public Arabic tool-calling training set.** The big ones were generated with GPT-4o, whose terms don't
    allow using its output to build competing models; QCRI's translated sets were never released. So we will generate
    our own with our rule-driven pipeline, and use the public sets only as tests.
  - **ArabicMMLU** (14,455 questions) and **Arabic IFEval** (404 prompts) cover knowledge and instruction following:
    the "no quality loss" check.
  - **LC-Eval**, HUMAIN's own long-context benchmark, gives realistic Arabic long-document questions next to our
    synthetic needle test.
- Set up a new repo with the same software versions as the receptionist project, so numbers stay comparable.

**What I learned.** _(after the Phase 1 check question)_

---

## Step 1 — Baselines: what ALLaM-7B can do before we touch it

**Concept.** ALLaM knows word order through RoPE, a set of "clock hands" turning at different speeds as the position
grows. It was trained on texts up to 4,096 tokens, so beyond that its hands reach angles it has never seen. **YaRN**
stretches the slow hands by a factor (here 4, so 16K positions look like 4K) and leaves the fast ones alone; it can be
switched on in the config without any training. The question for week 1: how far does that get us, and what does it
cost? To answer it we need tests at several levels: a **needle test** (can it *find* one hidden fact?), a realistic
**long-document test** (can it *understand* a long book?), and **short-text tests** (did anything break?).

**What we did.**
- **Needle test** (210 cases per version, Arabic book text). As released: perfect at 2K and 4K, then **0% at 8K and
  16K**: the model writes fluent-looking nonsense. With YaRN in the config: **90% at 8K and 16K**, and still perfect
  at 2K/4K. No-needle controls: never guessed.
- **LC-Eval** (HUMAIN's Arabic long-context benchmark, 8–16K-token books): multiple choice goes from 18% (worse than
  guessing) to **48%** with YaRN. Finding one sentence is much easier than understanding a book.
- **ArabicMMLU** (14,455 questions): 68.6% as released, which matches the model card (67.78), so our scorer is sound.
  **YaRN costs 1.4 points** here: static YaRN stretches positions in short texts too. **Arabic IFEval**: no change.
- **Tool calling** (AISA, 545 requests in 5 dialects): the base picks the right tool 65% of the time but gets every
  argument right only 9% of the time; it often copies the tool's schema instead of filling in values. The receptionist
  model (trained on 6 tools) picks the right tool 85% of the time on 27 unseen tools, but calls a tool even when none
  fits (31% vs 87%).
- **Memory.** In llama.cpp, 32K fits only with an 8- or 4-bit KV cache. For training, 12K fits once the loss is
  computed in slices; **16K fits too** once the tensors kept for the backward pass are parked in system RAM
  (12.1 GB, 21 s per step). I first stopped at 12K; Basel pushed for 16K, and one more test showed it works.
- **Four memory problems found and fixed on the way**, each a lesson about GPU memory on Windows (see below).

**What I learned.** _(after the check question)_

**Lessons from the bugs.**
- When a GPU job is suddenly 10–30× slower, check memory first: Windows lets programs use more GPU memory than the
  card has by borrowing system RAM, which is far slower. The tell-tale sign: GPU "100% busy" but drawing little power.
- PyTorch keeps freed GPU memory in a pool for reuse. A cache that grows step by step leaves holes the next request
  cannot use: 13.5 GB of data needed 26 GB of pool. Reserving the cache once, at full size, fixed it.
- A multiple-choice score only needs the scores at the last position; computing all positions for 16 long questions
  at once filled the card.
- `model.eval()` turns off gradient checkpointing in transformers: a training-memory test must run in `model.train()`
  mode, or it measures the wrong thing.
