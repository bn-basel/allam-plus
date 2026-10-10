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

**What I learned.**
- Check question: "Why did YaRN change answers on short MMLU questions?" I wasn't sure and guessed it was linked to
  the needle test's lab code. It isn't: YaRN is not a mode that starts after 4K. It re-labels *every* position (the
  slow clock hands turn 4× slower for all texts, and attention is sharpened ×1.14), so even in a 1,000-token question
  the distances between words look slightly different from what the model learned. Like a ruler re-labelled for
  long distances that now reads short ones slightly wrong. Hence the training stage with YaRN on, and plan B (YaRN
  only for long requests).
- 4K → 16K is 4× the context, not 3×.
- 16K training first looked impossible on 16 GB; parking the backward-pass tensors in system RAM made it fit
  (12.1 GB). Worth asking "what else could we move?" before accepting a hardware limit.

**Lessons from the bugs.**
- When a GPU job is suddenly 10–30× slower, check memory first: Windows lets programs use more GPU memory than the
  card has by borrowing system RAM, which is far slower. The tell-tale sign: GPU "100% busy" but drawing little power.
- PyTorch keeps freed GPU memory in a pool for reuse. A cache that grows step by step leaves holes the next request
  cannot use: 13.5 GB of data needed 26 GB of pool. Reserving the cache once, at full size, fixed it.
- A multiple-choice score only needs the scores at the last position; computing all positions for 16 long questions
  at once filled the card.
- `model.eval()` turns off gradient checkpointing in transformers: a training-memory test must run in `model.train()`
  mode, or it measures the wrong thing.

---

## Step 2 — Long context: from 4K to 16K tokens

**Concept.** YaRN alone let ALLaM *find* a fact at 16K but only half-*understand* a 16K book, and it cost 1.4 points
on short questions. A short training stage fixes both: the model practises reading long Arabic texts with the
stretched positions switched on, and at the same time practises giving the *same* short answers as the original
model. That second part is **self-distillation** (the original model is the teacher of its YaRN copy), used here as
**replay**: mixing old behaviour back in so new training does not wash it out. The training material was written by
ALLaM itself: shown one short Wikipedia section, it wrote two question-answer pairs; we then buried that section
inside 6–16K tokens of other articles and asked the question at the end. No paid API.

**What we did.**
- Cleaned 38,333 Arabic Wikipedia articles; ALLaM wrote 2,346 question-answer pairs (rules dropped pairs with Latin
  letters, answers not taken from the section, or no question mark) and 1,125 replay answers.
- Built 450 long chats (6–16K tokens, three questions each), 100 long articles and 919 short replay examples. Kept
  out of training: book text (LC-Eval's source) and anything like the needle test.
- Trained one QLoRA run: 183 steps, 2 h 19 min, 12.0 GB peak, using the memory tricks from Step 1 plus parking
  activations in system RAM for long examples.
- Chose checkpoint 183 by the rule fixed in advance (best long-question score on held-out articles).
- Results: needle **100% at every length** (was 0% at 8K/16K as released); LC-Eval multiple choice **63.8%** (18.3%
  as released, 47.8% with YaRN alone); ArabicMMLU **68.8%**, the same as the original 68.6%, so the YaRN cost is
  gone; Arabic IFEval unchanged. All four targets met.
- The model went from 4K to 16K tokens: **4×**, on one 16 GB card.

**What I learned.** _(after the check question)_

**Notes.**
- A scoring bug caught before it mattered: the first multiple-choice scorer read only the reply's first character,
  but the model writes «بحسب مقالة ...، ... A. ...». Fixed and the starting point re-scored before any checkpoint was
  scored, so the checkpoint choice stayed fair.
- The dev score moved little after checkpoint 40 (0.72 → 0.70 → 0.67 → 0.71 → 0.73), while the replay loss kept
  falling (0.45 → 0.36): the long-reading skill came early, the "behave like the original" part took the whole run.
