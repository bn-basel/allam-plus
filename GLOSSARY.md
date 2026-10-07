# Glossary

Plain meanings of the terms used in this project, each with an example. Newest additions at the end of each section.

## Long context

**Context window.** How many tokens the model can read at once (prompt + its reply). ALLaM-7B was trained on 4,096
tokens: about 1,300 Arabic words (ALLaM averages ~3.2 characters of Arabic per token).

**Token.** A piece of text the model reads as one unit. «المدرسة» may be one token, a rare name may be three.

**RoPE (rotary position embedding).** How a Llama-style model knows word order: each position turns the query and key
vectors by an angle, with many "clock hands" turning at different speeds. Fast hands tell neighbours apart, slow hands
give the rough position in a long text. Example: position 10 and position 11 differ a lot on the fastest hand, a
tiny amount on the slowest.

**rope_theta.** Sets how slow the slowest clock hands turn. ALLaM uses 1,000,000 (Llama 2 used 10,000), so its slow
hands barely move within 4K tokens.

**Extrapolation.** Using a model on positions longer than it was trained on. The hands reach angles it never saw, and
the attention breaks down: the model produces fluent-looking nonsense.

**YaRN (Yet another RoPE extensioN).** A way to stretch RoPE: the slow hands are slowed by a factor (here 4, so 16K
positions look like 4K), the fast hands are left alone (they never left their trained range), the ones in between are
blended, and attention is made slightly sharper (×1.14 for factor 4) to make up for the blur. Like re-labelling a 4K
ruler so it measures 16K, while keeping the millimetre marks exact. Can be switched on in the config alone, or followed
by a short training stage so the model gets used to it.

**Static vs dynamic scaling.** Static (transformers, llama.cpp): the stretch is applied to every text, short ones
too, so short-text quality can move a little. Dynamic: the stretch grows only once the text passes 4K. We measure the
static cost at 2K/4K.

**KV cache.** The model's notes on what it has already read: one key and one value vector per token, per layer, per
attention head. Without it, every new word would mean re-reading the whole text. For ALLaM: 2 × 32 layers × 32 heads ×
128 numbers × 2 bytes = 0.5 MiB per token, so 16K tokens = 8 GiB in 16-bit.

**KV-cache quantisation.** Storing those notes in 8 bits (q8_0, half the memory) or 4 bits (q4_0, a quarter). Example:
32K tokens = 16 GiB in 16-bit (doesn't fit next to the model on a 16 GB card), 8 GiB in q8_0, 4 GiB in q4_0.

**GQA (grouped-query attention).** Many models share one key/value head between several query heads to shrink the
KV cache. ALLaM does not (32 query heads, 32 KV heads), which is why its cache is large for a 7B model.

**Needle in a haystack.** A long-context test: hide one short invented fact (the needle, e.g. «رمز الدخول إلى مختبر
الفيزياء هو 9052») at a chosen depth in a long unrelated text (the haystack) and ask for it at the end. Run over a grid
of lengths × depths; a failure at a length shows where reading breaks down.

**Depth.** Where the needle sits: 10% = near the start, 90% = near the end. Models often lose facts in the middle
("lost in the middle").

**Control case.** The same question with no needle in the text. If the model still "answers", it is guessing, and the
test's hits mean less.

**Gradient checkpointing.** Saving memory in training: keep only each layer's input during the forward pass and
recompute the rest during the backward pass. About 30% more compute, much less memory, which matters at 16K tokens.

**Chunked loss.** At 16K tokens the model's raw output scores (logits) are 16,384 × 64,000 numbers, about 2–4 GB.
Computing the loss 1,024 positions at a time means only one slice exists at a time.

## Evaluation

**Exact match (needle).** Correct if the expected answer appears in the reply after normalisation (no diacritics, one
alef form, ة→ه, ى→ي, Hindi digits ٠-٩ turned into Arabic digits 0-9).

**Log-likelihood scoring.** For multiple choice, instead of reading the model's written answer, compare the
probabilities it gives to " A", " B", " C", " D" as the next token and take the highest. No parsing failures, and the
same method as the LM Evaluation Harness.

**Position bias (in a test).** When the right answer sits in one position far more often than chance. In LC-Eval's
Arabic multiple-choice items it is option 1 in 62% of cases, so "always A" would score 62%. Fix: shuffle the options
with a fixed seed.

**Wilson interval.** A 95% confidence range for a percentage from a limited number of cases. Example: 267/400 = 66.8%,
range 62.0–71.2%.

**Prompt-level vs instruction-level (IFEval).** Prompt-level: every instruction in the prompt was followed (strict,
the user's view). Instruction-level: share of single instructions followed (partial credit).

## Tool calling

**Tool call (function call).** The model replies with a structured request instead of text, e.g.
`<tool_call>{"name": "get_prayer_times", "arguments": {"city": "الرياض"}}</tool_call>`; your code runs it and sends
the result back.

**Tool selection accuracy (TSA / FnAcc).** Right tool chosen, or rightly no tool when none fits.

**Argument exact match (ArgEM).** Right tool and every argument right after normalisation.

**Negative case.** A request that needs none of the offered tools; the right answer is a normal reply. Tests whether the
model calls tools when it shouldn't.

## Engineering (added in Phase 1)

**Prefill vs decode.** Prefill: reading the whole prompt into the KV cache (thousands of tokens per second, all
positions at once). Decode: writing the answer one token at a time (tens to hundreds of tokens per second).

**Chunked prefill.** Reading a long prompt in slices (here 4,096 tokens), each slice attending to the cache of what
came before. Same result, much lower peak memory.

**Static KV cache.** A KV cache reserved once at its full size, instead of growing. Avoids memory fragmentation.

**Memory fragmentation.** Free memory split into pieces too small for the next request, so new memory has to be
reserved even though plenty is "free" in total.

**Shared-memory spill (Windows).** When a program asks for more GPU memory than the card has, Windows lends it system
RAM over the PCIe bus. Nothing crashes, but the job becomes many times slower.

**Paired test (McNemar).** Comparing two models on the same questions by counting only the questions where they
disagree (A right / B wrong vs the reverse). Much more sensitive than comparing two overall percentages. Example:
YaRN lost 676 ArabicMMLU questions and gained 476: p < 0.001, a real difference even though both scores are ~68%.

**Balanced accuracy.** The average of "true items called true" and "false items called false". A model that always
gives the same answer scores 50%, whatever the mix of true and false items.

**Schema copying.** A tool-calling mistake: the model pastes the tool's description (`"properties": {...}`) into the
arguments instead of filling in values.

**Activation offloading.** During training, the values kept for the backward pass (with gradient checkpointing: each
layer's input, 128 MB per layer at 16K tokens) are copied to system RAM and brought back when needed. Costs some
time over PCIe, saves ~4 GB of GPU memory at 16K.
