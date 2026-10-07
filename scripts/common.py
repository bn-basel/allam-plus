"""Shared helpers: load ALLaM with a chosen position setting, greedy generation, Arabic text normalisation.

Position settings ("rope"):
  shipped  the config as released: RoPE with theta 1,000,000, trained for 4,096 positions, no scaling.
  yarn4    YaRN with factor 4 over the original 4,096 positions (so 16,384), set in the config only, no training.
           The same numbers llama.cpp takes as --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 4096.
"""

import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "humain-ai/ALLaM-7B-Instruct-preview"
ORIGINAL_CTX = 4096

ROPE = {
    "shipped": None,
    "yarn4": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": ORIGINAL_CTX},
}


def load_model(rope: str = "shipped", adapter: str | None = None):
    """ALLaM in 4-bit NF4 (the same way it is trained here), bf16 compute, SDPA attention, on GPU 0."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    config = AutoConfig.from_pretrained(MODEL_ID)
    if ROPE[rope]:
        config.rope_parameters = {"rope_theta": config.rope_parameters["rope_theta"], **ROPE[rope]}
        config.max_position_embeddings = int(ORIGINAL_CTX * ROPE[rope]["factor"])
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, config=config, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def chat_prompt(tok, user: str, system: str | None = None) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


PREFILL_CHUNK = 4096


def new_cache(model, length: int):
    """A KV cache reserved once for `length` tokens. The default cache grows by copying itself at every step; at 16K
    tokens that fragments PyTorch's memory pool (26 GB reserved for 13.5 GB of data on our card), which Windows then
    spills into slow shared memory. A cache of fixed size never moves."""
    from transformers import StaticCache
    return StaticCache(config=model.config, max_cache_len=length)


def prefill(model, ids: list[int], cache, start: int = 0, chunk: int = PREFILL_CHUNK):
    """Read tokens into the cache at positions start, start+1, ... in slices of 4,096 ("chunked prefill").

    Reading 16K tokens in one pass needs memory for every intermediate result of all 16K positions at once (14.6 GB
    peak). In slices, each slice attends to the cache of everything before it: the same computation in a different
    order, with a peak of about weights + KV cache. Prompts up to 4,096 tokens are one slice, i.e. the usual forward
    pass. Writing at an earlier `start` overwrites the cache from there on (positions after the last one written are
    masked out), which is how a long shared text is read once and several questions are asked after it.
    Returns log-probabilities of the next token."""
    import torch

    # transformers' StaticCache writes at its own running count, not at the positions passed in: set that count to
    # `start` so the write lands where the attention mask expects it.
    for layer in getattr(cache, "layers", []):
        if getattr(layer, "is_initialized", False):
            layer.cumulative_length.fill_(start)
    out = None
    for s in range(0, len(ids), chunk):
        piece = ids[s:s + chunk]
        pos = torch.arange(start + s, start + s + len(piece), device=model.device)
        out = model(input_ids=torch.tensor([piece], device=model.device), past_key_values=cache, cache_position=pos,
                    use_cache=True, logits_to_keep=1)
    return torch.log_softmax(out.logits[0, -1].float(), -1)


def generate(tok, model, prompt: str, max_new_tokens: int = 64) -> dict:
    """Greedy continuation (chunked prefill, then one token at a time, always the most likely token: the same as
    generate(do_sample=False)). Returns the text, prompt length in tokens, seconds and peak GPU memory (GB)."""
    import torch

    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    # Give the previous case's freed blocks back: an 8K cache leaves blocks too small for a 16K one, and keeping both
    # reserved (19.6 GB) pushes a 16 GB card into shared memory.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    new = []
    cache = new_cache(model, len(ids) + max_new_tokens)
    with torch.no_grad():
        logp = prefill(model, ids, cache)
        for _ in range(max_new_tokens):
            t = int(logp.argmax())
            new.append(t)
            if t in eos:
                break
            logp = prefill(model, [t], cache, start=len(ids) + len(new) - 1)
    del cache
    return {"text": tok.decode(new, skip_special_tokens=True).strip(), "prompt_tokens": len(ids),
            "new_tokens": len(new), "seconds": round(time.time() - t0, 2),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}


HINDI_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # harakat, Quranic marks, tatweel


def normalize_ar(text: str) -> str:
    """For matching only: no diacritics or tatweel, one alef, ة→ه, ى→ي, Hindi digits → Arabic digits, one space."""
    text = DIACRITICS.sub("", text or "").translate(HINDI_DIGITS)
    text = re.sub("[أإآٱ]", "ا", text).replace("ة", "ه").replace("ى", "ي")
    return re.sub(r"\s+", " ", text).strip()
