"""Training memory probe: can QLoRA train ALLaM on 4K / 8K / 12K / 16K-token examples on a 16 GB card?

    uv run python scripts/train_probe.py --lengths 4096 8192 12288 16384 [--rank 16]

Same setup as the receptionist training (saudi-dialect scripts/train.py): 4-bit NF4 base, LoRA rank 16 on all linear
layers (alpha 32), gradient checkpointing (non-reentrant), bf16, paged 8-bit AdamW, batch size 1, YaRN factor 4 in the
config. For each length: 3 optimizer steps on one real Arabic text of that length (LC-Eval book text: only its token
count matters here, nothing is learned or saved). Reports peak GPU memory and seconds per step.

Gradient checkpointing, in one line: instead of keeping every layer's intermediate results for the backward pass (memory
grows with length × layers), keep only each layer's input and recompute the rest when needed: ~30% more compute for
much less memory.
"""

import argparse
import contextlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402


def chunked_loss(model, ids, chunk: int = 1024):
    """Next-token loss without ever holding the full logits. At 16K tokens the logits are 16,384 × 64,000 numbers
    (2 GB in bf16, 4 GB once turned to float32 for the loss): more than the LoRA training itself needs. Here the
    output layer runs on 1,024 positions at a time, each chunk under checkpointing, so only one chunk's logits exist
    at any moment."""
    import torch
    from torch.utils.checkpoint import checkpoint

    inner = model.get_base_model()
    hidden = inner.model(input_ids=ids).last_hidden_state[0, :-1]
    targets = ids[0, 1:]

    def part(h, t):
        return torch.nn.functional.cross_entropy(inner.lm_head(h).float(), t, reduction="sum")

    total = sum(checkpoint(part, hidden[i:i + chunk], targets[i:i + chunk], use_reentrant=False)
                for i in range(0, hidden.shape[0], chunk))
    return total / targets.shape[0]


def main() -> None:
    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig, get_peft_model
    from common import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 12288, 16384])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--chunked-loss", action="store_true",
                    help="compute the output layer + loss 1,024 tokens at a time (see chunked_loss)")
    ap.add_argument("--offload", action="store_true",
                    help="keep the tensors saved for the backward pass in system RAM (pinned) instead of on the GPU")
    args = ap.parse_args()

    tok, model = load_model("yarn4")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.config.use_cache = False
    model.train()  # load_model() leaves eval mode, where transformers skips gradient checkpointing altogether
    opt = bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=1e-5)

    text = " ".join(re.sub(r"\s+", " ", json.loads(l)["full_context"])
                    for l in open(ROOT / "data" / "raw" / "lceval" / "lceval_ar.jsonl", encoding="utf-8")
                    if "Claim" in l[:20000])
    all_ids = tok(text[:400_000], add_special_tokens=False)["input_ids"]
    results = []
    for L in args.lengths:
        ids = torch.tensor([all_ids[:L]], device=model.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        times, err = [], None
        try:
            for _ in range(args.steps):
                t0 = time.time()
                # --offload: what gradient checkpointing keeps (each layer's input, 128 MB per layer at 16K) goes to
                # system RAM and comes back during the backward pass: slower, but ~4 GB less GPU memory at 16K.
                with torch.autograd.graph.save_on_cpu(pin_memory=True) if args.offload else contextlib.nullcontext():
                    loss = chunked_loss(model, ids) if args.chunked_loss else model(input_ids=ids, labels=ids).loss
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                times.append(time.time() - t0)
        except torch.OutOfMemoryError as e:
            err = "out of memory"
            del e
        peak = torch.cuda.max_memory_allocated() / 1e9
        reserved = torch.cuda.max_memory_reserved() / 1e9
        r = {"length": L, "peak_allocated_gb": round(peak, 2), "peak_reserved_gb": round(reserved, 2),
             "s_per_step": round(sum(times[1:]) / max(1, len(times) - 1), 1) if len(times) > 1 else None,
             "tokens_per_s": round(L / (sum(times[1:]) / (len(times) - 1)), 0) if len(times) > 1 else None,
             "loss": round(float(loss), 3) if times else None, "error": err}
        print(json.dumps(r), flush=True)
        results.append(r)
        if err:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    out = ROOT / "results" / "phase1" / f"train_probe{'_chunked' if args.chunked_loss else ''}{'_offload' if args.offload else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rank": args.rank, "rope": "yarn4", "chunked_loss": args.chunked_loss, "offload": args.offload,
                               "results": results}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
