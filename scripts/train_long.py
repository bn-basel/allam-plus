"""Phase 2: teach ALLaM-7B-Instruct to read 16K tokens (QLoRA, YaRN ×4 in the config).

    uv run python scripts/train_long.py --run long_v1 --dry-run     # encode and check the data only
    uv run python scripts/train_long.py --run long_v1 --max-steps 2 # smoke test: memory and speed
    uv run python scripts/train_long.py --run long_v1               # the run; checkpoints in runs/long_v1/
    uv run python scripts/train_long.py --run long_v1 --resume      # continue from the latest checkpoint

Data: data/long/train.jsonl (scripts/longdata.py): long-context question answering (6K-16K tokens, three questions per
context as a chat), plain long Wikipedia text, and short "replay" answers from the original model.
Loss: chat examples train only what the assistant writes (each answer and its closing </s>), rendered with ALLaM's
own chat template exactly as the tests render prompts (checked per example); plain-text examples train every token.

Memory (Phase 1 probe, docs/DECISIONS.md D11): 4-bit NF4 base, LoRA rank 16 on all linear layers, gradient
checkpointing, the output layer and loss computed 1,024 positions at a time and only where the loss looks, and for
examples over 8K tokens the tensors kept for the backward pass parked in system RAM. Batch size 1; gradients are
accumulated over --accum examples; every example counts equally (its loss is a mean over its trained tokens).

Pause cleanly: create runs/<run>/PAUSE (scripts/pause_training.ps1); the run saves a checkpoint after the current
optimizer step and stops. --resume reloads the latest adapter and skips the examples already used (the optimizer
restarts; a small, documented difference from an uninterrupted run).
"""

import argparse
import contextlib
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402

DATA = ROOT / "data" / "long"
OFFLOAD_ABOVE = 8192


def encode(ex: dict, tok) -> dict:
    """Token ids and labels (-100 = not trained)."""
    if ex["kind"] == "lm":
        ids = [tok.bos_token_id] + tok(ex["text"], add_special_tokens=False)["input_ids"]
        return {"input_ids": ids, "labels": ids[:]}
    msgs = ex["messages"]
    text = tok.apply_chat_template(msgs, tokenize=False)
    spans = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prompt = tok.apply_chat_template(msgs[:i], add_generation_prompt=True, tokenize=False)
        upto = tok.apply_chat_template(msgs[: i + 1], tokenize=False)
        assert upto.startswith(prompt) and text.startswith(upto), "template prefixes are not consistent"
        spans.append((len(prompt), len(upto)))
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    labels = [tid if any(s <= a and b <= e for s, e in spans) and b > a else -100
              for tid, (a, b) in zip(enc["input_ids"], enc["offset_mapping"])]
    # train exactly as tested: the tokens before the first answer must equal the test prompt's tokens
    first = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    first_prompt = tok.apply_chat_template(msgs[:first], add_generation_prompt=True, tokenize=False)
    p_ids = tok(first_prompt, add_special_tokens=False)["input_ids"]
    assert enc["input_ids"][: len(p_ids)] == p_ids, "training tokens differ from the test prompt's tokens"
    return {"input_ids": enc["input_ids"], "labels": labels}


def chunked_loss(model, ids, labels, chunk: int = 1024):
    """Mean next-token loss over trained positions, with the 64,000-wide output layer run only on positions whose
    next token is trained, 1,024 at a time under checkpointing (never the full 16K × 64K logits table)."""
    import torch
    from torch.utils.checkpoint import checkpoint

    inner = model.get_base_model()
    hidden = inner.model(input_ids=ids).last_hidden_state[0]
    keep = (labels[0, 1:] != -100).nonzero().squeeze(-1)
    h, t = hidden[keep], labels[0, keep + 1]

    def part(hh, tt):
        return torch.nn.functional.cross_entropy(inner.lm_head(hh).float(), tt, reduction="sum")

    total = sum(checkpoint(part, h[i:i + chunk], t[i:i + chunk], use_reentrant=False) for i in range(0, len(t), chunk))
    return total / max(1, len(t))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default=str(DATA / "train.jsonl"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=8, help="optimizer steps of linear warm-up")
    ap.add_argument("--save-every", type=int, default=20, help="optimizer steps between checkpoints")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-adapter", help="start from this trained adapter (e.g. the Phase 2 checkpoint) instead of a "
                                           "fresh LoRA")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from common import MODEL_ID

    out = ROOT / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    examples = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    rng = random.Random(args.seed)
    order = []
    for e in range(math.ceil(args.epochs)):
        ep = list(range(len(examples)))
        rng.shuffle(ep)
        order += ep
    order = order[: int(len(examples) * args.epochs)]
    total_steps = len(order) // args.accum
    t0 = time.time()
    encoded = {}
    for i in set(order):
        encoded[i] = encode(examples[i], tok)
    n_tokens = sum(len(encoded[i]["input_ids"]) for i in order)
    n_trained = sum(sum(l != -100 for l in encoded[i]["labels"]) for i in order)
    info = {"args": vars(args), "examples": len(examples), "kinds": {k: sum(e["kind"] == k for e in examples)
            for k in sorted({e["kind"] for e in examples})}, "steps": total_steps, "tokens_seen": n_tokens,
            "tokens_trained": n_trained, "max_len": max(len(v["input_ids"]) for v in encoded.values()),
            "encode_seconds": round(time.time() - t0)}
    print(json.dumps(info, indent=1), flush=True)
    if not args.resume:
        (out / "data.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    if args.dry_run:
        return

    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from common import load_model

    torch.manual_seed(args.seed)
    _, model = load_model("yarn4")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    start_step = 0
    if args.resume:
        ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if not ckpts:
            sys.exit("no checkpoint to resume from")
        model = PeftModel.from_pretrained(model, ckpts[-1], is_trainable=True)
        start_step = int(ckpts[-1].name.split("-")[1])
        print(f"resuming from {ckpts[-1].name}", flush=True)
    elif args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        print(f"starting from {args.init_adapter}", flush=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                                 target_modules="all-linear", task_type="CAUSAL_LM"))
    model.config.use_cache = False
    model.train()  # gradient checkpointing only works in training mode
    params = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.PagedAdamW8bit(params, lr=args.lr, weight_decay=0.0)

    def lr_at(step):  # linear warm-up, then cosine to 10% of the peak
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    log = open(out / "train.log", "a", encoding="utf-8")
    step = start_step
    stop = args.max_steps or total_steps
    t_step, acc_loss, acc_kinds = time.time(), [], []
    for k in range(start_step * args.accum, len(order)):
        ex = examples[order[k]]
        enc = encoded[order[k]]
        ids = torch.tensor([enc["input_ids"]], device=model.device)
        labels = torch.tensor([enc["labels"]], device=model.device)
        offload = len(enc["input_ids"]) > OFFLOAD_ABOVE
        # Not pinned: pinned (page-locked) host memory is cached by PyTorch and never given back, and with examples of
        # different lengths it grew to 27.7 GB of system RAM in tools_v1 and slowed training 15x (12 Oct).
        with torch.autograd.graph.save_on_cpu(pin_memory=False) if offload else contextlib.nullcontext():
            loss = chunked_loss(model, ids, labels)
        (loss / args.accum).backward()
        if offload:  # long examples leave large freed blocks in the GPU pool; hand them back
            torch.cuda.empty_cache()
        acc_loss.append(float(loss))
        acc_kinds.append(ex["kind"])
        del ids, labels, loss
        if (k + 1) % args.accum:
            continue
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        rec = {"step": step, "of": total_steps, "loss": round(sum(acc_loss) / len(acc_loss), 4),
               "loss_by_kind": {kk: round(sum(l for l, kd in zip(acc_loss, acc_kinds) if kd == kk) /
                                          max(1, acc_kinds.count(kk)), 4) for kk in set(acc_kinds)},
               "lr": lr_at(step - 1), "seconds": round(time.time() - t_step, 1),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        t_step, acc_loss, acc_kinds = time.time(), [], []
        torch.cuda.reset_peak_memory_stats()
        pause = (out / "PAUSE").exists()
        if step % args.save_every == 0 or step == total_steps or pause or step >= stop:
            model.save_pretrained(out / f"checkpoint-{step}")
            print(f"saved checkpoint-{step}", flush=True)
        if pause:
            print(f"paused at step {step}; continue with --resume", flush=True)
            return
        if step >= stop:
            break
    print("done", flush=True)


if __name__ == "__main__":
    main()
