# Phase 2: long context, 4K → 16K tokens

One QLoRA run on one RTX 5080 (16 GB): `runs/long_v1`, checkpoint 183 (chosen on the dev set by the rule fixed
before training, [DECISIONS.md](DECISIONS.md) D14). Model served with YaRN ×4 in the config.

![Phase 2 before and after](charts/phase2_long_context.svg)

| Test | Target | As released | + YaRN ×4, no training | **+ YaRN ×4 + training** |
|---|---|---|---|---|
| Needle at 16K (50 cases) | ≥ 98% | 0% | 90% | **100%** |
| Needle, all lengths 2K–16K (200 cases) | | 49.5% | 94.5% | **100%** |
| LC-Eval multiple choice, 8–16K-token books (224) | ≥ 60% | 18.3% | 47.8% | **63.8%** |
| ArabicMMLU, 14,455 questions | ≥ 67.6% | 68.6% | 67.2% | **68.8%** (68.0–69.5) |
| Arabic IFEval, every instruction followed (404) | not lower | 40.8% | 42.1% | **40.3%** |

Paired tests against the right comparison:
- LC-Eval vs YaRN alone: 48 questions gained, 12 lost, p < 0.001.
- ArabicMMLU vs as released: 545 gained, 520 lost, p = 0.46: **no loss**. YaRN alone had lost 1.4 points (p < 0.001);
  the replay data won them back.
- IFEval vs as released: 37 gained, 39 lost, p = 0.91: no change.
- LC-Eval claim verification stays at chance (balanced accuracy 51%), as before; reported, not a target (D12).

![Needle](charts/needle_phase2.svg)

## How it was trained

| | |
|---|---|
| Data | 450 long question-answer chats (6–16K tokens, 3 questions each), 100 long Wikipedia articles, 919 short "replay" answers |
| Written by | ALLaM itself (questions about single Wikipedia sections; replay answers from the original model). No paid API. |
| Tokens | 6.56 M seen, 1.31 M trained on (the rest is context the model reads but is not graded on) |
| Settings | QLoRA rank 16 on all linear layers, lr 1e-4 (cosine), 8 examples per step, 183 steps, YaRN ×4 |
| Time and memory | 2 h 19 min, peak 12.0 GB |
| Memory tricks | gradient checkpointing; output layer and loss computed 1,024 positions at a time and only where graded; activations parked in system RAM for examples over 8K tokens |

Dev set (40 chats about held-out Wikipedia articles, 120 questions):

| | long score | open-question F1 | multiple choice | replay loss |
|---|---|---|---|---|
| Base + YaRN (start) | 0.381 | 0.298 | 66.7% | 0.671 |
| checkpoint 40 | 0.718 | 0.657 | 92.6% | 0.449 |
| checkpoint 80 | 0.698 | 0.632 | 92.6% | 0.397 |
| checkpoint 120 | 0.671 | 0.597 | 92.6% | 0.378 |
| checkpoint 160 | 0.711 | 0.648 | 92.6% | 0.366 |
| **checkpoint 183** | **0.730** | **0.672** | 92.6% | **0.364** |

## Release files (local, not yet published)

`C:\WIP\release-work\allam-plus-long_v1\gguf\`: Q8_0 (7.4 GB) and Q4_K_M (4.3 GB), adapter merged into all 224
linear weights. IFEval above was run on the Q8_0 file with YaRN set on the llama-server command line.
