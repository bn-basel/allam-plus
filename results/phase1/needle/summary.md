# Arabic needle-in-a-haystack (v1)

Share of needles found (10 needles × 5 depths per length). Controls: no needle in the text; a 'found' control means the model guessed the answer.

## rope = long_v1

| depth \ length | 2K | 4K | 8K | 16K |
|---|---|---|---|---|
| 10% | 100% | 100% | 100% | 100% |
| 30% | 100% | 100% | 100% | 100% |
| 50% | 100% | 100% | 100% | 100% |
| 70% | 100% | 100% | 100% | 100% |
| 90% | 100% | 100% | 100% | 100% |
| **all** | **100%** (50/50) | **100%** (50/50) | **100%** (50/50) | **100%** (50/50) |

Controls answered: 0/10. Peak GPU memory 13.99 GB. Seconds per case by length: {'2048': 0.9, '4096': 1.5, '8192': 2.8, '16384': 36.2}.

## rope = shipped

| depth \ length | 2K | 4K | 8K | 16K |
|---|---|---|---|---|
| 10% | 100% | 100% | 0% | 0% |
| 30% | 100% | 100% | 0% | 0% |
| 50% | 100% | 100% | 0% | 0% |
| 70% | 100% | 100% | 0% | 0% |
| 90% | 100% | 90% | 0% | 0% |
| **all** | **100%** (50/50) | **98%** (49/50) | **0%** (0/50) | **0%** (0/50) |

Controls answered: 0/10. Peak GPU memory 13.5 GB. Seconds per case by length: {'2048': 0.7, '4096': 1.0, '8192': 3.2, '16384': 7.9}.

## rope = yarn4

| depth \ length | 2K | 4K | 8K | 16K |
|---|---|---|---|---|
| 10% | 100% | 100% | 100% | 100% |
| 30% | 100% | 100% | 90% | 80% |
| 50% | 100% | 100% | 70% | 90% |
| 70% | 100% | 100% | 100% | 90% |
| 90% | 90% | 100% | 90% | 90% |
| **all** | **98%** (49/50) | **100%** (50/50) | **90%** (45/50) | **90%** (45/50) |

Controls answered: 0/10. Peak GPU memory 13.5 GB. Seconds per case by length: {'2048': 0.7, '4096': 1.0, '8192': 2.3, '16384': 6.4}.
