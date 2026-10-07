"""Speed and GPU memory of a GGUF build in llama.cpp, at several context lengths and KV-cache types.

    uv run python scripts/speed_memory.py --gguf C:/WIP/release-work/allam-7b-instruct-base/gguf/allam-7b-instruct-base-Q4_K_M.gguf

The KV cache is the model's notes on every token it has read (one key and one value vector per token, per layer, per
head), kept so each new token doesn't re-read the whole text. For ALLaM it is 0.5 MiB per token in 16-bit, so a 32K
context needs 16 GiB, more than the card. Storing it in 8 bits (q8_0) halves that, 4 bits (q4_0) quarters it, at some
cost in accuracy (measured elsewhere, with the needle test).

Memory: llama-server is started with the full context allocated (-c N, YaRN on, flash attention on, which llama.cpp
needs for a quantised V cache) and the GPU memory in use is read with nvidia-smi once it is ready; llama.cpp reserves
the KV cache and compute buffers at start, so this is the working size. Speed: llama-bench, prompt processing
(prefill) of N tokens and generation of 128 tokens after N tokens of context.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402

LLAMA = Path(r"C:\WIP\tools\llama.cpp\bin")
YARN = ["--rope-scaling", "yarn", "--rope-scale", "4", "--yarn-orig-ctx", "4096"]


def gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def server_memory(gguf: str, ctx: int, kv: str, port: int = 8093) -> dict:
    base = gpu_used_mib()
    p = subprocess.Popen([str(LLAMA / "llama-server.exe"), "-m", gguf, "-ngl", "99", "-c", str(ctx), "-fa", "on",
                          "-ctk", kv, "-ctv", kv, "--port", str(port), "-np", "1", *YARN],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(240):
            time.sleep(1)
            if p.poll() is not None:
                return {"error": f"server exited ({p.returncode})"}
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
        time.sleep(2)
        return {"gpu_mib": gpu_used_mib() - base}
    finally:
        p.terminate()
        p.wait(timeout=60)
        time.sleep(3)


def bench(gguf: str, ctx: int, kv: str) -> dict:
    cmd = [str(LLAMA / "llama-bench.exe"), "-m", gguf, "-ngl", "99", "-fa", "on", "-ctk", kv, "-ctv", kv,
           "-p", str(ctx), "-n", "0", "-r", "2", "-o", "json"]
    pp = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout or "[]")
    cmd = [str(LLAMA / "llama-bench.exe"), "-m", gguf, "-ngl", "99", "-fa", "on", "-ctk", kv, "-ctv", kv,
           "-p", "0", "-n", "128", "-d", str(ctx), "-r", "2", "-o", "json"]
    tg = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout or "[]")
    return {"prefill_tok_s": round(pp[0]["avg_ts"], 1) if pp else None,
            "gen_tok_s": round(tg[0]["avg_ts"], 1) if tg else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ctx", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--kv", nargs="+", default=["f16", "q8_0", "q4_0"])
    ap.add_argument("--name")
    args = ap.parse_args()
    name = args.name or Path(args.gguf).stem
    out = ROOT / "results" / "speed" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {(r["ctx"], r["kv"]) for r in rows}
    for kv in args.kv:
        for ctx in args.ctx:
            if (ctx, kv) in done:
                continue
            r = {"gguf": Path(args.gguf).name, "ctx": ctx, "kv": kv, **server_memory(args.gguf, ctx, kv)}
            if "error" not in r:
                r |= bench(args.gguf, ctx, kv)
            print(json.dumps(r), flush=True)
            rows.append(r)
            out.write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
