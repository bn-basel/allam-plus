"""Release: build GGUF files (llama.cpp, LM Studio, Ollama) from a base model plus a trained LoRA adapter.

    uv run python scripts/export_gguf.py --model allam --name allam-7b-instruct-base          # the base, no adapter
    uv run python scripts/export_gguf.py --model allam --adapter runs/<run>/checkpoint-<n> --name <name>

Steps:
1. Convert the base model to an 8-bit GGUF (Q8_0) with llama.cpp's converter, merging the adapter into each weight as
   it is read: W' = W + (alpha / r) * B @ A, computed in float32 and stored in the base model's 16-bit type. No merged
   copy is ever written to disk, so the peak disk use is the GGUF files themselves (a first version wrote a merged copy
   and ran out of disk on Gemma).
2. Make a 4-bit GGUF (Q4_K_M) from the 8-bit one. Q8_0 is close to lossless, so this is very close to quantizing from
   16 bits.

Both files are re-tested on the benchmark before any score is quoted (scripts/run_eval.py with a llamacpp: model) and
their built-in chat template is checked against the training prompt (scripts/check_gguf_template.py).

The adapters were trained against the 4-bit (NF4) base used everywhere in this project, and are merged here into the
16-bit original, the usual way QLoRA adapters are released; the re-test shows what that costs.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPOS = {"gemma": "google/gemma-4-12B-it", "allam": "humain-ai/ALLaM-7B-Instruct-preview"}


def load_lora(adapter_dir: Path) -> tuple[dict, float]:
    """{base weight name: (A, B)} and the scale alpha / r."""
    from safetensors import safe_open
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    parts: dict[str, dict] = {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            module, _, part = k.removeprefix("base_model.model.").rpartition(".lora_")
            parts.setdefault(module + ".weight", {})[part.split(".")[0]] = f.get_tensor(k)
    return {k: (v["A"], v["B"]) for k, v in parts.items()}, cfg["lora_alpha"] / cfg["r"]


def convert_with_lora(base_dir: Path, adapter_dir: Path, outfile: Path, llama_src: Path) -> int:
    """Run llama.cpp's converter on the base model with the adapter merged into each weight as it is loaded."""
    sys.path.insert(0, str(llama_src))
    import convert_hf_to_gguf as conv  # noqa: F401  (registers the model classes)
    from conversion import get_model_class
    from conversion.base import LazyTorchTensor

    lora, scale = load_lora(adapter_dir)
    arch = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))["architectures"][0]
    cls = get_model_class(arch)  # the class the converter will use for the text model
    original = cls.filter_tensors
    seen: set[str] = set()

    def merged(t, name):
        a, b = lora[name]
        return (t.float() + scale * (b.float() @ a.float())).to(t.dtype)

    def filter_tensors(klass, item):  # the converter declares it a classmethod
        name, gen = item
        if name in lora:
            seen.add(name)

            def gen_merged(gen=gen, name=name):
                lazy = gen()
                return LazyTorchTensor(meta=lazy._meta, args=(lazy,), func=lambda t: merged(t, name))
            item = (name, gen_merged)
        return original(item)  # already bound to the class

    cls.filter_tensors = classmethod(filter_tensors)
    sys.argv = ["convert_hf_to_gguf.py", str(base_dir), "--outtype", "q8_0", "--outfile", str(outfile),
                "--model-name", outfile.stem.rsplit("-", 1)[0]]
    conv.main()
    # Gemma's adapter also touched a few audio and vision projections, which a text GGUF leaves out
    missing = [k for k in lora if k not in seen]
    text_missing = [k for k in missing if "audio" not in k and "vision" not in k]
    assert not text_missing, f"{len(text_missing)} adapter weights never reached the converter, e.g. {text_missing[:3]}"
    return len(seen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=REPOS, required=True)
    ap.add_argument("--adapter", help="LoRA adapter folder to merge; leave out to convert the base model as it is")
    ap.add_argument("--name", required=True, help="file name stem of the GGUF files")
    ap.add_argument("--work", default=r"C:\WIP\release-work", help="folder for the GGUF files")
    ap.add_argument("--llama", default=r"C:\WIP\tools\llama.cpp", help="llama.cpp folder (src/ and bin/)")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    base_dir = Path(snapshot_download(REPOS[args.model], local_files_only=True))
    out = Path(args.work) / args.name / "gguf"
    out.mkdir(parents=True, exist_ok=True)
    need = sum(p.stat().st_size for p in base_dir.glob("*.safetensors")) * 0.85 / 2**30
    free = shutil.disk_usage(out).free / 2**30
    print(f"base {base_dir}\nfree {free:.0f} GB, needed about {need:.0f} GB")
    if free < need:
        sys.exit("not enough free disk space")

    t0 = time.time()
    q8, q4 = out / f"{args.name}-Q8_0.gguf", out / f"{args.name}-Q4_K_M.gguf"
    if args.adapter:
        n = convert_with_lora(base_dir, ROOT / args.adapter, q8, Path(args.llama) / "src")
    else:
        n = 0
        subprocess.run([sys.executable, str(Path(args.llama) / "src" / "convert_hf_to_gguf.py"), str(base_dir),
                        "--outtype", "q8_0", "--outfile", str(q8)], check=True)
    print(f"1. {q8.name}: {q8.stat().st_size / 2**30:.1f} GB, adapter merged into {n} weights, {time.time() - t0:.0f} s")
    subprocess.run([str(Path(args.llama) / "bin" / "llama-quantize.exe"), "--allow-requantize", str(q8), str(q4), "Q4_K_M"],
                   check=True)
    print(f"2. {q4.name}: {q4.stat().st_size / 2**30:.1f} GB")
    manifest = {"model": REPOS[args.model], "adapter": args.adapter, "merged_weights": n,
                "files": {p.name: p.stat().st_size for p in (q8, q4)}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"done in {(time.time() - t0) / 60:.0f} min: {out}")


if __name__ == "__main__":
    main()
