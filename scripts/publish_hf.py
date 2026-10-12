"""Publish the release model to Hugging Face: GGUF files, the LoRA adapter, the model card and the serving files.

    uv run python scripts/publish_hf.py --run tools_v2 --adapter runs/tools_v2/checkpoint-180 --dry-run
    uv run python scripts/publish_hf.py --run tools_v2 --adapter runs/tools_v2/checkpoint-180

Builds a staging folder (C:\\WIP\\release-work\\hf-allam-plus-7b) with hard links to the release GGUFs (no 12 GB copy;
same disk) and copies of the small files, prints what will be uploaded, then uploads it to bnbasel/allam-plus-7b with
the login already stored by `hf auth login` (this script never reads or prints a token).

    README.md                      release/README.md (the model card)
    allam-plus-7b-Q4_K_M.gguf      release GGUF (template and 16K inside)
    allam-plus-7b-Q8_0.gguf
    adapter/                       the LoRA adapter (adapter_config.json, adapter_model.safetensors), for transformers
    chat_template.jinja            the same template as inside the GGUFs
    Modelfile, Modelfile.go-template
"""

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "bnbasel/allam-plus-7b"
STAGE = Path(r"C:\WIP\release-work\hf-allam-plus-7b")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the release run, e.g. tools_v2")
    ap.add_argument("--adapter", required=True, help="its checkpoint folder")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rel = Path(rf"C:\WIP\release-work\allam-plus-{args.run}\release")
    files = {
        "README.md": ROOT / "release" / "README.md",
        "chat_template.jinja": ROOT / "release" / "chat_template.jinja",
        "Modelfile": ROOT / "release" / "Modelfile",
        "Modelfile.go-template": ROOT / "release" / "Modelfile.go-template",
        "adapter/adapter_config.json": ROOT / args.adapter / "adapter_config.json",
        "adapter/adapter_model.safetensors": ROOT / args.adapter / "adapter_model.safetensors",
    }
    big = {f"allam-plus-7b-{q}.gguf": rel / f"allam-plus-7b-{q}.gguf" for q in ("Q4_K_M", "Q8_0")}
    missing = [str(p) for p in {**files, **big}.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    STAGE.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        dst = STAGE / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for name, src in big.items():
        dst = STAGE / name
        if not dst.exists():
            os.link(src, dst)  # a second name for the same file: no copy, no extra disk
    total = 0
    for p in sorted(STAGE.rglob("*")):
        if p.is_file():
            total += p.stat().st_size
            print(f"{p.stat().st_size / 1e9:8.2f} GB  {p.relative_to(STAGE).as_posix()}")
    print(f"{total / 1e9:.1f} GB -> https://huggingface.co/{args.repo}")
    if args.dry_run:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=args.repo, folder_path=str(STAGE), commit_message=f"allam-plus-7b ({args.run})")
    print(f"published: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
