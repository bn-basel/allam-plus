"""Phase 4: make the release GGUF files self-contained: the tool chat template and the 16K (YaRN ×4) setting inside.

    uv run python scripts/finalize_gguf.py --src C:\\WIP\\release-work\\allam-plus-tools_v1\\gguf --name allam-plus-7b

export_gguf.py converts from the original model folder, so its files carry ALLaM's original chat template (no tools)
and a 4,096-token context with no position scaling; the tests so far set YaRN on the llama-server command line. This
copies each GGUF (the weights byte for byte) with new metadata:
- tokenizer.chat_template = release/chat_template.jinja (scripts/make_template.py);
- llama.context_length = 16384, rope scaling yarn, factor 4, original context 4096: the same YaRN ×4 the model was
  trained with (scripts/common.py ROPE), so llama.cpp, LM Studio and Ollama use 16K with no extra flags;
- tokenizer.ggml.add_space_prefix = false: llama.cpp's tokenizer otherwise puts an extra space token after every <s>
  in the prompt (" [INST]" becomes "▁" "▁[" instead of "▁["), which ALLaM's own tokenizer never does, so through the
  chat API the model would see slightly different tokens from training (scripts/check_gguf_template.py compares them);
- general.name.
The attention factor is left to llama.cpp, which derives the same 1 + 0.1 ln 4 that transformers uses.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with the GGUF files from export_gguf.py")
    ap.add_argument("--name", required=True, help="release name; files become <name>-<quant>.gguf")
    ap.add_argument("--out", help="output folder (default: <src>/../release)")
    ap.add_argument("--llama", default=r"C:\WIP\tools\llama.cpp", help="llama.cpp folder (src/gguf-py)")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.llama) / "src" / "gguf-py"))
    import gguf
    from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata

    template = (ROOT / "release" / "chat_template.jinja").read_text(encoding="utf-8")
    src = Path(args.src)
    out = Path(args.out) if args.out else src.parent / "release"
    out.mkdir(parents=True, exist_ok=True)
    S, U32, F32 = gguf.GGUFValueType.STRING, gguf.GGUFValueType.UINT32, gguf.GGUFValueType.FLOAT32
    made = []
    for f in sorted(src.glob("*.gguf")):
        quant = f.stem.rsplit("-", 1)[1]
        dst = out / f"{args.name}-{quant}.gguf"
        reader = gguf.GGUFReader(f, "r")
        arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
        meta = {
            gguf.Keys.General.NAME: MetadataDetails(S, args.name),
            gguf.Keys.Tokenizer.CHAT_TEMPLATE: MetadataDetails(S, template),
            gguf.Keys.Tokenizer.ADD_PREFIX: MetadataDetails(gguf.GGUFValueType.BOOL, False),
            gguf.Keys.LLM.CONTEXT_LENGTH.format(arch=arch): MetadataDetails(U32, 16384),
            gguf.Keys.Rope.SCALING_TYPE.format(arch=arch): MetadataDetails(S, "yarn"),
            gguf.Keys.Rope.SCALING_FACTOR.format(arch=arch): MetadataDetails(F32, 4.0),
            gguf.Keys.Rope.SCALING_ORIG_CTX_LEN.format(arch=arch): MetadataDetails(U32, 4096),
        }
        writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
        alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
        if alignment is not None:
            writer.data_alignment = alignment.contents()
        print(f"{f.name} -> {dst}", flush=True)
        copy_with_new_metadata(reader, writer, meta, [])
        made.append({"file": dst.name, "from": f.name, "bytes": dst.stat().st_size})
    (out / "manifest.json").write_text(json.dumps({"name": args.name, "files": made, "context": 16384,
                                                   "rope": "yarn x4 (original 4096)",
                                                   "chat_template": "release/chat_template.jinja"}, indent=1),
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
