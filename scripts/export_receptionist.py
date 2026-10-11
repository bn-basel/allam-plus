"""Export the receptionist training conversations from the earlier project, as plain records, for Phase 3.

    uv run python scripts/export_receptionist.py      # -> data/tools/receptionist.jsonl

Exactly the conversations the receptionist ALLaM (saudi-dialect runs/allam_v1) was trained and validated on (its
data.json lists them), with the same short system prompt per business and the same training copy of the tools, taken
through that project's own code. They are re-rendered later with this project's Arabic tool guide
(scripts/toolformat.py). The frozen test sets are not read here (only the earlier project's own leakage check reads
them, as it did then).
"""

import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SD = HERE.parent.parent / "saudi-dialect"
sys.path.insert(0, str(SD / "scripts"))

from datagen import train_tools  # noqa: E402
from prompt import build_system_prompt, load_spec  # noqa: E402
from train import GOLD_TOKENIZER, gold_conversations  # noqa: E402


def main() -> None:
    from transformers import AutoTokenizer

    ref = json.loads((SD / "runs" / "allam_v1" / "data.json").read_text(encoding="utf-8"))
    split = {i.split("#")[0]: "train" for i in ref["train"]} | {i.split("#")[0]: "dev" for i in ref["validation"]}
    businesses = {b["id"]: b for b in json.loads((SD / "spec" / "train_businesses.json").read_text(encoding="utf-8"))}
    _, styles, tools = load_spec()
    tools = train_tools(tools)
    out = HERE.parent / "data" / "tools" / "receptionist.jsonl"
    n = 0
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for rec in gold_conversations(AutoTokenizer.from_pretrained(GOLD_TOKENIZER)):
            base = f"{rec['set']}/{rec['id']}"
            if base not in split:
                continue
            meta = rec["meta"]
            system = build_system_prompt(businesses[meta["business_id"]], styles, meta["style"],
                                         datetime.fromisoformat(meta["now"]), meta["prayer_times"], "short")
            order = [next(t for t in tools if t["function"]["name"] == nm) for nm in meta["tools"]]
            f.write(json.dumps({"id": base, "split": split[base], "system": system, "messages": rec["messages"],
                                "tools": order}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} conversations (of {len(split)} in allam_v1) -> {out}")


if __name__ == "__main__":
    main()
