"""Charts for the README and LEARNING.md, drawn as plain SVG from the saved result files (no plotting library).

    uv run python scripts/make_charts.py      # writes docs/charts/*.svg

Every number comes from results/**/summary.json, so a chart can be redrawn whenever a result changes and never drifts
from the files it describes. Each chart carries a light and a dark set of colours (a CSS media query inside the SVG).
Same drawing helpers and colours as the receptionist project's charts.
"""

import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "charts"
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

# colour name: (light theme, dark theme). Shapes take class f-<name> (fill) or s-<name> (stroke).
PALETTE = {
    "bg": ("#ffffff", "#0d1117"), "ink": ("#1f2328", "#e6edf3"), "muted": ("#656d76", "#8d96a0"),
    "grid": ("#eaeef2", "#21262d"), "axis": ("#afb8c1", "#484f58"), "white": ("#ffffff", "#ffffff"),
    "base": ("#8c959f", "#6e7681"), "yarn": ("#0e7c86", "#2bb3c0"), "ours": ("#1a7f37", "#3fb950"),
    "rec": ("#bf8700", "#d29922"),
    # heatmap: all found, 90%+, 75%+, 50%+, some, none
    "h5": ("#1a7f37", "#238636"), "h4": ("#9bd5a9", "#1f5f30"), "h3": ("#f2d27a", "#6b5110"),
    "h2": ("#f3ac7a", "#7c3b12"), "h1": ("#f2a6a0", "#6e211d"), "h0": ("#cf222e", "#b62324"),
}


def load(path: str) -> dict | None:
    p = ROOT / "results" / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def style() -> str:
    light = "".join(f".f-{k}{{fill:{l}}}.s-{k}{{stroke:{l}}}" for k, (l, _) in PALETTE.items())
    dark = "".join(f".f-{k}{{fill:{d}}}.s-{k}{{stroke:{d}}}" for k, (_, d) in PALETTE.items())
    return f"<style>text{{font-family:{FONT}}}{light}@media (prefers-color-scheme:dark){{{dark}}}</style>"


def save(name: str, title: str, width: float, height: float, body: list[str]) -> None:
    # a rounded card with its own background, so the chart reads well on any page colour
    (OUT / f"{name}.svg").write_text(
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width:.0f}' height='{height:.0f}' "
        f"viewBox='0 0 {width:.0f} {height:.0f}' role='img'><title>{escape(title)}</title>{style()}"
        f"<rect x='0.5' y='0.5' width='{width - 1:.0f}' height='{height - 1:.0f}' rx='12' class='f-bg s-grid'/>"
        + "".join(body) + "</svg>\n", encoding="utf-8", newline="\n")


def text(x, y, s, size=13, anchor="start", colour="ink", weight="normal") -> str:
    return (f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' text-anchor='{anchor}' class='f-{colour}' "
            f"font-weight='{weight}'>{escape(str(s))}</text>")


def rect(x, y, w, h, colour, rx=0, opacity=1.0) -> str:
    op = f" fill-opacity='{opacity}'" if opacity < 1 else ""
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{max(w, 0):.1f}' height='{h:.1f}' rx='{rx}' class='f-{colour}'{op}/>"


def line(x1, y1, x2, y2, colour, width=1.0, dash="") -> str:
    d = f" stroke-dasharray='{dash}'" if dash else ""
    return (f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' class='s-{colour}' "
            f"stroke-width='{width}'{d}/>")


def circle(x, y, r, colour, hollow=False) -> str:
    if hollow:
        return f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' class='f-bg s-{colour}' stroke-width='2.2'/>"
    return f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' class='f-{colour}'/>"


def header(title: str, subtitle: str) -> list[str]:
    return [text(24, 36, title, 19, weight="bold"), text(24, 58, subtitle, 12.5, colour="muted")]




def heat_colour(v: float) -> str:
    return "h5" if v >= 1 else "h4" if v >= .9 else "h3" if v >= .75 else "h2" if v >= .5 else "h1" if v > 0 else "h0"


NEEDLE_PANELS = {"shipped": "As released (trained on 4K)", "yarn4": "YaRN ×4, config only",
                 "long_v1": "YaRN ×4 + long-context training"}


def needle(name: str = "needle_phase1", keys: tuple = ("shipped", "yarn4"), cell_w: int = 74,
           title: str = "Where ALLaM-7B stops finding the needle") -> None:
    """Heatmaps side by side, one per model version."""
    s = load("phase1/needle/summary.json")
    if not s or not set(keys) <= set(s):
        return
    lengths, depths = ["2048", "4096", "8192", "16384"], [10, 30, 50, 70, 90]
    cell_h, gap, left, top = 34, 40 if len(keys) > 2 else 80, 96, 118
    panel_w = cell_w * len(lengths)
    body = header(title, "Arabic needle-in-a-haystack: 10 invented facts hidden at 5 depths in Arabic book text. "
                         "Share found per cell (10 needles).")
    for p, key in enumerate(keys):
        x0 = left + p * (panel_w + gap)
        body.append(text(x0 + panel_w / 2, top - 30, NEEDLE_PANELS[key], 13 if len(keys) > 2 else 13.5, "middle",
                         weight="bold"))
        for j, L in enumerate(lengths):
            body.append(text(x0 + j * cell_w + cell_w / 2, top - 10, f"{int(L) // 1024}K", 12, "middle", "muted"))
        for i, d in enumerate(depths):
            if p == 0:
                body.append(text(left - 12, top + i * cell_h + cell_h / 2 + 4, f"{d}%", 12, "end", "muted"))
            for j, L in enumerate(lengths):
                v = s[key]["grid"].get(f"{L}|{d}")
                if v is None:
                    continue
                x, y = x0 + j * cell_w, top + i * cell_h
                body += [rect(x + 2, y + 2, cell_w - 4, cell_h - 4, heat_colour(v), 4),
                         text(x + cell_w / 2, y + cell_h / 2 + 5, f"{v:.0%}", 12.5, "middle",
                              "white" if v >= 1 or v == 0 else "ink", "bold")]
        totals = s[key]["by_length"]
        for j, L in enumerate(lengths):
            if L in totals:
                body.append(text(x0 + j * cell_w + cell_w / 2, top + len(depths) * cell_h + 20,
                                 f"{totals[L]['acc']:.0%}", 13, "middle", weight="bold"))
    body.append(text(left - 12, top + len(depths) * cell_h + 20, "all", 12, "end", "muted"))
    body.append(text(left - 12, top - 10, "depth", 12, "end", "muted"))
    h = top + len(depths) * cell_h + 58
    controls = ", ".join(f"{NEEDLE_PANELS[k].split(' (')[0].split(',')[0]} {s[k]['controls_answered']}" for k in keys)
    body.append(text(24, h - 8, f"Controls (no needle in the text) answered: {controls}.", 11.5, colour="muted"))
    save(name, title, left + len(keys) * panel_w + (len(keys) - 1) * gap + 30, h + 6, body)


BASELINE_ROWS = [  # label, note, how to read the score from the result files
    ("ArabicMMLU", "knowledge, 14,455 questions",
     {"shipped": ("arabicmmlu/shipped/summary.json", "acc"), "yarn4": ("arabicmmlu/yarn4/summary.json", "acc"),
      "rec": ("arabicmmlu/receptionist/summary.json", "acc")}),
    ("Arabic IFEval", "every instruction followed, 404 prompts",
     {"shipped": ("ifeval/base_q8/summary.json", "prompt_strict"), "yarn4": ("ifeval/base_q8_yarn4/summary.json", "prompt_strict"),
      "rec": ("ifeval/receptionist_q8/summary.json", "prompt_strict")}),
    ("LC-Eval, multiple choice", "8-16K-token Arabic books, 224 questions",
     {"shipped": ("lceval/shipped/summary.json", "mcq|all.acc"), "yarn4": ("lceval/yarn4/summary.json", "mcq|all.acc")}),
    ("AISA tools: right tool", "545 requests in 5 dialects",
     {"shipped": ("tools/aisa_dev/base__text/summary.json", "all.fn_acc"),
      "rec": ("tools/aisa_dev/receptionist__text/summary.json", "all.fn_acc")}),
    ("AISA tools: exact arguments", "right tool and every argument right",
     {"shipped": ("tools/aisa_dev/base__text/summary.json", "all.arg_em"),
      "rec": ("tools/aisa_dev/receptionist__text/summary.json", "all.arg_em")}),
    ("AISA tools: rightly no call", "45 requests no tool fits",
     {"shipped": ("tools/aisa_dev/base__text/summary.json", "all.no_call_acc"),
      "rec": ("tools/aisa_dev/receptionist__text/summary.json", "all.no_call_acc")}),
]
SERIES = [("shipped", "ALLaM-7B-Instruct as released", "base"), ("yarn4", "+ YaRN ×4 in the config", "yarn"),
          ("rec", "receptionist fine-tune (earlier project)", "rec")]


def value(path: str, key: str) -> float | None:
    s = load(path)
    for k in key.split("."):
        if s is None:
            return None
        s = s.get(k)
    return s


def baselines() -> None:
    """Every Phase 1 score, grouped by test: base as released, base + YaRN, and the receptionist model."""
    left, top, bar_w, bar_h, w = 300, 120, 420, 13, 860
    rows = []
    for label, note, src in BASELINE_ROWS:
        vals = [(key, value(*src[key])) for key, _, _ in SERIES if key in src]
        rows.append((label, note, [(k, v) for k, v in vals if v is not None]))
    body = header("ALLaM-7B before any training: the baselines",
                  "Phase 1 of allam-plus. Each group: one test. Higher is better.")
    x = 24
    for key, name, colour in SERIES:  # legend
        body += [rect(x, 76, 12, 12, colour, 3), text(x + 18, 86, name, 12)]
        x += 18 + len(name) * 6.6 + 26
    y = top
    for label, note, vals in rows:
        group_h = len(vals) * (bar_h + 4)
        body += [text(left - 14, y + group_h / 2, label, 13.5, "end", weight="bold"),
                 text(left - 14, y + group_h / 2 + 15, note, 11, "end", "muted")]
        for key, v in vals:
            colour = next(c for k, _, c in SERIES if k == key)
            body += [rect(left, y, bar_w, bar_h, "grid", 3), rect(left, y, bar_w * v, bar_h, colour, 3),
                     text(left + bar_w * v + 6, y + bar_h - 2, f"{v:.1%}", 11.5)]
            y += bar_h + 4
        y += 22
    for p in range(0, 101, 25):
        xx = left + bar_w * p / 100
        body.append(text(xx, y + 4, f"{p}%", 10.5, "middle", "muted"))
    save("baselines_phase1", "Phase 1 baselines", w, y + 20, body)


def memory() -> None:
    """GPU memory of the 4-bit build (Q4_K_M) in llama.cpp by context length and KV-cache type, with the card's 16 GB."""
    rows = load("speed/allam-7b-instruct-base-Q4_K_M.json")
    if not rows:
        return
    ctxs = sorted({r["ctx"] for r in rows})
    kinds = [("f16", "KV cache 16-bit", "base"), ("q8_0", "KV cache 8-bit", "yarn"), ("q4_0", "KV cache 4-bit", "ours")]
    left, top, cw, ch, w = 80, 116, 700, 260, 860
    gb_max = 18.0
    X = lambda i: left + cw * i / (len(ctxs) - 1)  # noqa: E731
    Y = lambda gb: top + ch - ch * gb / gb_max  # noqa: E731
    body = header("Long context on a 16 GB card: the KV cache decides",
                  "ALLaM-7B, 4-bit weights (Q4_K_M), llama.cpp. GPU memory with the whole context reserved; "
                  "labels: generation speed.")
    for gb in range(0, 19, 4):
        body += [line(left, Y(gb), left + cw, Y(gb), "grid"), text(left - 10, Y(gb) + 4, f"{gb} GB", 11, "end", "muted")]
    body += [line(left, Y(16.3), left + cw, Y(16.3), "h0", 1.5, "6 4"),
             text(left + cw - 4, Y(16.3) - 6, "RTX 5080: 16 GB", 11.5, "end", "h0", "bold")]
    for i, c in enumerate(ctxs):
        body.append(text(X(i), top + ch + 20, f"{c // 1024}K", 12, "middle", "muted"))
    lx = 24
    for n, (kv, name, colour) in enumerate(kinds):
        pts = [(i, next((r for r in rows if r["ctx"] == c and r["kv"] == kv), None)) for i, c in enumerate(ctxs)]
        pts = [(i, r) for i, r in pts if r and r.get("gpu_mib")]
        for (i0, r0), (i1, r1) in zip(pts, pts[1:]):
            body.append(line(X(i0), Y(r0["gpu_mib"] / 1024), X(i1), Y(r1["gpu_mib"] / 1024), colour, 2.5))
        for i, r in pts:
            spill = r["gen_tok_s"] is not None and r["gen_tok_s"] < 20
            body.append(circle(X(i), Y(r["gpu_mib"] / 1024), 4.5, "h0" if spill else colour, hollow=spill))
            if ctxs[i] >= 16384:  # shorter contexts sit on top of each other; their speeds are in the results file
                last = ctxs[i] == ctxs[-1] and not spill  # right of the point, where no line runs
                dy = 4 if last else 18 if kv == "q4_0" else -9
                body.append(text(X(i) + (10 if last else -8), Y(r["gpu_mib"] / 1024) + dy,
                                 f"{r['gen_tok_s']:.0f} tok/s" + (", spills to RAM" if spill else ""), 10.5,
                                 "start" if last else "end", colour="h0" if spill else "muted"))
        body += [rect(lx, 76, 12, 12, colour, 3), text(lx + 18, 86, name, 12)]
        lx += 18 + len(name) * 6.6 + 26
    save("memory_phase1", "GPU memory by context and KV cache", w, top + ch + 40, body)


PHASE2_ROWS = [  # label, note, {series: (file, key)}
    ("Needle at 16K tokens", "10 hidden facts × 5 depths",
     {"shipped": ("phase1/needle/summary.json", "shipped.by_length.16384.acc"),
      "yarn4": ("phase1/needle/summary.json", "yarn4.by_length.16384.acc"),
      "long": ("phase1/needle/summary.json", "long_v1.by_length.16384.acc")}),
    ("LC-Eval, multiple choice", "HUMAIN's long-book test, 8-16K tokens",
     {"shipped": ("lceval/shipped/summary.json", "mcq|all.acc"), "yarn4": ("lceval/yarn4/summary.json", "mcq|all.acc"),
      "long": ("lceval/long_v1/summary.json", "mcq|all.acc")}),
    ("ArabicMMLU", "knowledge, short questions",
     {"shipped": ("arabicmmlu/shipped/summary.json", "acc"), "yarn4": ("arabicmmlu/yarn4/summary.json", "acc"),
      "long": ("arabicmmlu/long_v1/summary.json", "acc")}),
    ("Arabic IFEval", "every instruction followed",
     {"shipped": ("ifeval/base_q8/summary.json", "prompt_strict"),
      "yarn4": ("ifeval/base_q8_yarn4/summary.json", "prompt_strict"),
      "long": ("ifeval/long_v1_q8_yarn4/summary.json", "prompt_strict")}),
]
PHASE2_SERIES = [("shipped", "As released (4K)", "base"), ("yarn4", "+ YaRN ×4, no training", "yarn"),
                 ("long", "+ YaRN ×4 + long-context training (allam-plus)", "ours")]


def phase2() -> None:
    """Before/after for the long-context stage: long reading up, short-text quality unchanged."""
    left, top, bar_w, bar_h, w = 300, 120, 420, 14, 860
    body = header("ALLaM-7B reads 16K tokens, with no loss on short texts",
                  "Long-context stage (Phase 2): QLoRA on one RTX 5080. Higher is better.")
    x = 24
    for _, name, colour in PHASE2_SERIES:
        body += [rect(x, 76, 12, 12, colour, 3), text(x + 18, 86, name, 12)]
        x += 18 + len(name) * 6.4 + 24
    y = top
    for label, note, src in PHASE2_ROWS:
        vals = [(c, value(*src[k])) for k, _, c in PHASE2_SERIES if k in src]
        vals = [(c, v) for c, v in vals if v is not None]
        gh = len(vals) * (bar_h + 4)
        body += [text(left - 14, y + gh / 2, label, 13.5, "end", weight="bold"),
                 text(left - 14, y + gh / 2 + 15, note, 11, "end", "muted")]
        for colour, v in vals:
            body += [rect(left, y, bar_w, bar_h, "grid", 3), rect(left, y, max(bar_w * v, 2), bar_h, colour, 3),
                     text(left + bar_w * v + 6, y + bar_h - 2, f"{v:.1%}", 11.5,
                          weight="bold" if colour == "ours" else "normal")]
            y += bar_h + 4
        y += 22
    for p in range(0, 101, 25):
        body.append(text(left + bar_w * p / 100, y + 4, f"{p}%", 10.5, "middle", "muted"))
    save("phase2_long_context", "Phase 2: long context before and after", w, y + 20, body)


PHASE3_ROWS = [  # label, note, {series: (file, key)}
    ("ArabFuncBench: right tool", "1,000 MSA requests, 50 unseen tools",
     {"shipped": ("tools/arabfuncbench/base__native/summary.json", "all.fn_acc"),
      "long": ("tools/arabfuncbench/long_v1__native__yarn4/summary.json", "all.fn_acc"),
      "tools": ("tools/arabfuncbench/tools_v1__native__yarn4/summary.json", "all.fn_acc")}),
    ("ArabFuncBench: exact arguments", "right tool and every argument right",
     {"shipped": ("tools/arabfuncbench/base__native/summary.json", "all.arg_em"),
      "long": ("tools/arabfuncbench/long_v1__native__yarn4/summary.json", "all.arg_em"),
      "tools": ("tools/arabfuncbench/tools_v1__native__yarn4/summary.json", "all.arg_em")}),
    ("AISA: right tool", "545 requests in five dialects",
     {"shipped": ("tools/aisa_dev/base__native/summary.json", "all.fn_acc"),
      "long": ("tools/aisa_dev/long_v1__native__yarn4/summary.json", "all.fn_acc"),
      "tools": ("tools/aisa_dev/tools_v1__native__yarn4/summary.json", "all.fn_acc")}),
    ("AISA: exact arguments", "right tool and every argument right",
     {"shipped": ("tools/aisa_dev/base__native/summary.json", "all.arg_em"),
      "long": ("tools/aisa_dev/long_v1__native__yarn4/summary.json", "all.arg_em"),
      "tools": ("tools/aisa_dev/tools_v1__native__yarn4/summary.json", "all.arg_em")}),
    ("Receptionist test v2", "160 Saudi-dialect booking chats",
     {"shipped": ("receptionist/base_native__test_v2/summary.json", "pass_rate"),
      "long": ("receptionist/long_v1_native__test_v2/summary.json", "pass_rate"),
      "tools": ("receptionist/tools_v1_native__test_v2/summary.json", "pass_rate")}),
]
PHASE3_SERIES = [("shipped", "As released", "base"), ("long", "After Phase 2 (16K)", "yarn"),
                 ("tools", "After Phase 3 (16K + tools): allam-plus", "ours")]


def phase3() -> None:
    """Tool calling before and after, every row in the same Arabic tool format."""
    left, top, bar_w, bar_h, w = 300, 120, 420, 14, 860
    title = "ALLaM-7B learns Arabic tool calling"
    body = header(title, "Phase 3: QLoRA on our Arabic tool conversations + the receptionist data. "
                         "Same Arabic tool format in every row.")
    x = 24
    for _, name, colour in PHASE3_SERIES:
        body += [rect(x, 76, 12, 12, colour, 3), text(x + 18, 86, name, 12)]
        x += 18 + len(name) * 6.4 + 24
    y = top
    for label, note, src in PHASE3_ROWS:
        vals = [(c, value(*src[k])) for k, _, c in PHASE3_SERIES if k in src]
        vals = [(c, v) for c, v in vals if v is not None]
        gh = len(vals) * (bar_h + 4)
        body += [text(left - 14, y + gh / 2, label, 13.5, "end", weight="bold"),
                 text(left - 14, y + gh / 2 + 15, note, 11, "end", "muted")]
        for colour, v in vals:
            body += [rect(left, y, bar_w, bar_h, "grid", 3), rect(left, y, max(bar_w * v, 2), bar_h, colour, 3),
                     text(left + bar_w * v + 6, y + bar_h - 2, f"{v:.1%}", 11.5,
                          weight="bold" if colour == "ours" else "normal")]
            y += bar_h + 4
        y += 22
    for p in range(0, 101, 25):
        body.append(text(left + bar_w * p / 100, y + 4, f"{p}%", 10.5, "middle", "muted"))
    save("phase3_tools", "Phase 3: tool calling before and after", w, y + 20, body)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    phase3()
    needle()
    needle("needle_phase2", ("shipped", "yarn4", "long_v1"), 60,
           "ALLaM-7B finds the needle anywhere in 16K tokens after training")
    baselines()
    memory()
    phase2()
    print(", ".join(sorted(p.name for p in OUT.glob("*.svg"))))


if __name__ == "__main__":
    main()
