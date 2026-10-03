"""Regenerate Figure 1 of paper/nla_three_families.md from recorded values.

Each value is copied from the committed report named beside it; the paper's
number-trace test also checks this file. Nothing here reruns a model.

    python paper/make_figures.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LIMIT_PP = 5  # frozen gate 2: one-sided 95% upper P2 accuracy loss <= 5 pp

# (family, closed-pilot upper bound, whitespace-stripped upper bound), in pp.
UPPER_BOUNDS = [
    # research/open_weight_lingua/reports/post_hoc_answer_lens.md (Qwen: identical under both)
    ("Qwen2.5-7B", 41.41, 41.41),
    # reports/gemma3_pilot.md (closed) and reports/gemma3_12b_confirmation.md (confirmation run)
    ("Gemma-3-12B", 3.906, 10.16),
    # reports/gemma3_27b_pilot.md (closed) and reports/gemma3_27b_confirmation.md (confirmation run)
    ("Gemma-3-27B", 2.344, 6.25),
]
# What the stripped value is, per family: shown on the bar so status is never color-alone.
STRIPPED_STATUS = {"Qwen2.5-7B": "", "Gemma-3-12B": "confirmation run",
                   "Gemma-3-27B": "confirmation run"}

# Reference palette slots 1 and 2 (validated: CVD dE 24.7, contrast >= 3:1 on #fcfcfb).
FROZEN, LENS = "#2a78d6", "#eb6834"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def main(out: Path = Path(__file__).parent / "figures" / "fig1_preservation_bounds.png"):
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    height, gap = 0.34, 0.03
    families = [row[0] for row in UPPER_BOUNDS]
    for index, (family, frozen, lens) in enumerate(UPPER_BOUNDS):
        y = len(UPPER_BOUNDS) - 1 - index
        status = STRIPPED_STATUS[family]
        for offset, value, color, note in ((height / 2 + gap, frozen, FROZEN, ""),
                                           (-(height / 2 + gap), lens, LENS, status)):
            ax.barh(y + offset, value, height=height, color=color, edgecolor=SURFACE,
                    linewidth=1.0, zorder=3)
            label = f"{value:.2f}" + (f"  ({note})" if note else "")
            # Knockout background so the limit line passes behind the label, not through it.
            ax.text(value + 0.6, y + offset, label, va="center", ha="left",
                    color=INK, fontsize=9, zorder=6,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})

    ax.axvline(LIMIT_PP, color=INK, linewidth=1.2, linestyle=(0, (4, 3)), zorder=5)
    ax.text(LIMIT_PP + 0.6, len(UPPER_BOUNDS) - 0.45, "frozen limit: 5 pp", color=INK,
            fontsize=9, va="bottom")

    ax.set_yticks(range(len(UPPER_BOUNDS)))
    ax.set_yticklabels(reversed(families), color=INK)
    ax.set_xlim(0, 47)
    ax.set_ylim(-0.6, len(UPPER_BOUNDS) - 0.25)
    ax.set_xlabel("One-sided 95% upper bound on P2 accuracy loss vs P0 (percentage points)",
                  color=MUTED)
    ax.tick_params(axis="x", colors=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    handles = [plt.Rectangle((0, 0), 1, 1, color=FROZEN), plt.Rectangle((0, 0), 1, 1, color=LENS)]
    ax.legend(handles, ["Closed pilots' metric (no stripping)", "Trailing whitespace stripped"],
              loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
