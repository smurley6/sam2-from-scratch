"""Render the case-study figures from docs/results/*.json into docs/figures/.

    PYTHONPATH=src python tools/make_figures.py

Figures (PNG, 2x for retina):
    davis_comparison.png   our Hiera-T / Hiera-S J&F next to the paper's published numbers
    ablations.png          memory-bank ablations on DAVIS 2017 val (whichever runs exist)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs" / "results"
OUT = ROOT / "docs" / "figures"

INK = "#141413"
MUTED = "#767676"
ACCENT = "#2e6b48"
PAPER = "#b8b8b8"

# SAM 2 paper (arXiv:2408.00714v2): Table 17(d) for T and S (models trained with internal
# data in addition to SA-1B + SA-V), Table 6 for B+ and L.
PAPER_JF = {"Hiera-T": 89.4, "Hiera-S": 89.6, "Hiera-B+": 90.2, "Hiera-L": 90.7}


def load(tag: str) -> dict | None:
    p = RESULTS / f"davis_{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=10)
    ax.yaxis.grid(True, color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)


def davis_comparison() -> None:
    ours = {"Hiera-T": load("tiny_base"), "Hiera-S": load("small_base")}
    names = list(PAPER_JF)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    style(ax)
    x = range(len(names))
    w = 0.36
    paper_vals = [PAPER_JF[n] for n in names]
    ax.bar(
        [i - w / 2 for i in x],
        paper_vals,
        w,
        color=PAPER,
        label="SAM 2 paper (A100, internal data)",
    )
    for i, n in enumerate(names):
        r = ours.get(n)
        if r:
            ax.bar(
                i + w / 2,
                r["J&F"],
                w,
                color=ACCENT,
                label="This reimplementation (SAM 2.1 weights)" if i == 0 else None,
            )
            ax.text(
                i + w / 2,
                r["J&F"] + 0.25,
                f"{r['J&F']:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )
        ax.text(
            i - w / 2,
            PAPER_JF[n] + 0.25,
            f"{PAPER_JF[n]:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=MUTED,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(80, 93)
    ax.set_ylabel("J&F, DAVIS 2017 val", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "davis_comparison.png")
    plt.close(fig)


def ablations() -> None:
    runs = [
        ("Released settings\n(7 memories, 16 pointers, 1024 px)", "tiny_base"),
        ("512 px input", "tiny_512"),
        ("3 memories", "tiny_mem3"),
        ("1 memory", "tiny_mem1"),
        ("No object pointers", "tiny_noptr"),
    ]
    rows = [(label, load(tag)) for label, tag in runs]
    rows = [(label, r) for label, r in rows if r]
    if len(rows) < 2:
        print("ablations: fewer than two runs, skipped")
        return
    fig, ax = plt.subplots(figsize=(7.2, 0.7 * len(rows) + 1.2), dpi=200)
    style(ax)
    ax.xaxis.grid(True, color="#e8e8e8", linewidth=0.8)
    ax.yaxis.grid(False)
    labels = [label for label, _ in rows]
    vals = [r["J&F"] for _, r in rows]
    colors = [ACCENT if i == 0 else PAPER for i in range(len(rows))]
    y = list(range(len(rows)))[::-1]
    ax.barh(y, vals, color=colors, height=0.55)
    for yi, v in zip(y, vals, strict=True):
        ax.text(v + 0.15, yi, f"{v:.1f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    lo = min(vals)
    ax.set_xlim(max(0, lo - 6), max(vals) + 2.5)
    ax.set_xlabel("J&F, DAVIS 2017 val, Hiera-T", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "ablations.png")
    plt.close(fig)


def main() -> None:
    davis_comparison()
    ablations()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
