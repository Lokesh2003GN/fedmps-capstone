#!/usr/bin/env python3
"""Thesis versions of two-panel result figures: one panel per row, larger text.

The review document keeps its side-by-side figures in figures/. The thesis
prints at a single column width, where side-by-side panels are too small to
read, so it uses these stacked versions instead. Plotted from the measured
results, nothing re-run:

    docs/thesis/figures/operating_points_stacked.png   (from results/experiments_vgghq_base.json)

The other stacked figures come from the report scripts' --stacked option:

    python report_figures/make_fig02_node_split.py --stacked
    python report_figures/make_fig06_roc_det_cmc.py --stacked

Usage:
    python docs/thesis/make_stacked_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "figures"

INK, MUTED, RULE = "#1a1a1a", "#6b6b6b", "#d9d9d9"
STOP, ALT, GREY = "#8f2d24", "#2c5a85", "#6b7778"
FS = 12


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=FS - 1)
    ax.grid(True, axis="y", color=RULE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def operating_points(results: str = "experiments_vgghq_base.json") -> Path:
    op = json.loads((ROOT / "results" / results).read_text())["operating_points"]
    rows, d = op["rows"], op["default"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.0, 10.0), dpi=200)

    enrols = sorted({r["n_enrol"] for r in rows})
    frames = sorted({r["n_probe_frames"] for r in rows})
    for colour, ne in zip([STOP, INK, ALT, GREY], enrols):
        ys = [next(r["rank1_hubness"] for r in rows
                   if r["n_enrol"] == ne and r["n_probe_frames"] == nf) for nf in frames]
        ax.plot(frames, ys, marker="o", ms=7, color=colour, linewidth=2.2,
                label=f"{ne} enrolment photo{'s' if ne > 1 else ''}")
    ax.set_xlabel("Camera frames fused into one probe", fontsize=FS)
    ax.set_ylabel("Rank-1 identification", fontsize=FS)
    ax.set_xticks(frames)
    ax.set_ylim(0.45, 1.02)
    ax.legend(frameon=False, fontsize=FS - 1, loc="lower right")
    ax.set_title(f"Rank-1 across the operating surface\n"
                 f"({d['n_gallery']:,}-identity gallery, frozen backbone)",
                 fontsize=FS + 1, color=INK)
    style(ax)

    keys = [("rank1_cosine", "Cosine"), ("rank1_centred", "+ centring"),
            ("rank1_rotated", "+ rotation"), ("rank1_hubness", "+ hubness")]
    vals = [d[k] for k, _ in keys]
    bars = ax2.bar([lbl for _, lbl in keys], vals, color=[GREY, ALT, ALT, INK], width=0.6)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=FS, color=INK)
    ax2.set_ylabel("Rank-1 identification", fontsize=FS)
    ax2.set_ylim(min(vals) - 0.03, max(vals) + 0.015)
    ax2.set_title(f"Scoring layer, fitted on training identities only\n"
                  f"({d['n_enrol']} photos × {d['n_probe_frames']} frames)",
                  fontsize=FS + 1, color=INK)
    style(ax2)

    fig.tight_layout(h_pad=3.0)
    out = OUT / "operating_points_stacked.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"wrote {operating_points()}")
