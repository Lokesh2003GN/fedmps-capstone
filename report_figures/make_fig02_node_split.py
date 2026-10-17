#!/usr/bin/env python3
"""Figure 2: how VGGFace2 is split across the five nodes.

Bars show each node's training and validation images. Validation is whole
identities held out of training (`fl.val_identity_frac` = 0.2, applied by
`split_node_identities`), so early stopping is measured on people the node
never trained on. These are the numbers in Table 2.

Usage:
    python report_figures/make_fig02_node_split.py --artifacts-suffix _vgg_big
"""
import sys as _sys

for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]   # project root
_sys.path.insert(0, str(ROOT))

from fedmps.config import default_config
from fedmps.contracts import Artefacts
from fedmps.fl.loop import split_node_identities

TRAIN = "#0b5c54"     # dark teal
VAL = "#7fbfb6"       # light teal - separated from TRAIN by lightness, so it
                      # survives any colour-vision deficiency and greyscale print
INK = "#1a1a1a"
MUTED = "#6b6b6b"
RULE = "#d9d9d9"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-suffix", default="_vgg_big")
    args = ap.parse_args()

    cfg = default_config()
    if args.artifacts_suffix:
        cfg.paths.artifacts = (cfg.paths.artifacts.parent
                               / f"artifacts{args.artifacts_suffix}")
    art = Artefacts.load(cfg)
    labels = art.labels
    train_rows, val_rows = split_node_identities(art, cfg)

    nodes = sorted(train_rows)
    tr_img, va_img, tr_id, va_id = [], [], [], []
    for k in nodes:
        t, v = np.asarray(train_rows[k]), np.asarray(val_rows[k])
        tr_img.append(t.size)
        va_img.append(v.size)
        tr_id.append(len(set(labels[t].tolist())))
        va_id.append(len(set(labels[v].tolist())) if v.size else 0)

    held_out = len(art.partition["test_identities"])
    held_img = len(art.partition["test_idx"])
    align = art.meta.get("alignment", {})

    fig, (ax, ax_log) = plt.subplots(
        1, 2, figsize=(13.5, 5.2), dpi=200,
        gridspec_kw={"width_ratios": [1.45, 1.0]})

    x = np.arange(len(nodes))
    w = 0.38
    ax.bar(x - w / 2, tr_img, w, color=TRAIN, label="train images")
    ax.bar(x + w / 2, va_img, w, color=VAL, label="validation images")

    # Direct-label the identity count per node - the quantity the split is
    # actually made on. Image counts are on the axis already.
    for i, (ti, vi) in enumerate(zip(tr_id, va_id)):
        ax.text(i, max(tr_img[i], va_img[i]) + max(tr_img) * 0.035,
                f"{ti + vi} IDs\n{ti} train / {vi} val",
                ha="center", fontsize=8.5, color=INK, linespacing=1.3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Node {k}" for k in nodes])
    ax.set_ylabel("Images", fontsize=11)
    ax.set_ylim(0, max(tr_img) * 1.30)
    ax.set_title("VGGFace2 split across 5 federated nodes\n"
                 "identities held out whole, so validation is people no node trained on",
                 fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=RULE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    missed = align.get("missed", 0)
    total = align.get("aligned", 0) + missed
    extract = "\n".join([
        f"[loader] folder: {art.meta.get('n_identities'):,} identities,",
        f"    reading {art.meta.get('n_images'):,} images at 160px",
        "",
        f"[prepare] {art.meta.get('n_images'):,} images, "
        f"{art.meta.get('n_identities'):,} identities,",
        "    20-20 images each (median 20)",
        "",
        f"[backbone] aligning {total:,} images via",
        f"    {align.get('detector', 'det_10g.onnx')} at "
        f"{align.get('det_size', 320)}px",
        "",
        f"[backbone] alignment done: {align.get('aligned', 0):,}/{total:,}",
        f"    ({100 * align.get('miss_rate', 0):.2f}% missed)",
        "",
        f"nodes: {len(nodes)}   held-out identities: {held_out:,}",
        f"    ({held_img:,} images)",
        "",
        f"[fl] per-node validation: whole identities,",
        f"    fl.val_identity_frac = {cfg.fl.val_identity_frac}",
    ])
    ax_log.axis("off")
    ax_log.set_title("Extract from pipeline log (full_run_vgg_big.log)",
                     fontsize=11, color=INK)
    ax_log.text(0.0, 0.97, extract, fontsize=8.6, family="monospace",
                va="top", ha="left", color=INK,
                bbox=dict(boxstyle="round,pad=0.7", facecolor="#f4f4f4",
                          edgecolor=RULE))

    fig.tight_layout()
    out = cfg.paths.figures / "fig02_node_split.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[fig] wrote {out}")
    print(f"[fig] {'node':6s} {'train':>8} {'val':>7} {'total':>8}  ids")
    for i, k in enumerate(nodes):
        print(f"[fig] node{k:<2d} {tr_img[i]:>8,} {va_img[i]:>7,} "
              f"{tr_img[i] + va_img[i]:>8,}  {tr_id[i] + va_id[i]} "
              f"({tr_id[i]}/{va_id[i]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
