#!/usr/bin/env python3
"""Figure 3: aligned crops as the backbone sees them, for a prepared dataset.

    (a) enrolment crops above their CCTV-degraded probes, held-out identities
    (b) one identity through each node's camera profile, at the profile's own
        settings, labelled with its face size and JPEG quality

Reads the sample crops that the preparation step keeps (samples.npz) and the
node camera profiles from partition.json, and degrades with the same code and
ranges as the pipeline.

Usage:
    python report_figures/make_fig03_aligned_crops.py --artifacts-suffix _vgghq
"""
import sys as _sys

for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(ROOT))

from fedmps.config import default_config                      # noqa: E402
from fedmps.data.degrade import degrade_image, resolve_params  # noqa: E402

INK, MUTED = "#1a1a1a", "#555555"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-suffix", default="_vgghq")
    ap.add_argument("--out", default="docs/thesis/figures/aligned_crops_vgghq.png")
    args = ap.parse_args()

    a = ROOT / f"artifacts{args.artifacts_suffix}"
    s = np.load(a / "samples.npz")
    part = json.loads((a / "partition.json").read_text(encoding="utf-8"))
    test = set(part["test_identities"])
    cfg = default_config()

    # One crop per person: held-out people first, then training people, 16 in all.
    order = sorted(range(len(s["labels"])), key=lambda i: int(s["labels"][i]) not in test)
    seen, picks = set(), []
    for i in order:
        if int(s["labels"][i]) not in seen:
            seen.add(int(s["labels"][i]))
            picks.append(i)
        if len(picks) == 16:
            break
    n_prof = 3                           # identities shown through every camera profile

    # Rows: two blocks of enrolment + probe, a gap for heading (b), n_prof profile rows.
    fig = plt.figure(figsize=(16, 13.6), dpi=200)
    gs = fig.add_gridspec(5 + n_prof, 8, height_ratios=[1, 1, 1, 1, 0.45] + [1] * n_prof,
                          hspace=0.1, wspace=0.06, left=0.04, right=0.99, top=0.94,
                          bottom=0.04)
    first = None
    for j, i in enumerate(picks):
        block, col = divmod(j, 8)
        for r, (key, name) in enumerate((("clean", "enrolment"), ("degraded", "CCTV probe"))):
            ax = fig.add_subplot(gs[2 * block + r, col])
            ax.imshow(s[key][i])
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(name, fontsize=12, color=INK)
                first = first or ax
    pos = first.get_position()
    fig.text(pos.x0, pos.y1 + 0.012, "(a)  Enrolment crop and the matching CCTV-degraded "
             f"probe, {len(picks)} identities", fontsize=14, fontweight="bold", color=INK)

    for row in range(n_prof):
        clean = s["clean"][picks[row]]
        rng = np.random.default_rng(cfg.seed)
        ax = fig.add_subplot(gs[5 + row, 0])
        ax.imshow(clean); ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"identity {row + 1}", fontsize=12, color=INK)
        if row == 0:
            ax.set_title("clean", fontsize=12, color=INK)
            pos = ax.get_position()
            fig.text(pos.x0, pos.y1 + 0.03, "(b)  The same identities through each node's "
                     "camera profile", fontsize=14, fontweight="bold", color=INK)
        for k in range(part["n_nodes"]):
            prof = part["nodes"][f"node_{k}"]["degrade_params"]
            p = resolve_params(prof, cfg.degrade.ranges, rng, 0.0)
            img = degrade_image(clean, p, cfg.degrade, np.random.default_rng(cfg.seed + k))
            ax = fig.add_subplot(gs[5 + row, k + 1])
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"node {k}", fontsize=12, color=INK)
            if row == n_prof - 1:
                ax.set_xlabel(f"{p.face_px}px  q{p.jpeg_quality}", fontsize=11, color=MUTED)
    ax = fig.add_subplot(gs[5:, 6:])
    ax.axis("off")
    ax.text(0.06, 0.5, "Each node's camera has a fixed\nprofile: face size, blur, noise,\n"
            "gamma and JPEG quality, so the\nfive nodes see genuinely\ndifferent distributions.",
            fontsize=12, color=MUTED, va="center", transform=ax.transAxes)

    out = ROOT / args.out
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
