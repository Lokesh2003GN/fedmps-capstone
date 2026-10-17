#!/usr/bin/env python3
"""Figure 5: open-set identification DET curves, 3,021-identity gallery.

  clean probes, raw cosine          results/benchmark_measured.json
  degraded, no adapter, raw cosine  results/benchmark_measured.json
  frozen backbone + scoring layer   results/experiments_vgg_big_threeway.json
  federated adapter                 results/experiments_vgg_big_threeway.json
  centralised adapter               results/experiments_vgg_big_threeway.json

Before drawing, the script checks that each curve reproduces the TPIR @ 1%
FPIR reported in the tables. Colours match Figure 6.

Usage:
    python report_figures/make_fig05_det_curves.py
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

ROOT = Path(__file__).resolve().parents[1]   # project root
RESULTS = ROOT / "results"

INK, MUTED, RULE = "#1a1a1a", "#6b6b6b", "#d9d9d9"

# (source, key, colour, label, linestyle)
ARMS = [
    ("measured", "clean",               "#8c564b", "Clean probes (raw cosine)",          "-"),
    ("measured", "degraded",            "#E69F00", "Degraded, no adapter (raw cosine)",  "--"),
    ("threeway", "baseline_no_adapter", "#0072B2", "Frozen backbone + scoring layer",    "-"),
    ("threeway", "federated",           "#009E73", "Federated adapter (FedMPS)",         "-"),
    ("threeway", "centralised",         "#CC79A7", "Centralised adapter",                "-"),
]


def log(m):
    print(f"[fig5] {m}", flush=True)


def tpir_at(det, fpir):
    """Best TPIR on the curve at or below the given FPIR."""
    f, t = np.asarray(det["fpir"]), np.asarray(det["tpir"])
    k = f <= fpir
    return float(t[k].max()) if k.any() else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--three-way", default="experiments_vgg_big_threeway.json")
    ap.add_argument("--measured", default="benchmark_measured.json")
    ap.add_argument("--out", default="fig05_det_curves.png")
    args = ap.parse_args()

    tw = json.loads((RESULTS / args.three_way).read_text())["three_way"]
    bm = json.loads((RESULTS / args.measured).read_text())
    proto = bm["protocol"]

    def arm(src, key):
        if src == "measured":
            a = bm["arms"][key]
            return a["det_curve"], a["tpir"]["tpir@fpir=0.01"]
        a = tw[key]["identification"]
        return a["det_curve"], a["tpir@fpir=0.01"]

    # The frozen arm appears in both files; they must be the same measurement.
    a, b = bm["arms"]["degraded_calibrated"]["det_curve"], \
        tw["baseline_no_adapter"]["identification"]["det_curve"]
    assert np.allclose(a["fpir"], b["fpir"]) and np.allclose(a["tpir"], b["tpir"]), \
        "frozen-backbone DET differs between benchmark_measured and three-way"

    fig, ax = plt.subplots(figsize=(6.6, 4.8), dpi=200)
    for src, key, colour, label, ls in ARMS:
        det, reported = arm(src, key)
        # The curve is sampled on a threshold grid; it must reproduce the
        # tabulated TPIR @ 1% FPIR to within that grid's resolution.
        on_curve = tpir_at(det, 0.01)
        assert abs(on_curve - reported) < 0.01, (key, on_curve, reported)
        f, t = np.asarray(det["fpir"]), np.asarray(det["tpir"])
        k = f > 0
        ax.plot(f[k], 1.0 - t[k], color=colour, lw=1.8, ls=ls, label=label)
        log(f"{key:20s} TPIR@1% table {reported:.4f}  curve {on_curve:.4f}")

    ax.set_xscale("log")
    ax.set_xlim(left=proto["finest_fpir"] * 0.8, right=1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("False positive identification rate (FPIR)", fontsize=10)
    ax.set_ylabel("False negative identification rate (FNIR)", fontsize=10)
    ax.set_title(f"DET (lower is better) - VGGFace2, "
                 f"{proto['n_gallery']:,}-identity gallery",
                 fontsize=11, color=INK)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=RULE, linewidth=0.6, alpha=0.7, which="both")
    ax.set_axisbelow(True)
    fig.tight_layout()

    out = ROOT / "figures" / args.out
    fig.savefig(out)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
