#!/usr/bin/env python3
"""Figure 6: verification ROC, identification DET and CMC for every arm.

Reads results/experiments_vgg_big_threeway.json (and the clean arm from
results/benchmark_measured.json) and plots only measured curves. The
local-only arm is the mean of the five nodes.

Usage:
    python training/02_run_experiments.py --only three_way         --artifacts-suffix _vgg_big --out experiments_vgg_big_threeway.json
    python report_figures/make_fig06_roc_det_cmc.py
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
FIGURES = ROOT / "figures"

INK, MUTED, RULE = "#1a1a1a", "#6b6b6b", "#d9d9d9"

# Okabe-Ito, assigned in fixed order so an arm keeps its colour across all
# three panels. Chosen for published colour-vision-deficiency separation.
ARMS = [
    ("clean",               "#8c564b", "Clean probes (no degradation)", "-"),
    ("baseline_no_adapter", "#0072B2", "Frozen backbone (ArcFace)",     "-"),
    ("centralised",         "#CC79A7", "Centralised adapter",           "-"),
    ("federated",           "#009E73", "Federated adapter (FedMPS)",    "-"),
    ("local_only",          "#D55E00", "Local only (mean of 5 nodes)",  "--"),
]


def log(m):
    print(f"[fig6] {m}", flush=True)


def style(ax, logx=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=RULE, linewidth=0.6, alpha=0.7,
            which="both" if logx else "major")
    ax.set_axisbelow(True)


def collect(tw, clean_block):
    """One dict per arm holding whatever was measured for it."""
    out = {}
    if clean_block:
        out["clean"] = clean_block
    for key in ("baseline_no_adapter", "federated", "centralised"):
        if key in tw:
            out[key] = {"identification": tw[key].get("identification", {}),
                        "verification": tw[key].get("verification", {})}

    # local_only is reported per node; average the curves on a common grid so
    # the figure shows one line rather than five near-identical ones.
    per = tw.get("local_only", {}).get("per_node", {})
    if per:
        cmcs, r1s = [], []
        for v in per.values():
            ident = v.get("identification", {})
            if ident.get("cmc"):
                cmcs.append(ident["cmc"])
            if "rank1" in ident:
                r1s.append(ident["rank1"])
        mean_cmc = {}
        if cmcs:
            for k in cmcs[0]:
                mean_cmc[k] = float(np.mean([c[k] for c in cmcs if k in c]))
        out["local_only"] = {
            "identification": {"cmc": mean_cmc,
                               "rank1": float(np.mean(r1s)) if r1s else None},
            "verification": {},
            "_per_node": len(per),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--three-way", default="experiments_vgg_big_threeway.json")
    ap.add_argument("--measured", default="benchmark_measured.json")
    args = ap.parse_args()

    p = RESULTS / args.three_way
    if not p.exists():
        raise SystemExit(f"{p} not found - run the three-way experiment first")
    tw = json.loads(p.read_text())["three_way"]

    clean = None
    m = RESULTS / args.measured
    if m.exists():
        bm = json.loads(m.read_text())
        c = bm["arms"].get("clean")
        if c:
            clean = {"identification": {"cmc": c.get("cmc", {}),
                                        "rank1": c.get("rank1"),
                                        "det_curve": c.get("det_curve")},
                     "verification": c.get("verification", {})}
            log("clean arm taken from benchmark_measured.json")

    arms = collect(tw, clean)
    log(f"arms available: {', '.join(arms)}")

    fig, (ax_roc, ax_det, ax_cmc) = plt.subplots(1, 3, figsize=(16.5, 4.9),
                                                 dpi=200)
    drew = {"roc": 0, "det": 0, "cmc": 0}

    for key, colour, label, ls in ARMS:
        a = arms.get(key)
        if not a:
            continue
        ident, ver = a.get("identification", {}), a.get("verification", {})

        roc = ver.get("roc_curve")
        if roc and roc.get("far"):
            far, tar = np.asarray(roc["far"]), np.asarray(roc["tar"])
            k = far > 0
            ax_roc.plot(far[k], tar[k], color=colour, lw=2.0, ls=ls,
                        label=label)
            drew["roc"] += 1

        det = ident.get("det_curve")
        if det and det.get("fpir"):
            f, t = np.asarray(det["fpir"]), np.asarray(det["tpir"])
            k = f > 0
            ax_det.plot(f[k], 1.0 - t[k], color=colour, lw=2.0, ls=ls,
                        label=label)
            drew["det"] += 1

        cmc = ident.get("cmc") or {}
        if cmc:
            ks = sorted(int(x[4:]) for x in cmc)
            ax_cmc.plot(ks, [cmc[f"rank{k}"] for k in ks], color=colour,
                        lw=2.0, ls=ls, marker="o", ms=5, label=label)
            drew["cmc"] += 1

    ax_roc.set_xscale("log")
    ax_roc.set_xlabel("False accept rate (FAR)", fontsize=10)
    ax_roc.set_ylabel("True accept rate (TAR)", fontsize=10)
    ax_roc.set_title("1:1 verification ROC\n(higher is better)", fontsize=11,
                     color=INK)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.legend(loc="lower right", frameon=False, fontsize=8)
    style(ax_roc, True)

    ax_det.set_xscale("log")
    ax_det.set_xlabel("False positive identification rate (FPIR)", fontsize=10)
    ax_det.set_ylabel("False negative identification rate (FNIR)", fontsize=10)
    ax_det.set_title("1:N identification DET\n(lower is better)", fontsize=11,
                     color=INK)
    ax_det.set_ylim(0, 1.02)
    ax_det.legend(loc="upper right", frameon=False, fontsize=8)
    style(ax_det, True)

    ax_cmc.set_xscale("log")
    ticks = sorted({int(x[4:]) for a in arms.values()
                    for x in (a.get("identification", {}).get("cmc") or {})})
    if ticks:
        ax_cmc.set_xticks(ticks)
        ax_cmc.set_xticklabels([str(t) for t in ticks])
    ax_cmc.set_xlabel("Rank", fontsize=10)
    ax_cmc.set_ylabel("Retrieval rate", fontsize=10)
    ax_cmc.set_title("1:N cumulative match characteristic\n(higher is better)",
                     fontsize=11, color=INK)
    ax_cmc.legend(loc="lower right", frameon=False, fontsize=8)
    style(ax_cmc)

    ident0 = arms.get("baseline_no_adapter", {}).get("identification", {})
    fig.suptitle(
        f"VGGFace2 surveillance benchmark, "
        f"{ident0.get('n_gallery', '?'):,}-identity gallery, "
        f"{ident0.get('n_mate', '?'):,} mate probes. "
        f"All curves measured; nothing interpolated.",
        fontsize=9, color=MUTED, y=1.03)
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "fig06_roc_det_cmc.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"wrote {out.name}  (roc {drew['roc']} arms, det {drew['det']}, "
        f"cmc {drew['cmc']})")

    print()
    print(f"  {'arm':30s} {'rank1':>8} {'rank5':>8} {'rank10':>8}")
    for key, _c, label, _ls in ARMS:
        a = arms.get(key)
        if not a:
            continue
        ident = a.get("identification", {})
        cmc = ident.get("cmc") or {}
        def f(v):
            return f"{v:.4f}" if isinstance(v, float) else "   -   "
        print(f"  {label:30s} {f(ident.get('rank1')):>8} "
              f"{f(cmc.get('rank5')):>8} {f(cmc.get('rank10')):>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
