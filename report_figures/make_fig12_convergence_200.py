#!/usr/bin/env python3
"""Federated training over 200 rounds, scored every round on the held-out 30%.

Reads the three-way results of the VGGFace2-HQ run (70% of identities train
across five nodes, 30% are held out as the validation set) and draws two
graphs, one above the other:

    upper  rank-1 of the federated model after every round, with the 95%
           sampling band of that rank-1, the centralised model when it was
           scored every round, and the frozen backbone
    lower  TPIR at FPIR 1%, the primary metric, the same way

No round is chosen on these curves: the run reports the last round, so the
held-out set is never used for model selection.

Usage:
    python report_figures/make_fig12_convergence_200.py \
        --three-way experiments_vgghq.json
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

FED = "#0b5c54"       # dark teal
CEN = "#8c510a"       # brown
FROZEN = "#d62728"


def curve(arm: dict, key: str):
    hist = [h for h in arm.get("history", []) if "eval" in h]
    rounds = np.arange(1, len(arm.get("history", [])) + 1)
    idx = [i for i, h in enumerate(arm.get("history", [])) if "eval" in h]
    return rounds[idx], np.array([h["eval"][key] for h in hist], dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--three-way", default="experiments_vgghq.json")
    ap.add_argument("--out", default="fig12_convergence_200.png")
    args = ap.parse_args()

    tw = json.loads((ROOT / "results" / args.three_way).read_text())["three_way"]
    fed, cen, base = tw["federated"], tw.get("centralised", {}), tw["baseline_no_adapter"]
    n_mate = fed["identification"]["n_mate"]

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.6), dpi=200, sharex=True)
    panels = [("rank1", "Rank-1", base["identification"]["rank1"]),
              ("tpir@fpir=0.01", "TPIR at FPIR 1%",
               base["identification"]["tpir@fpir=0.01"])]
    for ax, (key, label, frozen) in zip(axes, panels):
        r, y = curve(fed, key)
        se = np.sqrt(np.clip(y * (1 - y), 0, None) / max(n_mate, 1))
        ax.fill_between(r, y - 1.96 * se, y + 1.96 * se, color=FED, alpha=0.15,
                        linewidth=0, label="95% sampling band")
        ax.plot(r, y, color=FED, linewidth=1.8, label="Federated, FedMPS")
        rc, yc = curve(cen, key)
        if rc.size > 1:
            ax.plot(rc, yc, color=CEN, linewidth=1.4, linestyle="-.",
                    label="Centralised")
        ax.axhline(frozen, color=FROZEN, linestyle="--", linewidth=1.4,
                   label="Frozen backbone")
        ax.set_ylabel(label, fontsize=12)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=10)
        ax.legend(loc="lower right", fontsize=9)
    axes[-1].set_xlabel("Federated round", fontsize=12)
    axes[-1].set_xlim(0, len(fed["history"]) + 1)
    fig.tight_layout()

    out = ROOT / "figures" / args.out
    fig.savefig(out)
    r, y = curve(fed, "rank1")
    print(f"wrote {out}")
    print(f"rounds {int(r[0])}-{int(r[-1])}; rank-1 first {y[0]:.4f}, last {y[-1]:.4f}, "
          f"frozen {base['identification']['rank1']:.4f}; std of last 20 rounds "
          f"{np.std(y[-20:]):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
