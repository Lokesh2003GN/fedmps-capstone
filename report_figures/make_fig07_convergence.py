#!/usr/bin/env python3
"""Figure 7: validation rank-1 per federated round.

Reads the per-round `val_rank1` history and the early-stopping record from
results/experiments_vgg_big_threeway.json, the run behind Tables 5-7.
Validation uses 518 identities held out of training; early stopping keeps
round 1.

Usage:
    python report_figures/make_fig07_convergence.py
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

ROOT = Path(__file__).resolve().parents[1]   # project root

FED = "#0b5c54"       # dark teal
FROZEN = "#d62728"
KEPT = "#8c510a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--three-way", default="experiments_vgg_big_threeway.json")
    ap.add_argument("--out", default="fig07_convergence.png")
    args = ap.parse_args()

    tw = json.loads((ROOT / "results" / args.three_way).read_text())["three_way"]
    fed = tw["federated"]
    hist = [h for h in fed["history"] if "val_rank1" in h]
    rounds = [h["round"] for h in hist]
    val = [h["val_rank1"] for h in hist]
    es = fed["early_stop"]
    kept, frozen = es["selected_round"], es["val_rank1_no_adapter"]

    # The selected round must be the first round reaching the maximum;
    # otherwise the plot and the caption would disagree with the run.
    assert val[rounds.index(kept)] == max(val), "early stop is not the argmax"
    assert abs(val[rounds.index(kept)] - es["val_rank1_selected"]) < 1e-9

    fig, ax = plt.subplots(figsize=(6.2, 3.7), dpi=200)
    ax.plot(rounds, val, color=FED, marker="o", markersize=3.5,
            linewidth=1.6, label="Federated adapter")
    ax.axhline(frozen, color=FROZEN, linestyle="--", linewidth=1.4,
               label="Frozen backbone (no adapter)")
    ax.axvline(kept, color=KEPT, linestyle=":", linewidth=1.6,
               label=f"Round kept by early stopping ({kept})")
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Validation rank-1")
    ax.set_title("Federated training convergence (validation identities)",
                 fontsize=10)
    ax.set_xlim(0, max(rounds) + 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out = ROOT / "figures" / args.out
    fig.savefig(out)
    print(f"wrote {out}")
    print(f"kept round {kept}: val rank-1 {val[rounds.index(kept)]:.4f} "
          f"vs frozen {frozen:.4f}; rounds {rounds[0]}-{rounds[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
