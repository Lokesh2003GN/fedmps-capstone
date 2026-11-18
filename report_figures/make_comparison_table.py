#!/usr/bin/env python3
"""Comparison with five published federated / privacy-preserving face systems.

Each row carries the metric its paper reports, the paper's own number, and
FedMPS measured on that same metric from the VGGFace2-HQ run, so no cell is
empty. Paper numbers were read from each paper's own tables (see the
`source` field); FedMPS numbers come from results/<three-way json>.

The protocols differ (datasets, backbones, clients, test sets), so the table
supports a comparison of design and of the cost of privacy, not a ranking.

Usage:
    python report_figures/make_comparison_table.py --results experiments_vgghq.json
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAPERS = [
    {"ref": "Cao et al. 2018, VGGFace2", "reference_row": True,
     "data": "VGGFace2, 8,631 identities", "model": "ResNet-50, centralised",
     "metric": "IJB-C rank-1 / TAR@FAR 1e-4", "result": "0.898 / 0.825",
     "ours": ("rank1", "tar@far=0.0001"), "source": "Table VIII"},
    {"ref": "FedHide, Park & Yun 2024", "data": "VGGFace2, 8,631 clients, one identity each",
     "model": "MobileFaceNet, 64x64", "metric": "EER",
     "result": "9.5% unprotected; 10.4-12.9% protected", "ours": ("eer",),
     "source": "Table 4"},
    {"ref": "FedFS, Kim, Choi & Kwak, BMVC 2024", "data": "VGGFace, 8,673 identities",
     "model": "MobileFaceNet", "metric": "AUROC / TPIR@FPIR 0.1, 0.01, 0.001",
     "result": "0.9794 / 0.9786, 0.9337, 0.8721",
     "ours": ("auc", "tpir@fpir=0.1", "tpir@fpir=0.01", "tpir@fpir=0.001"),
     "source": "Tables 1 and 2"},
    {"ref": "Kim, Kim & Teoh, ESWA 2026", "data": "MS-Celeb-1M, 8,000 identities, 20 clients",
     "model": "Heterogeneous backbones", "metric": "TAR@FAR 1e-4 (IJB-C, per client)",
     "result": "57.14%", "ours": ("tar@far=0.0001",), "source": "Table 2"},
    {"ref": "DP-FedFace, Wang & Li, CIKM 2024", "data": "CASIA-WebFace, 4,000 clients",
     "model": "64-layer CNN", "metric": "TAR@FAR 1e-3 (IJB-C)",
     "result": "87.76%", "ours": ("tar@far=0.001",), "source": "Table 1"},
    {"ref": "Ji et al., ECCV 2022", "data": "VGGFace2", "model": "ResNet-50, ArcFace, DP",
     "metric": "TAR@FAR 1e-4 (IJB-C)", "result": "93.25% unprotected; 91.22% private",
     "ours": ("tar@far=0.0001",), "source": "Table 1"},
]


def metrics(result: dict) -> dict:
    ident, ver = result["identification"], result["verification"]
    out = {k: ident[k] for k in ident if k == "rank1" or (k.startswith("tpir@") and not k.endswith("_note"))}
    out.update({k: ver[k] for k in ("auc", "eer", "tar@far=0.001", "tar@far=0.0001") if k in ver})
    return out


def fmt(key: str, value: float) -> str:
    if key == "eer" or key.startswith("tar@"):
        return f"{100 * value:.2f}%"
    return f"{value:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments_vgghq.json")
    ap.add_argument("--out", default="comparison_table_vgghq.md")
    args = ap.parse_args()

    tw = json.loads((ROOT / "results" / args.results).read_text(encoding="utf-8"))["three_way"]
    fed, frozen = metrics(tw["federated"]), metrics(tw["baseline_no_adapter"])

    lines = ["| Reference | Training data | Model | Metric reported | Paper's result | "
             "FedMPS, same metric | Frozen backbone, same metric |",
             "|---|---|---|---|---|---|---|"]
    for p in PAPERS:
        ours = ", ".join(fmt(k, fed[k]) for k in p["ours"])
        base = ", ".join(fmt(k, frozen[k]) for k in p["ours"])
        lines.append(f"| {p['ref']} | {p['data']} | {p['model']} | {p['metric']} | "
                     f"{p['result']} | {ours} | {base} |")
    text = "\n".join(lines) + "\n"
    out = ROOT / "results" / args.out
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
