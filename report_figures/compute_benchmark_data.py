#!/usr/bin/env python3
"""Measure the benchmark arms used by the report figures and tables.

Writes results/benchmark_measured.json. Every number is computed here from
the saved embeddings; nothing is interpolated or assumed.

Arms, all on the same held-out identities and the same 3,021-identity gallery:

  clean                frozen backbone, undegraded probes  (upper bound)
  degraded             frozen backbone, CCTV-degraded probes, raw cosine
  degraded_calibrated  the scoring layer on degraded probes
  federated_adapter    the federated adapter on degraded probes

Usage:
    python report_figures/compute_benchmark_data.py --artifacts-suffix _vgg_big
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]   # project root
_sys.path.insert(0, str(ROOT))

from fedmps.config import default_config
from fedmps.contracts import Artefacts, save_json
from fedmps.eval.metrics import (
    build_eval_sets, evaluate, fit_calibrator,
)

RANKS = (1, 2, 3, 5, 10, 20, 50)


def log(msg: str) -> None:
    print(f"[bench] {msg}", flush=True)


def score_histogram(sets, cfg, calibrator=None, bins: int = 40):
    """Genuine and impostor similarity distributions, as actually measured.

    Genuine  - each mate probe against its own gallery entry.
    Impostor - each mate probe against its best *wrong* gallery entry, which
               is the distribution a threshold actually has to separate, not
               a random pairing that flatters the separation.
    """
    g, gid = sets.gallery_emb, sets.gallery_ids
    m, mid = sets.mate_emb, sets.mate_ids
    if calibrator is not None:
        g, m = calibrator.gallery(g), calibrator.probe(m)

    genuine, impostor = [], []
    for i in range(0, m.shape[0], 2048):
        sim = m[i:i + 2048] @ g.T
        own = gid[None, :] == mid[i:i + 2048, None]
        has = own.any(1)
        genuine.extend(sim[own])
        wrong = np.where(own, -np.inf, sim)
        impostor.extend(wrong.max(1)[has])

    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    lo = float(min(genuine.min(), impostor.min()))
    hi = float(max(genuine.max(), impostor.max()))
    edges = np.linspace(lo, hi, bins + 1)
    return {
        "bin_edges": edges.tolist(),
        "genuine": np.histogram(genuine, bins=edges)[0].tolist(),
        "impostor": np.histogram(impostor, bins=edges)[0].tolist(),
        "genuine_mean": float(genuine.mean()),
        "impostor_mean": float(impostor.mean()),
        "n_genuine": int(genuine.size),
        "n_impostor": int(impostor.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-suffix", default="_vgg_big")
    args = ap.parse_args()

    cfg = default_config()
    if args.artifacts_suffix:
        cfg.paths.artifacts = (cfg.paths.artifacts.parent
                               / f"artifacts{args.artifacts_suffix}")
    log(f"artefacts: {cfg.paths.artifacts}")
    art = Artefacts.load(cfg)
    cal = fit_calibrator(art, cfg)

    deg = build_eval_sets(art, cfg, probe_source="degraded")
    clean = build_eval_sets(art, cfg, probe_source="clean")
    log(deg.describe())

    out = {
        "protocol": {
            "dataset": "VGGFace2",
            "n_images": art.meta.get("n_images"),
            "n_identities": art.meta.get("n_identities"),
            "n_gallery": int(deg.gallery_ids.size),
            "n_mate_probes": int(deg.mate_ids.size),
            "n_nonmate_probes": int(deg.nonmate_emb.shape[0]),
            "n_verification_pairs": int(deg.ver_same.size),
            "n_enrol": deg.n_enrol,
            "n_probe_frames": deg.n_probe_frames,
            "backbone": art.meta.get("backbone"),
            "alignment": art.meta.get("alignment"),
            "cmc_ranks": list(RANKS),
            # The finest rate the probe counts can resolve. Anything stricter
            # comes back NaN ("not resolvable") instead of a number.
            "finest_fpir": 1.0 / max(1, deg.nonmate_emb.shape[0]),
            "finest_far": 2.0 / max(1, int(deg.ver_same.size)),
        },
        "arms": {},
    }

    arms = [
        ("clean", clean, None, None),
        ("degraded", deg, None, None),
        ("degraded_calibrated", deg, None, cal),
    ]

    ck = cfg.paths.checkpoints / "adapter_final.pt"
    if ck.exists():
        import torch
        from fedmps.models.adapter import build_adapter
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        if blob.get("backbone") == art.meta.get("backbone"):
            c2 = default_config()
            c2.adapter.hidden_dim = blob["adapter_config"]["hidden_dim"]
            mdl = build_adapter(c2.adapter)
            mdl.load_state_dict(blob["state_dict"])
            mdl.eval()
            arms.append(("federated_adapter", deg,
                         lambda e: mdl.transform_numpy(e), cal))
            log("including the federated adapter arm")
        else:
            log(f"skipping the adapter arm: checkpoint was trained on "
                f"'{blob.get('backbone')}' but the artefacts use "
                f"'{art.meta.get('backbone')}'")
    else:
        log("no adapter checkpoint; skipping that arm")

    for name, sets, tf, c in arms:
        log(f"measuring {name} ...")
        res = evaluate(sets, cfg, transform=tf, calibrator=c)
        ident, ver = res["identification"], res["verification"]
        out["arms"][name] = {
            "rank1": ident["rank1"],
            "cmc": ident.get("cmc", {}),
            "tpir": {k: ident[k] for k in ident
                     if k.startswith("tpir@") and not k.endswith("_note")},
            "det_curve": ident.get("det_curve"),
            "verification": {
                "auc": ver.get("auc"),
                "eer": ver.get("eer"),
                "roc_curve": ver.get("roc_curve"),
                **{k: ver[k] for k in ver if k.startswith("tar@")},
            },
            "scores": score_histogram(sets, cfg, c),
        }
        cmc = out["arms"][name]["cmc"]
        log(f"   rank-1 {ident['rank1']:.4f}  "
            + "  ".join(f"r{k}={cmc.get(f'rank{k}', float('nan')):.4f}"
                        for k in (5, 10)))

    dest = cfg.paths.results / "benchmark_measured.json"
    save_json(dest, out)
    log(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
