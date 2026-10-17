#!/usr/bin/env python3
"""Fairness check for the local-only arm of the three-way comparison.

In Table 5, local-only trains each node for 30 epochs and keeps the last
epoch, while the federated arm keeps its best round on held-out validation
people and trains under the FedProx penalty. This script retrains local-only
the same way the federated arm is treated:

  no_early_stop    30 epochs, last epoch kept (as in Table 5)
  early_stop       best epoch on the node's OWN validation identities
  early_stop_prox  the same, plus the proximal penalty (mu = fl.fedprox_mu)
                   towards the starting weights

All arms use the same test set and scoring layer as Table 5.

Usage:
    python training/03_check_local_only.py --artifacts-suffix _vgg_big
"""
import sys as _sys

for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # project root
_sys.path.insert(0, str(ROOT))

import torch                                                     # noqa: E402

from fedmps.config import default_config                         # noqa: E402
from fedmps.contracts import Artefacts, save_json                # noqa: E402
from fedmps.eval.metrics import (                                # noqa: E402
    build_eval_sets, evaluate, fit_calibrator,
)
from fedmps.fl.loop import (                                     # noqa: E402
    _transform_of, _val_rank1, build_val_sets, split_node_identities,
)
from fedmps.fl.node import local_train, make_node_data           # noqa: E402
from fedmps.models.adapter import build_adapter, clone_state     # noqa: E402


def log(msg: str) -> None:
    print(f"[local-check] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-suffix", default="_vgg_big")
    ap.add_argument("--out", default="local_only_check.json")
    args = ap.parse_args()

    cfg = default_config()
    cfg.paths.artifacts = ROOT / f"artifacts{args.artifacts_suffix}"
    dev = cfg.backbone.device
    epochs = cfg.fl.rounds * cfg.train.local_epochs
    mu = cfg.fl.fedprox_mu

    torch.manual_seed(cfg.seed)
    art = Artefacts.load(cfg)
    sets = build_eval_sets(art, cfg)
    cal = fit_calibrator(art, cfg)

    def score(model):
        res = evaluate(sets, cfg,
                       transform=_transform_of(model) if model else None,
                       calibrator=cal)
        i = res["identification"]
        return {"rank1": i["rank1"], "tpir@1%": i["tpir@fpir=0.01"],
                "tpir@0.1%": i["tpir@fpir=0.001"]}

    t0 = time.time()
    out = {"frozen": score(None), "epochs": epochs, "fedprox_mu": mu,
           "nodes": {}, "config": cfg.to_dict()}
    log(f"frozen backbone: rank1={out['frozen']['rank1']:.4f}")

    node_train, node_val = split_node_identities(art, cfg)
    for k in range(art.n_nodes):
        data = make_node_data(art, node_train[k], dev)
        own_val = build_val_sets(art, cfg, {k: node_val[k]})
        rec = {"n_train": data.n,
               "n_val_identities": int(own_val.gallery_ids.size)}

        # As in Table 5: all epochs in one call, last epoch kept.
        model = build_adapter(cfg.adapter, dev)
        local_train(model, data, cfg, epochs=epochs, seed=cfg.seed + k)
        rec["no_early_stop"] = score(model)

        # Epoch by epoch; keep the best epoch on this node's own validation set.
        for name, prox_mu in (("early_stop", 0.0), ("early_stop_prox", mu)):
            torch.manual_seed(cfg.seed + k)
            model = build_adapter(cfg.adapter, dev)
            anchor = [p.detach().clone() for p in model.parameters()]
            best, best_ep, best_state, curve = -1.0, 0, None, []
            for ep in range(1, epochs + 1):
                local_train(model, data, cfg, epochs=1,
                            seed=cfg.seed + 1000 * ep + k,
                            fedprox_mu=prox_mu,
                            global_params=anchor if prox_mu > 0 else None)
                v = _val_rank1(own_val, model, cal)
                curve.append(round(v, 4))
                if v > best:
                    best, best_ep, best_state = v, ep, clone_state(model.state_dict())
            model.load_state_dict(best_state)
            rec[name] = {**score(model), "best_epoch": best_ep, "val_curve": curve}

        out["nodes"][f"node_{k}"] = rec
        log(f"node_{k}: last-epoch {rec['no_early_stop']['rank1']:.4f} | "
            f"early-stop {rec['early_stop']['rank1']:.4f} "
            f"(epoch {rec['early_stop']['best_epoch']}) | "
            f"early-stop+prox {rec['early_stop_prox']['rank1']:.4f} "
            f"(epoch {rec['early_stop_prox']['best_epoch']})")

    def mean(key, metric):
        vals = [n[key][metric] for n in out["nodes"].values()]
        return sum(vals) / len(vals)

    out["summary"] = {
        key: {m: round(mean(key, m), 4) for m in ("rank1", "tpir@1%", "tpir@0.1%")}
        for key in ("no_early_stop", "early_stop", "early_stop_prox")
    }
    out["elapsed_sec"] = round(time.time() - t0, 1)

    path = cfg.paths.results / args.out
    save_json(path, out)
    for key, s in out["summary"].items():
        log(f"mean {key:16s} rank1={s['rank1']:.4f}  "
            f"tpir@1%={s['tpir@1%']:.4f}  tpir@0.1%={s['tpir@0.1%']:.4f}")
    log(f"wrote {path} ({out['elapsed_sec']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
