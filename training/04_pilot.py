"""Pilot: which federated module, and which settings, before the 200-round run.

Run from the project root, after 01b_prepare_stream.py:

    python training/04_pilot.py --artifacts-suffix _vgghq --device cuda

The final run trains on all of the 70% and is scored on the held-out 30%, so
nothing about it may be chosen by looking at the 30%. This script holds back
10% of the *training* identities instead, trains each candidate on the other
90% for a short federated run, and scores it on those held-back people with
the same protocol and scoring layer as the real evaluation. The held-out 30%
is never loaded into an evaluation here.

Candidates:
  frozen        the backbone alone (round 0 of every arm)
  mlp_h64_mu3   the first thesis run's adapter and FedProx weight
  mlp_h256_mu03 the same adapter, wider and less tied to the global model,
                now that each node has far more people
  head_r16_mu3  the backbone head: bn2 scale/shift + rank-16 fc update
  head_r16_mu03 the same with a weaker proximal term

Writes results/pilot<SUFFIX>.json.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedmps.config import default_config                         # noqa: E402
from fedmps.contracts import Artefacts, save_json                # noqa: E402
from fedmps.eval.metrics import build_eval_sets, evaluate, fit_calibrator  # noqa: E402
from fedmps.fl.loop import run_federated                         # noqa: E402

CANDIDATES: List[Tuple[str, Dict[str, Any]]] = [
    ("mlp_h64_mu3", {"adapter.kind": "mlp", "adapter.hidden_dim": 64, "fl.fedprox_mu": 3.0}),
    ("mlp_h256_mu03", {"adapter.kind": "mlp", "adapter.hidden_dim": 256, "fl.fedprox_mu": 0.3}),
    ("head_r16_mu3", {"adapter.kind": "head", "adapter.lora_rank": 16, "adapter.lora_alpha": 16.0,
                      "fl.fedprox_mu": 3.0}),
    ("head_r16_mu03", {"adapter.kind": "head", "adapter.lora_rank": 16, "adapter.lora_alpha": 16.0,
                       "fl.fedprox_mu": 0.3}),
    ("head_r16_mu03_lr3e-4", {"adapter.kind": "head", "adapter.lora_rank": 16,
                              "adapter.lora_alpha": 16.0, "fl.fedprox_mu": 0.3,
                              "train.lr": 3e-4}),
]


def log(msg: str) -> None:
    print(f"[pilot] {msg}", flush=True)


def pilot_artefacts(art: Artefacts, frac: float, seed: int) -> Artefacts:
    """The same arrays, with `frac` of every node's identities as the test set.

    The real held-out identities are dropped entirely, so nothing here can be
    scored on them.
    """
    labels = art.labels
    rng = np.random.default_rng(seed + 5151)
    part = copy.deepcopy(art.partition)
    test_rows: List[int] = []
    test_ids: List[int] = []
    test_prof: List[int] = []
    for k in range(art.n_nodes):
        e = part["nodes"][f"node_{k}"]
        rows = np.asarray(e["train_idx"] + e["val_idx"], dtype=np.int64)
        ids = np.unique(labels[rows])
        held = set(int(i) for i in rng.choice(ids, size=max(1, int(round(frac * ids.size))),
                                              replace=False))
        is_held = np.array([int(l) in held for l in labels[rows]])
        test_rows.extend(rows[is_held].tolist())
        test_prof.extend([k] * int(is_held.sum()))
        test_ids.extend(sorted(held))
        keep = set(rows[~is_held].tolist())
        e["train_idx"] = [r for r in e["train_idx"] if r in keep]
        e["val_idx"] = [r for r in e["val_idx"] if r in keep]
        e["identities"] = [i for i in e["identities"] if i not in held]
    order = np.argsort(test_rows)
    part["test_idx"] = np.asarray(test_rows)[order].tolist()
    part["test_profile_ids"] = np.asarray(test_prof)[order].tolist()
    part["test_identities"] = sorted(test_ids)
    out = Artefacts(art.emb_clean, art.emb_deg, art.labels, part, art.meta,
                    art.identity_names, None)
    return out


def apply(cfg, overrides: Dict[str, Any]):
    c = copy.deepcopy(cfg)
    for key, value in overrides.items():
        section, field = key.split(".")
        setattr(getattr(c, section), field, value)
    return c


def summary(res: Dict[str, Any]) -> Dict[str, float]:
    ident, ver = res["identification"], res["verification"]
    return {"rank1": ident["rank1"],
            "tpir@fpir=0.01": ident.get("tpir@fpir=0.01"),
            "tpir@fpir=0.001": ident.get("tpir@fpir=0.001"),
            "tar@far=0.001": ver.get("tar@far=0.001"),
            "eer": ver.get("eer")}


def main() -> int:
    ap = argparse.ArgumentParser(description="FedMPS pilot on held-back training identities")
    ap.add_argument("--artifacts-suffix", default="_vgghq")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--holdback", type=float, default=0.10)
    ap.add_argument("--head-epoch-frac", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    cfg = default_config()
    cfg.paths.artifacts = cfg.paths.artifacts.parent / f"artifacts{args.artifacts_suffix}"
    cfg.backbone.device = args.device
    cfg.fl.rounds = args.rounds
    cfg.eval.eval_every = max(1, args.rounds // 6)
    cfg.eval.n_verification_pairs = 60000
    cfg.train.store_fp16 = True
    if args.lr is not None:
        cfg.train.lr = args.lr

    art = Artefacts.load(cfg)
    pilot = pilot_artefacts(art, args.holdback, cfg.seed)
    log(f"pilot validation: {len(pilot.partition['test_identities'])} identities, "
        f"{len(pilot.partition['test_idx'])} images, all from the training 70%")

    torch.manual_seed(cfg.seed)
    sets = build_eval_sets(pilot, cfg)
    calibrator = fit_calibrator(pilot, cfg)
    log(sets.describe())

    out: Dict[str, Any] = {"rounds": args.rounds, "holdback": args.holdback,
                           "eval_sets": sets.describe(), "candidates": {}}
    frozen = summary(evaluate(sets, cfg, transform=None, calibrator=calibrator))
    out["candidates"]["frozen"] = frozen
    log(f"frozen: {frozen}")

    for name, overrides in CANDIDATES:
        if args.only and name not in args.only:
            continue
        c = apply(cfg, overrides)
        if c.adapter.kind == "head":
            c.train.local_epoch_frac = args.head_epoch_frac
        torch.manual_seed(cfg.seed)
        t0 = time.time()
        res = run_federated(pilot, c, sets, calibrator=calibrator)
        s = summary(res)
        s["seconds"] = round(time.time() - t0, 1)
        s["history_rank1"] = [h["eval"]["rank1"] for h in res["history"] if "eval" in h]
        s["payload_mb"] = res["payload"].get("mb_per_update")
        out["candidates"][name] = s
        log(f"{name}: rank1 {s['rank1']:.4f} ({s['rank1'] - frozen['rank1']:+.4f})  "
            f"tpir@1% {s['tpir@fpir=0.01']:.4f} ({s['tpir@fpir=0.01'] - frozen['tpir@fpir=0.01']:+.4f})  "
            f"{s['seconds']:.0f}s")
        save_json(cfg.paths.results / f"pilot{args.artifacts_suffix}.json", out)

    save_json(cfg.paths.results / f"pilot{args.artifacts_suffix}.json", out)
    log(f"wrote results/pilot{args.artifacts_suffix}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
