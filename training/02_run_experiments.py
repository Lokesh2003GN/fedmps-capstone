"""Phase 3 entry point: every experiment, every figure, one command.

Run from the project root, after 01_prepare_data.py:

    python training/02_run_experiments.py

Writes results/experiments.json and figures/training_run/*.png. The JSON carries a full
config snapshot, so a number in the report can always be traced back to the
settings that produced it.

Useful flags while you are still checking things work:

    --quick                 few rounds, the fast experiments only
    --only three_way        run one experiment
    --rounds 10             override the federated round count
    --no-figures            skip matplotlib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedmps.config import default_config                        # noqa: E402
from fedmps.contracts import Artefacts, save_json               # noqa: E402
from fedmps.eval.experiments import (                           # noqa: E402
    ALL_EXPERIMENTS, DEFAULT_EXPERIMENTS, run_all,
)


def log(msg: str) -> None:
    print(f"[run] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="FedMPS Phase 3 experiments")
    ap.add_argument("--only", nargs="+", choices=list(ALL_EXPERIMENTS),
                    default=None, help="run only these experiments")
    ap.add_argument("--all", action="store_true",
                    help="run every experiment including fedprox and mia")
    ap.add_argument("--quick", action="store_true",
                    help="short run for checking the pipeline works")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--aux-rounds", type=int, default=None,
                    help="rounds for the side experiments (DP sweep, masking, "
                         "FedProx, membership inference); the three-way "
                         "comparison keeps --rounds")
    ap.add_argument("--nodes-per-round", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None,
                    help="where the adapter trains. cuda keeps the node data "
                         "on the GPU, which millions of images need")
    ap.add_argument("--fp16-store", action="store_true",
                    help="keep node training embeddings in float16 on the "
                         "device (half the memory; batches are cast back)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None,
                    help="score the held-out 30%% every N rounds")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lr-schedule", choices=["cosine", "constant"],
                    default=None)
    ap.add_argument("--mu", type=float, default=None,
                    help="FedProx proximal weight; 0 is plain FedAvg")
    ap.add_argument("--hidden", type=int, default=None,
                    help="adapter hidden width (64 gives 66,240 parameters)")
    ap.add_argument("--adapter-kind", choices=["mlp", "head"], default=None,
                    help="mlp: the residual MLP on the output embedding. head: "
                         "the backbone's last BatchNorm and a low-rank update "
                         "of its last layer, on features from "
                         "01b_prepare_stream.py")
    ap.add_argument("--lora-rank", type=int, default=None)
    ap.add_argument("--epoch-frac", type=float, default=None,
                    help="fraction of each node's data passed over per round")
    ap.add_argument("--val-identity-frac", type=float, default=None,
                    help="fraction of each node's identities held back from "
                         "training as a validation set; 0 trains on all of "
                         "them. Used by hyper-parameter sweeps, so the "
                         "held-out 30%% is never used to choose settings")
    ap.add_argument("--out", default="experiments.json")
    ap.add_argument("--artifacts-suffix", default="",
                    help="read from artifacts<SUFFIX>/ instead of artifacts/, "
                         "matching 01_prepare_data.py --out-suffix. Lets two "
                         "prepared datasets be scored without either "
                         "overwriting the other")
    args = ap.parse_args()

    cfg = default_config()
    if args.artifacts_suffix:
        cfg.paths.artifacts = (cfg.paths.artifacts.parent
                               / f"artifacts{args.artifacts_suffix}")
        log(f"reading artefacts from {cfg.paths.artifacts}")
    if args.seed is not None:
        cfg.seed = args.seed
    if args.rounds:
        cfg.fl.rounds = args.rounds
    if args.nodes_per_round:
        cfg.fl.clients_per_round = args.nodes_per_round
    if args.device:
        cfg.backbone.device = args.device
    if args.fp16_store:
        cfg.train.store_fp16 = True
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.eval_every:
        cfg.eval.eval_every = args.eval_every
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.lr_schedule:
        cfg.train.lr_schedule = args.lr_schedule
    if args.mu is not None:
        cfg.fl.fedprox_mu = args.mu
    if args.hidden:
        cfg.adapter.hidden_dim = args.hidden
    if args.adapter_kind:
        cfg.adapter.kind = args.adapter_kind
    if args.lora_rank:
        cfg.adapter.lora_rank = args.lora_rank
        cfg.adapter.lora_alpha = float(args.lora_rank)
    if args.epoch_frac is not None:
        cfg.train.local_epoch_frac = args.epoch_frac
    if args.val_identity_frac is not None:
        cfg.fl.val_identity_frac = args.val_identity_frac
        cfg.fl.early_stop = args.val_identity_frac > 0

    which = (tuple(args.only) if args.only
             else ALL_EXPERIMENTS if args.all
             else DEFAULT_EXPERIMENTS)

    rounds_override = args.aux_rounds
    if args.quick:
        cfg.fl.rounds = min(cfg.fl.rounds, 8)
        cfg.train.centralised_epochs = min(cfg.train.centralised_epochs, 15)
        cfg.eval.eval_every = 2
        rounds_override = 5
        if not args.only and not args.all:
            which = ("degradation", "three_way", "masking")
        log("quick mode: shortened runs, results are not reportable")

    art = Artefacts.load(cfg)
    print(art.describe())
    print()

    if art.meta.get("backbone") == "dummy":
        log("NOTE: Phase 1 used the synthetic backbone. Everything below will "
            "run and the directions should be right, but the numbers are not "
            "reportable. Install a real backbone and re-run 01_prepare_data.py "
            "before writing results.")
        print()

    log(f"experiments: {', '.join(which)}")
    results = run_all(art, cfg, which=which,
                      rounds_override=rounds_override,
                      figures=not args.no_figures,
                      save_to=cfg.paths.results / args.out)

    out_path = cfg.paths.results / args.out
    save_json(out_path, results)

    print()
    log(f"wrote {out_path}")
    for f in results.get("figures", []):
        log(f"  figure {Path(f).name}")

    if results.get("warnings"):
        print()
        log("WARNINGS - read these before writing anything up:")
        for w in results["warnings"]:
            print(f"    - {w}")
    else:
        print()
        log("no warnings raised")

    # A compact table for pasting straight into the report.
    tw = results.get("three_way", {}).get("summary", {}).get("rank1")
    if tw:
        print()
        log("rank-1 on held-out identities")
        for name, key in (("no adapter", "no_adapter"),
                          ("local only", "local_only_mean"),
                          ("federated", "federated"),
                          ("centralised", "centralised")):
            print(f"    {name:<14} {tw[key]:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
