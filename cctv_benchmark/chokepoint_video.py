#!/usr/bin/env python3
"""CCTV benchmark, part B: the live pipeline end to end on real footage.

Runs the application's own components - SCRFD detection, ByteTrack tracking,
the quality gate, the backbone (+ federated adapter) and k-of-n track voting -
over full ChokePoint video sequences, and scores the alerts against the
ground truth. This is what an operator would experience, not a crop-level
accuracy number.

Protocol (same watch-list splits as part A):
  * half the people are on the watchlist, enrolled from --n-enrol frames of
    the other sequence group (G1 <-> G2); the rest are the crowd;
  * every frame at the pipeline's sample rate goes through the pipeline;
  * an alert is attributed to the annotated person whose eyes lie inside the
    alerted track's box on that frame.

Reported per arm, mean over splits and both directions:
  pass detection rate   enrolled people walking through who got a correct alert
  wrong-identity alerts alerts on an enrolled person naming someone else
  crowd false-alarm rate crowd walk-throughs that raised any alert
  unattributed alerts   alerts on a track no annotated face lies in
  time to alert         seconds from a person's first annotated frame

Detection, tracking, gating and embedding run once per sequence; matching and
voting are replayed per split, which is exactly equivalent and far cheaper.

Usage (from the project root; needs the full-frame archives P1E_S1.tar.xz ...):
    python cctv_benchmark/chokepoint_video.py --root <chokepoint folder> --device cuda
"""
import sys as _sys

for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]   # project root
_sys.path.insert(0, str(ROOT))
_sys.path.insert(0, str(Path(__file__).resolve().parent))

import chokepoint_recognition as cr                          # noqa: E402
from fedmps.config import default_config                     # noqa: E402
from fedmps.contracts import l2_normalise                    # noqa: E402
from fedmps.models.backbone import get_backbone              # noqa: E402
from fedmps.pipeline.detect import Detection, get_detector, get_tracker  # noqa: E402
from fedmps.pipeline.match import TrackVoter, load_adapter   # noqa: E402
from fedmps.pipeline.quality import QualityGate              # noqa: E402

FPS = 30.0   # ChokePoint frame rate


def log(msg: str) -> None:
    print(f"[chokepoint-video] {msg}", flush=True)


def attribute(bbox, people) -> Optional[str]:
    """The annotated person whose eye midpoint lies inside this box."""
    x1, y1, x2, y2 = bbox
    for pid, (lx, ly, rx, ry) in people:
        cx, cy = (lx + rx) / 2.0, (ly + ry) / 2.0
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return pid
    return None


def scan_sequence(seq_dir: Path, gt_rows, cfg, detector, backbone,
                  max_frames: Optional[int]) -> Dict:
    """Detect -> track -> gate -> embed every sampled frame, once."""
    tracker = get_tracker(cfg)
    gate = QualityGate(cfg)
    stride = max(1, int(round(FPS / max(1, cfg.pipeline.sample_fps))))
    size = cfg.data.image_size

    gt_by_frame = defaultdict(list)
    for n, pid, eyes in gt_rows:
        gt_by_frame[n].append((pid, eyes))
    first_seen = {}
    for n, pid, _ in sorted(gt_rows):
        first_seen.setdefault(pid, n)

    frames = sorted(p for p in seq_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".png"))
    recs, crops = [], []
    n_det = processed = 0
    for i, path in enumerate(frames):
        if i % stride:
            continue
        if max_frames and processed >= max_frames:
            break
        processed += 1
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        n = int(path.stem)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dets = detector.detect(rgb)
        n_det += len(dets)
        for tr in tracker.update(dets):
            det = tr.detection or Detection(tr.bbox, tr.score)
            if not gate.assess(rgb, det).passed:
                continue
            crop = det.crop(rgb)
            crops.append(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA))
            recs.append({"frame": n, "track": tr.track_id,
                         "pid": attribute(tr.bbox, gt_by_frame.get(n, []))})
    emb = (backbone.embed(np.stack(crops), batch_size=cfg.backbone.batch_size,
                          progress=False)
           if crops else np.zeros((0, cfg.adapter.in_dim), np.float32))
    return {"recs": recs, "emb": emb, "first_seen": first_seen,
            "people": sorted(first_seen), "frames_processed": processed,
            "detections": n_det, "gate": gate.report()}


def replay(scan: Dict, probe_emb: np.ndarray, gallery: np.ndarray,
           gallery_ids: np.ndarray, cfg) -> List[Dict]:
    """Matching and k-of-n voting, as VideoPipeline does it, for one watchlist."""
    voter = TrackVoter(cfg)
    alerts = []
    sims = probe_emb @ gallery.T if len(probe_emb) else np.zeros((0, len(gallery_ids)))
    for rec, row in zip(scan["recs"], sims):
        j = int(np.argmax(row))
        fired = voter.observe(rec["track"], str(gallery_ids[j]), float(row[j]))
        if fired:
            alerts.append({"frame": rec["frame"], "track": rec["track"],
                           "case": str(gallery_ids[j]), "pid": rec["pid"],
                           "score": float(row[j])})
    return alerts


def score_alerts(alerts, people, watch, first_seen) -> Dict:
    mates = [p for p in people if p in watch]
    crowd = [p for p in people if p not in watch]
    detected, latency = set(), []
    wrong = false_alarm = unattributed = 0
    crowd_alerted = set()
    for a in alerts:
        if a["pid"] is None:
            unattributed += 1
        elif a["pid"] in watch:
            if a["case"] == a["pid"]:
                if a["pid"] not in detected:
                    latency.append((a["frame"] - first_seen[a["pid"]]) / FPS)
                detected.add(a["pid"])
            else:
                wrong += 1
        else:
            false_alarm += 1
            crowd_alerted.add(a["pid"])
    return {"mate_passes": len(mates), "crowd_passes": len(crowd),
            "detected": len(detected), "wrong_identity_alerts": wrong,
            "crowd_alerts": false_alarm, "crowd_passes_alerted": len(crowd_alerted),
            "unattributed_alerts": unattributed, "latency_s": latency}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--groundtruth", type=Path, default=None)
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="per sequence, for a quick test")
    ap.add_argument("--n-enrol", type=int, default=None)
    ap.add_argument("--out", default="cctv_chokepoint_video.json")
    args = ap.parse_args()

    cfg = default_config()
    if args.device:
        cfg.backbone.device = args.device
    n_enrol = args.n_enrol or cfg.eval.n_enrol
    size = cfg.data.image_size
    t0 = time.time()

    gt = cr.read_groundtruth(args.groundtruth or args.root / "groundtruth")
    dirs = {d.name: d for d in args.root.rglob("*")
            if d.is_dir() and cr.SEQ_RE.match(d.name)}
    # Only the official protocol sequences (G1 and G2) are ever scored, so
    # only those are scanned - a third of the footage.
    protocol = set(cr.GROUPS["G1"]) | set(cr.GROUPS["G2"])
    available = [s for s in gt if s in dirs and cr.base_seq(s) in protocol]
    if not available:
        log(f"no frame sequences under {args.root} - extract P1E_S1.tar.xz etc.")
        return 1
    log(f"{len(available)} sequences on disk: {', '.join(available)}")

    backbone = get_backbone(cfg)
    detector = get_detector(cfg)
    adapter = load_adapter(cfg, backbone)

    # Enrolment material: ground-truth face crops from every available
    # sequence, embedded raw (the gallery is never adapted).
    log("embedding enrolment crops")
    enrol_faces = cr.load_frames(args.root, {s: gt[s] for s in available},
                                 size, step=5)
    enrol_raw = backbone.embed(np.stack([f.image for f in enrol_faces]),
                               batch_size=cfg.backbone.batch_size, progress=False)

    scans = {}
    for s in available:
        log(f"scanning {s}")
        scans[s] = scan_sequence(dirs[s], gt[s], cfg, detector, backbone,
                                 args.max_frames)
        log(f"  {scans[s]['frames_processed']} frames, {len(scans[s]['recs'])} "
            f"gated faces, {sum(r['pid'] is not None for r in scans[s]['recs'])} "
            f"attributed")

    people = sorted({f.pid for f in enrol_faces})
    arms = {"frozen_backbone": None}
    if adapter is not None:
        arms["federated_adapter"] = adapter
    results = {}
    for arm, ad in arms.items():
        tot = defaultdict(float)
        lat = []
        for split in range(args.splits):
            rng = np.random.default_rng(split)
            watch = set(rng.choice(people, size=len(people) // 2, replace=False))
            for enrol_g, probe_g in (("G1", "G2"), ("G2", "G1")):
                enrol_seqs = set(cr.GROUPS[enrol_g])
                rows = defaultdict(list)
                for i, f in enumerate(enrol_faces):
                    if f.seq in enrol_seqs and f.pid in watch:
                        rows[f.pid].append(i)
                g, gid = [], []
                for pid, r in sorted(rows.items()):
                    if len(r) >= n_enrol:
                        pick = [r[int(j)] for j in np.linspace(0, len(r) - 1, n_enrol)]
                        g.append(l2_normalise(enrol_raw[pick].mean(0, keepdims=True))[0])
                        gid.append(pid)
                if not gid:
                    continue
                gallery, gallery_ids = np.stack(g), np.array(gid)
                for s in available:
                    if cr.base_seq(s) not in cr.GROUPS[probe_g]:
                        continue
                    sc = scans[s]
                    emb = sc["emb"] if ad is None or not len(sc["emb"]) \
                        else ad.transform_numpy(sc["emb"])
                    alerts = replay(sc, emb, gallery, gallery_ids, cfg)
                    r = score_alerts(alerts, sc["people"], set(gid), sc["first_seen"])
                    lat += r.pop("latency_s")
                    for k, v in r.items():
                        tot[k] += v
        if not tot:
            log("no usable split: need sequences from both groups G1 and G2")
            return 1
        results[arm] = {
            "pass_detection_rate": tot["detected"] / max(1, tot["mate_passes"]),
            "crowd_false_alarm_rate": tot["crowd_passes_alerted"] / max(1, tot["crowd_passes"]),
            "wrong_identity_alerts": tot["wrong_identity_alerts"],
            "unattributed_alerts": tot["unattributed_alerts"],
            "median_time_to_alert_s": float(np.median(lat)) if lat else None,
            "totals": dict(tot),
        }

    out = {"dataset": "ChokePoint", "sequences": available,
           "splits": args.splits, "n_enrol": n_enrol,
           "threshold": cfg.pipeline.threshold(),
           "vote": f"{cfg.pipeline.track_vote_k}/{cfg.pipeline.track_vote_n}",
           "sample_fps": cfg.pipeline.sample_fps, "device": cfg.backbone.device,
           "max_frames": args.max_frames,
           "funnel": {s: {k: v for k, v in sc.items()
                          if k in ("frames_processed", "detections", "gate")}
                      for s, sc in scans.items()},
           "elapsed_sec": round(time.time() - t0, 1), "arms": results}
    path = cfg.paths.results / args.out
    path.write_text(json.dumps(out, indent=1))
    log(f"wrote {path}")
    for arm, r in results.items():
        log(f"{arm:18s} detected {r['pass_detection_rate']:.1%} of enrolled "
            f"walk-throughs; crowd false alarms {r['crowd_false_alarm_rate']:.1%}; "
            f"wrong-identity {int(r['wrong_identity_alerts'])}; "
            f"median time to alert {r['median_time_to_alert_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
