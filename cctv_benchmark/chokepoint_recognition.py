#!/usr/bin/env python3
"""CCTV benchmark, part A: recognition accuracy on real surveillance faces.

Dataset: ChokePoint (NICTA, CC BY-NC 4.0) - real CCTV cameras mounted above
two portals, 29 people walking through, 800x600 at 30 fps, every face
annotated with eye positions. https://zenodo.org/record/815657

Protocol - the same open-set watch-list protocol as the review document,
moved from synthetic degradation onto real footage:

  * Each split puts half the people on the watchlist; the rest are the crowd
    (non-mates) the system must NOT alert on. Repeated over --splits seeds.
  * The dataset's official sequence groups G1 and G2 are used in turn: one
    group supplies the enrolment frames, the other the probes, then swapped.
    Enrolment and probe frames therefore always come from different walks.
  * --n-enrol frames are averaged into one gallery entry per person and
    --n-probe-frames consecutive frames into one probe, as in the main
    evaluation (3 and 3).
  * Arms: the frozen ArcFace backbone, and the same plus the federated
    adapter from checkpoints/adapter_final.pt, applied to probes only as in
    the main evaluation and the live matcher.

Reported per arm (mean +/- std over splits and both directions): rank-1,
TPIR at FPIR 1% and 10%, and TPIR / FPIR at the live system's own cosine
threshold (pipeline.match_threshold), which is what an operator would see.

Inputs, either of:
  --source crops   the official cropped faces (P1E.tar.xz, P1L.tar.xz, ...)
  --source frames  the full video frames (P1E_S1.tar.xz, ...); faces are cut
                   out around the ground-truth eye positions

Usage (from the project root):
    python cctv_benchmark/chokepoint_recognition.py --root <chokepoint folder>
    python cctv_benchmark/chokepoint_recognition.py --root <...> --source frames --device cuda

<chokepoint folder> holds the extracted archives and groundtruth/.
"""
import sys as _sys

for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]   # project root
_sys.path.insert(0, str(ROOT))

from fedmps.config import default_config                     # noqa: E402
from fedmps.contracts import l2_normalise                    # noqa: E402
from fedmps.eval.metrics import open_set_identification      # noqa: E402
from fedmps.models.backbone import get_backbone              # noqa: E402
from fedmps.pipeline.match import load_adapter               # noqa: E402

# Official ChokePoint verification groups (arma.sourceforge.net/chokepoint).
# Portal-2 sequences were recorded in two sessions, stored as <name>.1/.2;
# both sessions belong to the listed sequence.
GROUPS = {
    "G1": ["P1E_S1_C1", "P1E_S2_C2", "P2E_S2_C2", "P2E_S1_C3",
           "P1L_S1_C1", "P1L_S2_C2", "P2L_S2_C2", "P2L_S1_C1"],
    "G2": ["P1E_S3_C3", "P1E_S4_C1", "P2E_S4_C2", "P2E_S3_C1",
           "P1L_S3_C3", "P1L_S4_C1", "P2L_S4_C2", "P2L_S3_C3"],
}
SEQ_RE = re.compile(r"^(P[12][EL]_S\d_C\d)(\.\d)?$")
PID_RE = re.compile(r"^\d{4}$")
IMG_EXT = {".jpg", ".jpeg", ".png", ".pgm", ".bmp"}


def log(msg: str) -> None:
    print(f"[chokepoint] {msg}", flush=True)


@dataclass
class Face:
    seq: str          # e.g. P1E_S1_C1 (session suffix dropped)
    frame: int
    pid: str
    image: np.ndarray  # RGB uint8, data.image_size square


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def base_seq(name: str) -> Optional[str]:
    m = SEQ_RE.match(name)
    return m.group(1) if m else None


def read_groundtruth(gt_dir: Path) -> Dict[str, List[Tuple[int, str, Tuple]]]:
    """xml stem -> [(frame, pid, (lx, ly, rx, ry)), ...]."""
    out: Dict[str, List[Tuple[int, str, Tuple]]] = {}
    for xml in sorted(gt_dir.rglob("*.xml")):
        rows = []
        for fr in ET.parse(xml).getroot().iter("frame"):
            n = int(fr.get("number"))
            for p in fr.iter("person"):
                le, re_ = p.find("leftEye"), p.find("rightEye")
                if le is None or re_ is None:
                    continue
                rows.append((n, p.get("id"), (int(le.get("x")), int(le.get("y")),
                                              int(re_.get("x")), int(re_.get("y")))))
        out[xml.stem] = rows
    return out


def to_rgb_square(img: np.ndarray, size: int) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA
                      if img.shape[0] > size else cv2.INTER_CUBIC)


def load_crops(root: Path, size: int, step: int) -> List[Face]:
    """The official cropped faces. The person id and sequence are read from
    the path components, so the exact folder nesting does not matter."""
    faces: List[Face] = []
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT]
    for p in sorted(files):
        seq = pid = None
        for part in p.parts:
            seq = base_seq(part) or seq
            if PID_RE.match(part):
                pid = part
        digits = re.findall(r"\d+", p.stem)
        if not (seq and pid and digits):
            continue
        frame = int(digits[-1])
        if frame % step:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        # The official crops are cut tight around the eyes (96x96). The
        # backbone's detector cannot find a face that fills the whole image -
        # it missed 32 of 34 in a test - so every embedding would fall back
        # to an unaligned resize. A 25% replicated border fixes that (0 of 34).
        pad = int(0.25 * max(img.shape[:2]))
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        faces.append(Face(seq, frame, pid, to_rgb_square(img, size)))
    return faces


def crop_from_frame(frame_bgr: np.ndarray, eyes: Tuple, size: int) -> np.ndarray:
    """A loose square around the face, from the two annotated eye centres.

    Loose on purpose: the backbone's own detector re-finds the face and its
    five landmarks inside the crop, exactly as it does for VGGFace2."""
    lx, ly, rx, ry = eyes
    iod = max(8.0, float(np.hypot(rx - lx, ry - ly)))
    cx, cy = (lx + rx) / 2.0, (ly + ry) / 2.0 + 0.5 * iod
    half = int(round(2.0 * iod))
    pad = half + 2
    big = cv2.copyMakeBorder(frame_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    x0, y0 = int(round(cx)) - half + pad, int(round(cy)) - half + pad
    return to_rgb_square(big[y0:y0 + 2 * half, x0:x0 + 2 * half], size)


def load_frames(root: Path, gt: Dict[str, List], size: int, step: int) -> List[Face]:
    """Full video frames plus ground-truth eyes."""
    dirs = {d.name: d for d in root.rglob("*") if d.is_dir() and SEQ_RE.match(d.name)}
    faces: List[Face] = []
    for stem, rows in gt.items():
        d = dirs.get(stem)
        if d is None:
            continue
        by_frame: Dict[int, List] = defaultdict(list)
        for n, pid, eyes in rows:
            if n % step == 0:
                by_frame[n].append((pid, eyes))
        for n, people in sorted(by_frame.items()):
            path = next((d / f"{n:08d}{e}" for e in (".jpg", ".png")
                         if (d / f"{n:08d}{e}").exists()), None)
            img = cv2.imread(str(path)) if path else None
            if img is None:
                continue
            for pid, eyes in people:
                faces.append(Face(base_seq(stem), n, pid,
                                  crop_from_frame(img, eyes, size)))
        log(f"  {stem}: {sum(len(v) for v in by_frame.values())} annotated faces")
    return faces


# --------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------

def passes(faces: List[Face], idx: np.ndarray) -> Dict[Tuple[str, str], List[int]]:
    """(sequence, person) -> face indices in frame order: one walk-through."""
    out: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i in idx:
        out[(faces[i].seq, faces[i].pid)].append(int(i))
    for k in out:
        out[k].sort(key=lambda i: faces[i].frame)
    return out


def fuse(emb: np.ndarray, rows: List[int]) -> np.ndarray:
    return l2_normalise(emb[rows].mean(axis=0, keepdims=True))[0]


def run_split(faces, gallery_emb, probe_emb, watch, enrol_seqs, probe_seqs,
              n_enrol, n_probe, threshold, fpir_points):
    """One watch-list split. As in the main evaluation and the live matcher,
    the adapter (if any) is applied to probes only: `gallery_emb` is always
    the raw backbone embedding, `probe_emb` the arm under test."""
    enrol_idx = np.array([i for i, f in enumerate(faces) if f.seq in enrol_seqs])
    probe_idx = np.array([i for i, f in enumerate(faces) if f.seq in probe_seqs])

    # Gallery: n_enrol frames spread evenly over all of a person's enrolment
    # walks, averaged into one entry.
    by_pid: Dict[str, List[int]] = defaultdict(list)
    for (seq, pid), rows in passes(faces, enrol_idx).items():
        by_pid[pid].extend(rows)
    g_emb, g_ids = [], []
    for pid in sorted(watch):
        rows = by_pid.get(pid, [])
        if len(rows) < n_enrol:
            continue
        pick = [rows[int(j)] for j in np.linspace(0, len(rows) - 1, n_enrol)]
        g_emb.append(fuse(gallery_emb, pick))
        g_ids.append(pid)
    if not g_ids:
        return None
    gallery, gallery_ids = np.stack(g_emb), np.array(g_ids)
    enrolled = set(g_ids)

    # Probes: consecutive n_probe-frame windows from every walk-through.
    m_emb, m_ids, n_emb = [], [], []
    for (seq, pid), rows in passes(faces, probe_idx).items():
        for s in range(0, len(rows) - n_probe + 1, n_probe):
            v = fuse(probe_emb, rows[s:s + n_probe])
            if pid in enrolled:
                m_emb.append(v); m_ids.append(pid)
            elif pid not in watch:
                n_emb.append(v)
    if not m_emb or not n_emb:
        return None
    mates, mate_ids, nonmates = np.stack(m_emb), np.array(m_ids), np.stack(n_emb)

    res = open_set_identification(gallery, gallery_ids, mates, mate_ids,
                                  nonmates, fpir_points, cmc_ranks=(1, 5))
    # At the live system's own threshold: what an operator would actually see.
    ms = mates @ gallery.T
    hit = (gallery_ids[ms.argmax(1)] == mate_ids) & (ms.max(1) >= threshold)
    res["tpir@system_threshold"] = float(hit.mean())
    res["fpir@system_threshold"] = float(((nonmates @ gallery.T).max(1) >= threshold).mean())
    res.pop("det_curve", None)
    return res


def summarise(runs: List[Dict]) -> Dict[str, Dict[str, float]]:
    keys = ["rank1", "tpir@fpir=0.1", "tpir@fpir=0.01",
            "tpir@system_threshold", "fpir@system_threshold",
            "mean_mate_score", "mean_nonmate_score"]
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if k in r and np.isfinite(r[k])]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                      "n": len(vals)}
    cmc5 = [r["cmc"]["rank5"] for r in runs if "rank5" in r.get("cmc", {})]
    if cmc5:
        out["rank5"] = {"mean": float(np.mean(cmc5)), "std": float(np.std(cmc5)),
                        "n": len(cmc5)}
    return out


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, required=True,
                    help="folder with the extracted ChokePoint archives")
    ap.add_argument("--groundtruth", type=Path, default=None,
                    help="default: <root>/groundtruth")
    ap.add_argument("--source", choices=["crops", "frames"], default="crops")
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--frame-step", type=int, default=1,
                    help="use every k-th annotated frame (raise on a CPU)")
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--n-enrol", type=int, default=None)
    ap.add_argument("--n-probe-frames", type=int, default=None)
    ap.add_argument("--out", default="cctv_chokepoint_recognition.json")
    args = ap.parse_args()

    cfg = default_config()
    if args.device:
        cfg.backbone.device = args.device
    n_enrol = args.n_enrol or cfg.eval.n_enrol
    n_probe = args.n_probe_frames or cfg.eval.n_probe_frames
    size = cfg.data.image_size
    gt_dir = args.groundtruth or args.root / "groundtruth"

    t0 = time.time()
    if args.source == "crops":
        faces = load_crops(args.root, size, args.frame_step)
    else:
        faces = load_frames(args.root, read_groundtruth(gt_dir), size,
                            args.frame_step)
    # Only the official protocol sequences (G1 and G2) are scored; skip the
    # rest rather than spend time embedding them.
    protocol = set(GROUPS["G1"]) | set(GROUPS["G2"])
    faces = [f for f in faces if f.seq in protocol]
    if not faces:
        log(f"no faces found under {args.root} for --source {args.source}")
        return 1
    seqs = sorted({f.seq for f in faces})
    people = sorted({f.pid for f in faces})
    log(f"{len(faces)} faces, {len(people)} people, sequences: {', '.join(seqs)}")

    backbone = get_backbone(cfg)
    raw = backbone.embed(np.stack([f.image for f in faces]),
                         batch_size=cfg.backbone.batch_size)
    hits = getattr(backbone, "align_hits", 0)
    misses = getattr(backbone, "align_misses", 0)
    miss_rate = misses / max(1, hits + misses)
    log(f"embedded in {time.time() - t0:.0f}s; face alignment missed "
        f"{misses}/{hits + misses} ({miss_rate:.1%})")

    arms = {"frozen_backbone": raw}
    adapter = load_adapter(cfg, backbone)
    if adapter is not None:
        arms["federated_adapter"] = adapter.transform_numpy(raw)

    directions = [(d, "G2" if d == "G1" else "G1") for d in ("G1", "G2")]
    fpir_points = [0.1, 0.01]
    results: Dict[str, Dict] = {}
    for arm, emb in arms.items():
        runs = []
        for split in range(args.splits):
            rng = np.random.default_rng(split)
            watch = set(rng.choice(people, size=len(people) // 2, replace=False))
            for enrol_g, probe_g in directions:
                r = run_split(faces, raw, emb, watch, set(GROUPS[enrol_g]),
                              set(GROUPS[probe_g]), n_enrol, n_probe,
                              cfg.pipeline.match_threshold, fpir_points)
                if r is not None:
                    r.update(split=split, enrol=enrol_g, probe=probe_g)
                    runs.append(r)
        if not runs:
            log("no usable split: the available sequences do not cover both "
                "groups G1 and G2 - download more sequences")
            return 1
        results[arm] = {"summary": summarise(runs), "runs": runs}

    out = {
        "dataset": "ChokePoint",
        "source": args.source,
        "sequences": seqs,
        "n_faces": len(faces),
        "n_people": len(people),
        "frame_step": args.frame_step,
        "n_enrol": n_enrol,
        "n_probe_frames": n_probe,
        "splits": args.splits,
        "system_threshold": cfg.pipeline.match_threshold,
        "alignment_miss_rate": miss_rate,
        "device": cfg.backbone.device,
        "elapsed_sec": round(time.time() - t0, 1),
        "arms": results,
    }
    path = cfg.paths.results / args.out
    path.write_text(json.dumps(out, indent=1))

    log(f"wrote {path}")
    print(f"\n{'arm':20s} {'rank-1':>14s} {'TPIR@FPIR10%':>14s} "
          f"{'TPIR@FPIR1%':>14s} {'TPIR@thr':>14s} {'FPIR@thr':>14s}")
    for arm, r in results.items():
        s = r["summary"]

        def cell(k):
            return (f"{s[k]['mean']:.3f}+/-{s[k]['std']:.3f}"
                    if k in s else "n/a")
        print(f"{arm:20s} {cell('rank1'):>14s} {cell('tpir@fpir=0.1'):>14s} "
              f"{cell('tpir@fpir=0.01'):>14s} {cell('tpir@system_threshold'):>14s} "
              f"{cell('fpir@system_threshold'):>14s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
