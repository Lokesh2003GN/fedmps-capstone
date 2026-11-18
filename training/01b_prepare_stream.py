"""Phase 1, streaming: the same artefacts as 01_prepare_data.py, for a dataset
too large to hold in memory.

Run from the project root, on the GPU instance:

    python training/01b_prepare_stream.py \
        --folder /home/ubuntu/data/VGGface2_HQ --out-suffix _vgghq \
        --feat-dir /opt/dlami/nvme/fedmps --device cuda

01_prepare_data.py decodes every image into one array before it aligns,
degrades and embeds anything. At 172,620 images that peaked at 20 GB; the full
VGGFace2-HQ set is over 500,000 images and would need about 60 GB. This script
does the same five steps chunk by chunk:

  1. index the folder and split identities: held-out vs. the five nodes
  2. decode, detect and align each face onto ArcFace's template (workers)
  3. degrade the aligned crop with its node's camera (workers)
  4. embed the clean and the degraded crop with the frozen ArcFace (GPU)
  5. write the artefacts, plus meta.json

Step 4 also keeps one more tensor for every degraded crop: the output of the
backbone's last residual stage (512 x 7 x 7), taken just before its final
BatchNorm and fully connected layer. Those last two layers are linear, so a
model trained on these features can adapt the end of the backbone itself
without ever running the network again. The features are large (50 KB per
image in float16), so they go to `--feat-dir`, the instance's local NVMe disk.

Alignment, degradation and the per-image random seeds follow 01_prepare_data.py
exactly; the only difference is that faces are detected on the decoded image
instead of on a 160-pixel copy of it.
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedmps.config import Config, default_config                # noqa: E402
from fedmps.contracts import (                                   # noqa: E402
    F_EMB_CLEAN, F_EMB_DEG, F_IDENTITY_NAMES, F_LABELS, F_META, F_PARTITION,
    F_SAMPLES, assert_embeddings, l2_normalise, save_json, save_npy,
)
from fedmps.data.degrade import degrade_image, resolve_params    # noqa: E402
from fedmps.data.partition import (                              # noqa: E402
    build_partition, describe_partition, profile_for_each_image,
)

F_FEAT_DEG = "features_deg.npy"     # [N, 25088] float16, in --feat-dir
FEAT_TENSOR = "679"                 # last residual stage, before bn2 and fc
FEAT_DIM = 512 * 7 * 7
N_SAMPLES_KEPT = 32
IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def log(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


# --------------------------------------------------------------------------
# 1. Index
# --------------------------------------------------------------------------

def index_folder(root: Path, min_images: int, max_per_id: Optional[int],
                 max_identities: Optional[int], seed: int
                 ) -> Tuple[List[str], np.ndarray, List[str]]:
    """List `identity/image` files. Returns (paths, labels, identity_names).

    Selection happens on the listing, before anything is decoded, the same
    way load_folder does it.
    """
    rng = np.random.default_rng(seed)
    people = sorted(p for p in root.iterdir() if p.is_dir())
    eligible: List[Tuple[str, List[Path]]] = []
    for person in people:
        files = sorted(f for f in person.iterdir()
                       if f.suffix.lower() in IMAGE_EXT)
        if len(files) >= min_images:
            eligible.append((person.name, files))
    if max_identities is not None and len(eligible) > max_identities:
        keep = rng.choice(len(eligible), size=max_identities, replace=False)
        eligible = [eligible[i] for i in sorted(keep)]

    paths: List[str] = []
    labels: List[int] = []
    names: List[str] = []
    for label, (name, files) in enumerate(eligible):
        if max_per_id is not None and len(files) > max_per_id:
            pick = rng.choice(len(files), size=max_per_id, replace=False)
            files = [files[i] for i in sorted(pick)]
        names.append(name)
        paths.extend(str(f) for f in files)
        labels.extend([label] * len(files))
    return paths, np.asarray(labels, dtype=np.int32), names


# --------------------------------------------------------------------------
# 2-3. Align and degrade (worker processes)
# --------------------------------------------------------------------------

_W: Dict[str, Any] = {}


def _init_worker(cfg: Config, profiles: Dict[int, Dict[str, float]],
                 reduce_decode: bool) -> None:
    import os

    from insightface.model_zoo import get_model
    from insightface.utils import storage

    cv2.setNumThreads(1)
    model_dir = storage.ensure_available("models", cfg.backbone.insightface_model)
    det_path = os.path.join(str(model_dir), cfg.backbone.insightface_det)
    cuda = cfg.backbone.device == "cuda"
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda
                 else ["CPUExecutionProvider"])
    det = get_model(det_path, providers=providers)
    det.prepare(ctx_id=0 if cuda else -1,
                input_size=(cfg.backbone.det_size, cfg.backbone.det_size),
                det_thresh=cfg.backbone.det_thresh)
    _W.update(cfg=cfg, det=det, profiles=profiles, reduce=reduce_decode)


def _align_one(bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
    from insightface.utils import face_align

    try:
        boxes, kpss = _W["det"].detect(bgr, max_num=0, metric="default")
    except Exception:
        boxes, kpss = None, None
    if boxes is None or len(boxes) == 0 or kpss is None or len(kpss) == 0:
        return cv2.resize(bgr, (112, 112), interpolation=cv2.INTER_AREA), False
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return face_align.norm_crop(bgr, kpss[int(np.argmax(areas))],
                                image_size=112), True


def _align_degrade(i: int, bgr: np.ndarray, prof: int
                   ) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Align one decoded image and degrade the crop. Returns RGB crops."""
    cfg: Config = _W["cfg"]
    crop, hit = _align_one(bgr)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    # Same per-image seed as degrade_all, so the degradation of image i
    # does not depend on chunking or worker order.
    rng = np.random.default_rng((cfg.seed * 7919 + int(i)) % (2 ** 63))
    params = resolve_params(_W["profiles"][int(prof)], cfg.degrade.ranges,
                            rng, cfg.degrade.per_image_jitter)
    return rgb, degrade_image(rgb, params, cfg.degrade, rng), hit


def _empty_chunk(n: int):
    return (np.empty(n, dtype=np.int64),
            np.zeros((n, 112, 112, 3), dtype=np.uint8),
            np.zeros((n, 112, 112, 3), dtype=np.uint8),
            np.zeros(n, dtype=bool), np.zeros(n, dtype=bool))


def _process(task: Sequence[Tuple[int, str, int]]):
    """Decode, align and degrade one chunk in a worker. RGB crops out."""
    idx, clean, deg, hit, unreadable = _empty_chunk(len(task))
    flag = cv2.IMREAD_REDUCED_COLOR_2 if _W["reduce"] else cv2.IMREAD_COLOR
    for j, (i, path, prof) in enumerate(task):
        idx[j] = i
        bgr = cv2.imread(path, flag)
        if bgr is None:
            unreadable[j] = True
            continue
        clean[j], deg[j], hit[j] = _align_degrade(i, bgr, prof)
    return idx, clean, deg, hit, unreadable


# --gpu-in-main: workers only decode, and the main process runs the detector
# on the same GPU context as ArcFace. Several processes each holding a CUDA
# context make the GPU switch between them, and with a detector issuing many
# small kernels that switching cost most of the time.

def _init_decoder(reduce_decode: bool) -> None:
    cv2.setNumThreads(1)
    _W.update(reduce=reduce_decode)


def _decode(task: Sequence[Tuple[int, str, int]]):
    flag = cv2.IMREAD_REDUCED_COLOR_2 if _W["reduce"] else cv2.IMREAD_COLOR
    return [(i, prof, cv2.imread(path, flag)) for i, path, prof in task]


def _align_chunk(decoded, pool) -> Tuple[np.ndarray, ...]:
    """Align and degrade a decoded chunk on a thread pool in the main process."""
    idx, clean, deg, hit, unreadable = _empty_chunk(len(decoded))
    for j, (i, _prof, bgr) in enumerate(decoded):
        idx[j] = i
        unreadable[j] = bgr is None
    jobs = [(j, pool.submit(_align_degrade, i, bgr, prof))
            for j, (i, prof, bgr) in enumerate(decoded) if bgr is not None]
    for j, fut in jobs:
        clean[j], deg[j], hit[j] = fut.result()
    return idx, clean, deg, hit, unreadable


# --------------------------------------------------------------------------
# 4. Embed (main process, GPU)
# --------------------------------------------------------------------------

class ArcFaceWithFeatures:
    """w600k_r50 with the last residual stage exposed as a second output.

    Pre-processing is read from insightface's own wrapper, so the embedding
    output is identical to `ArcFaceONNX.get_feat`.
    """

    def __init__(self, cfg: Config) -> None:
        import os

        import onnx
        import onnxruntime as ort
        from insightface.model_zoo import get_model
        from insightface.utils import storage

        model_dir = storage.ensure_available("models", cfg.backbone.insightface_model)
        path = os.path.join(str(model_dir), cfg.backbone.insightface_onnx)
        ref = get_model(path, providers=["CPUExecutionProvider"])
        self.mean, self.std = float(ref.input_mean), float(ref.input_std)
        self.input_name = ref.input_name
        self.emb_name = ref.output_names[0]

        model = onnx.shape_inference.infer_shapes(onnx.load(path))
        if not any(o.name == FEAT_TENSOR for o in model.graph.output):
            vi = [v for v in model.graph.value_info if v.name == FEAT_TENSOR]
            if not vi:
                raise RuntimeError(f"tensor {FEAT_TENSOR} not found in {path}")
            model.graph.output.append(vi[0])
        # Shape inference pins the batch dimension of the outputs to 1, which
        # makes onnxruntime warn on every batch. Mark it variable.
        for out in model.graph.output:
            out.type.tensor_type.shape.dim[0].dim_param = "N"
        cuda = cfg.backbone.device == "cuda"
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda
                     else ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(model.SerializeToString(),
                                            providers=providers)
        active = self.session.get_providers()
        log(f"ArcFace {cfg.backbone.insightface_onnx} with {FEAT_TENSOR} "
            f"exposed, providers={active}")
        if cuda and "CUDAExecutionProvider" not in active:
            raise RuntimeError("CUDA requested but onnxruntime is on CPU")

    def __call__(self, rgb: np.ndarray, features: bool
                 ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # get_feat takes BGR and swaps to RGB; these crops are RGB already.
        blob = cv2.dnn.blobFromImages(list(rgb), 1.0 / self.std, (112, 112),
                                      (self.mean,) * 3, swapRB=False)
        names = [self.emb_name, FEAT_TENSOR] if features else [self.emb_name]
        outs = self.session.run(names, {self.input_name: blob})
        emb = l2_normalise(np.asarray(outs[0], dtype=np.float32))
        feat = (outs[1].reshape(len(rgb), FEAT_DIM).astype(np.float16)
                if features else None)
        return emb, feat


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _health(emb_clean: np.ndarray, emb_deg: np.ndarray, labels: np.ndarray,
            rng: np.random.Generator) -> Dict[str, float]:
    spec = importlib.util.spec_from_file_location(
        "_prep", Path(__file__).with_name("01_prepare_data.py"))
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    return {
        "same_identity_clean": prep.mean_same_identity_similarity(emb_clean, labels, rng),
        "same_identity_degraded": prep.mean_same_identity_similarity(emb_deg, labels, rng),
        "cross_identity_clean": prep.mean_cross_identity_similarity(emb_clean, labels, rng),
        "cross_identity_degraded": prep.mean_cross_identity_similarity(emb_deg, labels, rng),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="FedMPS streaming data preparation")
    ap.add_argument("--folder", type=Path, required=True)
    ap.add_argument("--out-suffix", default="_vgghq")
    ap.add_argument("--feat-dir", type=Path, required=True,
                    help="where the [N, 25088] float16 features go (local NVMe)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--held-out-pct", type=float, default=30.0,
                    help="percent of identities held out for evaluation; the "
                         "rest is split across the nodes and trained on")
    ap.add_argument("--min-images", type=int, default=6,
                    help="drop identities with fewer images; 6 is the least "
                         "that can be both enrolled and probed at 3 x 3")
    ap.add_argument("--max-images-per-identity", type=int, default=None)
    ap.add_argument("--max-identities", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--embed-batch", type=int, default=256)
    ap.add_argument("--no-reduce-decode", action="store_true",
                    help="decode at full size; by default JPEGs are decoded at "
                         "half resolution, which is plenty for a 112-pixel crop")
    ap.add_argument("--save-crops", action="store_true",
                    help="also keep the aligned clean and degraded crops in "
                         "--feat-dir, for training inside the backbone later")
    ap.add_argument("--gpu-in-main", action="store_true",
                    help="workers only decode; the detector runs in the main "
                         "process on ArcFace's GPU context, over --threads")
    ap.add_argument("--threads", type=int, default=4,
                    help="--gpu-in-main: threads aligning and degrading")
    ap.add_argument("--resume-from", type=int, default=0,
                    help="rows below this are already written by an earlier "
                         "run with the same arguments; continue from here")
    ap.add_argument("--resume-aligned", type=int, default=None,
                    help="with --resume-from: faces aligned in those rows, "
                         "from the earlier run's last progress line")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = default_config()
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.backbone.device = args.device
    cfg.data.source = "folder"
    cfg.data.folder_path = args.folder
    cfg.data.max_identities = args.max_identities
    cfg.data.max_images_per_identity = args.max_images_per_identity
    cfg.data.folder_min_images_per_person = args.min_images
    cfg.data.image_size = 112
    # Everything not held out is node data and is trained on; no per-image
    # validation slice inside the nodes.
    cfg.partition.server_pct = float(args.held_out_pct)
    cfg.partition.node_train_pct = 100.0 - float(args.held_out_pct)
    cfg.partition.node_val_pct = 0.0
    cfg.partition.resolve()
    cfg.paths.artifacts = cfg.paths.artifacts.parent / f"artifacts{args.out_suffix}"
    cfg.paths.artifacts.mkdir(parents=True, exist_ok=True)
    args.feat_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    paths, labels, names = index_folder(
        args.folder, args.min_images, args.max_images_per_identity,
        args.max_identities, cfg.seed)
    n = len(paths)
    log(f"{n:,} images, {len(names):,} identities "
        f"({np.bincount(labels).min()}-{np.bincount(labels).max()} each, "
        f"median {int(np.median(np.bincount(labels)))})")

    log(f"partitioning across {cfg.partition.n_nodes} nodes, "
        f"{args.held_out_pct:g}% of identities held out")
    partition = build_partition(labels, cfg)
    print(describe_partition(partition))
    profile_ids = profile_for_each_image(partition, n)
    profiles = {k: partition["nodes"][f"node_{k}"]["degrade_params"]
                for k in range(partition["n_nodes"])}

    a = cfg.paths.artifacts
    start = int(args.resume_from)
    if start and start % args.chunk:
        raise SystemExit("--resume-from must be a multiple of --chunk")

    def out(path: Path, dtype, shape):
        # A resumed run reopens the earlier run's arrays and fills in the rest.
        if start:
            arr = np.load(path, mmap_mode="r+")
            if arr.shape != shape or arr.dtype != dtype:
                raise SystemExit(f"{path} is {arr.shape} {arr.dtype}, expected "
                                 f"{shape} {dtype}; resume with the same arguments")
            return arr
        return np.lib.format.open_memmap(path, "w+", dtype, shape)

    emb_clean = out(a / F_EMB_CLEAN, np.float32, (n, 512))
    emb_deg = out(a / F_EMB_DEG, np.float32, (n, 512))
    feat = out(args.feat_dir / F_FEAT_DEG, np.float16, (n, FEAT_DIM))
    crops = None
    if args.save_crops:
        crops = {k: out(args.feat_dir / f"crops_{k}.npy", np.uint8, (n, 112, 112, 3))
                 for k in ("clean", "degraded")}

    rng = np.random.default_rng(cfg.seed)
    keep = set(int(i) for i in rng.choice(n, size=min(N_SAMPLES_KEPT, n),
                                          replace=False))
    samples: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    tasks = [[(i, paths[i], int(profile_ids[i]))
              for i in range(s, min(s + args.chunk, n))]
             for s in range(start, n, args.chunk)]
    embedder = ArcFaceWithFeatures(cfg)
    hits = start if args.resume_aligned is None else int(args.resume_aligned)
    unreadable = 0
    done = start
    if start:
        log(f"resuming at row {start:,} ({hits:,} aligned before it)")
    t_embed = time.time()

    def write(idx, clean, deg, hit, bad) -> None:
        nonlocal hits, unreadable, done
        for s in range(0, len(idx), args.embed_batch):
            sl = slice(s, s + args.embed_batch)
            rows = idx[sl]
            e_c, _ = embedder(clean[sl], features=False)
            e_d, f_d = embedder(deg[sl], features=True)
            emb_clean[rows] = e_c
            emb_deg[rows] = e_d
            feat[rows] = f_d
        if crops is not None:
            crops["clean"][idx] = clean
            crops["degraded"][idx] = deg
        for j, i in enumerate(idx):
            if int(i) in keep:
                samples[int(i)] = (clean[j], deg[j])
        hits += int(hit.sum())
        unreadable += int(bad.sum())
        done += len(idx)
        if (done // args.chunk) % 40 == 0 or done == n:
            rate = (done - start) / max(time.time() - t_embed, 1e-6)
            eta = (n - done) / max(rate, 1e-6) / 60
            log(f"{done:,}/{n:,}  {rate:.0f} img/s  ETA {eta:.0f} min  "
                f"aligned {hits:,}  unreadable {unreadable}")

    def too_many_misses() -> bool:
        if done >= 20000 and done - hits > 0.05 * done:
            log(f"ABORT: {done - hits} of {done} faces not detected, over "
                f"the {100 * cfg.backbone.max_align_miss_rate:.0f}% limit")
            return True
        return False

    def bounded(pool, fn, items, inflight: int):
        """pool.imap with at most `inflight` chunks submitted and unconsumed.

        imap feeds the workers as fast as they take work, so when they are
        faster than the GPU their results pile up in memory; with decoded
        images that filled 30 GB in under ten minutes.
        """
        from collections import deque

        it = iter(items)
        queue = deque(pool.apply_async(fn, (t,)) for t in
                      (x for _, x in zip(range(inflight), it)))
        while queue:
            result = queue.popleft().get()
            nxt = next(it, None)
            if nxt is not None:
                queue.append(pool.apply_async(fn, (nxt,)))
            yield result

    ctx = mp.get_context("spawn")
    inflight = 2 * args.workers
    if not args.gpu_in_main:
        with ctx.Pool(args.workers, initializer=_init_worker,
                      initargs=(cfg, profiles, not args.no_reduce_decode)) as pool:
            for chunk in bounded(pool, _process, tasks, inflight):
                write(*chunk)
                if too_many_misses():
                    pool.terminate()
                    return 2
    else:
        from concurrent.futures import ThreadPoolExecutor

        _init_worker(cfg, profiles, not args.no_reduce_decode)
        aligners = ThreadPoolExecutor(args.threads)
        ahead = ThreadPoolExecutor(1)      # aligns chunk k+1 while k is embedded
        pending = None
        with ctx.Pool(args.workers, initializer=_init_decoder,
                      initargs=(not args.no_reduce_decode,)) as pool:
            for decoded in bounded(pool, _decode, tasks, inflight):
                nxt = ahead.submit(_align_chunk, decoded, aligners)
                if pending is not None:
                    write(*pending.result())
                    if too_many_misses():
                        pool.terminate()
                        return 2
                pending = nxt
            if pending is not None:
                write(*pending.result())
        aligners.shutdown()
        ahead.shutdown()

    if crops is not None:
        # Samples that fell in rows written by an earlier, resumed run.
        for i in keep - set(samples):
            samples[i] = (np.asarray(crops["clean"][i]), np.asarray(crops["degraded"][i]))

    for arr in (emb_clean, emb_deg, feat):
        arr.flush()
    miss_rate = (n - hits) / max(n, 1)
    align_report = {
        "detector": cfg.backbone.insightface_det,
        "det_size": cfg.backbone.det_size,
        "det_thresh": cfg.backbone.det_thresh,
        "out_size": 112, "aligned": hits, "missed": n - hits,
        "unreadable": unreadable, "miss_rate": miss_rate,
        "decode": "half resolution" if not args.no_reduce_decode else "full",
        "resumed_from_row": start,
        "detector_in_main_process": bool(args.gpu_in_main),
    }
    log(f"alignment: {hits:,}/{n:,} ({100 * miss_rate:.2f}% missed)")

    emb_clean_r = np.load(a / F_EMB_CLEAN, mmap_mode="r")
    emb_deg_r = np.load(a / F_EMB_DEG, mmap_mode="r")
    assert_embeddings(np.asarray(emb_clean_r[:1000]), "embeddings_clean")
    assert_embeddings(np.asarray(emb_deg_r[:1000]), "embeddings_deg")
    health = _health(emb_clean_r, emb_deg_r, labels, rng)
    print()
    print("  mean cosine similarity        clean     degraded")
    print(f"    same identity            {health['same_identity_clean']:+7.3f}     "
          f"{health['same_identity_degraded']:+7.3f}")
    print(f"    different identities     {health['cross_identity_clean']:+7.3f}     "
          f"{health['cross_identity_degraded']:+7.3f}")
    print()

    save_npy(a / F_LABELS, labels)
    save_json(a / F_PARTITION, partition)
    save_json(a / F_IDENTITY_NAMES, names)
    order = sorted(samples)
    np.savez_compressed(a / F_SAMPLES,
                        clean=np.stack([samples[i][0] for i in order]),
                        degraded=np.stack([samples[i][1] for i in order]),
                        labels=labels[order], index=np.asarray(order, dtype=np.int32))

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backbone": "insightface",
        "backbone_is_synthetic": False,
        "n_images": int(n),
        "n_identities": int(len(names)),
        "image_size": 112,
        "elapsed_sec": round(time.time() - t0, 1),
        "source_folder": str(args.folder),
        "features": {"path": str(args.feat_dir / F_FEAT_DEG), "tensor": FEAT_TENSOR,
                     "shape": [int(n), FEAT_DIM], "dtype": "float16",
                     "of": "degraded crops"},
        "crops": ({k: str(args.feat_dir / f"crops_{k}.npy") for k in crops}
                  if crops is not None else None),
        "celeba_nonmates": {"used": False},
        "nonmate_profiles": [],
        "alignment": align_report,
        "health": health,
        "config": cfg.to_dict(),
    }
    save_json(a / F_META, meta)
    log(f"wrote artefacts to {a} and features to {args.feat_dir}")
    log(f"done in {meta['elapsed_sec'] / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
