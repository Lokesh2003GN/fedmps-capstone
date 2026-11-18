"""Stream a very large folder of faces through align -> degrade -> embed.

`01_prepare_data.py` normally decodes every image into one array, which caps it
at about 170,000 images on a 30 GB machine. Using all of VGGFace2 (about 3.1
million training images) needs the images to pass through in chunks and be
thrown away once their two embeddings exist. This module does that.

For each chunk it, on a thread pool, decodes the files, finds the face with
SCRFD and warps it onto ArcFace's landmark template, then degrades the aligned
crop with the camera of the node that owns the image. The GPU then embeds the
clean and degraded crops, and the two [chunk, 512] blocks are written straight
into memory-mapped .npy files. Nothing larger than one chunk is ever in RAM.

Every step uses the same functions and the same per-image seeds as the
in-memory path, so a streamed run and an in-memory run of the same images give
the same embeddings.

The run can be interrupted and restarted: progress is saved after each chunk.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import Config
from .degrade import degrade_image, resolve_params

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EMBED_DIM = 512
ALIGNED_SIZE = 112

# One unreadable file is copied from its neighbour so row numbers stay aligned
# with labels. More than this fraction means the dataset is damaged.
MAX_UNREADABLE = 0.0005


def _log(msg: str) -> None:
    print(f"[stream] {msg}", flush=True)


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------

@dataclass
class FolderIndex:
    names: List[str]          # identity folder names, one per label
    paths: List[str]          # every image file, grouped by identity
    labels: np.ndarray        # [N] int32
    roots: List[str]

    @property
    def n(self) -> int:
        return len(self.paths)


def _is_image(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMAGE_SUFFIXES


def find_identity_roots(root: Path, max_depth: int = 3) -> List[Path]:
    """Directories whose sub-folders are identities that hold images directly.

    A dataset may be laid out as `root/id/img.jpg`, `root/train/id/img.jpg`,
    or `root/vggface2_train/train/id/img.jpg`; this finds each of them without
    being told. A directory counts when it has at least 20 sub-folders and at
    least 16 of the first 20 contain an image file.
    """
    def looks_like_identities(d: str) -> bool:
        with os.scandir(d) as it:
            kids = [e.path for e in it if e.is_dir()]
        if len(kids) < 20:
            return False
        hits = 0
        for k in sorted(kids)[:20]:
            with os.scandir(k) as it:
                if any(e.is_file() and _is_image(e.name) for e in it):
                    hits += 1
        return hits >= 16

    found: List[Path] = []

    def walk(d: str, depth: int) -> None:
        if looks_like_identities(d):
            found.append(Path(d))
            return
        if depth >= max_depth:
            return
        with os.scandir(d) as it:
            subs = sorted(e.path for e in it if e.is_dir())
        for s in subs:
            walk(s, depth + 1)

    walk(str(root), 0)
    return found


def build_folder_index(roots: Sequence[Path], per_identity: Optional[int],
                       min_images: int, seed: int,
                       max_identities: Optional[int] = None) -> FolderIndex:
    """List the files, cap images per identity, and number the identities."""
    people: List[Tuple[str, str]] = []
    for r in roots:
        with os.scandir(r) as it:
            for e in sorted(it, key=lambda e: e.name):
                if e.is_dir():
                    people.append((e.name, e.path))

    # An identity folder name can repeat across roots (train and test); keep
    # the two apart by prefixing the root's name when it does.
    seen: Dict[str, int] = {}
    for n, _ in people:
        seen[n] = seen.get(n, 0) + 1
    if any(v > 1 for v in seen.values()):
        people = [(f"{Path(p).parent.name}/{n}" if seen[n] > 1 else n, p)
                  for n, p in people]

    rng = np.random.default_rng(seed)
    if max_identities is not None and len(people) > max_identities:
        keep = rng.choice(len(people), size=max_identities, replace=False)
        people = [people[i] for i in sorted(keep)]

    def list_files(p: Tuple[str, str]) -> List[str]:
        with os.scandir(p[1]) as it:
            return sorted(e.path for e in it
                          if e.is_file() and _is_image(e.name))

    _log(f"listing {len(people)} identity folders")
    with ThreadPoolExecutor(max_workers=16) as pool:
        listed = list(pool.map(list_files, people))

    names: List[str] = []
    paths: List[str] = []
    labs: List[int] = []
    for (name, _), files in zip(people, listed):
        if len(files) < min_images:
            continue
        if per_identity and len(files) > per_identity:
            pick = rng.choice(len(files), size=per_identity, replace=False)
            files = [files[i] for i in sorted(pick)]
        ident = len(names)
        names.append(name)
        paths.extend(files)
        labs.extend([ident] * len(files))

    if not paths:
        raise RuntimeError("no identity folder holds enough images")
    return FolderIndex(names, paths, np.asarray(labs, dtype=np.int32),
                       [str(r) for r in roots])


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------

def ensure_fp16_onnx(cfg: Config) -> str:
    """Make a float16 copy of the ArcFace ONNX model next to the original.

    On a T4 the half-precision model embeds about 2.1 times faster. Its
    embeddings match the float32 ones to a cosine of 0.9999 or better (checked
    on 1,024 images), and inputs and outputs stay float32, so nothing else in
    the pipeline changes. Returns the file name to put in
    `cfg.backbone.insightface_onnx`.
    """
    from insightface.utils import storage

    model_dir = storage.ensure_available("models", cfg.backbone.insightface_model)
    src = os.path.join(str(model_dir), cfg.backbone.insightface_onnx)
    name = os.path.splitext(cfg.backbone.insightface_onnx)[0] + "_fp16.onnx"
    dst = os.path.join(str(model_dir), name)
    if not os.path.exists(dst):
        import onnx
        from onnxconverter_common import float16

        _log(f"converting {cfg.backbone.insightface_onnx} to float16")
        model = float16.convert_float_to_float16(onnx.load(src), keep_io_types=True)
        onnx.save(model, dst)
    return name


def _make_detector(cfg: Config, quiet: bool = False):
    from insightface.model_zoo import get_model
    from insightface.utils import storage

    model_dir = storage.ensure_available("models", cfg.backbone.insightface_model)
    det_path = os.path.join(str(model_dir), cfg.backbone.insightface_det)
    want_cuda = cfg.backbone.device == "cuda"
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if want_cuda else ["CPUExecutionProvider"])
    try:
        det = get_model(det_path, providers=providers)
    except TypeError:
        det = get_model(det_path)
    det.prepare(ctx_id=0 if want_cuda else -1,
                input_size=(cfg.backbone.det_size, cfg.backbone.det_size),
                det_thresh=cfg.backbone.det_thresh)
    session = getattr(det, "session", None)
    actual = list(session.get_providers()) if session is not None else []
    if not quiet:
        _log(f"detector {cfg.backbone.insightface_det} "
             f"det_size={cfg.backbone.det_size} providers={actual or 'unknown'}")
    return det


# --------------------------------------------------------------------------
# The streaming pass
# --------------------------------------------------------------------------

def stream_embed_folder(index: FolderIndex, profile_ids: np.ndarray,
                        partition: Dict[str, Any], cfg: Config, backbone,
                        out_dir: Path, chunk: int = 2048, workers: int = 8,
                        embed_batch: int = 256, use_detector: bool = True,
                        resume: bool = True, max_chunks: Optional[int] = None
                        ) -> Dict[str, Any]:
    """Fill embeddings_clean.npy and embeddings_deg.npy in `out_dir`.

    Returns a report with the alignment miss rate, the unreadable-file count
    and the elapsed time. `max_chunks` stops early, for timing a short run.
    """
    n = index.n
    out_dir.mkdir(parents=True, exist_ok=True)
    f_clean, f_deg = out_dir / "embeddings_clean.npy", out_dir / "embeddings_deg.npy"
    f_prog = out_dir / "_stream_progress.json"
    n_chunks = (n + chunk - 1) // chunk

    state = {"next_chunk": 0, "misses": 0, "unreadable": 0, "elapsed": 0.0,
             "n": n, "chunk": chunk}
    if (resume and f_prog.exists() and f_clean.exists() and f_deg.exists()):
        saved = json.loads(f_prog.read_text())
        if saved.get("n") == n and saved.get("chunk") == chunk:
            state = saved
            _log(f"resuming at chunk {state['next_chunk']}/{n_chunks}")
    mode = "r+" if state["next_chunk"] > 0 else "w+"
    emb_clean = np.lib.format.open_memmap(f_clean, mode=mode, dtype=np.float32,
                                          shape=(n, EMBED_DIM))
    emb_deg = np.lib.format.open_memmap(f_deg, mode=mode, dtype=np.float32,
                                        shape=(n, EMBED_DIM))

    # One detector session per worker thread. SCRFD's own Python pre- and
    # post-processing is the slow part (about 3 ms of a 4 ms call), so
    # sharing one session behind a lock capped the whole pipeline at about
    # 225 images per second however many cores were free.
    det_local = threading.local()
    det_logged = {"done": False}

    def get_det():
        d = getattr(det_local, "det", None)
        if d is None:
            d = _make_detector(cfg, quiet=det_logged["done"])
            det_logged["done"] = True
            det_local.det = d
        return d

    det = True if use_detector else None
    size = cfg.data.image_size
    ranges = cfg.degrade.ranges
    profiles = {k: partition["nodes"][f"node_{k}"]["degrade_params"]
                for k in range(partition["n_nodes"])}
    if det is not None:
        from insightface.utils import face_align

    def prepare_one(i: int):
        bgr = cv2.imread(index.paths[i], cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
        miss = False
        if det is not None:
            try:
                boxes, kpss = get_det().detect(bgr, max_num=0, metric="default")
            except Exception:
                boxes, kpss = None, None
            if boxes is None or len(boxes) == 0 or kpss is None or len(kpss) == 0:
                miss = True
                crop = cv2.resize(bgr, (ALIGNED_SIZE, ALIGNED_SIZE),
                                  interpolation=cv2.INTER_AREA)
            else:
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                crop = face_align.norm_crop(bgr, kpss[int(np.argmax(areas))],
                                            image_size=ALIGNED_SIZE)
        else:
            crop = cv2.resize(bgr, (ALIGNED_SIZE, ALIGNED_SIZE),
                              interpolation=cv2.INTER_AREA)
        clean = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rng = np.random.default_rng((cfg.seed * 7919 + i) % (2 ** 63))
        params = resolve_params(profiles[int(profile_ids[i])], ranges, rng,
                                cfg.degrade.per_image_jitter)
        deg = degrade_image(clean, params, cfg.degrade, rng)
        return clean, deg, miss

    pool = ThreadPoolExecutor(max_workers=workers)

    def prepare_chunk(ci: int):
        a, b = ci * chunk, min((ci + 1) * chunk, n)
        res = list(pool.map(prepare_one, range(a, b)))
        good = [r for r in res if r is not None]
        if not good:
            raise RuntimeError(f"chunk {ci}: no readable image")
        unreadable = len(res) - len(good)
        fill = good[0]
        clean = np.empty((b - a, ALIGNED_SIZE, ALIGNED_SIZE, 3), dtype=np.uint8)
        deg = np.empty_like(clean)
        misses = 0
        for j, r in enumerate(res):
            r = fill if r is None else r
            fill = r
            clean[j], deg[j] = r[0], r[1]
            misses += int(r[2])
        return a, b, clean, deg, misses, unreadable

    t_start = time.time()
    done_before = state["next_chunk"]
    last_chunk = n_chunks if max_chunks is None else min(n_chunks, done_before + max_chunks)
    feeder = ThreadPoolExecutor(max_workers=1)
    fut = feeder.submit(prepare_chunk, done_before) if done_before < last_chunk else None

    for ci in range(done_before, last_chunk):
        a, b, clean, deg, misses, unreadable = fut.result()
        if ci + 1 < last_chunk:
            fut = feeder.submit(prepare_chunk, ci + 1)
        labs = index.labels[a:b]
        emb_clean[a:b] = backbone.embed(clean, labs, batch_size=embed_batch,
                                        progress=False)
        emb_deg[a:b] = backbone.embed(deg, labs, batch_size=embed_batch,
                                      progress=False)
        state["misses"] += misses
        state["unreadable"] += unreadable
        state["next_chunk"] = ci + 1
        emb_clean.flush()
        emb_deg.flush()
        f_prog.write_text(json.dumps(state))

        done = (ci + 1 - done_before) * chunk
        rate = done / max(time.time() - t_start, 1e-9)
        left = (n - b) / max(rate, 1e-9)
        if (ci - done_before) % 10 == 0 or ci + 1 == last_chunk:
            _log(f"chunk {ci + 1}/{n_chunks}  {b}/{n} images  {rate:.0f} img/s  "
                 f"eta {left / 60:.0f} min  missed {state['misses']}  "
                 f"unreadable {state['unreadable']}")
        if state["unreadable"] > MAX_UNREADABLE * max(b, 1) and b > 20000:
            raise RuntimeError(
                f"{state['unreadable']} unreadable files in {b}: the dataset "
                f"looks damaged")

    pool.shutdown()
    feeder.shutdown()
    elapsed = time.time() - t_start + state.get("elapsed", 0.0)
    state["elapsed"] = elapsed
    f_prog.write_text(json.dumps(state))
    seen = max(1, min(n, state["next_chunk"] * chunk))
    return {
        "detector": cfg.backbone.insightface_det if det is not None else None,
        "det_size": cfg.backbone.det_size,
        "det_thresh": cfg.backbone.det_thresh,
        "out_size": ALIGNED_SIZE,
        "aligned": seen - state["misses"],
        "missed": state["misses"],
        "miss_rate": state["misses"] / seen,
        "unreadable": state["unreadable"],
        "complete": state["next_chunk"] >= n_chunks,
        "elapsed_sec": round(elapsed, 1),
        "images_per_sec": round(seen / max(elapsed, 1e-9), 1),
    }
