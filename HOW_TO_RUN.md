# FedMPS — how to run it

This guide covers the project's four parts:

1. **Model training**: prepares the VGGFace2 dataset and trains the federated adapter. It ran on an **AWS GPU instance**.
2. **Streamlit application**: the live missing-person demo, with a server console and a node review console.
3. **Report figures**: the graphs and tables in the review documents.
4. **Real CCTV benchmark**: the system tested on real surveillance footage (ChokePoint). It ran on an **AWS GPU instance**.

Run every command **from the project root** (the folder containing this file).

---

## 0. Project map

```
fedmps-capstone/
├── README.md                  project summary and headline results
├── HOW_TO_RUN.md              this guide
├── requirements.txt
│
├── training/                  PART 1 - model training on VGGFace2 (AWS GPU)
│   ├── 01_prepare_data.py       align, degrade and embed the dataset (GPU step)
│   ├── 02_run_experiments.py    federated training + all experiments
│   ├── 03_check_local_only.py   fairness check of the local-only arm
│   └── aws_logs/                logs of the AWS GPU runs
│
├── application/               PART 2 - Streamlit demo
│   ├── server_app.py            central server console   (port 8501)
│   ├── node_app.py              node review console      (port 8502)
│   ├── run_video_scan.py        scan one camera video against the watchlist
│   ├── demo_reset.py            empty the demo before presenting
│   └── tools/make_demo_video.py built the synthetic node videos (one-off)
│
├── report_figures/            PART 3 - figures and table for the review document
│   ├── compute_benchmark_data.py
│   ├── generate_benchmark_table.py
│   ├── make_fig02_node_split.py
│   ├── make_fig05_det_curves.py
│   ├── make_fig06_roc_det_cmc.py
│   └── make_fig07_convergence.py
│
├── cctv_benchmark/            PART 4 - real CCTV benchmark (ChokePoint, AWS GPU)
│   ├── chokepoint_recognition.py  recognition accuracy on real CCTV faces
│   ├── chokepoint_video.py        the live pipeline end to end on CCTV video
│   └── run_on_aws.sh              download, extract and run both on the GPU
│
├── fedmps/                    the shared Python library used by all four parts
│
├── artifacts_vgg_big/         VGGFace2 embeddings produced on AWS (input to training)
├── artifacts/                 demo database (fedmps.sqlite) and template selector
├── checkpoints/               adapter_final.pt - the trained federated adapter
├── results/                   measured results (JSON) and the benchmark table
├── figures/                   the 10 figures of the second review, fig01 ... fig10
├── media/                     demo videos and enrolment photos
└── docs/                      first review report, second review documentation
```

Old runs, backups and unused files were moved out of the project to `Downloads\fedmps-capstone-archive`. That folder's `MOVED_FILES.md` lists where everything went.

### Inside `fedmps/`: which modules belong to which part

| Part | Module | What it does |
|---|---|---|
| **Shared** | `config.py` | Every setting (paths, rounds, thresholds, privacy) in one place |
| | `contracts.py` | File names, array shapes and load/save checks shared by all steps |
| | `models/backbone.py` | Frozen ArcFace (`buffalo_l`) face embedder with 5-point alignment |
| | `models/adapter.py` | The residual adapter (66,240 parameters) that is federated |
| | `privacy/template.py` | Cancelable watchlist templates (512 bits = 64 bytes per node) |
| | `data/degrade.py` | The CCTV degradation model (blur, downscale, noise, JPEG) |
| **Training** | `data/loader.py`, `data/partition.py`, `data/celeba.py` | Read the dataset; split identities across 5 nodes |
| | `fl/node.py`, `fl/server.py`, `fl/loop.py` | Local training, FedProx aggregation, and the three-way comparison |
| | `privacy/dp.py`, `privacy/mask.py` | Differential privacy and secure aggregation |
| | `eval/metrics.py`, `eval/calibrate.py`, `eval/experiments.py` | Metrics, the scoring layer, and all 8 experiments |
| **Application** | `pipeline/detect.py`, `pipeline/quality.py`, `pipeline/match.py` | Detection and tracking, quality gate, matching and track voting |
| | `store.py` | SQLite store for cases, templates, alerts and the audit log |

---

## 1. Setup (laptop, Windows)

Use the **system Python** (3.13). Streamlit is not on PATH on this laptop, so always launch it as `python -m streamlit`.

```bash
pip install -r requirements.txt
pip install insightface onnxruntime
```

The first run downloads the ArcFace model (`buffalo_l`) to `~/.insightface`.

---

## 2. Part 1 — Model training (AWS GPU)

Training ran on an **AWS EC2 GPU instance**: Ubuntu, an **NVIDIA T4**, project at `/home/ubuntu/fedmps`. The logs in `training/aws_logs/` record the backbone running on `CUDAExecutionProvider`.

| Step | Script | Ran on | Time | Output |
|---|---|---|---|---|
| 1. Prepare data | `training/01_prepare_data.py` | AWS, GPU (`--device cuda`) | 38 min, peak RAM 20 GB | `artifacts_vgg_big/` |
| 2. Train and evaluate | `training/02_run_experiments.py --all` | AWS, same instance | 22 min | `results/experiments_vgg_big.json`, `checkpoints/adapter_final.pt`, `figures/training_run/` |
| 2b. Three-way re-run with rank-5/10 | `training/02_run_experiments.py --only three_way` | this laptop, 13 Oct, from the AWS embeddings | 11 min | `results/experiments_vgg_big_threeway.json`, `checkpoints/adapter_final.pt` |

Step 1 is the GPU-heavy step: it detects, aligns and embeds 172,620 face images with ArcFace. Step 2 trains the 66k-parameter adapter on the saved embeddings. The run's config snapshot records `device: cpu` for it, because the adapter is too small to need a GPU.

### 2.1 Set up the AWS instance

```bash
# on your machine: copy the project up, leaving out the generated data
rsync -av --exclude artifacts_vgg_big fedmps-capstone/ ubuntu@<aws-host>:/home/ubuntu/fedmps/

# on the instance
cd /home/ubuntu/fedmps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip uninstall -y torch && pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install insightface onnxruntime-gpu
pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu
```

Two install traps, both found on 15 Oct and both silent until the first inference:

- Plain `pip install torch` now installs a **CUDA 13** build, but `onnxruntime-gpu` is built for **CUDA 12**. They then load clashing cuDNN libraries and every convolution fails with `CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE`. Install the CUDA 12 build of torch (`cu128`), as above.
- `insightface` pulls in the CPU-only `onnxruntime`, which hides the GPU package. Reinstalling `onnxruntime-gpu` alone, as above, fixes it. Check that `python -c "import onnxruntime as o; print(o.get_available_providers())"` lists `CUDAExecutionProvider`.

The dataset is **VGGFace2**, laid out as `identity_name/image.jpg`. It is licence-restricted, so download it separately to the instance, for example to `/home/ubuntu/vggface2`.

### 2.2 Step 1: prepare the data (GPU)

```bash
python training/01_prepare_data.py \
    --source folder --folder /home/ubuntu/vggface2 \
    --backbone insightface --device cuda \
    --max-identities 8631 --max-images-per-identity 20 \
    --out-suffix _vgg_big
```

This writes `artifacts_vgg_big/`:

| File | Contents |
|---|---|
| `embeddings_clean.npy` | ArcFace embedding of each original image, `[172620, 512]` |
| `embeddings_deg.npy` | the same image after the CCTV degradation model |
| `labels.npy` | identity id per image |
| `partition.json` | the 5-node split, held-out identities, per-node camera profiles |
| `identity_names.json`, `samples.npz`, `meta.json` | names, sample crops, run metadata |

### 2.3 Step 2: federated training and all experiments

```bash
python training/02_run_experiments.py --all \
    --artifacts-suffix _vgg_big --out experiments_vgg_big.json
```

It runs 8 experiments: degradation, operating points, three-way (local / federated / centralised), DP sweep, masking, templates, FedProx, and membership inference. It writes:

- `results/experiments_vgg_big.json`: every number, plus a full config snapshot.
- `checkpoints/adapter_final.pt`: the federated adapter, loaded later by the application.
- `figures/training_run/*.png`: the run's own charts. Figure 8 of the document is `fig8_operating_points.png` and Figure 9 is `fig4_dp_sweep.png`.

The three-way comparison behind Tables 5–7 was re-run with rank-5/10 measurement:

```bash
python training/02_run_experiments.py --only three_way \
    --artifacts-suffix _vgg_big --out experiments_vgg_big_threeway.json
```

> **Note:** every three-way run overwrites `checkpoints/adapter_final.pt`. The demo currently uses the adapter from the 13 Oct re-run.

### 2.4 Fairness check of the local-only arm (about 6 min, CPU)

In Table 5, local-only trains 30 epochs and keeps the last one, while federated keeps its best round. This retrains local-only with the same early stopping and proximal penalty:

```bash
python training/03_check_local_only.py --artifacts-suffix _vgg_big
```

It writes `results/local_only_check.json`. Local-only then reaches rank-1 0.9473, the same as federated, and Section 4.3 of the thesis explains what this means.

### 2.5 Bring the results back

```bash
rsync -av ubuntu@<aws-host>:/home/ubuntu/fedmps/artifacts_vgg_big ./
rsync -av ubuntu@<aws-host>:/home/ubuntu/fedmps/results ./
rsync -av ubuntu@<aws-host>:/home/ubuntu/fedmps/checkpoints ./
```

---

## 3. Part 2 — Streamlit application (demo)

The demo runs on the laptop CPU. It needs `checkpoints/adapter_final.pt`, `artifacts/selector.json` and the videos in `media/`.

### 3.1 Before presenting: empty the database

```bash
python application/demo_reset.py
```

This deletes all cases, templates, alerts and audit entries, the annotated videos, the scan statistics and the alert crops.

### 3.2 Terminal 1: server console

```bash
python -m streamlit run application/server_app.py --server.port 8501
```

Open http://localhost:8501. The page stays blank for about 10 seconds while the ArcFace model loads.

### 3.3 Terminal 2: node console

```bash
python -m streamlit run application/node_app.py --server.port 8502
```

Open http://localhost:8502. In the sidebar, **set "Node id" to 2**. "Reviewing as" follows it (`operator@node2`).

### 3.4 Terminal 3: scan a camera video

This is the verified demo pairing. `person-1.jpg` raises an alert at score **0.553**, with 3 of 3 frames agreeing.

```bash
python application/run_video_scan.py --video media/demo_node2_person1.mp4 --node 2 --camera "cam-C (site 2)" --enrol media/person-1.jpg --enrol-name "Priya Sharma" --fps-cap 0 --max-frames 60
```

It takes about a minute on CPU. Expected output:

```
[video] enrolled 'Priya Sharma' as case-..., distributed to 5 node(s)
[match]   ALERT #..: case-... on track 1, score 0.553 (3/3 frames) - PENDING local review
[match] voting: 40 frame hits -> 1 track alerts
```

To show matching on the **protected templates only**, add `--match-mode template`. The same pairing then raises an alert at score 0.387, against a template-mode threshold of 0.2656.

You can also open a case in the server console (upload a photo) and press **Scan video** in the node console's sidebar. That button scans `media/node<id>.mp4` for the selected node.

### 3.5 What the demo shows

1. **Server console.** Open a case. The full photograph is not stored; a 200×200 thumbnail is kept for reviewers. Each node receives its own keyed 64-byte template, a different bit pattern per node. In the default cosine mode, each node also receives the enrolled 512-d embedding.
2. **Scan.** The node scans its own footage. Its raw video, face crops and embeddings stay on the node.
3. **Node console (node 2).** One candidate awaits review: the reference thumbnail beside what the camera saw, the score, and the vote (3 of 3 frames). A human decides.
4. **Confirm.** Only now does a crop reach the server. The server console shows **Confirmed sightings: 1**, and the audit log records who decided and when.

### 3.6 Which photo goes with which video

| Enrolment photo | Video | Result |
|---|---|---|
| `media/person-1.jpg` | `media/demo_node2_person1.mp4` | **alert, score 0.553**: use this for the demo |
| your own photo | `media/node2.mp4` (your own video, 13 Oct) | alert at 0.44 for the "vamsi" case |
| `media/Abdullah_Gul/*.jpg` | `media/node0.mp4` | no alert: a known issue, so don't demo node 0 |

A photo of someone who isn't in the video correctly produces `0 track alerts`.

### 3.7 If something goes wrong

| Symptom | Cause / fix |
|---|---|
| Node console says "Nothing awaiting review" | Node id in the sidebar is not the node you scanned |
| Server console is blank | Still loading the backbone; wait and reload |
| Scan ends with `0 track alerts` | The enrolled photo isn't of anyone in that video (see 3.6) |
| `[match] IGNORING the adapter ...` | `checkpoints/adapter_final.pt` was trained on a different backbone; restore the trained one |
| Many faces rejected as `too_small` | The gate measures inter-ocular distance, not box size; add `--min-face-px 10` |

---

## 4. Part 3 — Report figures and the benchmark table

`figures/` holds exactly the 10 figures of the second review document. The scripts plot only measured values from `results/`, and re-running them reproduces the files byte for byte.

| Figure | File | Made by |
|---|---|---|
| 1 System architecture | `fig01_system_architecture.png` | diagram, no script |
| 2 Node split | `fig02_node_split.png` | `report_figures/make_fig02_node_split.py` |
| 3 Aligned crops | `fig03_aligned_crops.png` | image from the document, no script |
| 4 Rank-1 at each step | `fig04_dataset_progression.png` | image from the document, no script |
| 5 DET curves | `fig05_det_curves.png` | `report_figures/make_fig05_det_curves.py` |
| 6 ROC / DET / CMC | `fig06_roc_det_cmc.png` | `report_figures/make_fig06_roc_det_cmc.py` |
| 7 Convergence | `fig07_convergence.png` | `report_figures/make_fig07_convergence.py` |
| 8 Operating points | `fig08_operating_points.png` | training step 2 |
| 9 DP sweep | `fig09_dp_sweep.png` | training step 2 |
| 10 Timeline | `fig10_timeline.png` | diagram, no script |

```bash
python report_figures/compute_benchmark_data.py --artifacts-suffix _vgg_big
python report_figures/generate_benchmark_table.py
python report_figures/make_fig02_node_split.py
python report_figures/make_fig05_det_curves.py
python report_figures/make_fig06_roc_det_cmc.py
python report_figures/make_fig07_convergence.py
```

`compute_benchmark_data.py` takes a few minutes and writes `results/benchmark_measured.json`. `generate_benchmark_table.py` writes `results/benchmark_comparison_table.md`.

| Result file | Used for |
|---|---|
| `results/experiments_vgg_big.json` | AWS full run: degradation, operating points (Table 8), DP, masking, FedProx, MIA (Table 10) |
| `results/experiments_vgg_big_threeway.json` | three-way re-run: Tables 5, 6, 7, 9 and Figures 5–7 |
| `results/benchmark_measured.json` | clean and degraded curves, CMC, benchmark table |
| `results/local_only_check.json` | local-only with the same early stopping as federated (fairness check) |
| `results/alignment_ab.json` | alignment A/B test (Section 5.1) |
| `results/template_sweep_wide.json` | template bit-width sweep (Table 10, templates row) |
| `results/regularisation_sweep.json` | adapter size and regularisation sweep (why hidden size is 64) |
| `results/video_node2.json` | statistics of the last node-2 demo scan |

---

## 5. Part 4 — Real CCTV benchmark (ChokePoint, AWS GPU)

The review results use synthetically degraded VGGFace2. This part tests the same system on **real surveillance footage**: [ChokePoint](https://zenodo.org/record/815657) (NICTA, CC BY-NC 4.0). Three ceiling cameras above two doorways film 29 people walking through, 800×600 at 30 fps, with eye positions annotated on every face.

**Protocol** (the review's open-set watch-list protocol, moved to real footage):

- In each split half the people are on the watchlist and the rest are the crowd. There are 10 splits.
- Enrolment and probes come from the dataset's official sequence groups G1 and G2, used both ways round, so a person is never matched against the same walk-through they were enrolled from.
- 3 enrolment frames per person and 3-frame probes, as in the review. The adapter is applied to probes only, as in the review and the live matcher.

### 5.1 Run it on AWS

On a g4dn instance set up as in section 2.1, from the project root, inside the virtualenv:

```bash
bash cctv_benchmark/run_on_aws.sh ~/chokepoint
```

The script:

1. downloads the 21 archives (about 12 GB) all at once, because Zenodo throttles each connection to about 200 KB/s but not the total, which takes about 20 minutes;
2. unpacks the face crops into `crops/` and the frames into `frames/`;
3. runs both benchmarks with `--device cuda`.

On the 15 Oct run the GPU steps took about 30 minutes. Copy `results/cctv_chokepoint_*.json` back, then terminate the instance.

> **Note:** don't edit `run_on_aws.sh` while it is running. Bash reads a script as it executes it, so the rest of the run breaks.

### 5.2 Results (15 Oct 2026, g4dn.2xlarge)

**A. Recognition on real CCTV faces** (`cctv_benchmark/chokepoint_recognition.py`), mean over 20 runs:

| Faces | Count | Rank-1 | TPIR @ FPIR 1% | Hits at threshold 0.24 | Crowd false alarms at 0.24 |
|---|---|---|---|---|---|
| Official face crops (greyscale) | 25,331 | 1.000 | 0.999 | 1.000 | 4.4% ± 3.3% |
| Cut from full colour frames | 28,162 | 1.000 | 0.997 | 0.999 | 4.6% ± 3.2% |

The federated adapter gives the same numbers to three decimals, the same finding as the review.

**B. The live pipeline end to end** (`cctv_benchmark/chokepoint_video.py`): detection, tracking, quality gate, matching and 3-of-5 track voting over 24 camera sequences, scored against the ground truth:

| Arm | Enrolled walk-throughs detected | Crowd walk-throughs alerted | Wrong-identity alerts | Median time to alert |
|---|---|---|---|---|
| Frozen backbone | **95.2%** (1,988 of 2,088) | **3.9%** (88 of 2,232) | **0** | 1.3 s |
| + federated adapter | 95.2% | 3.7% | 0 | 1.3 s |

What the numbers say:

- On real footage the system finds 95% of enrolled people within about a second and never names the wrong person.
- The cost is about 4 false alerts per 100 strangers at the current threshold (0.24). Section A shows a higher threshold would keep almost every hit (TPIR 0.997 at FPIR 1%), so the threshold is worth re-tuning on real footage.
- The quality gate passes 54% of detected faces.
- ChokePoint is easier than a real search: indoor and outdoor doorways, people walking towards the camera, and enrolment from the same cameras rather than a family photograph.

Result files: `results/cctv_chokepoint_recognition_crops.json`, `results/cctv_chokepoint_recognition_frames.json`, `results/cctv_chokepoint_video.json`. `results/cctv_chokepoint_recognition_P1E_crops.json` is the earlier laptop run on portal 1 only.

### 5.3 Quick run on the laptop (CPU)

With only the 66 MB `P1E.tar.xz` crop archive and `groundtruth.tar.xz` extracted into a folder:

```bash
python cctv_benchmark/chokepoint_recognition.py --root <folder>/crops --groundtruth <folder>/groundtruth --source crops
```

That covers portal 1 and 25 people, and takes about 30 minutes on this laptop.

---

## 6. Self-checks (no dataset, no GPU, a minute or two)

Every library module checks itself on synthetic data and writes only to a temporary folder.

```bash
python -m fedmps.config
python -m fedmps.contracts
python -m fedmps.data.degrade
python -m fedmps.data.loader
python -m fedmps.data.partition
python -m fedmps.data.celeba
python -m fedmps.models.backbone
python -m fedmps.models.adapter
python -m fedmps.eval.metrics
python -m fedmps.eval.calibrate
python -m fedmps.fl.node
python -m fedmps.fl.server
python -m fedmps.fl.loop
python -m fedmps.privacy.dp
python -m fedmps.privacy.mask
python -m fedmps.privacy.template
python -m fedmps.eval.experiments
python -m fedmps.store
python -m fedmps.pipeline.detect
python -m fedmps.pipeline.quality
python -m fedmps.pipeline.match
```

Each one ends with `<name> OK`.
