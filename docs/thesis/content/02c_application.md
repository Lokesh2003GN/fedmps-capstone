### 2.2.14 Operational Pipeline at a Node

When a node scans its camera footage, every sampled frame passes through the stages of {fig:pipeline}.

![Operational pipeline at a node: from camera frame to pending alert](docs/thesis/figures/node_pipeline.png){#fig:pipeline w=6.2}

- **Sampling, detection and tracking.** Frames are sampled at up to 8 per second. Faces and their five landmarks are found by SCRFD [@scrfd] (`det_10g`) and linked across frames into one *track* per person by ByteTrack [@bytetrack], through the supervision library [@supervision].
- **Quality gate.** A face is discarded if its eyes are less than 24 pixels apart, it is blurred, badly exposed (mean brightness outside 25–235) or turned more than 45° away. A tiny or side-on face does not produce a weak match but a *confidently wrong* one, so discarding it removes false alarms cheaply.
- **Embedding.** Surviving faces are aligned and embedded by the frozen backbone and the federated adapter. A node refuses an adapter trained on a different backbone instead of silently applying it.
- **Matching.** In **template** mode, the live embedding is binarised and keyed like the watchlist template and compared by Hamming similarity (threshold 0.2656). In **cosine** mode, the default, it is compared with the enrolled embedding (threshold 0.24). Cosine mode is more accurate but needs the enrolled embedding at the node; template mode uses protected codes only.
- **Track voting.** An alert is raised only when **3 of the last 5 frames** of the same track match the same case, at most once per track and case. This turns tens of frame-level hits into one alert per person, stored as *pending* for the node's own operator together with the face crop, the score and the vote.

### 2.2.15 Server Console, Node Console and Audit Log

The operational workflow is run through two web consoles built with Streamlit [@streamlit]. They share one SQLite [@sqlite] store holding four tables: cases, templates, alerts and audit.

**Server console.** An investigator opens a case by entering the person's name, a case reference, notes and an automatic expiry period, and uploading a reference photograph. The photograph is read once in memory. The server computes its embedding and issues one keyed template per node. Only a 200 × 200 thumbnail is kept, for the human reviewer; the full photograph is not stored. The console lists open cases and the templates issued, and shows confirmed sightings only: a pending alert, which no human has yet checked, is never visible at the server. Closing a case, or letting it expire, revokes its templates at every node.

**Node console** ({fig:node}). The operator at a camera site sees the review queue for that node only. Each pending alert shows the reference thumbnail beside the face the camera captured, the match score, and how many frames of the track agreed. The operator confirms or rejects it, and opening the reference thumbnail is itself recorded in the audit log. Only a confirmed alert sends its single face crop upstream. The console also shows the node's watchlist (as keyed codes and key fingerprints, not photographs), the counts of confirmed and rejected alerts with the resulting local precision, and the node's audit trail.

![Node console: human review of a pending alert](docs/thesis/figures/node_review.png){#fig:node w=4.4}

Once the operator confirms the alert, it appears at the server as a confirmed sighting with the face crop, the score, the vote and the name of the operator who confirmed it. The server console also reports the precision of reviewed alerts.

**Audit log.** Every case opening, template issue, alert, reference view, confirmation, rejection and case closure is appended to the audit log with a timestamp and the acting party ({fig:audit}). Server and node operators can both read it, so oversight runs in both directions.

![Audit log after a complete case: creation, template issue, alert, review and confirmation](docs/thesis/figures/audit_log.png){#fig:audit w=4.4}

## 2.3 Standards

{tab:standards} lists the standards and established practices that the design and evaluation of FedMPS follow.

Table: Standards followed {#tab:standards}

| Standard | How FedMPS applies it |
|---|---|
| ISO/IEC 19795-1 [@iso19795], NIST FRVT 1:N [@frvt] | open-set metrics (TPIR at fixed FPIR); identity-disjoint, fixed test set |
| ISO/IEC 24745 [@iso24745] | templates that are irreversible, unlinkable (per-node keys) and renewable (per-case keys) |
| RFC 3526 [@rfc3526], FIPS 180-4 [@fips180] | 2048-bit Diffie–Hellman key agreement; SHA-256 for mask seeds and template keys |
| ONNX [@onnx] | ArcFace and SCRFD run through ONNX Runtime [@onnxruntime] on GPU and CPU alike |
| DPDP Act, 2023 [@dpdp] | data minimisation by design: footage stays on site, one confirmed crop per alert, case expiry, audited access (not a compliance claim) |

## 2.4 System Details

### 2.4.1 Software Details

The system is written in Python as one shared library (`fedmps/`) used by training, the application, and the scripts that produce the figures ({tab:software}). The library is organised by function: data (`data/`), models (`models/`), federated learning (`fl/`), privacy (`privacy/`), evaluation (`eval/`), the camera pipeline (`pipeline/`) and the store (`store.py`). Every module has a self-check on synthetic data.

Table: Software components {#tab:software}

| Component | Version | Used for |
|---|---|---|
| Python | 3.13 | implementation language |
| PyTorch [@pytorch] | 2.13 | adapter, local training, aggregation |
| ONNX Runtime [@onnxruntime] | 1.24 (GPU build on AWS) | running ArcFace and SCRFD |
| InsightFace [@insightface] | 1.0.1 | `buffalo_l` models: SCRFD `det_10g`, ArcFace `w600k_r50` |
| OpenCV [@opencv] | 5.0 | image decoding, degradation, video input and output |
| NumPy / scikit-learn | 2.1 / 1.5 | arrays, metrics |
| supervision [@supervision] | 0.30 | ByteTrack face tracking |
| Streamlit [@streamlit] / SQLite [@sqlite] | 1.51 / 3.49 | consoles; cases, templates, alerts, audit log |

### 2.4.2 Hardware Details

The expensive steps were run on a GPU instance on Amazon Web Services [@aws] ({tab:hardware}): aligning and embedding 172,620 face images twice (clean and degraded), and the real-CCTV benchmark. The adapter is small, so federated training on the saved embeddings, and the whole application, run comfortably on a laptop CPU.

Table: Hardware used {#tab:hardware}

| Machine | Specification | Used for |
|---|---|---|
| AWS EC2 GPU instance (Sep 2026) | `g4dn.2xlarge`: 1 × NVIDIA T4 GPU (16 GB), 8 vCPU, 32 GB RAM, Ubuntu Linux | VGGFace2 data preparation (38 min) and experiment suite (22 min) |
| AWS EC2 GPU instance (Oct 2026) | G4dn family, 1 × NVIDIA T4 GPU, Ubuntu Linux | ChokePoint real-CCTV benchmark (30 min of GPU compute) |
| Team laptop | AMD Ryzen 5 7520U (4 cores, 8 threads), 16 GB RAM, Windows 11 | development, three-way re-run, application demonstration |
