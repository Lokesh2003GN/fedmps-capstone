### 2.2.14 Operational Pipeline at a Node

When a node scans its camera footage, every sampled frame passes through the stages of {fig:pipeline}.

![Operational pipeline at a node, from camera frame to pending alert](docs/thesis/figures/node_pipeline.png){#fig:pipeline w=6.0}

- **Sampling, detection and tracking.** Frames are sampled at up to 8 Frames Per Second (FPS). Faces and their five landmarks are found by the SCRFD detector `det_10g` and linked across frames into one *track* per person by ByteTrack, through the supervision library.
- **Quality gate.** A face is discarded if its eyes are less than 24 pixels apart, if it is blurred, if it is badly exposed with a mean brightness outside 25–235, or if it is turned more than 45° away. A tiny or side-on face does not produce a weak match but a *confidently wrong* one, so discarding it removes false alarms cheaply.
- **Embedding.** Surviving faces are aligned and embedded by the frozen backbone and the federated adapter. A node refuses an adapter trained on a different backbone instead of silently applying it. The application and the real-CCTV test of Section 4.9 use the adapter of the earlier VGGFace2 run; the adapter of the VGGFace2-HQ run is saved separately and gives a similarly small change over the frozen backbone, as Chapter 4 shows.
- **Matching.** In **template** mode, the live embedding is binarised and keyed like the watchlist template and compared by Hamming similarity against a threshold of 0.2656. In **cosine** mode, the default, it is compared with the enrolled embedding against a threshold of 0.24. Cosine mode is more accurate but needs the enrolled embedding at the node, while template mode uses protected codes only.
- **Track voting.** An alert is raised only when **3 of the last 5 frames** of the same track match the same case, at most once per track and case. This turns tens of frame-level hits into one alert per person, stored as *pending* for the node's own operator together with the face crop, the score and the vote.

### 2.2.15 Server Console, Node Console and Audit Log

The operational workflow is run through two web consoles built with Streamlit. They share one SQLite store holding four tables: cases, templates, alerts and audit. **Server console.** An investigator opens a case by entering the person's name, a case reference, notes and an automatic expiry period, and by uploading a reference photograph. The photograph is read once in memory. The server computes its embedding and issues one keyed template per node. Only a 200 × 200 thumbnail is kept, for the human reviewer, and the full photograph is not stored. The console lists open cases and the templates issued, and it shows confirmed sightings only. A pending alert, which no human has yet checked, is never visible at the server. Closing a case, or letting it expire, revokes its templates at every node.

**Node console.** The operator at a camera site sees the review queue for that node only, as shown in {fig:node}. Each pending alert shows the reference thumbnail beside the face the camera captured, the match score and how many frames of the track agreed. The operator confirms or rejects the alert. Opening the reference thumbnail is itself recorded in the audit log. Only a confirmed alert sends its single face crop upstream. The console also shows the node's watchlist as keyed codes and key fingerprints, never as photographs, together with the counts of confirmed and rejected alerts, the resulting local precision and the node's audit trail.

![Node console during the human review of a pending alert](docs/thesis/figures/node_review.png){#fig:node w=6.0}

Once the operator confirms the alert, it appears at the server as a confirmed sighting with the face crop, the score, the vote and the name of the operator who confirmed it. The server console also reports the precision of reviewed alerts. **Audit log.** Every case opening, template issue, alert, reference view, confirmation, rejection and case closure is appended to the audit log with a timestamp and the acting party, as {fig:audit} shows. Server and node operators can both read it, so oversight runs in both directions.

![Audit log after a complete case, covering creation, template issue, alert, review and confirmation](docs/thesis/figures/audit_log.png){#fig:audit w=6.0}

## 2.3 Standards

{tab:standards} lists the standards and established practices that the design and evaluation of FedMPS follow. They come from the International Organization for Standardization (ISO) and the International Electrotechnical Commission (IEC), the National Institute of Standards and Technology (NIST) Face Recognition Vendor Test (FRVT), the Federal Information Processing Standards (FIPS), the ONNX model format and the Digital Personal Data Protection (DPDP) Act of India. The ONNX format lets the same model files run on a GPU and on a Central Processing Unit (CPU).

Table: Standards followed {#tab:standards}

| Standard | How FedMPS applies it |
|---|---|
| ISO/IEC 19795-1, NIST FRVT 1:N | Open-set metrics, TPIR at fixed FPIR; identity-disjoint, fixed test set |
| ISO/IEC 24745 | Templates that are irreversible, unlinkable through per-node keys and renewable through per-case keys |
| RFC 3526, FIPS 180-4 | 2048-bit Diffie–Hellman key agreement; SHA-256 for mask seeds and template keys |
| ONNX | ArcFace and SCRFD run through ONNX Runtime on GPU and CPU alike |
| DPDP Act, 2023 | Data minimisation by design: footage stays on site, one confirmed crop per alert, case expiry, audited access; this is a design property, not a compliance claim |

## 2.4 System Details

This section describes the software and the hardware on which FedMPS was built and evaluated. The software is organised as one shared library, and the hardware combines a rented cloud GPU for the expensive steps with a laptop for everything else.

### 2.4.1 Software Details

The system is written in Python as one shared library, `fedmps/`, used by training, by the application and by the scripts that produce the figures; {tab:software} lists its components. The library is organised by function into data in `data/`, models in `models/`, FL in `fl/`, privacy in `privacy/`, evaluation in `eval/`, the camera pipeline in `pipeline/` and the store in `store.py`. Every module has a self-check on synthetic data.

Table: Software components {#tab:software}

| Component | Version | Used for |
|---|---|---|
| Python | 3.13 | Implementation language |
| PyTorch | 2.13 | Adapter, local training, aggregation |
| ONNX Runtime | 1.24, GPU build on AWS | Running ArcFace and SCRFD |
| InsightFace | 1.0.1 | `buffalo_l` models: SCRFD `det_10g`, ArcFace `w600k_r50` |
| OpenCV | 5.0 | Image decoding, degradation, video input and output |
| NumPy / scikit-learn | 2.1 / 1.5 | Arrays, metrics |
| supervision | 0.30 | ByteTrack face tracking |
| Streamlit / SQLite | 1.51 / 3.49 | Consoles; cases, templates, alerts, audit log |

### 2.4.2 Hardware Details

The expensive steps were run on an AWS Elastic Compute Cloud (EC2) GPU instance, as listed in {tab:hardware}. These steps were aligning and embedding the face images twice, once clean and once degraded, the federated experiments and the real-CCTV benchmark. For the final VGGFace2-HQ run, the 628,699 images were downloaded straight onto the instance and never passed through the laptop; only the embeddings and the results were copied back, through Amazon Simple Storage Service (S3). The adapter is small, so the whole application runs comfortably on a laptop CPU. The instance's virtual CPUs are counted as vCPU and its memory as Random Access Memory (RAM).

Table: Hardware used {#tab:hardware}

| Machine | Specification | Used for |
|---|---|---|
| AWS EC2 GPU instance, September 2026 | `g4dn.2xlarge`: 1 × NVIDIA T4 GPU with 16 GB, 8 vCPU, 32 GB RAM, Ubuntu Linux | VGGFace2 data preparation in 38 min and experiment suite in 22 min |
| AWS EC2 GPU instance, October 2026 | G4dn family, 1 × NVIDIA T4 GPU, Ubuntu Linux | ChokePoint real-CCTV benchmark, 30 min of GPU compute |
| AWS EC2 GPU instance, 25 October 2026 | `g4dn.2xlarge`: 1 × NVIDIA T4 GPU with 16 GB, 8 vCPU, 32 GB RAM, 225 GB local NVMe disk, Ubuntu Linux | VGGFace2-HQ download, preparation in 84 min, pilot in 19 min, 200-round experiment suite in 64 min |
| Team laptop | AMD Ryzen 5 7520U with 4 cores and 8 threads, 16 GB RAM, Windows 11 | Development, three-way re-run, application demonstration |
