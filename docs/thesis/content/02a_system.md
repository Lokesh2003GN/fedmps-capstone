# CHAPTER 2 | Proposed System: FedMPS

This chapter describes the proposed system, FedMPS. Section 2.1 gives the overall architecture. Section 2.2 explains the working methodology step by step: data preparation, the model, federated training, the privacy mechanisms, the evaluation protocol and the operational application. Section 2.3 lists the standards followed, and Section 2.4 gives the software and hardware details.

## 2.1 Proposed System

FedMPS consists of five camera **nodes** and one central **server** ({fig:workflow}). Each node stands for a camera site, such as a railway station or a bus terminal. It owns its video footage, its operator, and the faces of the people its camera has seen. The server coordinates the search and the federated training, but it never holds training data, raw video or the faces of uninvolved people.

The system has two phases.

- **Training phase.** The nodes improve a shared model together. A frozen, pretrained ArcFace network turns every face into a 512-dimensional embedding, and a small *adapter* on top of it is trained by federated learning. In each round, every node trains the adapter on its own data and sends only the change in the adapter's weights. The server combines these changes (optionally under secure aggregation and differential privacy) and sends the new adapter back.
- **Operational phase.** A missing-person case is opened at the server. The server reads the reference photograph once, in memory, and issues each node its own keyed watchlist template. Each node scans its own footage: it detects, tracks and quality-checks faces, embeds them with the backbone and the trained adapter, compares them with the watchlist, and votes across the frames of each track. A match becomes a *pending* alert that only that node's operator can see. A face crop reaches the server only after the operator confirms the alert, and every step is written to an audit log.

![Workflow of FedMPS: training phase and operational phase](docs/thesis/figures/workflow.png){#fig:workflow w=6.0}

{tab:flows} summarises what crosses the network between a node and the server, and what never leaves the node.

Table: Data that crosses the network, and data that stays at a node {#tab:flows}

| Direction | What is sent | Size | When |
|---|---|---|---|
| Node → server | change in adapter weights (optionally masked and noised) | 0.26 MB per round | training |
| Server → node | updated global adapter | 0.26 MB per round | training |
| Server → node | keyed cancelable template of the missing person (plus the enrolled embedding in cosine match mode) | 64 bytes (+2 KB) | case opened |
| Node → server | one face crop and its score | one image | only after a human confirms |
| Stays at node | raw video, all other face crops, embeddings of passers-by, training data, labels | — | never sent |

## 2.2 Working Methodology

The methodology has six parts: (i) preparing a realistic dataset and splitting it across nodes; (ii) the recognition model, a frozen backbone with a federated adapter; (iii) federated training with early stopping; (iv) the privacy mechanisms; (v) the evaluation protocol; and (vi) the operational pipeline and consoles. Each is described below.

### 2.2.1 Dataset and Identity Partition

All reported results use the training split of **VGGFace2** [@vggface2], a large face dataset with wide variation in pose, age, illumination and image quality. From each of its 8,631 training identities, 20 images were taken, giving 172,620 images ({tab:data}). The project's first experiments used CelebA [@celeba] and LFW [@lfw]. VGGFace2 replaced them because it is large enough for a watchlist of thousands of people, and because its loosely framed faces exposed an alignment defect that CelebA's pre-aligned crops had hidden (Section 2.2.3).

Table: Dataset statistics: published VGGFace2 and the subset used {#tab:data}

| Statistic | VGGFace2 (original) | This project |
|---|---|---|
| Subjects | 9,131 | 8,631 (training split) |
| Images | 3.31 million | 172,620 |
| Images per subject (min / avg / max) | 80 / 362.6 / 843 | 20 / 20 / 20 |
| Landmark alignment | 5 landmarks provided | re-aligned with SCRFD `det_10g`; 14 of 172,620 missed (0.01%) |

The data is split **by identity, not by image**, so that no person used for evaluation is ever seen in training, in any form:

- **70% of identities** (6,042 people, 120,840 images) are held out for evaluation and never used for training. Half of them form the *watchlist* (the missing persons to search for); the other half form the *crowd* of people who are on no watchlist.
- **30% of identities** (2,589 people) are distributed across the five nodes. No person appears at more than one node. The nodes are deliberately unequal in size, with the largest about three times the smallest, as real camera sites would be.
- At each node, 20% of its identities are held back from training as **validation identities**. They are used only for early stopping (Section 2.2.7).

{tab:nodes} and {fig:split} give the resulting split.

Table: Identities and images per federated node {#tab:nodes}

| Node | Identities | Training images | Validation images | Total images |
|---|---|---|---|---|
| Node 0 | 258 | 4,120 | 1,040 | 5,160 |
| Node 1 | 388 | 6,200 | 1,560 | 7,760 |
| Node 2 | 517 | 8,280 | 2,060 | 10,340 |
| Node 3 | 647 | 10,360 | 2,580 | 12,940 |
| Node 4 | 779 | 12,460 | 3,120 | 15,580 |
| Total | 2,589 | 41,420 | 10,360 | 51,780 |

![Node split of VGGFace2 and an extract of the pipeline log](figures/fig02_node_split.png){#fig:split w=5.2}

### 2.2.2 CCTV Degradation Model

Real CCTV footage with identity labels at this scale is not publicly available, so for training and the main evaluation the effect of a surveillance camera is simulated. A separate test on real CCTV video is reported in Section 4.9. Every image is passed through a degradation chain that follows the physical order of image formation:

1. **Motion blur** at capture resolution (blur kernel 1–9 pixels at a random angle);
2. **Down-scaling** to the effective face size of a distant subject (20–48 pixels);
3. **Gamma shift** for poor lighting and automatic exposure (γ from 0.7 to 1.6);
4. **Sensor noise**, Gaussian with standard deviation up to 12 grey levels;
5. **JPEG compression** in the camera or recorder (quality 25–70);
6. **Up-scaling** back to the working size, which restores the array size but not the lost detail.

Each node draws one fixed *camera profile* from these ranges, and every image at that node is degraded around that profile with a small per-image jitter (25% of each range). Node 1's camera therefore consistently renders faces at about 24 pixels and JPEG quality 40, while node 3's renders them at 44 pixels and quality 57 ({fig:crops}). This makes the federated data genuinely *non-IID*: each node faces a different, systematic distortion, which is the setting in which federated learning is expected to help.

![Aligned crops as the backbone sees them: (a) enrolment photographs and their CCTV-degraded probes; (b) one identity through each node's camera profile](figures/fig03_aligned_crops.png){#fig:crops w=4.9}

The degradation is severe. Measured on the prepared embeddings, the average cosine similarity between two images of the same person falls from 0.514 (both clean) to 0.219 (clean against degraded). Between different people it stays near zero (0.004 and 0.022).

### 2.2.3 Face Alignment and the Frozen Backbone

Faces are embedded with **ArcFace** [@arcface], using the `buffalo_l` model package from InsightFace [@insightface]. Its recognition network, `w600k_r50`, is a ResNet-50 with 43.6 million parameters that produces a 512-dimensional, L2-normalised embedding. The network is used through ONNX Runtime [@onnxruntime] and is **never trained** in this project. Training a face-recognition backbone needs millions of images and a classifier over all identities, which no single camera site has.

ArcFace is trained only on faces warped onto a fixed five-landmark template, so every face must be aligned the same way. Before degradation, each image passes through the SCRFD face detector [@scrfd] (`det_10g`, 320 × 320 input, detection threshold 0.3). The detector finds the five facial landmarks (eyes, nose tip, mouth corners), and a similarity transform warps the face onto ArcFace's 112 × 112 template. Clean and degraded copies are then cut from the same aligned face, so they differ only in pixel quality.

Alignment was the single largest improvement in the project. In a controlled A/B test on the same 5,000 identities, with the same seed and degradation, alignment raised the average same-person cosine from 0.292 to 0.514 and lowered the cross-person cosine from 0.103 to 0.004. Rank-1 accuracy at the 3 × 3 operating point rose from 0.206 to 0.985.

For every image, two embeddings are stored: one of the clean image and one of its degraded copy. Data preparation ran on an AWS GPU instance (Section 2.4.2) and took 38 minutes for all 172,620 images.

### 2.2.4 The Federated Adapter

Only a small **residual adapter** is federated. It takes the 512-dimensional embedding of a degraded face and nudges it towards where the same person's clean embedding lies:

$$ a(x) = normalise( x + W₂ · Dropout( GELU( LayerNorm( W₁ x ) ) ) ) $$

Here W₁ maps 512 → 64 dimensions, W₂ maps 64 → 512, and the output is renormalised to unit length so that cosine comparison still applies. {tab:adapter} lists its properties.

Table: Properties of the federated adapter {#tab:adapter}

| Property | Value |
|---|---|
| Structure | residual MLP, 512 → 64 → 512, LayerNorm, GELU, dropout 0.1 |
| Parameters | 66,240 |
| Size of one update | 0.26 MB (264,960 bytes) |
| Size if the backbone were federated | 174.4 MB (43.6 M parameters) |
| Reduction per round | 658× |
| Initialisation | last layer set to zero, so the untrained adapter is exactly the identity |

The last layer starts at zero, so an untrained adapter passes embeddings through unchanged. Round 0 of federated training is therefore exactly the frozen backbone, and any change after it is due to the adapter. The hidden size of 64 was chosen by a regularisation sweep: with 256 hidden units (263,000 parameters) the adapter overfitted the few hundred people at each node, reaching its best held-out accuracy at round 1 and then declining.
