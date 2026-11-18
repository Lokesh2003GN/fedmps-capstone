# CHAPTER 2 | Proposed System: FedMPS

This chapter describes the proposed system, FedMPS. Section 2.1 gives the overall architecture. Section 2.2 explains the working methodology step by step: data preparation, the model, federated training, the privacy mechanisms, the evaluation protocol and the operational application. Section 2.3 lists the standards followed, and Section 2.4 gives the software and hardware details.

## 2.1 Proposed System

FedMPS consists of five camera **nodes** and one central **server**, as shown in {fig:workflow}. Each node stands for a camera site, such as a railway station or a bus terminal. It owns its video footage, its operator and the faces of the people its camera has seen. The server coordinates the search and the federated training, but it never holds training data, raw video or the faces of uninvolved people. The system has two phases.

- **Training phase.** The nodes improve a shared model together. A frozen, pretrained ArcFace network turns every face into a 512-dimensional embedding, and a small *adapter* on top of it is trained by FL. In each round, every node trains the adapter on its own data and sends only the change in the adapter's weights. The server combines these changes, optionally under SA and DP, and sends the new adapter back.
- **Operational phase.** A missing-person case is opened at the server. The server reads the reference photograph once, in memory, and issues each node its own keyed watchlist template. Each node scans its own footage: it detects, tracks and quality-checks faces, embeds them with the backbone and the trained adapter, compares them with the watchlist and votes across the frames of each track. A match becomes a *pending* alert that only that node's operator can see. A face crop reaches the server only after the operator confirms the alert, and every step is written to an audit log.

![Workflow of FedMPS in the training phase and the operational phase](docs/thesis/figures/workflow.png){#fig:workflow w=6.0}

{tab:flows} summarises what crosses the network between a node and the server, and what never leaves the node.

Table: Data that crosses the network, and data that stays at a node {#tab:flows}

| Direction | What is sent | Size | When |
|---|---|---|---|
| Node → server | Change in adapter weights, optionally masked and noised | 1.05 MB per round | Training |
| Server → node | Updated global adapter | 1.05 MB per round | Training |
| Server → node | Keyed cancelable template of the missing person, plus the enrolled embedding in cosine match mode | 64 bytes + 2 KB | Case opened |
| Node → server | One face crop and its score | One image | Only after a human confirms |
| Stays at node | Raw video, all other face crops, embeddings of passers-by, training data, labels | — | Never sent |

## 2.2 Working Methodology

The methodology has six parts. The first prepares a realistic dataset and splits it across the nodes, and the second defines the recognition model, a frozen backbone with a federated adapter. The third is federated training for a fixed 200 rounds, and the fourth is the set of privacy mechanisms. The fifth is the evaluation protocol, and the sixth is the operational pipeline with its consoles. Each part is described below.

### 2.2.1 Dataset and Identity Partition

All reported results use **VGGFace2-HQ** [@vgghq], a high-resolution version of VGGFace2 [@vggface2]. VGGFace2 is a large face dataset with wide variation in pose, age, illumination and image quality. In VGGFace2-HQ every face has been restored with the Generative Facial Prior GAN (GFPGAN) [@gfpgan] and aligned to the 512 × 512 Flickr-Faces-HQ (FFHQ) layout. The identity labels are those of VGGFace2. The dataset was downloaded directly onto the AWS training instance from Kaggle, and **every image of every identity was used**: 628,699 images of 4,605 people, as listed in {tab:data}. The project's first experiments used CelebA and LFW, and a later run used a 20-image-per-person subset of the original VGGFace2. VGGFace2-HQ replaced them for two reasons. It gives a median of 128 images per person instead of 20, so each node trains on many views of each of its people. It also starts from sharp faces, so the CCTV degradation of Section 2.2.2 alone sets the probe quality, rather than an unknown mix of the original web-image blur and the simulated camera.

Table: Dataset statistics of the published VGGFace2 and the VGGFace2-HQ set used {#tab:data}

| Statistic | Original VGGFace2 | This project (VGGFace2-HQ) |
|---|---|---|
| Subjects | 9,131 | 4,605, every subject in VGGFace2-HQ |
| Images | 3.31 million | 628,699, every image in VGGFace2-HQ |
| Images per subject, min / avg / max | 80 / 362.6 / 843 | 9 / 136.5 / 414, median 128 |
| Image source | Web images at their original resolution | GFPGAN-restored, FFHQ-aligned, 512 × 512 |
| Landmark alignment | 5 landmarks provided | Re-aligned with SCRFD `det_10g`; 520 of 628,699 missed, which is 0.08% |

The data is split **by identity, not by image**, so that no person used for evaluation is ever seen in training, in any form. The split follows a 70/30 ratio:

- **70% of identities**, which is 3,223 people and 438,053 images, are distributed across the five nodes and used for training. No person appears at more than one node. The nodes are deliberately unequal in size, with the largest about three times the smallest, as real camera sites would be.
- **30% of identities**, which is 1,382 people and 190,646 images, are held out for evaluation and never used for training. Half of them, 691 people, form the *watchlist* of missing persons to search for. The other 691 form the *crowd* of people who are on no watchlist.
- No validation identities are taken from the nodes. Training runs for a fixed 200 rounds and the last round is reported, so the nodes keep all their images for training. The settings were chosen beforehand by a pilot run on training identities only, as Section 2.2.7 explains.

{tab:nodes} and {fig:split} give the resulting split.

Table: Identities and images per federated node {#tab:nodes}

| Node | Identities | Training images | Images per identity |
|---|---|---|---|
| Node 0 | 322 | 42,678 | 132.5 |
| Node 1 | 483 | 67,577 | 139.9 |
| Node 2 | 644 | 90,004 | 139.8 |
| Node 3 | 805 | 106,409 | 132.2 |
| Node 4 | 969 | 131,385 | 135.6 |
| Total at the nodes | 3,223 | 438,053 | 135.9 |
| Held out for evaluation | 1,382 | 190,646 | 137.9 |

![Node split of VGGFace2-HQ, with an extract of the pipeline log below the chart](docs/thesis/figures/node_split_stacked.png){#fig:split w=5.6}

### 2.2.2 CCTV Degradation Model

Real CCTV footage with identity labels at this scale is not publicly available, so for training and the main evaluation the effect of a surveillance camera is simulated. A separate test on real CCTV video is reported in Section 4.9. Every image is passed through a degradation chain that follows the physical order of image formation:

1. **Motion blur** at capture resolution, with a blur kernel of 1–9 pixels at a random angle.
2. **Down-scaling** to the effective face size of a distant subject, which is 20–48 pixels.
3. **Gamma shift** for poor lighting and automatic exposure, with γ from 0.7 to 1.6.
4. **Sensor noise**, Gaussian with a standard deviation of up to 12 grey levels.
5. **Joint Photographic Experts Group (JPEG) compression** in the camera or recorder, at a quality of 25–70.
6. **Up-scaling** back to the working size, which restores the array size but not the lost detail.

Each node draws one fixed *camera profile* from these ranges, and every image at that node is degraded around that profile with a small per-image jitter of 25% of each range. Node 0's camera therefore consistently renders faces at about 21 pixels, with heavy motion blur and JPEG quality 50, while node 4's camera renders them at 47 pixels and quality 66, as {fig:crops} shows. The held-out images are spread evenly over the five profiles, about 38,130 images each, so the evaluation covers every camera. This makes the federated data genuinely non-IID. Each node faces a different, systematic distortion, which is the setting in which FL is expected to help. The degradation is severe. Measured on the prepared embeddings, the average cosine similarity between two images of the same person falls from 0.509 when both are clean to 0.264 when a clean image is compared with a degraded one. Between different people it stays near zero, at 0.006 and 0.015 respectively.

![Aligned crops as the backbone sees them. The upper panel shows enrolment photographs above their CCTV-degraded probes, and the lower panel shows one identity through each node's camera profile](docs/thesis/figures/aligned_crops_vgghq.png){#fig:crops w=6.3}

### 2.2.3 Face Alignment and the Frozen Backbone

Faces are embedded with **ArcFace**, using the `buffalo_l` model package from InsightFace [@arcface]. Its recognition network, `w600k_r50`, is a ResNet-50 with 43.6 million parameters that produces a 512-dimensional, L2-normalised embedding. The network is run through ONNX Runtime and is **never trained** in this project. Training a face-recognition backbone needs millions of images and a classifier over all identities, which no single camera site has. ArcFace is trained only on faces warped onto a fixed five-landmark template, so every face must be aligned the same way. Before degradation, each image therefore passes through the SCRFD detector `det_10g`, with a 320 × 320 input and a detection threshold of 0.3. The detector finds the five facial landmarks, namely the two eyes, the nose tip and the two mouth corners, and a similarity transform warps the face onto the 112 × 112 template of ArcFace. Clean and degraded copies are then cut from the same aligned face, so they differ only in pixel quality.

Alignment was the single largest improvement in the project. It was measured in a controlled A/B test during the earlier run on the original VGGFace2, on the same 5,000 identities. Both arms of the test used the same seed and the same degradation. Alignment raised the average same-person cosine similarity from 0.292 to 0.514 and lowered the cross-person cosine similarity from 0.103 to 0.004. Rank-1 accuracy at the 3 × 3 operating point rose from 0.206 to 0.985. Alignment is therefore applied to every image in this run.

For every image, two embeddings are stored: one of the clean image and one of its degraded copy. Data preparation ran on an AWS GPU instance, described in Section 2.4.2. At 628,699 images the crops alone would need about 47 GB of memory, so preparation was rewritten to *stream*: images are decoded, aligned, degraded and embedded in bounded chunks, and the results are written straight to disk. With the detector and the recognition network sharing one GPU context, the instance processed 121 images per second, and preparation took 84 minutes for all 628,699 images.

### 2.2.4 The Federated Adapter

Only a small **residual adapter** is federated. It takes the 512-dimensional embedding x of a degraded face and nudges it towards where the same person's clean embedding lies:

$$ a(x) = normalise( x + W₂ · Dropout( GELU( LayerNorm( W₁ x ) ) ) ) $$ {#eq:adapter}

In {eq:adapter}, W₁ maps 512 to 256 dimensions, W₂ maps 256 back to 512, GELU is the Gaussian Error Linear Unit (GELU) activation, and the output is renormalised to unit length so that cosine comparison still applies. {tab:adapter} lists the properties of this Multi-Layer Perceptron (MLP).

Table: Properties of the federated adapter {#tab:adapter}

| Property | Value |
|---|---|
| Structure | Residual MLP, 512 → 256 → 512, LayerNorm, GELU, dropout 0.1 |
| Parameters | 263,424 |
| Size of one update | 1.05 MB, or 1,053,696 bytes |
| Size if the backbone were federated | 174.4 MB for 43.6 million parameters |
| Reduction per round | 165 times smaller: 174.4 MB ÷ 1.054 MB |
| Initialisation | Last layer set to zero, so the untrained adapter is exactly the identity |

The last layer starts at zero, so an untrained adapter passes embeddings through unchanged. Round 0 of federated training is therefore exactly the frozen backbone, and any change after it is due to the adapter. The hidden size of 256 was chosen by the pilot run of Section 2.2.7. In the earlier run, with only 16 training images per person, a 256-unit adapter overfitted and a 64-unit one had to be used. With about 136 images per person the larger adapter no longer overfits, and in the pilot it gave the best open-set accuracy of the candidates tried. A second kind of candidate, which adapts the backbone's own last layers rather than its output, was also tried in the pilot and was not better, as Section 4.2 reports.
