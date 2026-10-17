# CHAPTER 3 | Cost Analysis

This chapter estimates the cost of building and evaluating FedMPS, and the communication cost of its federated training. FedMPS is a software project, so the costs are cloud GPU time, software, datasets and network traffic.

## 3.1 Development Cost

All software used is open source, and all datasets are free for non-commercial research. The only paid resource was GPU time on Amazon Web Services [@aws]. {tab:gpu} lists every GPU run recorded in the project's logs and result files.

Table: GPU runs on AWS {#tab:gpu}

| Run | Record | Date | Duration |
|---|---|---|---|
| CelebA data preparation and experiments | `prep_celeba.log`, `experiments_gpu.log` | 13 Sep 2026 | 26.6 min |
| VGGFace2 alignment A/B test | `prep_vgg_alignment_ab.log` | 15 Sep 2026 | 33.9 min |
| VGGFace2 full run (preparation + all experiments) | `full_run_vgg_big.log` | 15 Sep 2026 | 63.3 min |
| ChokePoint real-CCTV benchmark (three runs) | `cctv_chokepoint_*.json` | 15 Oct 2026 | 29.7 min |
| Total GPU compute | | | 153.5 min (2.56 h) |

The September runs used a `g4dn.2xlarge` instance (one NVIDIA T4 GPU, 32 GB RAM); the full run's memory peaked at 20 GB. {tab:cost} prices the compute time at that instance's published on-demand rate in the Mumbai region, US$ 0.828 per hour [@aws], with ₹85 per US dollar assumed.

Table: Estimated project cost {#tab:cost}

| Item | Basis | Cost (US$) | Cost (₹) |
|---|---|---|---|
| AWS GPU compute | 2.56 h × US$ 0.828 / h | 2.12 | 180 |
| Software (Python, PyTorch, ONNX Runtime, InsightFace, OpenCV, Streamlit, SQLite) | open source | 0 | 0 |
| Datasets (VGGFace2, CelebA, LFW, ChokePoint) | free for non-commercial research | 0 | 0 |
| Development laptop | already owned by the team | 0 | 0 |
| Total (GPU compute time only) | | 2.12 | 180 |

This is an estimate. Billed time is somewhat higher than compute time, because the instance also ran while software was installed and the datasets were downloaded; the October ChokePoint session cost about US$ 1.60 in total including its 12 GB download. Spot instances would lower the hourly rate. Either way, the order of magnitude is clear. Because the backbone is frozen and its embeddings are computed once, the whole evaluation needed under three hours of a single mid-range GPU, and federated training itself runs on a laptop CPU.

## 3.2 Communication Cost of Federated Training

Freezing the backbone and federating only the adapter keeps network traffic small. {tab:comm} compares one complete training run (5 nodes × 30 rounds, one upload and one download per node per round) with a design that federates the whole backbone.

Table: Network traffic of one federated training run {#tab:comm}

| Design | Size of one update | Updates per run | Total traffic per run |
|---|---|---|---|
| FedMPS: federate the adapter only | 0.26 MB | 5 × 30 × 2 = 300 | 79.5 MB |
| Federate the whole ArcFace backbone | 174.4 MB | 300 | 52.3 GB |
| Reduction | | | 658× |

Secure aggregation adds no traffic to the updates, since a masked update is the same size as an unmasked one, and only a one-time key exchange between each pair of nodes. Distributing a case costs one 64-byte template per node, plus the 2 KB enrolled embedding in cosine match mode.

## 3.3 Operational Cost per Node

On the team laptop's CPU, the operational pipeline processed about 1–2.3 frames per second, depending on the number of faces in view. That suffices for scanning recorded footage after a report, the demonstrated use case, but not for live monitoring of a busy camera. On the AWS GPU, the whole ChokePoint video benchmark, 24 sequences and 14,751 frames, ran in 5.9 minutes: about 42 frames per second. Real-time operation at each site would therefore need a small GPU or an edge accelerator.
