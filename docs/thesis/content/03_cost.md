# CHAPTER 3 | Cost Analysis

This chapter estimates the cost of building and evaluating FedMPS, and the communication cost of its federated training. FedMPS is a software project, so the costs are cloud GPU time, storage, software, datasets and network traffic.

## 3.1 Development Cost

All software used is open source, and all datasets are free for non-commercial research. The only paid resource was AWS. {tab:gpu} lists every GPU run recorded in the project's logs and result files.

Table: GPU runs on AWS {#tab:gpu}

| Run | Record | Date | Duration |
|---|---|---|---|
| CelebA data preparation and experiments | `prep_celeba.log`, `experiments_gpu.log` | 13 Sep 2026 | 26.6 min |
| VGGFace2 alignment A/B test | `prep_vgg_alignment_ab.log` | 15 Sep 2026 | 33.9 min |
| VGGFace2 full run: preparation + all experiments | `full_run_vgg_big.log` | 15 Sep 2026 | 63.3 min |
| ChokePoint real-CCTV benchmark, three runs | `cctv_chokepoint_*.json` | 15 Oct 2026 | 29.7 min |
| VGGFace2-HQ preparation, in three parts; the first two were stopped to fix speed and memory use, and the third resumed from their output | `prep_vgghq.log`, `_part2`, `_part3` | 25 Oct 2026 | 109 min |
| Pilot for model selection | `pilot_vgghq.log` | 25 Oct 2026 | 19 min |
| First 200-round run, stopped because a full epoch per round overfitted | `experiments_vgghq_stopped_overfit.log` | 25 Oct 2026 | 25 min |
| Second 200-round run, which failed in the DP sweep | Log not kept; replaced by the re-run | 25 Oct 2026 | 45 min |
| Test of the corrected privacy experiments | `sidetest_vgghq.log` | 25 Oct 2026 | 9 min |
| Final 200-round run and all experiments; the base experiments ran alongside an earlier run | `experiments_vgghq.log`, `experiments_vgghq_base.log` | 25 Oct 2026 | 64 min |
| Total GPU compute | | | 424.5 min = 7.08 h |

All runs used `g4dn.2xlarge` instances, with one NVIDIA T4 GPU and 32 GB of RAM. {tab:cost} prices the compute time at that instance's published on-demand rate in the Mumbai region, US$ 0.828 per hour, with ₹85 per US dollar assumed. The compute cost is therefore:

$$ Cost_GPU = 7.08 h × US$ 0.828 / h ≈ US$ 5.86 ≈ ₹498 $$ {#eq:cost}

Table: Estimated project cost {#tab:cost}

| Item | Basis | Cost in US$ | Cost in ₹ |
|---|---|---|---|
| AWS GPU compute | 7.08 h × US$ 0.828 / h | 5.86 | 498 |
| AWS disk, data transfer and storage, 25 Oct session | 200 GB disk for the day, results copied out, embeddings kept in S3 | 0.96 | 82 |
| Software: Python, PyTorch, ONNX Runtime, InsightFace, OpenCV, Streamlit, SQLite | Open source | 0 | 0 |
| Datasets: VGGFace2, VGGFace2-HQ, CelebA, LFW, ChokePoint | Free for non-commercial research | 0 | 0 |
| Development laptop | Already owned by the team | 0 | 0 |
| Total | | 6.82 | 580 |

{eq:cost} is an estimate. Billed time is somewhat higher than compute time, because the instance also ran while software was installed and the datasets were downloaded. On 25 October the instance ran for about 7.1 hours in all, about US$ 5.9, of which about 1.1 hours went on downloading and unpacking the 628,699 images. The instance was terminated after the run; only the 2.6 GB of embeddings were kept in S3, at about US$ 0.06 per month. Spot instances would lower the hourly rate. Either way, the order of magnitude is clear. Because the backbone is frozen and its embeddings are computed once, the whole project needed about seven hours of a single mid-range GPU.

## 3.2 Communication Cost of Federated Training

Freezing the backbone and federating only the adapter keeps network traffic small. One complete training run has 5 nodes and 200 rounds, with one upload and one download per node per round. Each update of the adapter is 1,053,696 bytes, so the total traffic of a run is:

$$ Traffic = 5 × 200 × 2 × 1,053,696 bytes ≈ 2.11 GB $$ {#eq:traffic}

{tab:comm} compares this with a design that federates the whole backbone.

Table: Network traffic of one federated training run {#tab:comm}

| Design | Size of one update | Updates per run | Total traffic per run |
|---|---|---|---|
| FedMPS: federate the adapter only | 1.05 MB | 5 × 200 × 2 = 2,000 | 2.11 GB |
| Federate the whole ArcFace backbone | 174.4 MB | 2,000 | 348.8 GB |
| Reduction: adapter compared with whole backbone | 165 times smaller | | 165 times smaller |

Spread over 200 rounds, each node sends and receives about 2.1 MB per round, which an ordinary office or mobile connection handles in seconds. SA adds no traffic to the updates, since a masked update is the same size as an unmasked one, and needs only a one-time key exchange between each pair of nodes. Distributing a case costs one 64-byte template per node, plus the 2 KB enrolled embedding in cosine match mode.

## 3.3 Operational Cost per Node

On the CPU of the team laptop, the operational pipeline processed about 1–2.3 FPS, depending on the number of faces in view. That suffices for scanning recorded footage after a report, which is the demonstrated use case, but not for live monitoring of a busy camera. On the AWS GPU, the whole ChokePoint video benchmark of 24 sequences and 14,751 frames ran in 5.9 minutes, which is about 42 FPS. Real-time operation at each site would therefore need a small GPU or an edge accelerator.
