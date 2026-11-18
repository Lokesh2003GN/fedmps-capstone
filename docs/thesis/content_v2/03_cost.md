# CHAPTER 3 | Cost Analysis

This chapter estimated the cost of building and evaluating FedMPS and the communication cost of its federated training. FedMPS was a software project, so its costs were cloud GPU time, cloud storage and data transfer, software, datasets and network traffic, and no hardware was bought for it. Section 3.1 gave the development cost from the logged GPU runs and the published price of the instance used, Section 3.2 gave the network traffic of one federated training run and compared it with federating the whole backbone, and Section 3.3 gave the operational cost of running the pipeline at a node.

## 3.1 Development Cost

All software used was open source, and all datasets were free for non-commercial research, so the only paid resource was AWS. Every GPU run of the project was recorded in its logs and result files. The earlier runs on CelebA, on the alignment A/B test and on the original VGGFace2 took 26.6, 33.9 and 63.3 minutes in September 2026, and the first ChokePoint benchmark took 29.7 minutes in October 2026. The final VGGFace2-HQ session of 25 October 2026 took 271 minutes of GPU compute, made up of 109 minutes of preparation in three parts, of which the first two were stopped to fix speed and memory use, 19 minutes for the pilot, 25 minutes for a first 200-round run that was stopped because a full epoch per round overfitted, 45 minutes for a second run that failed in the DP sweep, 9 minutes for a test of the corrected privacy experiments and 64 minutes for the final 200-round run with all experiments. The ChokePoint benchmark with the latest adapter on 27 October 2026 took 26 minutes of GPU compute, after two earlier attempts that failed because of software versions on the instance and were discarded. All runs used `g4dn.2xlarge` instances, and {tab:cost} priced them at that instance's published on-demand rate in the Mumbai region, US$ 0.828 per hour, with ₹85 assumed per US dollar, which gave the compute cost of {eq:cost}.

$$ Cost_GPU = 7.51 h × US$ 0.828 / h ≈ US$ 6.22 ≈ ₹528 $$ {#eq:cost}

In {eq:cost}, 7.51 hours was the total GPU compute time of all logged runs and US$ 0.828 was the hourly on-demand price of the instance, so the product was the cost of the compute that the experiments actually used. Billed time was somewhat higher than compute time, because the instances also ran while software was installed and the datasets were downloaded. On 25 October the instance ran for about 7.1 hours in all, about US$ 5.9, of which about 1.1 hours went on downloading and unpacking the 628,699 images, and the instance was terminated after the run, with only the 2.6 GB of embeddings kept in S3 at about US$ 0.06 per month. Spot instances would have lowered the hourly rate, but either way the order of magnitude was clear, because the backbone was frozen and its embeddings were computed once, so the whole project needed only a few hours of a single mid-range GPU.

Table: Estimated project cost {#tab:cost}

| Item | Basis | Cost in US$ | Cost in ₹ |
|---|---|---|---|
| AWS GPU compute | 7.51 h at US$ 0.828 per hour | 6.22 | 528 |
| AWS disk, data transfer and storage | Disk for the sessions, results copied out, embeddings kept in S3 | 0.96 | 82 |
| Software: Python, PyTorch, ONNX Runtime, InsightFace, OpenCV, Streamlit, SQLite | Open source | 0 | 0 |
| Datasets: VGGFace2, VGGFace2-HQ, CelebA, LFW, ChokePoint | Free for non-commercial research | 0 | 0 |
| Development laptop | Already owned by the team | 0 | 0 |
| Total | | 7.18 | 610 |

## 3.2 Communication Cost of Federated Training

Freezing the backbone and federating only the adapter kept the network traffic small. One complete training run had 5 nodes and 200 rounds, with one upload and one download per node per round, and each update of the adapter was 1,053,696 bytes, so the total traffic of a run was given by {eq:traffic}.

$$ Traffic = 5 × 200 × 2 × 1,053,696 bytes ≈ 2.11 GB $$ {#eq:traffic}

In {eq:traffic}, the factor 5 was the number of nodes, 200 was the number of rounds, 2 counted the upload of each node's masked update and the download of the new global adapter, and 1,053,696 bytes was the size of one adapter update. Federating the whole ArcFace backbone instead would have sent 174.4 MB in each of the same 2,000 transfers, 348.8 GB per run, so the adapter design was 165 times cheaper in traffic. Spread over 200 rounds, each node sent and received about 2.1 MB per round, which an ordinary office or mobile connection handled in seconds. SA added no traffic to the updates, since a masked update was the same size as an unmasked one, and it needed only a one-time key exchange between each pair of nodes. Distributing a case cost one 64-byte template per node, plus the 2 KB enrolled embedding in cosine match mode.

## 3.3 Operational Cost per Node

On the CPU of the team laptop, the operational pipeline processed about one to two FPS, depending on the number of faces in view, and the node-0 event video of Section 2.3.12, with 437 sampled frames and 1,892 detected faces, took 7 minutes at 1.0 FPS. That rate sufficed for scanning recorded footage after a report, which was the demonstrated use case, but not for live monitoring of a busy camera. On the AWS GPU, the whole ChokePoint video benchmark of 24 sequences and 14,751 frames ran in 5.6 minutes, so real-time operation at each site would have needed a small GPU or an edge accelerator, which was the only hardware cost a deployment would have added.
