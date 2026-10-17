# FedMPS — Privacy-Preserving Federated Learning for Missing Person Identification

VIT-AP University, Capstone Project, Fall 2026–27.
Team: G N Lokesh, R Kumaresh, K Kishore, Aswin Maheswaran. Guide: Kanhaiya Sharma.

FedMPS is a missing-person search system for a network of CCTV cameras. The cameras (nodes) improve a shared model without sending raw video, face crops or unprotected biometric data to a central server.

- A pretrained **ArcFace** face model (43.6 M parameters) is **frozen**.
- Only a small **residual adapter** (66,240 parameters, 0.26 MB) is trained by **federated learning** (FedProx) across 5 nodes. That is 658× less data per round than federating the full model.
- Three privacy mechanisms are implemented and measured: **secure aggregation** (pairwise Diffie–Hellman masks), **differential privacy** on updates, and **cancelable watchlist templates**.
- A **Streamlit** server console and node console run the operational flow. A case goes out as a keyed template, nodes scan their own video, alerts wait for **human review**, and every action is **audited**.

## Where to start

| You want to | Open |
|---|---|
| Read the final thesis | [`docs/FedMPS_Capstone_Thesis.pdf`](docs/FedMPS_Capstone_Thesis.pdf) |
| Run training, the demo or the figures | [`HOW_TO_RUN.md`](HOW_TO_RUN.md) |
| Read the review documents | [`docs/`](docs/) |

## Project structure

```
training/          model training on VGGFace2 (AWS GPU)
application/       Streamlit server and node consoles, video scanning
report_figures/    figures and the benchmark table for the review document
fedmps/            shared library: data, models, federated learning, privacy, evaluation, pipeline
results/           measured results (JSON)
figures/           the 10 figures of the second review
docs/              review documents, thesis format, and the final thesis (sources in docs/thesis/)
cctv_benchmark/    real-CCTV benchmark on ChokePoint (AWS GPU)
```

## Headline results

These come from VGGFace2 with synthetic CCTV degradation: 172,620 images and 8,631 identities, tested on a 3,021-person watchlist with 18,126 non-enrolled probes. Rank-1 and TPIR use 3 enrolment photos × 3 fused frames.

| Arm | Rank-1 | TPIR @ FPIR 1% | TAR @ FAR 0.1% |
|---|---|---|---|
| Clean probes (upper bound) | 0.9962 | 0.9915 | 0.9247 |
| Frozen ArcFace + scoring layer | 0.9474 | 0.7968 | 0.6287 |
| Centralised adapter (pooled data) | 0.9476 | 0.8043 | 0.6087 |
| **Federated adapter (FedMPS)** | **0.9473** | **0.8020** | **0.6193** |
| Local-only adapter (mean of 5 nodes) | 0.9286 | 0.770 | 0.583 |

- **Keeping data at the sites costs no accuracy.** Federated matches centralised (rank-1 gap 0.0003).
- **Naive local fine-tuning hurts; our protocol doesn't.** Local-only training for 30 epochs loses 1.9 rank-1 points. With the same early stopping and proximal penalty as federated, local-only also reaches 0.9473 (`results/local_only_check.json`), so on this data federation gives no accuracy gain over a well-regularised local model.
- **The adapter adds no measurable gain over the frozen model** on this data (−0.0001 rank-1, +0.005 TPIR @ 1%). Early stopping keeps round 1.
- **Secure aggregation costs no accuracy.** Masked and unmasked runs give identical rank-1 (0.9376), with a 1.008× time overhead.
- **Membership inference finds no leakage** (attack AUC 0.496; chance is 0.5).
- **Differential privacy has no usable operating point.** ε = 252 at the only tolerable accuracy.
- **512-bit cancelable templates cost about 9 rank-1 points.**
- **On real CCTV video (ChokePoint), the live pipeline alerted on 95.2% of enrolled walk-throughs**, with a median of 1.3 s and no wrong-name alerts (`results/cctv_chokepoint_video.json`).

Full tables and the discussion of each result are in the thesis, `docs/FedMPS_Capstone_Thesis.pdf`.
