# FedMPS — Privacy-Preserving Federated Learning for Missing Person Identification

VIT-AP University, Capstone Project, Fall 2026–27.
Team: G N Lokesh, R Kumaresh, K Kishore, Aswin Maheswaran. Guide: Kanhaiya Sharma.

FedMPS is a missing-person search system for a network of CCTV cameras. The camera sites (nodes) improve a shared model without sending raw video, face crops or unprotected biometric data to a central server.

- A pretrained **ArcFace** face model (ResNet-50, 43.6 M parameters) is **frozen**.
- Only a small **residual adapter** (263,424 parameters, 1.05 MB per update) is trained by **federated learning** (FedProx + FedAvg) across 5 nodes for 200 rounds. That is 165× less data per round than federating the full model.
- Three privacy mechanisms are implemented and measured: **secure aggregation** (pairwise Diffie–Hellman masks), **differential privacy** on updates, and **cancelable watchlist templates** (512-bit keyed codes).
- A **Streamlit** server console and node console run the operational flow. A case goes out as a keyed template, nodes scan their own video, alerts wait for **human review**, and every action is **audited**.

## Where to start

| You want to | Open |
|---|---|
| Read the final thesis | [`docs/presentation/FedMPS_Capstone_Thesis_v3.pdf`](docs/presentation/FedMPS_Capstone_Thesis_v3.pdf) |
| See the final review presentation | [`docs/presentation/FedMPS_Capstone_Final_Review.pptx`](docs/presentation/FedMPS_Capstone_Final_Review.pptx) |
| Run training, the demo or the figures | [`HOW_TO_RUN.md`](HOW_TO_RUN.md) |
| Rebuild the thesis from its sources | `python docs/thesis/build_thesis.py --content content_v2 --out FedMPS_Capstone_Thesis_v3` |

## Project structure

```
training/          data preparation, pilot and federated experiments (AWS GPU); run logs in aws_logs/
application/       Streamlit server and node consoles, video scanning
fedmps/            shared library: data, models, federated learning, privacy, evaluation, pipeline
cctv_benchmark/    real-CCTV benchmark on ChokePoint (AWS GPU)
checkpoints/       trained adapters; adapter_vgghq_h256.pt is the final one
results/           measured results (JSON)
report_figures/    scripts for the result figures and comparison table
figures/           result figures
docs/thesis/       thesis sources (content_v2/), figures and the build script
docs/presentation/ final thesis, review deck, speaker script and their build scripts
```

## Dataset

- **VGGFace2-HQ**: all 628,699 images of 4,605 people, split by identity: 70% (3,223 people) across 5 unequal nodes for training, 30% (1,382 people) held out. VGGFace2 on Kaggle: <https://www.kaggle.com/datasets/hearfool/vggface2>.
- Each node's images are degraded with its own simulated CCTV camera profile (blur, 20–48 px faces, lighting, noise, JPEG).
- **ChokePoint** for testing on real CCTV video (29 people, 24 sequences).
- The datasets and embeddings are not in this repository; `HOW_TO_RUN.md` explains how to prepare them.

## Headline results

Test set: a 691-person watchlist with 30,540 probes, plus 31,850 probes of people not on it. 3 enrolment photos × 3 fused frames, the same scoring for every arm.

| Arm | Rank-1 | TPIR @ FPIR 1% | TAR @ FAR 0.1% | EER |
|---|---|---|---|---|
| Clean probes (upper bound) | 0.9989 | 0.9912 | 0.8903 | 8.3% |
| Frozen ArcFace + scoring layer | 0.9733 | 0.9054 | 0.6820 | 12.1% |
| Centralised adapter (pooled data) | 0.9733 | 0.9080 | 0.6801 | 11.7% |
| **Federated adapter (FedMPS)** | **0.9734** | **0.9107** | **0.6835** | **11.8%** |
| Local-only adapter (mean of 5 nodes) | 0.9620 | 0.8812 | 0.6455 | 12.5% |

- **Keeping data at the sites costs no accuracy.** Federated matches centralised training on pooled data.
- **Training alone hurts.** Every node's own adapter falls below the frozen model; federation is 1.1 rank-1 and 3.0 TPIR points better than training alone.
- **The adapter's gain over the frozen model is small but consistent** (+0.5 TPIR points at 1% FPIR, stable over 200 rounds), so the recommendation is to keep the backbone frozen.
- **Secure aggregation costs no accuracy**, and a membership-inference attack finds no leakage (AUC 0.501; chance is 0.5).
- **512-bit cancelable templates cost 4.7 rank-1 points.** Differential privacy has no usable setting with only five sites.
- **On real CCTV video (ChokePoint), the live pipeline alerted on 95.2% of enrolled walk-throughs**, with a median of 1.3 s and no wrong-name alerts.

Full tables and the discussion of each result are in the thesis.
