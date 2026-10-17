# CHAPTER 1 | Introduction

Thousands of people are reported missing every year, and the first hours after a disappearance are the most valuable for finding them. Cities already operate large networks of closed-circuit television (CCTV) cameras in railway stations, bus terminals, markets and streets, and a person who walks past several of these cameras leaves a trail that could, in principle, be searched automatically. Modern deep face-recognition models can compare a photograph of a missing person against faces seen by a camera in milliseconds, so the technical ingredients of an automated search already exist.

The obstacle is not accuracy alone; it is privacy. The straightforward way to search many cameras is to collect all footage, or all detected faces, on one central server and run face recognition there. That design copies the faces of every bystander who passed any camera into one database. It creates a single point of failure whose compromise would expose the movements of an entire city, and it conflicts with data-protection principles such as data minimisation and purpose limitation. Camera operators belonging to different organisations are also often unwilling, or not permitted, to hand over their footage.

This project, **FedMPS (Federated Missing-Person Search)**, asks whether camera sites can cooperate on a missing-person search, and even improve a shared recognition model, *without* sharing raw video, face crops or unprotected biometric data. It combines three ideas. *Federated learning* trains a shared model by exchanging only model updates. *Privacy mechanisms* (secure aggregation, differential privacy and cancelable templates) protect those updates and the watchlist. And a *human-in-the-loop* review requires a person to confirm every match before anything leaves a camera site. The system is implemented end to end, from dataset preparation on an AWS GPU instance to a working Streamlit application, and every claim made about it is measured.

## 1.1 Problem Statement

Given a photograph of a missing person and a set of camera sites (nodes), each holding its own video footage, the system must:

- flag the camera frames in which the missing person appears, at a false-alarm rate low enough that a human operator can review every alert;
- keep each site's raw video, the faces of uninvolved people, and the site's training data on that site;
- avoid sending the missing person's photograph, or a reusable biometric template of it, to the sites in a form that could be leaked, linked across sites, or reused for another purpose;
- allow the sites to improve a shared recognition model for their own camera conditions without pooling data;
- record who saw what, and who decided what, so that every disclosure can be audited.

Two properties make the problem hard. First, CCTV faces are small, blurred, compressed and badly lit, so recognition accuracy drops sharply compared with clean photographs. Second, the task is *open-set*: almost everyone a camera sees is on no watchlist at all, and the system must stay silent for them. Accuracy on a closed set of enrolled people, which most published work reports, says little about whether such a system is usable.

## 1.2 Objectives

The objectives of this project are:

1. To build a multi-camera missing-person search pipeline (face detection, tracking, quality gating, five-point landmark alignment and ArcFace embeddings) whose recognition model is trained by federated learning across camera nodes.
2. To implement pairwise-masked secure aggregation, differential privacy on model updates and cancelable watchlist templates, and to measure the accuracy cost of each.

## 1.3 Literature Survey

The literature survey for this project reviewed sixteen recent works on missing-person identification, privacy-preserving biometrics and federated face recognition. They fall into four groups.

**Missing-person identification and template protection.** Rakshika et al. [@lit1] detect people with YOLO and match faces with ArcFace or FaceNet embeddings, reaching 97.50% accuracy on 780 synthetic group images; their system decides frame by frame, and they name video tracking as the open problem. Hayat et al. [@lit2] (DeepShieldFed) protect InsightFace features as 96-bit keyed binary templates trained with federated averaging and differential privacy (99.75% on CelebA). Krishna Prakasha and Sumalatha [@lit3] survey biometric privacy techniques and find that each trades security against accuracy or latency.

**Federated face recognition.** Aldahiri et al. [@lit4] combine split federated learning with partial secure multi-party computation (94.68% accuracy; membership-inference AUC 0.507–0.532). Nusa et al. [@lit6] find FedProx better than FedAvg on non-IID LFW, with accuracy falling as DP noise grows. Zheng et al. [@lit7] replace the server with a blockchain (92.22% against 94.67% centralised), and Solomon et al. [@lit10] show that secure aggregation raises the equal error rate only slightly (1.66% to about 2.8%).

**Other facial tasks and de-identification.** Federated learning has also been applied to micro-expressions [@lit5], facial expressions [@lit8], [@lit14], factory attendance [@lit9], edge video sensing [@lit11] and blockchain-audited systems [@lit12], [@lit13]. Chriskos et al. [@lit15] study face de-identification, and Kutlugün et al. [@lit16] data augmentation for small face datasets. {tab:lit} summarises all sixteen works.

Table: Summary of the literature survey {#tab:lit}

| Ref. | Approach | Key result | Gap noted |
|---|---|---|---|
| [@lit1] | YOLO + ArcFace/FaceNet, cosine matching | 97.50% accuracy, near-zero false positives | frame-based; no video tracking |
| [@lit2] | Cancelable 96-bit templates + FedAvg + DP | 99.75% (CelebA), 99.65% (LFW) | personalised FL; verifiable aggregation |
| [@lit3] | Survey of biometric privacy techniques | privacy–accuracy–latency trade-offs | real-time, edge-scale methods |
| [@lit4] | Split vertical FL + partial SMPC + LDP | 94.68% accuracy; MIA AUC 0.507–0.532 | larger datasets; adaptive DP |
| [@lit5] | FL + homomorphic encryption (micro-expressions) | close to centralised accuracy | advanced encryption, DP |
| [@lit6] | FedAvg vs FedProx on non-IID LFW, with DP | FedProx best (89.65%) | client drift under DP |
| [@lit7] | Blockchain-based FL (BFLFace) | 92.22% vs 94.67% centralised | throughput, incentives |
| [@lit8] | FL for facial expressions (Flower, FedAvg) | 72.54% test accuracy | multimodal FL |
| [@lit9] | FL for factory face attendance | 92% accuracy | full automation |
| [@lit10] | Federated face verification, secure aggregator | EER 1.66% → ~2.8% with SA | SA bandwidth and latency |
| [@lit11] | Edge federated video sensing, DP + HE | no stream leakage | commercial hardware, dashboards |
| [@lit12] | Blockchain + FL recruitment system | tamper-proof records | edge FPS and accuracy |
| [@lit13] | Committee-consensus auditable FL, dual masking | matches FedAvg accuracy | scalability |
| [@lit14] | Few-shot federated expression recognition | 84.9% on FER-2013 | non-IID data |
| [@lit15] | Face de-identification vs deep CNNs | >90% recognition failure | reversible methods |
| [@lit16] | Data augmentation for small face datasets | 50% → 76% accuracy | architectures, filters |

## 1.4 Technical Background

**Federated learning.** In federated learning [@fedavg], a server repeatedly sends a global model to the participating clients. Each client trains it locally, and the server averages the returned models (FedAvg). Data never leaves the clients. When clients hold very different data (*non-IID*), local models drift apart. FedProx [@fedprox] adds a proximal term to each client's loss that penalises moving far from the global model.

**Face recognition.** ArcFace [@arcface] trains a deep network so that faces map to 512-dimensional unit vectors (embeddings), with an additive angular margin that pulls embeddings of the same person together on the hypersphere. Two faces are compared by the cosine of the angle between their embeddings. VGGFace2 [@vggface2] is a large face dataset of 9,131 identities with wide variation in pose, age and image quality.

**Secure aggregation.** Secure aggregation [@secagg] lets a server learn the sum of the clients' updates without seeing any individual update. In the pairwise-masking construction, every pair of clients agrees a shared secret (for example with Diffie–Hellman key exchange [@dh]) and derives a random mask from it. One client of the pair adds the mask and the other subtracts it, so every mask cancels in the sum.

**Differential privacy.** Differential privacy (DP) [@dpsgd, @dwork] bounds how much any single participant can change the output of a computation. In federated learning, each update is clipped and Gaussian noise is added; the guarantee is a privacy budget ε (smaller is more private), accounted over rounds with Rényi DP [@rdp].

**Cancelable biometrics.** A cancelable biometric template [@ratha] is a deliberately distorted, key-dependent version of a biometric feature. Templates with the same key still match; a leaked template is revoked by issuing a new key, and templates with different keys cannot be linked.

**Open-set identification metrics.** For watchlist search, the standard measures [@frvt, @iso19795] are the *false positive identification rate* (FPIR), the fraction of people not on the watchlist who still raise an alert, and the *true positive identification rate* (TPIR), the fraction of watchlist people who are flagged *and* correctly identified. A system is compared by its TPIR at a fixed FPIR, for example TPIR at 1% FPIR.

## 1.5 Research Gaps and Contributions

The surveyed work leaves five gaps that this project addresses:

1. **Frame-by-frame decisions.** Existing systems [@lit1] decide on single frames; FedMPS alerts only when several frames of one track agree.
2. **Closed-set accuracy.** Most results [@lit1, @lit2, @lit4] are closed-set accuracies; FedMPS is evaluated as an open-set search with 18,126 non-watchlist probes.
3. **Unmeasured privacy costs.** FedMPS measures each privacy mechanism against the same training run without it.
4. **Missing comparison bounds.** FedMPS compares federated training with local-only, centralised and frozen-model baselines on identical data.
5. **No human oversight.** FedMPS puts human review and an audit log before any disclosure.

## 1.6 Organization of the Report

Chapter 2 describes the proposed system, Chapter 3 gives the cost analysis, Chapter 4 presents and discusses the results, and Chapter 5 concludes the report. Chapter 6 is the appendix with key code listings, and Chapter 7 lists the references.
