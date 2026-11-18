# CHAPTER 1 | Introduction

Thousands of people are reported missing every year, and the first hours after a disappearance are the most valuable for finding them. Cities already operate large networks of CCTV cameras in railway stations, bus terminals, markets and streets. A person who walks past several of these cameras leaves a trail that could, in principle, be searched automatically. Modern deep face-recognition models can compare a photograph of a missing person against the faces seen by a camera in milliseconds, so the technical ingredients of an automated search already exist. The obstacle is not accuracy alone; it is privacy. The straightforward way to search many cameras is to collect all footage, or all detected faces, on one central server and run face recognition there. That design copies the faces of every bystander who passed any camera into one database. It creates a single point of failure whose compromise would expose the movements of an entire city, and it conflicts with data-protection principles such as data minimisation and purpose limitation. Camera operators belonging to different organisations are also often unwilling, or not permitted, to hand over their footage.

This project, **FedMPS**, asks whether camera sites can cooperate on a missing-person search, and even improve a shared recognition model, *without* sharing raw video, face crops or unprotected biometric data. It combines three ideas. FL trains a shared model by exchanging only model updates. Privacy mechanisms such as SA, DP and cancelable templates protect those updates and the watchlist. A *human-in-the-loop* review requires a person to confirm every match before anything leaves a camera site. The system is implemented end to end, from dataset preparation on an Amazon Web Services (AWS) Graphics Processing Unit (GPU) instance to a working Streamlit application, and every claim made about it is measured.

## 1.1 Problem Statement

Given a photograph of a missing person and a set of camera sites, called nodes, each holding its own video footage, the system must:

- Flag the camera frames in which the missing person appears, at a false-alarm rate low enough that a human operator can review every alert.
- Keep each site's raw video, the faces of uninvolved people and the site's training data on that site.
- Avoid sending the missing person's photograph, or a reusable biometric template of it, to the sites in a form that could be leaked, linked across sites or reused for another purpose.
- Allow the sites to improve a shared recognition model for their own camera conditions without pooling data.
- Record who saw what, and who decided what, so that every disclosure can be audited.

Two properties make the problem hard. First, CCTV faces are small, blurred, compressed and badly lit, so recognition accuracy drops sharply compared with clean photographs. Second, the task is *open-set*: almost everyone a camera sees is on no watchlist at all, and the system must stay silent for them. Accuracy on a closed set of enrolled people, which most published work reports, says little about whether such a system is usable.

## 1.2 Objectives

The objectives of this project are:

1. To build a multi-camera missing-person search pipeline, comprising face detection, tracking, quality gating, five-point landmark alignment and ArcFace embeddings, whose recognition model is trained by FL across camera nodes.
2. To implement pairwise-masked SA, DP on model updates and cancelable watchlist templates, and to measure the accuracy cost of each.

## 1.3 Literature Survey

The literature survey for this project reviewed sixteen recent works on missing-person identification, privacy-preserving biometrics and federated face recognition. They fall into three groups. The first group covers missing-person identification and template protection. Rakshika et al. detect people with You Only Look Once (YOLO) and match faces with ArcFace or FaceNet embeddings, reaching 97.50% accuracy on 780 synthetic group images; their system decides frame by frame, and they name video tracking as the open problem [@lit1]. Hayat et al. protect InsightFace features in DeepShieldFed as 96-bit keyed binary templates trained with FL and DP, reaching 99.75% on CelebA [@lit2]. Krishna Prakasha and Sumalatha survey biometric privacy techniques and find that each trades security against accuracy or latency [@lit3]. The second group covers federated face recognition. Aldahiri et al. combine split FL with partial Secure Multi-Party Computation (SMPC) and Local Differential Privacy (LDP), reaching 94.68% accuracy with a Membership Inference Attack (MIA) Area Under the Curve (AUC) of 0.507–0.532 [@lit4]. Nusa et al. find FedProx better than FedAvg (Federated Averaging) on LFW data that is not Independent and Identically Distributed (IID), with accuracy falling as DP noise grows [@lit6]. Zheng et al. replace the server with a blockchain and reach 92.22% against 94.67% for centralised training [@lit7]. Solomon et al. show that SA raises the Equal Error Rate (EER) only slightly, from 1.66% to about 2.8% [@lit10].

The third group applies FL to other facial tasks or studies de-identification. FL has been applied to micro-expression recognition with Homomorphic Encryption (HE) [@lit5]. It has also been applied to facial-expression recognition, including a few-shot variant [@lit8, @lit14]. Other systems use it for factory attendance and for edge video sensing [@lit9, @lit11]. Two further works combine it with blockchain-audited records and committee-based aggregation [@lit12, @lit13]. Chriskos et al. study face de-identification against deep Convolutional Neural Networks (CNNs) [@lit15]. Kutlugün et al. study data augmentation for small face datasets [@lit16]. {tab:lit} summarises all sixteen works, and Section 4.10 compares the results of FedMPS with the VGGFace2 reference models and with five recent federated or privacy-preserving face-recognition systems [@vggface2, @fedhide, @fedfs, @kimteoh, @dpfedface, @ji2022].

Table: Summary of the literature survey {#tab:lit}

| Ref. | Approach | Key result | Gap noted |
|---|---|---|---|
| [@lit1] | YOLO + ArcFace or FaceNet, cosine matching | 97.50% accuracy, near-zero false positives | Frame-based; no video tracking |
| [@lit2] | Cancelable 96-bit templates + FedAvg + DP | 99.75% on CelebA, 99.65% on LFW | Personalised FL; verifiable aggregation |
| [@lit3] | Survey of biometric privacy techniques | Privacy–accuracy–latency trade-offs | Real-time, edge-scale methods |
| [@lit4] | Split vertical FL + partial SMPC + LDP | 94.68% accuracy; MIA AUC 0.507–0.532 | Larger datasets; adaptive DP |
| [@lit5] | FL + HE for micro-expressions | Close to centralised accuracy | Advanced encryption, DP |
| [@lit6] | FedAvg against FedProx on non-IID LFW, with DP | FedProx best at 89.65% | Client drift under DP |
| [@lit7] | Blockchain-based FL, BFLFace | 92.22% against 94.67% centralised | Throughput, incentives |
| [@lit8] | FL for facial expressions with Flower and FedAvg | 72.54% test accuracy | Multimodal FL |
| [@lit9] | FL for factory face attendance | 92% accuracy | Full automation |
| [@lit10] | Federated face verification, secure aggregator | EER 1.66% → about 2.8% with SA | SA bandwidth and latency |
| [@lit11] | Edge federated video sensing, DP + HE | No stream leakage | Commercial hardware, dashboards |
| [@lit12] | Blockchain + FL recruitment system | Tamper-proof records | Edge frame rate and accuracy |
| [@lit13] | Committee-consensus auditable FL, dual masking | Matches FedAvg accuracy | Scalability |
| [@lit14] | Few-shot federated expression recognition | 84.9% on FER-2013 | Non-IID data |
| [@lit15] | Face de-identification against deep CNNs | Over 90% recognition failure | Reversible methods |
| [@lit16] | Data augmentation for small face datasets | 50% → 76% accuracy | Architectures, filters |

## 1.4 Technical Background

This section introduces the techniques that FedMPS combines and the metrics used to evaluate it. **Federated learning.** In FL, a server repeatedly sends a global model to the participating clients, each client trains it on its own data, and the server averages the returned models [@fedavg]. This averaging is FedAvg, and the data never leaves the clients. When clients hold very different data, that is, data that is not IID, the local models drift apart. FedProx adds a proximal term to each client's loss that penalises moving far from the global model [@fedprox]. **Face recognition.** ArcFace trains a deep network so that faces map to 512-dimensional unit vectors, called embeddings, with an additive angular margin that pulls embeddings of the same person together on the hypersphere [@arcface]. Two faces with embeddings x and y are compared by their cosine similarity:

$$ cos(x, y) = ( x · y ) / ( ‖x‖ ‖y‖ ) $$ {#eq:cos}

For unit-length embeddings, {eq:cos} reduces to the dot product x · y, so a value near 1 means the same person and a value near 0 means unrelated faces. VGGFace2 is a large face dataset of 9,131 identities with wide variation in pose, age and image quality [@vggface2]. **Secure aggregation.** SA lets a server learn the sum of the clients' updates without seeing any individual update. In the pairwise-masking construction, every pair of clients agrees a shared secret, for example with Diffie–Hellman key exchange, and derives a random mask from it. One client of the pair adds the mask and the other subtracts it, so every mask cancels in the sum. **Differential privacy.** DP bounds how much any single participant can change the output of a computation. In FL, each update is clipped and Gaussian noise is added. The guarantee is a privacy budget ε, where a smaller ε is more private, and it is accounted over rounds with Rényi DP.

**Cancelable biometrics.** A cancelable biometric template is a deliberately distorted, key-dependent version of a biometric feature. Templates made with the same key still match, a leaked template is revoked by issuing a new key, and templates made with different keys cannot be linked. **Open-set identification metrics.** For watchlist search, the standard measure of false alarms is the False Positive Identification Rate (FPIR), the fraction of people not on the watchlist who still raise an alert. The standard measure of success is the True Positive Identification Rate (TPIR), the fraction of watchlist people who are flagged *and* correctly identified. A system is compared by its TPIR at a fixed FPIR, for example TPIR at 1% FPIR. These two rates, rather than closed-set accuracy, decide whether an operator can trust the alerts.

## 1.5 Research Gaps and Contributions

The surveyed work leaves five gaps that this project addresses:

1. **Frame-by-frame decisions.** Existing systems decide on single frames, while FedMPS alerts only when several frames of one track agree [@lit1].
2. **Closed-set accuracy.** Most results are closed-set accuracies, while FedMPS is evaluated as an open-set search with 31,850 probes of people not on the watchlist [@lit1, @lit2, @lit4].
3. **Unmeasured privacy costs.** FedMPS measures each privacy mechanism against the same training run without it.
4. **Missing comparison bounds.** FedMPS compares federated training with local-only, centralised and frozen-model baselines on identical data.
5. **No human oversight.** FedMPS puts human review and an audit log before any disclosure.

## 1.6 Organization of the Report

Chapter 2 describes the proposed system: its architecture, data, training method, privacy mechanisms, evaluation protocol, application, standards, software and hardware. Chapter 3 gives the cost analysis of the project. Chapter 4 presents and discusses the results, including a test on real CCTV footage and a comparison with recent work. Chapter 5 concludes the report and lists future work. Chapter 6 is the appendix with the repository structure, the commands that reproduce the results and the key code listings. Chapter 7 lists the references.
