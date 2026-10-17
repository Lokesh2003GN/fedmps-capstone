# CHAPTER 4 | Results and Discussions

This chapter presents the measured results. All accuracy numbers come from the fixed test split described in Section 2.2.12, and every number can be traced to a results file in the project (`results/`). The VGGFace2 dataset preparation and the full experiment suite ran on the AWS GPU instance on 15 September 2026. The three-way comparison of training regimes (Section 4.2) was re-run on 13 October 2026 on the team laptop from the same AWS-computed embeddings, to add rank-5 and rank-10. Section 4.4 explains why the two runs differ slightly.

## 4.1 The Size of the CCTV Problem and the Operating Point

The degradation model makes recognition much harder. With the frozen backbone and plain cosine matching at the default operating point (3 enrolment photographs × 3 fused frames), rank-1 falls from 0.9962 on clean probes to 0.9294 on CCTV-degraded probes. With one photograph and one frame, the drop is far larger: from 0.918 to 0.524.

{fig:progression} shows how the project's headline rank-1 developed. The first baseline, on CelebA with unaligned faces at one photograph × one frame, scored 0.4807. Adding the scoring layer gave 0.5029. Moving to the 3 × 3 operating point gave 0.9186. Moving to VGGFace2 with ArcFace alignment gave 0.9848. Finally, the watchlist was deliberately enlarged from 1,750 to 3,021 people, and rank-1 fell to **0.9474**. That last step made the task harder on purpose: at 1,750 people the task was nearly saturated, leaving no room to measure whether the adapter helps. Because the dataset and alignment changed together, their separate effects are not isolated; the alignment A/B test of Section 2.2.3 measures alignment alone.

![Rank-1 at each step of the project](figures/fig04_dataset_progression.png){#fig:progression w=4.6}

{tab:oppoints} and {fig:oppoints} show how accuracy depends on the number of enrolment photographs and fused frames, and on the scoring layer.

Table: Rank-1 by enrolment photographs and fused probe frames (frozen backbone) {#tab:oppoints}

| Photos × frames | Raw cosine | + centring | + rotation | + hubness (full scoring layer) | Gain |
|---|---|---|---|---|---|
| 1 × 1 | 0.5240 | 0.5301 | 0.5344 | 0.5419 | +0.018 |
| 1 × 3 | 0.8238 | 0.8332 | 0.8393 | 0.8511 | +0.027 |
| 2 × 3 | 0.9016 | 0.9075 | 0.9115 | 0.9214 | +0.020 |
| 3 × 3 (default) | 0.9294 | 0.9355 | 0.9391 | 0.9474 | +0.018 |
| 3 × 5 | 0.9775 | 0.9798 | 0.9814 | 0.9863 | +0.009 |
| 5 × 5 | 0.9861 | 0.9882 | 0.9895 | 0.9918 | +0.006 |

![Rank-1 across enrolment and frame settings, and the contribution of each part of the scoring layer](figures/fig08_operating_points.png){#fig:oppoints w=5.4}

Fusing frames and averaging several enrolment photographs matter far more than any learned correction: going from 1 × 1 to 3 × 3 adds about 40 rank-1 points. The scoring layer adds a further 0.6 to 2.7 points at every operating point.

## 4.2 Federated, Local and Centralised Training

{tab:main} is the main result of the project. Every row is scored on the same test set, through the same scoring layer and at the same operating point; the clean-probe row uses plain cosine and is the upper limit set by the degradation.

Table: Verification and open-set identification by training regime {#tab:main}

| Training regime | TAR @ FAR 0.1% | TAR @ FAR 1% | TPIR @ FPIR 0.1% | TPIR @ FPIR 1% | Rank-1 |
|---|---|---|---|---|---|
| Clean probes (ceiling) | 0.9247 | 0.9347 | 0.9801 | 0.9915 | 0.9962 |
| Frozen backbone | 0.6287 | 0.7440 | 0.6725 | 0.7968 | 0.9474 |
| Centralised | 0.6087 | 0.7360 | 0.6847 | 0.8043 | 0.9476 |
| Federated (FedMPS) | 0.6193 | 0.7303 | 0.6820 | 0.8020 | 0.9473 |
| Local only (mean ± std) | 0.583 ± 0.024 | 0.719 ± 0.011 | 0.634 ± 0.023 | 0.770 ± 0.012 | 0.9286 ± 0.008 |

Table: Closed-set identification at rank 1, 5 and 10 (CMC) {#tab:cmc}

| Training regime | Rank-1 | Rank-5 | Rank-10 |
|---|---|---|---|
| Frozen backbone | 0.9474 | 0.9752 | 0.9814 |
| Centralised | 0.9476 | 0.9749 | 0.9817 |
| Federated (FedMPS) | 0.9473 | 0.9755 | 0.9817 |
| Local only (mean of 5 nodes) | 0.9286 | 0.9663 | 0.9745 |

In 1:1 verification the picture is the same: AUC is 0.9769 on clean probes and between 0.9400 and 0.9456 for every degraded arm, with equal error rates of 0.121 to 0.124.

The ROC, detection-error-trade-off (DET) and CMC curves of {fig:curves} show the same picture across all operating points.

![Verification ROC, identification DET and closed-set CMC for every arm](figures/fig06_roc_det_cmc.png){#fig:curves w=6.3}

The results are read as follows.

- **Federated vs centralised.** The rank-1 gap is 0.0003, and TPIR at 1% FPIR differs by 0.002. With a single random seed and no confidence intervals, the two are equal. *Keeping the data at the camera sites costs no measurable accuracy compared with pooling it.*
- **Federated vs frozen backbone.** Rank-1 differs by −0.0001, TPIR at 1% FPIR by +0.005, and TPIR at 0.1% FPIR by +0.0095. These differences are within the noise of one seed: *the adapter gives no clear gain on this data* (Section 4.4 discusses why).
- **Federated vs local only.** As trained in {tab:main}, the local-only arm is 1.87 rank-1 points and 3.2 TPIR points below federated, and all five local models are worse than the federated one. However, that comparison is not like-for-like, as Section 4.3 shows.
- **Clean vs degraded.** The clean ceiling (TPIR 0.9915 at 1% FPIR) is far above every degraded arm (about 0.80). Most of the difficulty lies in the camera degradation, not in the choice of training regime.

## 4.3 A Fairness Check of the Local-only Arm

In {tab:main}, the local-only arm trains each node's adapter for 30 epochs and keeps the final weights, with no early stopping and no proximal penalty. The federated arm, by contrast, keeps its best round on held-out validation identities and trains under FedProx. To separate the effect of *collaboration* from the effect of *regularisation*, the local-only arm was retrained in two further ways (`training/03_check_local_only.py`):

1. with early stopping on each node's **own** validation identities, as a site working alone could do;
2. additionally with the same proximal penalty (μ = 3) towards the starting weights.

Table: Local-only training with and without the federated arm's regularisation (mean of five nodes) {#tab:fair}

| Local-only variant | Rank-1 | TPIR @ FPIR 1% | TPIR @ FPIR 0.1% |
|---|---|---|---|
| 30 epochs, last epoch kept (as in {tab:main}) | 0.9288 | 0.7703 | 0.6363 |
| + early stopping on own validation identities | 0.9448 | 0.7984 | 0.6776 |
| + early stopping + proximal penalty (μ = 3) | 0.9473 | 0.8016 | 0.6817 |
| *Federated, for reference* | *0.9473* | *0.8020* | *0.6820* |
| *Frozen backbone, for reference* | *0.9474* | *0.7968* | *0.6725* |

With the same treatment as the federated arm, every node's local adapter reaches the frozen backbone's accuracy (node rank-1 from 0.9467 to 0.9476). The 1.87-point gap in {tab:main} therefore measures the harm of **naive local fine-tuning** (30 epochs on a few hundred people overfits) rather than a benefit of collaboration. On this data, federated training, centralised training and well-regularised local training all converge to the frozen model. This corrects the claim made at the second review, where the 1.87-point gap was presented as the gain from federation.

## 4.4 Convergence, Early Stopping and Reproducibility

{fig:convergence} shows the federated adapter's validation rank-1 after each round. It is highest after round 1 (0.9514, against 0.9486 for the frozen backbone) and then drifts down, sometimes below the frozen backbone (0.9475 at round 18). Early stopping keeps round 1. The adapter learns the little that is learnable in the first round, and further rounds only overfit the training identities.

![Validation rank-1 per federated round; round 1 is kept](figures/fig07_convergence.png){#fig:convergence w=4.4}

The September run on AWS, with the same code, seed, data and settings, kept round 26 instead and reached a test rank-1 of 0.9448 (instead of 0.9473). The two machines' floating-point arithmetic differs slightly. As a result, per-round validation scores differ by at most 11 of the 2,590 validation probes, and in the AWS run round 26 beat round 1 by only 3 probes. A three-probe difference decides which round is kept. This confirms the conclusion of Section 4.2: the adapter's effect is smaller than the validation set can resolve.

Why does the adapter not help? Three observations explain it.

1. **Even pooling all data does not help.** The centralised arm, which sees every node's data, also ties the frozen model. The limit lies in what the data allows an embedding-space correction to achieve, not in federated learning.
2. **Much of the degradation is lost information.** A face 20–48 pixels wide, blurred and JPEG-compressed has lost detail that no mapping of 512-dimensional embeddings can restore.
3. **The recoverable part is rigid, and the scoring layer already captures it.** The camera shifts all embeddings in a similar direction. Centring and a rotation, which preserve the geometry of the embedding space, capture this shift (+1.8 points, {tab:oppoints}) and are applied to every arm. A free-form learned map instead overfits the few hundred training people at each site, as the regularisation sweep showed for larger adapters.

## 4.5 Per-camera Results

{tab:percam} compares the frozen backbone and the federated adapter separately on probes from each node's camera profile.

Table: Per-camera rank-1 and TPIR at 1% FPIR, frozen backbone against federated adapter {#tab:percam}

| Camera | Frozen rank-1 | Federated rank-1 | Frozen TPIR @ 1% | Federated TPIR @ 1% | Gain @ 1% |
|---|---|---|---|---|---|
| Node 0 | 0.9539 | 0.9533 | 0.846 | 0.844 | −0.002 |
| Node 1 | 0.9200 | 0.9203 | 0.727 | 0.731 | +0.004 |
| Node 2 | 0.9234 | 0.9234 | 0.737 | 0.744 | +0.008 |
| Node 3 | 0.9782 | 0.9779 | 0.871 | 0.875 | +0.003 |
| Node 4 | 0.9608 | 0.9611 | 0.833 | 0.838 | +0.005 |

The cameras differ much more from each other (TPIR from 0.73 to 0.87) than the two models differ on any one camera (under one point). The harshest cameras, nodes 1 and 2, with small faces and low JPEG quality, are the hardest regardless of the model.

## 4.6 FedAvg against FedProx

In a sweep on the AWS run, plain FedAvg (μ = 0) reached rank-1 0.9372, and FedProx with μ = 0.01, 0.1 and 1.0 reached 0.9373, 0.9380 and 0.9374. All settings fall within 0.001 of each other: client drift, which FedProx corrects, is not a limiting factor here, because early stopping keeps the model near its starting point. The main runs use μ = 3.0, outside this sweep; since μ changes rank-1 by less than 0.1 point across two orders of magnitude, this choice does not affect any conclusion.

## 4.7 Privacy Evaluation

{tab:privacy} summarises the measured effect and cost of each privacy mechanism.

Table: Privacy mechanisms: measured effect and cost {#tab:privacy}

| Mechanism | Result | Cost or limit |
|---|---|---|
| Secure aggregation (pairwise masks, 2048-bit Diffie–Hellman) | rank-1 0.9376 without masking and 0.9376 with masking (10 rounds); difference 0.0000 | 1.008× run time (26.8 s → 27.0 s); no recovery if a node drops out |
| Differential privacy (clip C = 0.128, Gaussian noise) | rank-1 0.9391 without noise → 0.8751 at σ = 0.3 → 0.0216 at σ = 2.0 | ε = 252 at σ = 0.3; no setting gives ε < 10 with usable accuracy |
| Membership inference (loss-threshold attack) | attack AUC 0.4963 (0.5 = chance) | one attack type only |
| Cancelable templates (keyed binarisation, XOR) | rank-1 0.9226 float → 0.8321 at 512 bits | about 9 rank-1 points at 512 bits |

### 4.7.1 Secure Aggregation

Masked and unmasked training produced identical rank-1 (0.9376) over 10 rounds, because the masks cancel exactly in the sum. The cost was 0.8% more run time. This is the clearest positive privacy result: the server learns only the aggregate update, at no cost in accuracy. In the main three-way run masking was switched off; since it does not change the aggregated update, switching it on would leave {tab:main} unchanged.

### 4.7.2 Differential Privacy

{tab:dp} shows the trade-off between privacy budget and accuracy.

Table: Differential privacy: noise multiplier, privacy budget and accuracy (30 rounds, δ = 10⁻⁵) {#tab:dp}

| Noise multiplier σ | Privacy budget ε | Rank-1 |
|---|---|---|
| 0 (no noise) | ∞ (no guarantee) | 0.9391 |
| 0.3 | 251.9 | 0.8751 |
| 0.6 | 83.6 | 0.6254 |
| 1.0 | 40.0 | 0.2477 |
| 2.0 | 15.9 | 0.0216 |

A meaningful guarantee needs ε of roughly 10 or less. Every setting near that range destroys accuracy, and the only setting with tolerable accuracy (σ = 0.3, rank-1 0.875) has ε = 252, which guarantees almost nothing. With only five clients and a model this small, the added noise does not average out. Differential privacy is therefore implemented and measured but not usable at this scale, and it is reported as such.

### 4.7.3 Membership Inference

A loss-threshold membership-inference attack [@shokri] tries to tell whether an image was in a node's training data from how well the model scores it. On 17,260 training images and 17,260 held-out images of the same people, the attack AUC is 0.4963: indistinguishable from guessing (0.5), and in the same range reported by [@lit4] (0.507–0.532). A second, identity-level test (AUC 0.5456) measures whether a person was *enrolled* rather than trained on, so it is not comparable and is not used as evidence. Only one attack type was tested.

### 4.7.4 Cancelable Templates

{tab:templates} gives the accuracy of three cancelable-template schemes at increasing bit widths, measured on a smaller test set in a separate sweep, so only the relative costs matter. Scheme A is the one implemented in the system: dimension selection, mean thresholding and XOR keying. Scheme B uses a random projection before binarising, and scheme C is BioHashing.

Table: Template protection: rank-1 / TPIR at 1% FPIR by scheme and bit width {#tab:templates}

| Bits (bytes) | A: binarise (used) | B: random projection | C: BioHashing |
|---|---|---|---|
| 96 (12) | 0.396 / 0.108 | 0.636 / 0.274 | 0.377 / 0.049 |
| 128 (16) | 0.466 / 0.175 | 0.668 / 0.336 | 0.479 / 0.149 |
| 256 (32) | 0.694 / 0.377 | 0.804 / 0.515 | 0.636 / 0.330 |
| 384 (48) | 0.794 / 0.474 | 0.864 / 0.570 | 0.734 / 0.404 |
| 512 (64) | 0.832 / 0.574 | 0.857 / 0.594 | 0.766 / 0.436 |
| Float, unprotected (2,048 bytes) | 0.923 / 0.681 | 0.923 / 0.681 | 0.923 / 0.681 |

The 96-bit width used in the literature [@lit2] loses 53 rank-1 points under CCTV degradation, because the genuine score falls below the impostor score. At 512 bits (64 bytes) the cost is about 9 points, so 512 bits is the configured width. Random projection (scheme B) is somewhat more accurate at every width; adopting it is a straightforward improvement left for future work. Because templates still cost accuracy, the live pipeline defaults to cosine matching on the enrolled embedding, and template matching is available as an option (Section 4.8).

## 4.8 Operational Results of the Application

The complete case workflow was run with the demonstration video for node 2 and the reference photograph `media/person-1.jpg` ({tab:demo}). {fig:frame} shows a frame of the node's annotated output, and {fig:node} and {fig:audit} in Chapter 2 show the consoles during the same run.

Table: Demonstration scan of node 2 (60 frames, laptop CPU) {#tab:demo}

| Quantity | Cosine mode (default) | Template mode |
|---|---|---|
| Faces detected | 220 | 220 |
| Faces passed by the quality gate | 40 (82% rejected as too small) | 40 |
| Frame-level hits | 40 | 40 |
| Track-level alerts | 1 | 1 |
| Alert score (threshold) | 0.55 (0.24) | 0.387 (0.2656) |
| Frames of the track that agreed | 3 of 3 | 3 of 3 |
| Processing speed | 1.9–2.3 frames per second | — |

![Annotated frame from node 2: the missing person matched (red, track 1, score +0.71 in this frame); other faces rejected by the quality gate as too small (grey)](docs/thesis/figures/annotated_frame.png){#fig:frame w=4.4}

The quality gate rejected 82% of detected faces as too small, track voting turned 40 frame-level hits into a single alert, and template mode raised the same alert using only the keyed 64-byte template.

After the operator confirmed the alert, the server showed one confirmed sighting, and the audit log recorded the complete chain: case creation, five template issues, the alert, two reference views and the confirmation by `operator@node2` ({fig:audit}).

One limitation was found in testing: on the node-0 video, the pipeline raised no alert for its enrolled subject. It found about three faces per frame where plain detection finds seven, so faces are lost between tracking and matching. Relaxing the quality gate did not change this, so the cause is elsewhere and remains unresolved.

## 4.9 Evaluation on Real CCTV Footage (ChokePoint)

The evaluations above use simulated camera degradation. To test the system on real surveillance video, the frozen backbone and the federated adapter, unchanged from the VGGFace2 training, were evaluated on the **ChokePoint** dataset [@chokepoint]. ChokePoint is real CCTV footage of 29 people walking through two portals, filmed by three cameras above each portal at 800 × 600 pixels and 30 frames per second, with every face annotated with its eye positions. Nothing was trained on ChokePoint: the test measures how the system transfers to real cameras. The runs used the AWS GPU instance (`cctv_benchmark/`).

The protocol is the open-set protocol of Section 2.2.12, moved onto real footage. In each of 10 random splits, half the people (14) are put on the watchlist and the others form the crowd. The dataset's two official sequence groups are used in turn, one for enrolment and the other for probes, so a person is always enrolled from a different walk-through than the one searched. Each watchlist entry averages 3 frames, each probe fuses 3 consecutive frames, and the live system's threshold of 0.24 is used. Results are averaged over the 10 splits in both directions (20 evaluations).

#### Part A: recognition of real CCTV faces

{tab:cctv_rec} reports recognition on the annotated faces, either the dataset's official face crops or faces cut from the full video frames.

Table: Recognition on real CCTV faces (ChokePoint, frozen backbone, mean over 20 evaluations) {#tab:cctv_rec}

| Test set | Faces | Rank-1 | TPIR @ FPIR 1% | Hits at threshold 0.24 | Crowd false alarms at 0.24 |
|---|---|---|---|---|---|
| Portal 1 entering, official crops (laptop, 25 people) | 4,776 | 1.000 | 0.999 | 1.000 | 8.3% ± 4.4% |
| Both portals, official crops | 25,331 | 1.000 | 0.999 | 1.000 | 4.4% ± 3.3% |
| Both portals, faces cut from full frames | 28,162 | 1.000 | 0.997 | 0.999 | 4.6% ± 3.2% |

Enrolled people score 0.70–0.76 on average against their own watchlist entry, and people in the crowd score 0.14–0.15 against their best match. At 1% FPIR, 99.7–99.9% of enrolled people are found and correctly named. The federated adapter changes none of these figures by more than 0.003 (for example, TPIR at 1% FPIR on the official crops is 0.999 with or without it). On the full frames, 16.6% of faces could not be landmark-aligned and were passed as plain crops, yet accuracy stayed above 99.6%.

#### Part B: the live pipeline end to end

Part B runs the application's own pipeline over 24 full camera sequences (14,751 frames at the pipeline's sampling rate): SCRFD detection, ByteTrack tracking, the quality gate, the backbone (with or without the adapter) and 3-of-5 track voting. Every alert is scored against the annotations ({tab:cctv_video}).

Table: Live pipeline on real CCTV video (ChokePoint, 24 sequences, totals over 20 evaluations) {#tab:cctv_video}

| Arm | Enrolled walk-throughs alerted | Crowd walk-throughs falsely alerted | Wrong-name alerts | Median time to alert |
|---|---|---|---|---|
| Frozen backbone | 95.2% (1,988 of 2,088) | 3.9% (88 of 2,232) | 0 | 1.3 s |
| + federated adapter | 95.2% (1,988 of 2,088) | 3.7% (83 of 2,232) | 0 | 1.3 s |

In 95% of the walk-throughs by an enrolled person, the system raised a correct alert, typically 1.3 seconds after the person first appeared. It never named the wrong person. The quality gate passed 54% of the 7,433 detected faces (68% at portal 1 and 46% at portal 2), and almost every rejection (99.9%) was a face too small to recognise reliably. Face size at the camera, rather than the matching step, is therefore the main constraint on what the system can see.

These results show three things.

1. **The system works on real surveillance footage.** It found almost every enrolled person and never confused identities.
2. **The live threshold must be set per deployment.** The threshold of 0.24 was chosen for a small watchlist on the demonstration videos. On ChokePoint it lets 4–5% of the crowd raise alerts, while Part A shows that a threshold set for 1% FPIR would still find 99.7% of enrolled people. Re-deriving the threshold from the DET curve on the deployed cameras, as the configuration already advises, would cut false alarms roughly fourfold at almost no cost.
3. **The adapter adds nothing on real CCTV either.** This matches the VGGFace2 finding.

ChokePoint is easier than a real missing-person search. People walk towards the cameras, the watchlist is small (14 people), and enrolment frames come from the same cameras, not from a family photograph. The near-perfect recognition figures should therefore not be generalised; the large VGGFace2 evaluation remains the main measure of accuracy.

## 4.10 Context: Published VGGFace2 Results

For context, ResNet-50 models trained on VGGFace2 report rank-1 between 0.886 and 0.976 on the IJB-A, IJB-B and IJB-C benchmarks [@vggface2], against 0.947 for FedMPS here. The numbers are **not comparable**: those models were trained on VGGFace2 and tested on the IJB benchmarks with a ten-split protocol, while this project tests a different pretrained model on CCTV-degraded VGGFace2 with one fixed split. No claim of superiority is made.

## 4.11 Discussion and Limitations

The results support three claims and refute one.

**Supported.**

1. A privacy-preserving missing-person search system can be built end to end: keyed per-site templates, local scanning with track voting, human review and an audit log. On real CCTV video it alerted on 95% of enrolled walk-throughs, never with the wrong name.
2. Federated training reaches the accuracy of centralised training on pooled data, so keeping the data at the sites costs no accuracy.
3. Secure aggregation hides individual updates at no accuracy cost, and no membership leakage is detectable.

**Refuted.** On this data, a federated adapter does *not* improve recognition over the frozen ArcFace backbone, and federation does not beat a well-regularised local model. The value of FedMPS on this data lies in its privacy properties and its protocol, not in an accuracy gain.

The main limitations are these. The evaluation uses **simulated degradation**; the real-CCTV test (Section 4.9) covers only 29 people, with enrolment frames from the same cameras. All results come from **one random seed**, without confidence intervals. **Identity overlap** with the backbone's training data was not checked. The **scoring layer** is fitted on pooled embeddings and is not in the live matcher. The default **cosine mode** sends the enrolled embedding to each node; template mode avoids this at about 9 rank-1 points. Secure aggregation has **no dropout recovery**, **DP** has no usable operating point at this model size, and the **node-0 detection loss** (Section 4.8) is unresolved.
