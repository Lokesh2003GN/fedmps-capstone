# CHAPTER 4 | Results and Discussions

This chapter presents the measured results. All accuracy numbers come from the fixed test split described in Section 2.2.12, and every number can be traced to a results file in the `results/` folder of the project. The main results come from the VGGFace2-HQ run of 25 October 2026, prepared and trained on the AWS GPU instance: `experiments_vgghq.json` for the training regimes and privacy experiments, `experiments_vgghq_base.json` for the degradation and operating-point measurements, and `pilot_vgghq.json` for model selection. Sections 4.8 and 4.9, on the application and on real CCTV video, use the adapter of the earlier VGGFace2 run.

## 4.1 The Size of the CCTV Problem and the Operating Point

The degradation model makes recognition much harder. With the frozen backbone and plain cosine matching at the default operating point of 3 enrolment photographs × 3 fused frames, rank-1 falls from 0.9989 on clean probes to 0.9683 on CCTV-degraded probes, and TPIR at 1% FPIR falls from 0.9912 to 0.8744. With one photograph and one frame, the drop is far larger, from 0.8908 to 0.6621. {fig:progression} shows how the headline rank-1 developed over the earlier runs of the project, from 0.4807 on CelebA with unaligned faces at one photograph × one frame, to 0.9474 on the original VGGFace2 with a 3,021-person watchlist. The final run on VGGFace2-HQ reaches **0.9733** for the frozen backbone with the scoring layer. Its watchlist has 691 people, because the 70/30 split leaves fewer people for evaluation, so its task is somewhat easier and the two runs are not compared directly.

![Rank-1 at each step of the earlier runs of the project](figures/fig04_dataset_progression.png){#fig:progression w=6.0}

{tab:oppoints} and {fig:oppoints} show how accuracy depends on the number of enrolment photographs and fused frames, and on the scoring layer.

Table: Rank-1 by enrolment photographs and fused probe frames for the frozen backbone {#tab:oppoints}

| Photos × frames | Raw cosine | + centring | + rotation | + hubness = full scoring layer | Gain |
|---|---|---|---|---|---|
| 1 × 1 | 0.6621 | 0.6647 | 0.6701 | 0.6710 | +0.009 |
| 1 × 3 | 0.8889 | 0.8919 | 0.8955 | 0.8977 | +0.009 |
| 3 × 1 | 0.7677 | 0.7696 | 0.7757 | 0.7744 | +0.007 |
| 2 × 3 | 0.9489 | 0.9507 | 0.9535 | 0.9557 | +0.007 |
| 3 × 3, default | 0.9683 | 0.9691 | 0.9710 | 0.9733 | +0.005 |
| 3 × 5 | 0.9897 | 0.9901 | 0.9909 | 0.9921 | +0.002 |
| 5 × 5 | 0.9961 | 0.9964 | 0.9971 | 0.9976 | +0.002 |

![Rank-1 across enrolment and frame settings in the upper graph, and the contribution of each part of the scoring layer in the lower graph](docs/thesis/figures/operating_points_stacked.png){#fig:oppoints w=5.2}

Fusing frames and averaging several enrolment photographs matter far more than any learned correction: going from 1 × 1 to 3 × 3 adds about 31 rank-1 points, and fusing frames is worth more than adding photographs, since 1 × 3 beats 3 × 1 by 12 points. The scoring layer adds a further 0.2 to 0.9 points at every operating point.

## 4.2 Model Selection by the Pilot

{tab:pilot} gives the pilot of Section 2.2.7. Every candidate was trained for 30 federated rounds on 90% of the training identities and scored on the other 10%, 321 people whose 160-person gallery and 7,044 mate probes are drawn only from the training 70%. The test identities were not used.

Table: Pilot on training identities only, 30 rounds, with the scoring layer {#tab:pilot}

| Candidate | Rank-1 | TPIR @ FPIR 1% | TPIR @ FPIR 0.1% | TAR @ FAR 0.1% | EER | Update size |
|---|---|---|---|---|---|---|
| Frozen backbone | 0.9864 | 0.9299 | 0.8836 | 0.7236 | 0.1113 | None |
| MLP adapter, 64 hidden units, μ = 3 | 0.9865 | 0.9371 | 0.8863 | 0.7244 | 0.1106 | 0.26 MB |
| **MLP adapter, 256 hidden units, μ = 0.3, chosen** | 0.9862 | **0.9390** | 0.8857 | 0.7302 | **0.1087** | 1.05 MB |
| Head adapter, rank 16, μ = 3 | 0.9861 | 0.9327 | 0.8864 | 0.7211 | 0.1113 | 1.64 MB |
| Head adapter, rank 16, μ = 0.3 | 0.9848 | 0.9354 | 0.8888 | 0.7291 | 0.1096 | 1.64 MB |
| Head adapter, rank 16, μ = 0.3, learning rate 3 × 10⁻⁴ | 0.9842 | 0.9363 | 0.8911 | 0.7305 | 0.1098 | 1.64 MB |

Every trained candidate improves the open-set and verification metrics slightly and leaves rank-1 unchanged. The head adapter, which changes the backbone's own last layers, is no better than the MLP adapter on its output, while costing 1.6 times the traffic, so the 256-unit MLP adapter was chosen. Already at this stage the gains are below one point, which anticipates the main result.

## 4.3 Federated, Local and Centralised Training

{tab:main} is the main result of the project. Every row is scored on the same test set, through the same scoring layer and at the same operating point, while the clean-probe row uses plain cosine and is the upper limit set by the degradation.

Table: Verification and open-set identification by training regime {#tab:main}

| Training regime | TAR @ FAR 10⁻⁴ | TAR @ FAR 10⁻³ | TAR @ FAR 10⁻² | TPIR @ FPIR 10⁻³ | TPIR @ FPIR 10⁻² | Rank-1 |
|---|---|---|---|---|---|---|
| Clean probes, ceiling | 0.8779 | 0.8903 | 0.9002 | 0.9827 | 0.9912 | 0.9989 |
| Frozen backbone | 0.5747 | 0.6820 | 0.7811 | 0.8357 | 0.9054 | 0.9733 |
| Centralised | 0.5673 | 0.6801 | 0.7864 | 0.8312 | 0.9080 | 0.9733 |
| Federated, FedMPS | **0.5750** | **0.6835** | **0.7877** | **0.8363** | **0.9107** | **0.9734** |
| Local only, mean ± standard deviation | 0.527 ± 0.035 | 0.646 ± 0.028 | 0.762 ± 0.018 | 0.791 ± 0.030 | 0.881 ± 0.020 | 0.9620 ± 0.009 |

Table: Closed-set identification at rank 1, 5 and 10, and verification AUC and EER {#tab:cmc}

| Training regime | Rank-1 | Rank-5 | Rank-10 | AUC | EER |
|---|---|---|---|---|---|
| Frozen backbone | 0.9733 | 0.9871 | 0.9906 | 0.9389 | 0.1211 |
| Centralised | 0.9733 | 0.9877 | 0.9908 | 0.9403 | 0.1173 |
| Federated, FedMPS | 0.9734 | 0.9877 | 0.9910 | 0.9402 | 0.1178 |
| Local only, mean of 5 nodes | 0.9620 | 0.9827 | 0.9874 | 0.9357 | 0.1252 |

At the strictest open-set point that this test set can resolve, an FPIR of 10⁻⁴, the TPIR is 0.6473 for federated training, 0.6369 for the frozen backbone and 0.6065 for centralised training. The Receiver Operating Characteristic (ROC), Detection Error Trade-off (DET) and Cumulative Match Characteristic (CMC) curves of {fig:curves} show the same picture across all operating points; the DET curve plots the False Negative Identification Rate (FNIR) against the FPIR. In the DET panel, the clean-probe curve uses plain cosine, as in {tab:main}.

![Verification ROC, identification DET and closed-set CMC for every arm, from top to bottom](docs/thesis/figures/roc_det_cmc_stacked.png){#fig:curves w=5.6}

The results are read as follows.

- **Federated against the frozen backbone.** The federated adapter is ahead on every open-set and verification metric: by 0.53 points of TPIR at 1% FPIR, 1.04 points at 0.01% FPIR, 0.66 points of TAR at 1% FAR, and 0.33 points of EER. Rank-1 is unchanged. The gain is consistent but small. With 30,540 mate probes, the 95% sampling band of TPIR at 1% FPIR is about ±0.32 points, so the main gain is just outside one band and inside two. Before the run, a gain of at least 2 points of TPIR at 1% FPIR had been set as the bar for recommending that the adapter be trained. That bar is not met.
- **Federated against centralised.** The two are equal within the sampling noise; federated is 0.27 points ahead at 1% FPIR and 0.01 points behind in rank-1. The federated loop warned that federated beat centralised, which can point to leakage. The identities of the nodes and of the test set are disjoint by construction and are checked when the data is partitioned, so no leakage is possible. The more likely reading is that averaging five differently degraded clients acts as a mild regulariser. *Keeping the data at the camera sites costs no measurable accuracy compared with pooling it.*
- **Federated against local only.** Here collaboration clearly matters. Every node's local adapter is worse than the federated one and worse than the frozen backbone, by 1.1 rank-1 points and 2.4 TPIR points on average. The federated adapter is 1.14 rank-1 points and 2.95 TPIR points above the local mean. {tab:localnodes} gives the five nodes.
- **Clean against degraded.** The clean ceiling, a TPIR of 0.9912 at 1% FPIR, is far above every degraded arm at about 0.91. Most of the difficulty lies in the camera degradation, not in the choice of training regime.

Table: Local-only adapters, each trained at one node and scored on the full test set {#tab:localnodes}

| Node | Identities | Rank-1 | TPIR @ FPIR 1% | TPIR @ FPIR 0.1% | TAR @ FAR 0.1% |
|---|---|---|---|---|---|
| Node 0 | 322 | 0.9472 | 0.8467 | 0.7415 | 0.5967 |
| Node 1 | 483 | 0.9665 | 0.8907 | 0.7907 | 0.6536 |
| Node 2 | 644 | 0.9653 | 0.8877 | 0.8025 | 0.6625 |
| Node 3 | 805 | 0.9622 | 0.8849 | 0.7971 | 0.6468 |
| Node 4 | 969 | 0.9689 | 0.8961 | 0.8218 | 0.6677 |
| *Federated, for reference* | *3,223* | *0.9734* | *0.9107* | *0.8363* | *0.6835* |
| *Frozen backbone, for reference* | *None* | *0.9733* | *0.9054* | *0.8357* | *0.6820* |

The smallest node, node 0, with 322 people and the harshest camera, loses most: 2.6 rank-1 points against the frozen backbone. Larger nodes lose less, but none recovers the frozen accuracy. A site that fine-tunes alone on its own few hundred people, under one camera, learns that camera's people rather than a correction that transfers to new people. In the earlier run, a fairness check with the script `training/03_check_local_only.py` showed that early stopping and a proximal penalty bring a local adapter back to the frozen accuracy, but no higher. Taken together, the two runs show that a site working alone gains nothing from training and can lose by it, while federation turns the same data into a small gain.

## 4.4 Convergence over 200 Rounds

{fig:convergence} shows rank-1 and TPIR at 1% FPIR of the federated and centralised adapters after every one of the 200 rounds, with the frozen backbone as a dashed line and the 95% sampling band of the federated curve shaded.

![Rank-1 and TPIR at 1% FPIR after every round of 200, with the 95% sampling band of the federated curve](figures/fig12_convergence_200.png){#fig:convergence w=6.0}

The curves answer the instability seen in the earlier run, whose validation rank-1 still rose and fell after 30 rounds. Here the federated TPIR is above the frozen backbone from round 1 and stays between 0.9083, its lowest at round 30, and 0.9112, its highest at round 145. The last round, which is the one reported, gives 0.9107. Over the last 20 rounds the standard deviation is 0.003 points for TPIR and 0.002 points for rank-1, and the norm of the aggregated update falls from 0.178 in round 1 to 0.00002 in round 200, so the model has settled. The cosine learning-rate schedule is what makes it settle: the late rounds can only make very small changes. Rank-1 stays between 0.9728 and 0.9744 throughout, inside the sampling band around the frozen value. The centralised adapter starts below the frozen backbone, at 0.9038 TPIR in round 18, and recovers to 0.9080. The federated, centralised and local-only arms took 1,212 s, 1,203 s and 232 s of GPU time.

Why is the gain of the adapter so small, even with 136 images per person and 200 rounds? Three observations explain it.

1. **Even pooling all data does not help more.** The centralised arm, which sees every node's data, gains no more than the federated arm. The limit lies in what the data allows an embedding-space correction to achieve, not in FL.
2. **Much of the degradation is lost information.** A face 20–48 pixels wide, blurred and JPEG-compressed has lost detail that no mapping of 512-dimensional embeddings can restore. The gap of about 8 points between clean and degraded TPIR at 1% FPIR in {tab:main} is mostly of this kind.
3. **The recoverable part is rigid, and the scoring layer already captures it.** The camera shifts all embeddings in a similar direction. Centring and a rotation, which preserve the geometry of the embedding space, capture this shift and are applied to every arm. The adapter can only add what is left over.

## 4.5 Per-camera Results

{tab:percam} compares the frozen backbone and the federated adapter separately on probes from each node's camera profile, about 6,100 mate probes per camera.

Table: Per-camera rank-1 and TPIR at 1% FPIR, frozen backbone against federated adapter {#tab:percam}

| Camera | Face size, JPEG quality | Frozen rank-1 | Federated rank-1 | Frozen TPIR @ 1% | Federated TPIR @ 1% | Gain @ 1% |
|---|---|---|---|---|---|---|
| Node 0 | 21 px, q50 | 0.9549 | 0.9562 | 0.8385 | 0.8480 | +0.010 |
| Node 1 | 45 px, q61 | 0.9771 | 0.9765 | 0.9243 | 0.9242 | 0.000 |
| Node 2 | 38 px, q39 | 0.9820 | 0.9808 | 0.9245 | 0.9298 | +0.005 |
| Node 3 | 29 px, q31 | 0.9667 | 0.9672 | 0.8856 | 0.8862 | +0.001 |
| Node 4 | 47 px, q66 | 0.9859 | 0.9862 | 0.9633 | 0.9615 | −0.002 |

The cameras differ much more from each other, with TPIR from 0.84 to 0.96, than the two models differ on any one camera. The adapter helps most where the camera is worst: on node 0, with the smallest faces and heavy motion blur, it adds 0.95 points of TPIR, while on the best camera, node 4, it makes no difference. The ranking of the cameras follows face size, except that node 1, with the heaviest blur and sensor noise, falls behind node 2 despite its larger faces.

## 4.6 FedAvg against FedProx

In a sweep of 30 rounds, scored with plain cosine, plain FedAvg with μ = 0 reached a rank-1 of 0.9714, and FedProx with μ = 0.01, 0.1 and 1.0 reached 0.9715, 0.9715 and 0.9714. All settings fall within 0.0001 of each other. Client drift, which FedProx corrects, is therefore not a limiting factor here, because each round trains on only 15% of a node's data and the adapter starts as the identity. The main run uses μ = 0.3, chosen in the pilot; since μ changes rank-1 by less than 0.01 point across two orders of magnitude, this choice does not affect any conclusion.

## 4.7 Privacy Evaluation

{tab:privacy} summarises the measured effect and cost of each privacy mechanism. The privacy experiments run for 30 rounds and are scored with plain cosine, without the scoring layer, so their reference is the frozen backbone's raw rank-1 of 0.9683 rather than 0.9733.

Table: Privacy mechanisms with their measured effect and cost {#tab:privacy}

| Mechanism | Result | Cost or limit |
|---|---|---|
| SA with pairwise masks and 2048-bit Diffie–Hellman | Rank-1 0.9713 without masking and 0.9713 with masking; difference 0.0000 | Run time 3.1% longer: 92.0 s against 89.2 s; no recovery if a node drops out |
| DP with clip C = 0.313 and Gaussian noise | Rank-1 0.9707 without noise → 0.0566 at σ = 0.3 → 0.0011 at σ = 2.0 | ε = 252 at σ = 0.3; no noisy setting is usable |
| MIA, loss-threshold attack on held-out images of the same people | Attack AUC 0.5010, where 0.5 is chance | One attack type only |
| Cancelable templates, keyed binarisation with XOR | Rank-1 0.9715 float → 0.9244 at 512 bits | 4.7 rank-1 points at 512 bits |

### 4.7.1 Secure Aggregation

Masked and unmasked training produced an identical rank-1 of 0.9713 over 30 rounds, because the masks cancel exactly in the sum. The cost was 3.1% more run time. This is the clearest positive privacy result: the server learns only the aggregate update, at no cost in accuracy. In the main three-way run, masking was switched off. Since masking does not change the aggregated update, switching it on would leave {tab:main} unchanged.

### 4.7.2 Differential Privacy

{tab:dp} shows the trade-off between privacy budget and accuracy.

Table: DP noise multiplier, privacy budget and accuracy over 30 rounds at δ = 10⁻⁵ {#tab:dp}

| Noise multiplier σ | Privacy budget ε | Rank-1 | Verification AUC |
|---|---|---|---|
| 0, clipping only | ∞, no guarantee | 0.9707 | 0.9404 |
| 0.3 | 251.9 | 0.0566 | 0.7103 |
| 0.6 | 83.6 | 0.0118 | 0.6062 |
| 1.0 | 40.0 | 0.0046 | 0.5511 |
| 2.0 | 15.9 | 0.0011 | 0.5119 |

A meaningful guarantee needs an ε of roughly 10 or less. Here even the weakest noise destroys the model. The noise is added to each of the 263,424 coordinates of the update, so at σ = 0.3 its norm is about σ · C · √263,424 ≈ 48, against about 0.24 for a typical update. With only five clients the noise does not average out, and the adapter's residual output is swamped. Rank-1 then falls far *below* the frozen backbone, because the noisy adapter damages the embeddings rather than leaving them alone. DP is therefore implemented and measured but not usable at this scale, and it is reported as such. The earlier run, with a four times smaller adapter, kept a rank-1 of 0.875 at σ = 0.3, which is consistent with the noise norm growing with the square root of the number of parameters.

### 4.7.3 Membership Inference

A loss-threshold MIA tries to tell whether an image was in a node's training data from how well the model scores it. For this test, 20% of the images of every training identity were held out, the federated adapter was retrained without them, and the attack compared 87,611 training images with 87,611 held-out images of the same people. The attack AUC is 0.5010, which is indistinguishable from guessing at 0.5, and close to the 0.507–0.532 reported by Aldahiri et al. [@lit4]. The earlier run had reported an AUC of 0.4963, but its "held-out" images had in fact been trained on, so it did not test membership; the corrected test replaces it. A second, identity-level test, with an AUC of 0.5763, compares training identities with identities never seen by any node. It measures whether a person was *enrolled* rather than trained on, so it is not comparable and is not used as evidence. Only one attack type was tested.

### 4.7.4 Cancelable Templates

{tab:templates} gives the accuracy of the cancelable-template scheme used in the system, which combines dimension selection, mean thresholding and XOR keying, at increasing bit widths on the full test set.

Table: Template protection at increasing bit widths, rank-1 and TPIR at 1% FPIR {#tab:templates}

| Bits | Bytes | Rank-1 | TPIR @ FPIR 1% |
|---|---|---|---|
| 32 | 4 | 0.1069 | 0.0225 |
| 64 | 8 | 0.2931 | 0.0730 |
| 96 | 12 | 0.4602 | 0.1871 |
| 128 | 16 | 0.5826 | 0.2796 |
| 256 | 32 | 0.8138 | 0.5757 |
| 512 | 64 | 0.9244 | 0.7864 |
| Float, unprotected | 2,048 | 0.9715 | 0.8787 |

The 96-bit width used in the literature loses 51 rank-1 points under CCTV degradation, because the genuine score falls below the impostor score [@lit2]. At 512 bits, which is 64 bytes, the cost is 4.7 rank-1 points and 9.2 TPIR points, so 512 bits is the configured width. In the earlier run, a random projection before binarising was somewhat more accurate at every width, and adopting it is a straightforward improvement left for future work. Because templates still cost accuracy, the live pipeline defaults to cosine matching on the enrolled embedding, and template matching is available as an option, as Section 4.8 shows.

## 4.8 Operational Results of the Application

The complete case workflow was run with the demonstration video for node 2 and the reference photograph `media/person-1.jpg`, with the results in {tab:demo}. {fig:frame} shows a frame of the node's annotated output, and {fig:node} and {fig:audit} in Chapter 2 show the consoles during the same run.

Table: Demonstration scan of node 2 over 60 frames on the laptop CPU {#tab:demo}

| Quantity | Cosine mode, default | Template mode |
|---|---|---|
| Faces detected | 220 | 220 |
| Faces passed by the quality gate | 40; 82% rejected as too small | 40 |
| Frame-level hits | 40 | 40 |
| Track-level alerts | 1 | 1 |
| Alert score / threshold | 0.55 / 0.24 | 0.387 / 0.2656 |
| Frames of the track that agreed | 3 of 3 | 3 of 3 |
| Processing speed | 1.9–2.3 FPS | — |

![Annotated frame from node 2. The missing person is matched in red as track 1, with a score of +0.71 in this frame, and the other faces, in grey, are rejected by the quality gate as too small](docs/thesis/figures/annotated_frame.png){#fig:frame w=6.3}

The quality gate rejected 82% of detected faces as too small, track voting turned 40 frame-level hits into a single alert, and template mode raised the same alert using only the keyed 64-byte template. After the operator confirmed the alert, the server showed one confirmed sighting. The audit log recorded the complete chain, as {fig:audit} shows: case creation, five template issues, the alert, two reference views and the confirmation by `operator@node2`. One limitation was found in testing: on the node-0 video, the pipeline raised no alert for its enrolled subject. It found about three faces per frame where plain detection finds seven, so faces are lost between tracking and matching. Relaxing the quality gate did not change this, so the cause is elsewhere and remains unresolved.

## 4.9 Evaluation on Real CCTV Footage from ChokePoint

The evaluations above use simulated camera degradation. To test the system on real surveillance video, the frozen backbone and the federated adapter of the earlier VGGFace2 run, unchanged, were evaluated on the **ChokePoint** dataset [@chokepoint]. ChokePoint is real CCTV footage of 29 people walking through two portals, filmed by three cameras above each portal at 800 × 600 pixels and 30 FPS, with every face annotated with its eye positions. Nothing was trained on ChokePoint, so the test measures how the system transfers to real cameras. The runs used the AWS GPU instance and the scripts in `cctv_benchmark/`. The protocol is the open-set protocol of Section 2.2.12, moved onto real footage. In each of 10 random splits, half the people, 14 of them, are put on the watchlist and the others form the crowd. The two official sequence groups of the dataset are used in turn, one for enrolment and the other for probes, so a person is always enrolled from a different walk-through than the one searched. Each watchlist entry averages 3 frames, each probe fuses 3 consecutive frames, and the live system's threshold of 0.24 is used. Results are averaged over the 10 splits in both directions, which gives 20 evaluations.

#### Part A: Recognition of Real CCTV Faces

{tab:cctv_rec} reports recognition on the annotated faces, using either the dataset's official face crops or faces cut from the full video frames.

Table: Recognition on real CCTV faces from ChokePoint with the frozen backbone, mean over 20 evaluations {#tab:cctv_rec}

| Test set | Faces | Rank-1 | TPIR @ FPIR 1% | Hits at threshold 0.24 | Crowd false alarms at 0.24 |
|---|---|---|---|---|---|
| Portal 1 entering, official crops, laptop run, 25 people | 4,776 | 1.000 | 0.999 | 1.000 | 8.3% ± 4.4% |
| Both portals, official crops | 25,331 | 1.000 | 0.999 | 1.000 | 4.4% ± 3.3% |
| Both portals, faces cut from full frames | 28,162 | 1.000 | 0.997 | 0.999 | 4.6% ± 3.2% |

Enrolled people score 0.70–0.76 on average against their own watchlist entry, and people in the crowd score 0.14–0.15 against their best match. At 1% FPIR, 99.7–99.9% of enrolled people are found and correctly named. The federated adapter changes none of these figures by more than 0.003; for example, TPIR at 1% FPIR on the official crops is 0.999 with or without it. On the full frames, 16.6% of faces could not be landmark-aligned and were passed as plain crops, yet accuracy stayed above 99.6%.

#### Part B: The Live Pipeline End to End

Part B runs the application's own pipeline over 24 full camera sequences, which is 14,751 frames at the pipeline's sampling rate. The pipeline consists of SCRFD detection, ByteTrack tracking, the quality gate, the backbone with or without the adapter, and 3-of-5 track voting. Every alert is scored against the annotations, as {tab:cctv_video} shows.

Table: Live pipeline on 24 real CCTV sequences from ChokePoint, totals over 20 evaluations {#tab:cctv_video}

| Arm | Enrolled walk-throughs alerted | Crowd walk-throughs falsely alerted | Wrong-name alerts | Median time to alert |
|---|---|---|---|---|
| Frozen backbone | 95.2%, 1,988 of 2,088 | 3.9%, 88 of 2,232 | 0 | 1.3 s |
| + federated adapter | 95.2%, 1,988 of 2,088 | 3.7%, 83 of 2,232 | 0 | 1.3 s |

In 95% of the walk-throughs by an enrolled person, the system raised a correct alert, typically 1.3 seconds after the person first appeared. It never named the wrong person. The quality gate passed 54% of the 7,433 detected faces, 68% at portal 1 and 46% at portal 2, and almost every rejection, 99.9%, was a face too small to recognise reliably. Face size at the camera, rather than the matching step, is therefore the main constraint on what the system can see. These results show three things.

1. **The system works on real surveillance footage.** It found almost every enrolled person and never confused identities.
2. **The live threshold must be set per deployment.** The threshold of 0.24 was chosen for a small watchlist on the demonstration videos. On ChokePoint it lets 4–5% of the crowd raise alerts, while Part A shows that a threshold set for 1% FPIR would still find 99.7% of enrolled people. Re-deriving the threshold from the DET curve on the deployed cameras, as the configuration already advises, would cut false alarms roughly fourfold at almost no cost.
3. **The adapter adds nothing on real CCTV either.** This matches the small effect of the adapter on the simulated data.

ChokePoint is easier than a real missing-person search. People walk towards the cameras, the watchlist is small at 14 people, and enrolment frames come from the same cameras, not from a family photograph. The near-perfect recognition figures should therefore not be generalised, and the large VGGFace2-HQ evaluation remains the main measure of accuracy.

## 4.10 Comparison with Published Work

{tab:compare} compares FedMPS with five recent works that use VGGFace2 or report on the same metrics, three of them federated and two privacy-preserving, together with the VGGFace2 paper itself as a reference [@vggface2] [@fedhide] [@fedfs] [@kimteoh] [@dpfedface] [@ji2022]. Each paper is given with the metric it reports, and FedMPS and the frozen backbone are given on the *same metric*, measured on this project's test set. The reported figures are those of each paper. IJB-B and IJB-C belong to the Intelligence Advanced Research Projects Activity Janus Benchmark (IJB) family.

Table: Comparison with published work, each paper's metric against FedMPS on the same metric {#tab:compare}

| Reference | Dataset | Method | Metric | Paper's result | FedMPS | Frozen backbone |
|---|---|---|---|---|---|---|
| [@vggface2] | VGGFace2, 8,631 identities; tested on IJB-C | ResNet-50, centralised | Rank-1 / TAR @ FAR 10⁻⁴ | 0.898 / 82.5% | 0.9734 / 57.50% | 0.9733 / 57.47% |
| [@fedhide] | VGGFace2, 8,631 clients of one identity each | MobileFaceNet, 64 × 64 faces, federated | EER | 9.5% unprotected; 10.4–12.9% protected | 11.78% | 12.11% |
| [@fedfs] | VGGFace, 8,673 identities | MobileFaceNet, federated self-supervised | AUROC / TPIR @ FPIR 10⁻¹, 10⁻², 10⁻³ | 0.9794 / 0.9786, 0.9337, 0.8721 | 0.9402 / 0.9532, 0.9107, 0.8363 | 0.9389 / 0.9487, 0.9054, 0.8357 |
| [@kimteoh] | MS-Celeb-1M, 8,000 identities across 20 clients; tested on IJB-C | Heterogeneous backbones, federated prototypes | TAR @ FAR 10⁻⁴, per client | 57.14% | 57.50% | 57.47% |
| [@dpfedface] | CASIA-WebFace, 4,000 clients; tested on IJB-C | 64-layer CNN, federated with DP | TAR @ FAR 10⁻³ | 87.76% | 68.35% | 68.20% |
| [@ji2022] | VGGFace2; tested on IJB-C | ResNet-50 ArcFace with frequency-domain DP | TAR @ FAR 10⁻⁴ | 93.25% unprotected; 91.22% private | 57.50% | 57.47% |

The figures are measured on different test sets and are **not directly comparable**. The published works test on IJB-B, IJB-C or the original images, where both faces of a pair are web photographs. FedMPS tests every pair as a clean enrolment photograph against a CCTV-degraded probe whose face is 20–48 pixels wide, so its verification task is much harder, while its 691-person gallery makes its identification task easier than IJB-C. Four comparisons are nevertheless meaningful.

- **Accuracy on the same metric.** In open-set identification, which is how a missing-person system is used, FedMPS comes closest to the published work: at FPIR 10⁻³ its TPIR of 0.836 is 3.6 points below FedFS, which tests on undegraded faces. Its EER of 11.8% lies within the 10.4–12.9% range that FedHide reports for its protected models on 64 × 64 faces. Its TAR at FAR 10⁻⁴ of 57.5% matches the per-client result of Kim, Kim and Teoh, but is far below the centralised and privacy-preserving ResNet-50 models of Cao et al. and Ji et al., which are tested on high-quality IJB-C images. The difference is the probe quality, as the clean-probe ceiling of 87.8% at the same point in {tab:main} shows.
- **Cost of federation and privacy.** Ji et al. lose 2.0 points of TAR to their privacy mechanism, and FedHide loses 0.9–3.4 points of EER to protection. In FedMPS, federation costs nothing against centralised training, and SA costs no accuracy at all. DP, by contrast, is not usable at this scale, as Section 4.7.2 shows, so FedMPS does not match the formal guarantees of DP-FedFace or Ji et al.
- **Communication.** The five works train a whole backbone of millions of parameters, or exchange class prototypes, across clients. FedMPS sends 1.05 MB per update, 165 times less than its backbone, at the price of not adapting the backbone itself.
- **Evaluation.** Only FedFS reports open-set TPIR at fixed FPIR, and none tests on surveillance video. FedMPS is evaluated as an open-set search under camera degradation, and on real CCTV footage in Section 4.9.

## 4.11 Discussion and Limitations

The results support four claims and qualify one. **Supported.**

1. A privacy-preserving missing-person search system can be built end to end, and on real CCTV video it alerted on 95% of enrolled walk-throughs, never with the wrong name.
2. Federated training reaches the accuracy of centralised training on pooled data, so keeping the data at the sites costs no accuracy.
3. Federation is far better than training alone. Every node's local adapter loses accuracy against the frozen backbone, while the federated adapter gains slightly, so collaboration is worth 1.1 rank-1 points and 3.0 TPIR points over local training.
4. SA hides individual updates at no accuracy cost, and no membership leakage is detectable.

**Qualified.** On this data, the federated adapter gives a small but consistent gain over the frozen ArcFace backbone: 0.5 points of TPIR at 1% FPIR, 1.0 point at 0.01% FPIR and 0.3 points of EER, with rank-1 unchanged. It is stable over 200 rounds, but it is below the 2-point bar set before the run. The answer to whether the backbone should be frozen or trained is therefore: **freeze the backbone**, since it does almost all of the work; add the federated adapter where the communication of 1 MB per round is cheap, since it helps most on the worst cameras; and **never fine-tune at a single site alone**, which loses accuracy. The value of FedMPS on this data lies mainly in its privacy properties and its protocol, not in an accuracy gain.

The main limitations are these. The evaluation uses **simulated degradation**, and the real-CCTV test of Section 4.9 covers only 29 people, with enrolment frames from the same cameras. The clean images of VGGFace2-HQ were **restored by GFPGAN**, which may smooth or alter fine facial detail, so the clean enrolment photographs are cleaner than many real case photographs. All results come from **one random seed**, without confidence intervals over seeds. The backbone was trained by InsightFace on WebFace600K, and **identity overlap** with VGGFace2 was not checked, which may make the frozen baseline somewhat optimistic. The **scoring layer** is fitted on pooled embeddings and is not in the live matcher. The default **cosine mode** sends the enrolled embedding to each node; template mode avoids this at 4.7 rank-1 points. SA has **no dropout recovery**, **DP** has no usable operating point here, the **node-0 detection loss** is unresolved, and the **application** still uses the adapter of the earlier run.
