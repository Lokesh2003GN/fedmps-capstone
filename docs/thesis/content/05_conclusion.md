# CHAPTER 5 | Conclusion and Future Work

This chapter summarises what the project achieved, including the negative findings, and lists directions for future work.

## 5.1 Conclusion

This project built and evaluated FedMPS, a federated, privacy-preserving system for searching CCTV footage for a missing person. The two objectives set in Chapter 1 were met as follows.

1. **A federated multi-camera pipeline.** Face detection with SCRFD, tracking with ByteTrack, a quality gate, five-point alignment, a frozen ArcFace backbone and track-level voting run at each node. A 263,424-parameter adapter on top of the backbone is trained by FL with FedProx across five nodes with non-IID camera conditions. Raw video, face crops and training data never leave a node, and each update is 1.05 MB instead of 174.4 MB.
2. **Privacy mechanisms, each measured.** Pairwise-masked SA produced exactly the same model as plain aggregation at 3.1% extra run time. Update-level DP was implemented with a Rényi DP accountant, but even the weakest noise, at ε = 252, destroyed the adapter, so DP has no usable setting at this scale. Cancelable templates, keyed per case and per node, work end to end, and at 512 bits they cost 4.7 rank-1 points. A corrected MIA, on images held out of training, found no leakage, with an AUC of 0.501.

The system was evaluated with open-set identification metrics on every image of VGGFace2-HQ, 628,699 images of 4,605 people, split 70/30 by identity: 3,223 people trained across the five nodes, and 1,382 held-out people formed a 691-person watchlist and a crowd of 31,850 probes of people not on it. Federated training ran for 200 rounds and settled, with the last round reported and no round selection. It reached a rank-1 of 0.9734 and a TPIR of 0.9107 at 1% FPIR, against 0.9733 and 0.9080 for centralised training on pooled data. On real CCTV video from ChokePoint, the live pipeline alerted correctly on 95.2% of enrolled walk-throughs, never with the wrong name. The server and node consoles carry a case from enrolment, through scanning and human review, to a confirmed sighting. Every step is recorded in an audit log, and only a human-confirmed face crop ever reaches the server.

The effect of the adapter is reported as plainly as the rest. Against the frozen ArcFace backbone, the federated adapter gains 0.53 points of TPIR at 1% FPIR, 1.04 points at 0.01% FPIR and 0.33 points of EER, with rank-1 unchanged. The gain is consistent across metrics and stable over 200 rounds, and largest on the worst camera, but it is below the 2-point bar set before the run. Training alone is worse: every node's local adapter falls below the frozen backbone, by 1.1 rank-1 points on average, so federation is worth 1.1 rank-1 points and 3.0 TPIR points over local training. The recommendation is therefore to keep the backbone frozen, to add the federated adapter where its 1 MB per round is cheap, and never to fine-tune at a single site alone. The difficulty lies in information lost to camera degradation; the part that can be corrected, a rigid shift of the embeddings, is mostly captured by the scoring layer. Against five published works, FedMPS matches the published per-client and protected results on EER and TAR at FAR 10⁻⁴ despite its much harder CCTV probes, and stays within 3.6 points of open-set TPIR at FPIR 10⁻³. FedMPS therefore shows that camera sites *can* cooperate on a missing-person search without pooling footage or biometric data, at no accuracy cost relative to pooling, with human oversight and an audit trail on every disclosure. It also shows, by measurement, where the limits of the current design lie.

## 5.2 Future Work

The following directions would extend this work:

1. **Harder real footage.** Extend the real-CCTV evaluation to enrolment from ordinary photographs, probes from cameras at several distances as in the SCface dataset, and larger watchlists. Night-time and infrared footage may contain systematic distortions on which an adapter could help.
2. **Per-site thresholds.** Derive each site's threshold from its own DET curve, which on ChokePoint would cut false alerts about fourfold.
3. **Repeated runs.** Repeat every experiment over several seeds with confidence intervals, to resolve differences as small as those in {tab:main}.
4. **Learning from operator decisions.** Feed the confirmed and rejected alerts of each node, which are labelled examples from that camera, into training rounds.
5. **Scoring layer in the live matcher.** Move it into the live matcher, with its statistics computed through SA.
6. **Dropout-robust SA.** Add Shamir-shared self-masks so that aggregation survives a dropped node.
7. **Better templates.** Adopt the more accurate random-projection scheme and make template-only matching the default.
8. **Deployment and real-time operation.** Run the nodes and server on separate machines, with an edge GPU per site for 8 FPS.
