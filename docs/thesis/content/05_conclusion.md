# CHAPTER 5 | Conclusion and Future Work

## 5.1 Conclusion

This project built and evaluated FedMPS, a federated, privacy-preserving system for searching CCTV footage for a missing person. The four objectives set in Chapter 1 were met as follows.

1. **A federated multi-camera pipeline.** Face detection (SCRFD), tracking (ByteTrack), a quality gate, five-point alignment, a frozen ArcFace backbone and track-level voting run at each node. A 66,240-parameter adapter on top of the backbone is trained by federated learning (FedProx) across five nodes with non-IID camera conditions. Raw video, face crops and training data never leave a node, and each model update is 0.26 MB instead of the 174.4 MB the full backbone would need.
2. **Privacy mechanisms, each measured.** Pairwise-masked secure aggregation produced exactly the same model as plain aggregation at 0.8% extra run time. Update-level differential privacy was implemented with a Rényi-DP accountant, but no setting gave a meaningful guarantee (ε < 10) with usable accuracy. Cancelable templates, keyed per case and per node, work end to end; at 512 bits they cost about 9 rank-1 points. A membership-inference attack found no leakage (AUC 0.496).
3. **An open-set evaluation.** The system was evaluated on 172,620 CCTV-degraded VGGFace2 images with a 3,021-person watchlist and 18,126 probes of people not on it, using TPIR at fixed FPIR and identity-disjoint data. Federated training reached the accuracy of centralised training on pooled data (rank-1 0.9473 against 0.9476). On real CCTV video (ChokePoint), the live pipeline alerted correctly on 95.2% of enrolled walk-throughs, never with the wrong name.
4. **A working demonstration.** The server and node consoles carry a case from enrolment, through scanning and human review, to a confirmed sighting. Every step is recorded in an audit log, and only a human-confirmed face crop ever reaches the server.

The central negative finding is reported as clearly as the positive ones. On this data, no training regime improves on the frozen ArcFace backbone (rank-1 0.9474). Federated, centralised and well-regularised local training all converge to it, and early stopping keeps the adapter after its first round. The difficulty lies in information lost to camera degradation, and the part that can be corrected, a rigid shift of the embeddings, is captured by the scoring layer. A fairness check also corrected an earlier claim: the 1.87-point advantage of federated over local-only training came from local-only training without early stopping, not from collaboration.

FedMPS therefore shows that camera sites *can* cooperate on a missing-person search without pooling footage or biometric data, at no accuracy cost relative to pooling, with human oversight and an audit trail on every disclosure. It also shows, by measurement, where the limits of the current design lie.

## 5.2 Future Work

1. **Harder real footage.** Extend the real-CCTV evaluation to a true missing-person setting: enrolment from ordinary photographs and probes from surveillance cameras at several distances, as in SCface [@scface], with larger watchlists. Night-time and infrared footage may contain systematic distortions on which an adapter could improve over the frozen backbone.
2. **Per-site thresholds.** Derive each site's alert threshold from its own DET curve; on ChokePoint a threshold set for 1% FPIR would cut false alerts about fourfold while keeping 99.7% of hits.
3. **Repeated runs.** Repeat every experiment over several random seeds and report confidence intervals, so that differences as small as those in {tab:main} can be resolved.
4. **Learning from operator decisions.** Feed the confirmed and rejected alerts collected at each node, which are labelled examples from that camera, into federated training rounds.
5. **Scoring layer in the live matcher,** with thresholds from the DET curve and its statistics computed through secure aggregation instead of pooled data.
6. **Dropout-robust secure aggregation** with Shamir-shared self-masks [@secagg].
7. **Better templates:** the random-projection scheme, more accurate at every bit width, with template-only matching as the default.
8. **Deployment over a network** on separate machines with an established federated-learning framework [@kairouz].
9. **Real-time operation** at 8 frames per second on an edge GPU at each site.
