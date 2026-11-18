| Reference | Training data | Model | Metric reported | Paper's result | FedMPS, same metric | Frozen backbone, same metric |
|---|---|---|---|---|---|---|
| Cao et al. 2018, VGGFace2 | VGGFace2, 8,631 identities | ResNet-50, centralised | IJB-C rank-1 / TAR@FAR 1e-4 | 0.898 / 0.825 | 0.9734, 57.50% | 0.9733, 57.47% |
| FedHide, Park & Yun 2024 | VGGFace2, 8,631 clients, one identity each | MobileFaceNet, 64x64 | EER | 9.5% unprotected; 10.4-12.9% protected | 11.78% | 12.11% |
| FedFS, Kim, Choi & Kwak, BMVC 2024 | VGGFace, 8,673 identities | MobileFaceNet | AUROC / TPIR@FPIR 0.1, 0.01, 0.001 | 0.9794 / 0.9786, 0.9337, 0.8721 | 0.9402, 0.9532, 0.9107, 0.8363 | 0.9389, 0.9487, 0.9054, 0.8357 |
| Kim, Kim & Teoh, ESWA 2026 | MS-Celeb-1M, 8,000 identities, 20 clients | Heterogeneous backbones | TAR@FAR 1e-4 (IJB-C, per client) | 57.14% | 57.50% | 57.47% |
| DP-FedFace, Wang & Li, CIKM 2024 | CASIA-WebFace, 4,000 clients | 64-layer CNN | TAR@FAR 1e-3 (IJB-C) | 87.76% | 68.35% | 68.20% |
| Ji et al., ECCV 2022 | VGGFace2 | ResNet-50, ArcFace, DP | TAR@FAR 1e-4 (IJB-C) | 93.25% unprotected; 91.22% private | 57.50% | 57.47% |
