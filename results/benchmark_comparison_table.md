# FedMPS capstone benchmark results

**Every number below is measured.** Cells reading "not resolvable" are
operating points the sample size cannot support, and the code returns NaN for
them rather than a number; cells reading "not measured" are quantities that
run did not compute. Neither is filled in by estimation.

Protocol: **VGGFace2**, 8,631 identities,
172,620 images, ArcFace-aligned
(172,606/172,620
faces aligned). Gallery 3,021 identities;
15,105 mate probes; 18,126 non-mate
probes; 6,000 verification pairs.
Operating point: 3 enrolment photo(s) x
3 camera frame(s). Backbone `insightface`, frozen.

Resolution limits: the finest measurable FPIR is
5.52e-05 (18,126 non-mates) and the
finest FAR is 3.33e-04
(6,000 pairs). TAR@FAR=1e-4 is therefore below
what this verification set can resolve, which is why it reports as
not resolvable rather than as a value.

---

### Table 1: measured across the surveillance pipeline

| Arm | TAR@FAR=1e-2 | TAR@FAR=1e-3 | TAR@FAR=1e-4 | TPIR@FPIR=1e-2 | TPIR@FPIR=1e-3 | Rank-1 | Rank-5 | Rank-10 |
|---|---|---|---|---|---|---|---|---|
| Clean probes (upper bound) | 0.9347 | 0.9247 | not resolvable | 0.9915 | 0.9801 | 0.9962 | 0.9982 | 0.9985 |
| CCTV degraded (frozen backbone) | 0.7393 | 0.6133 | not resolvable | 0.7564 | 0.6079 | 0.9294 | 0.9664 | 0.9752 |
| + scoring layer | 0.7440 | 0.6287 | not resolvable | 0.7968 | 0.6725 | 0.9474 | 0.9752 | 0.9814 |
| + federated adapter | 0.7303 | 0.6193 | not resolvable | 0.8020 | 0.6820 | 0.9473 | 0.9755 | 0.9817 |


### Table 2: federated learning against its bounds

Closed-set identification on the same held-out identities, from the three-way run (`results/experiments_vgg_big_threeway.json`).

| Arm | Rank-1 | Rank-5 | Rank-10 |
|---|---|---|---|
| Frozen backbone, no adapter | 0.9474 | 0.9752 | 0.9814 |
| Local only (mean of 5 nodes) | 0.9286 | 0.9663 | 0.9745 |
| Federated (FedMPS) | 0.9473 | 0.9755 | 0.9817 |
| Centralised (upper bound) | 0.9476 | 0.9749 | 0.9817 |

---

### Reproducing

```bash
python report_figures/compute_benchmark_data.py    # measures, writes results/benchmark_measured.json
python report_figures/generate_benchmark_table.py  # tabulates only what the result files contain
```
