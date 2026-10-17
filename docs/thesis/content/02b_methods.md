### 2.2.5 Local Training at a Node

At each node, the adapter learns to map a degraded face embedding towards the clean embedding of the same person. Training uses a symmetric contrastive loss of the InfoNCE type [@cpc]. In a batch of B images, let aᵢ be the adapted degraded embedding of image i, cⱼ the clean embedding of image j, and sᵢⱼ = aᵢ · cⱼ / τ their scaled cosine similarity with temperature τ = 0.07. The loss for row i is

$$ ℓᵢ = − log [ exp(sᵢᵢ) / Σⱼ exp(sᵢⱼ) ] $$

and the batch loss averages it over rows and over columns (the symmetric direction). Other images of the *same* person in the batch are removed from the denominator, so the adapter is never pushed to separate a person from themselves. The negatives are the other people at the same node: faces degraded by the same camera, which can never be centralised and are the hardest negatives available.

To limit *client drift* on non-IID data, each node adds the FedProx proximal term [@fedprox], which penalises moving far from the global weights w_g received at the start of the round:

$$ L = L_InfoNCE + (μ / 2) · ‖ w − w_g ‖² ,   μ = 3.0 $$

Each node trains for one local epoch per round with AdamW (learning rate 10⁻⁴, weight decay 10⁻⁴, batch size 128). It then returns only the **delta**, the change in its weights, together with its number of training images. A node never sends images, embeddings, labels or identities.

### 2.2.6 Aggregation and the Federated Round

The server combines the deltas with FedAvg [@fedavg], weighting each node by its number of training images so that a node with few images cannot dominate. One federated round proceeds as follows:

```
for round r = 1 .. 30:
    server sends global adapter w_g to all 5 nodes
    for each node k (in parallel):
        w_k  <- local_train(w_g, data_k, InfoNCE + FedProx(mu = 3.0), 1 epoch)
        d_k  <- w_k - w_g                      # the only thing that leaves node k
        d_k  <- privatise(d_k)                 # optional: clip + Gaussian noise
        send mask_k(d_k)                       # optional: pairwise secure-aggregation masks
    server: w_g <- w_g + sum_k (n_k / n) * d_k  # masks cancel in the sum
    server: v_r <- validation rank-1 of w_g on held-out identities
keep the round r* with the highest v_r (early stopping)
```

Training runs for 30 rounds with all five nodes taking part in every round. That gives 30 epochs per node and 150 node-epochs in total; one round passes over 41,420 training images.

### 2.2.7 Early Stopping on Held-out Identities

A small adapter trained on a few hundred people can overfit: its accuracy on new people can peak early and then fall while the training loss keeps improving. After every round, the server-side loop therefore scores the global adapter on the **518 validation identities** held back at the nodes (Section 2.2.1). It uses the same gallery-and-probe protocol as the test set, and keeps the round with the best validation rank-1, not the last round. The validation identities are disjoint from both training and test identities, so selecting a round on them does not leak test information.

### 2.2.8 Comparison Arms

A federated result is only meaningful against the right baselines. Every arm in {tab:arms} is trained and evaluated on identical data and scored through the same scoring layer and operating point.

Table: Training regimes compared {#tab:arms}

| Arm | How it is trained | Role |
|---|---|---|
| Frozen backbone | no adapter (equivalent to round 0) | baseline: does the adapter add anything? |
| Local only | each node trains its own adapter on its own data only, for the same 30 epochs; no weights are shared | lower bound: does collaboration help? |
| Federated (FedMPS) | FedProx across the five nodes, with early stopping (Sections 2.2.5–2.2.7) | the proposed system |
| Centralised | all nodes' training data pooled at one client, trained with the same rounds, epochs and proximal term | upper bound: what does keeping data local cost? |

The federated loop also runs two automatic checks. It warns if federated fails to beat local-only, and it warns if federated beats centralised, which would point to leakage between training and test data.

### 2.2.9 Secure Aggregation

The server needs the *sum* of the nodes' deltas, not any individual delta. FedMPS implements secure aggregation by pairwise cancelling masks [@secagg]:

1. Every pair of nodes (i, j) agrees a shared secret by Diffie–Hellman key exchange [@dh] over the 2048-bit MODP group of RFC 3526 [@rfc3526], using 256-bit private exponents. Neither the server nor any other node can compute this secret.
2. For each round r, both nodes of a pair derive the same seed by hashing the secret, the round number and the pair indices with SHA-256 [@fips180]. A pseudo-random generator expands the seed into a Gaussian mask vector m_ij,r of the same length as the delta. Including the round number means masks never repeat across rounds.
3. The lower-indexed node adds m_ij,r to its delta; the higher-indexed node subtracts it.
4. The server adds up what it receives. Every mask appears once with each sign and cancels exactly, leaving the weighted sum, while each masked delta on its own is indistinguishable from noise.

The masks have elements of magnitude about 1, while the elements of a clipped delta are around 0.002. In the illustration recorded by the experiment, a random update of typical size has norm 2.6, and its masked version has norm 513. This construction is single-masked: if a node drops out mid-round its masks do not cancel, and recovering from that would reveal pairwise masks. Production protocols add a Shamir-shared self-mask for this case [@secagg], which is listed as future work.

### 2.2.10 Differential Privacy

Differential privacy is applied at the update level [@dpsgd, @dwork]. Each node clips its delta to an L2 norm C, which bounds how much any one node can move the global model, and then adds Gaussian noise with standard deviation σ·C. The clip norm is calibrated from real update norms rather than guessed: a short run without privacy records the norms, and C is set to their 70th percentile (C = 0.128). When masking is also on, each node adds noise of standard deviation σ·C/√n, because the server only ever sees the sum; the noise variances add up to give the aggregate exactly σ·C. The privacy budget ε over all rounds is computed with the Rényi-DP accountant for the Gaussian mechanism [@rdp], at δ = 10⁻⁵. The noise multiplier σ is swept from 0 to 2.0 to trace the privacy–utility curve.

### 2.2.11 Cancelable Watchlist Templates

A missing person's photograph is never sent to the nodes. The server sends a cancelable template [@ratha], built as follows ({fig:template}):

1. **Embed.** The server embeds the photograph with the frozen backbone.
2. **Select and binarise.** Dimensions are ranked by a Fisher ratio (spread between identities relative to spread within them), learned once offline, and the n best are kept. FedMPS uses n = 512, so every dimension is kept, and each is thresholded at its mean.
3. **Key.** A key specific to the case and the node is derived by hashing a master secret with the case identifier and node number using SHA-256 [@fips180], expanded to n bits.
4. **Protect.** The bits are XOR-ed with the key and packed into n/8 = 64 bytes.

![Derivation of a cancelable watchlist template](docs/thesis/figures/template_flow.png){#fig:template w=5.6}

Because XOR with a common key cancels inside a Hamming distance, a node applies the same key to its live faces and compares without undoing the protection. Hamming distance is mapped to the cosine scale as s = 1 − 2·d_H/n. This gives three properties.

- **Revocable:** closing or re-issuing a case retires its key.
- **Unlinkable:** the same person enrolled at two nodes has unrelated bit patterns.
- **Harder to invert:** thresholding discards the magnitude of each dimension.

The bit width matters: the 96 bits used in [@lit2] proved far too lossy under CCTV degradation (Chapter 4), which is why 512 bits are used. The protection defends stored templates and prevents linking watchlists across sites. It does not defend against a live node that is fully compromised, since such a node holds its own key.

### 2.2.12 Evaluation Protocol and Metrics

All models are evaluated on one fixed test split (seed 1337) built only from the 6,042 held-out identities. The protocol follows open-set identification practice [@frvt, @iso19795] and is summarised in {tab:protocol}.

Table: Evaluation protocol {#tab:protocol}

| Item | Setting |
|---|---|
| Watchlist (gallery) | 3,021 held-out people; each entry averages 3 clean enrolment photographs |
| Mate probes | 15,105 degraded probes of watchlist people, each fusing 3 consecutive frames; enrolment photographs are never reused as probes |
| Non-mate probes | 18,126 degraded probes of 3,021 other held-out people who are not on the watchlist |
| Verification pairs | 6,000 clean-versus-degraded pairs, half same person and half different |
| Success rule | the probe's best score clears the threshold **and** its top-ranked identity is correct |
| Fixed test set | identical for every model compared |

The metrics are those of Section 1.4. *Rank-k* is the fraction of mate probes whose correct identity is among the top k of the 3,021 watchlist entries. *TPIR at FPIR x* is the fraction of mate probes whose best score clears the threshold that lets a fraction x of non-mate probes through (x = 1% or 0.1%), **and** whose top match is the correct person; it is the primary metric. *TAR at FAR x*, AUC and EER describe 1:1 verification on the 6,000 pairs.

With 18,126 non-mate probes the smallest measurable FPIR is about 5.5 × 10⁻⁵, and with 3,000 different-person pairs the smallest measurable FAR is 3.3 × 10⁻⁴. Operating points below these limits are reported as "not resolvable" instead of being extrapolated.

The **operating point** is 3 enrolment photographs × 3 fused frames. A real case file usually has several photographs, and the tracker follows a person over several frames, so this is how the system operates. The single-photograph, single-frame result is also reported (Chapter 4). The same protocol is applied to real CCTV video in Section 4.9.

### 2.2.13 Scoring Layer

Before matching, a light **scoring layer** applies three rigid corrections to the embeddings:

1. **Centring.** Degraded embeddings share an offset, the "signature" of the cameras. The mean degraded embedding of the node-training people is subtracted from every probe, and the mean clean embedding from every gallery entry.
2. **Partial rotation.** The orthogonal Procrustes solution [@procrustes] that best rotates degraded embeddings onto their clean counterparts is computed, blended half-way with the identity, and applied to probes. A rotation preserves all angles, so it cannot distort the geometry the way a free-form learned map can.
3. **Hubness correction.** In high dimensions a few gallery entries are close to many probes and win matches they should not ("hubs") [@hubness]. Each gallery entry's score is reduced by its mean similarity to its 200 nearest neighbours in a cohort of unlabelled camera traffic. In the benchmark, the non-mate probes serve as this unlabelled cohort.

The centring and rotation statistics are fitted on the node-training identities only, pooled across nodes; in a deployment they could be aggregated with the same secure aggregation, which is not implemented here. Every arm in {tab:arms}, including the frozen backbone, is scored through the same scoring layer, so it cannot be credited to the adapter.
