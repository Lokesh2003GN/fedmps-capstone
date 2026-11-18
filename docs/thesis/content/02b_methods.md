### 2.2.5 Local Training at a Node

At each node, the adapter learns to map a degraded face embedding towards the clean embedding of the same person. Training uses a symmetric contrastive loss of the Information Noise-Contrastive Estimation (InfoNCE) type. In a batch of B images, let aᵢ be the adapted degraded embedding of image i and cⱼ the clean embedding of image j. Their scaled cosine similarity, with temperature τ = 0.07, is given by {eq:sim}, and the loss for row i of the batch by {eq:infonce}.

$$ sᵢⱼ = ( aᵢ · cⱼ ) / τ $$ {#eq:sim}


$$ ℓᵢ = − log [ exp( sᵢᵢ ) / Σⱼ exp( sᵢⱼ ) ] $$ {#eq:infonce}

The batch loss averages {eq:infonce} over the rows and, in the symmetric direction, over the columns. Other images of the *same* person in the batch are removed from the denominator, so the adapter is never pushed to separate a person from themselves. The negatives are the other people at the same node. They are faces degraded by the same camera, which can never be centralised and are the hardest negatives available. To limit *client drift* on non-IID data, each node adds the FedProx proximal term, which penalises moving far from the global weights w_g received at the start of the round [@fedprox]:

$$ L = L_InfoNCE + ( μ / 2 ) · ‖ w − w_g ‖² ,   μ = 0.3 $$ {#eq:fedprox}

In each round, each node trains on a fresh random 15% of its images with AdamW, at a weight decay of 10⁻⁴ and a batch size of 128. The learning rate starts at 10⁻⁴ and follows a cosine decay towards zero over the 200 rounds, so the late rounds make ever smaller changes and the model settles. Over 200 rounds each node therefore makes 30 passes' worth over its data. Passing over all of the data in every round was tried first and overfitted: after 200 full epochs the local-only adapters fell well below the frozen backbone. The node then returns only the **delta**, the change in its weights, together with its number of training images. A node never sends images, embeddings, labels or identities. The approach as a whole is transfer learning by adapter tuning, a form of parameter-efficient fine-tuning: the pretrained backbone stays frozen and only the small adapter learns. {tab:train} brings the complete training configuration together in one place.

Table: Training configuration of the federated adapter {#tab:train}

| Item | Setting |
|---|---|
| Approach | Transfer learning by adapter tuning, a form of parameter-efficient fine-tuning |
| Backbone | ArcFace ResNet-50 `w600k_r50`, 43.6 million parameters, pretrained by InsightFace and frozen |
| Trained part | Residual bottleneck adapter, 512 → 256 → 512, with 263,424 parameters |
| Loss function | Symmetric InfoNCE contrastive loss with τ = 0.07, plus the FedProx proximal term with μ = 0.3 |
| Optimiser | AdamW with a weight decay of 10⁻⁴; learning rate 10⁻⁴ with cosine decay over the rounds |
| Local training | Batch size 128; a fresh random 15% of the node's images per round |
| Federated algorithm | FedProx at the nodes and FedAvg weighting by n_k / n at the server |
| Rounds and nodes | 200 rounds, with all 5 nodes taking part in every round |
| Model selection | Settings fixed in advance by a pilot on training identities; the last round is reported |
| Optional privacy | DP clipping to C = 0.313 with Gaussian noise, and SA by pairwise masks |

### 2.2.6 Aggregation and the Federated Round

The server combines the deltas with FedAvg, weighting each node k by its number of training images n_k out of the total n, so that a node with few images cannot dominate [@fedavg]:

$$ w_g ← w_g + Σₖ ( n_k / n ) · d_k $$ {#eq:fedavg}

One federated round proceeds as follows:

```
for round r = 1 .. 200:
    lr_r <- cosine(1e-4, r, 200)               # decays towards 0
    server sends global adapter w_g to all 5 nodes
    for each node k (in parallel):
        w_k  <- local_train(w_g, 15% of data_k, InfoNCE + FedProx(mu = 0.3), lr_r)
        d_k  <- w_k - w_g                      # the only thing that leaves node k
        d_k  <- privatise(d_k)                 # optional: clip + Gaussian noise
        send mask_k(d_k)                       # optional: pairwise secure-aggregation masks
    server: w_g <- w_g + sum_k (n_k / n) * d_k  # masks cancel in the sum
report w_g after round 200 (no round selection)
```

Training runs for 200 rounds with all five nodes taking part in every round. One round passes over about 65,700 of the 438,053 training images, and the whole run gives 30 passes' worth per node and 150 in total. After every round the global adapter is scored on the test protocol, but only to draw the convergence curve of Section 4.4; the score never selects a round. {fig:flarch} shows the complete federated architecture: the server, the five camera nodes, the traffic between them and the steps that each node applies before its update leaves it.

### 2.2.7 Model Selection by a Pilot Run

The earlier run selected the best round on validation identities, and its learning curve still fluctuated after 30 rounds. This run instead fixes every setting **before** the main run and then reports the last of 200 rounds, so no choice is made by looking at the test result. The settings were chosen by a 30-round federated **pilot** that used the training identities only. At each node, 10% of its people were held back, 321 people and 42,898 images in all, and every candidate was trained on the other 90% and scored on those held-back people with the same protocol and scoring layer as the real evaluation. None of the 1,382 test identities was loaded. Six candidates were compared: the frozen backbone, residual MLP adapters with 64 and 256 hidden units at different FedProx strengths, and three versions of a *head* adapter that trains the scale and shift of the backbone's last normalisation layer together with a rank-16 update of its final fully connected layer. The residual MLP with 256 hidden units and μ = 0.3 gave the highest TPIR at an FPIR of 1% and the lowest EER, and was chosen. The full pilot table is given in Section 4.2. The fraction of data per round, 15%, was then set so that 200 rounds add up to the 30 epochs of the pilot.

![Federated learning architecture of FedMPS](docs/thesis/figures/fl_architecture.png){#fig:flarch w=5.5}

### 2.2.8 Comparison Arms

A federated result is only meaningful against the right baselines. Every arm in {tab:arms} is trained and evaluated on identical data and scored through the same scoring layer and operating point.

Table: Training regimes compared {#tab:arms}

| Arm | How it is trained | Role |
|---|---|---|
| Frozen backbone | No adapter, which is equivalent to round 0 | Baseline: does the adapter add anything? |
| Local only | Each node trains its own adapter on its own data only, for the same 200 passes over 15% of its data; no weights are shared | Lower bound: does collaboration help? |
| Federated, FedMPS | FedProx across the five nodes for 200 rounds, as in Sections 2.2.5–2.2.7 | The proposed system |
| Centralised | All nodes' training data pooled at one client, trained with the same rounds, data fraction, schedule and proximal term | Upper bound: what does keeping data local cost? |

The federated loop also runs two automatic checks. It warns if federated training fails to beat local-only training, and it warns if federated training beats centralised training, which would point to leakage between training and test data.

### 2.2.9 Secure Aggregation

The server needs the *sum* of the nodes' deltas, not any individual delta. FedMPS implements SA by pairwise cancelling masks:

1. Every pair of nodes i and j agrees a shared secret by Diffie–Hellman key exchange over the 2048-bit Modular Exponential (MODP) group of Request for Comments (RFC) 3526, using 256-bit private exponents. Neither the server nor any other node can compute this secret.
2. For each round r, both nodes of a pair derive the same seed by hashing the secret, the round number and the pair indices with Secure Hash Algorithm 256 (SHA-256). A pseudo-random generator expands the seed into a Gaussian mask vector m_ij,r of the same length as the delta. Including the round number means that masks never repeat across rounds.
3. The lower-indexed node adds m_ij,r to its delta, and the higher-indexed node subtracts it.
4. The server adds up what it receives. Every mask appears once with each sign and cancels exactly, leaving the weighted sum, while each masked delta on its own is indistinguishable from noise.

The masks have elements of magnitude about 1, while the elements of an adapter delta are a few thousandths. In the illustration recorded by the experiment, a node's update has a norm of 5.13, and its masked version has a norm of 1,026. This construction is single-masked: if a node drops out mid-round, its masks do not cancel, and recovering from that would reveal pairwise masks. Production protocols add a Shamir-shared self-mask for this case, which is listed as future work.

### 2.2.10 Differential Privacy

DP is applied at the update level. Each node clips its delta to an L2 norm C, which bounds how much any one node can move the global model, and then adds Gaussian noise with a standard deviation of σ·C. The clip norm is calibrated from real update norms rather than guessed: a short run of three rounds without privacy records the 15 node updates, and C is set to the 70th percentile of their norms, which gives C = 0.313. The median norm was 0.241 and the 95th percentile 0.399, so one update in three is clipped. When masking is also on, the server only ever sees the sum, so each of the n nodes adds noise with the standard deviation:

$$ σ_node = σ · C / √n $$ {#eq:dpnoise}

The noise variances of the n nodes add up, so by {eq:dpnoise} the aggregate carries noise of exactly σ·C. The privacy budget ε over all rounds is computed with the Rényi DP accountant for the Gaussian mechanism, at δ = 10⁻⁵. The noise multiplier σ is swept from 0 to 2.0 to trace the privacy–utility curve. Like the other privacy experiments, the sweep uses 30 rounds rather than 200, so that each point takes about 90 seconds on the GPU. A smaller ε gives a stronger guarantee, so the sweep shows directly how much accuracy each level of privacy costs. Chapter 4 reports the result of this sweep.

### 2.2.11 Cancelable Watchlist Templates

A missing person's photograph is never sent to the nodes. The server sends a cancelable template instead, built in four steps as shown in {fig:template}:

1. **Embed.** The server embeds the photograph with the frozen backbone.
2. **Select and binarise.** Dimensions are ranked by a Fisher ratio, the spread between identities relative to the spread within them, learned once offline, and the n best are kept. FedMPS uses n = 512, so every dimension is kept, and each is thresholded at its mean.
3. **Key.** A key specific to the case and the node is derived by hashing a master secret with the case identifier and the node number using SHA-256, expanded to n bits.
4. **Protect.** The bits are combined with the key by an exclusive OR (XOR) and packed into n/8 = 64 bytes.

![Derivation of a cancelable watchlist template](docs/thesis/figures/template_flow.png){#fig:template w=6.0}

Because XOR with a common key cancels inside a Hamming distance, a node applies the same key to its live faces and compares the two codes without undoing the protection. The Hamming distance d_H between two n-bit codes is mapped to the cosine scale by {eq:hamming}.

$$ s = 1 − 2 · d_H / n $$ {#eq:hamming}

With {eq:hamming}, identical codes score 1 and unrelated codes score about 0. The template scheme has three properties:

- **Revocable:** closing or re-issuing a case retires its key.
- **Unlinkable:** the same person enrolled at two nodes has unrelated bit patterns.
- **Harder to invert:** thresholding discards the magnitude of each dimension.

The bit width matters. The 96 bits used in earlier work proved far too lossy under CCTV degradation, as Chapter 4 shows, which is why 512 bits are used [@lit2]. The protection defends stored templates and prevents linking watchlists across sites. It does not defend against a live node that is fully compromised, since such a node holds its own key.

### 2.2.12 Evaluation Protocol and Metrics

All models are evaluated on one fixed test split, drawn with seed 1337 only from the 1,382 held-out identities. The protocol follows open-set identification practice and is summarised in {tab:protocol}.

Table: Evaluation protocol {#tab:protocol}

| Item | Setting |
|---|---|
| Watchlist or gallery | 691 held-out people; each entry averages 3 clean enrolment photographs |
| Mate probes | 30,540 degraded probes of watchlist people, each fusing 3 consecutive frames; enrolment photographs are never reused as probes |
| Non-mate probes | 31,850 degraded probes of the 691 other held-out people, who are not on the watchlist |
| Verification pairs | 200,000 clean-versus-degraded pairs, half same person and half different |
| Success rule | The probe's best score clears the threshold **and** its top-ranked identity is correct |
| Fixed test set | Identical for every model compared |

The metrics are those of Section 1.4. *Rank-k* is the fraction of mate probes whose correct identity is among the top k of the 691 watchlist entries. *TPIR at FPIR x* is the fraction of mate probes whose best score clears the threshold that lets a fraction x of non-mate probes through, for x from 10% down to 0.01%, **and** whose top match is the correct person; it is the primary metric. The True Accept Rate (TAR) at a False Accept Rate (FAR) of x, the AUC and the EER describe 1:1 verification on the 200,000 pairs. With 31,850 non-mate probes the smallest measurable FPIR is about 3.1 × 10⁻⁵, and with 100,000 different-person pairs the smallest measurable FAR is 10⁻⁵, so the 10⁻⁴ operating points reported in the published work of Section 4.10 can be measured here too. Operating points below these limits are reported as "not resolvable" instead of being extrapolated.

The **operating point** is 3 enrolment photographs × 3 fused frames. A real case file usually has several photographs. The tracker also follows a person over several frames. This is therefore how the system operates. The single-photograph, single-frame result is also reported in Chapter 4. The same protocol is applied to real CCTV video in Section 4.9.

### 2.2.13 Scoring Layer

Before matching, a light **scoring layer** applies three rigid corrections to the embeddings:

1. **Centring.** Degraded embeddings share an offset, the "signature" of the cameras. The mean degraded embedding of the node-training people is subtracted from every probe, and the mean clean embedding is subtracted from every gallery entry.
2. **Partial rotation.** The orthogonal Procrustes solution that best rotates degraded embeddings onto their clean counterparts is computed, blended half-way with the identity and applied to probes. A rotation preserves all angles, so it cannot distort the geometry the way a free-form learned map can.
3. **Hubness correction.** In high dimensions a few gallery entries, called hubs, are close to many probes and win matches they should not. Each gallery entry's score is reduced by its mean similarity to its 200 nearest neighbours in a cohort of unlabelled camera traffic. In the benchmark, the non-mate probes serve as this unlabelled cohort.

The centring and rotation statistics are fitted on the node-training identities only, pooled across nodes; in a deployment they could be aggregated with the same SA, which is not implemented here. Every arm in {tab:arms}, including the frozen backbone, is scored through the same scoring layer, so its gain cannot be credited to the adapter.
