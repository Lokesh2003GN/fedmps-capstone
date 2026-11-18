# CHAPTER 2 | Methodology

This chapter described the methodology of FedMPS in the order in which the work was carried out. Section 2.1 described the dataset used, its sample count and its source, together with the way it was split across the camera nodes. Section 2.2 described the preprocessing of the dataset, namely face alignment, the simulated CCTV degradation and the embedding of every image with the frozen backbone. Section 2.3 presented the proposed methodology and the architecture of the system, from the federated adapter and its training through the privacy mechanisms and the evaluation protocol to the operational pipeline and its consoles. Section 2.4 gave the software and hardware requirements, and Section 2.5 listed the standards that the design followed. The results obtained with this methodology were reported in Chapter 4.

## 2.1 Dataset

VGGFace2 was a large face dataset of 9,131 subjects and 3.31 million images with wide variation in pose, age, illumination and image quality, and it was available on Kaggle at https://www.kaggle.com/datasets/hearfool/vggface2. The VGGFace2-HQ version was downloaded directly onto the AWS training instance and every image of every identity was used, so the sample count was 628,699 images of 4,605 people, with between 9 and 414 images per person and a median of 128 [@vgghq]. The project's first experiments had used CelebA and LFW, and a later run had used a 20-image-per-person subset of the original VGGFace2. VGGFace2-HQ replaced them for two reasons. It gave a median of 128 images per person instead of 20, so that each node trained on many views of each of its people, and it started from sharp faces, so that the CCTV degradation of Section 2.2.2 alone set the probe quality rather than an unknown mix of web-image blur and the simulated camera.

The data was split by identity and not by image, so that no person used for evaluation was ever seen in training in any form. Seventy percent of the identities, which was 3,223 people and 438,053 images, were distributed across the five nodes and used for training, and no person appeared at more than one node. The nodes were deliberately unequal in size, with the largest about three times the smallest, as real camera sites would have been. The other thirty percent of the identities, which was 1,382 people and 190,646 images, were held out as the validation set and never used for training. Half of them, 691 people, formed the watchlist of missing persons to search for, and the other 691 formed the crowd of people who were on no watchlist. No validation identities were taken from the nodes, because training ran for a fixed 200 rounds and the last round was reported, so the nodes kept all of their images for training. {tab:split} and {fig:split} gave the resulting split.

Table: Identity-disjoint split of VGGFace2-HQ into nodes and held-out set {#tab:split}

| Part of the split | Identities | Images | Images per identity |
|---|---|---|---|
| Node 0 | 322 | 42,678 | 132.5 |
| Node 1 | 483 | 67,577 | 139.9 |
| Node 2 | 644 | 90,004 | 139.8 |
| Node 3 | 805 | 106,409 | 132.2 |
| Node 4 | 969 | 131,385 | 135.6 |
| Total at the nodes, 70% | 3,223 | 438,053 | 135.9 |
| Held out for validation, 30% | 1,382 | 190,646 | 137.9 |

![Node split of VGGFace2-HQ with an extract of the preparation log](docs/thesis/figures/node_split_stacked.png){#fig:split w=5.0}

## 2.2 Preprocessing of the Dataset

Before any training, every image of the dataset was prepared in the same way, so that the nodes, the evaluation and the live pipeline all saw faces in the form that the backbone expected. The preprocessing had three steps, which the sub-sections below described in turn. Every face was first aligned onto the five-landmark template of ArcFace, a copy of it was then degraded to CCTV quality with the camera profile of its node, and both the clean and the degraded face were finally embedded with the frozen backbone and stored, so that federated training and evaluation could run on the stored embeddings alone.

### 2.2.1 Face Alignment

The face-recognition backbone, ArcFace, had been trained only on faces warped onto a fixed five-landmark template, so every face had to be aligned in the same way. Before degradation, each image therefore passed through the SCRFD detector `det_10g`, with a 320 × 320 input and a detection threshold of 0.3, which found the two eyes, the nose tip and the two mouth corners, and a similarity transform warped the face onto the 112 × 112 template of ArcFace. The clean and degraded copies were then cut from the same aligned face, so they differed only in pixel quality, and 520 of the 628,699 images, which was 0.08%, could not be aligned.

### 2.2.2 CCTV Degradation Model

Real CCTV footage with identity labels at this scale was not publicly available, so for training and for the main evaluation the effect of a surveillance camera was simulated, and a separate test on real CCTV footage was reported in Section 4.9. Every image was passed through a degradation chain that followed the physical order of image formation. Motion blur was applied first at capture resolution, with a blur kernel of 1 to 9 pixels at a random angle. The face was then down-scaled to the effective size of a distant subject, between 20 and 48 pixels, and a gamma shift between 0.7 and 1.6 imitated poor lighting and automatic exposure. Gaussian sensor noise with a standard deviation of up to 12 grey levels was added next, followed by JPEG compression in the camera or recorder at a quality between 25 and 70. Finally, the image was up-scaled back to the working size, which restored the array size but not the lost detail.

Each node drew one fixed camera profile from these ranges, and every image at that node was degraded around that profile with a small per-image jitter of 25% of each range. Node 0's camera therefore consistently rendered faces at about 21 pixels with heavy motion blur and JPEG quality 50, while node 4's camera rendered them at 47 pixels and quality 66, as {fig:crops} showed. The held-out images were spread evenly over the five profiles, about 38,130 images each, so that the evaluation covered every camera. This made the federated data genuinely non-IID, since each node faced a different, systematic distortion, which was the setting in which FL was expected to help.

![Aligned enrolment crops, their CCTV probes and the five camera profiles](docs/thesis/figures/aligned_crops_vgghq.png){#fig:crops w=6.3}

### 2.2.3 Embedding with the Frozen Backbone

Faces were embedded with ArcFace, using the `buffalo_l` model package from InsightFace, whose recognition network `w600k_r50` was a ResNet-50 with 43.6 million parameters that produced a 512-dimensional, L2-normalised embedding [@arcface]. The network was run through ONNX Runtime and was never trained in this project, because training a face-recognition backbone needed millions of images and a classifier over all identities, which no single camera site had. For every image, two embeddings were stored, one of the clean image and one of its degraded copy. At 628,699 images the crops alone would have needed about 47 GB of memory, so the preparation was rewritten to stream the data, decoding, aligning, degrading and embedding the images in bounded chunks and writing the results straight to disk. With the detector and the recognition network sharing one GPU context, the instance processed 121 images per second, and the preparation took 84 minutes for all 628,699 images.

## 2.3 Proposed Methodology and System Architecture

The proposed methodology kept a pretrained face-recognition backbone frozen and trained only a small adapter on top of it by FL across the camera nodes, protected the training updates and the watchlist with privacy mechanisms, and put a human operator between every match and any disclosure. FedMPS consisted of five camera nodes and one central server. Each node stood for a camera site, such as a railway station or a bus terminal, and it owned its video footage, its operator and the faces of the people that its camera had seen. The server coordinated the search and the federated training, but it never held training data, raw video or the faces of uninvolved people. The system worked in two phases, a training phase and an operational phase, which {fig:workflow} showed one above the other, and the two phases were connected by the trained adapter that every node used.

![System architecture of FedMPS: camera nodes, cloud services and summary panels](docs/thesis/figures/system_architecture_v2.png){#fig:sysarch w=8.9 landscape}

As {fig:sysarch} showed, each camera node on the left ran the same local inference pipeline on its own CCTV video, from SCRFD detection and ByteTrack tracking through the quality gate and alignment to the frozen ArcFace backbone with the federated adapter, followed by matching against the keyed watchlist template received from the server and voting across the frames of each track. A match became a pending alert that the node's operator confirmed or rejected, and the node kept its alerts and audit trail in its local store. In the cloud in the middle, the federated coordinator combined the masked adapter updates of the nodes and distributed the global adapter of 1.05 MB in every round, and the training orchestrator on the AWS GPU instance prepared VGGFace2-HQ, ran the pilot, calibrated the DP clip and evaluated the model on the held-out identities and on ChokePoint. The secure storage held the adapter weights, checkpoints, embeddings and logs, the privacy modules provided DP, cancelable templates and SA, and the dashboard comprised the server console, the confirmed sightings, the node console and the audit log. The panels on the right summarised the technical stack, the privacy measures and the outputs of the system. Only keyed templates, masked adapter updates and a single face crop after an operator's confirmation crossed the network, while raw video, the faces of passers-by, training data, labels and pending alerts never left a node.

In the training phase, the nodes improved a shared model together. A frozen, pretrained ArcFace network turned every face into a 512-dimensional embedding, and a small adapter on top of it was trained by FL. In each round, every node trained the adapter on its own data and sent only the change in the adapter's weights, and the server combined these changes, optionally under SA and DP, and sent the new adapter back. In the operational phase, a missing-person case was opened at the server, which read the reference photograph once, in memory, and issued each node its own keyed watchlist template. Each node scanned its own footage by detecting, tracking and quality-checking faces, embedding them with the backbone and the trained adapter, comparing them with the watchlist and voting across the frames of each track. A match became a pending alert that only that node's operator could see, a face crop reached the server only after the operator confirmed the alert, and every step was written to an audit log.

![Workflow of FedMPS: training phase above, operational phase below](docs/thesis/figures/workflow.png){#fig:workflow w=5.6}

### 2.3.1 The Federated Adapter

Only a small residual adapter was federated. It took the 512-dimensional embedding x of a degraded face and nudged it towards the position of the same person's clean embedding, as given by {eq:adapter}.

$$ a(x) = normalise( x + W₂ · Dropout( GELU( LayerNorm( W₁ x ) ) ) ) $$ {#eq:adapter}

In {eq:adapter}, W₁ was a learned matrix that mapped the 512 input dimensions to 256 hidden dimensions, LayerNorm normalised the hidden vector, GELU was the activation function, dropout with a rate of 0.1 regularised training, and W₂ mapped the 256 hidden dimensions back to 512. The result was added to the input embedding x, so the adapter learned only a correction to the frozen embedding rather than a new one, and the sum was renormalised to unit length so that the cosine comparison of {eq:cos} still applied. This MLP had 263,424 parameters, so one update was 1,053,696 bytes or 1.05 MB, against 174.4 MB if the whole backbone of 43.6 million parameters had been federated, which made each update 165 times smaller. The last layer was initialised to zero, so an untrained adapter passed embeddings through unchanged, round 0 of federated training was exactly the frozen backbone, and any change after it was due to the adapter.

The hidden size of 256 was chosen by the pilot run of Section 2.3.4. The earlier run, with only 16 training images per person, had used a 64-unit adapter, while with about 136 images per person a larger adapter could be trained. A second kind of candidate, which adapted the backbone's own last layers rather than its output, was also included in the pilot, and the comparison of the candidates was reported in Section 4.2.

### 2.3.2 Local Training at a Node

At each node, the adapter learned to map a degraded face embedding towards the clean embedding of the same person, using a symmetric contrastive loss of the InfoNCE type. In a batch of B images, let aᵢ be the adapted degraded embedding of image i and cⱼ the clean embedding of image j. Their scaled cosine similarity was given by {eq:sim}.

$$ sᵢⱼ = ( aᵢ · cⱼ ) / τ $$ {#eq:sim}

In {eq:sim}, the dot product aᵢ · cⱼ was the cosine similarity of the two unit-length embeddings, and the temperature τ = 0.07 divided it so that small differences in similarity became large differences in the scores that the loss compared. The score sᵢᵢ belonged to the matching pair, a degraded image and the clean image of the same face, and every other score in row i belonged to a non-matching pair. The loss for row i of the batch was then given by {eq:infonce}.

$$ ℓᵢ = − log [ exp( sᵢᵢ ) / Σⱼ exp( sᵢⱼ ) ] $$ {#eq:infonce}

In {eq:infonce}, the fraction was the probability that the model assigned to the correct clean image among all the clean images of the batch, so the loss was small when the matching pair scored far above every other pair and large otherwise. The batch loss averaged {eq:infonce} over the rows and, in the symmetric direction, over the columns. Other images of the same person in the batch were removed from the denominator, so the adapter was never pushed to separate a person from themselves. The negatives were the other people at the same node, faces degraded by the same camera, which could never be centralised and were the hardest negatives available. To limit client drift on non-IID data, each node added the FedProx proximal term, which penalised moving far from the global weights received at the start of the round [@fedprox]. The full local objective was given by {eq:fedprox}.

$$ L = L_InfoNCE + ( μ / 2 ) · ‖ w − w_g ‖² ,   μ = 0.3 $$ {#eq:fedprox}

In {eq:fedprox}, L_InfoNCE was the batch loss built from {eq:infonce}, w was the node's current adapter weights, w_g was the global weights received from the server at the start of the round, and ‖ w − w_g ‖² was the squared distance between them. The weight μ = 0.3, chosen in the pilot of Section 2.3.4, set how strongly a node was held near the global model, so that a node with an unusual camera could still adapt to it but could not drag the shared model far towards its own data.

In each round, each node trained on a fresh random 15% of its images with AdamW, at a weight decay of 10⁻⁴ and a batch size of 128. The learning rate started at 10⁻⁴ and followed a cosine decay towards zero over the 200 rounds, so the late rounds made ever smaller changes and the model settled, and over 200 rounds each node therefore made 30 passes' worth over its data. The node then returned only the delta, the change in its weights, together with its number of training images, and it never sent images, embeddings, labels or identities. The approach as a whole was transfer learning by adapter tuning, a form of parameter-efficient fine-tuning in which the pretrained backbone stayed frozen and only the small adapter learned, and {tab:train} brought the complete training configuration together in one place.

Table: Training configuration of the federated adapter {#tab:train}

| Item | Setting |
|---|---|
| Approach | Transfer learning by adapter tuning, a form of parameter-efficient fine-tuning |
| Backbone | ArcFace ResNet-50 `w600k_r50`, 43.6 million parameters, pretrained by InsightFace and frozen |
| Trained part | Residual adapter, 512 to 256 to 512, with 263,424 parameters |
| Loss function | Symmetric InfoNCE contrastive loss with τ = 0.07, plus the FedProx proximal term with μ = 0.3 |
| Optimiser | AdamW with a weight decay of 10⁻⁴; learning rate 10⁻⁴ with cosine decay over the rounds |
| Local training | Batch size 128; a fresh random 15% of the node's images per round |
| Federated algorithm | FedProx at the nodes and FedAvg weighting by n_k / n at the server |
| Rounds and nodes | 200 rounds, with all 5 nodes taking part in every round |
| Model selection | Settings fixed in advance by a pilot on training identities; the last round reported |
| Optional privacy | DP clipping to C = 0.313 with Gaussian noise, and SA by pairwise masks |

### 2.3.3 Aggregation and the Federated Round

The server combined the deltas with FedAvg, weighting each node by its number of training images, so that a node with few images could not dominate the shared model [@fedavg]. The update of the global adapter was given by {eq:fedavg}.

$$ w_g ← w_g + Σₖ ( n_k / n ) · d_k $$ {#eq:fedavg}

In {eq:fedavg}, d_k was the delta returned by node k, n_k was that node's number of training images and n was the total over all five nodes, so the weights n_k / n summed to one and the step was a weighted average of the five deltas added to the current global weights w_g. Node 4, with 131,385 images, therefore counted about three times as much as node 0 with 42,678 images. One federated round proceeded as in the following listing, in which the comments marked the only quantity that left a node and the optional privacy steps.

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

Training ran for 200 rounds with all five nodes taking part in every round. One round passed over about 65,700 of the 438,053 training images, and the whole run gave 30 passes' worth per node and 150 in total. After every round the global adapter was scored on the test protocol, but only to draw the convergence curve of Section 4.4, and the score never selected a round. {fig:flarch} showed the complete federated architecture, with the server, the five camera nodes, the traffic between them and the steps that each node applied before its update left it.

![FL architecture of FedMPS: server, five nodes and the steps in each round](docs/thesis/figures/fl_architecture.png){#fig:flarch w=6.0}

### 2.3.4 Model Selection by a Pilot Run

The earlier run had selected the best round on validation identities, and its learning curve had still fluctuated after 30 rounds. The final run instead fixed every setting before the main run and then reported the last of 200 rounds, so that no choice was made by looking at the test result. The settings were chosen by a 30-round federated pilot that used the training identities only. At each node, 10% of its people were held back, 321 people and 42,898 images in all, and every candidate was trained on the other 90% and scored on those held-back people with the same protocol and scoring layer as the real evaluation, while none of the 1,382 test identities was loaded. Six candidates were compared, namely the frozen backbone, residual MLP adapters with 64 and 256 hidden units at different FedProx strengths, and three versions of a head adapter that trained the scale and shift of the backbone's last normalisation layer together with a rank-16 update of its final fully connected layer. The candidate with the best open-set accuracy on the held-back people was used for the main run, the comparison of all six candidates was reported in Section 4.2, and the fraction of data per round, 15%, was then set so that 200 rounds added up to the 30 epochs of the pilot.

### 2.3.5 Comparison Arms

A federated result was only meaningful against the right baselines, so four training regimes were trained and evaluated on identical data and scored through the same scoring layer at the same operating point. The frozen backbone, with no adapter, was equivalent to round 0 and answered whether the adapter added anything at all. The local-only arm trained each node's adapter on its own data only, for the same 200 passes over 15% of its data and with no weights shared, and it was the lower bound that answered whether collaboration helped. The federated arm was FedMPS itself, FedProx across the five nodes for 200 rounds as described above. The centralised arm pooled all nodes' training data at one client and trained it with the same rounds, data fraction, schedule and proximal term, and it was the upper bound that answered what keeping the data local cost. The federated loop also ran two automatic checks, warning if federated training failed to beat local-only training and warning if federated training beat centralised training, which could point to leakage between training and test data.

### 2.3.6 SA by Pairwise Masks

The server needed the sum of the nodes' deltas, not any individual delta, and FedMPS implemented SA by pairwise cancelling masks. Every pair of nodes first agreed a shared secret by Diffie–Hellman key exchange over the 2048-bit MODP group of RFC 3526 with 256-bit private exponents, a secret that neither the server nor any other node could compute. For each round, both nodes of a pair derived the same seed by hashing the secret, the round number and the pair indices with SHA-256, and a pseudo-random generator expanded the seed into a Gaussian mask vector of the same length as the delta, so the masks never repeated across rounds. The lower-indexed node of the pair added the mask to its delta and the higher-indexed node subtracted it. When the server added up what it received, every mask appeared once with each sign and cancelled exactly, leaving the weighted sum, while each masked delta on its own was indistinguishable from noise. This construction was single-masked, so if a node had dropped out mid-round its masks would not have cancelled, and production protocols added a Shamir-shared self-mask for that case, which was left as future work.

### 2.3.7 DP on Model Updates

DP was applied at the level of model updates. Each node clipped its delta to an L2 norm C, which bounded how much any one node could move the global model, and then added Gaussian noise with a standard deviation of σ·C. The clip norm was calibrated from real update norms rather than guessed, because a short run of three rounds without privacy recorded the 15 node updates and C was set to the 70th percentile of their norms, which gave C = 0.313. When masking was also on, the server only ever saw the sum, so each of the n nodes added noise with the smaller standard deviation of {eq:dpnoise}.

$$ σ_node = σ · C / √n $$ {#eq:dpnoise}

In {eq:dpnoise}, σ was the noise multiplier, C was the clip norm and n = 5 was the number of nodes. The noise variances of the n nodes added up in the sum, so the aggregate carried noise with a standard deviation of exactly σ·C, the same as a single trusted party adding it, while no node had to trust the others. The privacy budget ε over all rounds was computed with the Rényi DP accountant for the Gaussian mechanism at δ = 10⁻⁵, and the noise multiplier σ was swept from 0 to 2.0 to trace the privacy and utility curve. Like the other privacy experiments, the sweep used 30 rounds rather than 200, so that each point took about 90 seconds on the GPU, and Chapter 4 reported its result.

### 2.3.8 Cancelable Watchlist Templates

A missing person's photograph was never sent to the nodes. The server sent a cancelable template instead, built in four steps as shown in {fig:template}. The server first embedded the photograph with the frozen backbone. It then ranked the dimensions by a Fisher ratio, the spread between identities relative to the spread within them, learned once offline on the training identities, and kept the best n of them, and since FedMPS used n = 512 every dimension was kept and each was thresholded at its mean. A key specific to the case and the node was then derived by hashing a master secret with the case identifier and the node number using SHA-256, expanded to n bits. Finally, the bits were combined with the key by an XOR and packed into n / 8 = 64 bytes.

![Derivation of a cancelable template at the server and its use at node k](docs/thesis/figures/template_flow.png){#fig:template w=6.3}

Because an XOR with a common key cancelled inside a Hamming distance, a node applied the same key to its live faces and compared the two codes without undoing the protection. The Hamming distance d_H between two n-bit codes was mapped to the cosine scale by {eq:hamming}.

$$ s = 1 − 2 · d_H / n $$ {#eq:hamming}

In {eq:hamming}, d_H counted the bit positions in which the two codes differed and n = 512 was the code length, so identical codes scored 1, codes that differed in every bit scored −1, and unrelated codes, which differed in about half of their bits, scored about 0, the same scale as the cosine similarity of {eq:cos}. The template scheme had three properties. It was revocable, because closing or re-issuing a case retired its key, it was unlinkable, because the same person enrolled at two nodes had unrelated bit patterns, and it was harder to invert, because thresholding discarded the magnitude of each dimension. The bit width mattered, since the 96 bits used in earlier work proved far too lossy under CCTV degradation, as Chapter 4 showed, which was why 512 bits were used. The protection defended stored templates and prevented the linking of watchlists across sites, but it did not defend against a live node that was fully compromised, since such a node held its own key.

### 2.3.9 Evaluation Protocol and Metrics

All models were evaluated on one fixed test split, drawn with seed 1337 only from the 1,382 held-out identities, and the protocol followed open-set identification practice. The watchlist, or gallery, held 691 held-out people, and each of its entries averaged three clean enrolment photographs. The mate probes were 30,540 degraded probes of watchlist people, each fusing three consecutive frames, and the enrolment photographs were never reused as probes. The non-mate probes were 31,850 degraded probes of the 691 other held-out people, who were not on the watchlist, and 200,000 clean-versus-degraded pairs, half of the same person and half of different people, were used for verification. A probe counted as a success only when its best score cleared the threshold and its top-ranked identity was also correct, and the same fixed test set was used for every model compared.

The metrics were those of Section 1.4.5. Rank-k was the fraction of mate probes whose correct identity was among the top k of the 691 watchlist entries, and the TPIR at an FPIR of x was the fraction of mate probes whose best score cleared the threshold that let a fraction x of the non-mate probes through and whose top match was also the correct person, for x from 10% down to 0.01%, which was the primary metric. The TAR at a FAR of x, the AUC and the EER described one-to-one verification on the 200,000 pairs, and the identification error curves were plotted as FNIR against FPIR. With 31,850 non-mate probes the smallest measurable FPIR was about 3.1 × 10⁻⁵, and with 100,000 different-person pairs the smallest measurable FAR was 10⁻⁵, so the 10⁻⁴ operating points reported in the published work of Section 4.11 could be measured here too, and operating points below these limits were reported as not resolvable instead of being extrapolated. The operating point was three enrolment photographs by three fused frames, because a real case file usually had several photographs and the tracker followed a person over several frames, and the single-photograph, single-frame result was also reported in Chapter 4.

### 2.3.10 Scoring Layer

Before matching, a light scoring layer applied three rigid corrections to the embeddings. The first was centring. Degraded embeddings shared an offset, the signature of the cameras, so the mean degraded embedding of the node-training people was subtracted from every probe and the mean clean embedding was subtracted from every gallery entry. The second was a partial rotation, in which the orthogonal Procrustes solution that best rotated degraded embeddings onto their clean counterparts was computed, blended half-way with the identity and applied to the probes, and because a rotation preserved all angles it could not distort the geometry the way a free-form learned map could. The third was a hubness correction. In high dimensions a few gallery entries, called hubs, were close to many probes and won matches they should not have won, so each gallery entry's score was reduced by its mean similarity to its 200 nearest neighbours in a cohort of unlabelled camera traffic, for which the non-mate probes served in the benchmark. The centring and rotation statistics were fitted on the node-training identities only, pooled across nodes, and in a deployment they could have been aggregated with the same SA, which was not implemented here. Every arm, including the frozen backbone, was scored through the same scoring layer, so its gain could not be credited to the adapter.

### 2.3.11 Operational Pipeline at a Node

When a node scanned its camera footage, every sampled frame passed through the stages of {fig:pipeline}, and every stage ran on the node itself. Frames were sampled at up to 8 FPS, and faces with their five landmarks were found by the SCRFD detector `det_10g` and linked across frames into one track per person by ByteTrack, through the supervision library. A quality gate then discarded a face if its eyes were less than 24 pixels apart, if it was blurred, if it was badly exposed with a mean brightness outside 25 to 235, or if it was turned more than 45° away, because a tiny or side-on face did not produce a weak match but a confidently wrong one, so discarding it removed false alarms cheaply. The surviving faces were aligned and embedded by the frozen backbone and the federated adapter, and the application loaded the latest trained adapter, the one from the VGGFace2-HQ run, while a node refused an adapter trained on a different backbone instead of silently applying it.

![Operational pipeline at a node, from camera frame to pending alert](docs/thesis/figures/node_pipeline.png){#fig:pipeline w=6.3}

Matching worked in one of two modes. In template mode, the live embedding was binarised and keyed like the watchlist template and compared by the Hamming similarity of {eq:hamming} against a threshold of 0.2656. In cosine mode, the default, it was compared with the enrolled embedding against a threshold of 0.24, which was more accurate but needed the enrolled embedding at the node, while template mode used protected codes only. An alert was raised only when three of the last five frames of the same track matched the same case, at most once per track and case. This track voting turned tens of frame-level hits into one alert per person, stored as pending for the node's own operator together with the face crop, the score and the vote.

### 2.3.12 Server Console, Node Console and Audit Log

The operational workflow was run through two web consoles built with Streamlit, which shared one SQLite store holding four tables for cases, templates, alerts and the audit log. The consoles were tested end to end with the latest trained model on a recorded university event video for node 0, the video Experience Excellence: Life at VIT-AP available at https://youtu.be/EzHwEE6gV3E, and a reference photograph of one person who appeared in it, and the screenshots in this section and in Section 4.8 came from that run. In the server console, shown in {fig:server}, an investigator opened a case by entering the person's name, a case reference, notes and an automatic expiry period and by uploading a reference photograph. The photograph was read once in memory, the server computed its embedding and issued one keyed template per node, and only a 200 × 200 thumbnail was kept for the human reviewer while the full photograph was not stored. The console listed the open cases and the templates issued, it showed confirmed sightings only, and a pending alert, which no human had yet checked, was never visible at the server. Closing a case, or letting it expire, revoked its templates at every node.

![Server console with one open case and five live templates](docs/thesis/figures/v2_server_console.png){#fig:server w=6.3}

The operator at a camera site saw the review queue of that node only, as shown in {fig:node}. Each pending alert showed the reference thumbnail beside the face that the camera had captured, the match score and how many frames of the track had agreed, and the operator confirmed or rejected it, while opening the reference thumbnail was itself recorded in the audit log. Only a confirmed alert sent its single face crop upstream. The console also showed the node's watchlist as keyed codes and key fingerprints, never as photographs, together with the counts of confirmed and rejected alerts and the resulting local precision.

![Node 0 review console with pending alerts awaiting human review](docs/thesis/figures/v2_node_review.png){#fig:node w=6.3}

Once the operator had confirmed an alert, it appeared at the server as a confirmed sighting with the face crop, the score, the vote and the name of the operator who confirmed it, and the server console also reported the precision of the reviewed alerts. Every case opening, template issue, alert, reference view, confirmation, rejection and case closure was appended to the audit log with a timestamp and the acting party, as {fig:audit} showed. Server and node operators could both read the log, so oversight ran in both directions.

![Audit log at the server after the complete node 0 case](docs/thesis/figures/v2_audit_log.png){#fig:audit w=6.3}

## 2.4 System Details

The system was written in Python as one shared library used by training, by the application and by the scripts that produced the figures, and it ran on two kinds of hardware. A rented cloud GPU was used for the expensive steps, namely preparing the face embeddings, running the federated experiments and benchmarking on real CCTV video, and a laptop CPU was used for development and for the application. {tab:system} brought the software and the hardware together in one table, giving for each component the minimum that the project needed and the exact version or specification that was used, first on the laptop and then on the AWS GPU instance where the two differed.

Table: Software and hardware requirements and versions used {#tab:system}

| Component | Minimum requirement | Version or specification used |
|---|---|---|
| Operating system | 64-bit Windows or Linux | Windows 11 on the laptop; Ubuntu 22.04 deep learning image on AWS |
| Python | 3.10 or later | 3.13.5 on the laptop; 3.12 on AWS |
| PyTorch | 2.0 or later; a CUDA 12 build for GPU training | 2.13.0 CPU build on the laptop; 2.7.0 with CUDA 12.8 on AWS |
| ONNX Runtime | CPU build, or the CUDA 12 GPU build for GPU runs | ONNX Runtime 1.24.4 on the laptop; GPU build onnxruntime-gpu 1.23.2 on AWS |
| InsightFace models | `buffalo_l` package with SCRFD `det_10g` and ArcFace `w600k_r50` | InsightFace 1.0.1, `buffalo_l`, 326 MB |
| OpenCV | 4.8 or later | 5.0.0 |
| NumPy and scikit-learn | NumPy 1.24 and scikit-learn 1.3 or later | NumPy 2.1 and scikit-learn 1.5.2 |
| supervision, for ByteTrack | 0.18 to 0.30, the versions that still include ByteTrack | 0.30.2 |
| Streamlit and SQLite | Streamlit 1.50 or later and SQLite 3 | Streamlit 1.51.0 and SQLite 3.49.1 |
| Matplotlib, for the figures | 3.7 or later | 3.10.7 |
| GPU, for preparation and training | NVIDIA GPU with 16 GB of memory and CUDA 12 | NVIDIA T4 with 16 GB on AWS EC2 `g4dn.2xlarge` |
| CPU | 4 cores for the application; 8 virtual CPU cores for preparation | AMD Ryzen 5 7520U with 4 cores and 8 threads; 8 virtual CPU cores on AWS |
| RAM | 16 GB for the application; 32 GB for preparation | 16 GB on the laptop; 32 GB on AWS |
| Storage | 5 GB for the application, models and embeddings; 150 GB for preparing VGGFace2-HQ | Laptop solid-state disk; 150 GB disk and 225 GB local disk on AWS |
| Camera input | Recorded video file at any resolution | MP4 at 1912 × 880 and 30 FPS for node 0; ChokePoint at 800 × 600 and 30 FPS |

The shared library, `fedmps/`, was organised by function into data in `data/`, models in `models/`, FL in `fl/`, privacy in `privacy/`, evaluation in `eval/`, the camera pipeline in `pipeline/` and the store in `store.py`, and every module had a self-check on synthetic data. The consoles were built with Streamlit on an SQLite store that held the cases, templates, alerts and audit log, ONNX Runtime ran ArcFace and SCRFD, PyTorch ran the adapter, local training and aggregation, and supervision provided ByteTrack face tracking. The expensive steps ran on AWS EC2 GPU instances of type `g4dn.2xlarge`, each with one NVIDIA T4 GPU of 16 GB, 8 vCPU, 32 GB of RAM and Ubuntu Linux. In September 2026 such an instance prepared the original VGGFace2 run in 38 minutes and ran its experiment suite in 22 minutes, and in October 2026 one ran the first ChokePoint real-CCTV benchmark in 30 minutes of GPU compute. On 25 October 2026 an instance with a 225 GB local NVMe disk downloaded VGGFace2-HQ, prepared it in 84 minutes, ran the pilot in 19 minutes and ran the 200-round experiment suite in 64 minutes, and the 628,699 images never passed through the laptop, since only the embeddings and the results were copied back through S3. On 27 October 2026 a further instance repeated the ChokePoint benchmark with the latest trained adapter, downloading the dataset and running all three benchmark parts in 26 minutes of GPU compute. The adapter was small, so the whole application ran on the team laptop, an AMD Ryzen 5 7520U with four cores, eight threads and 16 GB of RAM under Windows 11, which was also used for development and for the console tests of Section 2.3.12.

## 2.5 Standards

The design and evaluation of FedMPS followed published standards and established practice from the ISO and the IEC, the NIST FRVT, the FIPS, the ONNX model format and the DPDP Act of India. The evaluation followed ISO/IEC 19795-1 and the NIST FRVT one-to-many protocol, which meant open-set metrics with the TPIR reported at a fixed FPIR on an identity-disjoint, fixed test set. The watchlist templates followed ISO/IEC 24745 on biometric information protection, being irreversible, unlinkable through per-node keys and renewable through per-case keys. The cryptography followed RFC 3526 for the 2048-bit Diffie–Hellman key agreement and FIPS 180-4 for the SHA-256 hashes used for the mask seeds and the template keys. ArcFace and SCRFD ran through ONNX Runtime, so the same model files ran on a GPU and on a CPU alike. Finally, the design applied data minimisation as required by the DPDP Act of 2023, since footage stayed on site, at most one confirmed crop left a node per alert, cases expired and every access was audited, which was a property of the design and not a claim of legal compliance.
