#!/usr/bin/env python3
"""Speaker script for FedMPS_Capstone_Final_Review.pptx: one script per slide,
split across three presenters, plus likely examiner questions with answers.

Usage:
    python docs/presentation/build_script.py
"""
from pathlib import Path

import docx
from docx.shared import Pt, RGBColor

OUT = Path(__file__).resolve().parent / "FedMPS_Presentation_Script.docx"
MAROON = RGBColor(0x57, 0x23, 0x14)

SPEAKERS = {
    "Person 1": "Slides 1–5: introduction, problem and objectives (about 4 minutes)",
    "Person 2": "Slides 6–10: related work and the proposed solution (about 5 minutes)",
    "Person 3": "Slides 11–16: dataset, results and summary (about 5 minutes)",
}

SLIDES = [
    ("Person 1", 1, "Title",
     "Good morning, respected examiners. We are presenting our capstone project, FedMPS, which "
     "stands for Federated Missing Person Search. It is a privacy-preserving system that helps "
     "find a missing person in CCTV footage from many camera sites, without collecting all the "
     "footage in one place. Our project was guided by Kanhaiya Sharma. I will introduce the "
     "problem, then my teammates will explain the solution and the results."),
    ("Person 1", 2, "Presentation Outline",
     "This is the flow of our talk. First the introduction: overview, problem and objectives. "
     "Then related work and how we improve on it. Then our proposed solution, with the "
     "architecture, modules and tools. After that the dataset, parameters and results, and "
     "finally a summary with limitations and future work."),
    ("Person 1", 3, "Project Overview",
     "In simple words, our system searches CCTV footage of several camera sites for a missing "
     "person. The important point is that each site keeps its own video. We use a pretrained "
     "face-recognition model called ArcFace, keep it frozen, and train only a small add-on "
     "called an adapter using federated learning across five camera nodes. Privacy is protected "
     "by secure aggregation, differential privacy and keyed watchlist templates, and a human "
     "operator must confirm every alert. We used about 6.3 lakh face images, and each model "
     "update is only 1.05 MB."),
    ("Person 1", 4, "Problem Statement & Motivations",
     "The problem is: given a photo of a missing person, find the frames where the person "
     "appears, while keeping all other data at the camera site. There are three motivations. "
     "First, privacy: a central server would store the face of every bystander and becomes a "
     "single point of failure. Second, CCTV faces are small, blurred and compressed, so accuracy "
     "drops a lot. Third, it is an open-set search: almost everyone on camera is not missing, so "
     "the system must stay silent for them."),
    ("Person 1", 5, "Objectives & Expected Impact",
     "We had two objectives. One, build a complete multi-camera search pipeline whose model is "
     "trained by federated learning. Two, add privacy mechanisms and measure how much accuracy "
     "each one costs. The expected impact is that camera sites can cooperate without sharing "
     "footage, the model adapts to CCTV conditions, and every disclosure is reviewed by a human "
     "and recorded. Now my teammate will explain the related work."),
    ("Person 2", 6, "Related Work & Their Limitations",
     "We surveyed sixteen papers; this table shows the six closest to our work. For example, "
     "paper one used YOLO with ArcFace and reached 97.5 percent accuracy, but it decides frame "
     "by frame without tracking. Paper two used keyed templates with federated learning, but on "
     "clean images. Others tried FedProx, blockchain and secure aggregation. The common gap is "
     "that none tested an open-set search on CCTV-quality faces with human review."),
    ("Person 2", 7, "Improvements over Existing Solutions",
     "This slide maps each gap to our improvement. Instead of single frames, we raise an alert "
     "only when three of the last five frames of a person agree. Instead of closed-set accuracy, "
     "we test with a 691-person watchlist and 31,850 people who are not on it. We measure the "
     "cost of each privacy method, we compare against proper baselines, and we keep a human in "
     "the loop with an audit log."),
    ("Person 2", 8, "System Architecture",
     "This is our system architecture. On the left are the camera nodes. Each node detects and "
     "tracks faces, filters poor-quality faces, creates embeddings and matches them with the "
     "watchlist, all on site. In the middle is the cloud side, which combines the masked model "
     "updates and prepares the data on a GPU. Only three things travel on the network: keyed "
     "templates, masked updates, and one face crop after an operator confirms an alert."),
    ("Person 2", 9, "Key Components & Modules",
     "Our system has six main modules. The node pipeline does detection, tracking, quality "
     "checking, matching and voting. The federated adapter is a small network of about 2.6 lakh "
     "parameters on top of frozen ArcFace. Secure aggregation hides each node's update from the "
     "server. Differential privacy adds noise to updates. Cancelable templates protect the "
     "watchlist. And the consoles let the investigator and operators work, with every action "
     "logged."),
    ("Person 2", 10, "Algorithms, Technologies & Tools",
     "On the algorithm side, we use SCRFD for face detection, ByteTrack for tracking and ArcFace "
     "for recognition with cosine similarity. The adapter is trained with an InfoNCE contrastive "
     "loss, FedProx at each node and FedAvg at the server. For privacy we use Diffie–Hellman "
     "masks, Gaussian noise and keyed binary templates. Everything is written in Python with "
     "PyTorch and ONNX Runtime, the consoles use Streamlit, and the heavy GPU work ran on an "
     "AWS T4 instance. Now my teammate will present the results."),
    ("Person 3", 11, "Dataset / Input–Output",
     "We used the VGGFace2-HQ dataset, with all 628,699 images of 4,605 people. We split it by "
     "person: 70 percent of people trained across five nodes and 30 percent were held out for "
     "testing, so no test person was ever seen in training. Each node's images were degraded to "
     "look like its own CCTV camera. For real CCTV we used the ChokePoint dataset. The input to "
     "the system is a reference photo and a video, and the output is an alert with the face crop "
     "and score, as you can see in this frame from our test video."),
    ("Person 3", 12, "Parameters",
     "These are our main training settings. The adapter is 512 to 256 to 512. We trained for 200 "
     "rounds with all five nodes, each using 15 percent of its data per round, with the AdamW "
     "optimiser and a cosine learning rate. FedProx strength mu is 0.3. An alert needs a cosine "
     "score of 0.24 in three of the last five frames. These settings were fixed by a small pilot "
     "run before the main training, so nothing was tuned on the test set."),
    ("Person 3", 13, "Experimental Outcomes",
     "On the left are our main results on the same test set. Federated training reached rank-1 "
     "of 0.9734, the same as centralised training on pooled data, and it was better than "
     "training at one site alone by 1.1 points. The gain over the frozen model was small but "
     "consistent. On real CCTV, the system alerted on 95.2 percent of enrolled people and never "
     "named the wrong person. Secure aggregation cost no accuracy and the membership attack was "
     "at chance level. On the right, we compare with six published papers on their own metrics; "
     "their tests use clean images, so this comparison is only indicative."),
    ("Person 3", 14, "Summary",
     "To summarise: federated training gives the same accuracy as pooling all data, while "
     "keeping data at each site, and training alone at one site actually loses accuracy, so the "
     "backbone should stay frozen. Our limitations are that the main test uses simulated CCTV "
     "degradation, the real CCTV test has only 29 people, we used one random seed, and "
     "differential privacy was not usable with only five sites. In future we plan to test on "
     "harder real footage, set thresholds per site and run more seeds."),
    ("Person 3", 15, "Bibliography",
     "These are the main references we used in this presentation. The complete list of 27 "
     "references is in our thesis."),
    ("Person 3", 16, "Thank you",
     "Thank you for your time. We are happy to answer your questions, and we can also show the "
     "live demo of the consoles."),
]

QA = [
    ("What is federated learning, in one line?", "Person 2",
     "Each site trains the model on its own data and sends only the model changes to the "
     "server; the server averages them. The data never leaves the site."),
    ("Why did you freeze ArcFace instead of training the whole model?", "Person 2",
     "Training a face-recognition backbone needs millions of images, which one camera site does "
     "not have. Freezing it also keeps each update to 1.05 MB instead of 174.4 MB, 165 times "
     "smaller."),
    ("What does the adapter do?", "Person 2",
     "It is a small residual network that slightly corrects the embedding of a blurred CCTV "
     "face so it moves closer to the clean embedding of the same person."),
    ("What is FedProx and why use it?", "Person 2",
     "FedProx adds a penalty when a node's model moves too far from the global model. Our nodes "
     "have different cameras, so this keeps them from drifting apart. We used mu equal to 0.3."),
    ("How does secure aggregation work?", "Person 2",
     "Every pair of nodes shares a secret key using Diffie–Hellman. One adds a random mask to "
     "its update and the other subtracts the same mask, so the masks cancel when the server adds "
     "all updates. The server sees only the total."),
    ("What happens if a node drops out during secure aggregation?", "Person 2",
     "In our version the masks would not cancel, so that round fails. Real systems add an extra "
     "self-mask shared by secret sharing to handle this; we listed it as future work."),
    ("Why did differential privacy not work?", "Person 3",
     "With only five nodes the noise does not average out. Even the weakest noise, with a privacy "
     "budget of 252, dropped rank-1 to about 0.06. We report it honestly as not usable at this "
     "scale."),
    ("How do the cancelable templates protect the missing person's photo?", "Person 2",
     "The photo is never sent to the nodes. The server turns its embedding into 512 bits and "
     "mixes it with a key unique to each case and node. A leaked template can be cancelled by "
     "changing the key, and two nodes cannot link their templates. It costs 4.7 rank-1 points."),
    ("The adapter gain is very small. Then what is the use of federated learning?", "Person 3",
     "The main benefit is privacy at no accuracy cost: federated equals pooled training. Also, "
     "every site training alone did worse than the frozen model, so federation is 1.1 rank-1 "
     "points better than working alone. The adapter helped most on the worst camera."),
    ("Federated beat centralised slightly. Isn't that suspicious?", "Person 3",
     "We checked for leakage: test and training people are completely separate. The difference "
     "is within sampling noise, and averaging across different cameras acts like mild "
     "regularisation."),
    ("Why simulated degradation instead of real CCTV for training?", "Person 3",
     "No public CCTV dataset with labels at this size exists. So we simulated blur, small faces, "
     "noise, lighting and JPEG compression, and then checked on real CCTV with ChokePoint."),
    ("Why split by identity and not by image?", "Person 3",
     "So that no test person is ever seen in training. Otherwise the model would just remember "
     "people and the accuracy would be misleading."),
    ("What is TPIR at 1 percent FPIR?", "Person 1",
     "FPIR is how many people not on the watchlist wrongly raise an alert; TPIR is how many "
     "watchlist people are found and correctly named. We fix false alarms at 1 percent and "
     "report how many we find: 0.9107 for federated training."),
    ("How do you avoid false alarms?", "Person 1",
     "Three ways: a quality gate drops tiny or blurred faces, an alert needs three of five "
     "frames to agree, and a human operator confirms every alert before anything leaves the "
     "site."),
    ("Can it run in real time?", "Person 3",
     "On a laptop CPU it runs about 1 frame per second, enough for recorded footage. On the AWS "
     "GPU the ChokePoint video ran at about 44 frames per second, so a small GPU per site would "
     "give real time."),
    ("What was the total cost?", "Person 3",
     "About US$ 7, roughly ₹610, mainly AWS GPU time of 7.5 hours. All software and datasets were "
     "free."),
    ("What did you personally learn or find hardest?", "Any",
     "Handling 6.3 lakh images in limited memory, which needed streaming preparation, and "
     "reporting negative results honestly, like differential privacy not working."),
]

d = docx.Document()
st = d.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(11.5)


def h(text, level):
    p = d.add_heading(text, level)
    for r in p.runs:
        r.font.color.rgb = MAROON
    return p


h("FedMPS Final Review – Presentation Script", 0)
d.add_paragraph("16 slides, about 14 minutes, three speakers. Speak naturally; you do not need "
                "to read word for word.")
h("Who presents what", 1)
for who, what in SPEAKERS.items():
    p = d.add_paragraph(style="List Bullet")
    p.add_run(f"{who}: ").bold = True
    p.add_run(what)

current = None
for who, n, title, text in SLIDES:
    if who != current:
        h(f"{who}", 1)
        current = who
    p = d.add_paragraph()
    r = p.add_run(f"Slide {n} – {title}")
    r.bold = True
    r.font.color.rgb = MAROON
    d.add_paragraph(text)
    if n in (5, 10):
        nxt = "Person 2" if n == 5 else "Person 3"
        p = d.add_paragraph()
        p.add_run(f"Hand over to {nxt}.").italic = True

h("Follow-up questions and answers", 1)
d.add_paragraph("The suggested person answers first; anyone can add to it.")
for k, (q, who, a) in enumerate(QA, 1):
    p = d.add_paragraph()
    p.add_run(f"Q{k}. {q}").bold = True
    p.add_run(f"  ({who})").italic = True
    d.add_paragraph(a)

d.save(str(OUT))
print(f"wrote {OUT}")
