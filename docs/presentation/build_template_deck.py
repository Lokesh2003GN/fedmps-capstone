#!/usr/bin/env python3
"""FedMPS final review deck in the university's Capstone_PPT_Template.pptx.

Keeps every template slide, its titles, sub-headings, logo and numbering, and
fills only the empty content area of each slide. Numbers come from the thesis
(docs/FedMPS_Capstone_Thesis_v3) and the results/ files.

Usage:
    python docs/presentation/build_template_deck.py
"""
import copy
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = ROOT / "docs" / "thesis" / "figures"
TEMPLATE = HERE / "Capstone_PPT_Template.pptx"
OUT = HERE / "FedMPS_Capstone_Final_Review.pptx"

MAROON = RGBColor(0x57, 0x23, 0x14)     # the template's title colour
BULLET = RGBColor(0x38, 0x91, 0xA7)     # the template's bullet colour
TINT = RGBColor(0xF5, 0xEE, 0xEB)       # light maroon tint for cards and table rows
INK = RGBColor(0x00, 0x00, 0x00)
MUTED = RGBColor(0x55, 0x55, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Gill Sans MT"                    # the template's body font

prs = Presentation(str(TEMPLATE))
S = list(prs.slides)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def body_box(slide):
    return next(sh for sh in slide.shapes if sh.name == "TextShape 2")


def keep_heading_only(slide, height=0.5):
    """Leave the template's bold sub-heading and drop its empty lines."""
    tb = body_box(slide)
    ps = tb.text_frame._txBody.findall(qn("a:p"))
    for p in ps[1:]:
        tb.text_frame._txBody.remove(p)
    tb.height = Inches(height)
    return tb


def run(p, text, size, color=INK, bold=False, italic=False):
    r = p.add_run()
    r.text = text
    f = r.font
    f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, FONT
    f.color.rgb = color
    return r


def textbox(slide, x, y, w, h, name, margin=0.04):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(margin))
    return tf


def bullet_para(p, level=0):
    """The template's square bullet in its blue, at a smaller indent for content."""
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(Inches(0.22 + 0.25 * level)))
    pPr.set("indent", str(-Inches(0.2)))
    clr = etree.SubElement(pPr, qn("a:buClr"))
    etree.SubElement(clr, qn("a:srgbClr")).set("val", "3891A7")
    etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Wingdings")
    etree.SubElement(pPr, qn("a:buChar")).set("char", "")


def bullets(tf, items, size=14, after=5, level=0):
    for k, item in enumerate(items):
        p = tf.paragraphs[0] if k == 0 and not tf.paragraphs[0].runs else tf.add_paragraph()
        p.space_after = Pt(after)
        bullet_para(p, level)
        lead, _, rest = item.partition(": ")
        if rest:
            run(p, lead + ": ", size, INK, bold=True)
            run(p, rest, size, INK)
        else:
            run(p, item, size, INK)


def heading(slide, x, y, w, text, name, size=15):
    tf = textbox(slide, x, y, w, 0.34, name, margin=0)
    run(tf.paragraphs[0], text, size, MAROON, bold=True)


def card(slide, x, y, w, h, head, body, name, size=13):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w),
                               Inches(h))
    s.name = f"{name} background"
    s.adjustments[0] = 0.08
    s.fill.solid()
    s.fill.fore_color.rgb = TINT
    s.line.fill.background()
    s.shadow.inherit = False
    heading(slide, x + 0.15, y + 0.1, w - 0.3, head, f"{name} heading", size=14)
    tf = textbox(slide, x + 0.15, y + 0.45, w - 0.3, h - 0.55, f"{name} text", margin=0)
    run(tf.paragraphs[0], body, size, INK)


def stat(slide, x, y, w, value, label, name):
    tf = textbox(slide, x, y, w, 0.5, f"{name} value", margin=0)
    run(tf.paragraphs[0], value, 24, MAROON, bold=True)
    tf = textbox(slide, x, y + 0.48, w, 0.45, f"{name} label", margin=0)
    run(tf.paragraphs[0], label, 11, MUTED)


def table(slide, x, y, w, rows, widths, name, size=11, row_h=0.3):
    gt = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w),
                                Inches(row_h * len(rows)))
    gt.name = name
    t = gt.table
    for j, cw in enumerate(widths):
        t.columns[j].width = Inches(cw)
    for i, r in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, v in enumerate(r):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = MAROON if i == 0 else (TINT if i % 2 == 0 else WHITE)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            p.text = ""
            run(p, v, size, WHITE if i == 0 else INK, bold=(i == 0))
    return gt


def picture(slide, path, x, y, w=None, h=None, name="Figure"):
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                   Inches(w) if w else None, Inches(h) if h else None)
    pic.name = name
    return pic


def caption(slide, x, y, w, text, name="Caption"):
    tf = textbox(slide, x, y, w, 0.28, name, margin=0)
    run(tf.paragraphs[0], text, 10, MUTED, italic=True)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# --------------------------------------------------------------------------
# 1  Title
# --------------------------------------------------------------------------
s = S[0]
sh = {x.name: x for x in s.shapes}
sh["TextShape 1"].text_frame.paragraphs[1].runs[0].text = (
    "FedMPS: Privacy-Preserving Federated Learning for Missing Person Identification")
sh["TextShape 1"].text_frame.paragraphs[1].runs[0].font.size = Pt(22)
sup = sh["TextShape 2"].text_frame.paragraphs[0].runs
sup[2].text, sup[3].text, sup[4].text = "Kanhaiya Sharma", "", ""
stu = sh["CustomShape 3"].text_frame.paragraphs
for k, (name, reg) in enumerate([("G N Lokesh", "23BCE9603"), ("R Kumaresh", "23BCE9585"),
                                 ("K Kishore", "23BCE9746"), ("Aswin Maheswaran", "23BCE8540")]):
    p = stu[2 + k]
    run(p, f"{name} ({reg})", 13, INK)
notes(s, "We present FedMPS, a privacy-preserving federated learning system for finding a missing "
         "person in CCTV footage held by several camera sites.")

# --------------------------------------------------------------------------
# 3  Project overview
# --------------------------------------------------------------------------
s = S[2]
keep_heading_only(s)
tf = textbox(s, 0.3, 1.2, 5.6, 3.8, "Overview points")
bullets(tf, ["Goal: find a missing person in CCTV footage held by several camera sites",
             "Privacy: sites never share raw video, faces of passers-by or training data",
             "Model: frozen ArcFace backbone with a small adapter trained by federated learning "
             "across five camera nodes",
             "Safeguards: secure aggregation, differential privacy, keyed watchlist templates",
             "Oversight: a human operator confirms every alert; every action is audited"],
        size=14, after=7)
for k, (v, lab) in enumerate([("628,699", "face images, VGGFace2-HQ"), ("5", "camera nodes"),
                              ("1.05 MB", "update per round, 165× smaller"),
                              ("95.2%", "real-CCTV walk-throughs alerted")]):
    x = 6.3 + (k % 2) * 1.75
    y = 1.3 + (k // 2) * 1.35
    stat(s, x, y, 1.65, v, lab, f"Overview stat {k}")
notes(s, "FedMPS lets camera sites search together for a missing person without pooling footage. "
         "Only a small adapter is trained federatedly, privacy mechanisms protect the updates and "
         "the watchlist, and a human confirms every alert.")

# --------------------------------------------------------------------------
# 4  Problem statement and motivations
# --------------------------------------------------------------------------
s = S[3]
keep_heading_only(s)
tf = textbox(s, 0.3, 1.15, 9.3, 0.75, "Problem statement")
run(tf.paragraphs[0], "Given a photograph of a missing person and camera sites that each keep their "
                      "own footage, flag the frames where the person appears, keep all other data "
                      "on site, and let a human confirm every disclosure.", 14, INK, italic=True)
for k, (head, body) in enumerate([
        ("Privacy risk", "Central search copies every bystander's face and is a single point of "
                         "failure; sites often cannot share footage."),
        ("Poor CCTV faces", "Faces are 20–48 px, blurred, compressed and badly lit, so accuracy "
                            "drops sharply against a clean photograph."),
        ("Open-set search", "Almost everyone on camera is on no watchlist; the system must stay "
                            "silent for them, which closed-set accuracy hides.")]):
    card(s, 0.3 + k * 3.13, 2.1, 2.95, 2.1, head, body, f"Motivation {k}")
notes(s, "The obvious design pools all footage centrally, which exposes every bystander. CCTV faces "
         "are also poor quality, and the task is open-set: almost everyone seen must be rejected.")

# --------------------------------------------------------------------------
# 5  Objectives and expected impact
# --------------------------------------------------------------------------
s = S[4]
keep_heading_only(s)
heading(s, 0.3, 1.15, 4.5, "Objectives", "Objectives heading")
tf = textbox(s, 0.3, 1.55, 4.5, 3.4, "Objectives")
bullets(tf, ["Objective 1: a multi-camera search pipeline (detection, tracking, quality gate, "
             "alignment, ArcFace) whose model is trained by federated learning",
             "Objective 2: secure aggregation, DP on updates and cancelable templates, each "
             "measured against the same run without it"], size=14, after=8)
heading(s, 5.2, 1.15, 4.5, "Expected impact", "Impact heading")
tf = textbox(s, 5.2, 1.55, 4.5, 3.4, "Impact")
bullets(tf, ["Cooperation: camera sites search together without pooling footage",
             "Accuracy: a shared model that adapts to CCTV conditions",
             "Trust: human review and an audit log on every disclosure",
             "Evidence: open-set metrics and real CCTV tests, not claims"], size=14, after=8)
notes(s, "Two objectives: a federated multi-camera pipeline, and privacy mechanisms whose cost is "
         "measured. The impact is cooperation without pooling, with human oversight.")

# --------------------------------------------------------------------------
# 6  Related work and their limitations
# --------------------------------------------------------------------------
s = S[5]
keep_heading_only(s)
rows = [("Ref.", "Approach", "Result", "Limitation"),
        ("[1]", "YOLO + ArcFace missing-person search", "97.5% accuracy", "Frame-by-frame, no tracking"),
        ("[2]", "96-bit keyed templates, FedAvg + DP", "99.75% on CelebA", "Clean images, closed-set"),
        ("[3]", "Split FL + partial SMPC + LDP", "94.68%, MIA AUC 0.51", "Small datasets"),
        ("[4]", "FedAvg vs FedProx on non-IID LFW + DP", "FedProx best, 89.65%", "Accuracy falls with DP"),
        ("[5]", "Blockchain-based FL (BFLFace)", "92.22% vs 94.67%", "Throughput, incentives"),
        ("[6]", "Federated verification, secure aggregator", "EER 1.66% → 2.8%", "Bandwidth, latency")]
table(s, 0.3, 1.2, 9.3, rows, (0.55, 3.45, 2.35, 2.95), "Related work table", size=12, row_h=0.42)
caption(s, 0.3, 4.3, 9.3, "Six of the sixteen papers in the literature survey; none evaluated an "
                          "open-set search on CCTV-quality faces with human review")
notes(s, "We surveyed sixteen papers; these six are the closest. Each leaves a gap: frame-level "
         "decisions, clean or closed-set evaluation, or unmeasured privacy cost.")

# --------------------------------------------------------------------------
# 7  Improvements over existing solutions
# --------------------------------------------------------------------------
s = S[6]
keep_heading_only(s)
rows = [("Existing work", "FedMPS"),
        ("Decides on single frames", "Alerts only when 3 of the last 5 frames of a track agree"),
        ("Closed-set accuracy", "Open-set search: 691-person watchlist, 31,850 crowd probes"),
        ("Privacy cost not measured", "Each mechanism measured against the same run"),
        ("No proper baselines", "Frozen, local-only and centralised baselines on identical data"),
        ("Matches disclosed automatically", "Human review and audit log before any disclosure")]
table(s, 0.3, 1.2, 9.3, rows, (3.4, 5.9), "Improvements table", size=13, row_h=0.5)
notes(s, "Each gap from the survey maps to a design decision in FedMPS: track voting, open-set "
         "evaluation, measured privacy, full baselines and human oversight.")

# --------------------------------------------------------------------------
# 8  System architecture
# --------------------------------------------------------------------------
s = S[7]
keep_heading_only(s)
picture(s, FIG / "system_architecture_v2.png", 0.3, 1.1, h=4.0, name="System architecture")
tf = textbox(s, 6.4, 1.2, 3.3, 3.9, "Architecture points")
bullets(tf, ["Nodes: detect, track, gate, embed and match on site",
             "Cloud: aggregates masked adapter updates; GPU prepares data",
             "Network: only keyed templates, masked updates and one confirmed crop"],
        size=13, after=8)
notes(s, "Each camera node runs the same pipeline on its own video. The cloud coordinates training. "
         "Only keyed templates, masked updates and a single confirmed crop cross the network.")

# --------------------------------------------------------------------------
# 9  Key components and modules
# --------------------------------------------------------------------------
s = S[8]
keep_heading_only(s)
comps = [("Node pipeline", "SCRFD detection, ByteTrack, quality gate, embedding, matching, "
                           "3-of-5 vote"),
         ("Federated adapter", "263,424-parameter residual MLP on frozen ArcFace, FedProx, "
                               "200 rounds"),
         ("Secure aggregation", "Pairwise Diffie–Hellman masks cancel in the sum"),
         ("Differential privacy", "Update clipping C = 0.313, Gaussian noise, Rényi accountant"),
         ("Cancelable templates", "512-bit keyed code per case and node; revocable, unlinkable"),
         ("Consoles and audit", "Server and node consoles; every action logged")]
for k, (head, body) in enumerate(comps):
    card(s, 0.3 + (k % 3) * 3.13, 1.2 + (k // 3) * 1.75, 2.95, 1.55, head, body, f"Component {k}")
notes(s, "Six modules: the node pipeline, the federated adapter, three privacy mechanisms, and the "
         "consoles with the audit log.")

# --------------------------------------------------------------------------
# 10  Algorithms, technologies, frameworks and tools
# --------------------------------------------------------------------------
s = S[9]
keep_heading_only(s)
heading(s, 0.3, 1.15, 4.6, "Algorithms and methods", "Algorithms heading")
tf = textbox(s, 0.3, 1.55, 4.6, 3.5, "Algorithms")
bullets(tf, ["Detection and tracking: SCRFD det_10g, ByteTrack",
             "Recognition: ArcFace ResNet-50 [15], cosine similarity",
             "Training: symmetric InfoNCE loss, FedProx [17], FedAvg [16]",
             "Privacy: Diffie–Hellman masks, Gaussian DP, keyed binarisation with XOR"],
        size=13, after=7)
heading(s, 5.2, 1.15, 4.5, "Technologies and tools", "Tools heading")
tf = textbox(s, 5.2, 1.55, 4.5, 3.5, "Tools")
bullets(tf, ["Language: Python 3.13",
             "Frameworks: PyTorch 2.13, ONNX Runtime 1.24, InsightFace, OpenCV 5.0",
             "Application: Streamlit 1.51 consoles, SQLite store, supervision",
             "Cloud: AWS EC2 g4dn.2xlarge with NVIDIA T4 GPU, S3"], size=13, after=7)
notes(s, "SCRFD and ByteTrack find and follow faces, ArcFace embeds them, and the adapter is trained "
         "with InfoNCE and FedProx. Everything is Python, with GPU work on AWS.")

# --------------------------------------------------------------------------
# 11  Dataset and input/output
# --------------------------------------------------------------------------
s = S[10]
keep_heading_only(s)
heading(s, 0.3, 1.15, 4.4, "Datasets", "Datasets heading")
tf = textbox(s, 0.3, 1.55, 4.4, 3.5, "Datasets")
bullets(tf, ["VGGFace2-HQ [13]: 628,699 images of 4,605 people",
             "Split by identity: 70% over 5 nodes, 30% held out",
             "Degraded to CCTV: one camera profile per node",
             "Real CCTV: ChokePoint [14], 29 people, 24 sequences"], size=13, after=7)
heading(s, 5.0, 1.15, 4.7, "Input → output", "IO heading")
tf = textbox(s, 5.0, 1.55, 4.7, 1.0, "IO")
bullets(tf, ["Input: reference photo and CCTV video",
             "Output: pending alert with crop and score"], size=13, after=4)
picture(s, FIG / "v2_annotated_frame.png", 5.0, 2.4, w=4.7, name="Annotated frame")
caption(s, 5.0, 4.08, 4.7, "Node 0 video: enrolled person matched in red, others in green")
notes(s, "We used every image of VGGFace2-HQ, split by identity so no test person is seen in "
         "training, and ChokePoint for real CCTV. The input is a reference photo and video; the "
         "output is an alert that an operator reviews.")

# --------------------------------------------------------------------------
# 12  Parameter descriptions
# --------------------------------------------------------------------------
s = S[11]
keep_heading_only(s)
rows = [("Parameter", "Value", "Parameter", "Value"),
        ("Adapter", "512 → 256 → 512, 263,424 params", "Rounds, nodes", "200 rounds, 5 nodes"),
        ("Loss", "Symmetric InfoNCE, τ = 0.07", "Data per round", "15% of each node"),
        ("FedProx", "μ = 0.3", "Optimiser", "AdamW, lr 10⁻⁴, cosine"),
        ("Batch size", "128", "Weight decay", "10⁻⁴"),
        ("Match threshold", "cosine 0.24, template 0.2656", "Track vote", "3 of last 5 frames"),
        ("DP clip", "C = 0.313", "Template", "512 bits, 64 bytes")]
table(s, 0.3, 1.2, 9.3, rows, (1.75, 2.9, 1.75, 2.9), "Parameters table", size=12, row_h=0.5)
notes(s, "Key settings: a 256-unit residual adapter, InfoNCE with FedProx at mu 0.3, 200 rounds "
         "with 15 percent of each node's data per round, all fixed by a pilot before the main run.")

# --------------------------------------------------------------------------
# 13  Experimental outcomes
# --------------------------------------------------------------------------
s = S[12]
keep_heading_only(s)
heading(s, 0.3, 1.1, 4.5, "Training regimes, same test set", "Results heading")
rows = [("Regime", "Rank-1", "TPIR @ 1%", "EER"),
        ("Frozen backbone", "0.9733", "0.9054", "12.1%"),
        ("Centralised", "0.9733", "0.9080", "11.7%"),
        ("Federated", "0.9734", "0.9107", "11.8%"),
        ("Local only", "0.9620", "0.8812", "12.5%")]
table(s, 0.3, 1.45, 4.5, rows, (1.65, 0.95, 1.0, 0.9), "Results table", size=12, row_h=0.36)
tf = textbox(s, 0.3, 3.4, 4.5, 1.7, "Other outcomes")
bullets(tf, ["Real CCTV: 95.2% walk-throughs alerted, 0 wrong names",
             "Privacy: SA cost 0; MIA AUC 0.501; templates −4.7 rank-1",
             "DP: no usable setting with five sites"], size=12, after=4)
heading(s, 5.1, 1.1, 4.6, "Against published work, same metric", "Comparison heading")
rows = [("Ref.", "Metric", "Paper", "FedMPS"),
        ("[7]", "TAR @ FAR 10⁻⁴", "82.5%", "57.5%"),
        ("[8]", "EER", "10.4–12.9%", "11.8%"),
        ("[9]", "TPIR @ FPIR 10⁻³", "0.872", "0.836"),
        ("[10]", "TAR @ FAR 10⁻⁴", "57.1%", "57.5%"),
        ("[11]", "TAR @ FAR 10⁻³", "87.8%", "68.4%"),
        ("[12]", "TAR @ FAR 10⁻⁴", "91.2%", "57.5%")]
table(s, 5.1, 1.45, 4.6, rows, (0.85, 1.55, 1.15, 1.05), "Comparison table", size=12, row_h=0.36)
caption(s, 5.1, 4.1, 4.6, "Papers test on clean IJB images, FedMPS on 20–48 px CCTV faces")
notes(s, "Federated training matched centralised training and beat training alone by 1.1 rank-1 "
         "points; the gain over the frozen backbone is small. On real CCTV the system alerted on "
         "95.2 percent of enrolled walk-throughs. Against published work, FedMPS matches the "
         "federated and protected results on EER and TAR at 10 to the minus 4.")

# --------------------------------------------------------------------------
# 14  Summary: findings, limitations, future directions
# --------------------------------------------------------------------------
s = S[13]
tb = body_box(s)
txb = tb.text_frame._txBody
ps = txb.findall(qn("a:p"))
heads = [ps[0], ps[4], ps[8]]
for p in ps:
    txb.remove(p)
content = [["Federated training equals pooling: rank-1 0.9734 against 0.9733",
            "Training alone loses accuracy; keep the backbone frozen"],
           ["Simulated degradation; real CCTV test covers only 29 people",
            "One random seed; DP has no usable setting here"],
           ["Harder real footage and per-site thresholds",
            "Several seeds and dropout-robust secure aggregation"]]
for hd, items in zip(heads, content):
    txb.append(copy.deepcopy(hd))
    for item in items:
        p = etree.SubElement(txb, qn("a:p"))
        pPr = etree.SubElement(p, qn("a:pPr"))
        pPr.set("marL", str(Inches(0.5)))
        pPr.set("indent", str(-Inches(0.2)))
        etree.SubElement(pPr, qn("a:spcBef")).append(etree.Element(qn("a:spcPts"), val="200"))
        clr = etree.SubElement(pPr, qn("a:buClr"))
        etree.SubElement(clr, qn("a:srgbClr")).set("val", "3891A7")
        etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Wingdings")
        etree.SubElement(pPr, qn("a:buChar")).set("char", "")
        r = etree.SubElement(p, qn("a:r"))
        rPr = etree.SubElement(r, qn("a:rPr"), lang="en-US", sz="1500", dirty="0")
        etree.SubElement(etree.SubElement(rPr, qn("a:solidFill")), qn("a:srgbClr")).set("val",
                                                                                        "000000")
        etree.SubElement(rPr, qn("a:latin")).set("typeface", FONT)
        etree.SubElement(r, qn("a:t")).text = item
notes(s, "The key finding: keep the backbone frozen, federate a small adapter, and never train at a "
         "single site alone. Limits are simulated degradation and one seed; next steps are harder "
         "real footage and per-site thresholds.")

# --------------------------------------------------------------------------
# 15  Bibliography: the references cited in this deck, in order
# --------------------------------------------------------------------------
refs = [
    "G. S. Rakshika et al., AI-based Missing Person Identification using YOLO and Deep Facial Embeddings, ICCMC 2026.",
    "A. Hayat et al., DeepShieldFed: Securing Face Templates via Deep Cancelable Transforms and FL, IJCB 2025.",
    "A. Aldahiri et al., Privacy-Preserving Vertical Split Federated Learning for Face Recognition, IEEE Trans. Privacy, 2026.",
    "S. Nusa et al., A Comparative Study of FL Methods for Face Recognition on Non-IID Dataset, AIMS 2024.",
    "H. Zheng et al., Blockchain-based Federated Learning Framework Applied in Face Recognition, ICSIP 2022.",
    "E. Solomon et al., Secure and Efficient Face Recognition via Supervised Federated Learning, AIBThings 2024.",
    "Q. Cao et al., VGGFace2: A Dataset for Recognising Faces across Pose and Age, FG 2018.",
    "H. Park and S. Yun, FedHide: Federated Learning by Hiding in the Neighbors, arXiv:2409.07808, 2024.",
    "H. Kim et al., Federated Learning for Face Recognition via Intra-subject Self-supervised Learning, BMVC 2024.",
    "T. Kim et al., A Prototype-based Framework for Open-set Heterogeneous Federated Face Recognition, ESWA, 2026.",
    "W. Wang and S. Li, DP-FedFace: Privacy-Preserving Facial Recognition in Real Federated Scenarios, CIKM 2024.",
    "J. Ji et al., Privacy-Preserving Face Recognition with Learnable Privacy Budgets in Frequency Domain, ECCV 2022.",
    "NNNNAI, VGGFace2-HQ: A High Resolution Version of VGGFace2, GitHub and Kaggle, 2022.",
    "Y. Wong et al., Patch-based Probabilistic Image Quality Assessment for Face Selection (ChokePoint), CVPRW 2011.",
    "J. Deng et al., ArcFace: Additive Angular Margin Loss for Deep Face Recognition, CVPR 2019.",
    "H. B. McMahan et al., Communication-Efficient Learning of Deep Networks from Decentralized Data, AISTATS 2017.",
    "T. Li et al., Federated Optimization in Heterogeneous Networks, MLSys 2020.",
]
s = S[14]
ph = next(x for x in s.placeholders if x.placeholder_format.idx != 0)
ph.left, ph.top, ph.width, ph.height = Inches(0.4), Inches(0.85), Inches(9.2), Inches(4.3)
tf = ph.text_frame
tf.word_wrap = True
for k, ref in enumerate(refs):
    p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
    p.alignment = PP_ALIGN.LEFT
    p.space_after = Pt(1)
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(Inches(0.38)))
    pPr.set("indent", str(-Inches(0.38)))
    etree.SubElement(pPr, qn("a:buNone"))
    run(p, f"[{k + 1}]\t{ref}", 10, INK)
notes(s, "References cited in this presentation; the full list of 27 is in the thesis.")

prs.save(str(OUT))
print(f"wrote {OUT} ({len(prs.slides)} slides)")
